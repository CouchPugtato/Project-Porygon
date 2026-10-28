'use strict';

const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {constructMoveHistoryWorld, replaySteps} = require('./move_history_world');
const {proposeOwnSwitchTurns} = require('./opening_proposals');
const {compareSwitchTurns, searchSettings} = require('./opening_search');
const {restorePosition, playerObservation, pendingSides} = require('./position_simulator');
const {replacementActions, isPostKnockoutReplacement} = require('./replacement_actions');
const {checkWorldChoice} = require('./opening_actions');

async function searchMoveHistory(history, options, gate) {
    const settings = searchSettings(options), replay = replaySteps(history), started = performance.now();
    const view = history.observations.at(-1), worlds = [], proposals = [];
    const rng = new PRNG(settings.seed.join(','));
    const seedOf = () => Array.from({length: 4}, () => rng.random(65536));
    let reconstructionNodes = 0;
    const remainingTime = () => Math.floor(settings.max_elapsed_ms - (performance.now() - started));
    const stop = (reason, unsupported = []) => ({version: 1, scope: 'observation_only_move_history_search',
        root_turn: view.turn, supported: false, selected_command: null, fallback_required: true,
        reason, unsupported, settings, reconstruction_nodes: reconstructionNodes, search_nodes: 0,
        nodes: reconstructionNodes, proposals, training_eligible: false, strength_claim: false,
        elapsed_seconds: (performance.now() - started) / 1000});
    if (view.ended) return {...stop('battle_ended'), terminal: true, fallback_required: false};
    if (view.phase === 'switch' && view.request?.wait)
        return {...stop('waiting_for_opponent_replacement'), waiting: true, fallback_required: false};
    if (replay.unsupported.length) return stop('unsupported_move_history', replay.unsupported);
    const replacing = view.phase === 'switch';
    let candidates;
    try { candidates = replacing ? replacementActions(view) :
        proposeOwnSwitchTurns(view, settings.candidates, rng, settings.proposal_mode); }
    catch (error) {
        if (!/move action space outside supported bound/.test(error.message)) throw error;
        return stop('no_supported_ordinary_moves');
    }
    for (let i = 0; i < settings.worlds; ++i) {
        await gate?.waitForLaunch();
        const constructionTime = remainingTime();
        if (constructionTime < 1 || reconstructionNodes >= settings.max_nodes)
            return stop('construction_budget_exhausted');
        const world = constructMoveHistoryWorld(history, {seed: seedOf(), chance_seed: seedOf(),
            max_nodes: Math.min(512, settings.max_nodes - reconstructionNodes),
            max_elapsed_ms: Math.min(10000, constructionTime)});
        reconstructionNodes += world.nodes;
        proposals.push({supported: world.supported, unsupported: world.unsupported,
            nodes: world.nodes, world_attempts: world.world_attempts, history_sha256: world.history_sha256});
        if (!world.supported) return stop('history_world_construction_failed', world.unsupported);
        const battle = restorePosition(world.position, {mode: 'sample', seed: seedOf()});
        try {
            if (replacing) {
                if (!isPostKnockoutReplacement(battle) || !pendingSides(battle).includes(view.side))
                    return stop('unsupported_midturn_replacement');
                worlds.push({position: world.position});
                continue;
            }
            const other = view.side === 'p1' ? 'p2' : 'p1';
            const proposal = proposeOwnSwitchTurns(playerObservation(battle, other), settings.responses, rng,
                settings.response_proposal_mode);
            worlds.push({position: world.position, responses: proposal.commands, proposal});
        } catch (error) {
            if (!/move action space outside supported bound/.test(error.message)) throw error;
            return stop('no_supported_opponent_moves');
        } finally { battle.destroy(); }
    }
    const comparisonTime = remainingTime();
    if (comparisonTime < 1 || reconstructionNodes >= settings.max_nodes)
        return stop('construction_budget_exhausted');
    if (replacing) {
        // legal replacement incumbent without inventing a strategic ranking
        for (const turn of candidates.turns) {
            if (remainingTime() < 1) return stop('replacement_validation_budget_exhausted');
            if (worlds.every(world => checkWorldChoice(world.position, view, turn.command,
                {mode: 'sample', seed: seedOf()}).accepted))
                return {...stop(null), supported: true, fallback_required: false, selected_command: turn.command,
                    decision_phase: 'replacement', ranked: false, selection_method: 'legal_replacement_incumbent',
                    legal_choice_count: candidates.turns.length,
                    limitation: 'Legal selection only without strategic replacement value estimation.'};
        }
        return stop('no_consistent_replacement');
    }
    // one shared transition budget for history fitting and forward comparisons
    const result = await compareSwitchTurns(view, worlds, candidates.commands,
        {...settings, max_nodes: settings.max_nodes - reconstructionNodes, max_elapsed_ms: comparisonTime}, gate);
    return {...result, scope: 'observation_only_move_history_search', settings,
        reconstruction_nodes: reconstructionNodes, search_nodes: result.nodes, nodes: reconstructionNodes + result.nodes,
        fallback_required: !result.supported, proposals, candidate_coverage: candidates.coverage,
        response_coverage: worlds.map(world => world.proposal.coverage),
        elapsed_seconds: (performance.now() - started) / 1000,
        limitations: ['Ordinary moves, voluntary switches and legal replacement incumbents only; no Tera or pivot policy.',
            'Accepted worlds match history but are not a calibrated posterior.',
            'Only one future turn is scored with the existing HP and fainting heuristic.',
            'No command is returned until a candidate has been compared across every sampled world.',
            'The caller must retain its baseline player when fallback is required.']};
}

module.exports = {searchMoveHistory};
