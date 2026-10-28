'use strict';

const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {constructMoveHistoryWorld, replaySteps} = require('./move_history_world');
const {proposeOrdinaryTurns} = require('./opening_proposals');
const {compareOrdinaryTurns, searchSettings} = require('./opening_search');
const {restorePosition, playerObservation} = require('./position_simulator');

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
    if (replay.unsupported.length) return stop('unsupported_move_history', replay.unsupported);
    let candidates;
    try { candidates = proposeOrdinaryTurns(view, settings.candidates, rng, settings.proposal_mode); }
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
            const other = view.side === 'p1' ? 'p2' : 'p1';
            const proposal = proposeOrdinaryTurns(playerObservation(battle, other), settings.responses, rng,
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
    // one shared transition budget for history fitting and forward comparisons
    const result = await compareOrdinaryTurns(view, worlds, candidates.commands,
        {...settings, max_nodes: settings.max_nodes - reconstructionNodes, max_elapsed_ms: comparisonTime}, gate);
    return {...result, scope: 'observation_only_move_history_search', settings,
        reconstruction_nodes: reconstructionNodes, search_nodes: result.nodes, nodes: reconstructionNodes + result.nodes,
        fallback_required: !result.supported, proposals, candidate_coverage: candidates.coverage,
        response_coverage: worlds.map(world => world.proposal.coverage),
        elapsed_seconds: (performance.now() - started) / 1000,
        limitations: ['Restricted ordinary moves only; no switching, Tera, pivot or replacement policy.',
            'Accepted worlds match history but are not a calibrated posterior.',
            'Only one future turn is scored with the existing HP and fainting heuristic.',
            'No command is returned until a candidate has been compared across every sampled world.',
            'The caller must retain its baseline player when fallback is required.']};
}

module.exports = {searchMoveHistory};
