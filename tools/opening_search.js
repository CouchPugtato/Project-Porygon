'use strict';

const {performance} = require('node:perf_hooks');
const {isDeepStrictEqual} = require('node:util');
const {createHash} = require('node:crypto');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {openingActions, ordinaryActions, checkWorldChoice} = require('./opening_actions');
const {constructOpeningWorld} = require('./opening_world');
const {proposeOpeningTurns} = require('./opening_proposals');
const {restorePosition, playerObservation, advancePosition} = require('./position_simulator');

function searchSettings(options) {
    const settings = {worlds: 2, candidates: 12, responses: 4, chance_samples: 2,
        max_nodes: 512, max_elapsed_ms: 5000, proposal_mode: 'coverage', response_proposal_mode: 'uniform', ...options};
    const limits = {worlds: 4, candidates: 32, responses: 8, chance_samples: 8,
        max_nodes: 1024, max_elapsed_ms: 20000};
    if (!Array.isArray(settings.seed) || settings.seed.length !== 4 ||
            settings.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535) ||
            !['coverage', 'uniform'].includes(settings.proposal_mode) ||
            !['coverage', 'uniform'].includes(settings.response_proposal_mode) ||
            Object.keys(settings).some(key => !['seed', 'proposal_mode', 'response_proposal_mode'].includes(key) && !Object.hasOwn(limits, key)) ||
            Object.entries(limits).some(([key, max]) => !Number.isInteger(settings[key]) ||
                settings[key] < 1 || settings[key] > max))
        throw new Error('opening search requires a four-word seed and bounded integer settings');
    return settings;
}

function nextSeed(rng) { return Array.from({length: 4}, () => rng.random(65536)); }

function material(battle, perspective) {
    const sideScore = side => side.pokemon.reduce((sum, mon) =>
        sum + mon.hp / mon.maxhp + (mon.hp > 0 ? 2 : 0), 0);
    const own = battle.getSide(perspective);
    return sideScore(own) - sideScore(own.foe);
}

function scoreLeaf(position, perspective, initialMaterial, initialTurn, randomness) {
    const battle = restorePosition(position, randomness);
    try {
        if (battle.ended) return {supported: true, score: !battle.winner ? 0 :
            battle.winner === battle.getSide(perspective).name ? 100 : -100};
        if (battle.turn <= initialTurn && !(battle.requestState === 'switch' && battle.queue.list.length === 0))
            return {supported: false, reason: 'unfinished_turn_requires_continuation'};
        return {supported: true, score: material(battle, perspective) - initialMaterial};
    } finally { battle.destroy(); }
}

// hypothetical backend without live battle inputs in player facing search
async function compareTurns(view, worlds, commands, options, gate, enumerate) {
    const settings = searchSettings(options), started = performance.now();
    const pool = new Set(enumerate(view).turns.map(turn => turn.command));
    if (!Array.isArray(worlds) || !worlds.length || worlds.length > settings.worlds ||
            !Array.isArray(commands) || !commands.length || commands.length > settings.candidates ||
            new Set(commands).size !== commands.length || commands.some(command => !pool.has(command)))
        throw new Error('supply bounded worlds and distinct observation-generated candidate commands');
    const rng = new PRNG(settings.seed.join(','));
    // response keyed chance streams shared across proposal ablations
    // stable samples when reference comparisons use different column chunks
    const chanceSeeds = worlds.map((world, i) => world.responses?.map(response =>
        Array.from({length: settings.chance_samples}, (_, sample) => {
            const bytes = createHash('sha256').update(JSON.stringify([settings.seed, i, response, sample])).digest();
            return Array.from({length: 4}, (_, word) => bytes.readUInt16BE(word * 2));
        })));
    const result = {version: 1, scope: enumerate === openingActions ? 'opening_hypothetical_matchups' :
        'ordinary_hypothetical_matchups', root_turn: view.turn, supported: false,
        selected_command: null, settings, nodes: 0, budget_exhausted: false, candidates: [],
        objective: 'uniform_world_mean_of_worst_response_chance_mean',
        leaf_score: 'opponent_minus_own_hp_fraction_loss_plus_2_per_net_faint; terminal_win_loss_100',
        world_weights: worlds.map(() => 1 / worlds.length), training_eligible: false,
        strength_claim: false, fixture_worlds: worlds.some(world => world.position?.origin === 'fixture')};
    const prepared = worlds.map((world, i) => {
        const randomness = {mode: 'sample', seed: nextSeed(rng)};
        const battle = restorePosition(world.position, randomness);
        try {
            if (!isDeepStrictEqual(playerObservation(battle, view.side), view))
                throw new Error('every search world must reproduce the same player observation');
            const opponent = view.side === 'p1' ? 'p2' : 'p1';
            const opponentView = playerObservation(battle, opponent);
            const opponentPool = new Set(enumerate(opponentView).turns.map(turn => turn.command));
            if (!Array.isArray(world.responses) || !world.responses.length ||
                    world.responses.length > settings.responses || new Set(world.responses).size !== world.responses.length ||
                    world.responses.some(command => !opponentPool.has(command)))
                throw new Error('responses must be distinct commands from the simulated opponent observation');
            for (const command of world.responses) {
                const check = checkWorldChoice(world.position, opponentView, command, randomness);
                if (!check.accepted) return {unsupportedResponse: {world: i, command, reason: check.error}};
            }
            return {...world, opponent, initialMaterial: material(battle, view.side), randomness};
        } finally { battle.destroy(); }
    });
    const rejectedResponse = prepared.find(world => world.unsupportedResponse);
    if (rejectedResponse) return {...result, reason: 'opponent_response_requires_retry',
        rejection: rejectedResponse.unsupportedResponse, elapsed_seconds: (performance.now() - started) / 1000,
        comparison_complete: false};
    const outOfTime = () => performance.now() - started >= settings.max_elapsed_ms;
    for (const command of commands) {
        const row = {command, status: 'running', worlds: [], score: null};
        result.candidates.push(row);
        for (const [i, world] of prepared.entries()) {
            if (outOfTime()) { result.budget_exhausted = true; break; }
            const check = checkWorldChoice(world.position, view, command, world.randomness);
            if (!check.accepted) {
                row.status = 'world_conditional_rejection'; row.reason = check.error; row.rejected_world = i;
                break;
            }
            const responses = [];
            for (const [j, response] of world.responses.entries()) {
                const scores = [];
                for (const seed of chanceSeeds[i][j]) {
                    if (result.nodes % 16 === 0) await gate?.waitForLaunch();
                    if (result.nodes >= settings.max_nodes || outOfTime()) {
                        result.budget_exhausted = true; break;
                    }
                    ++result.nodes;
                    const randomness = {mode: 'sample', seed};
                    const next = advancePosition(world.position,
                        {[view.side]: command, [world.opponent]: response}, randomness);
                    const leaf = scoreLeaf(next.position, view.side, world.initialMaterial, view.turn, randomness);
                    if (!leaf.supported) { row.status = 'unsupported_continuation'; row.reason = leaf.reason; break; }
                    scores.push(leaf.score);
                }
                if (result.budget_exhausted || row.status !== 'running') break;
                responses.push({command: response, mean: scores.reduce((a, b) => a + b, 0) / scores.length,
                    samples: scores.length, min: Math.min(...scores), max: Math.max(...scores)});
            }
            if (result.budget_exhausted || row.status !== 'running') break;
            row.worlds.push({index: i, responses, score: Math.min(...responses.map(r => r.mean))});
        }
        if (result.budget_exhausted) { row.status = 'incomplete_budget'; break; }
        if (row.status !== 'running') continue;
        row.status = 'completed';
        row.score = row.worlds.reduce((sum, world) => sum + world.score / worlds.length, 0);
        if (!result.supported || row.score > result.selected_score ||
                row.score === result.selected_score && command < result.selected_command) {
            result.supported = true; result.selected_command = command; result.selected_score = row.score;
        }
    }
    result.elapsed_seconds = (performance.now() - started) / 1000;
    result.comparison_complete = !result.budget_exhausted && result.candidates.every(row => row.status === 'completed');
    return result;
}

async function searchOpening(view, options, gate) {
    const settings = searchSettings(options), started = performance.now();
    const rng = new PRNG(settings.seed.join(',')), worlds = [], proposals = [];
    const candidateProposal = proposeOpeningTurns(view, settings.candidates, rng, settings.proposal_mode);
    const commands = candidateProposal.commands;
    for (let i = 0; i < settings.worlds; ++i) {
        await gate?.waitForLaunch();
        const world = constructOpeningWorld(view, {seed: nextSeed(rng), chance_seed: nextSeed(rng), attempts: 8});
        proposals.push({supported: world.supported, attempts: world.attempts, unsupported: world.unsupported});
        if (!world.supported || performance.now() - started >= settings.max_elapsed_ms)
            return {supported: false, selected_command: null, reason: world.supported ? 'construction_budget_exhausted' :
                'opening_world_construction_failed', proposals, training_eligible: false};
        const randomness = {mode: 'sample', seed: nextSeed(rng)};
        const battle = restorePosition(world.position, randomness);
        try {
            const opponent = view.side === 'p1' ? 'p2' : 'p1';
            const proposal = proposeOpeningTurns(playerObservation(battle, opponent), settings.responses, rng, settings.response_proposal_mode);
            worlds.push({position: world.position, responses: proposal.commands, proposal});
        } finally { battle.destroy(); }
    }
    const remaining = Math.max(1, Math.floor(settings.max_elapsed_ms - (performance.now() - started)));
    const result = await compareOpeningTurns(view, worlds, commands, {...settings, max_elapsed_ms: remaining}, gate);
    return {...result, scope: 'observation_only_opening_search', settings, proposals,
        candidate_coverage: candidateProposal.coverage, response_coverage: worlds.map(world => world.proposal.coverage),
        elapsed_seconds: (performance.now() - started) / 1000,
        limitations: ['Uniform rule-only worlds are not a calibrated posterior.',
            'Bounded proposal subsets can miss important turns or opposing responses; coverage is not strength.',
            'Pure-action worst-response scoring is not a mixed-strategy equilibrium.',
            'World-conditional rejections and unfinished turns are excluded, not assigned invented values.',
            'HP/fainting scores omit positional and long-term value; they are not win probabilities.']};
}

function compareOpeningTurns(view, worlds, commands, options, gate) {
    return compareTurns(view, worlds, commands, options, gate, openingActions);
}

function compareOrdinaryTurns(view, worlds, commands, options, gate) {
    return compareTurns(view, worlds, commands, options, gate, ordinaryActions);
}

module.exports = {searchOpening, compareOpeningTurns, compareOrdinaryTurns, searchSettings};
