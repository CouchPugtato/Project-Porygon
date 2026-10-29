'use strict';

const {performance} = require('node:perf_hooks');
const {isDeepStrictEqual} = require('node:util');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {restorePosition, playerObservation, pendingSides, advancePosition} = require('./position_simulator');
const {selectCommands, UnsupportedPlannerRequest} = require('./planner_actions');
const {nextSeed, choiceFailure} = require('./planner_beliefs');
const {evaluatePosition} = require('./planner_value');
const {solveBeliefGame} = require('./planner_matrix');

function plannerSettings(options) {
    const settings = {candidates: 8, responses: 3, chance_samples: 2, continuation_choices: 2,
        max_nodes: 256, max_elapsed_ms: 2000, depth_turns: 1, evaluator: 'material',
        strategy: 'pure', adaptive: false, ...options};
    const limits = {candidates: 32, responses: 8, chance_samples: 8, continuation_choices: 4,
        max_nodes: 1024, max_elapsed_ms: 20000, depth_turns: 3};
    if (!Array.isArray(settings.seed) || settings.seed.length !== 4 ||
            settings.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535) ||
            !['material', 'positional'].includes(settings.evaluator) || !['pure', 'mixed'].includes(settings.strategy) ||
            typeof settings.adaptive !== 'boolean' ||
            Object.keys(settings).some(key => !['seed', 'evaluator', 'strategy', 'adaptive'].includes(key) && !Object.hasOwn(limits, key)) ||
            Object.entries(limits).some(([key, max]) => !Number.isInteger(settings[key]) ||
                settings[key] < 1 || settings[key] > max)) throw new Error('invalid bounded planner settings');
    return settings;
}

async function searchBeliefWorlds(view, worlds, options, gate) {
    const settings = plannerSettings(options);
    if (!Array.isArray(worlds) || worlds.length > 4) throw new Error('at most four belief worlds required');
    const started = performance.now(), rng = new PRNG(settings.seed.join(','));
    const result = {supported: false, selected_command: null, selected_score: null, candidates: [], nodes: 0,
        budget_exhausted: false, comparison_complete: false, settings, decision_phase: view.phase,
        objective: 'uniform_world_mean_of_worst_response_chance_mean',
        leaf_score: 'hp_fraction_plus_two_per_survivor_delta_terminal_plus_or_minus_100',
        calibrated_beliefs: false, training_eligible: false, strength_claim: false};
    result.fixture_worlds = worlds.some(position => position.origin === 'fixture');
    const finish = reason => ({...result, reason, fallback_required: !result.supported,
        elapsed_seconds: (performance.now() - started) / 1000});
    if (view.ended || view.request?.wait)
        return {...finish(view.ended ? 'battle_ended' : 'waiting'), fallback_required: false};
    if (!worlds.length) return finish('no_consistent_worlds');
    const exhausted = () => result.nodes >= settings.max_nodes || performance.now() - started >= settings.max_elapsed_ms;
    const randomness = () => ({mode: 'sample', seed: nextSeed(rng)});
    const commands = selectCommands(view, settings.candidates, rng);
    const other = view.side === 'p1' ? 'p2' : 'p1';
    const targetTurn = view.turn + settings.depth_turns + (view.phase === 'switch' ? 1 : 0);
    const cache = new Map();
    let cacheBytes = 0;
    result.cache_hits = 0;
    const prepared = worlds.map(position => {
        const battle = restorePosition(position, randomness());
        try {
            if (!isDeepStrictEqual(playerObservation(battle, view.side), view))
                throw new Error('planner worlds must match the supplied observation');
            const opponentView = playerObservation(battle, other);
            return {position, baseline: evaluatePosition(battle, view.side, settings.evaluator),
                responses: pendingSides(battle).includes(other) ?
                    selectCommands(opponentView, settings.responses, rng) : [null]};
        } finally { battle.destroy(); }
    });
    async function transition(position, choices, seed) {
        const key = JSON.stringify([position.state_json, choices, seed]);
        if (cache.has(key)) { ++result.cache_hits; return cache.get(key); }
        if (result.nodes % 16 === 0) await gate?.waitForLaunch();
        if (exhausted()) { result.budget_exhausted = true; return null; }
        ++result.nodes;
        try {
            const next = advancePosition(position, choices, {mode: 'sample', seed});
            const bytes = Buffer.byteLength(key) + 4 * Buffer.byteLength(next.position.state_json);
            if (cacheBytes + bytes <= 4 * 1024 ** 2) { cache.set(key, next); cacheBytes += bytes; }
            return next;
        }
        catch (error) { if (choiceFailure(error)) return null; throw error; }
    }
    async function continuation(position, baseline, depth, seed) {
        if (depth > 6 || exhausted()) {
            if (exhausted()) result.budget_exhausted = true;
            return null;
        }
        const chance = new PRNG(seed.join(','));
        for (let i = 0; i <= depth; ++i) nextSeed(chance);
        const continuationSeed = nextSeed(chance);
        const battle = restorePosition(position, {mode: 'sample', seed: continuationSeed});
        let own, opposing;
        try {
            if (battle.ended) return !battle.winner ? 0 : battle.winner === battle.getSide(view.side).name ? 100 : -100;
            if (battle.requestState === 'move' && battle.turn >= targetTurn)
                return evaluatePosition(battle, view.side, settings.evaluator) - baseline;
            const pending = pendingSides(battle);
            own = pending.includes(view.side) ? selectCommands(playerObservation(battle, view.side),
                settings.continuation_choices, new PRNG(seed.join(','))) : [null];
            opposing = pending.includes(other) ? selectCommands(playerObservation(battle, other),
                settings.continuation_choices, new PRNG(seed.join(','))) : [null];
        } catch (error) {
            if (error instanceof UnsupportedPlannerRequest) return null;
            throw error;
        } finally { battle.destroy(); }
        // bounded complete continuation rows rather than a score for an unfinished queue
        let best = null;
        for (const command of own) {
            let worst = Infinity, complete = true;
            for (const response of opposing) {
                const choices = {};
                if (command !== null) choices[view.side] = command;
                if (response !== null) choices[other] = response;
                const next = await transition(position, choices, continuationSeed);
                const score = next && await continuation(next.position, baseline, depth + 1, seed);
                if (score === null) { complete = false; break; }
                worst = Math.min(worst, score);
            }
            if (!complete) return null;
            if (Number.isFinite(worst)) best = best === null ? worst : Math.max(best, worst);
        }
        return best;
    }
    // common random numbers for every candidate within the same response column
    const seeds = prepared.map(world => world.responses.map(() =>
        Array.from({length: settings.chance_samples}, () => nextSeed(rng))));
    for (const command of commands) {
        const row = {command, status: 'running', score: null, worlds: [], payoffs: []};
        result.candidates.push(row);
        for (const [i, world] of prepared.entries()) {
            const responseScores = [];
            for (const [j, response] of world.responses.entries()) {
                const scores = [];
                for (const seed of seeds[i][j]) {
                    const choices = {[view.side]: command};
                    if (response !== null) choices[other] = response;
                    const next = await transition(world.position, choices, seed);
                    const score = next && await continuation(next.position, world.baseline, 0, seed);
                    if (score === null) { row.status = 'incomplete_or_rejected'; break; }
                    scores.push(score);
                }
                if (row.status !== 'running') break;
                responseScores.push(scores.reduce((sum, score) => sum + score, 0) / scores.length);
            }
            if (row.status !== 'running') break;
            if (!responseScores.length) { row.status = 'incomplete_or_rejected'; break; }
            row.worlds.push(Math.min(...responseScores));
            row.payoffs.push(responseScores);
        }
        if (row.status === 'running') {
            row.status = 'completed'; row.score = row.worlds.reduce((a, b) => a + b, 0) / worlds.length;
            if (!result.supported || row.score > result.selected_score) {
                result.supported = true; result.selected_command = command; result.selected_score = row.score;
            }
        }
        if (result.budget_exhausted) break;
        if (settings.adaptive && result.candidates.length >= 4) {
            const complete = result.candidates.filter(candidate => candidate.status === 'completed').map(candidate => candidate.score).sort((a, b) => b - a);
            if (complete.length >= 2 && complete[0] - complete[1] >= 3) {
                result.adaptive_stop = 'large_heuristic_margin_not_statistical_confidence'; break;
            }
        }
    }
    if (settings.strategy === 'mixed' && result.supported) {
        const complete = result.candidates.filter(row => row.status === 'completed');
        const matrices = prepared.map((_, world) => complete.map(row => row.payoffs[world]));
        result.mixture = solveBeliefGame(matrices);
        result.mixture.commands = complete.map(row => row.command);
        let sample = rng.random();
        const index = result.mixture.policy.findIndex(probability => (sample -= probability) <= 0);
        result.selected_command = complete[index < 0 ? complete.length - 1 : index].command;
        result.selected_score = result.mixture.lower;
        result.objective = 'finite_sampled_belief_game_regret_matching';
    }
    result.comparison_complete = result.candidates.length === commands.length && !result.budget_exhausted && result.candidates.every(row => row.status === 'completed');
    return finish(result.supported ? null : 'no_completed_candidate');
}

module.exports = {searchBeliefWorlds, plannerSettings};
