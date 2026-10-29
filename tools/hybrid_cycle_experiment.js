'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const {isDeepStrictEqual} = require('node:util');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {playerObservation, pendingSides} = require('./position_simulator');
const {HybridPlanner} = require('./hybrid_planner');
const {decisionActions} = require('./planner_actions');
const {nextSeed} = require('./planner_beliefs');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');
const {writeJsonAtomic} = require('./counterfactual_rollout');
const {summarize} = require('./tactical_teacher_benchmark');

const CONFIG = 'config/experimental/hybrid_cycle_smoke.json';
const SOURCES = ['tools/hybrid_cycle_experiment.js', 'tools/hybrid_cycle.test.js', 'tools/hybrid_planner.js',
    'tools/planner_actions.js', 'tools/planner_beliefs.js', 'tools/planner_search.js', 'tools/opening_actions.js',
    'tools/opening_proposals.js', 'tools/opening_world.js', 'tools/belief_state.js', 'tools/replacement_actions.js',
    'tools/fixtures/hybrid_cycle.js', 'tools/fixtures/tactical_reliability.js', 'tools/tactical_teacher_benchmark.js'];

function validateConfig(config) {
    const limits = {games: 8, seed: 2147483647, max_boundaries: 256, experiment_budget_seconds: 120};
    if (!config || Object.keys(config).some(key => ![...Object.keys(limits), 'beliefs', 'search'].includes(key)) ||
            Object.entries(limits).some(([key, max]) => !Number.isInteger(config[key]) || config[key] < 1 || config[key] > max))
        throw new Error('invalid bounded hybrid experiment config');
    // constructors own the detailed planner setting limits
    new HybridPlanner({seed: [1, 2, 3, 4], beliefs: config.beliefs, search: config.search}, () => 'unused');
    return config;
}

async function playFixtureGame(config, index, gate, checkBudget = () => {}) {
    const rng = new PRNG(`${config.seed & 65535},${index + 1},3,4`);
    const candidateSide = index % 2 ? 'p2' : 'p1', opponent = candidateSide === 'p1' ? 'p2' : 'p1';
    const battleSeed = nextSeed(rng), battle = createHybridFixture({seed: battleSeed});
    const randomCommand = view => {
        const pool = decisionActions(view).turns;
        if (!pool.length) throw new Error('request random fallback has no supported action');
        return pool[rng.random(pool.length)].command;
    };
    const planner = new HybridPlanner({seed: nextSeed(rng), beliefs: config.beliefs, search: config.search},
        randomCommand, gate);
    const record = {index, opponent: 'request_random', teacher_side: candidateSide === 'p1' ? 'a' : 'b',
        policy_mode: 'hybrid', status: 'running', result: null, choice_rejections: [],
        training_eligible: false, boundaries: 0, phases: {move: 0, switch: 0}, decisions: [],
        battle_seed: battleSeed, choices: []};
    let acceptedOwn = null;
    try {
        assertLegalTeams(battle);
        for (let boundary = 0; boundary < config.max_boundaries && !battle.ended; ++boundary) {
            await gate?.waitForLaunch(); checkBudget();
            const view = playerObservation(battle, candidateSide), pending = pendingSides(battle);
            const decision = await planner.decide(view, acceptedOwn);
            const choices = {};
            if (pending.includes(candidateSide)) choices[candidateSide] = decision.command;
            if (pending.includes(opponent)) choices[opponent] = randomCommand(playerObservation(battle, opponent));
            record.decisions.push({turn: view.turn, phase: view.phase, source: decision.source,
                reason: decision.reason || null, command: decision.command,
                belief_particles: decision.belief.particles, retained_particles: decision.belief.retained,
                belief_nodes: decision.belief.nodes, search_nodes: decision.search?.nodes || 0});
            acceptedOwn = choices[candidateSide] || null;
            record.choices.push(choices);
            for (const side of pending) {
                try { assert(battle.choose(side, choices[side])); }
                catch (error) {
                    record.choice_rejections.push({side, command: choices[side], message: error.message});
                    throw error;
                }
            }
            ++record.boundaries; ++record.phases[view.phase];
        }
        if (!battle.ended) throw new Error('bounded battle did not reach a terminal result');
        record.result = {result: !battle.winner ? 'draw' :
            battle.winner === battle.getSide(candidateSide).name ? 'win' : 'loss'};
        record.final_observations = Object.fromEntries(['p1', 'p2'].map(side => [side, playerObservation(battle, side)]));
        record.status = 'completed';
    } catch (error) {
        record.status = 'failed'; record.failure = error.stack || error.message;
    } finally {
        record.coverage = {...planner.stats, search_rate: planner.stats.decisions ?
            planner.stats.searched / planner.stats.decisions : 0};
        battle.destroy();
    }
    return record;
}

function validateSaved(saved, identity, config) {
    if (!isDeepStrictEqual(saved.identity, identity) || !isDeepStrictEqual(saved.config, config) ||
            !Array.isArray(saved.games) || saved.games.length > config.games ||
            saved.games.some((game, i) => game.index !== i || !['completed', 'failed'].includes(game.status)))
        throw new Error('saved hybrid experiment is incompatible or corrupt');
    for (const game of saved.games) {
        if (game.status !== 'completed') throw new Error('saved failed game requires a new run name for another attempt');
        const rng = new PRNG(`${config.seed & 65535},${game.index + 1},3,4`), expectedSeed = nextSeed(rng);
        if (!isDeepStrictEqual(expectedSeed, game.battle_seed) || !Array.isArray(game.choices) ||
                game.choices.length !== game.boundaries || game.boundaries > config.max_boundaries ||
                game.decisions?.length !== game.boundaries || game.choice_rejections?.length !== 0)
            throw new Error('saved hybrid game trace is incomplete');
        const battle = createHybridFixture({seed: expectedSeed});
        try {
            for (const choices of game.choices) {
                const pending = pendingSides(battle);
                if (Object.keys(choices).length !== pending.length || pending.some(side => typeof choices[side] !== 'string'))
                    throw new Error('saved hybrid choices do not match pending players');
                for (const side of pending) assert(battle.choose(side, choices[side]));
            }
            const candidate = game.index % 2 ? 'p2' : 'p1';
            const result = !battle.winner ? 'draw' : battle.winner === battle.getSide(candidate).name ? 'win' : 'loss';
            if (!battle.ended || game.result?.result !== result || game.teacher_side !== (candidate === 'p1' ? 'a' : 'b') ||
                    !isDeepStrictEqual(game.final_observations,
                        Object.fromEntries(['p1', 'p2'].map(side => [side, playerObservation(battle, side)]))))
                throw new Error('saved hybrid terminal replay differs');
        } finally { battle.destroy(); }
    }
    return saved.games;
}

async function runCycleExperiment(gate, context, config) {
    const started = performance.now(), savedPath = path.join(context.directory, 'games.json');
    const report = {diagnostic: 'bounded_hybrid_full_battle_smoke', passed: false, status: 'running',
        settings: config, workers: 1, teams: 'hand_authored_mechanics_fixtures',
        published_sets_used: false, training_eligible: false, strength_claim: false,
        live_bridge_integrated: false, games: [], failure: null};
    const checkBudget = () => {
        if (performance.now() - started > config.experiment_budget_seconds * 1000)
            throw new Error('bounded hybrid experiment elapsed budget exhausted');
    };
    try {
        try { report.games = validateSaved(JSON.parse(await fs.readFile(savedPath, 'utf8')), context.identity, config); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        for (let i = report.games.length; i < config.games; ++i) {
            report.games.push(await playFixtureGame(config, i, gate, checkBudget));
            await writeJsonAtomic(savedPath, {identity: context.identity, config, games: report.games});
            const game = report.games.at(-1);
            console.log(`[hybrid-cycle] games=${i + 1}/${config.games} status=${game.status} ` +
                `searched=${game.coverage.searched}/${game.coverage.decisions} elapsed=${((performance.now() - started) / 1000).toFixed(1)}s`);
            if (game.status !== 'completed') throw new Error(game.failure);
        }
        report.results = summarize(report.games, ['request_random']);
        report.loaded_set_generators = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generators, []);
        assert.equal(report.results.request_random.valid_games, config.games);
        assert(report.games.some(game => game.coverage.searched > 0));
        report.passed = true; report.status = 'completed';
    } catch (error) { report.status = 'failed'; report.failure = error.stack || error.message; }
    report.elapsed_seconds = (performance.now() - started) / 1000;
    report.observed_process_rss_mib = process.memoryUsage().rss / 1024 ** 2;
    report.limitations = ['Small fixture matches measure integration rather than competitive strength.',
        'Exact observation rejection can exhaust the rule only proposal budget and require fallback.',
        'Uniform worlds and pure action worst response search are not calibrated beliefs or equilibrium play.',
        'Native move execution does not guarantee reconstruction coverage for every mechanic.',
        'No live process bridge or default runtime policy has been changed.'];
    return report;
}

async function main(argv) {
    if (![2, 4].includes(argv.length) || argv[0] !== '--run-name' || argv.length === 4 && argv[2] !== '--config')
        throw new Error('usage: node tools/hybrid_cycle_experiment.js --run-name run_NNNN_name [--config path]');
    const configPath = argv[3] || CONFIG;
    const config = validateConfig(JSON.parse(await fs.readFile(configPath, 'utf8')));
    return publishExperiment(argv[1], (gate, context) => runCycleExperiment(gate, context, config),
        [...SOURCES, configPath]);
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[hybrid-cycle] ${error.stack || error.message}`); process.exitCode = 1;
});

module.exports = {playFixtureGame, runCycleExperiment, validateConfig, validateSaved};
