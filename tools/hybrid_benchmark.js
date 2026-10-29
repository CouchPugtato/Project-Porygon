'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const {isDeepStrictEqual} = require('node:util');
const {createHash} = require('node:crypto');
const {AgentProcess, createResourceGate, acquireRunLock, writeJsonAtomic} = require('./counterfactual_rollout');
const {createBattleAgents, playGame, hashFile} = require('./tactical_teacher_smoke');
const {summarize, isValid} = require('./tactical_teacher_benchmark');
const {HybridAgent} = require('./hybrid_agent');
const {plannerSettings} = require('./planner_search');
const {PlannerBeliefs} = require('./planner_beliefs');

const ROOT = path.resolve(__dirname, '..');
const SOURCES = ['hybrid_benchmark.js', 'hybrid_agent.js', 'hybrid_planner.js', 'planner_beliefs.js',
    'planner_evidence.js', 'planner_recovery.js', 'planner_search.js', 'planner_matrix.js', 'planner_value.js',
    'planner_actions.js', 'opening_world.js', 'opening_actions.js', 'opening_proposals.js', 'belief_state.js',
    'position_simulator.js', 'replacement_actions.js', 'counterfactual_rollout.js', 'tactical_teacher_smoke.js',
    'tactical_teacher_benchmark.js'];

function validateBenchmark(config) {
    if (config.format !== 'gen9randomdoublesbattle' || config.agent_omp_threads !== 1 ||
            !Number.isInteger(config.games_per_side) || config.games_per_side < 1 || config.games_per_side > 1000 ||
            !Number.isInteger(config.seed) || config.seed < 1 || config.seed > 2147480000 ||
            !(config.battle_timeout_seconds >= 1 && config.battle_timeout_seconds <= 300) ||
            !(config.max_system_memory_percent > 0 && config.max_system_memory_percent <= 65) ||
            !(config.max_system_cpu_percent > 0 && config.max_system_cpu_percent <= 65) ||
            !Number.isFinite(config.resource_check_seconds) || config.resource_check_seconds < 0.1 ||
            !Number.isFinite(config.launch_stagger_ms) || config.launch_stagger_ms < 0 ||
            !Array.isArray(config.opponents) || !config.opponents.length ||
            config.opponents.some(opponent => !['random', 'parent', 'tactical'].includes(opponent)) ||
            new Set(config.opponents).size !== config.opponents.length ||
            !Array.isArray(config.profiles) || !config.profiles.length || config.profiles.length > 8 ||
            new Set(config.profiles.map(profile => profile.name)).size !== config.profiles.length)
        throw new Error('invalid bounded balanced hybrid benchmark config');
    for (const profile of config.profiles) {
        if (!/^[a-z0-9_]+$/.test(profile.name) || typeof profile.baseline !== 'boolean') throw new Error('invalid profile');
        plannerSettings({...profile.search, seed: [1, 2, 3, 4]});
        new PlannerBeliefs({...profile.beliefs, seed: [1, 2, 3, 4]});
    }
    return config;
}

function schedule(config, smoke = false) {
    const jobs = [];
    for (const profile of smoke ? config.profiles.filter(p => !p.baseline).slice(0, 1) : config.profiles)
        for (const opponent of smoke ? ['random'] : config.opponents)
            for (let index = 0; index < (smoke ? 2 : config.games_per_side * 2); ++index)
                jobs.push({profile: profile.name, opponent, index});
    return jobs;
}

async function runLiveGame(sim, config, profile, job, gate) {
    await gate.waitForLaunch();
    let wrapper;
    const started = performance.now();
    const game = await playGame(sim, {...config, tactical_teacher: false, candidate_inference_policy: 'sample'}, job.index,
        {opponent: job.opponent === 'tactical' ? 'parent' : job.opponent, matchedTeams: true,
            agentFactory: (options, seeds, side, opponent, observe) => {
                const agents = createBattleAgents(options, seeds, side, opponent, observe);
                if (job.opponent === 'tactical') agents[1 - side] = new AgentProcess(
                    path.resolve(ROOT, config.checkpoint), seeds.inference[1 - side], null, 1, true);
                if (!profile.baseline) {
                    const seed = [config.seed & 65535, job.index + 1, 11, 12];
                    wrapper = new HybridAgent(agents[side], {seed, beliefs: profile.beliefs, search: profile.search}, gate);
                    agents[side] = wrapper;
                }
                return agents;
            }});
    return {...game, opponent: job.opponent, profile: profile.name, policy_mode: profile.baseline ? 'neural' : 'hybrid',
        training_eligible: false, elapsed_seconds: (performance.now() - started) / 1000,
        hybrid_coverage: wrapper ? structuredClone(wrapper.planner.stats) : null,
        first_belief_failure: wrapper ? structuredClone(wrapper.planner.beliefs.firstFailure) : null};
}

function compareProfiles(games, config) {
    return Object.fromEntries(config.profiles.map(profile => {
        const subset = games.filter(game => game.profile === profile.name);
        const decisions = subset.reduce((sum, game) => sum + (game.hybrid_coverage?.decisions || 0), 0);
        const searched = subset.reduce((sum, game) => sum + (game.hybrid_coverage?.searched || 0), 0);
        const seconds = subset.reduce((sum, game) => sum + game.elapsed_seconds, 0);
        return [profile.name, {results: summarize(subset, config.opponents), decisions, searched,
            search_rate: decisions ? searched / decisions : null, elapsed_seconds: seconds,
            note: 'matched teams across profiles and sides are dependent observations not independent evidence'}];
    }));
}

function computeComparison(games, low = 'pure_low', high = 'pure_high') {
    const score = game => game.result.result === 'win' ? 1 : game.result.result === 'draw' ? 0.5 : 0;
    const comparisons = {};
    for (const opponent of new Set(games.map(game => game.opponent))) {
        const pairs = new Map();
        for (const game of games.filter(game => game.opponent === opponent && [low, high].includes(game.profile) && isValid(game))) {
            const key = Math.floor(game.index / 2);
            if (!pairs.has(key)) pairs.set(key, {});
            pairs.get(key)[`${game.profile}_${game.index % 2}`] = game;
        }
        const blocks = [...pairs.values()].filter(pair => [low, high].every(profile =>
            [0, 1].every(side => pair[`${profile}_${side}`])));
        const deltas = blocks.map(pair => [0, 1].reduce((sum, side) => sum +
            (score(pair[`${high}_${side}`]) - score(pair[`${low}_${side}`])) / 2, 0));
        const mean = deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null;
        const error = deltas.length >= 30 ? 1.96 * Math.sqrt(deltas.reduce((sum, value) =>
            sum + (value - mean) ** 2, 0) / (deltas.length - 1) / deltas.length) : null;
        comparisons[opponent] = {low_profile: low, high_profile: high, matched_team_blocks: blocks.length,
            mean_score_delta: mean, exploratory_cluster_interval: error === null ? null :
                {low: mean - error, high: mean + error},
            evidence: error === null ? 'insufficient_paired_blocks' : 'exploratory_not_a_promotion_rule'};
    }
    return comparisons;
}

function recordHash(game) {
    const {record_hash: ignored, ...record} = game;
    return createHash('sha256').update(JSON.stringify(record)).digest('hex');
}

function validateLiveResume(saved, identity, jobs) {
    if (!isDeepStrictEqual(saved.identity, identity) || !Array.isArray(saved.games) || saved.games.length > jobs.length ||
            saved.games.some((game, i) => game.record_hash !== recordHash(game) ||
                !isDeepStrictEqual({profile: game.profile, opponent: game.opponent, index: game.index}, jobs[i])))
        throw new Error('benchmark resume identity schedule or record integrity changed');
    if (saved.games.some(game => game.status !== 'completed'))
        throw new Error('saved failed game retained use a new run name after diagnosing it');
    return saved;
}

async function main(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i += 2) {
        if (!['--run-name', '--config', '--smoke'].includes(argv[i]) || !argv[i + 1]) throw new Error('expected named argument pairs');
        args[argv[i]] = argv[i + 1];
    }
    if (!/^run_\d+_[a-z0-9_]+$/.test(args['--run-name'] || '') ||
            args['--smoke'] !== undefined && !['0', '1'].includes(args['--smoke'])) throw new Error('invalid run name or smoke flag');
    const config = validateBenchmark(JSON.parse(await fs.readFile(path.resolve(ROOT,
        args['--config'] || 'config/experimental/hybrid_benchmark.json'), 'utf8')));
    const smoke = args['--smoke'] === '1', jobs = schedule(config, smoke);
    if (smoke) config.battle_timeout_seconds = Math.min(30, config.battle_timeout_seconds);
    const directory = path.join(ROOT, 'matches/runs', args['--run-name']);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory), filename = path.join(directory, 'benchmark.json');
    try {
        const identity = {config, smoke, executable: await hashFile(path.join(ROOT, 'build-fresh/showdown_client.exe')),
            checkpoint: await hashFile(path.resolve(ROOT, config.checkpoint)), sources: {}};
        for (const source of SOURCES) identity.sources[source] = await hashFile(path.join(__dirname, source));
        // environment set generation isolated from observation only planner inputs
        const sim = require('../external/pokemon-showdown/dist/sim');
        sim.Teams.getGenerator(config.format);
        identity.simulator = {};
        for (const file of Object.keys(require.cache).filter(file => file.startsWith(path.join(ROOT, 'external/pokemon-showdown') + path.sep)))
            identity.simulator[path.relative(ROOT, file)] = await hashFile(file);
        let report = {identity, status: 'running', games: [], training_eligible: false, strength_claim: false};
        try {
            const saved = JSON.parse(await fs.readFile(filename, 'utf8'));
            report = validateLiveResume(saved, identity, jobs);
        } catch (error) { if (error.code !== 'ENOENT') throw error; }
        const gate = createResourceGate({maxSystemMemoryPercent: config.max_system_memory_percent,
            maxSystemCpuPercent: config.max_system_cpu_percent, resourceCheckSeconds: config.resource_check_seconds,
            launchStaggerMilliseconds: config.launch_stagger_ms});
        for (let i = report.games.length; i < jobs.length; ++i) {
            const job = jobs[i], profile = config.profiles.find(profile => profile.name === job.profile);
            const game = await runLiveGame(sim, config, profile, job, gate);
            game.record_hash = recordHash(game); report.games.push(game);
            report.profiles = compareProfiles(report.games, config);
            report.compute_scaling = computeComparison(report.games);
            report.strategy_comparison = computeComparison(report.games, 'pure_high', 'mixed_high');
            report.depth_comparison = computeComparison(report.games, 'mixed_high', 'mixed_deep');
            report.status = game.status !== 'completed' ? 'failed' : report.games.length === jobs.length ? 'completed' : 'running';
            report.all_scheduled_games_valid = report.games.length === jobs.length && report.games.every(isValid);
            await writeJsonAtomic(filename, report);
            console.log(`[hybrid-benchmark] games=${i + 1}/${jobs.length} profile=${profile.name} ` +
                `opponent=${job.opponent} status=${report.games.at(-1).status} elapsed=${report.games.at(-1).elapsed_seconds.toFixed(1)}s`);
            if (report.games.at(-1).status !== 'completed') throw new Error(report.games.at(-1).error || 'live hybrid game failed');
        }
        console.log(`[hybrid-benchmark] report=${filename}`);
    } finally { await lock.release(); }
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[hybrid-benchmark] ${error.stack || error.message}`); process.exitCode = 1;
});

module.exports = {validateBenchmark, schedule, runLiveGame, compareProfiles, computeComparison, recordHash, validateLiveResume};
