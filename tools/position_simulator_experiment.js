'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {capturePosition, advancePosition} = require('./position_simulator');
const {REPLAY, SELF_BUFFS, createFixture, comparable, checkReplay, scenarios} = require('./fixtures/position_simulator');
const {acquireRunLock, createResourceGate, seedTuple, writeJsonAtomic} = require('./counterfactual_rollout');
const {hashFile} = require('./tactical_teacher_smoke');

const ROOT = path.resolve(__dirname, '..');
const SIMULATOR = path.join(ROOT, 'external', 'pokemon-showdown');
const SAMPLES = 192;
const SEED = 32201;
const RESOURCE_SETTINGS = {maxSystemMemoryPercent: 65, maxSystemCpuPercent: 65,
    resourceCheckSeconds: 2, launchStaggerMilliseconds: 0};

function loadedSetGenerators() {
    return Object.keys(require.cache).filter(file => file.startsWith(SIMULATOR + path.sep) &&
        /[\\/](?:random-battles|random-teams)(?:[\\/.])/.test(file));
}

async function sourceIdentity() {
    const files = [...Object.keys(require.cache).filter(file => file.startsWith(SIMULATOR + path.sep)),
        ...['tools/position_simulator.js', 'tools/position_simulator.test.js',
            'tools/position_simulator_experiment.js', 'tools/fixtures/position_simulator.js',
            'tools/counterfactual_rollout.js', 'tools/tactical_teacher_smoke.js']
            .map(file => path.join(ROOT, file))];
    const hashes = {};
    for (const filename of [...new Set(files)].sort())
        hashes[path.relative(ROOT, filename).replaceAll('\\', '/')] = await hashFile(filename);
    return {node: process.version, platform: process.platform, arch: process.arch,
        simulator_version: require('../external/pokemon-showdown/package.json').version,
        loaded_simulator_and_adapter_sha256: hashes};
}

async function runExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'position_simulator_fixture_experiment', version: 1,
        status: 'running', passed: false, direction_step: 'hybrid_engine_plan.txt: stage 1',
        scope: 'hypothetical_complete_state_fixtures', published_sets_used: false,
        teams: 'hand_authored_mechanics_fixtures', training_eligible: false,
        strength_claim: false, live_agent_ready: false, belief_search_ready: false,
        resource_settings: {...RESOURCE_SETTINGS, workers: 1, chance_samples: SAMPLES,
            active_experiment_budget_seconds: 30}, deterministic_scenarios: [],
        chance: {samples: 0, seed: SEED, expected_accuracy: 0.7, hits: 0,
            reference_matches: 0}, failure: null};
    const parentBattle = createFixture();
    const parent = capturePosition(parentBattle, 'fixture');
    const choices = {p1: 'move 2 1, move 2', p2: SELF_BUFFS};
    const durations = [];
    try {
        for (const scenario of scenarios) report.deterministic_scenarios.push(checkReplay(scenario));
        for (let i = 0; i < SAMPLES; ++i) {
            if (i % 16 === 0) await gate.waitForLaunch();
            if (performance.now() - started > 30000) throw new Error('30s fixture experiment budget exceeded');
            const seed = seedTuple(SEED, i, 0).split(',').map(Number);
            const sampleStarted = performance.now();
            const result = advancePosition(parent, choices, {mode: 'sample', seed});
            durations.push(performance.now() - sampleStarted);
            // Independent uninterrupted construction checks that sampled branch
            // restoration agrees with Showdown, not just with a second adapter call.
            const reference = createFixture();
            try {
                reference.prng = new PRNG(seed.join(','));
                for (const side of ['p1', 'p2']) assert(reference.choose(side, choices[side]));
                assert.deepEqual(comparable(result.position), comparable(capturePosition(reference, 'fixture')));
                ++report.chance.reference_matches;
            } finally { reference.destroy(); }
            ++report.chance.samples;
            if (!result.observations.p1.log.some(line => line.startsWith('|-miss|'))) ++report.chance.hits;
            if ((i + 1) % 32 === 0) {
                const elapsed = (performance.now() - started) / 1000;
                console.log(`[position-simulator] samples=${i + 1}/${SAMPLES} elapsed=${elapsed.toFixed(2)}s ` +
                    `eta=${(elapsed / (i + 1) * (SAMPLES - i - 1)).toFixed(2)}s`);
            }
        }
        assert.deepEqual(comparable(capturePosition(parentBattle, 'fixture')), comparable(parent));
        const chance = report.chance;
        chance.hit_rate = chance.hits / chance.samples;
        chance.deviation_standard_errors = (chance.hit_rate - chance.expected_accuracy) /
            Math.sqrt(chance.expected_accuracy * (1 - chance.expected_accuracy) / chance.samples);
        chance.sanity_check = 'absolute deviation <= 6 binomial standard errors; fixed-seed mechanics check, not strength evidence';
        assert(Math.abs(chance.deviation_standard_errors) <= 6, 'Thunder accuracy inconsistent with the fixture mechanics');
        report.loaded_set_generator_modules = loadedSetGenerators().map(file => path.relative(ROOT, file));
        assert.deepEqual(report.loaded_set_generator_modules, [], 'fixture experiment loaded set-generation code');
        report.status = 'completed'; report.passed = true;
    } catch (error) {
        report.status = 'failed'; report.failure = error.stack || error.message;
    } finally {
        parentBattle.destroy();
        report.elapsed_seconds = (performance.now() - started) / 1000;
        report.observed_process_rss_mib = process.memoryUsage().rss / 1024 ** 2;
        report.system_memory_used_percent = (1 - os.freemem() / os.totalmem()) * 100;
        if (durations.length) {
            const sorted = durations.toSorted((a, b) => a - b);
            report.sampled_branch_timing_ms = {count: sorted.length,
                mean: sorted.reduce((a, b) => a + b, 0) / sorted.length,
                p95: sorted[Math.ceil(sorted.length * .95) - 1], max: sorted.at(-1),
                note: 'restore, validate choices, simulate, capture and project; excludes independent reference and resource waits'};
        }
    }
    report.identity = await sourceIdentity();
    return report;
}

async function main(argv) {
    if (argv.length !== 2 || argv[0] !== '--run-name' || !/^run_\d+_[a-z0-9_]+$/.test(argv[1]))
        throw new Error('usage: node tools/position_simulator_experiment.js --run-name run_NNNN_name');
    const directory = path.join(ROOT, 'models', 'diagnostics', argv[1]);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const filename = path.join(directory, 'report.json');
        try { await fs.stat(filename); throw new Error('report already exists; use a new run name'); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        const report = await runExperiment(createResourceGate(RESOURCE_SETTINGS));
        report.run_name = argv[1];
        await writeJsonAtomic(filename, report);
        console.log(`[position-simulator] passed=${report.passed} report=${filename}`);
        if (!report.passed) throw new Error(report.failure);
    } finally { await lock.release(); }
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[position-simulator] ${error.stack || error.message}`); process.exitCode = 1;
});
module.exports = {loadedSetGenerators, runExperiment};
