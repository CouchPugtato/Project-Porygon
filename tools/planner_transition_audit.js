'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {isDeepStrictEqual} = require('node:util');
const {advancePosition, restorePosition, playerObservation} = require('./position_simulator');
const {observedResponses, nextSeed, choiceFailure} = require('./planner_beliefs');
const {mismatchEvidence} = require('./planner_evidence');
const {publishExperiment} = require('./position_simulator_experiment');
const {recordHash} = require('./hybrid_benchmark');

async function auditTransition(failure, gate, {samples = 128, max_elapsed_ms = 2000} = {}) {
    if (!Number.isInteger(samples) || samples < 1 || samples > 256 || !Number.isInteger(max_elapsed_ms) ||
            max_elapsed_ms < 1 || max_elapsed_ms > 5000) throw new Error('bounded transition audit settings required');
    const trace = failure.simulation_trace;
    if (!trace || trace.stage !== 'simulation') return {supported: false, reason: 'no_captured_simulation'};
    const started = performance.now(), side = trace.expected.side;
    const index = failure.history.observations.findIndex(view => isDeepStrictEqual(view, trace.expected));
    assert(index > 0, 'captured expected projection must be a recorded boundary');
    const before = failure.history.observations[index - 1], parent = trace.position.state_json;
    const replay = advancePosition(trace.position, trace.choices, {mode: 'sample', seed: trace.seed});
    assert.deepEqual(replay.observations[side], trace.actual, 'captured branch must replay independently');
    const opponent = side === 'p1' ? 'p2' : 'p1';
    const branch = restorePosition(trace.position, {mode: 'sample', seed: trace.seed});
    let inferred, synthetic;
    try {
        assert.deepEqual(playerObservation(branch, side), before);
        inferred = observedResponses(branch, before, trace.expected);
        synthetic = observedResponses(branch, before, trace.actual);
    } finally { branch.destroy(); }
    let syntheticControl = false;
    for (const command of synthetic.slice(0, 16)) {
        const choices = {...trace.choices};
        if (command !== null) choices[opponent] = command;
        try {
            const result = advancePosition(trace.position, choices, {mode: 'sample', seed: trace.seed});
            if (isDeepStrictEqual(result.observations[side], trace.actual)) { syntheticControl = true; break; }
        } catch (error) { if (!choiceFailure(error)) throw error; }
    }
    const report = {supported: true, exact_branch_replay: true, synthetic_inference_control: syntheticControl,
        origin: trace.position.origin, samples: 0, exact_matches: 0, mismatch_counts: {},
        recorded_pair_inferred: inferred.includes(trace.choices[opponent] || null),
        initial_mismatch: mismatchEvidence(trace.actual, trace.expected), closest_mismatch: null,
        training_eligible: false, strength_claim: false};
    const rng = new PRNG('343,1,2,3');
    for (let sample = 0; sample < samples; ++sample) {
        if (sample % 16 === 0) await (gate?.waitForCapacity?.() ?? gate?.waitForLaunch?.());
        if (performance.now() - started >= max_elapsed_ms) break;
        const next = advancePosition(trace.position, trace.choices, {mode: 'sample', seed: nextSeed(rng)});
        ++report.samples;
        const mismatch = mismatchEvidence(next.observations[side], trace.expected);
        if (mismatch.kind === 'matched') ++report.exact_matches;
        else report.mismatch_counts[mismatch.kind] = (report.mismatch_counts[mismatch.kind] || 0) + 1;
        if (!report.closest_mismatch || mismatch.distance < report.closest_mismatch.distance) report.closest_mismatch = mismatch;
    }
    assert.equal(trace.position.state_json, parent, 'audit cannot modify its parent hypothesis');
    report.elapsed_seconds = (performance.now() - started) / 1000;
    report.assessment = !syntheticControl ? 'action_inference_control_failed' : report.exact_matches ?
        'chance_resampling_found_an_exact_match' : 'no_exact_match_in_this_world_and_command_pair_within_budget';
    return report;
}

async function main(argv) {
    if (argv.length !== 4 || argv[0] !== '--run-name' || argv[2] !== '--input')
        throw new Error('expected --run-name NAME --input BENCHMARK_JSON');
    const filename = path.resolve(argv[3]);
    if ((await fs.stat(filename)).size > 4 * 1024 ** 2) throw new Error('transition audit input exceeds the 4 MiB limit');
    const run = JSON.parse(await fs.readFile(filename, 'utf8'));
    if (!Array.isArray(run.games) || run.games.length < 1 || run.games.length > 4)
        throw new Error('transition audit accepts one to four captured games');
    return publishExperiment(argv[1], async gate => {
        const report = {diagnostic: 'independent_first_transition_audit', passed: false, cases: [],
            training_eligible: false, strength_claim: false, failure: null};
        try {
            for (const game of run.games) {
                assert.equal(game.record_hash, recordHash(game), 'captured benchmark record integrity changed');
                assert.equal(game.first_belief_failure?.simulation_trace?.position?.origin, 'belief-sample');
                const result = await auditTransition(game.first_belief_failure, gate);
                report.cases.push({side: game.teacher_side, ...result});
                assert(result.exact_branch_replay && result.synthetic_inference_control);
            }
            assert(report.cases.length > 0); report.passed = true;
        } catch (error) { report.failure = error.stack || error.message; }
        report.limitations = ['Captured snapshots contain sampled hypotheses rather than actual opposing hidden state.',
            'The audit holds the sampled build and command pair fixed while changing chance seeds.',
            'Failure to match within this budget is not proof that no consistent hidden state exists.',
            'No observations are normalized away or patched to force a match.'];
        return report;
    }, ['tools/planner_transition_audit.js', 'tools/hybrid_benchmark.js', 'tools/planner_beliefs.js', 'tools/planner_evidence.js',
        'tools/belief_state.js', 'tools/planner_actions.js', 'tools/opening_actions.js',
        path.relative(path.resolve(__dirname, '..'), filename)]);
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[transition-audit] ${error.stack || error.message}`); process.exitCode = 1;
});

module.exports = {auditTransition};
