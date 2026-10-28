'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {capturePosition, playerObservation, advancePosition, restorePosition} = require('./position_simulator');
const {compareOpeningTurns} = require('./opening_search');
const {proposeOpeningTurns} = require('./opening_proposals');
const {constructOpeningWorld} = require('./opening_world');
const {assessTacticalProbe} = require('./tactical_reliability');
const {tacticalProbes, assertLegalTeams} = require('./fixtures/tactical_reliability');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const OPTIONS = {seed: [328, 1, 2, 3], worlds: 1, candidates: 2, responses: 1, chance_samples: 4};
const SAMPLE = {mode: 'sample', seed: [328, 4, 5, 6]};

async function runReliabilityExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now(), probes = tacticalProbes();
    const report = {diagnostic: 'tactical_reliability', version: 1, passed: false, status: 'running',
        scope: 'controlled_full_state_development_probes', settings: {...OPTIONS,
            workers: 1, proposed_choices: 8, experiment_budget_seconds: 30},
        published_sets_used: false, training_eligible: false, strength_claim: false,
        ready_for_strength: false, cases: [], world_construction_gaps: [], failure: null};
    try {
        for (const probe of probes) {
            await gate.waitForLaunch();
            if (performance.now() - started > 30000) throw new Error('30s tactical reliability budget exceeded');
            const battle = probe.create();
            try {
                assertLegalTeams(battle);
                const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
                const original = position.state_json;
                // private scenario truth confined to simulator mechanics verification
                const next = advancePosition(position, {p1: probe[probe.verify_choice || 'preferred'], p2: probe.response}, SAMPLE);
                const branch = restorePosition(next.position, SAMPLE);
                try { probe.verify(branch); } finally { branch.destroy(); }
                const result = await compareOpeningTurns(view, [{position, responses: [probe.response]}],
                    [probe.preferred, probe.alternative], OPTIONS, gate);
                const proposal = proposeOpeningTurns(view, 8, new PRNG(OPTIONS.seed.join(',')));
                const assessment = assessTacticalProbe(probe, result, proposal.commands);
                assert.equal(position.state_json, original);
                assert.deepEqual(playerObservation(battle, 'p1'), view);
                report.cases.push({name: probe.name, expectation: probe.expectation, mechanics_verified: true,
                    assessment, proposal, comparison: result, parent_unchanged: true});

                if (probe.name === 'protected_target') {
                    const laterView = next.observations.p1;
                    assert(laterView.turn > 1);
                    const world = constructOpeningWorld(laterView,
                        {seed: [328, 7, 8, 9], chance_seed: [328, 10, 11, 12], attempts: 2});
                    assert.equal(world.supported, false); assert.equal(world.position, null);
                    assert(world.unsupported.some(reason => reason.code === 'opening_move_request_only'));
                    report.world_construction_gaps.push({name: 'later_turn_observation', classification: 'world_construction_gap',
                        turn: laterView.turn, unsupported: world.unsupported});
                }
                const elapsed = (performance.now() - started) / 1000;
                console.log(`[tactical-reliability] cases=${report.cases.length}/${probes.length} ` +
                    `classification=${assessment.classification} elapsed=${elapsed.toFixed(1)}s ` +
                    `eta=${(elapsed / report.cases.length * (probes.length - report.cases.length)).toFixed(1)}s`);
            } finally { battle.destroy(); }
        }
        report.classification_counts = {};
        for (const record of [...report.cases.map(probe => probe.assessment), ...report.world_construction_gaps]) {
            const key = record.classification;
            report.classification_counts[key] = (report.classification_counts[key] || 0) + 1;
        }
        report.total_simulated_nodes = report.cases.reduce((sum, probe) => sum + probe.comparison.nodes, 0);
        report.interpretation = [
            'Completed diagnostics do not mean every tactical probe passed.',
            'Exact probe absence does not prove all equally good alternatives were absent.',
            'Field benefit probes test score sensitivity rather than universal action optimality.',
            'Inspected fixtures become development data and cannot serve as fresh strength confirmation.',
            'World reconstruction and continuation gaps block a full battle planning baseline.',
        ];
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.status = 'completed'; report.passed = true;
    } catch (error) {
        report.status = 'failed'; report.failure = error.stack || error.message;
    } finally {
        report.elapsed_seconds = (performance.now() - started) / 1000;
        report.observed_process_rss_mib = process.memoryUsage().rss / 1024 ** 2;
    }
    return report;
}

if (require.main === module) Promise.resolve().then(() => {
    const argv = process.argv.slice(2);
    if (argv.length !== 2 || argv[0] !== '--run-name')
        throw new Error('usage: node tools/tactical_reliability_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runReliabilityExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/opening_actions.js', 'tools/opening_proposals.js',
            'tools/opening_search.js', 'tools/tactical_reliability.js', 'tools/tactical_reliability.test.js',
            'tools/fixtures/tactical_reliability.js', 'tools/tactical_reliability_experiment.js']);
}).catch(error => { console.error(`[tactical-reliability] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runReliabilityExperiment};
