'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {openingActions} = require('./opening_actions');
const {proposeOpeningTurns} = require('./opening_proposals');
const {compareOpeningTurns} = require('./opening_search');
const {capturePosition, playerObservation} = require('./position_simulator');
const {coverageOpening} = require('./fixtures/opening_search');
const {comparable} = require('./fixtures/position_simulator');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [327, 1, 2, 3], worlds: 1, candidates: 32, responses: 8,
    chance_samples: 2, max_nodes: 256, max_elapsed_ms: 5000};

async function runProposalExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'opening_proposal_comparison', version: 1, passed: false, status: 'running',
        settings: {...SETTINGS, proposed_choices: 8, trials_per_fixture: 4, fixtures: 3,
            workers: 1, experiment_budget_seconds: 30},
        scope: 'fresh_mechanics_fixtures_not_a_held_out_strength_benchmark',
        published_sets_used: false, training_eligible: false, strength_claim: false,
        cases: [], total_simulated_nodes: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s proposal comparison budget exceeded');
    };
    try {
        for (let index = 0; index < 3; ++index) {
            await gate.waitForLaunch(); checkBudget();
            const battle = coverageOpening(index);
            try {
                const view = playerObservation(battle, 'p1'), opponentView = playerObservation(battle, 'p2');
                const position = capturePosition(battle, 'fixture'), root = comparable(position);
                const options = {...SETTINGS, seed: [327, index, 2, 3]};
                const fixedResponses = ['move 1 1, move 1 2', 'move 2, move 1 2'];
                const evaluate = async (commands, responses) => {
                    checkBudget();
                    const result = await compareOpeningTurns(view, [{position, responses}], commands, options, gate);
                    assert.equal(result.comparison_complete, true, JSON.stringify(result));
                    report.total_simulated_nodes += result.nodes;
                    return result;
                };
                const fullPool = openingActions(view).turns.map(turn => turn.command);
                const referenceRows = [];
                for (let offset = 0; offset < fullPool.length; offset += 32) {
                    const result = await evaluate(fullPool.slice(offset, offset + 32), fixedResponses);
                    referenceRows.push(...result.candidates);
                }
                referenceRows.sort((a, b) => b.score - a.score || a.command.localeCompare(b.command));
                const best = referenceRows[0];
                // Isolate opposing coverage: keep the same friendly turn in both
                // modes and compare against every opposing command in this fixture.
                const opponentPool = openingActions(opponentView).turns.map(turn => turn.command);
                let worstResponseScore = Infinity;
                for (let offset = 0; offset < opponentPool.length; offset += 8) {
                    const result = await evaluate([best.command], opponentPool.slice(offset, offset + 8));
                    worstResponseScore = Math.min(worstResponseScore, result.selected_score);
                }
                const record = {index, friendly_pool: fullPool.length, opponent_pool: opponentPool.length,
                    reference_best_score: best.score, fixed_friendly_command: best.command,
                    reference_worst_response_score: worstResponseScore, trials: []};
                for (let trial = 0; trial < 4; ++trial) {
                    const comparison = {trial};
                    for (const mode of ['uniform', 'coverage']) {
                        const seed = [327, index, trial, 9];
                        const candidates = proposeOpeningTurns(view, 8, new PRNG(seed.join(',')), mode);
                        const responses = proposeOpeningTurns(opponentView, 8, new PRNG(seed.join(',')), mode);
                        const candidateResult = await evaluate(candidates.commands, fixedResponses);
                        const responseResult = await evaluate([best.command], responses.commands);
                        // Chunking/order must not change a shared matchup's samples.
                        for (const row of candidateResult.candidates)
                            assert.equal(row.score, referenceRows.find(reference => reference.command === row.command).score);
                        const regret = best.score - candidateResult.selected_score;
                        const missedThreat = responseResult.selected_score - worstResponseScore;
                        assert(regret >= -1e-10); assert(missedThreat >= -1e-10);
                        comparison[mode] = {candidate_nodes: candidateResult.nodes, response_nodes: responseResult.nodes,
                            candidate_regret: Math.max(0, regret), missed_response_threat: Math.max(0, missedThreat),
                            selected_command: candidateResult.selected_command,
                            candidate_coverage: candidates.coverage, response_coverage: responses.coverage};
                    }
                    assert.equal(comparison.uniform.candidate_nodes, comparison.coverage.candidate_nodes);
                    assert.equal(comparison.uniform.response_nodes, comparison.coverage.response_nodes);
                    record.trials.push(comparison);
                }
                assert.deepEqual(comparable(position), root);
                report.cases.push(record);
            } finally { battle.destroy(); }
        }
        const trials = report.cases.flatMap(record => record.trials);
        report.summary = Object.fromEntries(['uniform', 'coverage'].map(mode => [mode, {
            trials: trials.length,
            mean_candidate_regret: trials.reduce((sum, trial) => sum + trial[mode].candidate_regret, 0) / trials.length,
            mean_missed_response_threat: trials.reduce((sum, trial) => sum + trial[mode].missed_response_threat, 0) / trials.length,
            mean_missing_candidate_categories: trials.reduce((sum, trial) =>
                sum + trial[mode].candidate_coverage.missing.length, 0) / trials.length,
            mean_missing_response_categories: trials.reduce((sum, trial) =>
                sum + trial[mode].response_coverage.missing.length, 0) / trials.length,
        } ]));
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
        throw new Error('usage: node tools/opening_proposals_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runProposalExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/opening_actions.js',
            'tools/opening_search.js', 'tools/opening_search.test.js', 'tools/opening_proposals.js',
            'tools/opening_proposals.test.js', 'tools/fixtures/opening_search.js', 'tools/opening_proposals_experiment.js']);
}).catch(error => { console.error(`[opening-proposals] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runProposalExperiment};
