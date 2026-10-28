'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {compareOpeningTurns, searchOpening} = require('./opening_search');
const {capturePosition, playerObservation} = require('./position_simulator');
const {createFixture, comparable} = require('./fixtures/position_simulator');
const {tacticalOpening} = require('./fixtures/opening_search');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [326, 1, 2, 3], worlds: 2, candidates: 6, responses: 2,
    chance_samples: 2, max_nodes: 128, max_elapsed_ms: 5000};
const ELECTRIC = 'move 1 1, move 2', OTHER = 'move 1 2, move 2', PASSIVE = 'move 2, move 2';
const WATER = 'move 2, move 1 1', IDLE = 'move 1, move 1', PROTECT = 'move 2, move 1';

async function runSearchExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'opening_one_turn_search', version: 1, status: 'running', passed: false,
        settings: {...SETTINGS, workers: 1, experiment_budget_seconds: 30},
        scope: 'hand_authored_tactics_and_observation_only_opening_integration',
        published_sets_used: false, training_eligible: false, strength_claim: false,
        cases: [], hidden_truth_invariance_pairs: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started >= 30000) throw new Error('30s opening search fixture budget exceeded');
    };
    try {
        const battle = tacticalOpening({protect: true});
        try {
            const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
            const root = comparable(position);
            for (const [name, responses, expected] of [
                ['super_effective_knockout', [IDLE], ELECTRIC],
                ['protect_changes_target', [PROTECT], OTHER],
                ['worst_sampled_response', [IDLE, PROTECT], OTHER],
            ]) {
                await gate.waitForLaunch(); checkBudget();
                const result = await compareOpeningTurns(view, [{position, responses}],
                    [PASSIVE, OTHER, ELECTRIC], SETTINGS, gate);
                assert.equal(result.selected_command, expected);
                assert.equal(result.comparison_complete, true);
                if (name === 'super_effective_knockout') assert(result.selected_score >= 3);
                report.cases.push({name, passed: true, result});
            }
            assert.deepEqual(comparable(position), root);
        } finally { battle.destroy(); }

        const opposingWorlds = ['Water Absorb', 'Volt Absorb'].map(absorb => tacticalOpening({absorb}));
        try {
            await gate.waitForLaunch(); checkBudget();
            const view = playerObservation(opposingWorlds[0], 'p1');
            assert.deepEqual(playerObservation(opposingWorlds[1], 'p1'), view);
            const worlds = opposingWorlds.map(b => ({position: capturePosition(b, 'fixture'), responses: [IDLE]}));
            const result = await compareOpeningTurns(view, worlds, [ELECTRIC, WATER], SETTINGS, gate);
            assert.equal(result.comparison_complete, true);
            const [electric, water] = result.candidates;
            assert(electric.worlds[0].score > water.worlds[0].score);
            assert(water.worlds[1].score > electric.worlds[1].score);
            const clairvoyant = [0, 1].reduce((sum, i) => sum +
                Math.max(...result.candidates.map(row => row.worlds[i].score)) / 2, 0);
            assert(clairvoyant > result.selected_score);
            assert.equal(result.selected_score, Math.max(...result.candidates.map(row => row.score)));
            report.cases.push({name: 'one_shared_action_across_hidden_worlds', passed: true,
                unattainable_clairvoyant_score: clairvoyant, result});
        } finally { opposingWorlds.forEach(b => b.destroy()); }

        const base = createFixture();
        try {
            const view = playerObservation(base, 'p1'), baseline = await searchOpening(view, SETTINGS, gate);
            assert(baseline.supported); assert.equal(baseline.fixture_worlds, false);
            report.observation_only_integration = baseline;
            for (let i = 0; i < 4; ++i) {
                await gate.waitForLaunch(); checkBudget();
                const hidden = createFixture({seed: [326, 30 + i, 4, 5], changeTeams: teams => {
                    Object.assign(teams[1][0], {ability: i % 2 ? 'Thick Fat' : 'Gluttony',
                        item: i % 2 ? 'Leftovers' : 'Choice Band', moves: ['bellydrum', 'rest', 'bodyslam']});
                    Object.assign(teams[1][2], {species: i % 2 ? 'Slowbro' : 'Alakazam',
                        ability: i % 2 ? 'Own Tempo' : 'Synchronize'});
                }});
                try {
                    const hiddenView = playerObservation(hidden, 'p1'); assert.deepEqual(hiddenView, view);
                    const result = await searchOpening(hiddenView, SETTINGS, gate);
                    assert.deepEqual(result.candidates, baseline.candidates);
                    assert.equal(result.selected_command, baseline.selected_command);
                    assert.equal(result.nodes, baseline.nodes);
                    ++report.hidden_truth_invariance_pairs;
                } finally { hidden.destroy(); }
            }
        } finally { base.destroy(); }
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
        throw new Error('usage: node tools/opening_search_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runSearchExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/opening_actions.js',
            'tools/opening_search.js', 'tools/opening_proposals.js', 'tools/opening_search.test.js',
            'tools/fixtures/opening_search.js', 'tools/opening_search_experiment.js']);
}).catch(error => { console.error(`[opening-search] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runSearchExperiment};
