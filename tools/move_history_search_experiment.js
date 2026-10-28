'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {searchMoveHistory} = require('./move_history_search');
const {ownSwitchActions} = require('./opening_actions');
const {historyFixture} = require('./fixtures/move_history');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [330, 1, 2, 3], worlds: 2, candidates: 6, responses: 2,
    chance_samples: 2, max_nodes: 256, max_elapsed_ms: 5000};

async function runHistorySearchExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'ordinary_move_history_search', version: 1, passed: false, status: 'running',
        scope: 'bounded_observation_only_later_turn_planning', settings: {...SETTINGS,
            workers: 1, experiment_budget_seconds: 30}, published_sets_used: false,
        training_eligible: false, strength_claim: false, runtime_integration: false,
        cases: [], invariant_hidden_world_pairs: 0, expected_fallbacks: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s history search experiment budget exceeded');
    };
    try {
        const specs = [
            {name: 'turn_two'},
            {name: 'turn_three', turns: [
                {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
                {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
            ]},
            {name: 'damaged_active', turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
            {name: 'paralyzed_active', status: true, turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
            {name: 'opposite_perspective', side: 'p2'},
        ];
        for (const spec of specs) {
            await gate.waitForLaunch(); checkBudget();
            const {battle, history} = historyFixture(spec);
            try {
                const before = structuredClone(history), result = await searchMoveHistory(history, SETTINGS, gate);
                assert(result.supported, JSON.stringify(result));
                assert.equal(result.fixture_worlds, false); assert.equal(result.fallback_required, false);
                assert.equal(result.root_turn, history.observations.at(-1).turn);
                assert.equal(result.nodes, result.reconstruction_nodes + result.search_nodes);
                assert(result.nodes <= SETTINGS.max_nodes);
                assert(ownSwitchActions(history.observations.at(-1)).turns.some(turn => turn.command === result.selected_command));
                assert.deepEqual(history, before);
                report.cases.push({name: spec.name, passed: true, result});
            } finally { battle.destroy(); }
        }
        const base = historyFixture();
        try {
            const reference = await searchMoveHistory(base.history, SETTINGS, gate); assert(reference.supported);
            for (let i = 0; i < 4; ++i) {
                await gate.waitForLaunch(); checkBudget();
                const other = historyFixture({seed: [330, 30 + i, 4, 5], changeTeams: teams => {
                    Object.assign(teams[1][0], {ability: i % 2 ? 'Thick Fat' : 'Gluttony',
                        item: i % 2 ? 'Leftovers' : 'Choice Band', moves: ['tackle', 'rest', 'bellydrum']});
                    Object.assign(teams[1][2], {species: i % 2 ? 'Slowbro' : 'Alakazam',
                        ability: i % 2 ? 'Own Tempo' : 'Synchronize'});
                }});
                try {
                    assert.deepEqual(other.history, base.history);
                    const result = await searchMoveHistory(other.history, SETTINGS, gate);
                    assert.equal(result.selected_command, reference.selected_command);
                    assert.deepEqual(result.candidates, reference.candidates);
                    assert.equal(result.nodes, reference.nodes); ++report.invariant_hidden_world_pairs;
                } finally { other.battle.destroy(); }
            }
            const unsupported = structuredClone(base.history); unsupported.own_choices[0] = 'switch 3, move 2';
            for (const [history, settings] of [[unsupported, SETTINGS], [base.history, {...SETTINGS, max_nodes: 1}]]) {
                const result = await searchMoveHistory(history, settings, gate);
                assert.equal(result.supported, false); assert.equal(result.selected_command, null);
                assert.equal(result.fallback_required, true); assert(result.nodes <= settings.max_nodes);
                ++report.expected_fallbacks;
            }
        } finally { base.battle.destroy(); }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.limitations = ['Nine ordinary moves only and at most three completed historical turns.',
            'Own voluntary switches only with no opposing switch, Tera, pivot or replacement policy.',
            'Caller retains the baseline player on reconstruction or search abstention.',
            'No strength claim from these controlled legality and isolation checks.'];
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
        throw new Error('usage: node tools/move_history_search_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runHistorySearchExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/ordinary_moves.js',
            'tools/opening_actions.js', 'tools/opening_proposals.js', 'tools/opening_search.js',
            'tools/move_history_world.js', 'tools/move_history_search.js', 'tools/move_history_search.test.js',
            'tools/fixtures/move_history.js', 'tools/move_history_search_experiment.js']);
}).catch(error => { console.error(`[move-history-search] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runHistorySearchExperiment};
