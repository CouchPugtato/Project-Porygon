'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {playerObservation, restorePosition} = require('./position_simulator');
const {historyFixture} = require('./fixtures/move_history');
const {comparable} = require('./fixtures/position_simulator');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [329, 1, 2, 3], chance_seed: [329, 4, 5, 6],
    world_attempts: 16, chance_attempts: 16, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [329, 7, 8, 9]};

async function runHistoryExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'ordinary_move_history_worlds', version: 1, status: 'running', passed: false,
        scope: 'up_to_three_completed_turns_with_explicit_ordinary_move_subset', settings: {...SETTINGS,
            workers: 1, experiment_budget_seconds: 30}, published_sets_used: false,
        training_eligible: false, strength_claim: false, search_ready: false,
        cases: [], invariant_hidden_world_pairs: 0, expected_abstentions: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s move history experiment budget exceeded');
    };
    try {
        const specs = [
            {name: 'protection'}, {name: 'opposite_perspective', side: 'p2'},
            {name: 'outgoing_damage', turns: [{p1: 'move 2, move 1 1', p2: 'move 1 1, move 2'}]},
            {name: 'incoming_damage', turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
            {name: 'own_status', status: true, turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
            {name: 'two_completed_turns', turns: [
                {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
                {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
            ]},
            {name: 'three_completed_turns', turns: [
                {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
                {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
                {p1: 'move 2, move 2', p2: 'move 2, move 2'},
            ]},
        ];
        for (const spec of specs) {
            await gate.waitForLaunch(); checkBudget();
            const {battle, history} = historyFixture(spec);
            try {
                assertLegalTeams(battle);
                const before = structuredClone(history), world = constructMoveHistoryWorld(history, SETTINGS);
                assert(world.supported, JSON.stringify({...world, position: null}));
                const restored = restorePosition(world.position, SAMPLE);
                try {
                    assertLegalTeams(restored);
                    assert.deepEqual(playerObservation(restored, spec.side || 'p1'), history.observations.at(-1));
                } finally { restored.destroy(); }
                assert.deepEqual(history, before);
                report.cases.push({name: spec.name, passed: true, nodes: world.nodes,
                    world_attempts: world.world_attempts, rejected: world.rejected,
                    matched_observations: world.matched_observations, history_sha256: world.history_sha256,
                    elapsed_seconds: world.elapsed_seconds});
            } finally { battle.destroy(); }
        }
        const base = historyFixture();
        try {
            const reference = constructMoveHistoryWorld(base.history, SETTINGS); assert(reference.supported);
            for (let i = 0; i < 8; ++i) {
                await gate.waitForLaunch(); checkBudget();
                const other = historyFixture({seed: [329, 30 + i, 4, 5], changeTeams: teams => {
                    Object.assign(teams[1][0], {ability: i % 2 ? 'Thick Fat' : 'Gluttony',
                        item: i % 2 ? 'Choice Band' : 'Leftovers', moves: ['tackle', 'bellydrum', 'rest']});
                    Object.assign(teams[1][2], {species: i % 2 ? 'Slowbro' : 'Alakazam',
                        ability: i % 2 ? 'Own Tempo' : 'Synchronize'});
                }});
                try {
                    assert.deepEqual(other.history, base.history);
                    const world = constructMoveHistoryWorld(other.history, SETTINGS); assert(world.supported);
                    assert.deepEqual(comparable(world.position), comparable(reference.position));
                    assert.equal(world.nodes, reference.nodes); ++report.invariant_hidden_world_pairs;
                } finally { other.battle.destroy(); }
            }
            const tera = structuredClone(base.history);
            tera.observations[1].log.push('|-terastallize|p2a: Snorlax|Normal');
            const switched = structuredClone(base.history); switched.own_choices[0] = 'switch 3, move 2';
            const invalidPp = structuredClone(base.history); invalidPp.observations[1].request.active[0].moves[0].pp = 999;
            const missingAction = structuredClone(base.history);
            missingAction.observations[1].log = missingAction.observations[1].log.filter(line => !line.startsWith('|move|p2a:'));
            for (const history of [tera, switched, invalidPp, missingAction]) {
                const world = constructMoveHistoryWorld(history,
                    {...SETTINGS, world_attempts: 2, chance_attempts: 2, max_nodes: 2});
                assert.equal(world.supported, false); assert.equal(world.position, null);
                assert(world.unsupported.length); assert(world.nodes <= 2); ++report.expected_abstentions;
            }
        } finally { base.battle.destroy(); }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.limitations = ['No pivot, replacement, switching or transformation reconstruction.',
            'Requires the opening observation and accepted own commands at each supplied move boundary.',
            'Only nine explicitly supported ordinary moves; opposing unexecuted choices remain unknown.',
            'Exact rejection sampling can exhaust its budget on valid histories.',
            'No planner integration, posterior calibration or strength evaluation.'];
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
        throw new Error('usage: node tools/move_history_world_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runHistoryExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/move_history_world.js',
            'tools/move_history_world.test.js', 'tools/fixtures/move_history.js',
            'tools/fixtures/tactical_reliability.js', 'tools/move_history_world_experiment.js']);
}).catch(error => { console.error(`[move-history-world] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runHistoryExperiment};
