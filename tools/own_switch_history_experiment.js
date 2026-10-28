'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {searchMoveHistory} = require('./move_history_search');
const {playerObservation, restorePosition} = require('./position_simulator');
const {ownSwitchActions} = require('./opening_actions');
const {ownSwitchSpecs, ownSwitchFixture} = require('./fixtures/own_switch_history');
const {comparable} = require('./fixtures/position_simulator');
const {historyFixture} = require('./fixtures/move_history');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const WORLD = {seed: [331, 1, 2, 3], chance_seed: [331, 4, 5, 6]};
const SEARCH = {seed: [331, 1, 2, 3], worlds: 1, candidates: 8, responses: 2,
    chance_samples: 2, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [331, 7, 8, 9]};

async function runOwnSwitchExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'own_voluntary_switch_history', version: 1, passed: false, status: 'running',
        scope: 'ordinary_moves_and_own_voluntary_switches', settings: {world: WORLD, search: SEARCH,
            workers: 1, experiment_budget_seconds: 30}, published_sets_used: false, training_eligible: false,
        strength_claim: false, runtime_integration: false, cases: [], invariant_hidden_world_pairs: 0,
        expected_abstentions: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s own switch experiment budget exceeded');
    };
    try {
        for (const spec of ownSwitchSpecs()) {
            await gate.waitForLaunch(); checkBudget();
            const {battle, history} = ownSwitchFixture(spec);
            try {
                assertLegalTeams(battle);
                const original = structuredClone(history), world = constructMoveHistoryWorld(history, WORLD);
                assert(world.supported, JSON.stringify({...world, position: null}));
                const restored = restorePosition(world.position, SAMPLE);
                try {
                    assertLegalTeams(restored);
                    assert.deepEqual(playerObservation(restored, spec.side || 'p1'), history.observations.at(-1));
                } finally { restored.destroy(); }
                if (spec.name.startsWith('preserve_')) {
                    const before = history.observations[1], after = history.observations.at(-1);
                    const outgoing = before.request.side.pokemon.find(mon => mon.ident === 'p1: Blastoise');
                    const incoming = after.request.side.pokemon.find(mon => mon.ident === outgoing.ident);
                    assert.equal(incoming.active, true); assert.equal(incoming.condition, outgoing.condition);
                    assert.deepEqual(after.request.active[1].moves, before.request.active[1].moves);
                }
                const search = await searchMoveHistory(history, SEARCH, gate);
                assert(search.supported, JSON.stringify(search)); assert(search.nodes <= SEARCH.max_nodes);
                assert(ownSwitchActions(history.observations.at(-1)).turns.some(turn => turn.command === search.selected_command));
                assert.deepEqual(history, original);
                report.cases.push({name: spec.name, passed: true, replay_nodes: world.nodes,
                    matched_observations: world.matched_observations, search});
            } finally { battle.destroy(); }
        }
        const spec = {...ownSwitchSpecs()[0], turns: ownSwitchSpecs()[0].turns.slice(0, 1)};
        const base = ownSwitchFixture(spec);
        try {
            const reference = constructMoveHistoryWorld(base.history, WORLD); assert(reference.supported);
            const decision = await searchMoveHistory(base.history, SEARCH, gate); assert(decision.supported);
            for (let i = 0; i < 4; ++i) {
                await gate.waitForLaunch(); checkBudget();
                const other = ownSwitchFixture({...spec, seed: [331, 30 + i, 4, 5], changeTeams: teams => {
                    Object.assign(teams[1][0], {ability: i % 2 ? 'Thick Fat' : 'Gluttony',
                        item: i % 2 ? 'Choice Band' : 'Leftovers', moves: ['tackle', 'protect', 'rest', 'bellydrum']});
                    Object.assign(teams[1][2], {species: i % 2 ? 'Slowbro' : 'Alakazam',
                        ability: i % 2 ? 'Own Tempo' : 'Synchronize'});
                }});
                try {
                    assert.deepEqual(other.history, base.history);
                    const world = constructMoveHistoryWorld(other.history, WORLD); assert(world.supported);
                    assert.deepEqual(comparable(world.position), comparable(reference.position));
                    const result = await searchMoveHistory(other.history, SEARCH, gate);
                    assert.equal(result.selected_command, decision.selected_command);
                    assert.deepEqual(result.candidates, decision.candidates);
                    assert.equal(result.nodes, decision.nodes); ++report.invariant_hidden_world_pairs;
                } finally { other.battle.destroy(); }
            }
            const duplicate = structuredClone(base.history); duplicate.own_choices[0] = 'switch 3, switch 3';
            const active = structuredClone(base.history); active.own_choices[0] = 'switch 1, move 2';
            for (const input of [duplicate, active]) {
                const result = constructMoveHistoryWorld(input, WORLD);
                assert.equal(result.supported, false); assert.equal(result.position, null); assert.equal(result.nodes, 0);
                ++report.expected_abstentions;
            }
        } finally { base.battle.destroy(); }
        const opposing = historyFixture({turns: [{p1: 'move 2, move 2', p2: 'switch 3, move 2'}]});
        try {
            const result = constructMoveHistoryWorld(opposing.history, WORLD);
            assert.equal(result.supported, false); assert(result.unsupported.includes('opponent_switch_not_supported'));
            ++report.expected_abstentions;
        } finally { opposing.battle.destroy(); }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.limitations = ['No opponent switch, forced replacement, pivot or Tera reconstruction.',
            'Own switches are evaluated against move only opposing responses.',
            'World conditional trap rejections are not proof of actual hidden trapping.',
            'No live deployment or strength claim from these fixtures.'];
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
        throw new Error('usage: node tools/own_switch_history_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runOwnSwitchExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/ordinary_moves.js', 'tools/opening_actions.js',
            'tools/opening_proposals.js', 'tools/opening_search.js', 'tools/move_history_world.js',
            'tools/move_history_search.js', 'tools/own_switch_history.test.js', 'tools/fixtures/move_history.js',
            'tools/fixtures/own_switch_history.js', 'tools/fixtures/tactical_reliability.js', 'tools/own_switch_history_experiment.js']);
}).catch(error => { console.error(`[own-switch-history] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runOwnSwitchExperiment};
