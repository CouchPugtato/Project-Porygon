'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {searchMoveHistory} = require('./move_history_search');
const {playerObservation, restorePosition} = require('./position_simulator');
const {opposingSwitchSpecs, opposingSwitchFixture} = require('./fixtures/opponent_switch_history');
const {comparable} = require('./fixtures/position_simulator');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const WORLD = {seed: [332, 1, 2, 3], chance_seed: [332, 4, 5, 6]};
const SEARCH = {seed: [332, 1, 2, 3], worlds: 1, candidates: 8, responses: 4,
    chance_samples: 2, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [332, 7, 8, 9]};

async function runOpponentSwitchExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'opposing_voluntary_switch_history', version: 1, passed: false, status: 'running',
        scope: 'ordinary_moves_and_voluntary_switches_by_both_players', settings: {world: WORLD, search: SEARCH,
            workers: 1, experiment_budget_seconds: 30}, published_sets_used: false, training_eligible: false,
        strength_claim: false, runtime_integration: false, cases: [], invariant_hidden_world_pairs: 0,
        expected_abstentions: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s opponent switch experiment budget exceeded');
    };
    try {
        for (const spec of opposingSwitchSpecs()) {
            await gate.waitForLaunch(); checkBudget();
            const {battle, history} = opposingSwitchFixture(spec);
            try {
                assertLegalTeams(battle);
                const original = structuredClone(history), world = constructMoveHistoryWorld(history, WORLD);
                assert(world.supported, JSON.stringify({...world, position: null}));
                const restored = restorePosition(world.position, SAMPLE);
                let differentBenchOrder = false;
                try {
                    const side = spec.side || 'p1', enemy = restored.getSide(side).foe;
                    assertLegalTeams(restored); assert.deepEqual(playerObservation(restored, side), history.observations.at(-1));
                    if (spec.name === 'opponent_damage_and_pp') {
                        const snorlax = enemy.pokemon.find(mon => mon.species.id === 'snorlax');
                        assert(snorlax.hp < snorlax.maxhp);
                        const tackle = snorlax.moveSlots.find(move => move.id === 'tackle');
                        assert.equal(tackle.maxpp - tackle.pp, 1);
                    }
                    if (spec.name === 'both_sides_switch_with_different_bench_order') {
                        assert.notDeepEqual(enemy.pokemon.slice(2).map(mon => mon.fullname),
                            battle.p2.pokemon.slice(2).map(mon => mon.fullname));
                        differentBenchOrder = true;
                    }
                } finally { restored.destroy(); }
                const search = await searchMoveHistory(history, SEARCH, gate);
                assert(search.supported, JSON.stringify(search)); assert(search.nodes <= SEARCH.max_nodes);
                assert(search.candidates.filter(row => row.status === 'completed').some(row =>
                    row.worlds.some(sample => sample.responses.some(response => response.command.includes('switch')))));
                assert.deepEqual(history, original);
                report.cases.push({name: spec.name, passed: true, replay_nodes: world.nodes,
                    different_hidden_bench_order_verified: differentBenchOrder,
                    matched_observations: world.matched_observations, search});
            } finally { battle.destroy(); }
        }
        const spec = {...opposingSwitchSpecs()[0], turns: opposingSwitchSpecs()[0].turns.slice(0, 1)};
        const base = opposingSwitchFixture(spec);
        try {
            const reference = constructMoveHistoryWorld(base.history, WORLD); assert(reference.supported);
            const decision = await searchMoveHistory(base.history, SEARCH, gate); assert(decision.supported);
            for (let i = 0; i < 4; ++i) {
                await gate.waitForLaunch(); checkBudget();
                const other = opposingSwitchFixture({...spec, seed: [332, 30 + i, 4, 5], changeTeams: teams => {
                    Object.assign(teams[1][2], {ability: i % 2 ? 'Unaware' : 'Magic Guard',
                        item: i % 2 ? 'Choice Specs' : 'Safety Goggles',
                        moves: ['cosmicpower', 'wish', 'moonlight', 'minimize'], nature: 'Timid',
                        evs: {hp: 252, spe: 252, def: 4}});
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
            const forced = structuredClone(base.history);
            forced.observations[1].log = forced.observations[1].log.map(line =>
                line.startsWith('|switch|p2a:') ? line.replace('|switch|', '|drag|') : line);
            const active = structuredClone(base.history);
            active.observations[1].log = active.observations[1].log.map(line =>
                line.startsWith('|switch|p2a: Clefable|') ?
                    line.replace('p2a: Clefable|Clefable, M', 'p2a: Snorlax|Snorlax, M') : line);
            for (const input of [forced, active]) {
                const result = constructMoveHistoryWorld(input, WORLD);
                assert.equal(result.supported, false); assert.equal(result.position, null); assert.equal(result.nodes, 0);
                ++report.expected_abstentions;
            }
        } finally { base.battle.destroy(); }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.limitations = ['No forced replacement, pivot or Tera reconstruction.',
            'Unrevealed builds are rule based hypotheses rather than recovered sets or calibrated probabilities.',
            'Nine supported moves only so opposing attack coverage remains incomplete.',
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
        throw new Error('usage: node tools/opponent_switch_history_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runOpponentSwitchExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/ordinary_moves.js', 'tools/opening_actions.js',
            'tools/opening_proposals.js', 'tools/opening_search.js', 'tools/move_history_world.js',
            'tools/move_history_search.js', 'tools/opponent_switch_history.test.js', 'tools/fixtures/move_history.js',
            'tools/fixtures/opponent_switch_history.js', 'tools/fixtures/tactical_reliability.js', 'tools/opponent_switch_history_experiment.js']);
}).catch(error => { console.error(`[opponent-switch-history] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runOpponentSwitchExperiment};
