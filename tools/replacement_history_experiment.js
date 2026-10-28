'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {replacementActions, isPostKnockoutReplacement} = require('./replacement_actions');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {searchMoveHistory} = require('./move_history_search');
const {restorePosition, playerObservation, advancePosition, pendingSides} = require('./position_simulator');
const {replacementFixture} = require('./fixtures/replacement_history');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {comparable} = require('./fixtures/position_simulator');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const WORLD = {seed: [333, 1, 2, 3], chance_seed: [333, 4, 5, 6]};
const SEARCH = {seed: [333, 1, 2, 3], worlds: 1, candidates: 6, responses: 2,
    chance_samples: 2, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [333, 7, 8, 9]};

async function runReplacementExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'post_knockout_replacement_history', version: 1, status: 'running', passed: false,
        settings: {world: WORLD, search: SEARCH, workers: 1, experiment_budget_seconds: 30},
        published_sets_used: false, training_eligible: false, strength_claim: false, runtime_integration: false,
        replacement_ranking_verified: false, cases: [], invariant_hidden_world_pairs: 0, failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s replacement experiment budget exceeded');
    };
    try {
        const specs = [
            {kind: 'single'}, {kind: 'double_one_reserve'}, {kind: 'double_two_reserves'}, {kind: 'both'},
            {kind: 'both', side: 'p2'}, {kind: 'single', side: 'p2'},
            {kind: 'terminal'}, {kind: 'terminal', side: 'p2'},
            {kind: 'single', finish: true}, {kind: 'single', finish: true, side: 'p2'},
            {kind: 'both', finish: true}, {kind: 'both', finish: true, side: 'p2'},
        ];
        for (const spec of specs) {
            await gate.waitForLaunch(); checkBudget();
            const {battle, history} = replacementFixture(spec);
            try {
                assertLegalTeams(battle);
                const before = structuredClone(history), world = constructMoveHistoryWorld(history, WORLD);
                assert(world.supported, JSON.stringify({...world, position: null}));
                const copy = restorePosition(world.position, SAMPLE);
                try {
                    assertLegalTeams(copy);
                    assert.deepEqual(playerObservation(copy, spec.side || 'p1'), history.observations.at(-1));
                    if (copy.requestState === 'switch') assert(isPostKnockoutReplacement(copy));
                } finally { copy.destroy(); }
                const result = await searchMoveHistory(history, SEARCH, gate);
                const view = history.observations.at(-1);
                if (view.ended) {
                    assert.equal(result.terminal, true); assert.equal(result.selected_command, null);
                } else if (view.request.wait) {
                    assert.equal(result.waiting, true); assert.equal(result.selected_command, null);
                } else if (view.phase === 'switch') {
                    assert(result.supported); assert.equal(result.ranked, false);
                    assert(replacementActions(view).turns.some(turn => turn.command === result.selected_command));
                    const restored = restorePosition(world.position, SAMPLE);
                    let choices;
                    try {
                        choices = Object.fromEntries(pendingSides(restored).map(side => [side,
                            side === view.side ? result.selected_command : replacementActions(playerObservation(restored, side)).turns[0].command]));
                    } finally { restored.destroy(); }
                    const next = advancePosition(world.position, choices, SAMPLE);
                    assert.equal(next.phase, 'move'); assert.equal(next.turn, view.turn + 1);
                } else {
                    assert.equal(view.phase, 'move');
                    assert(result.supported || result.fallback_required);
                }
                assert.deepEqual(history, before); assert(result.nodes <= SEARCH.max_nodes);
                report.cases.push({kind: spec.kind, side: spec.side || 'p1', through_replacement: !!spec.finish,
                    phase: view.phase, passed: true, replay_nodes: world.nodes, decision: result});
            } finally { battle.destroy(); }
        }
        const make = variant => replacementFixture({changeTeams: teams => {
            teams[0][0].ability = 'Lightning Rod';
            teams[1][0].ability = variant % 2 ? 'Lightning Rod' : 'Static';
            teams[1][0].item = variant % 2 ? 'Choice Band' : 'Leftovers';
            teams[1][0].evs = variant % 2 ? {atk: 252, spe: 4} : {hp: 1};
            Object.assign(teams[1][2], {species: variant % 2 ? 'Slowbro' : 'Espeon',
                ability: variant % 2 ? 'Own Tempo' : 'Synchronize',
                moves: variant % 2 ? ['watergun', 'protect'] : ['psychic', 'protect'], teraType: 'Psychic'});
        }});
        const base = make(0);
        try {
            assertLegalTeams(base.battle);
            const reference = constructMoveHistoryWorld(base.history, WORLD); assert(reference.supported);
            const decision = await searchMoveHistory(base.history, SEARCH, gate); assert(decision.supported);
            for (let i = 1; i <= 4; ++i) {
                await gate.waitForLaunch(); checkBudget();
                const other = make(i);
                try {
                    assertLegalTeams(other.battle); other.battle.prng = new PRNG(`333,${40 + i},4,5`);
                    assert.deepEqual(other.history, base.history);
                    const world = constructMoveHistoryWorld(other.history, WORLD); assert(world.supported);
                    assert.deepEqual(comparable(world.position), comparable(reference.position));
                    const selected = await searchMoveHistory(other.history, SEARCH, gate);
                    assert.equal(selected.selected_command, decision.selected_command);
                    assert.equal(selected.nodes, decision.nodes); assert.equal(selected.ranked, false);
                    ++report.invariant_hidden_world_pairs;
                } finally { other.battle.destroy(); }
            }
        } finally { base.battle.destroy(); }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.limitations = ['Replacement choices are legal incumbents without strategic value ranking.',
            'No midturn pivot, replacement cascade or Tera reconstruction.',
            'Reconstruction remains bounded and restricted to supported executed moves.',
            'No live deployment or strength claim.'];
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
        throw new Error('usage: node tools/replacement_history_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runReplacementExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/ordinary_moves.js', 'tools/opening_actions.js',
            'tools/opening_proposals.js', 'tools/opening_search.js', 'tools/move_history_world.js',
            'tools/move_history_search.js', 'tools/replacement_actions.js', 'tools/replacement_history.test.js',
            'tools/fixtures/replacement_history.js', 'tools/fixtures/tactical_reliability.js', 'tools/replacement_history_experiment.js']);
}).catch(error => { console.error(`[replacement-history] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runReplacementExperiment};
