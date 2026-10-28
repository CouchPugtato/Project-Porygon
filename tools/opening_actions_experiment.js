'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {openingActions, checkWorldChoice} = require('./opening_actions');
const {constructOpeningWorld} = require('./opening_world');
const {playerObservation, advancePosition} = require('./position_simulator');
const {createFixture, comparable, openingFixtureChoices, sixPokemon} = require('./fixtures/position_simulator');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [325, 1, 2, 3], chance_seed: [325, 4, 5, 6], attempts: 32};
const SAMPLE = {mode: 'sample', seed: [325, 7, 8, 9]};

async function runActionExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'opening_complete_turn_actions', version: 1, status: 'running', passed: false,
        scope: 'initial_two_active_move_requests', settings: {...SETTINGS, workers: 1, budget_seconds: 30},
        published_sets_used: false, training_eligible: false, strength_claim: false,
        search_policy_implemented: false, cases: [], failure: null};
    const checkBudget = () => {
        if (performance.now() - started > 30000) throw new Error('30s opening action experiment budget exceeded');
    };
    try {
        for (const size of [3, 6]) for (const side of ['p1', 'p2']) {
            await gate.waitForLaunch(); checkBudget();
            const live = createFixture({changeTeams: size === 6 ? sixPokemon : undefined});
            try {
                const view = playerObservation(live, side), world = constructOpeningWorld(view, SETTINGS);
                assert(world.supported, JSON.stringify(world.unsupported));
                const root = comparable(world.position), generated = openingActions(view);
                const accepted = [], rejected = [];
                for (const [i, turn] of generated.turns.entries()) {
                    if (i % 64 === 0) { await gate.waitForLaunch(); checkBudget(); }
                    const checked = checkWorldChoice(world.position, view, turn.command, SAMPLE);
                    if (checked.accepted) accepted.push(turn.command);
                    else rejected.push({command: turn.command, error: checked.error, request_updated: checked.request_updated});
                }
                assert(accepted.length > 0);
                const reference = openingFixtureChoices(world.position, SAMPLE.seed);
                let simulated = 0;
                // Spread samples across the entire deterministic action order,
                // rather than only exercising the first move or Tera setting.
                for (let i = 0; i < Math.min(16, accepted.length); ++i) {
                    checkBudget();
                    const command = accepted[Math.floor(i * accepted.length / Math.min(16, accepted.length))];
                    const next = advancePosition(world.position, {...reference, [side]: command}, SAMPLE);
                    assert(next.turn > 1 || next.phase === 'switch' || next.ended);
                    ++simulated;
                }
                assert.deepEqual(comparable(world.position), root);
                assert.deepEqual(openingActions(view), generated);
                report.cases.push({name: `${size}_pokemon_${side}`, passed: true,
                    proposed: generated.turns.length, accepted_in_world: accepted.length,
                    rejected_in_world: rejected, simulated, parent_unchanged: true});
            } finally { live.destroy(); }
        }
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
        throw new Error('usage: node tools/opening_actions_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runActionExperiment,
        ['tools/belief_state.js', 'tools/opening_world.js', 'tools/opening_actions.js',
            'tools/opening_actions.test.js', 'tools/opening_actions_experiment.js']);
}).catch(error => { console.error(`[opening-actions] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runActionExperiment};
