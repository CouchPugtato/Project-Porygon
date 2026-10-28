'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {constructOpeningWorld} = require('./opening_world');
const {playerObservation, restorePosition, advancePosition} = require('./position_simulator');
const {createFixture, comparable, openingFixtureChoices, sixPokemon, SELF_BUFFS} = require('./fixtures/position_simulator');
const {seedTuple} = require('./counterfactual_rollout');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [324, 1, 2, 3], chance_seed: [324, 4, 5, 6], attempts: 32};
const SAMPLE = {mode: 'sample', seed: [324, 7, 8, 9]};

async function runOpeningExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'opening_world_fixture_experiment', version: 1, status: 'running', passed: false,
        scope: 'observation_consistent_opening_worlds_only', direction_step: 'hybrid_engine_plan.txt: opening world constructor',
        settings: {...SETTINGS, workers: 1, active_experiment_budget_seconds: 30},
        published_sets_used: false, training_eligible: false, search_ready: false,
        strength_claim: false, stage_2_complete: false, cases: [], invariant_hidden_world_pairs: 0,
        expected_abstentions: 0, failure: null};
    const base = createFixture();
    const checkBudget = () => { if (performance.now() - started > 30000) throw new Error('30s opening experiment budget exceeded'); };
    try {
        const cases = [
            {name: 'three_pokemon_side_a', side: 'p1'}, {name: 'three_pokemon_side_b', side: 'p2'},
            {name: 'six_pokemon_side_a', side: 'p1', changeTeams: sixPokemon},
            {name: 'six_pokemon_side_b', side: 'p2', changeTeams: sixPokemon},
            {name: 'non_neutral_level80', side: 'p1', changeTeams: selected => {
                Object.assign(selected[0][0], {level: 80, nature: 'Adamant',
                    evs: {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85}});
            }},
            {name: 'startup_weather', side: 'p1', changeTeams: selected => {
                Object.assign(selected[0][0], {species: 'Ninetales', ability: 'Drought',
                    moves: ['flamethrower', 'protect', 'solarbeam', 'willowisp']});
            }},
        ];
        for (const spec of cases) {
            await gate.waitForLaunch(); checkBudget();
            const live = createFixture({changeTeams: spec.changeTeams});
            try {
                const view = playerObservation(live, spec.side), world = constructOpeningWorld(view, SETTINGS);
                assert.equal(world.supported, true, `${spec.name}: ${JSON.stringify(world.unsupported)}`);
                const copy = restorePosition(world.position, SAMPLE);
                try { assert.deepEqual(playerObservation(copy, spec.side), view); }
                finally { copy.destroy(); }
                const root = comparable(world.position);
                const next = advancePosition(world.position, openingFixtureChoices(world.position, SAMPLE.seed), SAMPLE);
                assert(next.turn > 1 || next.phase === 'switch' || next.ended);
                assert.deepEqual(comparable(world.position), root);
                report.cases.push({name: spec.name, passed: true, attempts: world.attempts,
                    rejected_proposals: world.rejected, opponent_roster: world.opponent_hypotheses.length,
                    next_phase: next.phase, next_turn: next.turn});
            } finally { live.destroy(); }
        }
        const view = playerObservation(base, 'p1'), reference = constructOpeningWorld(view, SETTINGS);
        assert(reference.supported);
        for (let i = 0; i < 16; ++i) {
            if (i % 4 === 0) await gate.waitForLaunch(); checkBudget();
            const hidden = createFixture({seed: seedTuple(32401, i, 0).split(',').map(Number),
                changeTeams: selected => {
                    selected[1][0].ability = i % 2 ? 'Thick Fat' : 'Gluttony';
                    selected[1][0].item = i % 2 ? 'Choice Band' : 'Leftovers';
                    selected[1][0].moves = ['bellydrum', 'rest', 'bodyslam', 'crunch'];
                    selected[1][0].evs = {hp: i * 4, atk: 252, spe: 4};
                    selected[1][2].species = i % 2 ? 'Alakazam' : 'Slowbro';
                    selected[1][2].ability = i % 2 ? 'Synchronize' : 'Own Tempo';
                }});
            try {
                const hiddenView = playerObservation(hidden, 'p1'); assert.deepEqual(hiddenView, view);
                const world = constructOpeningWorld(hiddenView, SETTINGS); assert(world.supported);
                assert.deepEqual(comparable(world.position), comparable(reference.position));
                ++report.invariant_hidden_world_pairs;
            } finally { hidden.destroy(); }
        }
        assert(base.choose('p1', 'move 4, move 2')); assert(base.choose('p2', SELF_BUFFS));
        const damaged = structuredClone(view); damaged.request.side.pokemon[0].condition = '1/211';
        for (const unsupported of [playerObservation(base, 'p1'), damaged,
            {...view, log: [...view.log, '|-weather|RainDance']}]) {
            const world = constructOpeningWorld(unsupported, {...SETTINGS, attempts: 2});
            assert.equal(world.supported, false); assert.equal(world.position, null);
            assert(world.unsupported.length); ++report.expected_abstentions;
        }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.remaining = ['Later turns, PP/volatile history and disguise reconstruction remain unsupported.',
            'Rule-based proposals and one stat-equivalent own build are not a calibrated posterior.',
            'No search policy, deployment or strength experiment ran.'];
        report.status = 'completed'; report.passed = true;
    } catch (error) {
        report.status = 'failed'; report.failure = error.stack || error.message;
    } finally {
        base.destroy(); report.elapsed_seconds = (performance.now() - started) / 1000;
        report.observed_process_rss_mib = process.memoryUsage().rss / 1024 ** 2;
    }
    return report;
}

if (require.main === module) Promise.resolve().then(() => {
    const argv = process.argv.slice(2);
    if (argv.length !== 2 || argv[0] !== '--run-name')
        throw new Error('usage: node tools/opening_world_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runOpeningExperiment,
        ['tools/belief_state.js', 'tools/belief_state.test.js', 'tools/opening_world.js',
            'tools/opening_world.test.js', 'tools/opening_world_experiment.js']);
}).catch(error => { console.error(`[opening-world] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runOpeningExperiment};
