'use strict';

const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {playerObservation} = require('./position_simulator');
const {createFixture} = require('./fixtures/position_simulator');
const {observeBeliefs, sampleRevealedSets} = require('./belief_state');
const {seedTuple} = require('./counterfactual_rollout');
const {loadedSetGenerators, publishExperiment} = require('./position_simulator_experiment');

const SETTINGS = {seed: [323, 1, 2, 3], samples: 8};

async function runBeliefExperiment(gate) {
    await gate.waitForLaunch();
    const started = performance.now();
    const report = {diagnostic: 'observation_belief_fixture_experiment', version: 1,
        status: 'running', passed: false, direction_step: 'hybrid_engine_plan.txt: initial stage 2',
        scope: 'visible_facts_and_revealed_set_completions', published_sets_used: false,
        training_eligible: false, strength_claim: false, world_ready: false, stage_2_complete: false,
        settings: {hidden_world_pairs: 32, particles_per_observation: SETTINGS.samples,
            sample_seed: SETTINGS.seed, workers: 1, active_experiment_budget_seconds: 30},
        checks: {invariant_hidden_world_pairs: 0, evidence_updates: 0,
            abstentions: 0, sampled_particles: 0}, failure: null};
    const base = createFixture();
    try {
        const view = playerObservation(base, 'p1'), beliefs = observeBeliefs(view);
        const reference = sampleRevealedSets(beliefs, SETTINGS);
        assert.equal(reference.particles.length, SETTINGS.samples);
        for (let i = 0; i < 32; ++i) {
            if (i % 8 === 0) await gate.waitForLaunch();
            if (performance.now() - started > 30000) throw new Error('30s belief experiment budget exceeded');
            const changed = createFixture({seed: seedTuple(32301, i, 0).split(',').map(Number),
                changeTeams: selected => {
                    selected[1][0].ability = i % 2 ? 'Thick Fat' : 'Gluttony';
                    selected[1][0].item = i % 2 ? 'Choice Band' : 'Leftovers';
                    selected[1][0].moves = ['bellydrum', 'rest', 'bodyslam', 'crunch'];
                    selected[1][0].evs = {hp: i * 4, atk: 252, spe: 4};
                    selected[1][2].species = i % 2 ? 'Alakazam' : 'Slowbro';
                    selected[1][2].ability = i % 2 ? 'Synchronize' : 'Own Tempo';
                }});
            try {
                const changedView = playerObservation(changed, 'p1');
                assert.deepEqual(changedView, view);
                assert.deepEqual(observeBeliefs(changedView), beliefs);
                const sampled = sampleRevealedSets(observeBeliefs(changedView), SETTINGS);
                assert.deepEqual(sampled, reference);
                ++report.checks.invariant_hidden_world_pairs;
                report.checks.sampled_particles += sampled.particles.length;
            } finally { changed.destroy(); }
        }
        const updates = [
            ['|move|p2a: Snorlax|Belly Drum|p2a: Snorlax', p => p.set.moves.includes('bellydrum')],
            ['|-ability|p2a: Snorlax|Thick Fat', p => p.set.ability === 'Thick Fat'],
            ['|-item|p2a: Snorlax|Leftovers', p => p.set.item === 'Leftovers'],
            ['|-terastallize|p2a: Snorlax|Ghost', p => p.set.teraType === 'Ghost'],
        ];
        const history = [...view.log];
        for (const [line, consistent] of updates) {
            history.push(line);
            const next = observeBeliefs({...view, log: [...history]});
            const sampled = sampleRevealedSets(next, SETTINGS);
            assert.equal(sampled.particles.length, SETTINGS.samples);
            for (const particle of sampled.particles) {
                const p = particle.sets.find(s => s.key === 'p2: Snorlax');
                for (const [, earlier] of updates.slice(0, report.checks.evidence_updates + 1)) assert(earlier(p));
                assert(consistent(p));
            }
            ++report.checks.evidence_updates;
            report.checks.sampled_particles += sampled.particles.length;
        }
        const damaged = observeBeliefs({...view, log: [...view.log, '|-damage|p2a: Snorlax|75/100 par']});
        const hp = damaged.opponent.pokemon.find(p => p.species === 'snorlax').health;
        assert.deepEqual([hp.lower, hp.upper, hp.status], [.74, .75, 'par']);
        ++report.checks.evidence_updates;
        for (const line of [
            '|replace|p2a: Snorlax|Zoroark, M',
            '|move|p2a: Snorlax|Thunderbolt|p1a: Pikachu|[from] move: Metronome',
            '|-ability|p2a: Snorlax|Levitate|[from] move: Skill Swap',
            '|move|p2a: Snorlax|Not A Real Move|p1a: Pikachu',
            '|-damage|p2a: Snorlax|400/461',
        ]) {
            const sampled = sampleRevealedSets(observeBeliefs({...view, log: [...view.log, line]}), SETTINGS);
            assert.equal(sampled.particles.length, 0); assert(sampled.unsupported.length);
            ++report.checks.abstentions;
        }
        report.loaded_set_generator_modules = loadedSetGenerators();
        assert.deepEqual(report.loaded_set_generator_modules, []);
        report.remaining = ['Unrevealed teammates and complete dynamic worlds are not constructed.',
            'Proposals are not calibrated beliefs; damage-to-stat inference and move-slot/PP recovery remain.',
            'No planner, network update or battle-strength comparison ran.'];
        report.status = 'completed'; report.passed = true;
    } catch (error) {
        report.status = 'failed'; report.failure = error.stack || error.message;
    } finally {
        base.destroy();
        report.elapsed_seconds = (performance.now() - started) / 1000;
        report.observed_process_rss_mib = process.memoryUsage().rss / 1024 ** 2;
    }
    return report;
}

if (require.main === module) Promise.resolve().then(() => {
    const argv = process.argv.slice(2);
    if (argv.length !== 2 || argv[0] !== '--run-name')
        throw new Error('usage: node tools/belief_state_experiment.js --run-name run_NNNN_name');
    return publishExperiment(argv[1], runBeliefExperiment,
        ['tools/belief_state.js', 'tools/belief_state.test.js', 'tools/belief_state_experiment.js']);
}).catch(error => { console.error(`[belief-state] ${error.stack || error.message}`); process.exitCode = 1; });

module.exports = {runBeliefExperiment};
