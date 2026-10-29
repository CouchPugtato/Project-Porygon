'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, advancePosition} = require('./position_simulator');
const {firstDamageFeasibility} = require('./planner_damage_feasibility');
const {defensiveGrid} = require('./planner_stat_constraints');
const {validateTargetEpisode} = require('./target_context');
const {validateBenchmark} = require('./hybrid_benchmark');

const SEED = [349, 1, 2, 3], SAMPLE = {mode: 'sample', seed: SEED};

function damageTrace(battle) {
    const position = capturePosition(battle, 'fixture'), choices = {p1: 'move 1 1, move 2', p2: 'move 3, move 3'};
    const expected = advancePosition(position, choices, SAMPLE).observations.p1;
    return {position, choices, seed: SEED, expected};
}

test('native damage factor enumeration contains its known outcome and preserves the parent', () => {
    const battle = createHybridFixture();
    try {
        const trace = damageTrace(battle), parent = trace.position.state_json;
        const result = firstDamageFeasibility(trace);
        assert(result.supported); assert.equal(result.outcomes.length, 16);
        assert(result.context_matched); assert(result.feasible_in_captured_context);
        assert.equal(result.full_transition_verified, false);
        assert.equal(trace.position.state_json, parent);
    } finally { battle.destroy(); }
});

test('an incompatible damage display is rejected only within the declared captured context', () => {
    const battle = createHybridFixture();
    try {
        const trace = damageTrace(battle);
        const index = trace.expected.log.findIndex(line => line.startsWith('|-damage|'));
        const parts = trace.expected.log[index].split('|'); parts[3] = '100/100';
        trace.expected.log[index] = parts.join('|');
        const result = firstDamageFeasibility(trace);
        assert(result.supported); assert(result.context_matched);
        assert.equal(result.feasible_in_captured_context, false);
        assert.equal(result.scope, 'first_direct_hit_with_other_chance_draws_fixed');
    } finally { battle.destroy(); }
});

test('a hit without a randomized damage calculation does not receive a fabricated range', () => {
    const battle = createHybridFixture({changeTeams: teams => { teams[0][0].moves[0] = 'seismictoss'; }});
    try {
        const result = firstDamageFeasibility(damageTrace(battle));
        assert.equal(result.supported, false);
        assert.equal(result.reason, 'first_hit_not_a_single_randomized_damage_calculation');
    } finally { battle.destroy(); }
});

test('earlier hazard damage cannot masquerade as the first direct hit on the same target', () => {
    const battle = createHybridFixture({changeTeams: teams => {
        Object.assign(teams[0][0], {species: 'Mew', ability: 'Synchronize', gender: '',
            moves: ['stealthrock', 'psychic', 'protect', 'helpinghand']});
        Object.assign(teams[1][2], {species: 'Espeon', ability: 'Synchronize', moves: ['psychic', 'protect']});
    }});
    try {
        assert(battle.choose('p1', 'move 1, move 2')); assert(battle.choose('p2', 'move 3, move 3'));
        const position = capturePosition(battle, 'fixture'), choices = {p1: 'move 2 1, move 2', p2: 'switch 3, move 3'};
        const expected = advancePosition(position, choices, SAMPLE).observations.p1;
        const result = firstDamageFeasibility({position, choices, expected, seed: SEED});
        assert.equal(result.supported, false);
        assert.equal(result.reason, 'first_logged_damage_not_the_first_calculated_hit');
    } finally { battle.destroy(); }
});

test('bounded stat proposals alter only the selected sampled build and respect the investment budget', () => {
    const sets = [{key: 'p2: Target', set: {ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
        evs: {hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0}, moves: ['protect'], nature: 'Serious'}},
        {key: 'p2: Other', set: {moves: ['protect']}}];
    const before = structuredClone(sets), proposals = defensiveGrid(sets, 'p2: Target', 'spd');
    assert.equal(proposals.length, 16); assert.deepEqual(sets, before);
    for (const proposal of proposals) {
        assert.deepEqual(proposal[1], sets[1]);
        assert(Object.values(proposal[0].set.evs).reduce((a, b) => a + b, 0) <= 508);
        assert.deepEqual(proposal[0].set.moves, sets[0].set.moves);
    }
});

test('hybrid evaluation preserves and validates the strongest native targeting episode', () => {
    const config = {target_slot_local: true, target_decoder: 'argmax'};
    const native = {policy_tag: 'target-context-slot-local-argmax-v1', training_eligible: false,
        target_decoder: 'argmax', target_slot_local: true, target_decisions: 10, learned_target_decisions: 4};
    const wrapped = {policy_tag: 'hybrid-search-eval-v1', training_eligible: false, hybrid_baseline_episode: native};
    assert.equal(validateTargetEpisode(native, config), native);
    assert.equal(validateTargetEpisode(wrapped, config), native);
    assert.throws(() => validateTargetEpisode({...wrapped, training_eligible: true}, config), /evaluation-only/);
    assert.throws(() => validateTargetEpisode({...wrapped, hybrid_baseline_episode: {...native, policy_tag: 'parent'}}, config), /evaluation-only/);
    assert.throws(() => validateTargetEpisode({...native, learned_target_decisions: 11}, config), /evaluation-only/);
});

test('confirmed fallback config retains its student pairing and evaluation only decoder restrictions', () => {
    const config = structuredClone(require('../config/experimental/hybrid_benchmark.json'));
    validateBenchmark(config);
    assert(config.candidate_checkpoint.includes('run_0275'));
    assert(config.target_sidecar.includes('run_0294'));
    assert.throws(() => validateBenchmark({...config, target_slot_local: false}), /confirmed student/);
    assert.throws(() => validateBenchmark({...config, candidate_checkpoint: undefined}), /confirmed student/);
});
