'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, advancePosition} = require('./position_simulator');
const {rollReplay, diagnoseRolls, mechanicalAlternatives, investmentRanges} = require('./planner_mechanics');
const {fitJointTransition} = require('./planner_joint_constraints');
const {observeBeliefs} = require('./belief_state');
const {playerObservation} = require('./position_simulator');

const SEED = [378, 1, 2, 3];
const choices = {p1: 'move 1 1, move 2', p2: 'move 3, move 3'};
const budget = () => ({available: () => true, consume() {}});
const loss = (actual, expected) => Number(JSON.stringify(actual) !== JSON.stringify(expected));

test('native damage roll enumeration includes a known legal hit without mutating its parent', async () => {
    const battle = createHybridFixture();
    try {
        const position = capturePosition(battle, 'fixture'), original = position.state_json;
        const expected = advancePosition(position, choices, {mode: 'sample', seed: SEED}).observations.p1;
        const result = await diagnoseRolls(position, choices, SEED, expected, budget(), loss);
        assert.equal(result.status, 'conditional_legal_roll_match');
        assert.equal(result.exhaustive, true); assert.equal(result.max_damage_events, 1);
        assert.equal(result.witness_eligible, false); assert.equal(position.state_json, original);
    } finally { battle.destroy(); }
});

test('an impossible private request is excluded only within the explicitly fixed event scope', async () => {
    const battle = createHybridFixture();
    try {
        const position = capturePosition(battle, 'fixture');
        const expected = advancePosition(position, choices, {mode: 'sample', seed: SEED}).observations.p1;
        expected.request.side.pokemon[0].condition = '999/999';
        const result = await diagnoseRolls(position, choices, SEED, expected, budget(), loss);
        assert.equal(result.status, 'no_roll_match_under_fixed_events');
        assert.equal(result.exhaustive, true); assert.equal(result.conditional, true);
        const limited = await diagnoseRolls(position, choices, SEED, expected,
            {available: () => false, consume() { throw new Error('budget exceeded'); }}, loss);
        assert.equal(limited.status, 'inconclusive'); assert.equal(limited.exhaustive, false);
    } finally { battle.destroy(); }
});

test('multiple damage events never turn a nonexhaustive search into an impossibility claim', async () => {
    const battle = createHybridFixture();
    try {
        const position = capturePosition(battle, 'fixture');
        const attacks = {p1: 'move 1 1, move 1 1', p2: 'move 3, move 3'};
        const expected = advancePosition(position, attacks, {mode: 'sample', seed: SEED}).observations.p1;
        expected.request.side.pokemon[0].condition = '999/999';
        const result = await diagnoseRolls(position, attacks, SEED, expected, budget(), loss);
        assert(result.max_damage_events > 1); assert.equal(result.exhaustive, false);
        assert.equal(result.status, 'inconclusive'); assert(result.trials <= 64);
    } finally { battle.destroy(); }
});

test('mechanical alternatives preserve revealed items abilities moves and original proposals', () => {
    const sets = [{key: 'p2: Snorlax', set: {species: 'Snorlax', ability: 'Immunity', item: '',
        nature: 'Serious', moves: ['tackle']}}];
    const facts = {opponent: {pokemon: [{key: sets[0].key, ability: 'immunity',
        item: {current: '', last_revealed: 'sitrusberry'}}]}};
    const variables = [{key: sets[0].key, stat: 'spe'}], original = structuredClone(sets);
    const proposals = mechanicalAlternatives(sets, facts, variables, true);
    assert(proposals.length); assert(proposals.every(p => p[0].set.ability === 'Immunity' && p[0].set.item === ''));
    assert(proposals.every(p => p[0].set.moves[0] === 'tackle')); assert.deepEqual(sets, original);
    facts.opponent.pokemon[0].ability = null; facts.opponent.pokemon[0].item = null;
    const unknown = mechanicalAlternatives(sets, facts, variables, true);
    assert(unknown.some(p => p[0].set.item === 'Choice Scarf'));
    assert(unknown.some(p => p[0].set.ability === 'Thick Fat'));
});

test('mechanics recovery still requires full natural replay and retains only sampled witness ranges', async () => {
    const battle = createHybridFixture();
    try {
        const before = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const next = advancePosition(position, choices, {mode: 'sample', seed: SEED});
        const failure = {history: {observations: [before, next.observations.p1], own_choices: [choices.p1]},
            simulation_trace: {position, choices, expected: next.observations.p1, seed: SEED}};
        const original = structuredClone(failure);
        const report = await fitJointTransition(failure, null, {mechanics: true, max_nodes: 1, seeds: 1});
        assert(report.independently_verified); assert.equal(report.investment_constraints.status, 'sampled_exact_witnesses');
        assert.equal(report.investment_constraints.exhaustive, false);
        assert(report.investment_constraints.ranges.some(range => range.sampled_values.length));
        assert.deepEqual(failure, original);
        assert.equal(observeBeliefs(before).opponent.pokemon.length, 2);
        assert.deepEqual(investmentRanges([], [{key: 'p2: Unknown', stat: 'hp'}])[0].sampled_values, []);
    } finally { battle.destroy(); }
});

test('recoil uses the native chosen damage roll and remains a diagnostic not an accepted witness', () => {
    const battle = createHybridFixture({changeTeams: teams => { teams[0][0].moves[0] = 'wildcharge'; }});
    try {
        const position = capturePosition(battle, 'fixture');
        const low = rollReplay(position, choices, SEED, 'p1', [15]);
        const high = rollReplay(position, choices, SEED, 'p1', [0]);
        assert(low.observation.log.some(line => line.includes('[from] Recoil')));
        assert.notDeepEqual(low.observation, high.observation);
    } finally { battle.destroy(); }
});

test('mechanics recovery explains hidden offensive items without copying an actual hidden build', async () => {
    const actual = createHybridFixture({changeTeams: teams => { teams[1][0].item = 'Choice Band'; }});
    const hypothesis = createHybridFixture();
    try {
        const before = playerObservation(actual, 'p1'), position = capturePosition(hypothesis, 'fixture');
        assert.deepEqual(before, playerObservation(hypothesis, 'p1'));
        const attacks = {p1: 'move 3 -2, move 2', p2: 'move 1 1, move 3'};
        const next = advancePosition(capturePosition(actual, 'fixture'), attacks, {mode: 'sample', seed: SEED});
        const failure = {history: {observations: [before, next.observations.p1], own_choices: [attacks.p1]},
            simulation_trace: {position, choices: attacks, expected: next.observations.p1, seed: SEED}};
        const original = structuredClone(failure);
        const result = await fitJointTransition(failure, null, {mechanics: true, max_nodes: 192, seeds: 1});
        assert(result.validation_gate_passed, JSON.stringify({loss: result.best_loss, rolls: result.roll_diagnostics}));
        assert(result.mechanical_proposals > 0); assert.deepEqual(failure, original);
        assert(result.roll_diagnostics.length);
    } finally { actual.destroy(); hypothesis.destroy(); }
});

test('native speed recovery accounts for choice scarf without imposing a raw speed ordering', async () => {
    const make = item => createHybridFixture({changeTeams: teams => {
        Object.assign(teams[1][0], {species: 'Gallade', ability: 'Steadfast', item,
            moves: ['psychocut', 'protect', 'swordsdance', 'rest']});
    }});
    const actual = make('Choice Scarf'), hypothesis = make('');
    try {
        const before = playerObservation(actual, 'p1'), position = capturePosition(hypothesis, 'fixture');
        assert.deepEqual(before, playerObservation(hypothesis, 'p1'));
        const attacks = {p1: 'move 1 1, move 2', p2: 'move 1 1, move 3'};
        const next = advancePosition(capturePosition(actual, 'fixture'), attacks, {mode: 'sample', seed: SEED});
        const failure = {history: {observations: [before, next.observations.p1], own_choices: [attacks.p1]},
            simulation_trace: {position, choices: attacks, expected: next.observations.p1, seed: SEED}};
        const result = await fitJointTransition(failure, null, {mechanics: true, max_nodes: 192, seeds: 1});
        assert(result.validation_gate_passed, JSON.stringify({loss: result.best_loss, mismatch: result.best_mismatch}));
        assert(result.mechanical_proposals > 0);
    } finally { actual.destroy(); hypothesis.destroy(); }
});

test('multiple observed hits share one build and a forged earlier hit cannot be discarded', async () => {
    const battle = createHybridFixture();
    try {
        const before = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const first = advancePosition(position, choices, {mode: 'sample', seed: SEED});
        const second = advancePosition(first.position, choices, {mode: 'sample', seed: SEED});
        const failure = {opening_position: position, replay_steps: [{choices, seed: SEED}],
            history: {observations: [before, first.observations.p1, second.observations.p1], own_choices: [choices.p1, choices.p1]},
            simulation_trace: {position: first.position, choices, seed: SEED, expected: second.observations.p1}};
        const result = await fitJointTransition(failure, null, {mechanics: true, max_nodes: 1, seeds: 1});
        assert(result.validation_gate_passed); assert.equal(result.witness.steps.length, 2);
        failure.history.observations[1].request.side.pokemon[0].condition = '999/999';
        const rejected = await fitJointTransition(failure, null, {mechanics: true, max_nodes: 64, seeds: 1});
        assert.equal(rejected.validation_gate_passed, false); assert.equal(rejected.witness, null);
    } finally { battle.destroy(); }
});
