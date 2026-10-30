'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, advancePosition, playerObservation} = require('./position_simulator');
const {damageVariables, withInvestment, transitionLoss, fitJointTransition} = require('./planner_joint_constraints');
const {HybridPlanner} = require('./hybrid_planner');

const SEED = [351, 1, 2, 3];

function control(battle) {
    const before = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
    const choices = {p1: 'move 1 1, move 2', p2: 'move 3, move 3'};
    const expected = advancePosition(position, choices, {mode: 'sample', seed: SEED}).observations.p1;
    return {history: {observations: [before, expected], own_choices: [choices.p1]},
        simulation_trace: {position, choices, expected, seed: SEED}};
}

test('joint fit preserves a known complete transition and records a reproducible native witness', async () => {
    const battle = createHybridFixture();
    try {
        const failure = control(battle), original = structuredClone(failure);
        const report = await fitJointTransition(failure, null, {max_nodes: 1, seeds: 1});
        assert(report.validation_gate_passed); assert(report.independently_verified);
        assert.equal(report.native_replays, 1); assert.equal(report.best_loss, 0);
        const result = advancePosition(report.witness.opening, report.witness.choices,
            {mode: 'sample', seed: report.witness.seed});
        assert.deepEqual(result.observations.p1, failure.simulation_trace.expected);
        assert.deepEqual(result.position, report.witness.position);
        assert.deepEqual(failure, original);
    } finally { battle.destroy(); }
});

test('numeric loss cannot accept a reordered log or a corrupted private request', async () => {
    const battle = createHybridFixture();
    try {
        const failure = control(battle), expected = failure.simulation_trace.expected;
        const changed = structuredClone(expected);
        changed.request.side.pokemon[0].condition = '999/999';
        assert(transitionLoss(changed, expected, 0) > 0);
        const reordered = structuredClone(expected); reordered.log.reverse();
        assert(transitionLoss(reordered, expected, 0) > 0);
        failure.simulation_trace.expected = changed; failure.history.observations[1] = changed;
        const report = await fitJointTransition(failure, null, {max_nodes: 8, seeds: 1});
        assert.equal(report.validation_gate_passed, false); assert.equal(report.witness, null);
    } finally { battle.destroy(); }
});

test('damage variables include incoming offense and outgoing defenses without changing observed friendly stats', () => {
    const before = {side: 'p1', log: []};
    const after = {log: ['|move|p1a: Latios|Luster Purge|p2a: Chesnaught',
        '|-damage|p2a: Chesnaught|5/100', '|move|p2b: Whiscash|Icy Wind|p1a: Latios',
        '|-damage|p1a: Latios|206/256', '|-damage|p1b: Gyarados|257/283']};
    assert.deepEqual(damageVariables(before, after), [{key: 'p2: Chesnaught', stat: 'hp'},
        {key: 'p2: Chesnaught', stat: 'spd'}, {key: 'p2: Whiscash', stat: 'spa'},
        {key: 'p2: Whiscash', stat: 'spe'}]);
    assert.deepEqual(damageVariables(before, {log: ['|move|p1a: Chansey|Seismic Toss|p2a: Target',
        '|-damage|p2a: Target|50/100']}), []);
});

test('investment updates preserve unrelated hypotheses and reject the total effort budget overflow', () => {
    const sets = [{key: 'p2: Target', set: {ivs: {hp: 31, spa: 31}, evs: {hp: 252, spa: 0, spe: 252}}}];
    const original = structuredClone(sets), variable = {key: 'p2: Target', stat: 'spa'};
    assert.equal(withInvestment(sets, variable, 94), null);
    const proposal = withInvestment(sets, variable, 32);
    assert.equal(proposal[0].set.evs.spa, 4); assert.deepEqual(sets, original);
    assert.equal(proposal[0].set.evs.spe, 252);
});

test('joint fitting corrects a wrong hidden defensive hypothesis rather than merely replaying a supplied match', async () => {
    const actual = createHybridFixture({changeTeams: teams => {
        teams[1][0].ivs = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};
        teams[1][0].evs = {hp: 252, spd: 252, spe: 1};
    }});
    const wrong = createHybridFixture({changeTeams: teams => {
        teams[1][0].ivs = {hp: 0, atk: 31, def: 31, spa: 31, spd: 0, spe: 31};
        teams[1][0].evs = {spe: 1};
    }});
    try {
        const failure = control(actual);
        assert.deepEqual(playerObservation(actual, 'p1'), playerObservation(wrong, 'p1'));
        failure.simulation_trace.position = capturePosition(wrong, 'fixture');
        const incorrect = advancePosition(failure.simulation_trace.position, failure.simulation_trace.choices,
            {mode: 'sample', seed: SEED});
        assert.notDeepEqual(incorrect.observations.p1, failure.simulation_trace.expected);
        const original = structuredClone(failure);
        const report = await fitJointTransition(failure, null, {max_nodes: 256, seeds: 2});
        assert(report.validation_gate_passed); assert(report.proposals > 1);
        assert.notEqual(report.witness.opening.state_json, failure.simulation_trace.position.state_json);
        assert.deepEqual(failure, original);
    } finally { actual.destroy(); wrong.destroy(); }
});

test('joint reconstruction retains explicit proposal and time limits', async () => {
    await assert.rejects(fitJointTransition({}, null, {max_nodes: 8193}), /bounded/);
    await assert.rejects(fitJointTransition({}, null, {max_elapsed_ms: 60001}), /bounded/);
});

test('joint fitting refuses observations or accepted choices detached from the captured history', async () => {
    const battle = createHybridFixture();
    try {
        const failure = control(battle);
        failure.history.own_choices[0] = 'move 2, move 2';
        await assert.rejects(fitJointTransition(failure, null), /accepted choice/);
        const other = control(battle);
        other.history.observations[0].request.side.pokemon[0].condition = '999/999';
        await assert.rejects(fitJointTransition(other, null), /initial observation/);
    } finally { battle.destroy(); }
});

test('opt in planner recovery adopts only complete verified worlds and respects the requested particle cap', async () => {
    const battle = createHybridFixture();
    try {
        const failure = control(battle), view = failure.simulation_trace.expected;
        const planner = new HybridPlanner({seed: SEED, reconstruction: {max_nodes: 1, seeds: 1},
            search: {max_nodes: 1, max_elapsed_ms: 1}}, () => 'move 2, move 2');
        planner.beliefs.update = () => {
            planner.beliefs.history = failure.history; planner.beliefs.firstFailure = failure;
            return {supported: false, nodes: 0, elapsed_seconds: 0, particles: 0, rejections: {}};
        };
        const decision = await planner.decide(view, failure.history.own_choices[0]);
        assert(decision.belief.joint_reconstructed);
        assert.equal(planner.beliefs.worlds.length, 1);
        assert.equal(planner.stats.reconstructed_particles, 1);
        assert.equal(JSON.parse(planner.beliefs.worlds[0].state_json).sides.length, 2);
        assert.equal(planner.stats.reconstruction_attempts, 1);
        await planner.decide(view, failure.history.own_choices[0]);
        assert.equal(planner.stats.reconstruction_attempts, 1);
        const ordinary = new HybridPlanner({seed: SEED}, () => 'move 2, move 2');
        assert.equal(ordinary.reconstruction, null);
    } finally { battle.destroy(); }
});

test('opt in recovery refuses a forged private request and leaves the belief pool empty', async () => {
    const battle = createHybridFixture();
    try {
        const failure = control(battle), view = failure.simulation_trace.expected;
        view.request.side.pokemon[0].condition = '999/999';
        const planner = new HybridPlanner({seed: SEED, reconstruction: {max_nodes: 2, seeds: 1}}, () => 'move 2, move 2');
        planner.beliefs.update = () => {
            planner.beliefs.history = failure.history; planner.beliefs.firstFailure = failure;
            return {supported: false, nodes: 0, elapsed_seconds: 0, particles: 0, rejections: {}};
        };
        const decision = await planner.decide(view, failure.history.own_choices[0]);
        assert.equal(decision.source, 'fallback'); assert.equal(planner.beliefs.worlds.length, 0);
        assert.equal(planner.stats.reconstructed_particles, 0);
    } finally { battle.destroy(); }
});

test('multi turn joint fitting independently verifies every earlier boundary and rejects a corrupted prefix', async () => {
    const battle = createHybridFixture();
    try {
        const failure = control(battle), trace = failure.simulation_trace;
        const first = advancePosition(trace.position, trace.choices, {mode: 'sample', seed: SEED});
        const seed = [367, 4, 5, 6];
        const last = advancePosition(first.position, trace.choices, {mode: 'sample', seed});
        failure.opening_position = trace.position;
        failure.replay_steps = [{choices: trace.choices, seed: SEED}];
        failure.history.observations.push(last.observations.p1);
        failure.history.own_choices.push(trace.choices.p1);
        failure.simulation_trace = {...trace, position: first.position, seed, expected: last.observations.p1};
        const original = structuredClone(failure);
        const report = await fitJointTransition(failure, null, {max_nodes: 1, seeds: 1});
        assert(report.validation_gate_passed); assert.equal(report.matched_boundaries, 2);
        assert.equal(report.witness.steps.length, 2); assert.deepEqual(failure, original);
        failure.history.observations[1].request.side.pokemon[0].condition = '999/999';
        const rejected = await fitJointTransition(failure, null, {max_nodes: 16, seeds: 1});
        assert.equal(rejected.validation_gate_passed, false); assert.equal(rejected.witness, null);
        const lastCorrupt = structuredClone(original);
        lastCorrupt.history.observations[2].request.side.pokemon[0].condition = '999/999';
        lastCorrupt.simulation_trace.expected = lastCorrupt.history.observations[2];
        const cached = await fitJointTransition(lastCorrupt, null, {max_nodes: 8, seeds: 1, persistent: true});
        assert.equal(cached.validation_gate_passed, false);
        assert(cached.prefix_cache_hits > 0); assert(cached.opening_cache_hits > 0);
        assert(cached.native_replays <= cached.proposals + 1);
    } finally { battle.destroy(); }
});

test('a newly revealed reserve replaces only an unobserved root proposal before full native replay', async () => {
    const actual = createHybridFixture();
    const hypothesis = createHybridFixture({changeTeams: teams => {
        Object.assign(teams[1][2], {species: 'Bulbasaur', name: 'Unrevealed 1', ability: 'Overgrow',
            moves: ['tackle', 'protect', 'growl', 'vinewhip'], gender: 'M'});
    }});
    try {
        const before = playerObservation(actual, 'p1');
        assert.deepEqual(before, playerObservation(hypothesis, 'p1'));
        const position = capturePosition(hypothesis, 'fixture');
        const choices = {p1: 'move 1 1, move 2', p2: 'switch 3, move 3'};
        const expected = advancePosition(capturePosition(actual, 'fixture'), choices, {mode: 'sample', seed: SEED}).observations.p1;
        const failure = {history: {observations: [before, expected], own_choices: [choices.p1]},
            simulation_trace: {position, choices, expected, seed: SEED}};
        const original = structuredClone(failure);
        const report = await fitJointTransition(failure, null, {max_nodes: 512, seeds: 4});
        assert(report.validation_gate_passed, JSON.stringify({loss: report.best_loss, reason: report.reason}));
        assert.equal(report.witness.position.origin, 'belief-sample'); assert.deepEqual(failure, original);
        const decoded = JSON.parse(report.witness.opening.state_json);
        assert(decoded.sides[1].pokemon.some(mon => mon.set.species === 'Clefable'));
        assert(!decoded.sides[1].pokemon.some(mon => mon.set.species === 'Bulbasaur'));
    } finally { actual.destroy(); hypothesis.destroy(); }
});

test('revealed friendly illusion makes opponent reconstruction abstain instead of aborting the live battle', async () => {
    const battle = createHybridFixture({changeTeams: teams => {
        Object.assign(teams[0][0], {species: 'Zoroark-Hisui', ability: 'Illusion', gender: 'M',
            moves: ['shadowball', 'protect', 'nastyplot', 'flamethrower']});
        teams[1][0].moves[0] = 'crunch';
        Object.assign(teams[1][1], {species: 'Regieleki', ability: 'Transistor', gender: 'N',
            moves: ['thunderbolt', 'protect', 'explosion', 'charge']});
    }});
    try {
        const before = playerObservation(battle, 'p1'), opening = capturePosition(battle, 'fixture');
        const choices = {p1: 'move 3, move 2', p2: 'move 1 1, move 3'};
        const first = advancePosition(opening, choices, {mode: 'sample', seed: SEED});
        assert(first.observations.p1.log.some(line => line.includes('|Illusion')));
        assert.equal(first.observations.p1.request.wait, true);
        assert.deepEqual(first.pending_sides, ['p2']);
        const replacement = {p2: 'pass, switch 3'};
        const seed = [376, 4, 5, 6], last = advancePosition(first.position, replacement, {mode: 'sample', seed});
        const failure = {opening_position: opening, replay_steps: [{seed: SEED, choices}],
            history: {observations: [before, first.observations.p1, last.observations.p1], own_choices: [choices.p1, null]},
            simulation_trace: {position: first.position, choices: replacement, seed, expected: last.observations.p1}};
        const original = structuredClone(failure);
        const report = await fitJointTransition(failure, null, {max_nodes: 4, seeds: 1});
        assert.equal(report.validation_gate_passed, false); assert(report.unsupported_opponent_requests > 0);
        assert.equal(report.witness, null); assert.deepEqual(failure, original);
    } finally { battle.destroy(); }
});
