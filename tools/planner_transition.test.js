'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, advancePosition, playerObservation} = require('./position_simulator');
const {PlannerBeliefs} = require('./planner_beliefs');
const {mismatchEvidence} = require('./planner_evidence');
const {auditTransition} = require('./planner_transition_audit');

const SEED = [344, 1, 2, 3], SAMPLE = {mode: 'sample', seed: SEED};

function controlFailure(battle) {
    const before = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
    const choices = {p1: 'move 4, move 2', p2: 'move 3, move 3'};
    const actual = advancePosition(position, choices, SAMPLE).observations.p1;
    const expected = structuredClone(actual);
    expected.request.active[0].moves[0].pp = 999;
    return {history: {observations: [before, expected], own_choices: [choices.p1]},
        simulation_trace: {stage: 'simulation', position, choices, seed: SEED, expected, actual}};
}

test('captured branches replay independently and never accept an altered private request', async () => {
    const battle = createHybridFixture();
    try {
        const failure = controlFailure(battle), before = structuredClone(failure);
        const result = await auditTransition(failure, null, {samples: 4, max_elapsed_ms: 2000});
        assert(result.exact_branch_replay); assert(result.synthetic_inference_control);
        assert(result.recorded_pair_inferred); assert.equal(result.samples, 4);
        assert.equal(result.exact_matches, 0);
        assert.equal(result.initial_mismatch.kind, 'private_request_or_hidden_stats_mismatch');
        assert.deepEqual(failure, before);
    } finally { battle.destroy(); }
});

test('independent replay refuses a corrupted captured simulator projection', async () => {
    const battle = createHybridFixture();
    try {
        const failure = controlFailure(battle);
        failure.simulation_trace.actual.request.active[0].moves[0].pp = 999;
        await assert.rejects(auditTransition(failure, null, {samples: 1}), /replay independently/);
    } finally { battle.destroy(); }
});

test('spread display targets remain exact mismatches without being labelled action order failures', () => {
    const expected = {log: ['|move|p2b: Whiscash|Icy Wind|p1a: Latios|[spread] p1a,p1b'], request: null};
    const actual = {log: ['|move|p2b: Whiscash|Icy Wind|p1b: Gyarados|[spread] p1a,p1b'], request: null};
    const mismatch = mismatchEvidence(actual, expected);
    assert.equal(mismatch.kind, 'move_target_or_chance_mismatch'); assert(mismatch.distance > 0);
    assert.notEqual(mismatch.observed_event, mismatch.simulated_event);
});

test('belief failure capture is bounded detached and retained only for the first failed transition', () => {
    const battle = createHybridFixture();
    try {
        const tracker = new PlannerBeliefs({seed: SEED, particles: 1, max_nodes: 64, max_elapsed_ms: 2000});
        const failure = controlFailure(battle);
        assert(tracker.update(failure.history.observations[0]).supported);
        const failed = tracker.update(failure.history.observations[1], failure.history.own_choices[0]);
        assert(!failed.supported); assert(tracker.firstFailure);
        assert.equal(tracker.firstFailure.source, 'player_observations_and_sampled_world_only');
        assert.equal(tracker.firstFailure.simulation_trace.position.origin, 'belief-sample');
        assert(Buffer.byteLength(JSON.stringify(tracker.firstFailure)) <= 512 * 1024);
        const saved = structuredClone(tracker.firstFailure);
        tracker.update(failure.history.observations[1]);
        assert.deepEqual(tracker.firstFailure, saved);
        failure.history.observations[1].request.active[0].moves[0].pp = 998;
        assert.deepEqual(tracker.firstFailure, saved);
    } finally { battle.destroy(); }
});
