'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
    buildCounterfactualSample,
    choicesDiffer,
    equalPrefix,
    seedTuple,
} = require('./counterfactual_rollout');

function sampleAction() {
    return {
        counterfactual: {decision_index: 3, action_rank: 1},
        action: 2,
        action2: 17,
        slot0_has_action: 1,
        slot0_kind: 0,
        slot0_move_index: 2,
        slot0_switch_index: 0,
        slot0_use_tera: 0,
        slot0_target_index: 1,
        slot0_target_mask: 3,
        slot1_has_action: 1,
        slot1_kind: 0,
        slot1_move_index: 3,
        slot1_switch_index: 0,
        slot1_use_tera: 0,
        slot1_target_index: 0,
        slot1_target_mask: 1,
        baseline_value: 0.2,
        behavior_log_prob: -2.3,
        hidden_dim: 3,
        hidden_state: [0.1, 0.2, 0.3],
        legal_mask: Array(28).fill(1),
    };
}

test('counterfactual records store the compact recurrent state and intervention', () => {
    const record = buildCounterfactualSample(sampleAction(), 'pair-7', 'parent.chk', -1);
    assert.equal(record.type, 'counterfactual_sample');
    assert.equal(record.pair_id, 'pair-7');
    assert.equal(record.action_rank, 1);
    assert.equal(record.decision_index, 3);
    assert.equal(record.action2, 17);
    assert.equal(record.target_value, -1);
    assert.deepEqual(record.hidden_state, [0.1, 0.2, 0.3]);
    assert.equal(record.legal_mask.length, 28);
});

test('paired-state and action comparisons are strict', () => {
    assert.equal(equalPrefix([1, 2, 9], [1, 2, 8], 2), true);
    assert.equal(equalPrefix([1, 2], [1, 3], 2), false);
    assert.equal(choicesDiffer(
        {action: 1, action2: 15, slot0_target: 0, slot1_target: 1},
        {action: 1, action2: 15, slot0_target: 0, slot1_target: 2}
    ), true);
});

test('Showdown seed tuples are deterministic and attempt-specific', () => {
    assert.equal(seedTuple(17, 4, 2), seedTuple(17, 4, 2));
    assert.notEqual(seedTuple(17, 4, 2), seedTuple(17, 5, 2));
    assert.match(seedTuple(17, 4, 2), /^\d+,\d+,\d+,\d+$/);
});
