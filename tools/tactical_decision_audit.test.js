'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const {createDecisionAudit, summarizeDecisions} = require('./tactical_decision_audit');

function decision(overrides = {}) {
    return {version: 1, forced_switch: false, selected_score: 50, friendly_power: 0,
        unscored_moves: 0, teacher_score: 100, retargeted_score: 80,
        teacher_command: '/choose move 2 1', slots: [
            {action: 0, move_id: 1, damaging: true, protect: false, tera: false, target_choices: 2},
            {action: -1, move_id: 0, damaging: null, protect: false, tera: false, target_choices: 0}], ...overrides};
}

test('audit records accepted choices only, excluding rejected retries', () => {
    const audit = createDecisionAudit();
    const action = {request_id: 2, command: '/choose move 1 2', training_eligible: false, decision_audit: decision()};
    audit.propose(action);
    audit.observe(JSON.stringify({...action, type: 'decision_rejected'}));
    assert.equal(audit.decisions.length, 0);
    audit.propose({...action, request_id: 3});
    assert.throws(() => audit.observe(JSON.stringify({type: 'decision_accepted', request_id: 2})), /does not match/);
    audit.observe(JSON.stringify({...action, request_id: 3, type: 'decision_accepted'}));
    assert.equal(audit.finish({policy_tag: 'tactical-audit-v1', training_eligible: false, audit_decisions: 1}).length, 1);
    assert.throws(() => audit.finish({policy_tag: 'tactical-audit-v1', training_eligible: false, audit_decisions: 2}), /terminal count/);
});

test('audit separates retargeting from remaining action gap and does not score unsupported status choices', () => {
    const protect = decision({unscored_moves: 1, selected_score: 0, retargeted_score: 0,
        slots: [{action: 1, move_id: 2, damaging: false, protect: true, tera: false, target_choices: 1},
            decision().slots[1]]});
    const results = summarizeDecisions([{opponent: 'random', index: 0, teacher_side: 'a',
        decision_audit: [decision(), protect, protect, decision({forced_switch: true}), protect]}]);
    assert.equal(results.decisions, 5);
    assert.equal(results.mean_teacher_gap, 50);
    assert.equal(results.mean_target_gap, 30);
    assert.equal(results.mean_action_gap_after_retargeting, 20);
    assert.equal(results.target_comparisons, 1);
    assert.equal(results.partially_scored, 3);
    assert.equal(results.protect_moves, 3);
    assert.equal(results.consecutive_protect, 1);
    assert.equal(results.forced_switches, 1);
    assert.equal(results.examples.length, 1);
    assert.equal(summarizeDecisions([]).mean_teacher_gap, null);
});

test('audit weights decisions rather than averaging battles and bounds examples', () => {
    const result = summarizeDecisions([{decision_audit: [decision({selected_score: 0})]},
        {decision_audit: Array.from({length: 20}, () => decision())}]);
    assert.equal(result.mean_teacher_gap, 1100 / 21);
    assert.equal(result.examples.length, 10);
    assert.equal(result.examples[0].gap, 100);
    assert.throws(() => summarizeDecisions([{}]), /no accepted decisions/);
    assert.throws(() => summarizeDecisions([{decision_audit: [decision({teacher_score: NaN})]}]), /malformed/);
});
