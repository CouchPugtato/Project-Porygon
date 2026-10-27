'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const {createTargetAudit, summarizeTargetAudit, validateTargetAudit, rescoreRetainedTargetAudit} = require('./target_audit');
const {validateConfig, validateResume, nextGame} = require('./tactical_teacher_benchmark');
const {createBattleAgents, battleSeeds} = require('./tactical_teacher_smoke');

function decision() {
    const context = {version: 1, move_id: 935, move_type_id: 2, move_type_known: true,
        targets: Array.from({length: 4}, (_, position) => ({position, present: true, friendly: position < 2,
            type1_id: position + 1, type1_known_mode: 2, type2_id: 0, type2_known_mode: 2}))};
    return {version: 1, turn: 8, learned_used: true, support_reason: 'supported', comparable: true,
        selected_score: 40, reference_score: 100, gap: 60, slots: [
            {slot: 0, kind: 1, selected_target: 2, legal: 12, best_mask: 8, gap: 60, target_scores: [null, null, 40, 100], context},
            {slot: 1, kind: 0, selected_target: 0, legal: 0, best_mask: 0, gap: null, target_scores: [null, null, null, null], context: null}]};
}
function unsupported() {
    const d = decision();
    Object.assign(d, {learned_used: false, comparable: false, support_reason: 'status_move', selected_score: null, reference_score: null, gap: null});
    Object.assign(d.slots[0], {best_mask: 0, gap: null, target_scores: [null, null, null, null]});
    return d;
}
test('passive target audit retains only accepted proposals and checks terminal accounting', () => {
    const audit = createTargetAudit(), action = {request_id: 1, command: '/choose move 1 1', training_eligible: false, target_audit: decision()};
    audit.propose(action); audit.observe(JSON.stringify({...action, type: 'decision_rejected'}));
    audit.propose({...action, request_id: 2});
    assert.throws(() => audit.observe(JSON.stringify({...action, type: 'decision_accepted'})), /does not match/);
    audit.observe(JSON.stringify({...action, request_id: 2, type: 'decision_accepted'}));
    const episode = {training_eligible: false, target_audit: true, policy_tag: 'target-context-argmax-v1', target_decoder: 'argmax', target_decisions: 1, learned_target_decisions: 1};
    assert.equal(audit.finish(episode).length, 1);
    assert.throws(() => audit.finish({...episode, learned_target_decisions: 0}), /accounting/);
});
test('target audit separates unsupported opportunities, tie agreement and heuristic-only impact', () => {
    const tie = decision(); tie.reference_score = tie.selected_score; tie.gap = 0;
    Object.assign(tie.slots[0], {best_mask: 12, gap: 0, target_scores: [null, null, 40, 40]});
    const s = summarizeTargetAudit([{target_audit: [decision(), tie, unsupported()]}]);
    assert.equal(s.decisions, 3); assert.equal(s.learned_used, 2); assert.equal(s.target_opportunities, 3);
    assert.equal(s.supported_target_opportunities, 2); assert.equal(s.informative_slots, 1); assert.equal(s.disagreements, 1);
    assert.equal(s.joint_gap_sum, 60); assert.equal(s.slot_gap_sum, 60); assert.equal(s.support_reasons.status_move, 1);
    assert.equal(s.by_move[935].disagreements, 1); assert.equal(s.by_stage.middle_6_15.slot_gap_sum, 60);
    assert.match(s.heuristic_gap_note, /not actual HP/);
    const malformed = decision(); malformed.slots[0].target_scores[1] = 1000;
    assert.throws(() => validateTargetAudit(malformed), /mask/);
});
test('passive target audit config freezes policy, routes only candidate telemetry and guards resume', () => {
    const config = require('../config/experimental/target_policy_audit.json');
    validateConfig(config);
    assert.equal(config.games_per_side * config.opponents.length * 2, 100);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(config, battleSeeds(config.seed, side, false), side, 'parent', null);
        assert.equal(agents[side].targetOptions.audit, true); assert.equal(agents[1 - side].targetOptions.audit, false);
        assert.equal(agents[side].targetOptions.decoder, 'argmax');
    }
    assert.throws(() => validateConfig({...config, target_decoder: 'sample'}), /passive target audit/);
    const identity = {config}, game = {...nextGame(config, []), status: 'completed', duration_seconds: 1, policy_mode: 'neural', coverage: null,
        result: {result: 'win'}, choice_rejections: [], policy_tag: 'target-context-argmax-v1', target_audit: [decision()],
        learned_targets: {training_eligible: false, target_decoder: 'argmax', target_audit: true, target_decisions: 1, learned_target_decisions: 1}};
    assert.deepEqual(validateResume({identity, games: [game]}, identity, config), [game]);
    assert.throws(() => validateResume({identity, games: [{...game, target_audit: []}]}, identity, config), /no accepted/);
    assert.throws(() => validateResume({identity, games: [game]}, identity, {...config, target_audit: false}), /unaudited/);
});

test('Showdown retargets a fainted foe location to the sole surviving foe in doubles', () => {
    const {Battle} = require('../external/pokemon-showdown/dist/sim');
    const team = [{species: 'Mew', moves: ['psychic']}, {species: 'Magikarp', moves: ['splash']}];
    const battle = new Battle({formatid: 'gen9doublescustomgame', seed: [1, 2, 3, 4],
        p1: {name: 'AuditFixture', team}, p2: {name: 'Opponent', team}});
    try {
        if (battle.requestState === 'teampreview') battle.makeChoices('team 12', 'team 12');
        battle.p2.active[0].fainted = true; battle.p2.active[0].hp = 0; battle.p2.pokemonLeft = 1;
        const attacker = battle.p1.active[0], survivor = battle.p2.active[1];
        assert.equal(battle.getTarget(attacker, 'psychic', 1), survivor);
        assert.equal(battle.getTarget(attacker, 'psychic', 2), survivor);
        const ally = battle.p1.active[1];
        ally.fainted = true; ally.hp = 0;
        assert.equal(battle.getTarget(attacker, 'psychic', -2), ally,
            'A fainted ally does not redirect an ordinary attack to a foe');
        assert.equal(battle.getTarget(attacker, 'helpinghand', -2), ally,
            'An ally-only move keeps the fainted ally target');
        assert.equal(battle.getTarget(attacker, 'acupressure', -2), attacker,
            'An ally-or-self move falls back to self in generation nine');
    } finally { battle.destroy(); }
});

test('retained rescoring abstains on absent alternatives without replacing original metrics', () => {
    const safe = decision(), missing = decision();
    Object.assign(missing.slots[0].context.targets[3], {present: false,
        type1_id: 0, type2_id: 0, type1_known_mode: 0, type2_known_mode: 0});
    const games = [{target_audit: [safe, missing, unsupported()]}], before = JSON.stringify(games);
    const result = rescoreRetainedTargetAudit(games);
    assert.equal(result.retained_decisions, 1); assert.equal(result.unresolved_decisions, 1);
    assert.equal(result.originally_unscored_decisions, 1); assert.equal(result.corrected_cached_decisions, 0);
    assert.equal(result.retained.joint_gap_sum, 60); assert.equal(JSON.stringify(games), before);
    const fresh = decision(); fresh.version = 2; validateTargetAudit(fresh);
});

function localDecision() {
    const d = unsupported();
    Object.assign(d, {version: 3, learned_used: true});
    Object.assign(d.slots[0], decision().slots[0], {support_reason: 'supported', learned_used: true, request_target_kind: 1});
    Object.assign(d.slots[1], {kind: 1, legal: 1, context: structuredClone(d.slots[0].context),
        support_reason: 'status_move', learned_used: false, request_target_kind: 6});
    return d;
}

test('slot-local audit grades the attack without inventing a joint or status-move score', () => {
    const d = localDecision(), s = summarizeTargetAudit([{target_audit: [d]}]);
    assert.equal(s.comparable_decisions, 0); assert.equal(s.joint_gap_sum, 0);
    assert.equal(s.supported_target_opportunities, 1); assert.equal(s.slot_coverage_rate, 1);
    assert.equal(s.informative_slots, 1); assert.equal(s.disagreements, 1);
    assert.equal(s.newly_supported_slot_opportunities, 1); assert.equal(s.newly_supported_slot_gap_sum, 60);
    assert.equal(s.learned_slots, 1); assert.equal(s.slot_examples[0].newly_supported, true);
    const malformed = structuredClone(d); malformed.slots[1].learned_used = true;
    assert.throws(() => validateTargetAudit(malformed), /eligibility/);
    const audit = createTargetAudit(), action = {request_id: 1, command: '/choose move 1 1, move 1', training_eligible: false, target_audit: d};
    audit.propose(action); audit.observe(JSON.stringify({...action, type: 'decision_accepted'}));
    const terminal = {policy_tag: 'target-context-slot-local-argmax-v1', target_decoder: 'argmax', target_slot_local: true,
        training_eligible: false, target_audit: true, target_decisions: 1, learned_target_decisions: 1};
    assert.equal(audit.finish(terminal).length, 1);
    assert.throws(() => audit.finish({...terminal, target_slot_local: false}), /accounting/);
});

test('slot-local audit uses a distinct policy identity and routes only the candidate', () => {
    const config = require('../config/experimental/target_policy_slot_local_audit.json');
    validateConfig(config);
    assert.equal(config.games_per_side * config.opponents.length * 2, 100);
    assert.notEqual(config.seed, require('../config/experimental/target_policy_audit.json').seed);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(config, battleSeeds(config.seed, side, false), side, 'parent', null);
        assert.equal(agents[side].targetOptions.slotLocal, true);
        assert.equal(agents[1 - side].targetOptions.slotLocal, false);
    }
    assert.throws(() => validateConfig({...config, context_target_evaluation: false}), /slot-local/);
    const identity = {config}, game = {...nextGame(config, []), status: 'completed', duration_seconds: 1, policy_mode: 'neural', coverage: null,
        result: {result: 'win'}, choice_rejections: [], policy_tag: 'target-context-slot-local-argmax-v1', target_audit: [localDecision()],
        learned_targets: {training_eligible: false, target_decoder: 'argmax', target_slot_local: true, target_audit: true,
            target_decisions: 1, learned_target_decisions: 1}};
    assert.deepEqual(validateResume({identity, games: [game]}, identity, config), [game]);
    assert.throws(() => validateResume({identity, games: [{...game, learned_targets: {...game.learned_targets, target_slot_local: false}}]}, identity, config), /accounting/);
});
