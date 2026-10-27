'use strict';
const {validateContext, targetPolicyTag} = require('./target_context');

function validateTargetAudit(a) {
    const reasons = ['supported', 'forced_switch', 'unknown_visible_types', 'tera', 'unknown_move',
        'status_move', 'unsupported_move_metadata', 'unsupported_target_kind'];
    if (![1, 2, 3].includes(a?.version) || !Number.isSafeInteger(a.turn) || a.turn < 0 ||
            typeof a.learned_used !== 'boolean' || !reasons.includes(a.support_reason) || typeof a.comparable !== 'boolean' ||
            ![a.selected_score, a.reference_score, a.gap].every(v => v === null || Number.isFinite(v)) ||
            (a.comparable ? a.support_reason !== 'supported' || ![a.selected_score, a.reference_score, a.gap].every(Number.isFinite) || a.gap < 0 :
                [a.selected_score, a.reference_score, a.gap].some(v => v !== null)) ||
            !Array.isArray(a.slots) || a.slots.length !== 2) throw new Error('invalid target audit decision');
    if (a.comparable && Math.abs(a.gap - Math.max(0, a.reference_score - a.selected_score)) > .002)
        throw new Error('inconsistent target audit joint gap');
    for (const [i, s] of a.slots.entries()) {
        if (a.version === 3 && (![...reasons, 'not_move'].includes(s.support_reason) ||
                typeof s.learned_used !== 'boolean' || !Number.isSafeInteger(s.request_target_kind) ||
                s.request_target_kind < 0 || s.request_target_kind > 12 ||
                (s.learned_used && (s.support_reason !== 'supported' || s.kind !== 1 || !s.legal)) ||
                (s.gap !== null && s.support_reason !== 'supported')))
            throw new Error('invalid slot-local target eligibility');
        if (s.slot !== i || ![0, 1, 2].includes(s.kind) || !Number.isInteger(s.selected_target) || s.selected_target < 0 || s.selected_target > 3 ||
                !Number.isInteger(s.legal) || s.legal < 0 || s.legal > 15 || !Number.isInteger(s.best_mask) || s.best_mask < 0 ||
                (s.best_mask & ~s.legal) || !Array.isArray(s.target_scores) || s.target_scores.length !== 4 ||
                !s.target_scores.every(v => v === null || Number.isFinite(v)) ||
                (s.gap !== null && (!Number.isFinite(s.gap) || s.gap < 0 || (a.version !== 3 && !a.comparable) || !s.best_mask || !(s.legal & (1 << s.selected_target)))) ||
                (s.gap === null && (s.best_mask || s.target_scores.some(v => v !== null)))) throw new Error('invalid target audit slot');
        if (s.kind === 1) validateContext(s.context, true); else if (s.context !== null) throw new Error('non-move target context');
        if (s.gap !== null) {
            if (s.kind !== 1 || s.target_scores.some((v, t) => (s.legal & (1 << t)) ? !Number.isFinite(v) : v !== null))
                throw new Error('invalid scored target mask');
            const best = Math.max(...s.target_scores.filter(Number.isFinite));
            if (Math.abs(s.gap - Math.max(0, best - s.target_scores[s.selected_target])) > .002 ||
                    s.target_scores.some((v, t) => (s.best_mask & (1 << t)) && Math.abs(v - best) > .002))
                throw new Error('inconsistent fixed-partner target gap');
        }
    }
    if (a.version === 3 && a.learned_used !== a.slots.some(s => s.learned_used))
        throw new Error('slot-local learned target accounting mismatch');
}

function createTargetAudit() {
    let pending = null;
    const decisions = [], accepted = new Set();
    return {
        propose(action) {
            if (pending || action.training_eligible !== false) throw new Error('target audit proposal is not evaluation-only or is still pending');
            validateTargetAudit(action.target_audit);
            pending = action;
        },
        observe(line) {
            const m = JSON.parse(line);
            if (!['decision_accepted', 'decision_rejected'].includes(m.type)) return;
            if (!pending || m.request_id !== pending.request_id || m.command !== pending.command)
                throw new Error('target audit acceptance does not match proposal');
            if (m.type === 'decision_accepted') {
                if (accepted.has(m.request_id)) throw new Error('duplicate accepted target audit request');
                accepted.add(m.request_id);
                decisions.push({request_id: pending.request_id, command: pending.command, ...pending.target_audit});
            }
            pending = null;
        },
        finish(episode) {
            if (pending || !decisions.length || episode.training_eligible !== false || episode.target_audit !== true ||
                    episode.policy_tag !== targetPolicyTag({...episode, context_target_evaluation: true}) || episode.target_decoder !== 'argmax' ||
                    decisions.some(d => (d.version === 3) !== !!episode.target_slot_local) ||
                    episode.target_decisions !== decisions.length || episode.learned_target_decisions !== decisions.filter(d => d.learned_used).length)
                throw new Error('target audit terminal accounting mismatch');
            return decisions;
        },
    };
}

function summarizeTargetAudit(games) {
    const result = {decisions: 0, learned_used: 0, comparable_decisions: 0, target_opportunities: 0,
        supported_target_opportunities: 0, informative_slots: 0, disagreements: 0, joint_gap_sum: 0, slot_gap_sum: 0, friendly_targets: 0,
        support_reasons: {}, opportunity_support_reasons: {}, by_move: {}, by_move_type: {}, by_selected_target_types: {}, by_relation: {}, by_stage: {}, examples: [],
        slot_opportunities: 0, supported_slot_opportunities: 0, learned_slots: 0, slot_support_reasons: {},
        newly_supported_slot_opportunities: 0, newly_supported_informative_slots: 0, newly_supported_disagreements: 0,
        newly_supported_slot_gap_sum: 0, slot_examples: [], metrics_versions: {}};
    const add = (table, key, gap, mismatch) => {
        const row = table[key] ||= {opportunities: 0, disagreements: 0, slot_gap_sum: 0};
        ++row.opportunities; row.disagreements += +mismatch; row.slot_gap_sum += gap;
    };
    for (const game of games) {
        if (!Array.isArray(game.target_audit) || !game.target_audit.length) throw new Error('saved target audit has no accepted decisions');
        for (const d of game.target_audit) {
            validateTargetAudit(d); ++result.decisions; result.learned_used += +d.learned_used;
            result.metrics_versions[d.version] = (result.metrics_versions[d.version] || 0) + 1;
            result.support_reasons[d.support_reason] = (result.support_reasons[d.support_reason] || 0) + 1;
            const opportunity = d.slots.some(s => s.legal && (s.legal & (s.legal - 1)));
            if (opportunity) {
                ++result.target_opportunities;
                result.opportunity_support_reasons[d.support_reason] = (result.opportunity_support_reasons[d.support_reason] || 0) + 1;
            }
            result.supported_target_opportunities += +(d.version === 3 ? d.slots.some(s =>
                s.support_reason === 'supported' && s.legal && (s.legal & (s.legal - 1))) : d.comparable && opportunity);
            if (d.comparable) { ++result.comparable_decisions; result.joint_gap_sum += d.gap; }
            for (const s of d.slots) {
                const slotOpportunity = s.legal && (s.legal & (s.legal - 1));
                const supported = d.version === 3 ? s.support_reason === 'supported' : d.support_reason === 'supported';
                const newlySupported = d.version === 3 && supported && d.support_reason !== 'supported';
                if (d.version === 3) result.learned_slots += +s.learned_used;
                if (slotOpportunity) {
                    ++result.slot_opportunities;
                    result.supported_slot_opportunities += +supported;
                    result.newly_supported_slot_opportunities += +newlySupported;
                    const reason = d.version === 3 ? s.support_reason : d.support_reason;
                    result.slot_support_reasons[reason] = (result.slot_support_reasons[reason] || 0) + 1;
                }
                if (s.gap === null || s.best_mask === s.legal) continue;
                ++result.informative_slots;
                const mismatch = !(s.best_mask & (1 << s.selected_target));
                result.disagreements += +mismatch; result.slot_gap_sum += s.gap;
                if (newlySupported) {
                    ++result.newly_supported_informative_slots;
                    result.newly_supported_disagreements += +mismatch;
                    result.newly_supported_slot_gap_sum += s.gap;
                }
                if (s.gap > 1e-5) {
                    result.slot_examples.push({opponent: game.opponent, game_index: game.index, side: game.teacher_side,
                        request_id: d.request_id, command: d.command, turn: d.turn, newly_supported: newlySupported, ...s});
                    result.slot_examples.sort((a, b) => b.gap - a.gap); result.slot_examples.length = Math.min(12, result.slot_examples.length);
                }
                const p = s.context.targets[s.selected_target], relation = p.friendly ? (s.selected_target === 0 ? 'self' : 'ally') : 'foe';
                result.friendly_targets += +p.friendly;
                add(result.by_move, s.context.move_id, s.gap, mismatch);
                add(result.by_move_type, s.context.move_type_id, s.gap, mismatch);
                add(result.by_selected_target_types, [p.type1_id, p.type2_id].join('/'), s.gap, mismatch);
                add(result.by_relation, relation, s.gap, mismatch);
                add(result.by_stage, d.turn <= 5 ? 'opening_through_5' : d.turn <= 15 ? 'middle_6_15' : 'late_16_plus', s.gap, mismatch);
            }
            if (d.gap > 1e-5) {
                result.examples.push({opponent: game.opponent, game_index: game.index, side: game.teacher_side,
                    request_id: d.request_id, command: d.command, ...d});
                result.examples.sort((a, b) => b.gap - a.gap); result.examples.length = Math.min(12, result.examples.length);
            }
        }
    }
    result.agreement_rate = result.informative_slots ? 1 - result.disagreements / result.informative_slots : null;
    result.learned_use_rate = result.decisions ? result.learned_used / result.decisions : null;
    result.slot_coverage_rate = result.slot_opportunities ? result.supported_slot_opportunities / result.slot_opportunities : null;
    result.newly_supported_agreement_rate = result.newly_supported_informative_slots ?
        1 - result.newly_supported_disagreements / result.newly_supported_informative_slots : null;
    result.heuristic_gap_note = 'Effective-power score differences only; not actual HP damage, counterfactual wins, or causal win impact. Joint and slot gaps are separate measures. Version 3 slot scores include only that attack, preserving an incoming ally switch; unsupported partner moves are not assigned a score.';
    return result;
}

/* Version-one contexts omit request target kinds and why a target is absent.
 * Only reuse scores when every scored location is present. In particular, do
 * not turn a presence filter into a claimed correction of the missing cases. */
function rescoreRetainedTargetAudit(games) {
    let retained = 0, unresolved = 0, unscored = 0;
    const safeGames = [];
    for (const game of games) {
        const decisions = [];
        for (const d of game.target_audit) {
            validateTargetAudit(d);
            if (!d.comparable) { ++unscored; continue; }
            const missing = d.slots.some(s => s.gap !== null && s.context.targets.some((p, t) =>
                (s.legal & (1 << t)) && !p.present));
            if (missing) { ++unresolved; continue; }
            decisions.push(d); ++retained;
        }
        if (decisions.length) safeGames.push({...game, target_audit: decisions});
    }
    return {metrics_version: 2, retained_decisions: retained, unresolved_decisions: unresolved,
        originally_unscored_decisions: unscored, corrected_cached_decisions: 0,
        limitation: 'Partial retained-context view, not corrected full-audit totals. Absent-location cases lack request target kinds and confirmed fainted state; they remain unresolved. Present-location scores are unchanged.',
        retained: summarizeTargetAudit(safeGames)};
}

module.exports = {createTargetAudit, summarizeTargetAudit, validateTargetAudit, rescoreRetainedTargetAudit};

if (require.main === module) {
    const fs = require('node:fs'), crypto = require('node:crypto');
    if (process.argv.length !== 3) throw new Error('usage: node tools/target_audit.js <saved-summary.json>');
    const source = fs.readFileSync(process.argv[2]);
    const saved = JSON.parse(source);
    console.log(JSON.stringify({diagnostic: 'retained_target_audit_rescore', source: process.argv[2],
        source_sha256: crypto.createHash('sha256').update(source).digest('hex'),
        ...rescoreRetainedTargetAudit(saved.games)}, null, 2));
}
