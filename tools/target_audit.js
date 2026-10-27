'use strict';
const {validateContext} = require('./target_context');

function validateTargetAudit(a) {
    const reasons = ['supported', 'forced_switch', 'unknown_visible_types', 'tera', 'unknown_move',
        'status_move', 'unsupported_move_metadata', 'unsupported_target_kind'];
    if (a?.version !== 1 || !Number.isSafeInteger(a.turn) || a.turn < 0 ||
            typeof a.learned_used !== 'boolean' || !reasons.includes(a.support_reason) || typeof a.comparable !== 'boolean' ||
            ![a.selected_score, a.reference_score, a.gap].every(v => v === null || Number.isFinite(v)) ||
            (a.comparable ? a.support_reason !== 'supported' || ![a.selected_score, a.reference_score, a.gap].every(Number.isFinite) || a.gap < 0 :
                [a.selected_score, a.reference_score, a.gap].some(v => v !== null)) ||
            !Array.isArray(a.slots) || a.slots.length !== 2) throw new Error('invalid target audit decision');
    if (a.comparable && Math.abs(a.gap - Math.max(0, a.reference_score - a.selected_score)) > .002)
        throw new Error('inconsistent target audit joint gap');
    for (const [i, s] of a.slots.entries()) {
        if (s.slot !== i || ![0, 1, 2].includes(s.kind) || !Number.isInteger(s.selected_target) || s.selected_target < 0 || s.selected_target > 3 ||
                !Number.isInteger(s.legal) || s.legal < 0 || s.legal > 15 || !Number.isInteger(s.best_mask) || s.best_mask < 0 ||
                (s.best_mask & ~s.legal) || !Array.isArray(s.target_scores) || s.target_scores.length !== 4 ||
                !s.target_scores.every(v => v === null || Number.isFinite(v)) ||
                (s.gap !== null && (!Number.isFinite(s.gap) || s.gap < 0 || !a.comparable || !s.best_mask || !(s.legal & (1 << s.selected_target)))) ||
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
                    episode.policy_tag !== 'target-context-argmax-v1' || episode.target_decoder !== 'argmax' ||
                    episode.target_decisions !== decisions.length || episode.learned_target_decisions !== decisions.filter(d => d.learned_used).length)
                throw new Error('target audit terminal accounting mismatch');
            return decisions;
        },
    };
}

function summarizeTargetAudit(games) {
    const result = {decisions: 0, learned_used: 0, comparable_decisions: 0, target_opportunities: 0,
        supported_target_opportunities: 0, informative_slots: 0, disagreements: 0, joint_gap_sum: 0, slot_gap_sum: 0, friendly_targets: 0,
        support_reasons: {}, opportunity_support_reasons: {}, by_move: {}, by_move_type: {}, by_selected_target_types: {}, by_relation: {}, by_stage: {}, examples: []};
    const add = (table, key, gap, mismatch) => {
        const row = table[key] ||= {opportunities: 0, disagreements: 0, slot_gap_sum: 0};
        ++row.opportunities; row.disagreements += +mismatch; row.slot_gap_sum += gap;
    };
    for (const game of games) {
        if (!Array.isArray(game.target_audit) || !game.target_audit.length) throw new Error('saved target audit has no accepted decisions');
        for (const d of game.target_audit) {
            validateTargetAudit(d); ++result.decisions; result.learned_used += +d.learned_used;
            result.support_reasons[d.support_reason] = (result.support_reasons[d.support_reason] || 0) + 1;
            const opportunity = d.slots.some(s => s.legal && (s.legal & (s.legal - 1)));
            if (opportunity) {
                ++result.target_opportunities;
                result.opportunity_support_reasons[d.support_reason] = (result.opportunity_support_reasons[d.support_reason] || 0) + 1;
            }
            if (!d.comparable) continue;
            result.supported_target_opportunities += +opportunity;
            ++result.comparable_decisions; result.joint_gap_sum += d.gap;
            for (const s of d.slots) {
                if (s.gap === null || s.best_mask === s.legal) continue;
                ++result.informative_slots;
                const mismatch = !(s.best_mask & (1 << s.selected_target));
                result.disagreements += +mismatch; result.slot_gap_sum += s.gap;
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
    result.heuristic_gap_note = 'Effective-power score differences only; not actual HP damage, counterfactual wins, or causal win impact. Joint and fixed-partner slot gaps are separate measures.';
    return result;
}

module.exports = {createTargetAudit, summarizeTargetAudit, validateTargetAudit};
