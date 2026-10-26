'use strict';

function validateDecision(audit) {
    if (audit?.version !== 1 || typeof audit.forced_switch !== 'boolean' ||
            !Number.isFinite(audit.selected_score) || !Number.isFinite(audit.friendly_power) ||
            !Number.isSafeInteger(audit.unscored_moves) || audit.unscored_moves < 0 ||
            ![audit.teacher_score, audit.retargeted_score].every(value => value === null || Number.isFinite(value)) ||
            !Array.isArray(audit.slots) || audit.slots.length !== 2 || audit.slots.some(slot =>
                !Number.isInteger(slot.action) || slot.action < -1 || slot.action >= 28 ||
                !Number.isInteger(slot.move_id) || !Number.isInteger(slot.target_choices) ||
                slot.target_choices < 0 || slot.target_choices > 4 ||
                ![true, false, null].includes(slot.damaging) ||
                typeof slot.protect !== 'boolean' || typeof slot.tera !== 'boolean'))
        throw new Error('missing or malformed native decision audit');
}

function createDecisionAudit() {
    let pending = null;
    const decisions = [];
    return {decisions,
        propose(action) {
            if (pending) throw new Error('audit proposal was not accepted or rejected');
            if (action.training_eligible !== false) throw new Error('audit action must be evaluation-only');
            validateDecision(action.decision_audit);
            pending = action;
        },
        observe(line) {
            const message = JSON.parse(line);
            if (!['decision_accepted', 'decision_rejected'].includes(message.type)) return;
            if (!pending || message.request_id !== pending.request_id || message.command !== pending.command)
                throw new Error('audit acceptance does not match the proposed choice');
            if (message.type === 'decision_accepted') decisions.push({request_id: pending.request_id,
                command: pending.command, ...pending.decision_audit});
            pending = null;
        },
        finish(episode) {
            if (pending || episode.policy_tag !== 'tactical-audit-v1' || episode.training_eligible !== false ||
                    episode.audit_decisions !== decisions.length || !decisions.length)
                throw new Error('audit terminal count does not match accepted decisions');
            return decisions;
        },
    };
}

function summarizeDecisions(games) {
    const totals = {decisions: 0, forced_switches: 0, ordinary_decisions: 0, teacher_supported: 0,
        fully_scored: 0, partially_scored: 0, target_comparisons: 0, target_gap_sum: 0,
        teacher_gap_sum: 0, teacher_gap_count: 0, action_gap_sum: 0, action_gap_count: 0, friendly_power_sum: 0,
        moves: 0, damaging_moves: 0, status_moves: 0, unknown_moves: 0, protect_moves: 0,
        consecutive_protect: 0, tera_moves: 0, voluntary_switches: 0, explicit_target_moves: 0,
        move_counts: {}, examples: []};
    for (const game of games) {
        if (!Array.isArray(game.decision_audit) || !game.decision_audit.length)
            throw new Error('saved audit game has no accepted decisions');
        let previousProtect = [false, false];
        for (const [index, decision] of game.decision_audit.entries()) {
            validateDecision(decision);
            ++totals.decisions;
            if (decision.forced_switch) {
                ++totals.forced_switches;
                previousProtect = [false, false];
                continue;
            }
            ++totals.ordinary_decisions;
            ++totals[decision.unscored_moves ? 'partially_scored' : 'fully_scored'];
            totals.friendly_power_sum += decision.friendly_power;
            if (decision.teacher_score !== null) ++totals.teacher_supported;
            // Unsupported status/Tera choices are not evidence of lost damage.
            if (!decision.unscored_moves && decision.teacher_score !== null) {
                ++totals.teacher_gap_count;
                totals.teacher_gap_sum += decision.teacher_score - decision.selected_score;
                if (decision.retargeted_score !== null) {
                    ++totals.action_gap_count;
                    totals.action_gap_sum += decision.teacher_score - decision.retargeted_score;
                }
            }
            if (!decision.unscored_moves && decision.retargeted_score !== null &&
                    decision.slots.some(slot => slot.target_choices > 1)) {
                ++totals.target_comparisons;
                totals.target_gap_sum += decision.retargeted_score - decision.selected_score;
            }
            const nowProtect = [false, false];
            decision.slots.forEach((slot, side) => {
                if (slot.action < 0) return;
                if (slot.action % 14 >= 8) { ++totals.voluntary_switches; return; }
                ++totals.moves;
                ++totals[slot.damaging === null ? 'unknown_moves' : slot.damaging ? 'damaging_moves' : 'status_moves'];
                totals.move_counts[slot.move_id] = (totals.move_counts[slot.move_id] || 0) + 1;
                if (slot.protect) {
                    ++totals.protect_moves;
                    if (previousProtect[side]) ++totals.consecutive_protect;
                    nowProtect[side] = true;
                }
                if (slot.tera) ++totals.tera_moves;
                if (slot.target_choices > 1) ++totals.explicit_target_moves;
            });
            previousProtect = nowProtect;
            if (!decision.unscored_moves && decision.teacher_score !== null) {
                totals.examples.push({opponent: game.opponent, game_index: game.index, side: game.teacher_side,
                    decision_index: index, request_id: decision.request_id, command: decision.command,
                    teacher_command: decision.teacher_command,
                    gap: decision.teacher_score - decision.selected_score,
                    target_gap: decision.retargeted_score === null ? null : decision.retargeted_score - decision.selected_score});
                totals.examples.sort((a, b) => b.gap - a.gap);
                totals.examples.length = Math.min(10, totals.examples.length);
            }
        }
    }
    totals.mean_teacher_gap = totals.teacher_gap_count ? totals.teacher_gap_sum / totals.teacher_gap_count : null;
    totals.mean_target_gap = totals.target_comparisons ? totals.target_gap_sum / totals.target_comparisons : null;
    totals.mean_action_gap_after_retargeting = totals.action_gap_count ? totals.action_gap_sum / totals.action_gap_count : null;
    totals.protect_rate = totals.moves ? totals.protect_moves / totals.moves : null;
    return totals;
}

module.exports = {createDecisionAudit, summarizeDecisions, validateDecision};
