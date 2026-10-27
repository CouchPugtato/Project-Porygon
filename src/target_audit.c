#include "target_audit.h"
#include "target_context.h"
#include "target_sidecar.h"
#include <float.h>
#include <math.h>

void target_audit_write(FILE* out, const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int turn, unsigned char learned_slots, int slot_local
) {
    TacticalAudit audit;
    const char* reason = target_sidecar_support_reason(observation, request, choice);
    int comparable = !reason && tactical_diagnostic_targets(observation, request, choice, &audit) &&
        !audit.selected.unscored_moves && audit.retargeted_available;
    int slot;
    fprintf(out, "{\"version\":%d,\"turn\":%d,\"learned_used\":%s,\"support_reason\":\"%s\",\"comparable\":%s,",
        slot_local ? 3 : 2, turn, learned_slots ? "true" : "false", reason ? reason : "supported", comparable ? "true" : "false");
    if (comparable) fprintf(out, "\"selected_score\":%.9g,\"reference_score\":%.9g,\"gap\":%.9g,",
        audit.selected.score, audit.retargeted.evaluation.score, fmaxf(0, audit.retargeted.evaluation.score - audit.selected.score));
    else fputs("\"selected_score\":null,\"reference_score\":null,\"gap\":null,", out);
    fputs("\"slots\":[", out);
    for (slot = 0; slot < 2; ++slot) {
        int has = slot ? choice->slot1_has_action : choice->slot0_has_action;
        int kind = slot ? choice->slot1_kind : choice->slot0_kind;
        int selected = slot ? choice->slot1_target_index : choice->slot0_target_index;
        unsigned char legal = slot ? choice->slot1_target_mask : choice->slot0_target_mask;
        const char* slot_reason = slot_local ? target_sidecar_slot_support_reason(observation, request, choice, slot) : reason;
        float scores[4] = {0}, best = -FLT_MAX;
        unsigned char best_mask = 0;
        int target, scored = (slot_local ? !slot_reason : comparable) && has && kind == FACTORIZED_ACTION_MOVE && legal;
        if (scored) for (target = 0; target < 4; ++target) if (legal & (1 << target)) {
            FactorizedActionChoice changed = *choice;
            TacticalChoiceEvaluation evaluation;
            if (slot) changed.slot1_target_index = (unsigned char)target;
            else changed.slot0_target_index = (unsigned char)target;
            int evaluated = slot_local ? tactical_diagnostic_slot_power(observation, request, &changed, slot, &evaluation) :
                tactical_diagnostic_power(observation, request, &changed, &evaluation, NULL);
            if (!evaluated || evaluation.unscored_moves) {
                scored = 0; break;
            }
            scores[target] = evaluation.score;
            if (scores[target] > best) best = scores[target];
        }
        if (scored) for (target = 0; target < 4; ++target)
            if ((legal & (1 << target)) && fabsf(scores[target] - best) <= 1e-5f) best_mask |= (unsigned char)(1 << target);
        fprintf(out, "%s{\"slot\":%d,\"kind\":%d,\"selected_target\":%d,\"legal\":%u,\"best_mask\":%u,\"gap\":",
            slot ? "," : "", slot, has ? kind : FACTORIZED_ACTION_NONE, selected, legal, best_mask);
        if (scored) fprintf(out, "%.9g", fmaxf(0, best - scores[selected])); else fputs("null", out);
        fputs(",\"target_scores\":[", out);
        for (target = 0; target < 4; ++target) {
            if (target) fputc(',', out);
            if (scored && (legal & (1 << target))) fprintf(out, "%.9g", scores[target]); else fputs("null", out);
        }
        fputs("],\"context\":", out);
        if (has && kind == FACTORIZED_ACTION_MOVE) {
            TargetContext context;
            target_context_collect(observation, request, choice, slot, &context);
            target_context_write(out, &context);
        } else fputs("null", out);
        if (slot_local) {
            int move = slot ? choice->slot1_move_index : choice->slot0_move_index;
            fprintf(out, ",\"support_reason\":\"%s\",\"learned_used\":%s,\"request_target_kind\":%d",
                slot_reason ? slot_reason : "supported", (learned_slots & (1u << slot)) ? "true" : "false",
                has && kind == FACTORIZED_ACTION_MOVE ? request->active[slot].move_target[move] : REQUEST_TARGET_UNKNOWN);
        }
        fputc('}', out);
    }
    fputs("]}", out);
}
