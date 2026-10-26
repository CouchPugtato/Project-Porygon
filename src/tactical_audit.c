#include "tactical_audit.h"
#include "id_tables.h"

#include <string.h>

static int evaluate_teacher(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    return tactical_evaluate_power(observation, request, choice, evaluation, context) &&
        evaluation->unscored_moves == 0 && (request->forced_switch_any || evaluation->score > 0.0f);
}

static int evaluate_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    const int* selected = context;
    int actions[2];
    if (!factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            actions[0] != selected[0] || actions[1] != selected[1]) return 0;
    return tactical_evaluate_power(observation, request, choice, evaluation, NULL) &&
        evaluation->unscored_moves == 0;
}

int tactical_audit_choice(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit
) {
    int actions[2];
    if (!observation || !request || !choice || !audit) return 0;
    memset(audit, 0, sizeof(*audit));
    if (!factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            !tactical_evaluate_power(observation, request, choice, &audit->selected, NULL)) return 0;
    audit->teacher_available = tactical_select_choice(observation, request, evaluate_teacher, NULL, &audit->teacher);
    audit->retargeted_available = tactical_select_choice(observation, request, evaluate_targets, actions, &audit->retargeted);
    return 1;
}

void tactical_audit_write_json(FILE* out, const Observation* observation,
    const ParsedRequest* request, const FactorizedActionChoice* choice
) {
    TacticalAudit audit;
    char command[256];
    int actions[2], slot;
    if (!tactical_audit_choice(observation, request, choice, &audit)) {
        fputs("null", out);
        return;
    }
    factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]);
    fprintf(out, "{\"version\":1,\"forced_switch\":%s,\"selected_score\":%.9g,"
        "\"friendly_power\":%.9g,\"unscored_moves\":%zu,\"teacher_score\":",
        request->forced_switch_any ? "true" : "false", audit.selected.score,
        audit.selected.friendly_power, audit.selected.unscored_moves);
    if (audit.teacher_available) fprintf(out, "%.9g", audit.teacher.evaluation.score);
    else fputs("null", out);
    fputs(",\"retargeted_score\":", out);
    if (audit.retargeted_available) fprintf(out, "%.9g", audit.retargeted.evaluation.score);
    else fputs("null", out);
    fputs(",\"teacher_command\":", out);
    /* Commands contain only generated action tokens and numeric targets. */
    if (audit.teacher_available && factorized_choice_to_command(request, &audit.teacher.choice, command, sizeof(command)))
        fprintf(out, "\"%s\"", command);
    else fputs("null", out);
    fputs(",\"slots\":[", out);
    for (slot = 0; slot < 2; ++slot) {
        int local = actions[slot] - slot * 14;
        int move = actions[slot] >= 0 && local < 8;
        int id = move ? request->active[slot].move_id[local % 4] : 0;
        MoveSemanticMetadata metadata;
        int known = move && move_semantics_from_id(id, &metadata);
        unsigned char targets = slot ? choice->slot1_target_mask : choice->slot0_target_mask;
        int target_count = 0;
        while (targets) { target_count += targets & 1; targets >>= 1; }
        fprintf(out, "%s{\"action\":%d,\"move_id\":%d,\"damaging\":%s,\"protect\":%s,"
            "\"tera\":%s,\"target_choices\":%d}", slot ? "," : "", actions[slot], id,
            known ? (metadata.damaging ? "true" : "false") : "null",
            move && id == move_id_from_name("protect") ? "true" : "false",
            move && local >= 4 ? "true" : "false", target_count);
    }
    fputs("]}", out);
}
