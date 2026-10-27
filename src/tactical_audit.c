#include "tactical_audit.h"
#include "id_tables.h"

#include <string.h>

static int present(const ObsPokemon* team, int slot) {
    int i;
    for (i = 0; i < OBS_TEAM_SIZE; ++i)
        if (team[i].active && !team[i].fainted && team[i].active_slot == slot + 1) return 1;
    return 0;
}

int tactical_diagnostic_power(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    FactorizedActionChoice resolved;
    int slot;
    (void)context;
    if (!observation || !request || !choice || !evaluation) return 0;
    resolved = *choice;
    for (slot = 0; slot < 2; ++slot) {
        int has = slot ? choice->slot1_has_action : choice->slot0_has_action;
        int kind = slot ? choice->slot1_kind : choice->slot0_kind;
        int move = slot ? choice->slot1_move_index : choice->slot0_move_index;
        unsigned char* target = slot ? &resolved.slot1_target_index : &resolved.slot0_target_index;
        ParsedMoveTarget target_kind;
        if (!has || kind != FACTORIZED_ACTION_MOVE) continue;
        target_kind = request->active[slot].move_target[move];
        if (*target >= FACTORIZED_TARGET_FOE_LEFT &&
                build_move_target_mask(request, slot, move) &&
                !present(observation->opp_team, *target - FACTORIZED_TARGET_FOE_LEFT)) {
            int other = 1 - (*target - FACTORIZED_TARGET_FOE_LEFT);
            /* This is a current-observation proxy, not a prediction of the board
             * at execution time. Unknown target kinds and empty boards abstain. */
            if (!request->is_doubles || !present(observation->opp_team, other) ||
                    (target_kind != REQUEST_TARGET_NORMAL && target_kind != REQUEST_TARGET_ADJACENT_FOE &&
                     target_kind != REQUEST_TARGET_ANY)) return 0;
            *target = (unsigned char)(FACTORIZED_TARGET_FOE_LEFT + other);
        } else if (*target == FACTORIZED_TARGET_ALLY && target_kind == REQUEST_TARGET_ADJACENT_ALLY_OR_SELF &&
                !present(observation->self_team, 1 - slot)) {
            int switching = slot ? choice->slot0_has_action && choice->slot0_kind == FACTORIZED_ACTION_SWITCH :
                choice->slot1_has_action && choice->slot1_kind == FACTORIZED_ACTION_SWITCH;
            if (!switching) *target = FACTORIZED_TARGET_SELF;
        }
    }
    return tactical_evaluate_power(observation, request, &resolved, evaluation, NULL);
}

int tactical_diagnostic_slot_power(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int slot, TacticalChoiceEvaluation* evaluation
) {
    FactorizedActionChoice isolated = *choice;
    /* Preserve an incoming partner switch when scoring friendly damage, but
     * do not assign a made-up score to the partner's unsupported status move. */
    if (slot == 0 && isolated.slot1_kind != FACTORIZED_ACTION_SWITCH) isolated.slot1_has_action = 0;
    if (slot == 1 && isolated.slot0_kind != FACTORIZED_ACTION_SWITCH) isolated.slot0_has_action = 0;
    return tactical_diagnostic_power(observation, request, &isolated, evaluation, NULL);
}

static int evaluate_diagnostic_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    const int* selected = context;
    int actions[2];
    return factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) &&
        actions[0] == selected[0] && actions[1] == selected[1] &&
        tactical_diagnostic_power(observation, request, choice, evaluation, NULL) && !evaluation->unscored_moves;
}

int tactical_diagnostic_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit
) {
    int actions[2];
    if (!observation || !request || !choice || !audit) return 0;
    memset(audit, 0, sizeof(*audit));
    if (!factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            !tactical_diagnostic_power(observation, request, choice, &audit->selected, NULL)) return 0;
    audit->retargeted_available = tactical_select_choice(observation, request,
        evaluate_diagnostic_targets, actions, &audit->retargeted);
    return 1;
}

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

int tactical_audit_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit
) {
    int actions[2];
    if (!observation || !request || !choice || !audit) return 0;
    memset(audit, 0, sizeof(*audit));
    if (!factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            !tactical_evaluate_power(observation, request, choice, &audit->selected, NULL)) return 0;
    audit->retargeted_available = tactical_select_choice(observation, request, evaluate_targets, actions, &audit->retargeted);
    return 1;
}

int tactical_audit_choice(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit
) {
    if (!tactical_audit_targets(observation, request, choice, audit)) return 0;
    audit->teacher_available = tactical_select_choice(observation, request, evaluate_teacher, NULL, &audit->teacher);
    return 1;
}

int tactical_retarget_choice(const Observation* observation, const ParsedRequest* request,
    FactorizedActionChoice* choice
) {
    TacticalChoiceEvaluation selected;
    TacticalDecision retargeted;
    int actions[2];
    if (!observation || !request || !choice || request->forced_switch_any ||
            !factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            !tactical_evaluate_power(observation, request, choice, &selected, NULL) ||
            selected.unscored_moves ||
            !tactical_select_choice(observation, request, evaluate_targets, actions, &retargeted) ||
            !(retargeted.evaluation.score > selected.score)) return 0;
    choice->slot0_target_index = retargeted.choice.slot0_target_index;
    choice->slot1_target_index = retargeted.choice.slot1_target_index;
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
