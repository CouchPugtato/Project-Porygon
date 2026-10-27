#include "target_labels.h"
#include "target_context.h"
#include <float.h>
#include <math.h>
#include <string.h>

typedef struct {
    int actions[2];
    unsigned char legal[2];
    float scores[2][4];
} TargetScores;

static int score_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    TargetScores* scores = context;
    int actions[2], slot;
    if (!factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            actions[0] != scores->actions[0] || actions[1] != scores->actions[1] ||
            !tactical_evaluate_power(observation, request, choice, evaluation, NULL) || evaluation->unscored_moves) return 0;
    for (slot = 0; slot < 2; ++slot) {
        int target = slot ? choice->slot1_target_index : choice->slot0_target_index;
        unsigned char mask = slot ? choice->slot1_target_mask : choice->slot0_target_mask;
        if (!mask || target >= 4) continue;
        scores->legal[slot] |= 1 << target;
        if (evaluation->score > scores->scores[slot][target]) scores->scores[slot][target] = evaluation->score;
    }
    return 1;
}

int target_labels_collect(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TargetLabels* labels
) {
    TargetScores scores = {0};
    TacticalDecision ignored;
    int slot, target, count = 0;
    memset(labels, 0, sizeof(*labels));
    if (request->forced_switch_any || !factorized_action_choice_to_flat_actions(choice,
            &scores.actions[0], &scores.actions[1])) return 0;
    for (slot = 0; slot < 2; ++slot) for (target = 0; target < 4; ++target) scores.scores[slot][target] = -FLT_MAX;
    if (!tactical_select_choice(observation, request, score_targets, &scores, &ignored)) return 0;
    for (slot = 0; slot < 2; ++slot) {
        float best = -FLT_MAX;
        for (target = 0; target < 4; ++target) if (scores.legal[slot] & (1 << target))
            if (scores.scores[slot][target] > best) best = scores.scores[slot][target];
        for (target = 0; target < 4; ++target) if ((scores.legal[slot] & (1 << target)) &&
                fabsf(scores.scores[slot][target] - best) <= 1e-5f) labels->best[slot] |= 1 << target;
        if (labels->best[slot] == scores.legal[slot]) labels->best[slot] = 0;
        if (labels->best[slot]) { labels->legal[slot] = scores.legal[slot]; ++count; }
    }
    return count;
}

void target_labels_write(FILE* out, const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, const float* hidden, size_t hidden_dim, uint64_t fingerprint
) {
    TargetLabels labels;
    int slot, first = 1;
    size_t h;
    int count = target_labels_collect(observation, request, choice, &labels);
    fprintf(out, "{\"parent_fingerprint\":\"%016llx\",\"examples\":[", (unsigned long long)fingerprint);
    for (slot = 0; slot < 2; ++slot) if (labels.best[slot]) {
        TargetContext context;
        target_context_collect(observation, request, choice, slot, &context);
        fprintf(out, "%s{\"slot\":%d,\"move\":%u,\"legal\":%u,\"best\":%u,\"context\":", first ? "" : ",", slot,
            slot ? choice->slot1_move_index : choice->slot0_move_index, labels.legal[slot], labels.best[slot]);
        target_context_write(out, &context);
        fputc('}', out);
        first = 0;
    }
    fputs("],\"hidden\":[", out);
    if (count) for (h = 0; h < hidden_dim; ++h) fprintf(out, "%s%.9g", h ? "," : "", hidden[h]);
    fputs("]}", out);
}
