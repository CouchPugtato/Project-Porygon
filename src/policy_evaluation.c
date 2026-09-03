#include "policy_evaluation.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MIN_POLICY_PROBABILITY 1.0e-8f

static int argmax_masked(const float* policy, const unsigned char* mask, size_t count) {
    int best = -1;
    size_t i;
    for (i = 0; i < count; ++i) {
        if ((!mask || mask[i]) && (best < 0 || policy[i] > policy[best])) {
            best = (int)i;
        }
    }
    return best;
}

static int is_top_k(const float* policy, const unsigned char* mask, size_t count, size_t target, size_t k) {
    size_t better = 0;
    size_t i;
    if (target >= count || (mask && !mask[target])) {
        return 0;
    }
    for (i = 0; i < count; ++i) {
        if ((!mask || mask[i]) && policy[i] > policy[target]) {
            ++better;
        }
    }
    return better < k;
}

static float masked_probability(const float* policy, const unsigned char* mask, size_t count, size_t target) {
    float sum = 0.0f;
    size_t i;
    if (!policy || target >= count || (mask && !mask[target])) {
        return 0.0f;
    }
    for (i = 0; i < count; ++i) {
        if (!mask || mask[i]) {
            sum += policy[i];
        }
    }
    return sum > 0.0f ? policy[target] / sum : 0.0f;
}

static double negative_log_probability(float probability) {
    return -log((double)(probability > MIN_POLICY_PROBABILITY ? probability : MIN_POLICY_PROBABILITY));
}

static void build_factor_masks(
    const unsigned char* legal_mask,
    int slot,
    unsigned char* kind_mask,
    unsigned char* move_mask,
    unsigned char* switch_mask
) {
    int base = slot == 0 ? 0 : FACTORIZED_LOCAL_ACTION_DIM;
    int i;
    memset(kind_mask, 0, FACTORIZED_KIND_DIM);
    memset(move_mask, 0, FACTORIZED_MOVE_DIM);
    memset(switch_mask, 0, FACTORIZED_SWITCH_DIM);
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        if (legal_mask[base + i] || legal_mask[base + FACTORIZED_MOVE_DIM + i]) {
            kind_mask[0] = 1;
            move_mask[i] = 1;
        }
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        if (legal_mask[base + 8 + i]) {
            kind_mask[1] = 1;
            switch_mask[i] = 1;
        }
    }
}

static void build_local_policy(
    const FactorizedPolicySnapshot* snapshot,
    const unsigned char* legal_mask,
    int slot,
    float* local_policy
) {
    const float* kind = slot == 0 ? snapshot->slot0_kind_policy : snapshot->slot1_kind_policy;
    const float* move = slot == 0 ? snapshot->slot0_move_policy : snapshot->slot1_move_policy;
    const float* sw = slot == 0 ? snapshot->slot0_switch_policy : snapshot->slot1_switch_policy;
    const float* tera = slot == 0 ? snapshot->slot0_tera_policy : snapshot->slot1_tera_policy;
    int base = slot == 0 ? 0 : FACTORIZED_LOCAL_ACTION_DIM;
    float total = 0.0f;
    int i;

    memset(local_policy, 0, FACTORIZED_LOCAL_ACTION_DIM * sizeof(float));
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        float tera_total = 0.0f;
        if (legal_mask[base + i]) tera_total += tera[0];
        if (legal_mask[base + FACTORIZED_MOVE_DIM + i]) tera_total += tera[1];
        if (tera_total > 0.0f) {
            if (legal_mask[base + i]) local_policy[i] = kind[0] * move[i] * tera[0] / tera_total;
            if (legal_mask[base + FACTORIZED_MOVE_DIM + i]) {
                local_policy[FACTORIZED_MOVE_DIM + i] = kind[0] * move[i] * tera[1] / tera_total;
            }
        }
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        if (legal_mask[base + 8 + i]) local_policy[8 + i] = kind[1] * sw[i];
    }
    for (i = 0; i < FACTORIZED_LOCAL_ACTION_DIM; ++i) total += local_policy[i];
    if (total > 0.0f) {
        for (i = 0; i < FACTORIZED_LOCAL_ACTION_DIM; ++i) local_policy[i] /= total;
    }
}

static void build_joint_marginals(const float* joint_policy, float* slot0, float* slot1) {
    int action0;
    int action1;
    memset(slot0, 0, FACTORIZED_LOCAL_ACTION_DIM * sizeof(float));
    memset(slot1, 0, FACTORIZED_LOCAL_ACTION_DIM * sizeof(float));
    for (action0 = 0; action0 < FACTORIZED_LOCAL_ACTION_DIM; ++action0) {
        for (action1 = 0; action1 < FACTORIZED_LOCAL_ACTION_DIM; ++action1) {
            float probability = joint_policy[action0 * FACTORIZED_LOCAL_ACTION_DIM + action1];
            slot0[action0] += probability;
            slot1[action1] += probability;
        }
    }
}

static int record_component_metrics(
    const FactorizedPolicySnapshot* snapshot,
    const unsigned char* legal_mask,
    const FactorizedActionChoice* choice,
    int slot,
    PolicyEvaluationMetrics* metrics
) {
    const float* kind = slot == 0 ? snapshot->slot0_kind_policy : snapshot->slot1_kind_policy;
    const float* move = slot == 0 ? snapshot->slot0_move_policy : snapshot->slot1_move_policy;
    const float* sw = slot == 0 ? snapshot->slot0_switch_policy : snapshot->slot1_switch_policy;
    const float* tera = slot == 0 ? snapshot->slot0_tera_policy : snapshot->slot1_tera_policy;
    const float* target = slot == 0 ? snapshot->slot0_target_policy : snapshot->slot1_target_policy;
    int has_action = slot == 0 ? choice->slot0_has_action : choice->slot1_has_action;
    int action_kind = slot == 0 ? choice->slot0_kind : choice->slot1_kind;
    int move_index = slot == 0 ? choice->slot0_move_index : choice->slot1_move_index;
    int switch_index = slot == 0 ? choice->slot0_switch_index : choice->slot1_switch_index;
    int use_tera = slot == 0 ? choice->slot0_use_tera : choice->slot1_use_tera;
    int target_index = slot == 0 ? choice->slot0_target_index : choice->slot1_target_index;
    unsigned char target_bits = slot == 0 ? choice->slot0_target_mask : choice->slot1_target_mask;
    unsigned char kind_mask[FACTORIZED_KIND_DIM];
    unsigned char move_mask[FACTORIZED_MOVE_DIM];
    unsigned char switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char tera_mask[FACTORIZED_TERA_DIM];
    unsigned char target_mask[FACTORIZED_TARGET_DIM];
    int base = slot == 0 ? 0 : FACTORIZED_LOCAL_ACTION_DIM;
    int kind_index;
    int targets_correct = 1;

    if (!has_action) return 1;
    build_factor_masks(legal_mask, slot, kind_mask, move_mask, switch_mask);
    kind_index = action_kind == FACTORIZED_ACTION_MOVE ? 0 : 1;
    ++metrics->kind_labels;
    if (argmax_masked(kind, kind_mask, FACTORIZED_KIND_DIM) == kind_index) ++metrics->kind_hits;

    if (action_kind == FACTORIZED_ACTION_MOVE) {
        int predicted;
        ++metrics->move_labels;
        if (argmax_masked(move, move_mask, FACTORIZED_MOVE_DIM) == move_index) ++metrics->move_hits;

        tera_mask[0] = legal_mask[base + move_index] ? 1 : 0;
        tera_mask[1] = legal_mask[base + FACTORIZED_MOVE_DIM + move_index] ? 1 : 0;
        ++metrics->tera_labels;
        predicted = argmax_masked(tera, tera_mask, FACTORIZED_TERA_DIM);
        if (predicted == use_tera) ++metrics->tera_hits;

        if (target_bits != 0u) {
            size_t i;
            for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) target_mask[i] = (target_bits & (1u << i)) ? 1 : 0;
            ++metrics->target_labels;
            predicted = argmax_masked(target, target_mask, FACTORIZED_TARGET_DIM);
            if (predicted == target_index) ++metrics->target_hits;
            else targets_correct = 0;
            if (target_index < 0 || target_index >= FACTORIZED_TARGET_DIM || !target_mask[target_index] ||
                    predicted < 0 || !target_mask[predicted]) {
                ++metrics->illegal_predictions;
                targets_correct = 0;
            }
            metrics->target_nll_sum += negative_log_probability(
                masked_probability(target, target_mask, FACTORIZED_TARGET_DIM, (size_t)target_index));
        }
    } else {
        ++metrics->switch_labels;
        if (argmax_masked(sw, switch_mask, FACTORIZED_SWITCH_DIM) == switch_index) ++metrics->switch_hits;
    }
    return targets_correct;
}

void policy_evaluation_init(PolicyEvaluationMetrics* metrics) {
    if (metrics) memset(metrics, 0, sizeof(*metrics));
}

int policy_evaluation_add_episode(
    const GruModel* model,
    size_t bptt_window,
    const Episode* episode,
    PolicyEvaluationMetrics* metrics
) {
    float* hidden;
    float flat_policy[OBS_NUM_ACTIONS];
    size_t hidden_dim;
    size_t t;
    if (!model || !episode || !metrics) return 0;

    hidden_dim = gru_model_hidden_dim(model);
    hidden = (float*)malloc(hidden_dim * sizeof(float));
    if (!hidden) return 0;

    ++metrics->sessions;
    for (t = 0; t < episode->count; ++t) {
        const FactorizedActionChoice* choice = &episode->factorized_actions[t];
        const unsigned char* legal_mask = episode->legal_masks + t * OBS_NUM_ACTIONS;
        size_t start = t + 1 > bptt_window ? t + 1 - bptt_window : 0;
        size_t steps = t - start + 1;
        FactorizedPolicySnapshot snapshot;
        float value = 0.0f;
        float action_probability = 0.0f;
        double target_nll_before;
        int decision_correct = 0;
        int components_correct;
        int slot0_targets_correct;
        int slot1_targets_correct;

        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            ++metrics->skipped_turns;
            continue;
        }
        gru_model_zero_state(model, hidden);
        gru_model_forward_sequence(model, episode->observations + start * episode->obs_dim,
            steps, hidden, flat_policy, &value);
        if (!gru_model_evaluate_policy_snapshot(model, hidden, legal_mask,
                choice->slot0_has_action && choice->slot1_has_action, &snapshot, &value)) {
            free(hidden);
            return 0;
        }

        ++metrics->decision_turns;
        if (choice->slot0_has_action) {
            ++metrics->slot0_labels;
            ++metrics->action_labels;
        }
        if (choice->slot1_has_action) {
            ++metrics->slot1_labels;
            ++metrics->action_labels;
        }

        if (snapshot.has_joint_policy) {
            float slot0_marginal[FACTORIZED_LOCAL_ACTION_DIM];
            float slot1_marginal[FACTORIZED_LOCAL_ACTION_DIM];
            int local0 = episode->actions[t];
            int local1 = episode->actions2[t] - FACTORIZED_LOCAL_ACTION_DIM;
            size_t pair_index;
            int predicted_pair;
            if (local0 < 0 || local0 >= FACTORIZED_LOCAL_ACTION_DIM ||
                    local1 < 0 || local1 >= FACTORIZED_LOCAL_ACTION_DIM) {
                ++metrics->illegal_predictions;
                continue;
            }
            pair_index = (size_t)local0 * FACTORIZED_LOCAL_ACTION_DIM + (size_t)local1;
            predicted_pair = argmax_masked(snapshot.joint_policy, NULL, FACTORIZED_JOINT_DIM);
            build_joint_marginals(snapshot.joint_policy, slot0_marginal, slot1_marginal);
            if (argmax_masked(slot0_marginal, NULL, FACTORIZED_LOCAL_ACTION_DIM) == local0) ++metrics->slot0_hits;
            if (argmax_masked(slot1_marginal, NULL, FACTORIZED_LOCAL_ACTION_DIM) == local1) ++metrics->slot1_hits;
            ++metrics->joint_pair_labels;
            if (predicted_pair == (int)pair_index) {
                ++metrics->joint_pair_hits;
                decision_correct = 1;
            }
            if (is_top_k(snapshot.joint_policy, NULL, FACTORIZED_JOINT_DIM, pair_index, 3)) ++metrics->top3_hits;
            action_probability = snapshot.joint_policy[pair_index];
            if (predicted_pair < 0 || snapshot.joint_policy[predicted_pair] <= 0.0f) ++metrics->illegal_predictions;
        } else {
            float local_policy[FACTORIZED_LOCAL_ACTION_DIM];
            int slot = choice->slot0_has_action ? 0 : 1;
            int action = slot == 0 ? episode->actions[t] : episode->actions2[t] - FACTORIZED_LOCAL_ACTION_DIM;
            int predicted;
            if (action < 0 || action >= FACTORIZED_LOCAL_ACTION_DIM) {
                ++metrics->illegal_predictions;
                continue;
            }
            build_local_policy(&snapshot, legal_mask, slot, local_policy);
            predicted = argmax_masked(local_policy, NULL, FACTORIZED_LOCAL_ACTION_DIM);
            if (slot == 0 && predicted == action) ++metrics->slot0_hits;
            if (slot == 1 && predicted == action) ++metrics->slot1_hits;
            decision_correct = predicted == action;
            if (is_top_k(local_policy, NULL, FACTORIZED_LOCAL_ACTION_DIM, (size_t)action, 3)) ++metrics->top3_hits;
            action_probability = local_policy[action];
            if (predicted < 0 || local_policy[predicted] <= 0.0f) ++metrics->illegal_predictions;
        }

        target_nll_before = metrics->target_nll_sum;
        slot0_targets_correct = record_component_metrics(&snapshot, legal_mask, choice, 0, metrics);
        slot1_targets_correct = record_component_metrics(&snapshot, legal_mask, choice, 1, metrics);
        components_correct = slot0_targets_correct && slot1_targets_correct;
        metrics->action_nll_sum += negative_log_probability(action_probability);
        metrics->full_turn_nll_sum += negative_log_probability(action_probability) +
            (metrics->target_nll_sum - target_nll_before);
        if (decision_correct && components_correct) ++metrics->full_turn_hits;
        if (isfinite(value)) {
            double error = (double)value - (double)episode->rewards[t];
            metrics->value_loss_sum += 0.5 * error * error;
        } else {
            ++metrics->nonfinite_values;
        }
    }
    free(hidden);
    return 1;
}

void policy_evaluation_merge(PolicyEvaluationMetrics* total, const PolicyEvaluationMetrics* part) {
    if (!total || !part) return;
    total->sessions += part->sessions;
    total->decision_turns += part->decision_turns;
    total->skipped_turns += part->skipped_turns;
    total->action_labels += part->action_labels;
    total->slot0_labels += part->slot0_labels;
    total->slot1_labels += part->slot1_labels;
    total->joint_pair_labels += part->joint_pair_labels;
    total->kind_labels += part->kind_labels;
    total->move_labels += part->move_labels;
    total->switch_labels += part->switch_labels;
    total->tera_labels += part->tera_labels;
    total->target_labels += part->target_labels;
    total->full_turn_hits += part->full_turn_hits;
    total->top3_hits += part->top3_hits;
    total->slot0_hits += part->slot0_hits;
    total->slot1_hits += part->slot1_hits;
    total->joint_pair_hits += part->joint_pair_hits;
    total->kind_hits += part->kind_hits;
    total->move_hits += part->move_hits;
    total->switch_hits += part->switch_hits;
    total->tera_hits += part->tera_hits;
    total->target_hits += part->target_hits;
    total->illegal_predictions += part->illegal_predictions;
    total->nonfinite_values += part->nonfinite_values;
    total->action_nll_sum += part->action_nll_sum;
    total->target_nll_sum += part->target_nll_sum;
    total->full_turn_nll_sum += part->full_turn_nll_sum;
    total->value_loss_sum += part->value_loss_sum;
}

static double ratio(size_t numerator, size_t denominator) {
    return denominator > 0 ? (double)numerator / (double)denominator : 0.0;
}

double policy_evaluation_action_nll(const PolicyEvaluationMetrics* m) { return m && m->decision_turns ? m->action_nll_sum / m->decision_turns : 0.0; }
double policy_evaluation_target_nll(const PolicyEvaluationMetrics* m) { return m && m->target_labels ? m->target_nll_sum / m->target_labels : 0.0; }
double policy_evaluation_full_turn_nll(const PolicyEvaluationMetrics* m) { return m && m->decision_turns ? m->full_turn_nll_sum / m->decision_turns : 0.0; }
double policy_evaluation_value_loss(const PolicyEvaluationMetrics* m) { return m && m->decision_turns ? m->value_loss_sum / m->decision_turns : 0.0; }
double policy_evaluation_full_turn_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->full_turn_hits, m->decision_turns) : 0.0; }
double policy_evaluation_top3_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->top3_hits, m->decision_turns) : 0.0; }
double policy_evaluation_slot0_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->slot0_hits, m->slot0_labels) : 0.0; }
double policy_evaluation_slot1_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->slot1_hits, m->slot1_labels) : 0.0; }
double policy_evaluation_joint_pair_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->joint_pair_hits, m->joint_pair_labels) : 0.0; }
double policy_evaluation_kind_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->kind_hits, m->kind_labels) : 0.0; }
double policy_evaluation_move_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->move_hits, m->move_labels) : 0.0; }
double policy_evaluation_switch_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->switch_hits, m->switch_labels) : 0.0; }
double policy_evaluation_tera_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->tera_hits, m->tera_labels) : 0.0; }
double policy_evaluation_target_accuracy(const PolicyEvaluationMetrics* m) { return m ? ratio(m->target_hits, m->target_labels) : 0.0; }
