#ifndef POLICY_EVALUATION_H
#define POLICY_EVALUATION_H

#include <stddef.h>

#include "episode.h"
#include "gru_model.h"

#define POLICY_EVALUATION_METRICS_VERSION 2

typedef struct {
    size_t sessions;
    size_t decision_turns;
    size_t skipped_turns;
    size_t action_labels;
    size_t slot0_labels;
    size_t slot1_labels;
    size_t joint_pair_labels;
    size_t kind_labels;
    size_t move_labels;
    size_t switch_labels;
    size_t tera_labels;
    size_t target_labels;

    size_t full_turn_hits;
    size_t top3_hits;
    size_t slot0_hits;
    size_t slot1_hits;
    size_t joint_pair_hits;
    size_t kind_hits;
    size_t move_hits;
    size_t switch_hits;
    size_t tera_hits;
    size_t target_hits;
    size_t illegal_predictions;
    size_t nonfinite_values;

    double action_nll_sum;
    double target_nll_sum;
    double full_turn_nll_sum;
    double value_loss_sum;
} PolicyEvaluationMetrics;

void policy_evaluation_init(PolicyEvaluationMetrics* metrics);
int policy_evaluation_add_episode(
    const GruModel* model,
    size_t bptt_window,
    const Episode* episode,
    PolicyEvaluationMetrics* metrics
);
void policy_evaluation_merge(PolicyEvaluationMetrics* total, const PolicyEvaluationMetrics* part);

double policy_evaluation_action_nll(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_target_nll(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_full_turn_nll(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_value_loss(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_full_turn_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_top3_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_slot0_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_slot1_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_joint_pair_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_kind_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_move_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_switch_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_tera_accuracy(const PolicyEvaluationMetrics* metrics);
double policy_evaluation_target_accuracy(const PolicyEvaluationMetrics* metrics);

#endif
