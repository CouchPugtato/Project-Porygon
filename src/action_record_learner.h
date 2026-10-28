#ifndef ACTION_RECORD_LEARNER_H
#define ACTION_RECORD_LEARNER_H
#include "gru_model.h"

typedef struct {
    const float* hidden;
    unsigned char legal[28];
    FactorizedActionChoice choice;
    float terminal_return;
    double behavior_log_probability;
    double conditional_target_log_probability;
} ActionLearningRecord;

typedef struct {
    size_t minibatch, max_updates, invocation_updates;
    float learning_rate, max_abs_log_ratio;
    enum { ACTION_RETURN_DIAGNOSTIC = 0, ACTION_PPO_FROZEN_VALUE = 1 } objective;
    float ppo_clip_epsilon;
} ActionLearningOptions;

typedef struct {
    double ratio, surrogate, log_probability_gradient;
    int clipped;
} ActionPpoSurrogate;

/* Unfloored double-precision full-turn likelihoods. Gradient is with respect
 * to current log probability, holding the advantage and behavior fixed. */
int action_record_ppo_surrogate(double current_log, double behavior_log, double advantage,
    double epsilon, ActionPpoSurrogate* result);

typedef struct {
    size_t cursor, updates;
    int guard_stopped;
    double max_abs_log_ratio;
    double rejected_abs_log_ratio;
    double ratio_min, ratio_max, mean_advantage, mean_value, surrogate_mean, clipped_fraction;
} ActionLearningProgress;

/* Frozen hidden/target conditionals must have passed complete bundle replay.
 * PPO mode uses undiscounted terminal returns minus frozen behavior values.
 * Neither mode learns values, shuffles or repeats dataset epochs. */
int action_record_learn(const GruModel* reference, const GruModel* behavior, GruModel* actor,
    GruActionHeadOptimizer** optimizer, const ActionLearningRecord* records, size_t count,
    const ActionLearningOptions* options, ActionLearningProgress* progress);
int action_record_learner_cli(int argc, char** argv);
#endif
