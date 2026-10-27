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
} ActionLearningOptions;

typedef struct {
    size_t cursor, updates;
    int guard_stopped;
    double max_abs_log_ratio;
    double rejected_abs_log_ratio;
} ActionLearningProgress;

/* Diagnostic terminal-return score-function updates only. Frozen hidden/target
 * conditionals must already have passed complete behavior-bundle replay. No
 * PPO, value learning, shuffling or repeated dataset epochs are implemented. */
int action_record_learn(const GruModel* reference, const GruModel* behavior, GruModel* actor,
    GruActionHeadOptimizer** optimizer, const ActionLearningRecord* records, size_t count,
    const ActionLearningOptions* options, ActionLearningProgress* progress);
int action_record_learner_cli(int argc, char** argv);
#endif
