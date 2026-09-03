#ifndef LEARNING_DIAGNOSTICS_H
#define LEARNING_DIAGNOSTICS_H

#include <stddef.h>

#include "episode.h"
#include "gru_model.h"
#include "gru_trainer.h"
#include "policy_evaluation.h"

typedef struct {
    PolicyEvaluationMetrics before;
    PolicyEvaluationMetrics after;
    double action_loss_reduction;
    int training_completed;
    int action_loss_reduced;
    int full_turn_accuracy_reached;
    int action_probability_increased;
    int target_probability_increased;
    int value_error_decreased;
    int outputs_finite;
    int predictions_legal;
    int passed;
} SupervisedOverfitResult;

int learning_diagnostic_run_supervised_overfit(
    GruTrainer* trainer,
    GruModel* model,
    const Episode* const* episodes,
    size_t episode_count,
    size_t epochs,
    SupervisedOverfitResult* result
);

int learning_diagnostic_write_supervised_report(
    const char* report_path,
    const char* source_path,
    const char* first_battle_id,
    const char* second_battle_id,
    unsigned int seed,
    size_t epochs,
    const GruTrainer* trainer,
    const SupervisedOverfitResult* result
);

#endif
