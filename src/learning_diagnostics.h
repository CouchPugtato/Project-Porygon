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

typedef struct {
    size_t episode_count;
    size_t sample_count;
    size_t nonfinite_count;
    double value_loss;
    double mean_return;
    double mean_value;
    double value_bias;
    double explained_variance;
    double return_value_correlation;
} CriticFitMetrics;

typedef struct {
    CriticFitMetrics overall;
    CriticFitMetrics wins;
    CriticFitMetrics losses;
} CriticFitEvaluation;

typedef struct {
    CriticFitEvaluation before_train;
    CriticFitEvaluation before_selection;
    CriticFitEvaluation before_holdout;
    CriticFitEvaluation head_after_train;
    CriticFitEvaluation head_after_selection;
    CriticFitEvaluation head_after_holdout;
    CriticFitEvaluation recurrent_after_train;
    CriticFitEvaluation recurrent_after_selection;
    CriticFitEvaluation recurrent_after_holdout;
    double head_policy_probability_delta;
    double recurrent_policy_probability_delta;
    int head_policy_unchanged;
    int head_training_completed;
    int recurrent_training_completed;
    size_t head_epochs_completed;
    size_t head_best_epoch;
    size_t recurrent_epochs_completed;
    size_t recurrent_best_epoch;
    int head_stopped_early;
    int recurrent_stopped_early;
    int head_generalizes;
    int recurrent_aggregate_generalizes;
    int recurrent_policy_drift_acceptable;
    int recurrent_outcome_consistent;
    int recurrent_overfit;
    double recurrent_generalization_gap;
    int recurrent_generalizes;
    int critic_learnable;
} CriticFitResult;

#define PPO_UPDATE_AUDIT_BIN_COUNT 5

typedef struct {
    size_t sample_count;
    double mean_raw_advantage;
    double mean_standardized_advantage;
    double mean_before_probability;
    double mean_after_probability;
    double mean_probability_delta;
    double mean_log_probability_delta;
    double probability_increased_fraction;
    double mean_legal_policy_kl;
} PpoUpdateAuditBin;

typedef struct {
    size_t episode_count;
    size_t sample_count;
    size_t nonfinite_count;
    double raw_advantage_mean;
    double raw_advantage_standard_deviation;
    double advantage_log_probability_delta_correlation;
    double behavior_log_probability_mean_absolute_error;
    double behavior_value_mean_absolute_error;
    double mean_legal_policy_kl;
    double max_legal_policy_kl;
    CriticFitMetrics before_value;
    CriticFitMetrics after_value;
    PpoUpdateAuditBin bins[PPO_UPDATE_AUDIT_BIN_COUNT];
    int behavior_policy_matches;
    int value_loss_decreased;
    int actor_direction_consistent;
} PpoUpdateAuditResult;

void learning_diagnostic_assess_critic_fit(CriticFitResult* result);

int learning_diagnostic_publish_critic_checkpoint(
    const char* output_path,
    const GruModel* model,
    const GruTrainer* trainer,
    const CriticFitResult* result
);

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

int learning_diagnostic_run_critic_fit(
    GruTrainer* head_trainer,
    GruModel* head_model,
    GruTrainer* recurrent_trainer,
    GruModel* recurrent_model,
    const Episode* const* train_episodes,
    size_t train_count,
    const Episode* const* selection_episodes,
    size_t selection_count,
    const Episode* const* holdout_episodes,
    size_t holdout_count,
    size_t epochs,
    size_t minibatch_episodes,
    size_t early_stop_patience,
    unsigned int shuffle_seed,
    CriticFitResult* result
);

int learning_diagnostic_write_critic_report(
    const char* report_path,
    const char* source_path,
    const char* selection_source_path,
    const char* holdout_source_path,
    const char* checkpoint_path,
    const char* output_checkpoint_path,
    int publication_requested,
    int checkpoint_published,
    unsigned int validation_seed,
    unsigned int shuffle_seed,
    size_t epochs,
    size_t minibatch_episodes,
    size_t early_stop_patience,
    const GruTrainer* head_trainer,
    const GruTrainer* recurrent_trainer,
    const CriticFitResult* result
);

int learning_diagnostic_run_ppo_update_audit(
    const GruTrainer* trainer,
    const GruModel* before_model,
    const GruModel* after_model,
    const Episode* const* episodes,
    size_t episode_count,
    PpoUpdateAuditResult* result
);

int learning_diagnostic_write_ppo_update_report(
    const char* report_path,
    const char* episode_batch_path,
    const char* before_checkpoint_path,
    const char* after_checkpoint_path,
    float gamma,
    float gae_lambda,
    size_t episode_limit,
    unsigned int selection_seed,
    const PpoUpdateAuditResult* result
);

#endif
