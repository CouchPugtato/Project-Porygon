#ifndef GRU_TRAINER_H
#define GRU_TRAINER_H

#include "checkpoint.h"
#include "episode.h"
#include "gru_model.h"

typedef enum {
    GRU_SUPERVISED_OPTIMIZER_SGD = 0,
    GRU_SUPERVISED_OPTIMIZER_ADAM = 1
} GruSupervisedOptimizer;

typedef struct {
    size_t step;
    float learning_rate;
    size_t bptt_window;
    float gradient_clip;
    unsigned int seed;
    float last_action_loss;
    float last_value_loss;
    float last_accuracy;
    float gamma;
    float entropy_coef;
    int advantage_norm;
    float last_policy_loss;
    float last_mean_return;
    float last_mean_advantage;
    float last_mean_abs_advantage;
    float last_mean_value;
    double last_return_sum;
    double last_return_square_sum;
    double last_value_sum;
    double last_value_square_sum;
    double last_value_error_sum;
    double last_value_error_square_sum;
    double last_return_value_product_sum;
    size_t last_critic_samples;
    float last_entropy;
    float last_anchor_kl_mean;
    float last_anchor_kl_max;
    float last_anchor_loss;
    float last_approx_kl;
    float last_clip_fraction;
    size_t last_rl_labels;
    size_t supervised_minibatch_size;
    GruSupervisedOptimizer supervised_optimizer;
    int supervised_profile_enabled;
    double last_supervised_cache_seconds;
    double last_supervised_update_seconds;
    size_t last_supervised_label_count;
    size_t last_supervised_window_count;
    size_t last_supervised_batch_flushes;
    const GruModel* anchor_model;
    float anchor_kl_coef;
    float ppo_clip_epsilon;
    float ppo_value_clip_epsilon;
    float target_kl;
    float gae_lambda;
    float adam_beta1;
    float adam_beta2;
    float adam_epsilon;
} GruTrainer;

void gru_trainer_init(GruTrainer* trainer, float learning_rate, size_t bptt_window, float gradient_clip, unsigned int seed);
TrainerCheckpointState gru_trainer_checkpoint_state(const GruTrainer* trainer);
int gru_trainer_supervised_episode(GruTrainer* trainer, GruModel* model, const Episode* episode);
int gru_trainer_policy_gradient_episode(GruTrainer* trainer, GruModel* model, const Episode* episode);
int gru_trainer_ppo_episode(GruTrainer* trainer, GruModel* model, const Episode* episode);
int gru_trainer_ppo_minibatch(
    GruTrainer* trainer,
    GruModel* model,
    const Episode* const* episodes,
    size_t episode_count);
int gru_trainer_ppo_hard_kl_stop_update(
    float approx_kl,
    float target_kl,
    float hard_multiplier,
    int required_consecutive_updates,
    int* consecutive_breaches);
double gru_trainer_critic_explained_variance(const GruTrainer* trainer);
double gru_trainer_return_value_correlation(const GruTrainer* trainer);
const char* gru_supervised_optimizer_name(GruSupervisedOptimizer optimizer);

#endif
