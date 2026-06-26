#include "gru_trainer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int train_prefix(
    GruTrainer* trainer,
    GruModel* model,
    const float* observations,
    size_t obs_dim,
    size_t steps,
    int target_action,
    float target_value,
    float* action_loss,
    float* value_loss,
    float* accuracy
) {
    if (!gru_model_supervised_update_sequence(model, observations, steps, target_action, target_value,
        trainer->learning_rate, action_loss, value_loss, accuracy)) {
        return 0;
    }
    (void)obs_dim;
    return 1;
}

void gru_trainer_init(GruTrainer* trainer, float learning_rate, size_t bptt_window, float gradient_clip, unsigned int seed) {
    if (!trainer) {
        return;
    }
    memset(trainer, 0, sizeof(*trainer));
    trainer->learning_rate = learning_rate;
    trainer->bptt_window = bptt_window ? bptt_window : 16;
    trainer->gradient_clip = gradient_clip;
    trainer->seed = seed;
    trainer->gamma = 1.0f;
    trainer->entropy_coef = 0.001f;
    trainer->advantage_norm = 1;
}

TrainerCheckpointState gru_trainer_checkpoint_state(const GruTrainer* trainer) {
    TrainerCheckpointState state;
    memset(&state, 0, sizeof(state));
    if (trainer) {
        state.step = trainer->step;
        state.learning_rate = trainer->learning_rate;
        state.bptt_window = trainer->bptt_window;
        state.gradient_clip = trainer->gradient_clip;
        state.seed = trainer->seed;
    }
    return state;
}

int gru_trainer_supervised_episode(GruTrainer* trainer, GruModel* model, const Episode* episode) {
    size_t t;
    float action_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float accuracy_sum = 0.0f;
    size_t trained = 0;
    if (!trainer || !model || !episode) {
        return 0;
    }
    for (t = 0; t < episode->count; ++t) {
        size_t start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        size_t steps = (t - start) + 1;
        float action_loss = 0.0f;
        float value_loss = 0.0f;
        float accuracy = 0.0f;
        if (episode->actions[t] < 0) {
            continue;
        }
        if (!train_prefix(trainer, model,
                episode->observations + (start * episode->obs_dim),
                episode->obs_dim,
                steps,
                episode->actions[t],
                episode->rewards[t],
                &action_loss,
                &value_loss,
                &accuracy)) {
            return 0;
        }
        action_loss_sum += action_loss;
        value_loss_sum += value_loss;
        accuracy_sum += accuracy;
        ++trained;
        ++trainer->step;
        if (episode->actions2[t] >= 0) {
            action_loss = 0.0f;
            value_loss = 0.0f;
            accuracy = 0.0f;
            if (!train_prefix(trainer, model,
                    episode->observations + (start * episode->obs_dim),
                    episode->obs_dim,
                    steps,
                    episode->actions2[t],
                    episode->rewards[t],
                    &action_loss,
                    &value_loss,
                    &accuracy)) {
                return 0;
            }
            action_loss_sum += action_loss;
            value_loss_sum += value_loss;
            accuracy_sum += accuracy;
            ++trained;
            ++trainer->step;
        }
    }
    if (trained > 0) {
        trainer->last_action_loss = action_loss_sum / (float)trained;
        trainer->last_value_loss = value_loss_sum / (float)trained;
        trainer->last_accuracy = accuracy_sum / (float)trained;
    }
    return 1;
}

int gru_trainer_policy_gradient_episode(GruTrainer* trainer, GruModel* model, const Episode* episode) {
    size_t t;
    size_t labeled_steps = 0;
    size_t trained_labels = 0;
    float* returns = NULL;
    float* advantages = NULL;
    float* values = NULL;
    float* hidden = NULL;
    float* hidden_states = NULL;
    float* policies = NULL;
    size_t* labeled_indices = NULL;
    float return_sum = 0.0f;
    float advantage_sum = 0.0f;
    float policy_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float entropy_sum = 0.0f;
    float advantage_mean = 0.0f;
    float advantage_std = 0.0f;
    size_t action_dim;
    size_t hidden_dim;

    if (!trainer || !model || !episode) {
        return 0;
    }
    if (episode->count == 0) {
        trainer->last_policy_loss = 0.0f;
        trainer->last_value_loss = 0.0f;
        trainer->last_mean_return = 0.0f;
        trainer->last_mean_advantage = 0.0f;
        trainer->last_entropy = 0.0f;
        trainer->last_rl_labels = 0;
        return 1;
    }

    returns = (float*)calloc(episode->count, sizeof(float));
    advantages = (float*)calloc(episode->count, sizeof(float));
    values = (float*)calloc(episode->count, sizeof(float));
    action_dim = gru_model_num_actions(model);
    hidden_dim = gru_model_hidden_dim(model);
    hidden = (float*)malloc(hidden_dim * sizeof(float));
    hidden_states = (float*)calloc(episode->count * hidden_dim, sizeof(float));
    policies = (float*)calloc(episode->count * action_dim, sizeof(float));
    labeled_indices = (size_t*)calloc(episode->count, sizeof(size_t));
    if (!returns || !advantages || !values || !hidden || !hidden_states || !policies || !labeled_indices) {
        free(returns);
        free(advantages);
        free(values);
        free(hidden);
        free(hidden_states);
        free(policies);
        free(labeled_indices);
        return 0;
    }

    {
        float running_return = 0.0f;
        for (t = episode->count; t-- > 0;) {
            running_return = episode->rewards[t] + (trainer->gamma * running_return);
            returns[t] = running_return;
        }
    }

    for (t = 0; t < episode->count; ++t) {
        size_t start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        size_t steps = (t - start) + 1;
        float* stored_hidden;
        float* stored_policy;
        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            continue;
        }
        stored_hidden = hidden_states + (labeled_steps * hidden_dim);
        stored_policy = policies + (labeled_steps * action_dim);
        gru_model_zero_state(model, hidden);
        gru_model_forward_sequence(
            model,
            episode->observations + (start * episode->obs_dim),
            steps,
            hidden,
            stored_policy,
            &values[labeled_steps]);
        memcpy(stored_hidden, hidden, hidden_dim * sizeof(float));
        labeled_indices[labeled_steps] = t;
        advantages[labeled_steps] = returns[t] - values[labeled_steps];
        advantage_sum += advantages[labeled_steps];
        return_sum += returns[t];
        ++labeled_steps;
    }

    if (labeled_steps == 0) {
        trainer->last_policy_loss = 0.0f;
        trainer->last_value_loss = 0.0f;
        trainer->last_mean_return = 0.0f;
        trainer->last_mean_advantage = 0.0f;
        trainer->last_entropy = 0.0f;
        trainer->last_rl_labels = 0;
        free(returns);
        free(advantages);
        free(values);
        free(hidden);
        free(hidden_states);
        free(policies);
        free(labeled_indices);
        return 1;
    }

    advantage_mean = advantage_sum / (float)labeled_steps;
    if (trainer->advantage_norm && labeled_steps > 1) {
        float variance = 0.0f;
        for (t = 0; t < labeled_steps; ++t) {
            float centered = advantages[t] - advantage_mean;
            variance += centered * centered;
        }
        variance /= (float)labeled_steps;
        advantage_std = sqrtf(variance);
        if (advantage_std > 1.0e-6f) {
            for (t = 0; t < labeled_steps; ++t) {
                advantages[t] = (advantages[t] - advantage_mean) / advantage_std;
            }
        } else {
            for (t = 0; t < labeled_steps; ++t) {
                advantages[t] = 0.0f;
            }
        }
    }

    advantage_sum = 0.0f;
    for (t = 0; t < labeled_steps; ++t) {
        size_t episode_t = labeled_indices[t];
        const float* stored_hidden = hidden_states + (t * hidden_dim);
        const float* stored_policy = policies + (t * action_dim);
        float entropy = 0.0f;
        float prob_action = 0.0f;
        float prob_action2 = 0.0f;
        size_t a;

        for (a = 0; a < action_dim; ++a) {
            float p = stored_policy[a] > 1.0e-8f ? stored_policy[a] : 1.0e-8f;
            entropy -= p * logf(p);
        }
        if (episode->actions[episode_t] >= 0) {
            prob_action = stored_policy[episode->actions[episode_t]] > 1.0e-8f ? stored_policy[episode->actions[episode_t]] : 1.0e-8f;
            policy_loss_sum += -logf(prob_action) * advantages[t];
            value_loss_sum += 0.5f * (values[t] - returns[episode_t]) * (values[t] - returns[episode_t]);
            entropy_sum += entropy;
            advantage_sum += advantages[t];
            ++trained_labels;
            if (!gru_model_policy_gradient_update_heads(model,
                    stored_hidden,
                    NULL,
                    episode->actions[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate)) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(hidden_states);
                free(policies);
                free(labeled_indices);
                return 0;
            }
            ++trainer->step;
        }
        if (episode->actions2[episode_t] >= 0) {
            prob_action2 = stored_policy[episode->actions2[episode_t]] > 1.0e-8f ? stored_policy[episode->actions2[episode_t]] : 1.0e-8f;
            policy_loss_sum += -logf(prob_action2) * advantages[t];
            value_loss_sum += 0.5f * (values[t] - returns[episode_t]) * (values[t] - returns[episode_t]);
            entropy_sum += entropy;
            advantage_sum += advantages[t];
            ++trained_labels;
            if (!gru_model_policy_gradient_update_heads(model,
                    stored_hidden,
                    NULL,
                    episode->actions2[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate)) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(hidden_states);
                free(policies);
                free(labeled_indices);
                return 0;
            }
            ++trainer->step;
        }
    }
    trainer->last_policy_loss = trained_labels > 0 ? (policy_loss_sum / (float)trained_labels) : 0.0f;
    trainer->last_value_loss = trained_labels > 0 ? (value_loss_sum / (float)trained_labels) : 0.0f;
    trainer->last_mean_return = labeled_steps > 0 ? (return_sum / (float)labeled_steps) : 0.0f;
    trainer->last_mean_advantage = trained_labels > 0 ? (advantage_sum / (float)trained_labels) : 0.0f;
    trainer->last_entropy = trained_labels > 0 ? (entropy_sum / (float)trained_labels) : 0.0f;
    trainer->last_rl_labels = trained_labels;
    trainer->last_action_loss = trainer->last_policy_loss;
    trainer->last_accuracy = 0.0f;
    free(returns);
    free(advantages);
    free(values);
    free(hidden);
    free(hidden_states);
    free(policies);
    free(labeled_indices);
    return 1;
}
