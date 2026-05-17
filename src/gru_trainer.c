#include "gru_trainer.h"

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
    float* hidden;
    float* policy;
    float value;
    hidden = (float*)malloc(gru_model_hidden_dim(model) * sizeof(float));
    policy = (float*)malloc(gru_model_num_actions(model) * sizeof(float));
    if (!hidden || !policy) {
        free(hidden);
        free(policy);
        return 0;
    }
    gru_model_zero_state(model, hidden);
    gru_model_forward_sequence(model, observations, steps, hidden, policy, &value);
    gru_model_supervised_update_heads(model, hidden, NULL, target_action, target_value,
        trainer->learning_rate, action_loss, value_loss, accuracy);
    free(hidden);
    free(policy);
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
    float running_return = 0.0f;
    if (!trainer || !model || !episode) {
        return 0;
    }
    for (t = episode->count; t-- > 0;) {
        float* hidden;
        float* policy;
        float value;
        size_t start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        size_t steps = (t - start) + 1;
        if (episode->actions[t] < 0) {
            continue;
        }
        running_return += episode->rewards[t];
        hidden = (float*)malloc(gru_model_hidden_dim(model) * sizeof(float));
        policy = (float*)malloc(gru_model_num_actions(model) * sizeof(float));
        if (!hidden || !policy) {
            free(hidden);
            free(policy);
            return 0;
        }
        gru_model_zero_state(model, hidden);
        gru_model_forward_sequence(model, episode->observations + (start * episode->obs_dim), steps, hidden, policy, &value);
        gru_model_policy_gradient_update_heads(model, hidden, NULL, episode->actions[t], running_return - value, running_return, 0.0f, trainer->learning_rate);
        free(hidden);
        free(policy);
        ++trainer->step;
    }
    return 1;
}
