#include "gru_trainer.h"
#include "action_mapper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void build_step_slot_legal_mask(const Episode* episode, size_t step_index, int slot, uint8_t* out) {
    if (!episode || !out || step_index >= episode->count) {
        return;
    }
    build_slot_legal_mask(episode->legal_masks + (step_index * OBS_NUM_ACTIONS), slot, out);
}

static int train_prefix(
    GruTrainer* trainer,
    GruModel* model,
    const float* observations,
    size_t obs_dim,
    size_t steps,
    const unsigned char* legal_mask,
    int target_action,
    float target_value,
    float* action_loss,
    float* value_loss,
    float* accuracy
) {
    if (!gru_model_supervised_update_sequence(model, observations, steps, legal_mask, target_action, target_value,
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
    uint8_t slot_mask[OBS_NUM_ACTIONS];
    if (!trainer || !model || !episode) {
        return 0;
    }
    for (t = 0; t < episode->count; ++t) {
        size_t start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        size_t steps = (t - start) + 1;
        float action_loss = 0.0f;
        float value_loss = 0.0f;
        float accuracy = 0.0f;
        int slot;
        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            continue;
        }
        if (episode->actions[t] >= 0) {
            slot = obs_action_slot((enum ObsAction)episode->actions[t]);
            build_step_slot_legal_mask(episode, t, slot, slot_mask);
            if (!train_prefix(trainer, model,
                    episode->observations + (start * episode->obs_dim),
                    episode->obs_dim,
                    steps,
                    slot_mask,
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
        if (episode->actions2[t] >= 0) {
            action_loss = 0.0f;
            value_loss = 0.0f;
            accuracy = 0.0f;
            slot = obs_action_slot((enum ObsAction)episode->actions2[t]);
            build_step_slot_legal_mask(episode, t, slot, slot_mask);
            if (!train_prefix(trainer, model,
                    episode->observations + (start * episode->obs_dim),
                    episode->obs_dim,
                    steps,
                    slot_mask,
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
    float* next_hidden = NULL;
    float* hidden_states = NULL;
    float* policy_step = NULL;
    float* masked_policy = NULL;
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
    uint8_t slot_mask[OBS_NUM_ACTIONS];

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
    next_hidden = (float*)malloc(hidden_dim * sizeof(float));
    hidden_states = (float*)calloc(episode->count * hidden_dim, sizeof(float));
    policy_step = (float*)malloc(action_dim * sizeof(float));
    masked_policy = (float*)malloc(action_dim * sizeof(float));
    labeled_indices = (size_t*)calloc(episode->count, sizeof(size_t));
    if (!returns || !advantages || !values || !hidden || !next_hidden || !hidden_states || !policy_step || !masked_policy || !labeled_indices) {
        free(returns);
        free(advantages);
        free(values);
        free(hidden);
        free(next_hidden);
        free(hidden_states);
        free(policy_step);
        free(masked_policy);
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

    gru_model_zero_state(model, hidden);
    for (t = 0; t < episode->count; ++t) {
        float value = 0.0f;
        float* stored_hidden;
        gru_model_forward_step(
            model,
            episode->observations + (t * episode->obs_dim),
            hidden,
            next_hidden,
            policy_step,
            &value);
        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
            continue;
        }
        stored_hidden = hidden_states + (labeled_steps * hidden_dim);
        memcpy(stored_hidden, next_hidden, hidden_dim * sizeof(float));
        values[labeled_steps] = value;
        labeled_indices[labeled_steps] = t;
        advantages[labeled_steps] = returns[t] - values[labeled_steps];
        advantage_sum += advantages[labeled_steps];
        return_sum += returns[t];
        ++labeled_steps;
        memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
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
        free(next_hidden);
        free(hidden_states);
        free(policy_step);
        free(masked_policy);
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
        float prob_action = 0.0f;
        float prob_action2 = 0.0f;
        size_t a;
        if (episode->actions[episode_t] >= 0) {
            build_step_slot_legal_mask(episode, episode_t, 0, slot_mask);
            gru_model_evaluate_hidden(model, stored_hidden, slot_mask, masked_policy, &values[t]);
            prob_action = masked_policy[episode->actions[episode_t]] > 1.0e-8f ? masked_policy[episode->actions[episode_t]] : 1.0e-8f;
            policy_loss_sum += -logf(prob_action) * advantages[t];
            value_loss_sum += 0.5f * (values[t] - returns[episode_t]) * (values[t] - returns[episode_t]);
            {
                float slot_entropy = 0.0f;
                for (a = 0; a < action_dim; ++a) {
                    float p;
                    if (!slot_mask[a]) {
                        continue;
                    }
                    p = masked_policy[a] > 1.0e-8f ? masked_policy[a] : 1.0e-8f;
                    slot_entropy -= p * logf(p);
                }
                entropy_sum += slot_entropy;
            }
            advantage_sum += advantages[t];
            ++trained_labels;
            if (!gru_model_policy_gradient_update_heads(model,
                    stored_hidden,
                    slot_mask,
                    episode->actions[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate)) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(next_hidden);
                free(hidden_states);
                free(policy_step);
                free(masked_policy);
                free(labeled_indices);
                return 0;
            }
            ++trainer->step;
        }
        if (episode->actions2[episode_t] >= 0) {
            build_step_slot_legal_mask(episode, episode_t, 1, slot_mask);
            gru_model_evaluate_hidden(model, stored_hidden, slot_mask, masked_policy, &values[t]);
            prob_action2 = masked_policy[episode->actions2[episode_t]] > 1.0e-8f ? masked_policy[episode->actions2[episode_t]] : 1.0e-8f;
            policy_loss_sum += -logf(prob_action2) * advantages[t];
            value_loss_sum += 0.5f * (values[t] - returns[episode_t]) * (values[t] - returns[episode_t]);
            {
                float slot_entropy = 0.0f;
                for (a = 0; a < action_dim; ++a) {
                    float p;
                    if (!slot_mask[a]) {
                        continue;
                    }
                    p = masked_policy[a] > 1.0e-8f ? masked_policy[a] : 1.0e-8f;
                    slot_entropy -= p * logf(p);
                }
                entropy_sum += slot_entropy;
            }
            advantage_sum += advantages[t];
            ++trained_labels;
            if (!gru_model_policy_gradient_update_heads(model,
                    stored_hidden,
                    slot_mask,
                    episode->actions2[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate)) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(next_hidden);
                free(hidden_states);
                free(policy_step);
                free(masked_policy);
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
    free(next_hidden);
    free(hidden_states);
    free(policy_step);
    free(masked_policy);
    free(labeled_indices);
    return 1;
}
