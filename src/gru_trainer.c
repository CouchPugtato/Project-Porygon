#include "gru_trainer.h"
#include "action_mapper.h"

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

static void build_step_slot_legal_mask(const Episode* episode, size_t step_index, int slot, uint8_t* out) {
    if (!episode || !out || step_index >= episode->count) {
        return;
    }
    build_slot_legal_mask(episode->legal_masks + (step_index * OBS_NUM_ACTIONS), slot, out);
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
    trainer->anchor_kl_coef = 0.0f;
    trainer->ppo_clip_epsilon = 0.2f;
    trainer->ppo_value_clip_epsilon = 0.2f;
    trainer->target_kl = 0.02f;
    trainer->gae_lambda = 0.95f;
    trainer->adam_beta1 = 0.9f;
    trainer->adam_beta2 = 0.999f;
    trainer->adam_epsilon = 1.0e-8f;
    trainer->supervised_minibatch_size = 8;
    trainer->supervised_optimizer = GRU_SUPERVISED_OPTIMIZER_SGD;
    trainer->supervised_profile_enabled = 1;
}

const char* gru_supervised_optimizer_name(GruSupervisedOptimizer optimizer) {
    return optimizer == GRU_SUPERVISED_OPTIMIZER_ADAM ? "adam" : "sgd";
}

static int apply_supervised_updates(GruTrainer* trainer, GruModel* model) {
    if (trainer->supervised_optimizer == GRU_SUPERVISED_OPTIMIZER_ADAM) {
        return gru_model_apply_accumulated_adam_updates(
            model,
            trainer->learning_rate,
            trainer->adam_beta1,
            trainer->adam_beta2,
            trainer->adam_epsilon,
            trainer->gradient_clip);
    }
    return gru_model_apply_accumulated_supervised_updates(model, trainer->learning_rate);
}

static float masked_action_log_prob(const float* policy, int action) {
    float p;
    if (!policy || action < 0) {
        return 0.0f;
    }
    p = policy[action];
    if (p < 1.0e-8f) {
        p = 1.0e-8f;
    }
    return logf(p);
}

static float masked_small_log_prob(const float* policy, const unsigned char* mask, size_t dim, int index) {
    float sum = 0.0f;
    float p;
    size_t i;
    if (!policy || !mask || index < 0 || (size_t)index >= dim || !mask[index]) {
        return 0.0f;
    }
    for (i = 0; i < dim; ++i) {
        if (mask[i]) {
            sum += policy[i];
        }
    }
    if (sum <= 1.0e-8f) {
        return 0.0f;
    }
    p = policy[index] / sum;
    if (p < 1.0e-8f) {
        p = 1.0e-8f;
    }
    return logf(p);
}

static float masked_small_entropy(const float* policy, const unsigned char* mask, size_t dim) {
    float sum = 0.0f;
    float entropy = 0.0f;
    size_t i;
    for (i = 0; i < dim; ++i) {
        if (mask[i]) {
            sum += policy[i];
        }
    }
    if (sum <= 1.0e-8f) {
        return 0.0f;
    }
    for (i = 0; i < dim; ++i) {
        float p;
        if (!mask[i]) {
            continue;
        }
        p = policy[i] / sum;
        if (p < 1.0e-8f) {
            p = 1.0e-8f;
        }
        entropy -= p * logf(p);
    }
    return entropy;
}

static float masked_policy_kl(
    const float* anchor_policy,
    const float* current_policy,
    const unsigned char* mask,
    size_t dim
) {
    float anchor_sum = 0.0f;
    float current_sum = 0.0f;
    float kl = 0.0f;
    size_t i;
    if (!anchor_policy || !current_policy) {
        return 0.0f;
    }
    for (i = 0; i < dim; ++i) {
        if (!mask || mask[i]) {
            anchor_sum += anchor_policy[i];
            current_sum += current_policy[i];
        }
    }
    if (anchor_sum <= 1.0e-8f || current_sum <= 1.0e-8f) {
        return 0.0f;
    }
    for (i = 0; i < dim; ++i) {
        float q;
        float p;
        if (mask && !mask[i]) {
            continue;
        }
        q = anchor_policy[i] / anchor_sum;
        if (q <= 0.0f) {
            continue;
        }
        p = current_policy[i] / current_sum;
        if (p < 1.0e-8f) {
            p = 1.0e-8f;
        }
        kl += q * (logf(q > 1.0e-8f ? q : 1.0e-8f) - logf(p));
    }
    return kl > 0.0f ? kl : 0.0f;
}

static void build_factor_masks_from_episode(const Episode* episode, size_t step_index, int slot, unsigned char* kind_mask, unsigned char* move_mask, unsigned char* switch_mask) {
    const uint8_t* legal_mask = episode->legal_masks + (step_index * OBS_NUM_ACTIONS);
    int base = slot == 0 ? 0 : 14;
    int i;
    memset(kind_mask, 0, FACTORIZED_KIND_DIM);
    memset(move_mask, 0, FACTORIZED_MOVE_DIM);
    memset(switch_mask, 0, FACTORIZED_SWITCH_DIM);
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        if (legal_mask[base + i] || legal_mask[base + 4 + i]) {
            move_mask[i] = 1;
            kind_mask[0] = 1;
        }
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        if (legal_mask[base + 8 + i]) {
            switch_mask[i] = 1;
            kind_mask[1] = 1;
        }
    }
}

static int evaluate_joint_step(
    const GruModel* model,
    const float* hidden_state,
    const Episode* episode,
    size_t step_index,
    float* joint_log_prob_out,
    float* value_out,
    float* entropy_out,
    FactorizedPolicySnapshot* snapshot_out
) {
    unsigned char kind_mask[FACTORIZED_KIND_DIM];
    unsigned char move_mask[FACTORIZED_MOVE_DIM];
    unsigned char switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char tera_mask[FACTORIZED_TERA_DIM];
    FactorizedPolicySnapshot snapshot;
    float* slot0_kind_policy = snapshot.slot0_kind_policy;
    float* slot0_move_policy = snapshot.slot0_move_policy;
    float* slot0_switch_policy = snapshot.slot0_switch_policy;
    float* slot0_tera_policy = snapshot.slot0_tera_policy;
    float* slot0_target_policy = snapshot.slot0_target_policy;
    float* slot1_kind_policy = snapshot.slot1_kind_policy;
    float* slot1_move_policy = snapshot.slot1_move_policy;
    float* slot1_switch_policy = snapshot.slot1_switch_policy;
    float* slot1_tera_policy = snapshot.slot1_tera_policy;
    float* slot1_target_policy = snapshot.slot1_target_policy;
    float* joint_policy = snapshot.joint_policy;
    unsigned char target_mask[FACTORIZED_TARGET_DIM];
    float joint_log_prob = 0.0f;
    float entropy = 0.0f;
    float value = 0.0f;
    const FactorizedActionChoice* choice;
    if (!model || !hidden_state || !episode || step_index >= episode->count) {
        return 0;
    }
    choice = &episode->factorized_actions[step_index];
    if (!gru_model_evaluate_policy_snapshot(
            model,
            hidden_state,
            episode->legal_masks + (step_index * OBS_NUM_ACTIONS),
            choice->slot0_has_action && choice->slot1_has_action,
            &snapshot,
            &value)) {
        return 0;
    }
    if (snapshot_out) *snapshot_out = snapshot;
    if (choice->slot0_has_action && choice->slot1_has_action) {
        int flat0 = -1;
        int flat1 = -1;
        int selected;
        int i;
        if (!factorized_action_choice_to_flat_actions(choice, &flat0, &flat1) || flat0 < 0 || flat1 < 14) {
            return 0;
        }
        selected = flat0 * FACTORIZED_LOCAL_ACTION_DIM + (flat1 - 14);
        joint_log_prob += masked_action_log_prob(joint_policy, selected);
        for (i = 0; i < FACTORIZED_JOINT_DIM; ++i) {
            if (joint_policy[i] > 0.0f) {
                entropy -= joint_policy[i] * logf(joint_policy[i] > 1.0e-8f ? joint_policy[i] : 1.0e-8f);
            }
        }
        if (choice->slot0_kind == FACTORIZED_ACTION_MOVE && choice->slot0_target_mask != 0u) {
            factorized_target_mask_to_array(choice->slot0_target_mask, target_mask);
            joint_log_prob += masked_small_log_prob(slot0_target_policy, target_mask, FACTORIZED_TARGET_DIM, choice->slot0_target_index);
            entropy += masked_small_entropy(slot0_target_policy, target_mask, FACTORIZED_TARGET_DIM);
        }
        if (choice->slot1_kind == FACTORIZED_ACTION_MOVE && choice->slot1_target_mask != 0u) {
            factorized_target_mask_to_array(choice->slot1_target_mask, target_mask);
            joint_log_prob += masked_small_log_prob(slot1_target_policy, target_mask, FACTORIZED_TARGET_DIM, choice->slot1_target_index);
            entropy += masked_small_entropy(slot1_target_policy, target_mask, FACTORIZED_TARGET_DIM);
        }
        if (joint_log_prob_out) *joint_log_prob_out = joint_log_prob;
        if (value_out) *value_out = value;
        if (entropy_out) *entropy_out = entropy;
        return 1;
    }
    if (choice->slot0_has_action) {
        build_factor_masks_from_episode(episode, step_index, 0, kind_mask, move_mask, switch_mask);
        if (choice->slot0_kind == FACTORIZED_ACTION_MOVE) {
            tera_mask[0] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + choice->slot0_move_index] ? 1 : 0;
            tera_mask[1] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + 4 + choice->slot0_move_index] ? 1 : 0;
            joint_log_prob += masked_small_log_prob(slot0_kind_policy, kind_mask, FACTORIZED_KIND_DIM, 0);
            joint_log_prob += masked_small_log_prob(slot0_move_policy, move_mask, FACTORIZED_MOVE_DIM, choice->slot0_move_index);
            joint_log_prob += masked_small_log_prob(slot0_tera_policy, tera_mask, FACTORIZED_TERA_DIM, choice->slot0_use_tera ? 1 : 0);
            entropy += masked_small_entropy(slot0_kind_policy, kind_mask, FACTORIZED_KIND_DIM);
            entropy += masked_small_entropy(slot0_move_policy, move_mask, FACTORIZED_MOVE_DIM);
            entropy += masked_small_entropy(slot0_tera_policy, tera_mask, FACTORIZED_TERA_DIM);
            if (choice->slot0_target_mask != 0u) {
                factorized_target_mask_to_array(choice->slot0_target_mask, target_mask);
                joint_log_prob += masked_small_log_prob(slot0_target_policy, target_mask, FACTORIZED_TARGET_DIM, choice->slot0_target_index);
                entropy += masked_small_entropy(slot0_target_policy, target_mask, FACTORIZED_TARGET_DIM);
            }
        } else if (choice->slot0_kind == FACTORIZED_ACTION_SWITCH) {
            joint_log_prob += masked_small_log_prob(slot0_kind_policy, kind_mask, FACTORIZED_KIND_DIM, 1);
            joint_log_prob += masked_small_log_prob(slot0_switch_policy, switch_mask, FACTORIZED_SWITCH_DIM, choice->slot0_switch_index);
            entropy += masked_small_entropy(slot0_kind_policy, kind_mask, FACTORIZED_KIND_DIM);
            entropy += masked_small_entropy(slot0_switch_policy, switch_mask, FACTORIZED_SWITCH_DIM);
        }
    }
    if (choice->slot1_has_action) {
        build_factor_masks_from_episode(episode, step_index, 1, kind_mask, move_mask, switch_mask);
        if (choice->slot1_kind == FACTORIZED_ACTION_MOVE) {
            tera_mask[0] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + 14 + choice->slot1_move_index] ? 1 : 0;
            tera_mask[1] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + 18 + choice->slot1_move_index] ? 1 : 0;
            joint_log_prob += masked_small_log_prob(slot1_kind_policy, kind_mask, FACTORIZED_KIND_DIM, 0);
            joint_log_prob += masked_small_log_prob(slot1_move_policy, move_mask, FACTORIZED_MOVE_DIM, choice->slot1_move_index);
            joint_log_prob += masked_small_log_prob(slot1_tera_policy, tera_mask, FACTORIZED_TERA_DIM, choice->slot1_use_tera ? 1 : 0);
            entropy += masked_small_entropy(slot1_kind_policy, kind_mask, FACTORIZED_KIND_DIM);
            entropy += masked_small_entropy(slot1_move_policy, move_mask, FACTORIZED_MOVE_DIM);
            entropy += masked_small_entropy(slot1_tera_policy, tera_mask, FACTORIZED_TERA_DIM);
            if (choice->slot1_target_mask != 0u) {
                factorized_target_mask_to_array(choice->slot1_target_mask, target_mask);
                joint_log_prob += masked_small_log_prob(slot1_target_policy, target_mask, FACTORIZED_TARGET_DIM, choice->slot1_target_index);
                entropy += masked_small_entropy(slot1_target_policy, target_mask, FACTORIZED_TARGET_DIM);
            }
        } else if (choice->slot1_kind == FACTORIZED_ACTION_SWITCH) {
            joint_log_prob += masked_small_log_prob(slot1_kind_policy, kind_mask, FACTORIZED_KIND_DIM, 1);
            joint_log_prob += masked_small_log_prob(slot1_switch_policy, switch_mask, FACTORIZED_SWITCH_DIM, choice->slot1_switch_index);
            entropy += masked_small_entropy(slot1_kind_policy, kind_mask, FACTORIZED_KIND_DIM);
            entropy += masked_small_entropy(slot1_switch_policy, switch_mask, FACTORIZED_SWITCH_DIM);
        }
    }
    if (joint_log_prob_out) {
        *joint_log_prob_out = joint_log_prob;
    }
    if (value_out) {
        *value_out = value;
    }
    if (entropy_out) {
        *entropy_out = entropy;
    }
    return 1;
}

static float factorized_step_anchor_kl(
    const FactorizedPolicySnapshot* current,
    const FactorizedPolicySnapshot* anchor,
    const Episode* episode,
    size_t step_index
) {
    unsigned char kind_mask[FACTORIZED_KIND_DIM];
    unsigned char move_mask[FACTORIZED_MOVE_DIM];
    unsigned char switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char tera_mask[FACTORIZED_TERA_DIM];
    unsigned char target_mask[FACTORIZED_TARGET_DIM];
    const FactorizedActionChoice* choice;
    float kl = 0.0f;
    if (!current || !anchor || !episode || step_index >= episode->count) {
        return 0.0f;
    }
    choice = &episode->factorized_actions[step_index];
    if (choice->slot0_has_action && choice->slot1_has_action) {
        kl += masked_policy_kl(anchor->joint_policy, current->joint_policy, NULL, FACTORIZED_JOINT_DIM);
        if (choice->slot0_kind == FACTORIZED_ACTION_MOVE && choice->slot0_target_mask != 0u) {
            factorized_target_mask_to_array(choice->slot0_target_mask, target_mask);
            kl += masked_policy_kl(anchor->slot0_target_policy, current->slot0_target_policy,
                target_mask, FACTORIZED_TARGET_DIM);
        }
        if (choice->slot1_kind == FACTORIZED_ACTION_MOVE && choice->slot1_target_mask != 0u) {
            factorized_target_mask_to_array(choice->slot1_target_mask, target_mask);
            kl += masked_policy_kl(anchor->slot1_target_policy, current->slot1_target_policy,
                target_mask, FACTORIZED_TARGET_DIM);
        }
        return kl;
    }
    if (choice->slot0_has_action) {
        build_factor_masks_from_episode(episode, step_index, 0, kind_mask, move_mask, switch_mask);
        kl += masked_policy_kl(anchor->slot0_kind_policy, current->slot0_kind_policy,
            kind_mask, FACTORIZED_KIND_DIM);
        if (choice->slot0_kind == FACTORIZED_ACTION_MOVE) {
            tera_mask[0] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + choice->slot0_move_index] ? 1 : 0;
            tera_mask[1] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + 4 + choice->slot0_move_index] ? 1 : 0;
            kl += masked_policy_kl(anchor->slot0_move_policy, current->slot0_move_policy,
                move_mask, FACTORIZED_MOVE_DIM);
            kl += masked_policy_kl(anchor->slot0_tera_policy, current->slot0_tera_policy,
                tera_mask, FACTORIZED_TERA_DIM);
            if (choice->slot0_target_mask != 0u) {
                factorized_target_mask_to_array(choice->slot0_target_mask, target_mask);
                kl += masked_policy_kl(anchor->slot0_target_policy, current->slot0_target_policy,
                    target_mask, FACTORIZED_TARGET_DIM);
            }
        } else if (choice->slot0_kind == FACTORIZED_ACTION_SWITCH) {
            kl += masked_policy_kl(anchor->slot0_switch_policy, current->slot0_switch_policy,
                switch_mask, FACTORIZED_SWITCH_DIM);
        }
    }
    if (choice->slot1_has_action) {
        build_factor_masks_from_episode(episode, step_index, 1, kind_mask, move_mask, switch_mask);
        kl += masked_policy_kl(anchor->slot1_kind_policy, current->slot1_kind_policy,
            kind_mask, FACTORIZED_KIND_DIM);
        if (choice->slot1_kind == FACTORIZED_ACTION_MOVE) {
            tera_mask[0] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + 14 + choice->slot1_move_index] ? 1 : 0;
            tera_mask[1] = episode->legal_masks[step_index * OBS_NUM_ACTIONS + 18 + choice->slot1_move_index] ? 1 : 0;
            kl += masked_policy_kl(anchor->slot1_move_policy, current->slot1_move_policy,
                move_mask, FACTORIZED_MOVE_DIM);
            kl += masked_policy_kl(anchor->slot1_tera_policy, current->slot1_tera_policy,
                tera_mask, FACTORIZED_TERA_DIM);
            if (choice->slot1_target_mask != 0u) {
                factorized_target_mask_to_array(choice->slot1_target_mask, target_mask);
                kl += masked_policy_kl(anchor->slot1_target_policy, current->slot1_target_policy,
                    target_mask, FACTORIZED_TARGET_DIM);
            }
        } else if (choice->slot1_kind == FACTORIZED_ACTION_SWITCH) {
            kl += masked_policy_kl(anchor->slot1_switch_policy, current->slot1_switch_policy,
                switch_mask, FACTORIZED_SWITCH_DIM);
        }
    }
    return kl;
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
    size_t hidden_dim;
    size_t action_dim;
    clock_t cache_start_clock = 0;
    clock_t update_start_clock = 0;
    float action_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float accuracy_sum = 0.0f;
    size_t trained = 0;
    size_t window_count = 0;
    size_t batch_flushes = 0;
    int profile_enabled;
    float* hidden = NULL;
    float* next_hidden = NULL;
    float* hidden_after = NULL;
    float* policy = NULL;
    float value = 0.0f;
    uint8_t slot_mask_a[OBS_NUM_ACTIONS];
    uint8_t slot_mask_b[OBS_NUM_ACTIONS];
    if (!trainer || !model || !episode) {
        return 0;
    }
    profile_enabled = trainer->supervised_profile_enabled ? 1 : 0;

    hidden_dim = gru_model_hidden_dim(model);
    action_dim = gru_model_num_actions(model);
    hidden = (float*)calloc(hidden_dim, sizeof(float));
    next_hidden = (float*)malloc(hidden_dim * sizeof(float));
    hidden_after = (float*)calloc(episode->count * hidden_dim, sizeof(float));
    policy = (float*)malloc(action_dim * sizeof(float));
    if (!hidden || !next_hidden || !hidden_after || !policy) {
        free(hidden);
        free(next_hidden);
        free(hidden_after);
        free(policy);
        return 0;
    }

    trainer->last_supervised_cache_seconds = 0.0;
    trainer->last_supervised_update_seconds = 0.0;
    trainer->last_supervised_label_count = 0;
    trainer->last_supervised_window_count = 0;
    trainer->last_supervised_batch_flushes = 0;
    gru_model_clear_accumulated_supervised_updates(model);

    if (profile_enabled) {
        cache_start_clock = clock();
    }
    gru_model_zero_state(model, hidden);
    for (t = 0; t < episode->count; ++t) {
        gru_model_forward_step(
            model,
            episode->observations + (t * episode->obs_dim),
            hidden,
            next_hidden,
            policy,
            &value);
        memcpy(hidden_after + (t * hidden_dim), next_hidden, hidden_dim * sizeof(float));
        memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
    }
    if (profile_enabled) {
        trainer->last_supervised_cache_seconds =
            (double)(clock() - cache_start_clock) / (double)CLOCKS_PER_SEC;
    }

    if (profile_enabled) {
        update_start_clock = clock();
    }
    for (t = 0; t < episode->count; ++t) {
        size_t start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        size_t steps = (t - start) + 1;
        const float* initial_hidden = start > 0 ? (hidden_after + ((start - 1) * hidden_dim)) : NULL;
        float action_loss = 0.0f;
        float value_loss = 0.0f;
        float accuracy = 0.0f;
        size_t label_count = 0;
        int accumulated = 0;

        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            continue;
        }
        if (episode->actions[t] >= 0) {
            build_step_slot_legal_mask(episode, t, obs_action_slot((enum ObsAction)episode->actions[t]), slot_mask_a);
            ++label_count;
        }
        if (episode->actions2[t] >= 0) {
            build_step_slot_legal_mask(episode, t, obs_action_slot((enum ObsAction)episode->actions2[t]), slot_mask_b);
            ++label_count;
        }

        if (!gru_model_supervised_accumulate_sequence_window_factorized(
                model,
                episode->observations + (start * episode->obs_dim),
                steps,
                initial_hidden,
                slot_mask_a,
                slot_mask_b,
                &episode->factorized_actions[t],
                episode->rewards[t],
                &action_loss,
                &value_loss,
                &accuracy)) {
            free(hidden);
            free(next_hidden);
            free(hidden_after);
            free(policy);
            return 0;
        }
        accumulated = 1;

        action_loss_sum += action_loss * (float)label_count;
        value_loss_sum += value_loss * (float)label_count;
        accuracy_sum += accuracy * (float)label_count;
        trained += label_count;
        trainer->step += label_count;
        window_count += 1;
        if (accumulated && trainer->supervised_minibatch_size > 0 &&
                (window_count % trainer->supervised_minibatch_size) == 0) {
            if (!apply_supervised_updates(trainer, model)) {
                free(hidden);
                free(next_hidden);
                free(hidden_after);
                free(policy);
                return 0;
            }
            batch_flushes += 1;
        }
    }
    if (!apply_supervised_updates(trainer, model)) {
        free(hidden);
        free(next_hidden);
        free(hidden_after);
        free(policy);
        return 0;
    }
    if (trainer->supervised_minibatch_size > 0 &&
            window_count > 0 &&
            (window_count % trainer->supervised_minibatch_size) != 0) {
        batch_flushes += 1;
    }
    if (profile_enabled) {
        trainer->last_supervised_update_seconds =
            (double)(clock() - update_start_clock) / (double)CLOCKS_PER_SEC;
        trainer->last_supervised_label_count = trained;
        trainer->last_supervised_window_count = window_count;
        trainer->last_supervised_batch_flushes = batch_flushes;
    }

    if (trained > 0) {
        trainer->last_action_loss = action_loss_sum / (float)trained;
        trainer->last_value_loss = value_loss_sum / (float)trained;
        trainer->last_accuracy = accuracy_sum / (float)trained;
    }
    free(hidden);
    free(next_hidden);
    free(hidden_after);
    free(policy);
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
    float* hidden_after = NULL;
    float* policy_step = NULL;
    float* masked_policy = NULL;
    float* anchor_hidden = NULL;
    float* anchor_next_hidden = NULL;
    float* anchor_hidden_after = NULL;
    float* anchor_policy = NULL;
    size_t* labeled_indices = NULL;
    float return_sum = 0.0f;
    float advantage_sum = 0.0f;
    float abs_advantage_sum = 0.0f;
    float mean_value_sum = 0.0f;
    float policy_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float entropy_sum = 0.0f;
    float anchor_kl_sum = 0.0f;
    float anchor_loss_sum = 0.0f;
    float anchor_kl_max = 0.0f;
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
    hidden_after = (float*)calloc(episode->count * hidden_dim, sizeof(float));
    policy_step = (float*)malloc(action_dim * sizeof(float));
    masked_policy = (float*)malloc(action_dim * sizeof(float));
    labeled_indices = (size_t*)calloc(episode->count, sizeof(size_t));
    if (trainer->anchor_model && trainer->anchor_kl_coef > 0.0f) {
        anchor_hidden = (float*)calloc(hidden_dim, sizeof(float));
        anchor_next_hidden = (float*)malloc(hidden_dim * sizeof(float));
        anchor_hidden_after = (float*)calloc(episode->count * hidden_dim, sizeof(float));
        anchor_policy = (float*)malloc(action_dim * sizeof(float));
    }
    if (!returns || !advantages || !values || !hidden || !next_hidden || !hidden_after || !policy_step || !masked_policy || !labeled_indices ||
            ((trainer->anchor_model && trainer->anchor_kl_coef > 0.0f) && (!anchor_hidden || !anchor_next_hidden || !anchor_hidden_after || !anchor_policy))) {
        free(returns);
        free(advantages);
        free(values);
        free(hidden);
        free(next_hidden);
        free(hidden_after);
        free(policy_step);
        free(masked_policy);
        free(anchor_hidden);
        free(anchor_next_hidden);
        free(anchor_hidden_after);
        free(anchor_policy);
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
    if (trainer->anchor_model && trainer->anchor_kl_coef > 0.0f) {
        gru_model_zero_state(trainer->anchor_model, anchor_hidden);
    }
    for (t = 0; t < episode->count; ++t) {
        float value = 0.0f;
        gru_model_forward_step(
            model,
            episode->observations + (t * episode->obs_dim),
            hidden,
            next_hidden,
            policy_step,
            &value);
        memcpy(hidden_after + (t * hidden_dim), next_hidden, hidden_dim * sizeof(float));
        if (trainer->anchor_model && trainer->anchor_kl_coef > 0.0f) {
            float anchor_value = 0.0f;
            gru_model_forward_step(
                trainer->anchor_model,
                episode->observations + (t * episode->obs_dim),
                anchor_hidden,
                anchor_next_hidden,
                anchor_policy,
                &anchor_value);
            memcpy(anchor_hidden_after + (t * hidden_dim), anchor_next_hidden, hidden_dim * sizeof(float));
            memcpy(anchor_hidden, anchor_next_hidden, hidden_dim * sizeof(float));
        }
        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
            continue;
        }
        values[labeled_steps] = value;
        labeled_indices[labeled_steps] = t;
        advantages[labeled_steps] = returns[t] - values[labeled_steps];
        advantage_sum += advantages[labeled_steps];
        return_sum += returns[t];
        mean_value_sum += values[labeled_steps];
        ++labeled_steps;
        memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
    }

    if (labeled_steps == 0) {
        trainer->last_policy_loss = 0.0f;
        trainer->last_value_loss = 0.0f;
        trainer->last_mean_return = 0.0f;
        trainer->last_mean_advantage = 0.0f;
        trainer->last_mean_abs_advantage = 0.0f;
        trainer->last_mean_value = 0.0f;
        trainer->last_entropy = 0.0f;
        trainer->last_anchor_kl_mean = 0.0f;
        trainer->last_anchor_kl_max = 0.0f;
        trainer->last_anchor_loss = 0.0f;
        trainer->last_rl_labels = 0;
        free(returns);
        free(advantages);
        free(values);
        free(hidden);
        free(next_hidden);
        free(hidden_after);
        free(policy_step);
        free(masked_policy);
        free(anchor_hidden);
        free(anchor_next_hidden);
        free(anchor_hidden_after);
        free(anchor_policy);
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
        size_t start = (episode_t + 1 > trainer->bptt_window) ? (episode_t + 1 - trainer->bptt_window) : 0;
        size_t steps = (episode_t - start) + 1;
        const float* initial_hidden = start > 0 ? (hidden_after + ((start - 1) * hidden_dim)) : NULL;
        const float* stored_hidden = hidden_after + (episode_t * hidden_dim);
        const float* stored_anchor_hidden = (trainer->anchor_model && trainer->anchor_kl_coef > 0.0f)
            ? (anchor_hidden_after + (episode_t * hidden_dim))
            : NULL;
        float prob_action = 0.0f;
        float prob_action2 = 0.0f;
        size_t a;
        if (episode->actions[episode_t] >= 0 && episode->actions2[episode_t] >= 0) {
            uint8_t slot_mask_secondary[OBS_NUM_ACTIONS];
            float* anchor_policy_secondary = NULL;
            build_step_slot_legal_mask(episode, episode_t, 0, slot_mask);
            build_step_slot_legal_mask(episode, episode_t, 1, slot_mask_secondary);
            gru_model_evaluate_hidden(model, stored_hidden, slot_mask, masked_policy, &values[t]);
            prob_action = masked_policy[episode->actions[episode_t]] > 1.0e-8f ? masked_policy[episode->actions[episode_t]] : 1.0e-8f;
            policy_loss_sum += -logf(prob_action) * advantages[t];
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
            if (stored_anchor_hidden && anchor_policy) {
                size_t k;
                float anchor_value = 0.0f;
                float slot_kl = 0.0f;
                gru_model_evaluate_hidden(trainer->anchor_model, stored_anchor_hidden, slot_mask, anchor_policy, &anchor_value);
                for (k = 0; k < action_dim; ++k) {
                    float q;
                    float p;
                    if (!slot_mask[k]) {
                        continue;
                    }
                    q = anchor_policy[k] > 1.0e-8f ? anchor_policy[k] : 1.0e-8f;
                    p = masked_policy[k] > 1.0e-8f ? masked_policy[k] : 1.0e-8f;
                    slot_kl += q * (logf(q) - logf(p));
                }
                anchor_kl_sum += slot_kl;
                anchor_loss_sum += trainer->anchor_kl_coef * slot_kl;
                if (slot_kl > anchor_kl_max) {
                    anchor_kl_max = slot_kl;
                }
                anchor_policy_secondary = policy_step;
                gru_model_evaluate_hidden(trainer->anchor_model, stored_anchor_hidden, slot_mask_secondary, anchor_policy_secondary, &anchor_value);
            }
            gru_model_evaluate_hidden(model, stored_hidden, slot_mask_secondary, masked_policy, &values[t]);
            prob_action2 = masked_policy[episode->actions2[episode_t]] > 1.0e-8f ? masked_policy[episode->actions2[episode_t]] : 1.0e-8f;
            policy_loss_sum += -logf(prob_action2) * advantages[t];
            value_loss_sum += 0.5f * (values[t] - returns[episode_t]) * (values[t] - returns[episode_t]);
            {
                float slot_entropy = 0.0f;
                for (a = 0; a < action_dim; ++a) {
                    float p;
                    if (!slot_mask_secondary[a]) {
                        continue;
                    }
                    p = masked_policy[a] > 1.0e-8f ? masked_policy[a] : 1.0e-8f;
                    slot_entropy -= p * logf(p);
                }
                entropy_sum += slot_entropy;
            }
            if (anchor_policy_secondary) {
                size_t k;
                float slot_kl = 0.0f;
                for (k = 0; k < action_dim; ++k) {
                    float q;
                    float p;
                    if (!slot_mask_secondary[k]) {
                        continue;
                    }
                    q = anchor_policy_secondary[k] > 1.0e-8f ? anchor_policy_secondary[k] : 1.0e-8f;
                    p = masked_policy[k] > 1.0e-8f ? masked_policy[k] : 1.0e-8f;
                    slot_kl += q * (logf(q) - logf(p));
                }
                anchor_kl_sum += slot_kl;
                anchor_loss_sum += trainer->anchor_kl_coef * slot_kl;
                if (slot_kl > anchor_kl_max) {
                    anchor_kl_max = slot_kl;
                }
            }
            advantage_sum += advantages[t] * 2.0f;
            abs_advantage_sum += fabsf(advantages[t]) * 2.0f;
            trained_labels += 2;
            if (!((stored_anchor_hidden && anchor_policy && anchor_policy_secondary && trainer->anchor_kl_coef > 0.0f)
                    ? gru_model_policy_gradient_update_sequence_window_dual_anchored(
                        model,
                        episode->observations + (start * episode->obs_dim),
                        steps,
                        initial_hidden,
                        slot_mask,
                        episode->actions[episode_t],
                        slot_mask_secondary,
                        episode->actions2[episode_t],
                        advantages[t],
                        returns[episode_t],
                        trainer->entropy_coef,
                        trainer->learning_rate,
                        anchor_policy,
                        anchor_policy_secondary,
                        trainer->anchor_kl_coef)
                    : gru_model_policy_gradient_update_sequence_window_dual(
                    model,
                    episode->observations + (start * episode->obs_dim),
                    steps,
                    initial_hidden,
                    slot_mask,
                    episode->actions[episode_t],
                    slot_mask_secondary,
                    episode->actions2[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate))) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(next_hidden);
                free(hidden_after);
                free(policy_step);
                free(masked_policy);
                free(anchor_hidden);
                free(anchor_next_hidden);
                free(anchor_hidden_after);
                free(anchor_policy);
                free(labeled_indices);
                return 0;
            }
            trainer->step += 2;
            continue;
        }
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
            if (stored_anchor_hidden && anchor_policy) {
                size_t k;
                float anchor_value = 0.0f;
                float slot_kl = 0.0f;
                gru_model_evaluate_hidden(trainer->anchor_model, stored_anchor_hidden, slot_mask, anchor_policy, &anchor_value);
                for (k = 0; k < action_dim; ++k) {
                    float q;
                    float p;
                    if (!slot_mask[k]) {
                        continue;
                    }
                    q = anchor_policy[k] > 1.0e-8f ? anchor_policy[k] : 1.0e-8f;
                    p = masked_policy[k] > 1.0e-8f ? masked_policy[k] : 1.0e-8f;
                    slot_kl += q * (logf(q) - logf(p));
                }
                anchor_kl_sum += slot_kl;
                anchor_loss_sum += trainer->anchor_kl_coef * slot_kl;
                if (slot_kl > anchor_kl_max) {
                    anchor_kl_max = slot_kl;
                }
            }
            advantage_sum += advantages[t];
            abs_advantage_sum += fabsf(advantages[t]);
            ++trained_labels;
            if (!((stored_anchor_hidden && anchor_policy && trainer->anchor_kl_coef > 0.0f)
                    ? gru_model_policy_gradient_update_sequence_window_anchored(
                        model,
                        episode->observations + (start * episode->obs_dim),
                        steps,
                        initial_hidden,
                        slot_mask,
                        episode->actions[episode_t],
                        advantages[t],
                        returns[episode_t],
                        trainer->entropy_coef,
                        trainer->learning_rate,
                        anchor_policy,
                        trainer->anchor_kl_coef)
                    : gru_model_policy_gradient_update_sequence_window(
                    model,
                    episode->observations + (start * episode->obs_dim),
                    steps,
                    initial_hidden,
                    slot_mask,
                    episode->actions[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate))) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(next_hidden);
                free(hidden_after);
                free(policy_step);
                free(masked_policy);
                free(anchor_hidden);
                free(anchor_next_hidden);
                free(anchor_hidden_after);
                free(anchor_policy);
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
            if (stored_anchor_hidden && anchor_policy) {
                size_t k;
                float anchor_value = 0.0f;
                float slot_kl = 0.0f;
                gru_model_evaluate_hidden(trainer->anchor_model, stored_anchor_hidden, slot_mask, anchor_policy, &anchor_value);
                for (k = 0; k < action_dim; ++k) {
                    float q;
                    float p;
                    if (!slot_mask[k]) {
                        continue;
                    }
                    q = anchor_policy[k] > 1.0e-8f ? anchor_policy[k] : 1.0e-8f;
                    p = masked_policy[k] > 1.0e-8f ? masked_policy[k] : 1.0e-8f;
                    slot_kl += q * (logf(q) - logf(p));
                }
                anchor_kl_sum += slot_kl;
                anchor_loss_sum += trainer->anchor_kl_coef * slot_kl;
                if (slot_kl > anchor_kl_max) {
                    anchor_kl_max = slot_kl;
                }
            }
            advantage_sum += advantages[t];
            abs_advantage_sum += fabsf(advantages[t]);
            ++trained_labels;
            if (!((stored_anchor_hidden && anchor_policy && trainer->anchor_kl_coef > 0.0f)
                    ? gru_model_policy_gradient_update_sequence_window_anchored(
                        model,
                        episode->observations + (start * episode->obs_dim),
                        steps,
                        initial_hidden,
                        slot_mask,
                        episode->actions2[episode_t],
                        advantages[t],
                        returns[episode_t],
                        trainer->entropy_coef,
                        trainer->learning_rate,
                        anchor_policy,
                        trainer->anchor_kl_coef)
                    : gru_model_policy_gradient_update_sequence_window(
                    model,
                    episode->observations + (start * episode->obs_dim),
                    steps,
                    initial_hidden,
                    slot_mask,
                    episode->actions2[episode_t],
                    advantages[t],
                    returns[episode_t],
                    trainer->entropy_coef,
                    trainer->learning_rate))) {
                free(returns);
                free(advantages);
                free(values);
                free(hidden);
                free(next_hidden);
                free(hidden_after);
                free(policy_step);
                free(masked_policy);
                free(anchor_hidden);
                free(anchor_next_hidden);
                free(anchor_hidden_after);
                free(anchor_policy);
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
    trainer->last_mean_abs_advantage = trained_labels > 0 ? (abs_advantage_sum / (float)trained_labels) : 0.0f;
    trainer->last_mean_value = labeled_steps > 0 ? (mean_value_sum / (float)labeled_steps) : 0.0f;
    trainer->last_entropy = trained_labels > 0 ? (entropy_sum / (float)trained_labels) : 0.0f;
    trainer->last_anchor_kl_mean = trained_labels > 0 ? (anchor_kl_sum / (float)trained_labels) : 0.0f;
    trainer->last_anchor_kl_max = anchor_kl_max;
    trainer->last_anchor_loss = trained_labels > 0 ? (anchor_loss_sum / (float)trained_labels) : 0.0f;
    trainer->last_rl_labels = trained_labels;
    trainer->last_action_loss = trainer->last_policy_loss;
    trainer->last_accuracy = 0.0f;
    free(returns);
    free(advantages);
    free(values);
    free(hidden);
    free(next_hidden);
    free(hidden_after);
    free(policy_step);
    free(masked_policy);
    free(anchor_hidden);
    free(anchor_next_hidden);
    free(anchor_hidden_after);
    free(anchor_policy);
    free(labeled_indices);
    return 1;
}

static int gru_trainer_ppo_episode_accumulate(
    GruTrainer* trainer,
    GruModel* model,
    const Episode* episode,
    int clear_before,
    int apply_after
) {
    size_t t;
    size_t hidden_dim;
    size_t labeled_steps = 0;
    size_t trained_labels = 0;
    float* advantages = NULL;
    float* returns = NULL;
    size_t* labeled_indices = NULL;
    float* hidden = NULL;
    float* next_hidden = NULL;
    float* hidden_after = NULL;
    float* anchor_hidden = NULL;
    float* anchor_next_hidden = NULL;
    float* anchor_hidden_after = NULL;
    float gae = 0.0f;
    float advantage_sum = 0.0f;
    float return_sum = 0.0f;
    float mean_value_sum = 0.0f;
    float abs_advantage_sum = 0.0f;
    float policy_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float entropy_sum = 0.0f;
    float approx_kl_sum = 0.0f;
    float clip_fraction_sum = 0.0f;
    float anchor_kl_sum = 0.0f;
    float anchor_kl_max = 0.0f;
    uint8_t slot_mask_a[OBS_NUM_ACTIONS];
    uint8_t slot_mask_b[OBS_NUM_ACTIONS];
    int anchor_enabled;

    if (!trainer || !model || !episode) {
        return 0;
    }
    anchor_enabled = trainer->anchor_model && trainer->anchor_kl_coef > 0.0f;
    if (anchor_enabled &&
            (gru_model_input_dim(trainer->anchor_model) != gru_model_input_dim(model) ||
             gru_model_hidden_dim(trainer->anchor_model) != gru_model_hidden_dim(model) ||
             gru_model_num_actions(trainer->anchor_model) != gru_model_num_actions(model))) {
        return 0;
    }
    if (episode->count == 0) {
        trainer->last_policy_loss = 0.0f;
        trainer->last_value_loss = 0.0f;
        trainer->last_mean_return = 0.0f;
        trainer->last_mean_advantage = 0.0f;
        trainer->last_mean_abs_advantage = 0.0f;
        trainer->last_mean_value = 0.0f;
        trainer->last_entropy = 0.0f;
        trainer->last_approx_kl = 0.0f;
        trainer->last_clip_fraction = 0.0f;
        trainer->last_anchor_kl_mean = 0.0f;
        trainer->last_anchor_kl_max = 0.0f;
        trainer->last_anchor_loss = 0.0f;
        trainer->last_rl_labels = 0;
        return 1;
    }

    hidden_dim = gru_model_hidden_dim(model);
    advantages = (float*)calloc(episode->count, sizeof(float));
    returns = (float*)calloc(episode->count, sizeof(float));
    labeled_indices = (size_t*)malloc(episode->count * sizeof(size_t));
    hidden = (float*)calloc(hidden_dim, sizeof(float));
    next_hidden = (float*)malloc(hidden_dim * sizeof(float));
    hidden_after = (float*)calloc(episode->count * hidden_dim, sizeof(float));
    if (anchor_enabled) {
        anchor_hidden = (float*)calloc(hidden_dim, sizeof(float));
        anchor_next_hidden = (float*)malloc(hidden_dim * sizeof(float));
        anchor_hidden_after = (float*)calloc(episode->count * hidden_dim, sizeof(float));
    }
    if (!advantages || !returns || !labeled_indices || !hidden || !next_hidden || !hidden_after) {
        free(advantages);
        free(returns);
        free(labeled_indices);
        free(hidden);
        free(next_hidden);
        free(hidden_after);
        free(anchor_hidden);
        free(anchor_next_hidden);
        free(anchor_hidden_after);
        return 0;
    }
    if (anchor_enabled && (!anchor_hidden || !anchor_next_hidden || !anchor_hidden_after)) {
        free(advantages); free(returns); free(labeled_indices);
        free(hidden); free(next_hidden); free(hidden_after);
        free(anchor_hidden); free(anchor_next_hidden); free(anchor_hidden_after);
        return 0;
    }

    if (clear_before) {
        gru_model_clear_accumulated_supervised_updates(model);
    }

    gru_model_zero_state(model, hidden);
    if (anchor_enabled) {
        gru_model_zero_state(trainer->anchor_model, anchor_hidden);
    }
    for (t = 0; t < episode->count; ++t) {
        gru_model_forward_step(
            model,
            episode->observations + (t * episode->obs_dim),
            hidden,
            next_hidden,
            NULL,
            NULL);
        memcpy(hidden_after + (t * hidden_dim), next_hidden, hidden_dim * sizeof(float));
        memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
        if (anchor_enabled) {
            gru_model_forward_step(
                trainer->anchor_model,
                episode->observations + (t * episode->obs_dim),
                anchor_hidden,
                anchor_next_hidden,
                NULL,
                NULL);
            memcpy(anchor_hidden_after + (t * hidden_dim), anchor_next_hidden, hidden_dim * sizeof(float));
            memcpy(anchor_hidden, anchor_next_hidden, hidden_dim * sizeof(float));
        }
    }

    for (t = episode->count; t > 0; --t) {
        size_t idx = t - 1;
        float next_value = 0.0f;
        float nonterminal = episode->dones[idx] ? 0.0f : 1.0f;
        float delta;
        if (idx + 1 < episode->count) {
            next_value = episode->old_values[idx + 1];
        }
        delta = episode->rewards[idx] + trainer->gamma * next_value * nonterminal - episode->old_values[idx];
        gae = delta + trainer->gamma * trainer->gae_lambda * nonterminal * gae;
        advantages[idx] = gae;
        returns[idx] = advantages[idx] + episode->old_values[idx];
    }

    if (trainer->advantage_norm && episode->count > 1) {
        float mean = 0.0f;
        float variance = 0.0f;
        for (t = 0; t < episode->count; ++t) {
            mean += advantages[t];
        }
        mean /= (float)episode->count;
        for (t = 0; t < episode->count; ++t) {
            float centered = advantages[t] - mean;
            variance += centered * centered;
        }
        variance /= (float)episode->count;
        variance = sqrtf(variance);
        if (variance > 1.0e-6f) {
            for (t = 0; t < episode->count; ++t) {
                advantages[t] = (advantages[t] - mean) / variance;
            }
        }
    }

    for (t = 0; t < episode->count; ++t) {
        float current_log_prob;
        float current_value;
        float current_entropy;
        float ratio;
        float clipped_ratio;
        float effective_advantage;
        float approx_kl;
        float surrogate_a;
        float surrogate_b;
        float value_delta;
        float clipped_value_target;
        float unclipped_value_loss;
        float clipped_value_loss;
        float value_target;
        size_t start;
        size_t steps;
        const float* initial_hidden;
        const float* stored_hidden;
        FactorizedPolicySnapshot current_snapshot;
        FactorizedPolicySnapshot anchor_snapshot;

        if (episode->actions[t] < 0 && episode->actions2[t] < 0) {
            continue;
        }
        stored_hidden = hidden_after + (t * hidden_dim);
        if (!evaluate_joint_step(model, stored_hidden, episode, t, &current_log_prob, &current_value,
                &current_entropy, &current_snapshot)) {
            free(advantages);
            free(returns);
            free(labeled_indices);
            free(hidden);
            free(next_hidden);
            free(hidden_after);
            free(anchor_hidden); free(anchor_next_hidden); free(anchor_hidden_after);
            return 0;
        }
        if (anchor_enabled) {
            float anchor_log_prob;
            float anchor_value;
            float anchor_entropy;
            float step_anchor_kl;
            if (!evaluate_joint_step(trainer->anchor_model,
                    anchor_hidden_after + (t * hidden_dim), episode, t,
                    &anchor_log_prob, &anchor_value, &anchor_entropy, &anchor_snapshot)) {
                free(advantages); free(returns); free(labeled_indices);
                free(hidden); free(next_hidden); free(hidden_after);
                free(anchor_hidden); free(anchor_next_hidden); free(anchor_hidden_after);
                return 0;
            }
            step_anchor_kl = factorized_step_anchor_kl(&current_snapshot, &anchor_snapshot, episode, t);
            anchor_kl_sum += step_anchor_kl;
            if (step_anchor_kl > anchor_kl_max) {
                anchor_kl_max = step_anchor_kl;
            }
        }
        ratio = expf(current_log_prob - episode->old_log_probs[t]);
        clipped_ratio = ratio;
        if (clipped_ratio < 1.0f - trainer->ppo_clip_epsilon) {
            clipped_ratio = 1.0f - trainer->ppo_clip_epsilon;
        }
        if (clipped_ratio > 1.0f + trainer->ppo_clip_epsilon) {
            clipped_ratio = 1.0f + trainer->ppo_clip_epsilon;
        }
        surrogate_a = ratio * advantages[t];
        surrogate_b = clipped_ratio * advantages[t];
        effective_advantage = 0.0f;
        if ((advantages[t] >= 0.0f && ratio <= 1.0f + trainer->ppo_clip_epsilon) ||
                (advantages[t] < 0.0f && ratio >= 1.0f - trainer->ppo_clip_epsilon)) {
            effective_advantage = ratio * advantages[t];
        }
        approx_kl = episode->old_log_probs[t] - current_log_prob;
        value_delta = returns[t] - episode->old_values[t];
        if (value_delta < -trainer->ppo_value_clip_epsilon) {
            value_delta = -trainer->ppo_value_clip_epsilon;
        }
        if (value_delta > trainer->ppo_value_clip_epsilon) {
            value_delta = trainer->ppo_value_clip_epsilon;
        }
        clipped_value_target = episode->old_values[t] + value_delta;
        unclipped_value_loss = 0.5f * (current_value - returns[t]) * (current_value - returns[t]);
        clipped_value_loss = 0.5f * (current_value - clipped_value_target) * (current_value - clipped_value_target);
        value_target = unclipped_value_loss >= clipped_value_loss ? returns[t] : clipped_value_target;
        approx_kl_sum += approx_kl;
        if ((advantages[t] >= 0.0f && surrogate_b < surrogate_a) ||
                (advantages[t] < 0.0f && surrogate_b < surrogate_a)) {
            clip_fraction_sum += 1.0f;
        }
        policy_loss_sum += -(surrogate_a < surrogate_b ? surrogate_a : surrogate_b);
        value_loss_sum += unclipped_value_loss >= clipped_value_loss ? unclipped_value_loss : clipped_value_loss;
        entropy_sum += current_entropy;
        advantage_sum += advantages[t];
        abs_advantage_sum += fabsf(advantages[t]);
        return_sum += returns[t];
        mean_value_sum += current_value;

        start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        steps = (t - start) + 1;
        initial_hidden = start > 0 ? (hidden_after + ((start - 1) * hidden_dim)) : NULL;
        if (episode->actions[t] >= 0) {
            build_step_slot_legal_mask(episode, t, 0, slot_mask_a);
        }
        if (episode->actions2[t] >= 0) {
            build_step_slot_legal_mask(episode, t, 1, slot_mask_b);
        }
        if ((fabsf(effective_advantage) > 0.0f || anchor_enabled) &&
                !(anchor_enabled ?
                gru_model_policy_gradient_accumulate_sequence_window_factorized_anchored(
                    model,
                    episode->observations + (start * episode->obs_dim),
                    steps,
                    initial_hidden,
                    slot_mask_a,
                    slot_mask_b,
                    &episode->factorized_actions[t],
                    effective_advantage,
                    value_target,
                    trainer->entropy_coef,
                    &anchor_snapshot,
                    trainer->anchor_kl_coef) :
                gru_model_policy_gradient_accumulate_sequence_window_factorized(
                    model,
                    episode->observations + (start * episode->obs_dim),
                    steps,
                    initial_hidden,
                    slot_mask_a,
                    slot_mask_b,
                    &episode->factorized_actions[t],
                    effective_advantage,
                    value_target,
                    trainer->entropy_coef))) {
            free(advantages); free(returns); free(labeled_indices); free(hidden); free(next_hidden); free(hidden_after);
            free(anchor_hidden); free(anchor_next_hidden); free(anchor_hidden_after);
            return 0;
        }
        ++trained_labels;
        labeled_indices[labeled_steps++] = t;
        ++trainer->step;
    }

    if (apply_after && !gru_model_apply_accumulated_adam_updates(
            model,
            trainer->learning_rate,
            trainer->adam_beta1,
            trainer->adam_beta2,
            trainer->adam_epsilon,
            trainer->gradient_clip)) {
        free(advantages);
        free(returns);
        free(labeled_indices);
        free(hidden);
        free(next_hidden);
        free(hidden_after);
        free(anchor_hidden);
        free(anchor_next_hidden);
        free(anchor_hidden_after);
        return 0;
    }

    trainer->last_policy_loss = labeled_steps > 0 ? policy_loss_sum / (float)labeled_steps : 0.0f;
    trainer->last_value_loss = labeled_steps > 0 ? value_loss_sum / (float)labeled_steps : 0.0f;
    trainer->last_mean_return = labeled_steps > 0 ? return_sum / (float)labeled_steps : 0.0f;
    trainer->last_mean_advantage = labeled_steps > 0 ? advantage_sum / (float)labeled_steps : 0.0f;
    trainer->last_mean_abs_advantage = labeled_steps > 0 ? abs_advantage_sum / (float)labeled_steps : 0.0f;
    trainer->last_mean_value = labeled_steps > 0 ? mean_value_sum / (float)labeled_steps : 0.0f;
    trainer->last_entropy = labeled_steps > 0 ? entropy_sum / (float)labeled_steps : 0.0f;
    trainer->last_approx_kl = labeled_steps > 0 ? approx_kl_sum / (float)labeled_steps : 0.0f;
    trainer->last_clip_fraction = labeled_steps > 0 ? clip_fraction_sum / (float)labeled_steps : 0.0f;
    trainer->last_anchor_kl_mean = labeled_steps > 0 ? anchor_kl_sum / (float)labeled_steps : 0.0f;
    trainer->last_anchor_kl_max = anchor_kl_max;
    trainer->last_anchor_loss = trainer->anchor_kl_coef * trainer->last_anchor_kl_mean;
    trainer->last_rl_labels = trained_labels;

    free(advantages);
    free(returns);
    free(labeled_indices);
    free(hidden);
    free(next_hidden);
    free(hidden_after);
    free(anchor_hidden);
    free(anchor_next_hidden);
    free(anchor_hidden_after);
    return 1;
}

int gru_trainer_ppo_episode(GruTrainer* trainer, GruModel* model, const Episode* episode) {
    return gru_trainer_ppo_episode_accumulate(trainer, model, episode, 1, 1);
}

int gru_trainer_ppo_hard_kl_stop_update(
    float approx_kl,
    float target_kl,
    float hard_multiplier,
    int required_consecutive_updates,
    int* consecutive_breaches
) {
    int required = required_consecutive_updates > 0 ? required_consecutive_updates : 1;
    if (!consecutive_breaches) {
        return 0;
    }
    if (target_kl <= 0.0f || hard_multiplier <= 0.0f ||
            approx_kl <= target_kl * hard_multiplier) {
        *consecutive_breaches = 0;
        return 0;
    }
    ++(*consecutive_breaches);
    return *consecutive_breaches >= required;
}

int gru_trainer_ppo_minibatch(
    GruTrainer* trainer,
    GruModel* model,
    const Episode* const* episodes,
    size_t episode_count
) {
    size_t i;
    size_t labels = 0;
    double policy_loss_sum = 0.0;
    double value_loss_sum = 0.0;
    double mean_return_sum = 0.0;
    double mean_advantage_sum = 0.0;
    double mean_abs_advantage_sum = 0.0;
    double mean_value_sum = 0.0;
    double entropy_sum = 0.0;
    double approx_kl_sum = 0.0;
    double clip_fraction_sum = 0.0;
    double anchor_kl_sum = 0.0;
    double anchor_loss_sum = 0.0;
    float anchor_kl_max = 0.0f;

    if (!trainer || !model || !episodes || episode_count == 0) {
        return 0;
    }
    gru_model_clear_accumulated_supervised_updates(model);
    for (i = 0; i < episode_count; ++i) {
        size_t episode_labels;
        if (!episodes[i] || !gru_trainer_ppo_episode_accumulate(trainer, model, episodes[i], 0, 0)) {
            gru_model_clear_accumulated_supervised_updates(model);
            return 0;
        }
        episode_labels = trainer->last_rl_labels;
        labels += episode_labels;
        policy_loss_sum += (double)trainer->last_policy_loss * episode_labels;
        value_loss_sum += (double)trainer->last_value_loss * episode_labels;
        mean_return_sum += (double)trainer->last_mean_return * episode_labels;
        mean_advantage_sum += (double)trainer->last_mean_advantage * episode_labels;
        mean_abs_advantage_sum += (double)trainer->last_mean_abs_advantage * episode_labels;
        mean_value_sum += (double)trainer->last_mean_value * episode_labels;
        entropy_sum += (double)trainer->last_entropy * episode_labels;
        approx_kl_sum += (double)trainer->last_approx_kl * episode_labels;
        clip_fraction_sum += (double)trainer->last_clip_fraction * episode_labels;
        anchor_kl_sum += (double)trainer->last_anchor_kl_mean * episode_labels;
        anchor_loss_sum += (double)trainer->last_anchor_loss * episode_labels;
        if (trainer->last_anchor_kl_max > anchor_kl_max) {
            anchor_kl_max = trainer->last_anchor_kl_max;
        }
    }
    if (!gru_model_apply_accumulated_adam_updates(
            model,
            trainer->learning_rate,
            trainer->adam_beta1,
            trainer->adam_beta2,
            trainer->adam_epsilon,
            trainer->gradient_clip)) {
        gru_model_clear_accumulated_supervised_updates(model);
        return 0;
    }
    trainer->last_policy_loss = labels > 0 ? (float)(policy_loss_sum / labels) : 0.0f;
    trainer->last_value_loss = labels > 0 ? (float)(value_loss_sum / labels) : 0.0f;
    trainer->last_mean_return = labels > 0 ? (float)(mean_return_sum / labels) : 0.0f;
    trainer->last_mean_advantage = labels > 0 ? (float)(mean_advantage_sum / labels) : 0.0f;
    trainer->last_mean_abs_advantage = labels > 0 ? (float)(mean_abs_advantage_sum / labels) : 0.0f;
    trainer->last_mean_value = labels > 0 ? (float)(mean_value_sum / labels) : 0.0f;
    trainer->last_entropy = labels > 0 ? (float)(entropy_sum / labels) : 0.0f;
    trainer->last_approx_kl = labels > 0 ? (float)(approx_kl_sum / labels) : 0.0f;
    trainer->last_clip_fraction = labels > 0 ? (float)(clip_fraction_sum / labels) : 0.0f;
    trainer->last_anchor_kl_mean = labels > 0 ? (float)(anchor_kl_sum / labels) : 0.0f;
    trainer->last_anchor_kl_max = anchor_kl_max;
    trainer->last_anchor_loss = labels > 0 ? (float)(anchor_loss_sum / labels) : 0.0f;
    trainer->last_rl_labels = labels;
    return 1;
}
