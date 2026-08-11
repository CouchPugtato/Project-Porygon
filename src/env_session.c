#include "env_session.h"

#include "observation_builder.h"
#include "request_parser.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ENV_DENSE_HP_SWING_WEIGHT_DEFAULT 0.10f
#define ENV_DENSE_FAINT_SWING_WEIGHT_DEFAULT 0.25f
#define ENV_DENSE_REWARD_CLIP_DEFAULT 0.40f

typedef struct {
    float self_hp_frac_sum;
    float opp_hp_frac_sum;
    int self_fainted_count;
    int opp_fainted_count;
} EnvRewardSnapshot;

static float action_log_prob(const float* policy, int action) {
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

static float masked_binary_log_prob(const float* policy, const unsigned char* mask, int choice_index) {
    float probs[FACTORIZED_TERA_DIM];
    float sum = 0.0f;
    size_t i;
    if (!policy || !mask || choice_index < 0 || choice_index >= FACTORIZED_TERA_DIM || !mask[choice_index]) {
        return 0.0f;
    }
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) {
        probs[i] = mask[i] ? policy[i] : 0.0f;
        sum += probs[i];
    }
    if (sum <= 1.0e-8f) {
        return 0.0f;
    }
    probs[choice_index] /= sum;
    if (probs[choice_index] < 1.0e-8f) {
        probs[choice_index] = 1.0e-8f;
    }
    return logf(probs[choice_index]);
}

static int sample_small_masked(const float* policy, const unsigned char* mask, size_t n) {
    float total = 0.0f;
    float draw;
    size_t i;
    for (i = 0; i < n; ++i) {
        if (mask[i]) {
            total += policy[i] > 0.0f ? policy[i] : 0.0f;
        }
    }
    if (total <= 1.0e-8f) {
        for (i = 0; i < n; ++i) {
            if (mask[i]) {
                return (int)i;
            }
        }
        return -1;
    }
    draw = ((float)rand() / (float)RAND_MAX) * total;
    for (i = 0; i < n; ++i) {
        if (!mask[i]) {
            continue;
        }
        draw -= policy[i] > 0.0f ? policy[i] : 0.0f;
        if (draw <= 0.0f) {
            return (int)i;
        }
    }
    for (i = 0; i < n; ++i) {
        if (mask[i]) {
            return (int)i;
        }
    }
    return -1;
}

static void build_runtime_factor_masks(const unsigned char* legal_mask, int slot, unsigned char* kind_mask, unsigned char* move_mask, unsigned char* switch_mask) {
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

static EnvSession* find_session(EnvRuntime* runtime, const char* battle_id) {
    size_t i;
    for (i = 0; i < runtime->count; ++i) {
        if (strcmp(runtime->sessions[i].battle_id, battle_id) == 0) {
            return &runtime->sessions[i];
        }
    }
    return NULL;
}

static EnvSession* ensure_session(EnvRuntime* runtime, const char* battle_id, int is_doubles) {
    EnvSession* session;
    if (!runtime || !battle_id || !*battle_id) {
        return NULL;
    }
    session = find_session(runtime, battle_id);
    if (session) {
        return session;
    }
    if (runtime->count == runtime->capacity) {
        size_t new_capacity = runtime->capacity ? runtime->capacity * 2 : 4;
        EnvSession* resized = (EnvSession*)realloc(runtime->sessions, new_capacity * sizeof(EnvSession));
        if (!resized) {
            return NULL;
        }
        runtime->sessions = resized;
        runtime->capacity = new_capacity;
    }
    session = &runtime->sessions[runtime->count++];
    memset(session, 0, sizeof(*session));
    strncpy(session->battle_id, battle_id, sizeof(session->battle_id) - 1);
    raw_battle_state_init(&session->raw_state, is_doubles);
    observation_init(&session->observation);
    action_mask_init(&session->action_mask);
    parsed_request_init(&session->parsed_request);
    session->pending_action = -1;
    session->pending_action2 = -1;
    if (!episode_init(&session->episode, 32, runtime->obs_dim)) {
        return NULL;
    }
    session->hidden_state = (float*)calloc(gru_model_hidden_dim(runtime->model), sizeof(float));
    session->flat_observation = (float*)calloc(runtime->obs_dim, sizeof(float));
    if (!session->hidden_state || !session->flat_observation) {
        return NULL;
    }
    return session;
}

static void free_session(EnvSession* session) {
    if (!session) {
        return;
    }
    episode_free(&session->episode);
    free(session->hidden_state);
    free(session->flat_observation);
    memset(session, 0, sizeof(*session));
}

static int queue_prestart_message(EnvSession* session, const RuntimeMessage* msg) {
    if (!session || !msg) {
        return 0;
    }
    if (session->pending_prestart_count >= ENV_PRESTART_QUEUE_MAX) {
        return 0;
    }
    session->pending_prestart_messages[session->pending_prestart_count++] = *msg;
    return 1;
}

static float clamp_dense_reward(float reward, float reward_clip) {
    if (reward > reward_clip) {
        return reward_clip;
    }
    if (reward < -reward_clip) {
        return -reward_clip;
    }
    return reward;
}

static void compute_reward_snapshot(const RawBattleState* state, EnvRewardSnapshot* snapshot) {
    int i;
    if (!state || !snapshot) {
        return;
    }
    memset(snapshot, 0, sizeof(*snapshot));
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        const RawPokemon* self = &state->self_team[i];
        const RawPokemon* opp = &state->opp_team[i];
        if (self->fainted) {
            snapshot->self_fainted_count += 1;
        } else if (self->max_hp > 0 && self->current_hp > 0) {
            snapshot->self_hp_frac_sum += (float)self->current_hp / (float)self->max_hp;
        }
        if (opp->fainted) {
            snapshot->opp_fainted_count += 1;
        } else if (opp->max_hp > 0 && opp->current_hp > 0) {
            snapshot->opp_hp_frac_sum += (float)opp->current_hp / (float)opp->max_hp;
        }
    }
}

static float compute_dense_reward_delta(
    const EnvRewardSnapshot* previous,
    const EnvRewardSnapshot* current,
    const EnvDenseRewardConfig* dense_reward_config
) {
    float opp_hp_loss;
    float self_hp_loss;
    int opp_faints_gained;
    int self_faints_gained;
    float reward;

    if (!previous || !current) {
        return 0.0f;
    }
    opp_hp_loss = previous->opp_hp_frac_sum - current->opp_hp_frac_sum;
    self_hp_loss = previous->self_hp_frac_sum - current->self_hp_frac_sum;
    opp_faints_gained = current->opp_fainted_count - previous->opp_fainted_count;
    self_faints_gained = current->self_fainted_count - previous->self_fainted_count;
    reward =
        dense_reward_config->hp_swing_weight * (opp_hp_loss - self_hp_loss) +
        dense_reward_config->faint_swing_weight * (float)(opp_faints_gained - self_faints_gained);
    return clamp_dense_reward(reward, dense_reward_config->reward_clip);
}

static float compute_request_reward(const EnvRuntime* runtime, const EnvSession* session) {
    EnvRewardSnapshot current;
    EnvRewardSnapshot previous;

    if (!runtime || !session || runtime->reward_mode != ENV_REWARD_DENSE_ADDITIVE) {
        return 0.0f;
    }
    compute_reward_snapshot(&session->raw_state, &current);
    if (!session->reward_snapshot_valid) {
        return 0.0f;
    }
    previous.self_hp_frac_sum = session->prev_self_hp_frac_sum;
    previous.opp_hp_frac_sum = session->prev_opp_hp_frac_sum;
    previous.self_fainted_count = session->prev_self_fainted_count;
    previous.opp_fainted_count = session->prev_opp_fainted_count;
    return compute_dense_reward_delta(&previous, &current, &runtime->dense_reward_config);
}

static void update_request_reward_snapshot(EnvSession* session) {
    EnvRewardSnapshot current;
    if (!session) {
        return;
    }
    compute_reward_snapshot(&session->raw_state, &current);
    session->prev_self_hp_frac_sum = current.self_hp_frac_sum;
    session->prev_opp_hp_frac_sum = current.opp_hp_frac_sum;
    session->prev_self_fainted_count = current.self_fainted_count;
    session->prev_opp_fainted_count = current.opp_fainted_count;
    session->reward_snapshot_valid = 1;
}

static void clear_request_reward_snapshot(EnvSession* session) {
    if (!session) {
        return;
    }
    session->reward_snapshot_valid = 0;
    session->prev_self_hp_frac_sum = 0.0f;
    session->prev_opp_hp_frac_sum = 0.0f;
    session->prev_self_fainted_count = 0;
    session->prev_opp_fainted_count = 0;
}

int env_runtime_init(
    EnvRuntime* runtime,
    GruModel* model,
    FILE* replay_file,
    int replay_only,
    EnvRewardMode reward_mode,
    const EnvDenseRewardConfig* dense_reward_config,
    const char* policy_tag
) {
    if (!runtime || !model) {
        return 0;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->model = model;
    runtime->obs_dim = gru_model_input_dim(model);
    runtime->replay_file = replay_file;
    runtime->replay_only = replay_only;
    runtime->reward_mode = reward_mode;
    runtime->policy_tag[0] = '\0';
    if (policy_tag && *policy_tag) {
        snprintf(runtime->policy_tag, sizeof(runtime->policy_tag), "%s", policy_tag);
    }
    runtime->dense_reward_config.hp_swing_weight = dense_reward_config ? dense_reward_config->hp_swing_weight : ENV_DENSE_HP_SWING_WEIGHT_DEFAULT;
    runtime->dense_reward_config.faint_swing_weight = dense_reward_config ? dense_reward_config->faint_swing_weight : ENV_DENSE_FAINT_SWING_WEIGHT_DEFAULT;
    runtime->dense_reward_config.reward_clip = dense_reward_config ? dense_reward_config->reward_clip : ENV_DENSE_REWARD_CLIP_DEFAULT;
    return 1;
}

void env_runtime_free(EnvRuntime* runtime) {
    size_t i;
    if (!runtime) {
        return;
    }
    for (i = 0; i < runtime->count; ++i) {
        free_session(&runtime->sessions[i]);
    }
    free(runtime->sessions);
    memset(runtime, 0, sizeof(*runtime));
}

static int write_action(EnvRuntime* runtime, EnvSession* session, FILE* out) {
    float slot0_kind_policy[FACTORIZED_KIND_DIM];
    float slot0_move_policy[FACTORIZED_MOVE_DIM];
    float slot0_switch_policy[FACTORIZED_SWITCH_DIM];
    float slot0_tera_policy[FACTORIZED_TERA_DIM];
    float slot1_kind_policy[FACTORIZED_KIND_DIM];
    float slot1_move_policy[FACTORIZED_MOVE_DIM];
    float slot1_switch_policy[FACTORIZED_SWITCH_DIM];
    float slot1_tera_policy[FACTORIZED_TERA_DIM];
    float* pair_policy;
    float value;
    int action = -1;
    int action2 = -1;
    int legal_count = 0;
    int slot0_needs_action;
    int slot1_needs_action;
    unsigned char slot0_mask[OBS_NUM_ACTIONS];
    unsigned char slot1_mask[OBS_NUM_ACTIONS];
    unsigned char slot0_kind_mask[FACTORIZED_KIND_DIM];
    unsigned char slot0_move_mask[FACTORIZED_MOVE_DIM];
    unsigned char slot0_switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char slot1_kind_mask[FACTORIZED_KIND_DIM];
    unsigned char slot1_move_mask[FACTORIZED_MOVE_DIM];
    unsigned char slot1_switch_mask[FACTORIZED_SWITCH_DIM];
    char command[256];
    char json[512];
    int i;
    ValidatedRequestChoice validated;
    FactorizedActionChoice sampled_choice;
    if (!runtime || !session || !out || runtime->replay_only) {
        return 1;
    }
    pair_policy = (float*)calloc(OBS_NUM_ACTIONS, sizeof(float));
    if (!pair_policy) {
        free(pair_policy);
        return 0;
    }
    for (i = 0; i < OBS_NUM_ACTIONS; ++i) {
        if (session->observation.legal_mask[i]) {
            ++legal_count;
        }
    }
    if (legal_count == 0) {
        runtime_emit_error_json(json, sizeof(json), session->battle_id, "no legal actions available");
        fputs(json, out);
        fputc('\n', out);
        fflush(out);
        free(pair_policy);
        return 0;
    }
    slot0_needs_action = parsed_request_slot_needs_choice(&session->parsed_request, 0);
    slot1_needs_action = parsed_request_slot_needs_choice(&session->parsed_request, 1);
    build_slot_legal_mask(session->observation.legal_mask, 0, slot0_mask);
    build_slot_legal_mask(session->observation.legal_mask, 1, slot1_mask);
    build_runtime_factor_masks(session->observation.legal_mask, 0, slot0_kind_mask, slot0_move_mask, slot0_switch_mask);
    build_runtime_factor_masks(session->observation.legal_mask, 1, slot1_kind_mask, slot1_move_mask, slot1_switch_mask);
    factorized_action_choice_init(&sampled_choice);
    gru_model_forward_step(runtime->model, session->flat_observation, session->hidden_state, session->hidden_state, NULL, &value);
    if (!gru_model_evaluate_factorized_hidden(
            runtime->model,
            session->hidden_state,
            session->observation.legal_mask,
            slot0_kind_policy,
            slot0_move_policy,
            slot0_switch_policy,
            slot0_tera_policy,
            slot1_kind_policy,
            slot1_move_policy,
            slot1_switch_policy,
            slot1_tera_policy,
            &value)) {
        free(pair_policy);
        return 0;
    }
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        pair_policy[i] = slot0_kind_policy[0] * slot0_move_policy[i] * (slot0_mask[i] ? slot0_tera_policy[0] : 0.0f);
        pair_policy[4 + i] = slot0_kind_policy[0] * slot0_move_policy[i] * (slot0_mask[4 + i] ? slot0_tera_policy[1] : 0.0f);
        pair_policy[14 + i] = slot1_kind_policy[0] * slot1_move_policy[i] * (slot1_mask[i] ? slot1_tera_policy[0] : 0.0f);
        pair_policy[18 + i] = slot1_kind_policy[0] * slot1_move_policy[i] * (slot1_mask[4 + i] ? slot1_tera_policy[1] : 0.0f);
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        pair_policy[8 + i] = slot0_kind_policy[1] * slot0_switch_policy[i];
        pair_policy[22 + i] = slot1_kind_policy[1] * slot1_switch_policy[i];
    }
    if (slot0_needs_action) {
        int kind = sample_small_masked(slot0_kind_policy, slot0_kind_mask, FACTORIZED_KIND_DIM);
        sampled_choice.slot0_has_action = 1;
        if (kind == 0) {
            unsigned char tera_mask[FACTORIZED_TERA_DIM];
            int move = sample_small_masked(slot0_move_policy, slot0_move_mask, FACTORIZED_MOVE_DIM);
            sampled_choice.slot0_kind = FACTORIZED_ACTION_MOVE;
            sampled_choice.slot0_move_index = (unsigned char)(move + 1);
            tera_mask[0] = slot0_mask[move] ? 1 : 0;
            tera_mask[1] = slot0_mask[4 + move] ? 1 : 0;
            sampled_choice.slot0_use_tera = (unsigned char)(sample_small_masked(slot0_tera_policy, tera_mask, FACTORIZED_TERA_DIM) == 1 ? 1 : 0);
        } else {
            int sw = sample_small_masked(slot0_switch_policy, slot0_switch_mask, FACTORIZED_SWITCH_DIM);
            sampled_choice.slot0_kind = FACTORIZED_ACTION_SWITCH;
            sampled_choice.slot0_switch_index = (unsigned char)(sw + 1);
        }
    }
    if (slot1_needs_action) {
        int kind = sample_small_masked(slot1_kind_policy, slot1_kind_mask, FACTORIZED_KIND_DIM);
        sampled_choice.slot1_has_action = 1;
        if (kind == 0) {
            unsigned char tera_mask[FACTORIZED_TERA_DIM];
            int move = sample_small_masked(slot1_move_policy, slot1_move_mask, FACTORIZED_MOVE_DIM);
            sampled_choice.slot1_kind = FACTORIZED_ACTION_MOVE;
            sampled_choice.slot1_move_index = (unsigned char)(move + 1);
            tera_mask[0] = slot1_mask[move] ? 1 : 0;
            tera_mask[1] = slot1_mask[4 + move] ? 1 : 0;
            sampled_choice.slot1_use_tera = (unsigned char)(sample_small_masked(slot1_tera_policy, tera_mask, FACTORIZED_TERA_DIM) == 1 ? 1 : 0);
        } else {
            int sw = sample_small_masked(slot1_switch_policy, slot1_switch_mask, FACTORIZED_SWITCH_DIM);
            sampled_choice.slot1_kind = FACTORIZED_ACTION_SWITCH;
            sampled_choice.slot1_switch_index = (unsigned char)(sw + 1);
        }
    }
    if (!factorized_action_choice_to_flat_actions(&sampled_choice, &action, &action2)) {
        free(pair_policy);
        return 0;
    }
    if (!validate_or_resample_request_choice(
            &session->parsed_request,
            &session->action_mask,
            pair_policy,
            slot0_needs_action,
            (enum ObsAction)(action >= 0 ? action : OBS_A1_MOVE1),
            slot1_needs_action,
            (enum ObsAction)(action2 >= 0 ? action2 : OBS_A2_MOVE1),
            &validated)) {
        free(pair_policy);
        runtime_emit_error_json(json, sizeof(json), session->battle_id, "failed to validate or resample request choice");
        fputs(json, out);
        fputc('\n', out);
        fflush(out);
        return 0;
    }
    action = validated.slot0_has_action ? (int)validated.action0 : -1;
    action2 = validated.slot1_has_action ? (int)validated.action1 : -1;
    factorized_action_choice_from_flat_actions(&session->pending_factorized_choice, action, action2);
    session->pending_old_log_prob = 0.0f;
    if (session->pending_factorized_choice.slot0_has_action) {
        if (session->pending_factorized_choice.slot0_kind == FACTORIZED_ACTION_MOVE) {
            unsigned char tera_mask[FACTORIZED_TERA_DIM] = {
                slot0_mask[session->pending_factorized_choice.slot0_move_index - 1] ? 1 : 0,
                slot0_mask[4 + session->pending_factorized_choice.slot0_move_index - 1] ? 1 : 0
            };
            session->pending_old_log_prob += action_log_prob(slot0_kind_policy, 0);
            session->pending_old_log_prob += action_log_prob(slot0_move_policy, session->pending_factorized_choice.slot0_move_index - 1);
            session->pending_old_log_prob += masked_binary_log_prob(slot0_tera_policy, tera_mask, session->pending_factorized_choice.slot0_use_tera ? 1 : 0);
        } else {
            session->pending_old_log_prob += action_log_prob(slot0_kind_policy, 1);
            session->pending_old_log_prob += action_log_prob(slot0_switch_policy, session->pending_factorized_choice.slot0_switch_index - 1);
        }
    }
    if (session->pending_factorized_choice.slot1_has_action) {
        if (session->pending_factorized_choice.slot1_kind == FACTORIZED_ACTION_MOVE) {
            unsigned char tera_mask[FACTORIZED_TERA_DIM] = {
                slot1_mask[session->pending_factorized_choice.slot1_move_index - 1] ? 1 : 0,
                slot1_mask[4 + session->pending_factorized_choice.slot1_move_index - 1] ? 1 : 0
            };
            session->pending_old_log_prob += action_log_prob(slot1_kind_policy, 0);
            session->pending_old_log_prob += action_log_prob(slot1_move_policy, session->pending_factorized_choice.slot1_move_index - 1);
            session->pending_old_log_prob += masked_binary_log_prob(slot1_tera_policy, tera_mask, session->pending_factorized_choice.slot1_use_tera ? 1 : 0);
        } else {
            session->pending_old_log_prob += action_log_prob(slot1_kind_policy, 1);
            session->pending_old_log_prob += action_log_prob(slot1_switch_policy, session->pending_factorized_choice.slot1_switch_index - 1);
        }
    }
    session->pending_old_value = value;
    strncpy(command, validated.command, sizeof(command) - 1);
    command[sizeof(command) - 1] = '\0';
    free(pair_policy);
    session->pending_action = action;
    session->pending_action2 = action2;
    strncpy(session->pending_command, command, sizeof(session->pending_command) - 1);
    session->pending_command[sizeof(session->pending_command) - 1] = '\0';
    runtime_emit_action_json(json, sizeof(json), session->battle_id, session->last_request_id, action, action2, command);
    fputs(json, out);
    fputc('\n', out);
    fflush(out);
    return 1;
}

int env_runtime_handle_message(EnvRuntime* runtime, const RuntimeMessage* msg, FILE* out) {
    EnvSession* session;
    char json[512];
    if (!runtime || !msg) {
        return 0;
    }
    if (runtime->replay_file) {
        replay_write_runtime_message(runtime->replay_file, msg);
    }

    switch (msg->type) {
        case RUNTIME_MSG_BATTLE_START:
            session = ensure_session(runtime, msg->battle_id, msg->is_doubles);
            if (!session) {
                return 0;
            }
            session->raw_state.is_doubles = msg->is_doubles ? 1 : 0;
            session->format_known = 1;
            if (session->pending_prestart_count > 0) {
                size_t i;
                size_t count = session->pending_prestart_count;
                session->pending_prestart_count = 0;
                for (i = 0; i < count; ++i) {
                    if (!env_runtime_handle_message(runtime, &session->pending_prestart_messages[i], out)) {
                        return 0;
                    }
                }
            }
            return 1;
        case RUNTIME_MSG_EVENT:
            session = ensure_session(runtime, msg->battle_id, msg->is_doubles);
            if (!session) {
                return 0;
            }
            if (!session->format_known) {
                if (!queue_prestart_message(session, msg)) {
                    if (out) {
                        runtime_emit_error_json(json, sizeof(json), msg->battle_id, "pre-start event queue overflow");
                        fputs(json, out);
                        fputc('\n', out);
                        fflush(out);
                    }
                    return 0;
                }
                return 1;
            }
            raw_battle_state_update_from_event_line(&session->raw_state, msg->line);
            return 1;
        case RUNTIME_MSG_REQUEST:
            session = ensure_session(runtime, msg->battle_id, msg->is_doubles);
            if (!session) {
                return 0;
            }
            if (!session->format_known) {
                if (!queue_prestart_message(session, msg)) {
                    if (out) {
                        runtime_emit_error_json(json, sizeof(json), msg->battle_id, "pre-start request queue overflow");
                        fputs(json, out);
                        fputc('\n', out);
                        fflush(out);
                    }
                    return 0;
                }
                return 1;
            }
            session->last_request_id = msg->request_id;
            if (!parse_request_payload(&session->parsed_request, msg->payload, msg->request_id, session->raw_state.is_doubles)) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "parse_request_payload failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            if (!raw_battle_state_update_from_request(&session->raw_state, &session->parsed_request)) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "raw_battle_state_update_from_request failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            if (!build_action_mask_from_request(&session->action_mask, &session->parsed_request)) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "build_action_mask_from_request failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            observation_from_raw_state(&session->observation, &session->raw_state, &session->parsed_request, &session->action_mask);
            if (observation_flatten(session->flat_observation, runtime->obs_dim, &session->observation) != runtime->obs_dim) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "observation_flatten failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            if (session->parsed_request.wait) {
                session->ready_for_decision = 0;
                return 1;
            }
            {
                float step_reward = compute_request_reward(runtime, session);
                if (!episode_append(&session->episode, session->flat_observation, session->observation.legal_mask, -1, step_reward, 0)) {
                    if (out) {
                        runtime_emit_error_json(json, sizeof(json), msg->battle_id, "episode_append failed");
                        fputs(json, out);
                        fputc('\n', out);
                        fflush(out);
                    }
                    return 0;
                }
            }
            update_request_reward_snapshot(session);
            session->ready_for_decision = 1;
            return write_action(runtime, session, out);
        case RUNTIME_MSG_TERMINAL:
            session = find_session(runtime, msg->battle_id);
            if (!session) return 1;
            if (session->episode.count > 0) {
                if (runtime->reward_mode == ENV_REWARD_DENSE_ADDITIVE) {
                    session->episode.rewards[session->episode.count - 1] += msg->reward;
                } else {
                    session->episode.rewards[session->episode.count - 1] = msg->reward;
                }
                session->episode.dones[session->episode.count - 1] = 1;
            }
            clear_request_reward_snapshot(session);
            session->terminal = 1;
            if (out && !runtime->replay_only && session->episode.count > 0) {
                episode_write_json_record(out, &session->episode, session->battle_id, runtime->policy_tag);
                fflush(out);
            }
            return 1;
        case RUNTIME_MSG_DECISION:
            session = ensure_session(runtime, msg->battle_id, msg->is_doubles);
            if (!session) {
                return 0;
            }
            if (!session->format_known) {
                if (!queue_prestart_message(session, msg)) {
                    if (out) {
                        runtime_emit_error_json(json, sizeof(json), msg->battle_id, "pre-start decision queue overflow");
                        fputs(json, out);
                        fputc('\n', out);
                        fflush(out);
                    }
                    return 0;
                }
                return 1;
            }
            if (msg->accepted > 0 && session->episode.count > 0) {
                int accepted_action = msg->action;
                int accepted_action2 = msg->action2;
                if ((accepted_action < 0 || accepted_action2 < 0) && msg->command[0] != '\0') {
                    int slot0_has_action = 0;
                    int slot1_has_action = 0;
                    enum ObsAction inferred_action0 = OBS_A1_MOVE1;
                    enum ObsAction inferred_action1 = OBS_A2_MOVE1;
                    if (command_to_request_choice(msg->command, &session->parsed_request,
                            &slot0_has_action, &inferred_action0, &slot1_has_action, &inferred_action1)) {
                        runtime->accepted_label_reconstructed_count += 1;
                        if (accepted_action < 0 && slot0_has_action) {
                            accepted_action = (int)inferred_action0;
                        }
                        if (accepted_action2 < 0 && slot1_has_action) {
                            accepted_action2 = (int)inferred_action1;
                        }
                        if (!slot0_has_action && slot1_has_action &&
                                accepted_action2 >= 0 &&
                                accepted_action == accepted_action2) {
                            accepted_action = -1;
                        }
                    } else {
                        runtime->accepted_label_failed_count += 1;
                    }
                } else {
                    runtime->accepted_label_direct_count += 1;
                }
                session->episode.actions[session->episode.count - 1] = accepted_action;
                session->episode.actions2[session->episode.count - 1] = accepted_action2;
                session->episode.factorized_actions[session->episode.count - 1] = session->pending_factorized_choice;
                session->episode.old_log_probs[session->episode.count - 1] = session->pending_old_log_prob;
                session->episode.old_values[session->episode.count - 1] = session->pending_old_value;
                session->pending_action = -1;
                session->pending_action2 = -1;
                session->pending_old_log_prob = 0.0f;
                session->pending_old_value = 0.0f;
                factorized_action_choice_init(&session->pending_factorized_choice);
                session->pending_command[0] = '\0';
            } else if (msg->accepted == 0) {
                session->pending_action = -1;
                session->pending_action2 = -1;
                session->pending_old_log_prob = 0.0f;
                session->pending_old_value = 0.0f;
                factorized_action_choice_init(&session->pending_factorized_choice);
                session->pending_command[0] = '\0';
            }
            return 1;
        case RUNTIME_MSG_BATTLE_END:
            return 1;
        default:
            return 1;
    }
}
