#include "env_session.h"

#include "observation_builder.h"
#include "request_parser.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ENV_DENSE_HP_SWING_WEIGHT 0.30f
#define ENV_DENSE_FAINT_SWING_WEIGHT 0.40f
#define ENV_DENSE_REWARD_CLIP 0.75f

typedef struct {
    float self_hp_frac_sum;
    float opp_hp_frac_sum;
    int self_fainted_count;
    int opp_fainted_count;
} EnvRewardSnapshot;

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

static float clamp_dense_reward(float reward) {
    if (reward > ENV_DENSE_REWARD_CLIP) {
        return ENV_DENSE_REWARD_CLIP;
    }
    if (reward < -ENV_DENSE_REWARD_CLIP) {
        return -ENV_DENSE_REWARD_CLIP;
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

static float compute_dense_reward_delta(const EnvRewardSnapshot* previous, const EnvRewardSnapshot* current) {
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
        ENV_DENSE_HP_SWING_WEIGHT * (opp_hp_loss - self_hp_loss) +
        ENV_DENSE_FAINT_SWING_WEIGHT * (float)(opp_faints_gained - self_faints_gained);
    return clamp_dense_reward(reward);
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
    return compute_dense_reward_delta(&previous, &current);
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

int env_runtime_init(EnvRuntime* runtime, GruModel* model, FILE* replay_file, int replay_only, EnvRewardMode reward_mode) {
    if (!runtime || !model) {
        return 0;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->model = model;
    runtime->obs_dim = gru_model_input_dim(model);
    runtime->replay_file = replay_file;
    runtime->replay_only = replay_only;
    runtime->reward_mode = reward_mode;
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
    float* policy;
    float value;
    int action = -1;
    int action2 = -1;
    int legal_count = 0;
    char command[256];
    char json[512];
    int i;
    ValidatedRequestChoice validated;
    if (!runtime || !session || !out || runtime->replay_only) {
        return 1;
    }
    policy = (float*)malloc(OBS_NUM_ACTIONS * sizeof(float));
    if (!policy) {
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
        free(policy);
        return 0;
    }
    gru_model_forward_step(runtime->model, session->flat_observation, session->hidden_state, session->hidden_state, policy, &value);
    if (!validate_or_resample_request_choice(
            &session->parsed_request,
            &session->action_mask,
            policy,
            1,
            (enum ObsAction)gru_model_sample_action_range(policy, session->observation.legal_mask, OBS_A1_MOVE1, OBS_A1_SWITCH6, OBS_NUM_ACTIONS),
            1,
            (enum ObsAction)gru_model_sample_action_range(policy, session->observation.legal_mask, OBS_A2_MOVE1, OBS_A2_SWITCH6, OBS_NUM_ACTIONS),
            &validated)) {
        free(policy);
        runtime_emit_error_json(json, sizeof(json), session->battle_id, "failed to validate or resample request choice");
        fputs(json, out);
        fputc('\n', out);
        fflush(out);
        return 0;
    }
    action = validated.slot0_has_action ? (int)validated.action0 : -1;
    action2 = validated.slot1_has_action ? (int)validated.action1 : -1;
    strncpy(command, validated.command, sizeof(command) - 1);
    command[sizeof(command) - 1] = '\0';
    free(policy);
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
                if (!episode_append(&session->episode, session->flat_observation, -1, step_reward, 0)) {
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
                session->pending_action = -1;
                session->pending_action2 = -1;
                session->pending_command[0] = '\0';
            } else if (msg->accepted == 0) {
                session->pending_action = -1;
                session->pending_action2 = -1;
                session->pending_command[0] = '\0';
            }
            return 1;
        case RUNTIME_MSG_BATTLE_END:
            return 1;
        default:
            return 1;
    }
}
