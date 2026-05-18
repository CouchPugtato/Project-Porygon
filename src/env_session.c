#include "env_session.h"

#include "observation_builder.h"
#include "request_parser.h"

#include <stdlib.h>
#include <string.h>

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

int env_runtime_init(EnvRuntime* runtime, GruModel* model, FILE* replay_file, int replay_only) {
    if (!runtime || !model) {
        return 0;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->model = model;
    runtime->obs_dim = gru_model_input_dim(model);
    runtime->replay_file = replay_file;
    runtime->replay_only = replay_only;
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
    int action;
    int action2 = -1;
    int legal_count = 0;
    char command[256];
    char json[512];
    int i;
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
    if (session->parsed_request.is_doubles && session->parsed_request.active_count > 1 && !session->parsed_request.team_preview) {
        action = gru_model_select_action_range(policy, session->observation.legal_mask, OBS_A1_MOVE1, OBS_A1_SWITCH6, OBS_NUM_ACTIONS);
        action2 = gru_model_select_action_range(policy, session->observation.legal_mask, OBS_A2_MOVE1, OBS_A2_SWITCH6, OBS_NUM_ACTIONS);
        if (action < 0 || action2 < 0 ||
            !doubles_actions_to_showdown_command(command, sizeof(command),
                (enum ObsAction)action, (enum ObsAction)action2, &session->parsed_request)) {
            free(policy);
            runtime_emit_error_json(json, sizeof(json), session->battle_id, "failed to map doubles actions");
            fputs(json, out);
            fputc('\n', out);
            fflush(out);
            return 0;
        }
    } else {
        action = gru_model_select_action(policy, session->observation.legal_mask, OBS_NUM_ACTIONS);
        if (action < 0 || !action_to_showdown_command(command, sizeof(command), (enum ObsAction)action, &session->parsed_request)) {
            free(policy);
            runtime_emit_error_json(json, sizeof(json), session->battle_id, "failed to map action");
            fputs(json, out);
            fputc('\n', out);
            fflush(out);
            return 0;
        }
    }
    free(policy);
    session->episode.actions[session->episode.count - 1] = action;
    if (runtime->replay_file) {
        replay_write_decision(runtime->replay_file, session->battle_id, session->last_request_id, action, command);
    }
    runtime_emit_action_json(json, sizeof(json), session->battle_id, session->last_request_id, command);
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
            return ensure_session(runtime, msg->battle_id, msg->is_doubles) != NULL;
        case RUNTIME_MSG_EVENT:
            session = ensure_session(runtime, msg->battle_id, 1);
            if (!session) return 0;
            raw_battle_state_update_from_event_line(&session->raw_state, msg->line);
            return 1;
        case RUNTIME_MSG_REQUEST:
            session = ensure_session(runtime, msg->battle_id, 1);
            if (!session) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "failed to create or find session");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            session->last_request_id = msg->request_id;
            if (!parse_request_payload(&session->parsed_request, msg->payload, msg->request_id, 1)) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "parse_request_payload failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            raw_battle_state_update_from_request(&session->raw_state, &session->parsed_request);
            if (!build_action_mask_from_request(&session->action_mask, &session->parsed_request)) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "build_action_mask_from_request failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            observation_from_raw_state(&session->observation, &session->raw_state, &session->action_mask);
            if (observation_flatten(session->flat_observation, runtime->obs_dim, &session->observation) != runtime->obs_dim) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "observation_flatten failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            if (!episode_append(&session->episode, session->flat_observation, -1, 0.0f, 0)) {
                if (out) {
                    runtime_emit_error_json(json, sizeof(json), msg->battle_id, "episode_append failed");
                    fputs(json, out);
                    fputc('\n', out);
                    fflush(out);
                }
                return 0;
            }
            session->ready_for_decision = 1;
            return write_action(runtime, session, out);
        case RUNTIME_MSG_TERMINAL:
            session = find_session(runtime, msg->battle_id);
            if (!session) return 1;
            if (session->episode.count > 0) {
                session->episode.rewards[session->episode.count - 1] = msg->reward;
                session->episode.dones[session->episode.count - 1] = 1;
            }
            session->terminal = 1;
            return 1;
        case RUNTIME_MSG_DECISION:
            session = ensure_session(runtime, msg->battle_id, 1);
            if (!session) return 0;
            if (session->episode.count > 0) {
                session->episode.actions[session->episode.count - 1] = msg->action;
            }
            return 1;
        case RUNTIME_MSG_BATTLE_END:
            return 1;
        default:
            return 1;
    }
}
