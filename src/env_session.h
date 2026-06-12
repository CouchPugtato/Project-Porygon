#ifndef ENV_SESSION_H
#define ENV_SESSION_H

#include "action_mapper.h"
#include "episode.h"
#include "gru_model.h"
#include "observation.h"
#include "raw_battle_state.h"
#include "replay_io.h"
#include "runtime_protocol.h"

#include <stdio.h>

typedef struct {
    char battle_id[RUNTIME_BATTLE_ID_LEN];
    RawBattleState raw_state;
    Observation observation;
    ActionMask action_mask;
    ParsedRequest parsed_request;
    Episode episode;
    float* hidden_state;
    float* flat_observation;
    int last_request_id;
    int terminal;
    int ready_for_decision;
    int pending_action;
    int pending_action2;
    char pending_command[RUNTIME_COMMAND_LEN];
} EnvSession;

typedef struct {
    EnvSession* sessions;
    size_t count;
    size_t capacity;
    GruModel* model;
    size_t obs_dim;
    FILE* replay_file;
    int replay_only;
    size_t accepted_label_direct_count;
    size_t accepted_label_reconstructed_count;
    size_t accepted_label_failed_count;
} EnvRuntime;

int env_runtime_init(EnvRuntime* runtime, GruModel* model, FILE* replay_file, int replay_only);
void env_runtime_free(EnvRuntime* runtime);
int env_runtime_handle_message(EnvRuntime* runtime, const RuntimeMessage* msg, FILE* out);

#endif
