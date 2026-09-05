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

#define ENV_PRESTART_QUEUE_MAX 32
#define ENV_POLICY_TAG_LEN 512

typedef enum {
    ENV_REWARD_TERMINAL = 0,
    ENV_REWARD_DENSE_ADDITIVE = 1
} EnvRewardMode;

const char* env_reward_mode_name(EnvRewardMode reward_mode);

typedef struct {
    float hp_swing_weight;
    float faint_swing_weight;
    float reward_clip;
} EnvDenseRewardConfig;

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
    int format_known;
    RuntimeMessage pending_prestart_messages[ENV_PRESTART_QUEUE_MAX];
    size_t pending_prestart_count;
    int terminal;
    int ready_for_decision;
    int reward_snapshot_valid;
    float prev_self_hp_frac_sum;
    float prev_opp_hp_frac_sum;
    int prev_self_fainted_count;
    int prev_opp_fainted_count;
    int pending_action;
    int pending_action2;
    float pending_old_log_prob;
    float pending_old_value;
    FactorizedActionChoice pending_factorized_choice;
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
    EnvRewardMode reward_mode;
    EnvDenseRewardConfig dense_reward_config;
    char policy_tag[ENV_POLICY_TAG_LEN];
    size_t accepted_label_direct_count;
    size_t accepted_label_reconstructed_count;
    size_t accepted_label_failed_count;
} EnvRuntime;

int env_runtime_init(
    EnvRuntime* runtime,
    GruModel* model,
    FILE* replay_file,
    int replay_only,
    EnvRewardMode reward_mode,
    const EnvDenseRewardConfig* dense_reward_config,
    const char* policy_tag
);
void env_runtime_free(EnvRuntime* runtime);
int env_runtime_handle_message(EnvRuntime* runtime, const RuntimeMessage* msg, FILE* out);

#endif
