#ifndef EPISODE_H
#define EPISODE_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "action_mapper.h"

#define EPISODE_REWARD_MODE_LEN 32

typedef struct {
    float* observations;
    uint8_t* legal_masks;
    int* actions;
    int* actions2;
    FactorizedActionChoice* factorized_actions;
    float* old_log_probs;
    float* old_values;
    float* rewards;
    uint8_t* dones;
    size_t count;
    size_t capacity;
    size_t obs_dim;
    char reward_mode[EPISODE_REWARD_MODE_LEN];
    float dense_hp_swing_weight;
    float dense_faint_swing_weight;
    float dense_reward_clip;
    int reward_config_present;
} Episode;

int episode_init(Episode* episode, size_t capacity, size_t obs_dim);
void episode_free(Episode* episode);
int episode_append(
    Episode* episode,
    const float* observation,
    const uint8_t* legal_mask,
    int action,
    float reward,
    uint8_t done
);
const float* episode_observation_at(const Episode* episode, size_t index);
int episode_write_json_record(FILE* out, const Episode* episode, const char* battle_id, const char* policy_tag);
int episode_parse_json_record(
    const char* json,
    Episode* episode,
    char* battle_id,
    size_t battle_id_len,
    char* policy_tag,
    size_t policy_tag_len
);

#endif
