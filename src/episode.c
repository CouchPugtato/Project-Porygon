#include "episode.h"
#include "observation.h"

#include <stdlib.h>
#include <string.h>

static int episode_grow(Episode* episode, size_t min_capacity) {
    size_t next_capacity;
    float* new_observations;
    uint8_t* new_legal_masks;
    int* new_actions;
    int* new_actions2;
    float* new_rewards;
    uint8_t* new_dones;

    if (!episode || episode->obs_dim == 0) {
        return 0;
    }

    next_capacity = episode->capacity ? episode->capacity * 2 : 4;
    if (next_capacity < min_capacity) {
        next_capacity = min_capacity;
    }

    new_observations = (float*)malloc(next_capacity * episode->obs_dim * sizeof(float));
    new_legal_masks = (uint8_t*)malloc(next_capacity * OBS_NUM_ACTIONS * sizeof(uint8_t));
    new_actions = (int*)malloc(next_capacity * sizeof(int));
    new_actions2 = (int*)malloc(next_capacity * sizeof(int));
    new_rewards = (float*)malloc(next_capacity * sizeof(float));
    new_dones = (uint8_t*)malloc(next_capacity * sizeof(uint8_t));

    if (!new_observations || !new_legal_masks || !new_actions || !new_actions2 || !new_rewards || !new_dones) {
        free(new_observations);
        free(new_legal_masks);
        free(new_actions);
        free(new_actions2);
        free(new_rewards);
        free(new_dones);
        return 0;
    }

    if (episode->count > 0) {
        memcpy(new_observations, episode->observations, episode->count * episode->obs_dim * sizeof(float));
        memcpy(new_legal_masks, episode->legal_masks, episode->count * OBS_NUM_ACTIONS * sizeof(uint8_t));
        memcpy(new_actions, episode->actions, episode->count * sizeof(int));
        memcpy(new_actions2, episode->actions2, episode->count * sizeof(int));
        memcpy(new_rewards, episode->rewards, episode->count * sizeof(float));
        memcpy(new_dones, episode->dones, episode->count * sizeof(uint8_t));
    }

    free(episode->observations);
    free(episode->legal_masks);
    free(episode->actions);
    free(episode->actions2);
    free(episode->rewards);
    free(episode->dones);

    episode->observations = new_observations;
    episode->legal_masks = new_legal_masks;
    episode->actions = new_actions;
    episode->actions2 = new_actions2;
    episode->rewards = new_rewards;
    episode->dones = new_dones;
    episode->capacity = next_capacity;
    return 1;
}

int episode_init(Episode* episode, size_t capacity, size_t obs_dim) {
    if (!episode || obs_dim == 0) {
        return 0;
    }
    memset(episode, 0, sizeof(*episode));
    episode->obs_dim = obs_dim;
    return episode_grow(episode, capacity ? capacity : 4);
}

void episode_free(Episode* episode) {
    if (!episode) {
        return;
    }
    free(episode->observations);
    free(episode->legal_masks);
    free(episode->actions);
    free(episode->actions2);
    free(episode->rewards);
    free(episode->dones);
    memset(episode, 0, sizeof(*episode));
}

int episode_append(
    Episode* episode,
    const float* observation,
    const uint8_t* legal_mask,
    int action,
    float reward,
    uint8_t done
) {
    float* dst;
    uint8_t* legal_dst;

    if (!episode || !observation) {
        return 0;
    }
    if (episode->count == episode->capacity && !episode_grow(episode, episode->count + 1)) {
        return 0;
    }

    dst = episode->observations + (episode->count * episode->obs_dim);
    memcpy(dst, observation, episode->obs_dim * sizeof(float));
    legal_dst = episode->legal_masks + (episode->count * OBS_NUM_ACTIONS);
    if (legal_mask) {
        memcpy(legal_dst, legal_mask, OBS_NUM_ACTIONS * sizeof(uint8_t));
    } else {
        memset(legal_dst, 0, OBS_NUM_ACTIONS * sizeof(uint8_t));
    }
    episode->actions[episode->count] = action;
    episode->actions2[episode->count] = -1;
    episode->rewards[episode->count] = reward;
    episode->dones[episode->count] = done;
    episode->count += 1;
    return 1;
}

const float* episode_observation_at(const Episode* episode, size_t index) {
    if (!episode || index >= episode->count) {
        return NULL;
    }
    return episode->observations + (index * episode->obs_dim);
}
