#ifndef EPISODE_H
#define EPISODE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float* observations;
    int* actions;
    float* rewards;
    uint8_t* dones;
    size_t count;
    size_t capacity;
    size_t obs_dim;
} Episode;

int episode_init(Episode* episode, size_t capacity, size_t obs_dim);
void episode_free(Episode* episode);
int episode_append(Episode* episode, const float* observation, int action, float reward, uint8_t done);
const float* episode_observation_at(const Episode* episode, size_t index);

#endif
