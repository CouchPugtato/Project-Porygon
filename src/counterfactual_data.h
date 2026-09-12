#ifndef COUNTERFACTUAL_DATA_H
#define COUNTERFACTUAL_DATA_H

#include <stddef.h>

#include "episode.h"
#include "observation.h"

#define COUNTERFACTUAL_PAIR_ID_LEN 192
#define COUNTERFACTUAL_POLICY_TAG_LEN 512

typedef struct {
    char pair_id[COUNTERFACTUAL_PAIR_ID_LEN];
    char policy_tag[COUNTERFACTUAL_POLICY_TAG_LEN];
    int action_rank;
    int decision_index;
    int action;
    int action2;
    FactorizedActionChoice choice;
    float baseline_value;
    float target_value;
    float* hidden_state;
    unsigned char legal_mask[OBS_NUM_ACTIONS];
} CounterfactualSample;

typedef struct {
    CounterfactualSample* samples;
    size_t count;
    size_t capacity;
    size_t hidden_dim;
    size_t pair_count;
} CounterfactualDataset;

int counterfactual_dataset_load(
    CounterfactualDataset* dataset,
    const char* path,
    size_t expected_hidden_dim,
    const char* expected_policy_tag
);
void counterfactual_dataset_free(CounterfactualDataset* dataset);

#endif
