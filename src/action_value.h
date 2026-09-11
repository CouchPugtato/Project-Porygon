#ifndef ACTION_VALUE_H
#define ACTION_VALUE_H

#include <stddef.h>

#include "episode.h"
#include "gru_model.h"

typedef struct ActionValueModel ActionValueModel;

typedef struct {
    float q_value;
    float baseline_value;
    float advantage;
    float legal_action_spread;
    size_t legal_action_count;
} ActionValuePrediction;

ActionValueModel* action_value_model_create(
    size_t hidden_dim,
    size_t latent_dim,
    unsigned int seed
);
void action_value_model_destroy(ActionValueModel* model);

size_t action_value_model_hidden_dim(const ActionValueModel* model);
size_t action_value_model_latent_dim(const ActionValueModel* model);
size_t action_value_model_parameter_count(const ActionValueModel* model);
int action_value_model_export_parameters(
    const ActionValueModel* model,
    float* parameters,
    size_t count
);
int action_value_model_import_parameters(
    ActionValueModel* model,
    const float* parameters,
    size_t count
);

int action_value_model_predict(
    const ActionValueModel* model,
    const GruModel* policy_model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    const FactorizedActionChoice* choice,
    int action0,
    int action1,
    float baseline_value,
    ActionValuePrediction* prediction
);

void action_value_model_clear_gradients(ActionValueModel* model);
int action_value_model_accumulate(
    ActionValueModel* model,
    const GruModel* policy_model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    const FactorizedActionChoice* choice,
    int action0,
    int action1,
    float baseline_value,
    float target_value,
    float* loss_out
);
int action_value_model_apply_adam(
    ActionValueModel* model,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float gradient_clip,
    float l2_coefficient
);

int action_value_model_save(const char* path, const ActionValueModel* model);
ActionValueModel* action_value_model_load(const char* path, size_t expected_hidden_dim);

#endif
