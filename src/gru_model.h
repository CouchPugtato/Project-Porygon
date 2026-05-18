#ifndef GRU_MODEL_H
#define GRU_MODEL_H

#include <stddef.h>

typedef struct GruModel GruModel;

GruModel* gru_model_create(size_t input_dim, size_t hidden_dim, size_t num_actions);
void gru_model_destroy(GruModel* model);

size_t gru_model_input_dim(const GruModel* model);
size_t gru_model_hidden_dim(const GruModel* model);
size_t gru_model_num_actions(const GruModel* model);

void gru_model_zero_state(const GruModel* model, float* hidden_state_out);
void gru_model_forward_step(
    const GruModel* model,
    const float* input,
    const float* hidden_state_in,
    float* hidden_state_out,
    float* policy_out,
    float* value_out
);
void gru_model_forward_sequence(
    const GruModel* model,
    const float* sequence,
    size_t steps,
    float* hidden_state_io,
    float* policy_out,
    float* value_out
);
int gru_model_select_action(const float* policy, const unsigned char* legal_mask, size_t num_actions);
int gru_model_select_action_range(
    const float* policy,
    const unsigned char* legal_mask,
    size_t start_index,
    size_t end_index,
    size_t num_actions
);
int gru_model_sample_action(
    const float* policy,
    const unsigned char* legal_mask,
    size_t num_actions
);
int gru_model_sample_action_range(
    const float* policy,
    const unsigned char* legal_mask,
    size_t start_index,
    size_t end_index,
    size_t num_actions
);
void gru_model_evaluate_hidden(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    float* policy_out,
    float* value_out
);
int gru_model_supervised_update_heads(
    GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    int target_action,
    float target_value,
    float learning_rate,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
);
int gru_model_policy_gradient_update_heads(
    GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate
);
size_t gru_model_parameter_count(const GruModel* model);
int gru_model_export_parameters(const GruModel* model, float* out, size_t count);
int gru_model_import_parameters(GruModel* model, const float* in, size_t count);

#endif
