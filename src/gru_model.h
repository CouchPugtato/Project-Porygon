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

#endif
