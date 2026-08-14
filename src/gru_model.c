#include "gru_model.h"
#include "action_mapper.h"
#include "observation.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

typedef struct {
    size_t hidden_capacity;
    size_t action_capacity;
    float* z;
    float* r;
    float* n;
    float* gated_hidden;
    float* logits;
    float* next_hidden;
    float* transformed_input;
    float* entity_hidden;
} GruForwardScratch;

typedef struct {
    size_t step_capacity;
    size_t hidden_capacity;
    size_t input_capacity;
    size_t action_capacity;
    float* h_states;
    float* z_cache;
    float* r_cache;
    float* n_cache;
    float* gated_cache;
    float* zero_hidden;
    float* logits;
    float* policy;
    float* grad_h;
    float* next_grad_h;
    float* grad_logits;
    float* d_gated;
    float* d_pre_z;
    float* d_pre_r;
    float* d_pre_n;
    float* grad_wzx;
    float* grad_wzh;
    float* grad_bz;
    float* grad_wrx;
    float* grad_wrh;
    float* grad_br;
    float* grad_wnx;
    float* grad_wnh;
    float* grad_bn;
    float* grad_policy_head;
    float* grad_policy_bias;
    float* grad_value_head;
    float* transformed_inputs;
    float* entity_hidden;
    float* grad_input;
    float* grad_entity_encoder;
    float* grad_entity_bias;
    float* grad_entity_decoder;
} GruRecurrentScratch;

typedef struct {
    float* wzx;
    float* wzh;
    float* bz;
    float* wrx;
    float* wrh;
    float* br;
    float* wnx;
    float* wnh;
    float* bn;
    float* policy_head;
    float* policy_bias;
    float* slot0_kind_head;
    float* slot0_kind_bias;
    float* slot0_move_head;
    float* slot0_move_bias;
    float* slot0_switch_head;
    float* slot0_switch_bias;
    float* slot0_tera_head;
    float* slot0_tera_bias;
    float* slot0_target_head;
    float* slot0_target_bias;
    float* slot1_kind_head;
    float* slot1_kind_bias;
    float* slot1_move_head;
    float* slot1_move_bias;
    float* slot1_switch_head;
    float* slot1_switch_bias;
    float* slot1_tera_head;
    float* slot1_tera_bias;
    float* slot1_target_head;
    float* slot1_target_bias;
    float* joint_pair_head;
    float* joint_pair_bias;
    float* entity_encoder;
    float* entity_bias;
    float* entity_decoder;
    float* value_head;
    float value_bias;
    size_t count;
} GruGradientAccum;

typedef struct {
    float* wzx_m;
    float* wzx_v;
    float* wzh_m;
    float* wzh_v;
    float* bz_m;
    float* bz_v;
    float* wrx_m;
    float* wrx_v;
    float* wrh_m;
    float* wrh_v;
    float* br_m;
    float* br_v;
    float* wnx_m;
    float* wnx_v;
    float* wnh_m;
    float* wnh_v;
    float* bn_m;
    float* bn_v;
    float* policy_head_m;
    float* policy_head_v;
    float* policy_bias_m;
    float* policy_bias_v;
    float* slot0_kind_head_m; float* slot0_kind_head_v;
    float* slot0_kind_bias_m; float* slot0_kind_bias_v;
    float* slot0_move_head_m; float* slot0_move_head_v;
    float* slot0_move_bias_m; float* slot0_move_bias_v;
    float* slot0_switch_head_m; float* slot0_switch_head_v;
    float* slot0_switch_bias_m; float* slot0_switch_bias_v;
    float* slot0_tera_head_m; float* slot0_tera_head_v;
    float* slot0_tera_bias_m; float* slot0_tera_bias_v;
    float* slot0_target_head_m; float* slot0_target_head_v;
    float* slot0_target_bias_m; float* slot0_target_bias_v;
    float* slot1_kind_head_m; float* slot1_kind_head_v;
    float* slot1_kind_bias_m; float* slot1_kind_bias_v;
    float* slot1_move_head_m; float* slot1_move_head_v;
    float* slot1_move_bias_m; float* slot1_move_bias_v;
    float* slot1_switch_head_m; float* slot1_switch_head_v;
    float* slot1_switch_bias_m; float* slot1_switch_bias_v;
    float* slot1_tera_head_m; float* slot1_tera_head_v;
    float* slot1_tera_bias_m; float* slot1_tera_bias_v;
    float* slot1_target_head_m; float* slot1_target_head_v;
    float* slot1_target_bias_m; float* slot1_target_bias_v;
    float* joint_pair_head_m; float* joint_pair_head_v;
    float* joint_pair_bias_m; float* joint_pair_bias_v;
    float* entity_encoder_m; float* entity_encoder_v;
    float* entity_bias_m; float* entity_bias_v;
    float* entity_decoder_m; float* entity_decoder_v;
    float* value_head_m;
    float* value_head_v;
    float value_bias_m;
    float value_bias_v;
    size_t step;
} GruAdamState;

struct GruModel {
    size_t input_dim;
    size_t hidden_dim;
    size_t num_actions;

    Matrix wzx;
    Matrix wzh;
    float* bz;

    Matrix wrx;
    Matrix wrh;
    float* br;

    Matrix wnx;
    Matrix wnh;
    float* bn;

    Matrix policy_head;
    float* policy_bias;
    Matrix slot0_kind_head;
    float* slot0_kind_bias;
    Matrix slot0_move_head;
    float* slot0_move_bias;
    Matrix slot0_switch_head;
    float* slot0_switch_bias;
    Matrix slot0_tera_head;
    float* slot0_tera_bias;
    Matrix slot0_target_head;
    float* slot0_target_bias;
    Matrix slot1_kind_head;
    float* slot1_kind_bias;
    Matrix slot1_move_head;
    float* slot1_move_bias;
    Matrix slot1_switch_head;
    float* slot1_switch_bias;
    Matrix slot1_tera_head;
    float* slot1_tera_bias;
    Matrix slot1_target_head;
    float* slot1_target_bias;
    Matrix joint_pair_head;
    float* joint_pair_bias;
    Matrix entity_encoder;
    float* entity_bias;
    Matrix entity_decoder;
    int entity_encoder_enabled;

    float* value_head;
    float value_bias;
    GruForwardScratch forward_scratch;
    GruRecurrentScratch recurrent_scratch;
    GruGradientAccum grad_accum;
    GruAdamState adam_state;
};

static void bootstrap_factorized_heads_from_flat(GruModel* model);
static void sync_flat_heads_from_factorized(GruModel* model);

static float rand_uniform(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static float sigmoidf_approx(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static Matrix matrix_make(size_t rows, size_t cols, float scale) {
    Matrix m;
    size_t n;
    size_t i;

    m.rows = rows;
    m.cols = cols;
    m.data = NULL;
    n = rows * cols;
    m.data = (float*)malloc(n * sizeof(float));
    if (!m.data) {
        m.rows = 0;
        m.cols = 0;
        return m;
    }

    for (i = 0; i < n; ++i) {
        m.data[i] = rand_uniform() * scale;
    }
    return m;
}

static void matrix_free(Matrix* matrix) {
    if (!matrix) {
        return;
    }
    free(matrix->data);
    matrix->data = NULL;
    matrix->rows = 0;
    matrix->cols = 0;
}

static float* vector_make(size_t len) {
    return (float*)calloc(len, sizeof(float));
}

static void gru_forward_scratch_free(GruForwardScratch* scratch) {
    if (!scratch) {
        return;
    }
    free(scratch->z);
    free(scratch->r);
    free(scratch->n);
    free(scratch->gated_hidden);
    free(scratch->logits);
    free(scratch->next_hidden);
    free(scratch->transformed_input);
    free(scratch->entity_hidden);
    memset(scratch, 0, sizeof(*scratch));
}

static int gru_forward_scratch_ensure(GruModel* model) {
    GruForwardScratch* scratch;
    if (!model) {
        return 0;
    }
    scratch = &model->forward_scratch;
    if (scratch->hidden_capacity == model->hidden_dim && scratch->action_capacity == model->num_actions &&
            scratch->z && scratch->r && scratch->n && scratch->gated_hidden && scratch->logits && scratch->next_hidden &&
            scratch->transformed_input && scratch->entity_hidden) {
        return 1;
    }
    gru_forward_scratch_free(scratch);
    scratch->z = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->r = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->n = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->gated_hidden = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->logits = (float*)calloc(model->num_actions, sizeof(float));
    scratch->next_hidden = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->transformed_input = (float*)calloc(model->input_dim, sizeof(float));
    scratch->entity_hidden = (float*)calloc(2u * OBS_TEAM_SIZE * GRU_ENTITY_EMBED_DIM, sizeof(float));
    if (!scratch->z || !scratch->r || !scratch->n || !scratch->gated_hidden || !scratch->logits || !scratch->next_hidden ||
            !scratch->transformed_input || !scratch->entity_hidden) {
        gru_forward_scratch_free(scratch);
        return 0;
    }
    scratch->hidden_capacity = model->hidden_dim;
    scratch->action_capacity = model->num_actions;
    return 1;
}

static void gru_recurrent_scratch_free(GruRecurrentScratch* scratch) {
    if (!scratch) {
        return;
    }
    free(scratch->h_states);
    free(scratch->z_cache);
    free(scratch->r_cache);
    free(scratch->n_cache);
    free(scratch->gated_cache);
    free(scratch->zero_hidden);
    free(scratch->logits);
    free(scratch->policy);
    free(scratch->grad_h);
    free(scratch->next_grad_h);
    free(scratch->grad_logits);
    free(scratch->d_gated);
    free(scratch->d_pre_z);
    free(scratch->d_pre_r);
    free(scratch->d_pre_n);
    free(scratch->grad_wzx);
    free(scratch->grad_wzh);
    free(scratch->grad_bz);
    free(scratch->grad_wrx);
    free(scratch->grad_wrh);
    free(scratch->grad_br);
    free(scratch->grad_wnx);
    free(scratch->grad_wnh);
    free(scratch->grad_bn);
    free(scratch->grad_policy_head);
    free(scratch->grad_policy_bias);
    free(scratch->grad_value_head);
    free(scratch->transformed_inputs);
    free(scratch->entity_hidden);
    free(scratch->grad_input);
    free(scratch->grad_entity_encoder);
    free(scratch->grad_entity_bias);
    free(scratch->grad_entity_decoder);
    memset(scratch, 0, sizeof(*scratch));
}

static int gru_recurrent_scratch_ensure(GruModel* model, size_t steps) {
    GruRecurrentScratch* scratch;
    if (!model || steps == 0) {
        return 0;
    }
    scratch = &model->recurrent_scratch;
    if (scratch->step_capacity >= steps &&
            scratch->hidden_capacity == model->hidden_dim &&
            scratch->input_capacity == model->input_dim &&
            scratch->action_capacity == model->num_actions &&
            scratch->h_states && scratch->z_cache && scratch->r_cache && scratch->n_cache && scratch->gated_cache &&
            scratch->zero_hidden && scratch->logits && scratch->policy && scratch->grad_h && scratch->next_grad_h &&
            scratch->grad_logits && scratch->d_gated && scratch->d_pre_z && scratch->d_pre_r && scratch->d_pre_n &&
            scratch->grad_wzx && scratch->grad_wzh && scratch->grad_bz && scratch->grad_wrx && scratch->grad_wrh &&
            scratch->grad_br && scratch->grad_wnx && scratch->grad_wnh && scratch->grad_bn &&
            scratch->grad_policy_head && scratch->grad_policy_bias && scratch->grad_value_head &&
            scratch->transformed_inputs && scratch->entity_hidden && scratch->grad_input &&
            scratch->grad_entity_encoder && scratch->grad_entity_bias && scratch->grad_entity_decoder) {
        return 1;
    }
    gru_recurrent_scratch_free(scratch);
    scratch->h_states = (float*)calloc(steps * model->hidden_dim, sizeof(float));
    scratch->z_cache = (float*)calloc(steps * model->hidden_dim, sizeof(float));
    scratch->r_cache = (float*)calloc(steps * model->hidden_dim, sizeof(float));
    scratch->n_cache = (float*)calloc(steps * model->hidden_dim, sizeof(float));
    scratch->gated_cache = (float*)calloc(steps * model->hidden_dim, sizeof(float));
    scratch->zero_hidden = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->logits = (float*)calloc(model->num_actions, sizeof(float));
    scratch->policy = (float*)calloc(model->num_actions, sizeof(float));
    scratch->grad_h = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->next_grad_h = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->grad_logits = (float*)calloc(model->num_actions, sizeof(float));
    scratch->d_gated = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->d_pre_z = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->d_pre_r = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->d_pre_n = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->grad_wzx = (float*)calloc(model->wzx.rows * model->wzx.cols, sizeof(float));
    scratch->grad_wzh = (float*)calloc(model->wzh.rows * model->wzh.cols, sizeof(float));
    scratch->grad_bz = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->grad_wrx = (float*)calloc(model->wrx.rows * model->wrx.cols, sizeof(float));
    scratch->grad_wrh = (float*)calloc(model->wrh.rows * model->wrh.cols, sizeof(float));
    scratch->grad_br = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->grad_wnx = (float*)calloc(model->wnx.rows * model->wnx.cols, sizeof(float));
    scratch->grad_wnh = (float*)calloc(model->wnh.rows * model->wnh.cols, sizeof(float));
    scratch->grad_bn = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->grad_policy_head = (float*)calloc(model->policy_head.rows * model->policy_head.cols, sizeof(float));
    scratch->grad_policy_bias = (float*)calloc(model->num_actions, sizeof(float));
    scratch->grad_value_head = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->transformed_inputs = (float*)calloc(steps * model->input_dim, sizeof(float));
    scratch->entity_hidden = (float*)calloc(steps * 2u * OBS_TEAM_SIZE * GRU_ENTITY_EMBED_DIM, sizeof(float));
    scratch->grad_input = (float*)calloc(model->input_dim, sizeof(float));
    scratch->grad_entity_encoder = (float*)calloc(model->entity_encoder_enabled ? model->entity_encoder.rows * model->entity_encoder.cols : 1u, sizeof(float));
    scratch->grad_entity_bias = (float*)calloc(model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 1u, sizeof(float));
    scratch->grad_entity_decoder = (float*)calloc(model->entity_encoder_enabled ? model->entity_decoder.rows * model->entity_decoder.cols : 1u, sizeof(float));
    if (!scratch->h_states || !scratch->z_cache || !scratch->r_cache || !scratch->n_cache || !scratch->gated_cache ||
            !scratch->zero_hidden || !scratch->logits || !scratch->policy || !scratch->grad_h ||
            !scratch->next_grad_h || !scratch->grad_logits || !scratch->d_gated || !scratch->d_pre_z ||
            !scratch->d_pre_r || !scratch->d_pre_n || !scratch->grad_wzx || !scratch->grad_wzh ||
            !scratch->grad_bz || !scratch->grad_wrx || !scratch->grad_wrh || !scratch->grad_br ||
            !scratch->grad_wnx || !scratch->grad_wnh || !scratch->grad_bn || !scratch->grad_policy_head ||
            !scratch->grad_policy_bias || !scratch->grad_value_head || !scratch->transformed_inputs ||
            !scratch->entity_hidden || !scratch->grad_input || !scratch->grad_entity_encoder ||
            !scratch->grad_entity_bias || !scratch->grad_entity_decoder) {
        gru_recurrent_scratch_free(scratch);
        return 0;
    }
    scratch->step_capacity = steps;
    scratch->hidden_capacity = model->hidden_dim;
    scratch->input_capacity = model->input_dim;
    scratch->action_capacity = model->num_actions;
    return 1;
}

static void gru_gradient_accum_free(GruGradientAccum* accum) {
    if (!accum) {
        return;
    }
    free(accum->wzx);
    free(accum->wzh);
    free(accum->bz);
    free(accum->wrx);
    free(accum->wrh);
    free(accum->br);
    free(accum->wnx);
    free(accum->wnh);
    free(accum->bn);
    free(accum->policy_head);
    free(accum->policy_bias);
    free(accum->slot0_kind_head); free(accum->slot0_kind_bias);
    free(accum->slot0_move_head); free(accum->slot0_move_bias);
    free(accum->slot0_switch_head); free(accum->slot0_switch_bias);
    free(accum->slot0_tera_head); free(accum->slot0_tera_bias);
    free(accum->slot0_target_head); free(accum->slot0_target_bias);
    free(accum->slot1_kind_head); free(accum->slot1_kind_bias);
    free(accum->slot1_move_head); free(accum->slot1_move_bias);
    free(accum->slot1_switch_head); free(accum->slot1_switch_bias);
    free(accum->slot1_tera_head); free(accum->slot1_tera_bias);
    free(accum->slot1_target_head); free(accum->slot1_target_bias);
    free(accum->joint_pair_head); free(accum->joint_pair_bias);
    free(accum->entity_encoder); free(accum->entity_bias); free(accum->entity_decoder);
    free(accum->value_head);
    memset(accum, 0, sizeof(*accum));
}

static void gru_adam_state_free(GruAdamState* state) {
    if (!state) {
        return;
    }
    free(state->wzx_m); free(state->wzx_v);
    free(state->wzh_m); free(state->wzh_v);
    free(state->bz_m); free(state->bz_v);
    free(state->wrx_m); free(state->wrx_v);
    free(state->wrh_m); free(state->wrh_v);
    free(state->br_m); free(state->br_v);
    free(state->wnx_m); free(state->wnx_v);
    free(state->wnh_m); free(state->wnh_v);
    free(state->bn_m); free(state->bn_v);
    free(state->policy_head_m); free(state->policy_head_v);
    free(state->policy_bias_m); free(state->policy_bias_v);
    free(state->slot0_kind_head_m); free(state->slot0_kind_head_v);
    free(state->slot0_kind_bias_m); free(state->slot0_kind_bias_v);
    free(state->slot0_move_head_m); free(state->slot0_move_head_v);
    free(state->slot0_move_bias_m); free(state->slot0_move_bias_v);
    free(state->slot0_switch_head_m); free(state->slot0_switch_head_v);
    free(state->slot0_switch_bias_m); free(state->slot0_switch_bias_v);
    free(state->slot0_tera_head_m); free(state->slot0_tera_head_v);
    free(state->slot0_tera_bias_m); free(state->slot0_tera_bias_v);
    free(state->slot0_target_head_m); free(state->slot0_target_head_v);
    free(state->slot0_target_bias_m); free(state->slot0_target_bias_v);
    free(state->slot1_kind_head_m); free(state->slot1_kind_head_v);
    free(state->slot1_kind_bias_m); free(state->slot1_kind_bias_v);
    free(state->slot1_move_head_m); free(state->slot1_move_head_v);
    free(state->slot1_move_bias_m); free(state->slot1_move_bias_v);
    free(state->slot1_switch_head_m); free(state->slot1_switch_head_v);
    free(state->slot1_switch_bias_m); free(state->slot1_switch_bias_v);
    free(state->slot1_tera_head_m); free(state->slot1_tera_head_v);
    free(state->slot1_tera_bias_m); free(state->slot1_tera_bias_v);
    free(state->slot1_target_head_m); free(state->slot1_target_head_v);
    free(state->slot1_target_bias_m); free(state->slot1_target_bias_v);
    free(state->joint_pair_head_m); free(state->joint_pair_head_v);
    free(state->joint_pair_bias_m); free(state->joint_pair_bias_v);
    free(state->entity_encoder_m); free(state->entity_encoder_v);
    free(state->entity_bias_m); free(state->entity_bias_v);
    free(state->entity_decoder_m); free(state->entity_decoder_v);
    free(state->value_head_m); free(state->value_head_v);
    memset(state, 0, sizeof(*state));
}

static int gru_adam_state_ensure(GruModel* model) {
    GruAdamState* state;
    if (!model) {
        return 0;
    }
    state = &model->adam_state;
    if (state->wzx_m && state->wzx_v && state->wzh_m && state->wzh_v && state->bz_m && state->bz_v &&
            state->wrx_m && state->wrx_v && state->wrh_m && state->wrh_v && state->br_m && state->br_v &&
            state->wnx_m && state->wnx_v && state->wnh_m && state->wnh_v && state->bn_m && state->bn_v &&
            state->policy_head_m && state->policy_head_v && state->policy_bias_m && state->policy_bias_v &&
            state->slot0_kind_head_m && state->slot0_kind_head_v && state->slot0_kind_bias_m && state->slot0_kind_bias_v &&
            state->slot0_move_head_m && state->slot0_move_head_v && state->slot0_move_bias_m && state->slot0_move_bias_v &&
            state->slot0_switch_head_m && state->slot0_switch_head_v && state->slot0_switch_bias_m && state->slot0_switch_bias_v &&
            state->slot0_tera_head_m && state->slot0_tera_head_v && state->slot0_tera_bias_m && state->slot0_tera_bias_v &&
            state->slot0_target_head_m && state->slot0_target_head_v && state->slot0_target_bias_m && state->slot0_target_bias_v &&
            state->slot1_kind_head_m && state->slot1_kind_head_v && state->slot1_kind_bias_m && state->slot1_kind_bias_v &&
            state->slot1_move_head_m && state->slot1_move_head_v && state->slot1_move_bias_m && state->slot1_move_bias_v &&
            state->slot1_switch_head_m && state->slot1_switch_head_v && state->slot1_switch_bias_m && state->slot1_switch_bias_v &&
            state->slot1_tera_head_m && state->slot1_tera_head_v && state->slot1_tera_bias_m && state->slot1_tera_bias_v &&
            state->slot1_target_head_m && state->slot1_target_head_v && state->slot1_target_bias_m && state->slot1_target_bias_v &&
            state->joint_pair_head_m && state->joint_pair_head_v && state->joint_pair_bias_m && state->joint_pair_bias_v &&
            state->entity_encoder_m && state->entity_encoder_v && state->entity_bias_m && state->entity_bias_v &&
            state->entity_decoder_m && state->entity_decoder_v &&
            state->value_head_m && state->value_head_v) {
        return 1;
    }
    gru_adam_state_free(state);
    state->wzx_m = (float*)calloc(model->wzx.rows * model->wzx.cols, sizeof(float));
    state->wzx_v = (float*)calloc(model->wzx.rows * model->wzx.cols, sizeof(float));
    state->wzh_m = (float*)calloc(model->wzh.rows * model->wzh.cols, sizeof(float));
    state->wzh_v = (float*)calloc(model->wzh.rows * model->wzh.cols, sizeof(float));
    state->bz_m = (float*)calloc(model->hidden_dim, sizeof(float));
    state->bz_v = (float*)calloc(model->hidden_dim, sizeof(float));
    state->wrx_m = (float*)calloc(model->wrx.rows * model->wrx.cols, sizeof(float));
    state->wrx_v = (float*)calloc(model->wrx.rows * model->wrx.cols, sizeof(float));
    state->wrh_m = (float*)calloc(model->wrh.rows * model->wrh.cols, sizeof(float));
    state->wrh_v = (float*)calloc(model->wrh.rows * model->wrh.cols, sizeof(float));
    state->br_m = (float*)calloc(model->hidden_dim, sizeof(float));
    state->br_v = (float*)calloc(model->hidden_dim, sizeof(float));
    state->wnx_m = (float*)calloc(model->wnx.rows * model->wnx.cols, sizeof(float));
    state->wnx_v = (float*)calloc(model->wnx.rows * model->wnx.cols, sizeof(float));
    state->wnh_m = (float*)calloc(model->wnh.rows * model->wnh.cols, sizeof(float));
    state->wnh_v = (float*)calloc(model->wnh.rows * model->wnh.cols, sizeof(float));
    state->bn_m = (float*)calloc(model->hidden_dim, sizeof(float));
    state->bn_v = (float*)calloc(model->hidden_dim, sizeof(float));
    state->policy_head_m = (float*)calloc(model->policy_head.rows * model->policy_head.cols, sizeof(float));
    state->policy_head_v = (float*)calloc(model->policy_head.rows * model->policy_head.cols, sizeof(float));
    state->policy_bias_m = (float*)calloc(model->num_actions, sizeof(float));
    state->policy_bias_v = (float*)calloc(model->num_actions, sizeof(float));
    state->slot0_kind_head_m = (float*)calloc(model->slot0_kind_head.rows * model->slot0_kind_head.cols, sizeof(float));
    state->slot0_kind_head_v = (float*)calloc(model->slot0_kind_head.rows * model->slot0_kind_head.cols, sizeof(float));
    state->slot0_kind_bias_m = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    state->slot0_kind_bias_v = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    state->slot0_move_head_m = (float*)calloc(model->slot0_move_head.rows * model->slot0_move_head.cols, sizeof(float));
    state->slot0_move_head_v = (float*)calloc(model->slot0_move_head.rows * model->slot0_move_head.cols, sizeof(float));
    state->slot0_move_bias_m = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    state->slot0_move_bias_v = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    state->slot0_switch_head_m = (float*)calloc(model->slot0_switch_head.rows * model->slot0_switch_head.cols, sizeof(float));
    state->slot0_switch_head_v = (float*)calloc(model->slot0_switch_head.rows * model->slot0_switch_head.cols, sizeof(float));
    state->slot0_switch_bias_m = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    state->slot0_switch_bias_v = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    state->slot0_tera_head_m = (float*)calloc(model->slot0_tera_head.rows * model->slot0_tera_head.cols, sizeof(float));
    state->slot0_tera_head_v = (float*)calloc(model->slot0_tera_head.rows * model->slot0_tera_head.cols, sizeof(float));
    state->slot0_tera_bias_m = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    state->slot0_tera_bias_v = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    state->slot0_target_head_m = (float*)calloc(model->slot0_target_head.rows * model->slot0_target_head.cols, sizeof(float));
    state->slot0_target_head_v = (float*)calloc(model->slot0_target_head.rows * model->slot0_target_head.cols, sizeof(float));
    state->slot0_target_bias_m = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    state->slot0_target_bias_v = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    state->slot1_kind_head_m = (float*)calloc(model->slot1_kind_head.rows * model->slot1_kind_head.cols, sizeof(float));
    state->slot1_kind_head_v = (float*)calloc(model->slot1_kind_head.rows * model->slot1_kind_head.cols, sizeof(float));
    state->slot1_kind_bias_m = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    state->slot1_kind_bias_v = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    state->slot1_move_head_m = (float*)calloc(model->slot1_move_head.rows * model->slot1_move_head.cols, sizeof(float));
    state->slot1_move_head_v = (float*)calloc(model->slot1_move_head.rows * model->slot1_move_head.cols, sizeof(float));
    state->slot1_move_bias_m = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    state->slot1_move_bias_v = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    state->slot1_switch_head_m = (float*)calloc(model->slot1_switch_head.rows * model->slot1_switch_head.cols, sizeof(float));
    state->slot1_switch_head_v = (float*)calloc(model->slot1_switch_head.rows * model->slot1_switch_head.cols, sizeof(float));
    state->slot1_switch_bias_m = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    state->slot1_switch_bias_v = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    state->slot1_tera_head_m = (float*)calloc(model->slot1_tera_head.rows * model->slot1_tera_head.cols, sizeof(float));
    state->slot1_tera_head_v = (float*)calloc(model->slot1_tera_head.rows * model->slot1_tera_head.cols, sizeof(float));
    state->slot1_tera_bias_m = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    state->slot1_tera_bias_v = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    state->slot1_target_head_m = (float*)calloc(model->slot1_target_head.rows * model->slot1_target_head.cols, sizeof(float));
    state->slot1_target_head_v = (float*)calloc(model->slot1_target_head.rows * model->slot1_target_head.cols, sizeof(float));
    state->slot1_target_bias_m = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    state->slot1_target_bias_v = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    state->joint_pair_head_m = (float*)calloc(model->joint_pair_head.rows * model->joint_pair_head.cols, sizeof(float));
    state->joint_pair_head_v = (float*)calloc(model->joint_pair_head.rows * model->joint_pair_head.cols, sizeof(float));
    state->joint_pair_bias_m = (float*)calloc(FACTORIZED_PAIR_DIM, sizeof(float));
    state->joint_pair_bias_v = (float*)calloc(FACTORIZED_PAIR_DIM, sizeof(float));
    state->entity_encoder_m = (float*)calloc(model->entity_encoder_enabled ? model->entity_encoder.rows * model->entity_encoder.cols : 1u, sizeof(float));
    state->entity_encoder_v = (float*)calloc(model->entity_encoder_enabled ? model->entity_encoder.rows * model->entity_encoder.cols : 1u, sizeof(float));
    state->entity_bias_m = (float*)calloc(model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 1u, sizeof(float));
    state->entity_bias_v = (float*)calloc(model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 1u, sizeof(float));
    state->entity_decoder_m = (float*)calloc(model->entity_encoder_enabled ? model->entity_decoder.rows * model->entity_decoder.cols : 1u, sizeof(float));
    state->entity_decoder_v = (float*)calloc(model->entity_encoder_enabled ? model->entity_decoder.rows * model->entity_decoder.cols : 1u, sizeof(float));
    state->value_head_m = (float*)calloc(model->hidden_dim, sizeof(float));
    state->value_head_v = (float*)calloc(model->hidden_dim, sizeof(float));
    if (!state->wzx_m || !state->wzx_v || !state->wzh_m || !state->wzh_v || !state->bz_m || !state->bz_v ||
            !state->wrx_m || !state->wrx_v || !state->wrh_m || !state->wrh_v || !state->br_m || !state->br_v ||
            !state->wnx_m || !state->wnx_v || !state->wnh_m || !state->wnh_v || !state->bn_m || !state->bn_v ||
            !state->policy_head_m || !state->policy_head_v || !state->policy_bias_m || !state->policy_bias_v ||
            !state->slot0_kind_head_m || !state->slot0_kind_head_v || !state->slot0_kind_bias_m || !state->slot0_kind_bias_v ||
            !state->slot0_move_head_m || !state->slot0_move_head_v || !state->slot0_move_bias_m || !state->slot0_move_bias_v ||
            !state->slot0_switch_head_m || !state->slot0_switch_head_v || !state->slot0_switch_bias_m || !state->slot0_switch_bias_v ||
            !state->slot0_tera_head_m || !state->slot0_tera_head_v || !state->slot0_tera_bias_m || !state->slot0_tera_bias_v ||
            !state->slot0_target_head_m || !state->slot0_target_head_v || !state->slot0_target_bias_m || !state->slot0_target_bias_v ||
            !state->slot1_kind_head_m || !state->slot1_kind_head_v || !state->slot1_kind_bias_m || !state->slot1_kind_bias_v ||
            !state->slot1_move_head_m || !state->slot1_move_head_v || !state->slot1_move_bias_m || !state->slot1_move_bias_v ||
            !state->slot1_switch_head_m || !state->slot1_switch_head_v || !state->slot1_switch_bias_m || !state->slot1_switch_bias_v ||
            !state->slot1_tera_head_m || !state->slot1_tera_head_v || !state->slot1_tera_bias_m || !state->slot1_tera_bias_v ||
            !state->slot1_target_head_m || !state->slot1_target_head_v || !state->slot1_target_bias_m || !state->slot1_target_bias_v ||
            !state->joint_pair_head_m || !state->joint_pair_head_v || !state->joint_pair_bias_m || !state->joint_pair_bias_v ||
            !state->entity_encoder_m || !state->entity_encoder_v || !state->entity_bias_m || !state->entity_bias_v ||
            !state->entity_decoder_m || !state->entity_decoder_v ||
            !state->value_head_m || !state->value_head_v) {
        gru_adam_state_free(state);
        return 0;
    }
    return 1;
}

static int gru_gradient_accum_ensure(GruModel* model) {
    GruGradientAccum* accum;
    if (!model) {
        return 0;
    }
    accum = &model->grad_accum;
    if (accum->wzx && accum->wzh && accum->bz && accum->wrx && accum->wrh && accum->br &&
            accum->wnx && accum->wnh && accum->bn && accum->policy_head && accum->policy_bias &&
            accum->slot0_kind_head && accum->slot0_kind_bias && accum->slot0_move_head && accum->slot0_move_bias &&
            accum->slot0_switch_head && accum->slot0_switch_bias && accum->slot0_tera_head && accum->slot0_tera_bias &&
            accum->slot0_target_head && accum->slot0_target_bias &&
            accum->slot1_kind_head && accum->slot1_kind_bias && accum->slot1_move_head && accum->slot1_move_bias &&
            accum->slot1_switch_head && accum->slot1_switch_bias && accum->slot1_tera_head && accum->slot1_tera_bias &&
            accum->slot1_target_head && accum->slot1_target_bias &&
            accum->joint_pair_head && accum->joint_pair_bias &&
            accum->entity_encoder && accum->entity_bias && accum->entity_decoder &&
            accum->value_head) {
        return 1;
    }
    gru_gradient_accum_free(accum);
    accum->wzx = (float*)calloc(model->wzx.rows * model->wzx.cols, sizeof(float));
    accum->wzh = (float*)calloc(model->wzh.rows * model->wzh.cols, sizeof(float));
    accum->bz = (float*)calloc(model->hidden_dim, sizeof(float));
    accum->wrx = (float*)calloc(model->wrx.rows * model->wrx.cols, sizeof(float));
    accum->wrh = (float*)calloc(model->wrh.rows * model->wrh.cols, sizeof(float));
    accum->br = (float*)calloc(model->hidden_dim, sizeof(float));
    accum->wnx = (float*)calloc(model->wnx.rows * model->wnx.cols, sizeof(float));
    accum->wnh = (float*)calloc(model->wnh.rows * model->wnh.cols, sizeof(float));
    accum->bn = (float*)calloc(model->hidden_dim, sizeof(float));
    accum->policy_head = (float*)calloc(model->policy_head.rows * model->policy_head.cols, sizeof(float));
    accum->policy_bias = (float*)calloc(model->num_actions, sizeof(float));
    accum->slot0_kind_head = (float*)calloc(model->slot0_kind_head.rows * model->slot0_kind_head.cols, sizeof(float));
    accum->slot0_kind_bias = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    accum->slot0_move_head = (float*)calloc(model->slot0_move_head.rows * model->slot0_move_head.cols, sizeof(float));
    accum->slot0_move_bias = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    accum->slot0_switch_head = (float*)calloc(model->slot0_switch_head.rows * model->slot0_switch_head.cols, sizeof(float));
    accum->slot0_switch_bias = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    accum->slot0_tera_head = (float*)calloc(model->slot0_tera_head.rows * model->slot0_tera_head.cols, sizeof(float));
    accum->slot0_tera_bias = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    accum->slot0_target_head = (float*)calloc(model->slot0_target_head.rows * model->slot0_target_head.cols, sizeof(float));
    accum->slot0_target_bias = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    accum->slot1_kind_head = (float*)calloc(model->slot1_kind_head.rows * model->slot1_kind_head.cols, sizeof(float));
    accum->slot1_kind_bias = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    accum->slot1_move_head = (float*)calloc(model->slot1_move_head.rows * model->slot1_move_head.cols, sizeof(float));
    accum->slot1_move_bias = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    accum->slot1_switch_head = (float*)calloc(model->slot1_switch_head.rows * model->slot1_switch_head.cols, sizeof(float));
    accum->slot1_switch_bias = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    accum->slot1_tera_head = (float*)calloc(model->slot1_tera_head.rows * model->slot1_tera_head.cols, sizeof(float));
    accum->slot1_tera_bias = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    accum->slot1_target_head = (float*)calloc(model->slot1_target_head.rows * model->slot1_target_head.cols, sizeof(float));
    accum->slot1_target_bias = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    accum->joint_pair_head = (float*)calloc(model->joint_pair_head.rows * model->joint_pair_head.cols, sizeof(float));
    accum->joint_pair_bias = (float*)calloc(FACTORIZED_PAIR_DIM, sizeof(float));
    accum->entity_encoder = (float*)calloc(model->entity_encoder_enabled ? model->entity_encoder.rows * model->entity_encoder.cols : 1u, sizeof(float));
    accum->entity_bias = (float*)calloc(model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 1u, sizeof(float));
    accum->entity_decoder = (float*)calloc(model->entity_encoder_enabled ? model->entity_decoder.rows * model->entity_decoder.cols : 1u, sizeof(float));
    accum->value_head = (float*)calloc(model->hidden_dim, sizeof(float));
    if (!accum->wzx || !accum->wzh || !accum->bz || !accum->wrx || !accum->wrh || !accum->br ||
            !accum->wnx || !accum->wnh || !accum->bn || !accum->policy_head || !accum->policy_bias ||
            !accum->slot0_kind_head || !accum->slot0_kind_bias || !accum->slot0_move_head || !accum->slot0_move_bias ||
            !accum->slot0_switch_head || !accum->slot0_switch_bias || !accum->slot0_tera_head || !accum->slot0_tera_bias ||
            !accum->slot0_target_head || !accum->slot0_target_bias ||
            !accum->slot1_kind_head || !accum->slot1_kind_bias || !accum->slot1_move_head || !accum->slot1_move_bias ||
            !accum->slot1_switch_head || !accum->slot1_switch_bias || !accum->slot1_tera_head || !accum->slot1_tera_bias ||
            !accum->slot1_target_head || !accum->slot1_target_bias ||
            !accum->joint_pair_head || !accum->joint_pair_bias ||
            !accum->entity_encoder || !accum->entity_bias || !accum->entity_decoder ||
            !accum->value_head) {
        gru_gradient_accum_free(accum);
        return 0;
    }
    return 1;
}

void gru_model_clear_accumulated_supervised_updates(GruModel* model) {
    GruGradientAccum* accum;
    if (!model || !gru_gradient_accum_ensure(model)) {
        return;
    }
    accum = &model->grad_accum;
    memset(accum->wzx, 0, model->wzx.rows * model->wzx.cols * sizeof(float));
    memset(accum->wzh, 0, model->wzh.rows * model->wzh.cols * sizeof(float));
    memset(accum->bz, 0, model->hidden_dim * sizeof(float));
    memset(accum->wrx, 0, model->wrx.rows * model->wrx.cols * sizeof(float));
    memset(accum->wrh, 0, model->wrh.rows * model->wrh.cols * sizeof(float));
    memset(accum->br, 0, model->hidden_dim * sizeof(float));
    memset(accum->wnx, 0, model->wnx.rows * model->wnx.cols * sizeof(float));
    memset(accum->wnh, 0, model->wnh.rows * model->wnh.cols * sizeof(float));
    memset(accum->bn, 0, model->hidden_dim * sizeof(float));
    memset(accum->policy_head, 0, model->policy_head.rows * model->policy_head.cols * sizeof(float));
    memset(accum->policy_bias, 0, model->num_actions * sizeof(float));
    memset(accum->slot0_kind_head, 0, model->slot0_kind_head.rows * model->slot0_kind_head.cols * sizeof(float));
    memset(accum->slot0_kind_bias, 0, FACTORIZED_KIND_DIM * sizeof(float));
    memset(accum->slot0_move_head, 0, model->slot0_move_head.rows * model->slot0_move_head.cols * sizeof(float));
    memset(accum->slot0_move_bias, 0, FACTORIZED_MOVE_DIM * sizeof(float));
    memset(accum->slot0_switch_head, 0, model->slot0_switch_head.rows * model->slot0_switch_head.cols * sizeof(float));
    memset(accum->slot0_switch_bias, 0, FACTORIZED_SWITCH_DIM * sizeof(float));
    memset(accum->slot0_tera_head, 0, model->slot0_tera_head.rows * model->slot0_tera_head.cols * sizeof(float));
    memset(accum->slot0_tera_bias, 0, FACTORIZED_TERA_DIM * sizeof(float));
    memset(accum->slot0_target_head, 0, model->slot0_target_head.rows * model->slot0_target_head.cols * sizeof(float));
    memset(accum->slot0_target_bias, 0, FACTORIZED_TARGET_DIM * sizeof(float));
    memset(accum->slot1_kind_head, 0, model->slot1_kind_head.rows * model->slot1_kind_head.cols * sizeof(float));
    memset(accum->slot1_kind_bias, 0, FACTORIZED_KIND_DIM * sizeof(float));
    memset(accum->slot1_move_head, 0, model->slot1_move_head.rows * model->slot1_move_head.cols * sizeof(float));
    memset(accum->slot1_move_bias, 0, FACTORIZED_MOVE_DIM * sizeof(float));
    memset(accum->slot1_switch_head, 0, model->slot1_switch_head.rows * model->slot1_switch_head.cols * sizeof(float));
    memset(accum->slot1_switch_bias, 0, FACTORIZED_SWITCH_DIM * sizeof(float));
    memset(accum->slot1_tera_head, 0, model->slot1_tera_head.rows * model->slot1_tera_head.cols * sizeof(float));
    memset(accum->slot1_tera_bias, 0, FACTORIZED_TERA_DIM * sizeof(float));
    memset(accum->slot1_target_head, 0, model->slot1_target_head.rows * model->slot1_target_head.cols * sizeof(float));
    memset(accum->slot1_target_bias, 0, FACTORIZED_TARGET_DIM * sizeof(float));
    memset(accum->joint_pair_head, 0, model->joint_pair_head.rows * model->joint_pair_head.cols * sizeof(float));
    memset(accum->joint_pair_bias, 0, FACTORIZED_PAIR_DIM * sizeof(float));
    if (model->entity_encoder_enabled) {
        memset(accum->entity_encoder, 0, model->entity_encoder.rows * model->entity_encoder.cols * sizeof(float));
        memset(accum->entity_bias, 0, GRU_ENTITY_EMBED_DIM * sizeof(float));
        memset(accum->entity_decoder, 0, model->entity_decoder.rows * model->entity_decoder.cols * sizeof(float));
    }
    memset(accum->value_head, 0, model->hidden_dim * sizeof(float));
    accum->value_bias = 0.0f;
    accum->count = 0;
}

int gru_model_apply_accumulated_supervised_updates(GruModel* model, float learning_rate) {
    size_t i;
    float scale;
    GruGradientAccum* accum;
    if (!model || !gru_gradient_accum_ensure(model)) {
        return 0;
    }
    accum = &model->grad_accum;
    if (accum->count == 0) {
        return 1;
    }
    scale = learning_rate / (float)accum->count;
#ifdef _OPENMP
    #pragma omp parallel for if(model->wzx.rows * model->wzx.cols >= 512)
#endif
    for (i = 0; i < model->wzx.rows * model->wzx.cols; ++i) model->wzx.data[i] -= scale * accum->wzx[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->wzh.rows * model->wzh.cols >= 512)
#endif
    for (i = 0; i < model->wzh.rows * model->wzh.cols; ++i) model->wzh.data[i] -= scale * accum->wzh[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->hidden_dim >= 64)
#endif
    for (i = 0; i < model->hidden_dim; ++i) model->bz[i] -= scale * accum->bz[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->wrx.rows * model->wrx.cols >= 512)
#endif
    for (i = 0; i < model->wrx.rows * model->wrx.cols; ++i) model->wrx.data[i] -= scale * accum->wrx[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->wrh.rows * model->wrh.cols >= 512)
#endif
    for (i = 0; i < model->wrh.rows * model->wrh.cols; ++i) model->wrh.data[i] -= scale * accum->wrh[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->hidden_dim >= 64)
#endif
    for (i = 0; i < model->hidden_dim; ++i) model->br[i] -= scale * accum->br[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->wnx.rows * model->wnx.cols >= 512)
#endif
    for (i = 0; i < model->wnx.rows * model->wnx.cols; ++i) model->wnx.data[i] -= scale * accum->wnx[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->wnh.rows * model->wnh.cols >= 512)
#endif
    for (i = 0; i < model->wnh.rows * model->wnh.cols; ++i) model->wnh.data[i] -= scale * accum->wnh[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->hidden_dim >= 64)
#endif
    for (i = 0; i < model->hidden_dim; ++i) model->bn[i] -= scale * accum->bn[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->policy_head.rows * model->policy_head.cols >= 512)
#endif
    for (i = 0; i < model->policy_head.rows * model->policy_head.cols; ++i) model->policy_head.data[i] -= scale * accum->policy_head[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->num_actions >= 16)
#endif
    for (i = 0; i < model->num_actions; ++i) model->policy_bias[i] -= scale * accum->policy_bias[i];
    for (i = 0; i < model->slot0_kind_head.rows * model->slot0_kind_head.cols; ++i) model->slot0_kind_head.data[i] -= scale * accum->slot0_kind_head[i];
    for (i = 0; i < FACTORIZED_KIND_DIM; ++i) model->slot0_kind_bias[i] -= scale * accum->slot0_kind_bias[i];
    for (i = 0; i < model->slot0_move_head.rows * model->slot0_move_head.cols; ++i) model->slot0_move_head.data[i] -= scale * accum->slot0_move_head[i];
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) model->slot0_move_bias[i] -= scale * accum->slot0_move_bias[i];
    for (i = 0; i < model->slot0_switch_head.rows * model->slot0_switch_head.cols; ++i) model->slot0_switch_head.data[i] -= scale * accum->slot0_switch_head[i];
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) model->slot0_switch_bias[i] -= scale * accum->slot0_switch_bias[i];
    for (i = 0; i < model->slot0_tera_head.rows * model->slot0_tera_head.cols; ++i) model->slot0_tera_head.data[i] -= scale * accum->slot0_tera_head[i];
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) model->slot0_tera_bias[i] -= scale * accum->slot0_tera_bias[i];
    for (i = 0; i < model->slot0_target_head.rows * model->slot0_target_head.cols; ++i) model->slot0_target_head.data[i] -= scale * accum->slot0_target_head[i];
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) model->slot0_target_bias[i] -= scale * accum->slot0_target_bias[i];
    for (i = 0; i < model->slot1_kind_head.rows * model->slot1_kind_head.cols; ++i) model->slot1_kind_head.data[i] -= scale * accum->slot1_kind_head[i];
    for (i = 0; i < FACTORIZED_KIND_DIM; ++i) model->slot1_kind_bias[i] -= scale * accum->slot1_kind_bias[i];
    for (i = 0; i < model->slot1_move_head.rows * model->slot1_move_head.cols; ++i) model->slot1_move_head.data[i] -= scale * accum->slot1_move_head[i];
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) model->slot1_move_bias[i] -= scale * accum->slot1_move_bias[i];
    for (i = 0; i < model->slot1_switch_head.rows * model->slot1_switch_head.cols; ++i) model->slot1_switch_head.data[i] -= scale * accum->slot1_switch_head[i];
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) model->slot1_switch_bias[i] -= scale * accum->slot1_switch_bias[i];
    for (i = 0; i < model->slot1_tera_head.rows * model->slot1_tera_head.cols; ++i) model->slot1_tera_head.data[i] -= scale * accum->slot1_tera_head[i];
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) model->slot1_tera_bias[i] -= scale * accum->slot1_tera_bias[i];
    for (i = 0; i < model->slot1_target_head.rows * model->slot1_target_head.cols; ++i) model->slot1_target_head.data[i] -= scale * accum->slot1_target_head[i];
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) model->slot1_target_bias[i] -= scale * accum->slot1_target_bias[i];
    for (i = 0; i < model->joint_pair_head.rows * model->joint_pair_head.cols; ++i) model->joint_pair_head.data[i] -= scale * accum->joint_pair_head[i];
    for (i = 0; i < FACTORIZED_PAIR_DIM; ++i) model->joint_pair_bias[i] -= scale * accum->joint_pair_bias[i];
    for (i = 0; i < model->entity_encoder.rows * model->entity_encoder.cols; ++i) model->entity_encoder.data[i] -= scale * accum->entity_encoder[i];
    for (i = 0; i < (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u); ++i) model->entity_bias[i] -= scale * accum->entity_bias[i];
    for (i = 0; i < model->entity_decoder.rows * model->entity_decoder.cols; ++i) model->entity_decoder.data[i] -= scale * accum->entity_decoder[i];
#ifdef _OPENMP
    #pragma omp parallel for if(model->hidden_dim >= 64)
#endif
    for (i = 0; i < model->hidden_dim; ++i) model->value_head[i] -= scale * accum->value_head[i];
    model->value_bias -= scale * accum->value_bias;
    sync_flat_heads_from_factorized(model);
    gru_model_clear_accumulated_supervised_updates(model);
    return 1;
}

static void adam_apply_array(float* params, float* m, float* v, const float* grads, size_t count,
        float learning_rate, float beta1, float beta2, float epsilon, float bias_correction1, float bias_correction2) {
    size_t i;
    for (i = 0; i < count; ++i) {
        float grad = grads[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        {
            float m_hat = m[i] / bias_correction1;
            float v_hat = v[i] / bias_correction2;
            params[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
        }
    }
}

int gru_model_apply_accumulated_adam_updates(
    GruModel* model,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float gradient_clip
) {
    GruGradientAccum* accum;
    GruAdamState* state;
    size_t i;
    double global_sq_norm = 0.0;
    float scale = 1.0f;
    float bias_correction1;
    float bias_correction2;
    if (!model || !gru_gradient_accum_ensure(model) || !gru_adam_state_ensure(model)) {
        return 0;
    }
    accum = &model->grad_accum;
    state = &model->adam_state;
    if (accum->count == 0) {
        return 1;
    }
    scale = 1.0f / (float)accum->count;
    for (i = 0; i < model->wzx.rows * model->wzx.cols; ++i) { float g = accum->wzx[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->wzh.rows * model->wzh.cols; ++i) { float g = accum->wzh[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->hidden_dim; ++i) { float g = accum->bz[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->wrx.rows * model->wrx.cols; ++i) { float g = accum->wrx[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->wrh.rows * model->wrh.cols; ++i) { float g = accum->wrh[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->hidden_dim; ++i) { float g = accum->br[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->wnx.rows * model->wnx.cols; ++i) { float g = accum->wnx[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->wnh.rows * model->wnh.cols; ++i) { float g = accum->wnh[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->hidden_dim; ++i) { float g = accum->bn[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->policy_head.rows * model->policy_head.cols; ++i) { float g = accum->policy_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->num_actions; ++i) { float g = accum->policy_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot0_kind_head.rows * model->slot0_kind_head.cols; ++i) { float g = accum->slot0_kind_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_KIND_DIM; ++i) { float g = accum->slot0_kind_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot0_move_head.rows * model->slot0_move_head.cols; ++i) { float g = accum->slot0_move_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) { float g = accum->slot0_move_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot0_switch_head.rows * model->slot0_switch_head.cols; ++i) { float g = accum->slot0_switch_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) { float g = accum->slot0_switch_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot0_tera_head.rows * model->slot0_tera_head.cols; ++i) { float g = accum->slot0_tera_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) { float g = accum->slot0_tera_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot0_target_head.rows * model->slot0_target_head.cols; ++i) { float g = accum->slot0_target_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) { float g = accum->slot0_target_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot1_kind_head.rows * model->slot1_kind_head.cols; ++i) { float g = accum->slot1_kind_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_KIND_DIM; ++i) { float g = accum->slot1_kind_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot1_move_head.rows * model->slot1_move_head.cols; ++i) { float g = accum->slot1_move_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) { float g = accum->slot1_move_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot1_switch_head.rows * model->slot1_switch_head.cols; ++i) { float g = accum->slot1_switch_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) { float g = accum->slot1_switch_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot1_tera_head.rows * model->slot1_tera_head.cols; ++i) { float g = accum->slot1_tera_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) { float g = accum->slot1_tera_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->slot1_target_head.rows * model->slot1_target_head.cols; ++i) { float g = accum->slot1_target_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) { float g = accum->slot1_target_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->joint_pair_head.rows * model->joint_pair_head.cols; ++i) { float g = accum->joint_pair_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < FACTORIZED_PAIR_DIM; ++i) { float g = accum->joint_pair_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->entity_encoder.rows * model->entity_encoder.cols; ++i) { float g = accum->entity_encoder[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u); ++i) { float g = accum->entity_bias[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->entity_decoder.rows * model->entity_decoder.cols; ++i) { float g = accum->entity_decoder[i] * scale; global_sq_norm += (double)g * (double)g; }
    for (i = 0; i < model->hidden_dim; ++i) { float g = accum->value_head[i] * scale; global_sq_norm += (double)g * (double)g; }
    {
        float g = accum->value_bias * scale;
        global_sq_norm += (double)g * (double)g;
    }
    if (gradient_clip > 0.0f && global_sq_norm > 0.0) {
        double global_norm = sqrt(global_sq_norm);
        if (global_norm > (double)gradient_clip) {
            scale *= (float)((double)gradient_clip / global_norm);
        }
    }
    state->step += 1u;
    bias_correction1 = 1.0f - powf(beta1, (float)state->step);
    bias_correction2 = 1.0f - powf(beta2, (float)state->step);
    for (i = 0; i < model->wzx.rows * model->wzx.cols; ++i) accum->wzx[i] *= scale;
    for (i = 0; i < model->wzh.rows * model->wzh.cols; ++i) accum->wzh[i] *= scale;
    for (i = 0; i < model->hidden_dim; ++i) accum->bz[i] *= scale;
    for (i = 0; i < model->wrx.rows * model->wrx.cols; ++i) accum->wrx[i] *= scale;
    for (i = 0; i < model->wrh.rows * model->wrh.cols; ++i) accum->wrh[i] *= scale;
    for (i = 0; i < model->hidden_dim; ++i) accum->br[i] *= scale;
    for (i = 0; i < model->wnx.rows * model->wnx.cols; ++i) accum->wnx[i] *= scale;
    for (i = 0; i < model->wnh.rows * model->wnh.cols; ++i) accum->wnh[i] *= scale;
    for (i = 0; i < model->hidden_dim; ++i) accum->bn[i] *= scale;
    for (i = 0; i < model->policy_head.rows * model->policy_head.cols; ++i) accum->policy_head[i] *= scale;
    for (i = 0; i < model->num_actions; ++i) accum->policy_bias[i] *= scale;
    for (i = 0; i < model->slot0_kind_head.rows * model->slot0_kind_head.cols; ++i) accum->slot0_kind_head[i] *= scale;
    for (i = 0; i < FACTORIZED_KIND_DIM; ++i) accum->slot0_kind_bias[i] *= scale;
    for (i = 0; i < model->slot0_move_head.rows * model->slot0_move_head.cols; ++i) accum->slot0_move_head[i] *= scale;
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) accum->slot0_move_bias[i] *= scale;
    for (i = 0; i < model->slot0_switch_head.rows * model->slot0_switch_head.cols; ++i) accum->slot0_switch_head[i] *= scale;
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) accum->slot0_switch_bias[i] *= scale;
    for (i = 0; i < model->slot0_tera_head.rows * model->slot0_tera_head.cols; ++i) accum->slot0_tera_head[i] *= scale;
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) accum->slot0_tera_bias[i] *= scale;
    for (i = 0; i < model->slot0_target_head.rows * model->slot0_target_head.cols; ++i) accum->slot0_target_head[i] *= scale;
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) accum->slot0_target_bias[i] *= scale;
    for (i = 0; i < model->slot1_kind_head.rows * model->slot1_kind_head.cols; ++i) accum->slot1_kind_head[i] *= scale;
    for (i = 0; i < FACTORIZED_KIND_DIM; ++i) accum->slot1_kind_bias[i] *= scale;
    for (i = 0; i < model->slot1_move_head.rows * model->slot1_move_head.cols; ++i) accum->slot1_move_head[i] *= scale;
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) accum->slot1_move_bias[i] *= scale;
    for (i = 0; i < model->slot1_switch_head.rows * model->slot1_switch_head.cols; ++i) accum->slot1_switch_head[i] *= scale;
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) accum->slot1_switch_bias[i] *= scale;
    for (i = 0; i < model->slot1_tera_head.rows * model->slot1_tera_head.cols; ++i) accum->slot1_tera_head[i] *= scale;
    for (i = 0; i < FACTORIZED_TERA_DIM; ++i) accum->slot1_tera_bias[i] *= scale;
    for (i = 0; i < model->slot1_target_head.rows * model->slot1_target_head.cols; ++i) accum->slot1_target_head[i] *= scale;
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) accum->slot1_target_bias[i] *= scale;
    for (i = 0; i < model->joint_pair_head.rows * model->joint_pair_head.cols; ++i) accum->joint_pair_head[i] *= scale;
    for (i = 0; i < FACTORIZED_PAIR_DIM; ++i) accum->joint_pair_bias[i] *= scale;
    for (i = 0; i < model->entity_encoder.rows * model->entity_encoder.cols; ++i) accum->entity_encoder[i] *= scale;
    for (i = 0; i < (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u); ++i) accum->entity_bias[i] *= scale;
    for (i = 0; i < model->entity_decoder.rows * model->entity_decoder.cols; ++i) accum->entity_decoder[i] *= scale;
    for (i = 0; i < model->hidden_dim; ++i) accum->value_head[i] *= scale;
    accum->value_bias *= scale;
    adam_apply_array(model->wzx.data, state->wzx_m, state->wzx_v, accum->wzx, model->wzx.rows * model->wzx.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->wzh.data, state->wzh_m, state->wzh_v, accum->wzh, model->wzh.rows * model->wzh.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->bz, state->bz_m, state->bz_v, accum->bz, model->hidden_dim, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->wrx.data, state->wrx_m, state->wrx_v, accum->wrx, model->wrx.rows * model->wrx.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->wrh.data, state->wrh_m, state->wrh_v, accum->wrh, model->wrh.rows * model->wrh.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->br, state->br_m, state->br_v, accum->br, model->hidden_dim, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->wnx.data, state->wnx_m, state->wnx_v, accum->wnx, model->wnx.rows * model->wnx.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->wnh.data, state->wnh_m, state->wnh_v, accum->wnh, model->wnh.rows * model->wnh.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->bn, state->bn_m, state->bn_v, accum->bn, model->hidden_dim, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->policy_head.data, state->policy_head_m, state->policy_head_v, accum->policy_head, model->policy_head.rows * model->policy_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->policy_bias, state->policy_bias_m, state->policy_bias_v, accum->policy_bias, model->num_actions, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_kind_head.data, state->slot0_kind_head_m, state->slot0_kind_head_v, accum->slot0_kind_head, model->slot0_kind_head.rows * model->slot0_kind_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_kind_bias, state->slot0_kind_bias_m, state->slot0_kind_bias_v, accum->slot0_kind_bias, FACTORIZED_KIND_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_move_head.data, state->slot0_move_head_m, state->slot0_move_head_v, accum->slot0_move_head, model->slot0_move_head.rows * model->slot0_move_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_move_bias, state->slot0_move_bias_m, state->slot0_move_bias_v, accum->slot0_move_bias, FACTORIZED_MOVE_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_switch_head.data, state->slot0_switch_head_m, state->slot0_switch_head_v, accum->slot0_switch_head, model->slot0_switch_head.rows * model->slot0_switch_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_switch_bias, state->slot0_switch_bias_m, state->slot0_switch_bias_v, accum->slot0_switch_bias, FACTORIZED_SWITCH_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_tera_head.data, state->slot0_tera_head_m, state->slot0_tera_head_v, accum->slot0_tera_head, model->slot0_tera_head.rows * model->slot0_tera_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_tera_bias, state->slot0_tera_bias_m, state->slot0_tera_bias_v, accum->slot0_tera_bias, FACTORIZED_TERA_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_target_head.data, state->slot0_target_head_m, state->slot0_target_head_v, accum->slot0_target_head, model->slot0_target_head.rows * model->slot0_target_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot0_target_bias, state->slot0_target_bias_m, state->slot0_target_bias_v, accum->slot0_target_bias, FACTORIZED_TARGET_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_kind_head.data, state->slot1_kind_head_m, state->slot1_kind_head_v, accum->slot1_kind_head, model->slot1_kind_head.rows * model->slot1_kind_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_kind_bias, state->slot1_kind_bias_m, state->slot1_kind_bias_v, accum->slot1_kind_bias, FACTORIZED_KIND_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_move_head.data, state->slot1_move_head_m, state->slot1_move_head_v, accum->slot1_move_head, model->slot1_move_head.rows * model->slot1_move_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_move_bias, state->slot1_move_bias_m, state->slot1_move_bias_v, accum->slot1_move_bias, FACTORIZED_MOVE_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_switch_head.data, state->slot1_switch_head_m, state->slot1_switch_head_v, accum->slot1_switch_head, model->slot1_switch_head.rows * model->slot1_switch_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_switch_bias, state->slot1_switch_bias_m, state->slot1_switch_bias_v, accum->slot1_switch_bias, FACTORIZED_SWITCH_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_tera_head.data, state->slot1_tera_head_m, state->slot1_tera_head_v, accum->slot1_tera_head, model->slot1_tera_head.rows * model->slot1_tera_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_tera_bias, state->slot1_tera_bias_m, state->slot1_tera_bias_v, accum->slot1_tera_bias, FACTORIZED_TERA_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_target_head.data, state->slot1_target_head_m, state->slot1_target_head_v, accum->slot1_target_head, model->slot1_target_head.rows * model->slot1_target_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->slot1_target_bias, state->slot1_target_bias_m, state->slot1_target_bias_v, accum->slot1_target_bias, FACTORIZED_TARGET_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->joint_pair_head.data, state->joint_pair_head_m, state->joint_pair_head_v, accum->joint_pair_head, model->joint_pair_head.rows * model->joint_pair_head.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    adam_apply_array(model->joint_pair_bias, state->joint_pair_bias_m, state->joint_pair_bias_v, accum->joint_pair_bias, FACTORIZED_PAIR_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    if (model->entity_encoder_enabled) {
        adam_apply_array(model->entity_encoder.data, state->entity_encoder_m, state->entity_encoder_v, accum->entity_encoder, model->entity_encoder.rows * model->entity_encoder.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
        adam_apply_array(model->entity_bias, state->entity_bias_m, state->entity_bias_v, accum->entity_bias, GRU_ENTITY_EMBED_DIM, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
        adam_apply_array(model->entity_decoder.data, state->entity_decoder_m, state->entity_decoder_v, accum->entity_decoder, model->entity_decoder.rows * model->entity_decoder.cols, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    }
    adam_apply_array(model->value_head, state->value_head_m, state->value_head_v, accum->value_head, model->hidden_dim, learning_rate, beta1, beta2, epsilon, bias_correction1, bias_correction2);
    state->value_bias_m = beta1 * state->value_bias_m + (1.0f - beta1) * accum->value_bias;
    state->value_bias_v = beta2 * state->value_bias_v + (1.0f - beta2) * accum->value_bias * accum->value_bias;
    model->value_bias -= learning_rate * (state->value_bias_m / bias_correction1) / (sqrtf(state->value_bias_v / bias_correction2) + epsilon);
    sync_flat_heads_from_factorized(model);
    gru_model_clear_accumulated_supervised_updates(model);
    return 1;
}

static void matrix_vec_mul_accum(const Matrix* matrix, const float* vec, float* out) {
    size_t r;
#ifdef _OPENMP
    #pragma omp parallel for private(r) if(matrix->rows >= 32)
#endif
    for (r = 0; r < matrix->rows; ++r) {
        size_t c;
        const float* row = matrix->data + (r * matrix->cols);
        float sum = 0.0f;
        for (c = 0; c < matrix->cols; ++c) {
            sum += row[c] * vec[c];
        }
        out[r] += sum;
    }
}

static void matrix_transpose_vec_mul_accum(const Matrix* matrix, const float* vec, float* out) {
    size_t r;
#ifdef _OPENMP
    int thread_count;
    float* partials;
    if (matrix->rows >= 32 && matrix->cols >= 32) {
        thread_count = omp_get_max_threads();
        partials = (float*)calloc((size_t)thread_count * matrix->cols, sizeof(float));
        if (partials) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                float* local = partials + ((size_t)tid * matrix->cols);
                #pragma omp for
                for (r = 0; r < matrix->rows; ++r) {
                    size_t c;
                    const float* row = matrix->data + (r * matrix->cols);
                    float vr = vec[r];
                    for (c = 0; c < matrix->cols; ++c) {
                        local[c] += row[c] * vr;
                    }
                }
            }
            for (r = 0; r < (size_t)thread_count; ++r) {
                size_t c;
                float* local = partials + (r * matrix->cols);
                for (c = 0; c < matrix->cols; ++c) {
                    out[c] += local[c];
                }
            }
            free(partials);
            return;
        }
    }
#endif
    for (r = 0; r < matrix->rows; ++r) {
        size_t c;
        const float* row = matrix->data + (r * matrix->cols);
        for (c = 0; c < matrix->cols; ++c) {
            out[c] += row[c] * vec[r];
        }
    }
}

static void outer_product_accum(float* dst, size_t rows, size_t cols, const float* lhs, const float* rhs) {
    size_t r;
#ifdef _OPENMP
    #pragma omp parallel for private(r) if(rows >= 32)
#endif
    for (r = 0; r < rows; ++r) {
        size_t c;
        float* dst_row = dst + (r * cols);
        float lhs_r = lhs[r];
        for (c = 0; c < cols; ++c) {
            dst_row[c] += lhs_r * rhs[c];
        }
    }
}

static void transform_observation_input(
    const GruModel* model,
    const float* input,
    float* transformed,
    float* entity_hidden
) {
    const size_t entity_offset = OBS_GLOBAL_FEATURES + 2u * OBS_SIDE_FEATURES;
    size_t entity;
    memcpy(transformed, input, model->input_dim * sizeof(float));
    memset(entity_hidden, 0, 2u * OBS_TEAM_SIZE * GRU_ENTITY_EMBED_DIM * sizeof(float));
    if (!model->entity_encoder_enabled) {
        return;
    }
    for (entity = 0; entity < 2u * OBS_TEAM_SIZE; ++entity) {
        const float* features = input + entity_offset + entity * OBS_POKEMON_FEATURES;
        float* output_features = transformed + entity_offset + entity * OBS_POKEMON_FEATURES;
        float* embedding = entity_hidden + entity * GRU_ENTITY_EMBED_DIM;
        float present = features[0];
        size_t e;
        size_t f;
        if (present <= 0.0f) {
            continue;
        }
        memcpy(embedding, model->entity_bias, GRU_ENTITY_EMBED_DIM * sizeof(float));
        matrix_vec_mul_accum(&model->entity_encoder, features, embedding);
        for (e = 0; e < GRU_ENTITY_EMBED_DIM; ++e) {
            embedding[e] = tanhf(embedding[e]);
        }
        for (f = 0; f < OBS_POKEMON_FEATURES; ++f) {
            const float* decoder_row = model->entity_decoder.data + f * GRU_ENTITY_EMBED_DIM;
            float residual = 0.0f;
            for (e = 0; e < GRU_ENTITY_EMBED_DIM; ++e) {
                residual += decoder_row[e] * embedding[e];
            }
            output_features[f] += present * residual;
        }
    }
}

static void backprop_entity_encoder(
    const GruModel* model,
    const float* raw_input,
    const float* entity_hidden,
    const float* grad_input,
    float* grad_encoder,
    float* grad_bias,
    float* grad_decoder
) {
    const size_t entity_offset = OBS_GLOBAL_FEATURES + 2u * OBS_SIDE_FEATURES;
    size_t entity;
    if (!model->entity_encoder_enabled) {
        return;
    }
    for (entity = 0; entity < 2u * OBS_TEAM_SIZE; ++entity) {
        const float* features = raw_input + entity_offset + entity * OBS_POKEMON_FEATURES;
        const float* embedding = entity_hidden + entity * GRU_ENTITY_EMBED_DIM;
        const float* entity_grad = grad_input + entity_offset + entity * OBS_POKEMON_FEATURES;
        float grad_embedding[GRU_ENTITY_EMBED_DIM] = {0};
        float present = features[0];
        size_t e;
        size_t f;
        if (present <= 0.0f) {
            continue;
        }
        for (f = 0; f < OBS_POKEMON_FEATURES; ++f) {
            float feature_gradient = entity_grad[f] * present;
            for (e = 0; e < GRU_ENTITY_EMBED_DIM; ++e) {
                grad_decoder[f * GRU_ENTITY_EMBED_DIM + e] += feature_gradient * embedding[e];
                grad_embedding[e] += model->entity_decoder.data[f * GRU_ENTITY_EMBED_DIM + e] * feature_gradient;
            }
        }
        for (e = 0; e < GRU_ENTITY_EMBED_DIM; ++e) {
            float grad_pre = grad_embedding[e] * (1.0f - embedding[e] * embedding[e]);
            grad_bias[e] += grad_pre;
            for (f = 0; f < OBS_POKEMON_FEATURES; ++f) {
                grad_encoder[e * OBS_POKEMON_FEATURES + f] += grad_pre * features[f];
            }
        }
    }
}

static void softmax(const float* logits, size_t n, float* out) {
    size_t i;
    float max_logit = logits[0];
    float sum = 0.0f;

    for (i = 1; i < n; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    for (i = 0; i < n; ++i) {
        out[i] = expf(logits[i] - max_logit);
        sum += out[i];
    }
    if (sum <= 0.0f) {
        sum = 1.0f;
    }
    for (i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

static void masked_softmax_small(const float* logits, const unsigned char* mask, size_t n, float* out) {
    size_t i;
    float max_logit = 0.0f;
    float sum = 0.0f;
    int any = 0;
    for (i = 0; i < n; ++i) {
        out[i] = 0.0f;
        if (!mask || mask[i]) {
            if (!any || logits[i] > max_logit) {
                max_logit = logits[i];
            }
            any = 1;
        }
    }
    if (!any) {
        return;
    }
    for (i = 0; i < n; ++i) {
        if (!mask || mask[i]) {
            out[i] = expf(logits[i] - max_logit);
            sum += out[i];
        }
    }
    if (sum <= 0.0f) {
        return;
    }
    for (i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

static void evaluate_small_head(const Matrix* head, const float* bias, const float* hidden_state, const unsigned char* mask, size_t dim, float* out) {
    float logits[FACTORIZED_SWITCH_DIM];
    size_t i;
    size_t h;
    for (i = 0; i < dim; ++i) {
        float sum = bias ? bias[i] : 0.0f;
        for (h = 0; h < head->cols; ++h) {
            sum += head->data[i * head->cols + h] * hidden_state[h];
        }
        logits[i] = sum;
    }
    masked_softmax_small(logits, mask, dim, out);
}

static void build_factorized_masks(const unsigned char* legal_mask, int slot, unsigned char* kind_mask, unsigned char* move_mask, unsigned char* switch_mask, unsigned char* tera_mask) {
    int base = slot == 0 ? 0 : 14;
    int i;
    memset(kind_mask, 0, FACTORIZED_KIND_DIM);
    memset(move_mask, 0, FACTORIZED_MOVE_DIM);
    memset(switch_mask, 0, FACTORIZED_SWITCH_DIM);
    memset(tera_mask, 0, FACTORIZED_TERA_DIM);
    if (!legal_mask) {
        return;
    }
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        if (legal_mask[base + i] || legal_mask[base + 4 + i]) {
            move_mask[i] = 1;
            kind_mask[0] = 1;
        }
        if (legal_mask[base + i]) {
            tera_mask[0] = 1;
        }
        if (legal_mask[base + 4 + i]) {
            tera_mask[1] = 1;
        }
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        if (legal_mask[base + 8 + i]) {
            switch_mask[i] = 1;
            kind_mask[1] = 1;
        }
    }
}

static size_t joint_pair_index(int action0, int action1) {
    size_t lo = (size_t)(action0 < action1 ? action0 : action1);
    size_t hi = (size_t)(action0 < action1 ? action1 : action0);
    return lo * FACTORIZED_LOCAL_ACTION_DIM - (lo * (lo - 1u)) / 2u + (hi - lo);
}

static int joint_local_pair_legal(int action0, int action1) {
    if (action0 >= 4 && action0 < 8 && action1 >= 4 && action1 < 8) {
        return 0;
    }
    if (action0 >= 8 && action1 >= 8 && action0 == action1) {
        return 0;
    }
    return 1;
}

static float linear_row_logit(const Matrix* head, const float* bias, size_t row, const float* hidden_state) {
    float value = bias[row];
    size_t h;
    for (h = 0; h < head->cols; ++h) {
        value += head->data[row * head->cols + h] * hidden_state[h];
    }
    return value;
}

static float factorized_local_action_logit(const GruModel* model, const float* hidden_state, int slot, int action) {
    const Matrix* kind_head = slot == 0 ? &model->slot0_kind_head : &model->slot1_kind_head;
    const float* kind_bias = slot == 0 ? model->slot0_kind_bias : model->slot1_kind_bias;
    const Matrix* move_head = slot == 0 ? &model->slot0_move_head : &model->slot1_move_head;
    const float* move_bias = slot == 0 ? model->slot0_move_bias : model->slot1_move_bias;
    const Matrix* switch_head = slot == 0 ? &model->slot0_switch_head : &model->slot1_switch_head;
    const float* switch_bias = slot == 0 ? model->slot0_switch_bias : model->slot1_switch_bias;
    const Matrix* tera_head = slot == 0 ? &model->slot0_tera_head : &model->slot1_tera_head;
    const float* tera_bias = slot == 0 ? model->slot0_tera_bias : model->slot1_tera_bias;
    if (action < 8) {
        int move = action & 3;
        int tera = action >= 4 ? 1 : 0;
        return linear_row_logit(kind_head, kind_bias, 0u, hidden_state) +
            linear_row_logit(move_head, move_bias, (size_t)move, hidden_state) +
            linear_row_logit(tera_head, tera_bias, (size_t)tera, hidden_state);
    }
    return linear_row_logit(kind_head, kind_bias, 1u, hidden_state) +
        linear_row_logit(switch_head, switch_bias, (size_t)(action - 8), hidden_state);
}

static void sync_flat_heads_from_factorized(GruModel* model) {
    size_t h;
    int i;
    if (!model) {
        return;
    }
    for (h = 0; h < model->hidden_dim; ++h) {
        for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
            model->policy_head.data[(0 + i) * model->hidden_dim + h] =
                model->slot0_kind_head.data[0 * model->hidden_dim + h] +
                model->slot0_move_head.data[i * model->hidden_dim + h] +
                model->slot0_tera_head.data[0 * model->hidden_dim + h];
            model->policy_head.data[(4 + i) * model->hidden_dim + h] =
                model->slot0_kind_head.data[0 * model->hidden_dim + h] +
                model->slot0_move_head.data[i * model->hidden_dim + h] +
                model->slot0_tera_head.data[1 * model->hidden_dim + h];
            model->policy_head.data[(14 + i) * model->hidden_dim + h] =
                model->slot1_kind_head.data[0 * model->hidden_dim + h] +
                model->slot1_move_head.data[i * model->hidden_dim + h] +
                model->slot1_tera_head.data[0 * model->hidden_dim + h];
            model->policy_head.data[(18 + i) * model->hidden_dim + h] =
                model->slot1_kind_head.data[0 * model->hidden_dim + h] +
                model->slot1_move_head.data[i * model->hidden_dim + h] +
                model->slot1_tera_head.data[1 * model->hidden_dim + h];
        }
        for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
            model->policy_head.data[(8 + i) * model->hidden_dim + h] =
                model->slot0_kind_head.data[1 * model->hidden_dim + h] +
                model->slot0_switch_head.data[i * model->hidden_dim + h];
            model->policy_head.data[(22 + i) * model->hidden_dim + h] =
                model->slot1_kind_head.data[1 * model->hidden_dim + h] +
                model->slot1_switch_head.data[i * model->hidden_dim + h];
        }
    }
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        model->policy_bias[0 + i] = model->slot0_kind_bias[0] + model->slot0_move_bias[i] + model->slot0_tera_bias[0];
        model->policy_bias[4 + i] = model->slot0_kind_bias[0] + model->slot0_move_bias[i] + model->slot0_tera_bias[1];
        model->policy_bias[14 + i] = model->slot1_kind_bias[0] + model->slot1_move_bias[i] + model->slot1_tera_bias[0];
        model->policy_bias[18 + i] = model->slot1_kind_bias[0] + model->slot1_move_bias[i] + model->slot1_tera_bias[1];
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        model->policy_bias[8 + i] = model->slot0_kind_bias[1] + model->slot0_switch_bias[i];
        model->policy_bias[22 + i] = model->slot1_kind_bias[1] + model->slot1_switch_bias[i];
    }
}

static void factorized_slot_policy_gradients(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    int slot,
    const FactorizedActionChoice* choice,
    float policy_scale,
    float entropy_coef,
    float* action_loss_sum,
    float* accuracy_sum,
    float* grad_h,
    float* grad_kind_head,
    float* grad_kind_bias,
    float* grad_move_head,
    float* grad_move_bias,
    float* grad_switch_head,
    float* grad_switch_bias,
    float* grad_tera_head,
    float* grad_tera_bias,
    float* grad_target_head,
    float* grad_target_bias
) {
    unsigned char kind_mask[FACTORIZED_KIND_DIM];
    unsigned char move_mask[FACTORIZED_MOVE_DIM];
    unsigned char switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char tera_mask[FACTORIZED_TERA_DIM];
    float kind_policy[FACTORIZED_KIND_DIM];
    float move_policy[FACTORIZED_MOVE_DIM];
    float switch_policy[FACTORIZED_SWITCH_DIM];
    float tera_policy[FACTORIZED_TERA_DIM];
    unsigned char target_mask[FACTORIZED_TARGET_DIM];
    float target_policy[FACTORIZED_TARGET_DIM];
    float grad_logits[FACTORIZED_SWITCH_DIM];
    const Matrix* kind_head = slot == 0 ? &model->slot0_kind_head : &model->slot1_kind_head;
    const float* kind_bias = slot == 0 ? model->slot0_kind_bias : model->slot1_kind_bias;
    const Matrix* move_head = slot == 0 ? &model->slot0_move_head : &model->slot1_move_head;
    const float* move_bias = slot == 0 ? model->slot0_move_bias : model->slot1_move_bias;
    const Matrix* switch_head = slot == 0 ? &model->slot0_switch_head : &model->slot1_switch_head;
    const float* switch_bias = slot == 0 ? model->slot0_switch_bias : model->slot1_switch_bias;
    const Matrix* tera_head = slot == 0 ? &model->slot0_tera_head : &model->slot1_tera_head;
    const float* tera_bias = slot == 0 ? model->slot0_tera_bias : model->slot1_tera_bias;
    const Matrix* target_head = slot == 0 ? &model->slot0_target_head : &model->slot1_target_head;
    const float* target_bias = slot == 0 ? model->slot0_target_bias : model->slot1_target_bias;
    int target_kind;
    int accuracy_ok = 1;
    int i;
    int h;
    build_factorized_masks(legal_mask, slot, kind_mask, move_mask, switch_mask, tera_mask);
    evaluate_small_head(kind_head, kind_bias, hidden_state, kind_mask, FACTORIZED_KIND_DIM, kind_policy);
    evaluate_small_head(move_head, move_bias, hidden_state, move_mask, FACTORIZED_MOVE_DIM, move_policy);
    evaluate_small_head(switch_head, switch_bias, hidden_state, switch_mask, FACTORIZED_SWITCH_DIM, switch_policy);
    target_kind = (slot == 0 ? choice->slot0_kind : choice->slot1_kind) == FACTORIZED_ACTION_SWITCH ? 1 : 0;
    *action_loss_sum += -logf(kind_policy[target_kind] > 1.0e-8f ? kind_policy[target_kind] : 1.0e-8f);
    accuracy_ok &= (kind_policy[1] > kind_policy[0] ? 1 : 0) == target_kind;
    {
        float neg_entropy = 0.0f;
        for (i = 0; i < FACTORIZED_KIND_DIM; ++i) if (kind_mask[i]) { float p = kind_policy[i] > 1.0e-8f ? kind_policy[i] : 1.0e-8f; neg_entropy += p * logf(p); }
        for (i = 0; i < FACTORIZED_KIND_DIM; ++i) {
            if (!kind_mask[i]) { grad_logits[i] = 0.0f; continue; }
            grad_logits[i] = (kind_policy[i] - (i == target_kind ? 1.0f : 0.0f)) * policy_scale;
            if (entropy_coef != 0.0f) { float p = kind_policy[i] > 1.0e-8f ? kind_policy[i] : 1.0e-8f; grad_logits[i] += entropy_coef * p * (logf(p) - neg_entropy); }
            grad_kind_bias[i] += grad_logits[i];
            for (h = 0; h < (int)model->hidden_dim; ++h) grad_kind_head[i * model->hidden_dim + h] += grad_logits[i] * hidden_state[h];
        }
        matrix_transpose_vec_mul_accum(kind_head, grad_logits, grad_h);
    }
    if (target_kind == 0) {
        int move_index = slot == 0 ? choice->slot0_move_index : choice->slot1_move_index;
        int tera_index = (slot == 0 ? choice->slot0_use_tera : choice->slot1_use_tera) ? 1 : 0;
        int target_index = slot == 0 ? choice->slot0_target_index : choice->slot1_target_index;
        unsigned char target_mask_bits = slot == 0 ? choice->slot0_target_mask : choice->slot1_target_mask;
        if (target_index < 0 || target_index >= FACTORIZED_TARGET_DIM ||
                !(target_mask_bits & FACTORIZED_TARGET_BIT(target_index))) {
            target_mask_bits = 0u;
        }
        tera_mask[0] = legal_mask[(slot == 0 ? 0 : 14) + move_index] ? 1 : 0;
        tera_mask[1] = legal_mask[(slot == 0 ? 4 : 18) + move_index] ? 1 : 0;
        evaluate_small_head(tera_head, tera_bias, hidden_state, tera_mask, FACTORIZED_TERA_DIM, tera_policy);
        *action_loss_sum += -logf(move_policy[move_index] > 1.0e-8f ? move_policy[move_index] : 1.0e-8f);
        *action_loss_sum += -logf(tera_policy[tera_index] > 1.0e-8f ? tera_policy[tera_index] : 1.0e-8f);
        for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) if (move_policy[i] > move_policy[move_index]) accuracy_ok = 0;
        for (i = 0; i < FACTORIZED_TERA_DIM; ++i) if (tera_mask[i] && tera_policy[i] > tera_policy[tera_index]) accuracy_ok = 0;
        {
            float neg_entropy = 0.0f;
            for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) if (move_mask[i]) { float p = move_policy[i] > 1.0e-8f ? move_policy[i] : 1.0e-8f; neg_entropy += p * logf(p); }
            for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
                if (!move_mask[i]) { grad_logits[i] = 0.0f; continue; }
                grad_logits[i] = (move_policy[i] - (i == move_index ? 1.0f : 0.0f)) * policy_scale;
                if (entropy_coef != 0.0f) { float p = move_policy[i] > 1.0e-8f ? move_policy[i] : 1.0e-8f; grad_logits[i] += entropy_coef * p * (logf(p) - neg_entropy); }
                grad_move_bias[i] += grad_logits[i];
                for (h = 0; h < (int)model->hidden_dim; ++h) grad_move_head[i * model->hidden_dim + h] += grad_logits[i] * hidden_state[h];
            }
            matrix_transpose_vec_mul_accum(move_head, grad_logits, grad_h);
        }
        {
            float neg_entropy = 0.0f;
            for (i = 0; i < FACTORIZED_TERA_DIM; ++i) if (tera_mask[i]) { float p = tera_policy[i] > 1.0e-8f ? tera_policy[i] : 1.0e-8f; neg_entropy += p * logf(p); }
            for (i = 0; i < FACTORIZED_TERA_DIM; ++i) {
                if (!tera_mask[i]) { grad_logits[i] = 0.0f; continue; }
                grad_logits[i] = (tera_policy[i] - (i == tera_index ? 1.0f : 0.0f)) * policy_scale;
                if (entropy_coef != 0.0f) { float p = tera_policy[i] > 1.0e-8f ? tera_policy[i] : 1.0e-8f; grad_logits[i] += entropy_coef * p * (logf(p) - neg_entropy); }
                grad_tera_bias[i] += grad_logits[i];
                for (h = 0; h < (int)model->hidden_dim; ++h) grad_tera_head[i * model->hidden_dim + h] += grad_logits[i] * hidden_state[h];
            }
            matrix_transpose_vec_mul_accum(tera_head, grad_logits, grad_h);
        }
        if (target_mask_bits != 0u) {
            factorized_target_mask_to_array(target_mask_bits, target_mask);
            evaluate_small_head(target_head, target_bias, hidden_state, target_mask, FACTORIZED_TARGET_DIM, target_policy);
            *action_loss_sum += -logf(target_policy[target_index] > 1.0e-8f ? target_policy[target_index] : 1.0e-8f);
            for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) if (target_mask[i] && target_policy[i] > target_policy[target_index]) accuracy_ok = 0;
            {
                float neg_entropy = 0.0f;
                for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) if (target_mask[i]) { float p = target_policy[i] > 1.0e-8f ? target_policy[i] : 1.0e-8f; neg_entropy += p * logf(p); }
                for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) {
                    if (!target_mask[i]) { grad_logits[i] = 0.0f; continue; }
                    grad_logits[i] = (target_policy[i] - (i == target_index ? 1.0f : 0.0f)) * policy_scale;
                    if (entropy_coef != 0.0f) { float p = target_policy[i] > 1.0e-8f ? target_policy[i] : 1.0e-8f; grad_logits[i] += entropy_coef * p * (logf(p) - neg_entropy); }
                    grad_target_bias[i] += grad_logits[i];
                    for (h = 0; h < (int)model->hidden_dim; ++h) grad_target_head[i * model->hidden_dim + h] += grad_logits[i] * hidden_state[h];
                }
                matrix_transpose_vec_mul_accum(target_head, grad_logits, grad_h);
            }
        }
    } else {
        int switch_index = slot == 0 ? choice->slot0_switch_index : choice->slot1_switch_index;
        *action_loss_sum += -logf(switch_policy[switch_index] > 1.0e-8f ? switch_policy[switch_index] : 1.0e-8f);
        for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) if (switch_policy[i] > switch_policy[switch_index]) accuracy_ok = 0;
        {
            float neg_entropy = 0.0f;
            for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) if (switch_mask[i]) { float p = switch_policy[i] > 1.0e-8f ? switch_policy[i] : 1.0e-8f; neg_entropy += p * logf(p); }
            for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
                if (!switch_mask[i]) { grad_logits[i] = 0.0f; continue; }
                grad_logits[i] = (switch_policy[i] - (i == switch_index ? 1.0f : 0.0f)) * policy_scale;
                if (entropy_coef != 0.0f) { float p = switch_policy[i] > 1.0e-8f ? switch_policy[i] : 1.0e-8f; grad_logits[i] += entropy_coef * p * (logf(p) - neg_entropy); }
                grad_switch_bias[i] += grad_logits[i];
                for (h = 0; h < (int)model->hidden_dim; ++h) grad_switch_head[i * model->hidden_dim + h] += grad_logits[i] * hidden_state[h];
            }
            matrix_transpose_vec_mul_accum(switch_head, grad_logits, grad_h);
        }
    }
    *accuracy_sum += accuracy_ok ? 1.0f : 0.0f;
}

typedef struct {
    float* slot0_kind_head; float* slot0_kind_bias;
    float* slot0_move_head; float* slot0_move_bias;
    float* slot0_switch_head; float* slot0_switch_bias;
    float* slot0_tera_head; float* slot0_tera_bias;
    float* slot1_kind_head; float* slot1_kind_bias;
    float* slot1_move_head; float* slot1_move_bias;
    float* slot1_switch_head; float* slot1_switch_bias;
    float* slot1_tera_head; float* slot1_tera_bias;
    float* pair_head; float* pair_bias;
} JointHeadGradients;

static void accumulate_linear_row_gradient(
    const Matrix* head,
    const float* hidden_state,
    size_t row,
    float gradient,
    float* grad_head,
    float* grad_bias,
    float* grad_h
) {
    size_t h;
    grad_bias[row] += gradient;
    for (h = 0; h < head->cols; ++h) {
        grad_head[row * head->cols + h] += gradient * hidden_state[h];
        grad_h[h] += head->data[row * head->cols + h] * gradient;
    }
}

static void accumulate_local_action_gradient(
    const GruModel* model,
    const float* hidden_state,
    int slot,
    int action,
    float gradient,
    JointHeadGradients* grads,
    float* grad_h
) {
    const Matrix* kind_head = slot == 0 ? &model->slot0_kind_head : &model->slot1_kind_head;
    const Matrix* move_head = slot == 0 ? &model->slot0_move_head : &model->slot1_move_head;
    const Matrix* switch_head = slot == 0 ? &model->slot0_switch_head : &model->slot1_switch_head;
    const Matrix* tera_head = slot == 0 ? &model->slot0_tera_head : &model->slot1_tera_head;
    float* kind_grad_head = slot == 0 ? grads->slot0_kind_head : grads->slot1_kind_head;
    float* kind_grad_bias = slot == 0 ? grads->slot0_kind_bias : grads->slot1_kind_bias;
    float* move_grad_head = slot == 0 ? grads->slot0_move_head : grads->slot1_move_head;
    float* move_grad_bias = slot == 0 ? grads->slot0_move_bias : grads->slot1_move_bias;
    float* switch_grad_head = slot == 0 ? grads->slot0_switch_head : grads->slot1_switch_head;
    float* switch_grad_bias = slot == 0 ? grads->slot0_switch_bias : grads->slot1_switch_bias;
    float* tera_grad_head = slot == 0 ? grads->slot0_tera_head : grads->slot1_tera_head;
    float* tera_grad_bias = slot == 0 ? grads->slot0_tera_bias : grads->slot1_tera_bias;
    if (action < 8) {
        accumulate_linear_row_gradient(kind_head, hidden_state, 0u, gradient, kind_grad_head, kind_grad_bias, grad_h);
        accumulate_linear_row_gradient(move_head, hidden_state, (size_t)(action & 3), gradient, move_grad_head, move_grad_bias, grad_h);
        accumulate_linear_row_gradient(tera_head, hidden_state, (size_t)(action >= 4 ? 1 : 0), gradient, tera_grad_head, tera_grad_bias, grad_h);
    } else {
        accumulate_linear_row_gradient(kind_head, hidden_state, 1u, gradient, kind_grad_head, kind_grad_bias, grad_h);
        accumulate_linear_row_gradient(switch_head, hidden_state, (size_t)(action - 8), gradient, switch_grad_head, switch_grad_bias, grad_h);
    }
}

static int joint_policy_gradients(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    const FactorizedActionChoice* choice,
    float policy_scale,
    float entropy_coef,
    float* action_loss_sum,
    float* accuracy_sum,
    float* grad_h,
    JointHeadGradients* grads
) {
    float joint_policy[FACTORIZED_JOINT_DIM];
    float grad_joint[FACTORIZED_JOINT_DIM];
    float local0_grad[FACTORIZED_LOCAL_ACTION_DIM] = {0};
    float local1_grad[FACTORIZED_LOCAL_ACTION_DIM] = {0};
    float pair_grad[FACTORIZED_PAIR_DIM] = {0};
    float neg_entropy = 0.0f;
    int flat0 = -1;
    int flat1 = -1;
    int selected;
    int best = 0;
    int i;
    if (!factorized_action_choice_to_flat_actions(choice, &flat0, &flat1) || flat0 < 0 || flat1 < 14) {
        return 0;
    }
    selected = flat0 * FACTORIZED_LOCAL_ACTION_DIM + (flat1 - 14);
    if (!gru_model_evaluate_joint_hidden(model, hidden_state, legal_mask, joint_policy, NULL) ||
            joint_policy[selected] <= 0.0f) {
        return 0;
    }
    *action_loss_sum += -2.0f * logf(joint_policy[selected] > 1.0e-8f ? joint_policy[selected] : 1.0e-8f);
    for (i = 0; i < FACTORIZED_JOINT_DIM; ++i) {
        float p = joint_policy[i] > 1.0e-8f ? joint_policy[i] : 1.0e-8f;
        if (joint_policy[i] > joint_policy[best]) best = i;
        if (joint_policy[i] > 0.0f) neg_entropy += p * logf(p);
    }
    *accuracy_sum += best == selected ? 2.0f : 0.0f;
    for (i = 0; i < FACTORIZED_JOINT_DIM; ++i) {
        int action0 = i / FACTORIZED_LOCAL_ACTION_DIM;
        int action1 = i % FACTORIZED_LOCAL_ACTION_DIM;
        float p = joint_policy[i];
        float gradient;
        if (p <= 0.0f) {
            grad_joint[i] = 0.0f;
            continue;
        }
        gradient = (p - (i == selected ? 1.0f : 0.0f)) * policy_scale;
        if (entropy_coef != 0.0f) {
            float safe_p = p > 1.0e-8f ? p : 1.0e-8f;
            gradient += entropy_coef * p * (logf(safe_p) - neg_entropy);
        }
        grad_joint[i] = gradient;
        local0_grad[action0] += gradient;
        local1_grad[action1] += gradient;
        pair_grad[joint_pair_index(action0, action1)] += gradient;
    }
    for (i = 0; i < FACTORIZED_LOCAL_ACTION_DIM; ++i) {
        accumulate_local_action_gradient(model, hidden_state, 0, i, local0_grad[i], grads, grad_h);
        accumulate_local_action_gradient(model, hidden_state, 1, i, local1_grad[i], grads, grad_h);
    }
    for (i = 0; i < FACTORIZED_PAIR_DIM; ++i) {
        accumulate_linear_row_gradient(&model->joint_pair_head, hidden_state, (size_t)i, pair_grad[i],
            grads->pair_head, grads->pair_bias, grad_h);
    }
    return 1;
}

static void factorized_target_policy_gradients(
    const GruModel* model,
    const float* hidden_state,
    int slot,
    const FactorizedActionChoice* choice,
    float policy_scale,
    float entropy_coef,
    float* action_loss_sum,
    float* grad_h,
    float* grad_target_head,
    float* grad_target_bias
) {
    unsigned char target_bits = slot == 0 ? choice->slot0_target_mask : choice->slot1_target_mask;
    int target_index = slot == 0 ? choice->slot0_target_index : choice->slot1_target_index;
    unsigned char target_mask[FACTORIZED_TARGET_DIM];
    float target_policy[FACTORIZED_TARGET_DIM];
    float neg_entropy = 0.0f;
    float grad_logits[FACTORIZED_TARGET_DIM];
    const Matrix* head = slot == 0 ? &model->slot0_target_head : &model->slot1_target_head;
    const float* bias = slot == 0 ? model->slot0_target_bias : model->slot1_target_bias;
    int i;
    if (target_bits == 0u || target_index < 0 || target_index >= FACTORIZED_TARGET_DIM ||
            !(target_bits & FACTORIZED_TARGET_BIT(target_index))) {
        return;
    }
    factorized_target_mask_to_array(target_bits, target_mask);
    evaluate_small_head(head, bias, hidden_state, target_mask, FACTORIZED_TARGET_DIM, target_policy);
    *action_loss_sum += -logf(target_policy[target_index] > 1.0e-8f ? target_policy[target_index] : 1.0e-8f);
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) if (target_mask[i]) {
        float p = target_policy[i] > 1.0e-8f ? target_policy[i] : 1.0e-8f;
        neg_entropy += p * logf(p);
    }
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) {
        if (!target_mask[i]) {
            grad_logits[i] = 0.0f;
            continue;
        }
        grad_logits[i] = (target_policy[i] - (i == target_index ? 1.0f : 0.0f)) * policy_scale;
        if (entropy_coef != 0.0f) {
            float p = target_policy[i] > 1.0e-8f ? target_policy[i] : 1.0e-8f;
            grad_logits[i] += entropy_coef * p * (logf(p) - neg_entropy);
        }
        accumulate_linear_row_gradient(head, hidden_state, (size_t)i, grad_logits[i],
            grad_target_head, grad_target_bias, grad_h);
    }
}

static void evaluate_hidden_internal(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    float* logits,
    float* policy_out,
    float* value_out
) {
    size_t i;
    if (policy_out) {
        memcpy(logits, model->policy_bias, model->num_actions * sizeof(float));
        matrix_vec_mul_accum(&model->policy_head, hidden_state, logits);
        if (legal_mask) {
            for (i = 0; i < model->num_actions; ++i) {
                if (!legal_mask[i]) {
                    logits[i] = -1.0e9f;
                }
            }
        }
        softmax(logits, model->num_actions, policy_out);
    }
    if (value_out) {
        *value_out = model->value_bias;
        for (i = 0; i < model->hidden_dim; ++i) {
            *value_out += model->value_head[i] * hidden_state[i];
        }
    }
}

static void bootstrap_factorized_heads_from_flat(GruModel* model) {
    size_t h;
    int i;
    if (!model) {
        return;
    }
    for (h = 0; h < model->hidden_dim; ++h) {
        model->slot0_kind_head.data[0 * model->hidden_dim + h] = 0.5f * (model->policy_head.data[0 * model->hidden_dim + h] + model->policy_head.data[4 * model->hidden_dim + h]);
        model->slot0_kind_head.data[1 * model->hidden_dim + h] = (model->policy_head.data[8 * model->hidden_dim + h] + model->policy_head.data[9 * model->hidden_dim + h] + model->policy_head.data[10 * model->hidden_dim + h] + model->policy_head.data[11 * model->hidden_dim + h] + model->policy_head.data[12 * model->hidden_dim + h] + model->policy_head.data[13 * model->hidden_dim + h]) / 6.0f;
        model->slot1_kind_head.data[0 * model->hidden_dim + h] = 0.5f * (model->policy_head.data[14 * model->hidden_dim + h] + model->policy_head.data[18 * model->hidden_dim + h]);
        model->slot1_kind_head.data[1 * model->hidden_dim + h] = (model->policy_head.data[22 * model->hidden_dim + h] + model->policy_head.data[23 * model->hidden_dim + h] + model->policy_head.data[24 * model->hidden_dim + h] + model->policy_head.data[25 * model->hidden_dim + h] + model->policy_head.data[26 * model->hidden_dim + h] + model->policy_head.data[27 * model->hidden_dim + h]) / 6.0f;
    }
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        memcpy(model->slot0_move_head.data + (i * model->hidden_dim), model->policy_head.data + (i * model->hidden_dim), model->hidden_dim * sizeof(float));
        memcpy(model->slot1_move_head.data + (i * model->hidden_dim), model->policy_head.data + ((14 + i) * model->hidden_dim), model->hidden_dim * sizeof(float));
        model->slot0_move_bias[i] = model->policy_bias[i];
        model->slot1_move_bias[i] = model->policy_bias[14 + i];
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        memcpy(model->slot0_switch_head.data + (i * model->hidden_dim), model->policy_head.data + ((8 + i) * model->hidden_dim), model->hidden_dim * sizeof(float));
        memcpy(model->slot1_switch_head.data + (i * model->hidden_dim), model->policy_head.data + ((22 + i) * model->hidden_dim), model->hidden_dim * sizeof(float));
        model->slot0_switch_bias[i] = model->policy_bias[8 + i];
        model->slot1_switch_bias[i] = model->policy_bias[22 + i];
    }
    model->slot0_kind_bias[0] = 0.5f * (model->policy_bias[0] + model->policy_bias[4]);
    model->slot0_kind_bias[1] = (model->policy_bias[8] + model->policy_bias[9] + model->policy_bias[10] + model->policy_bias[11] + model->policy_bias[12] + model->policy_bias[13]) / 6.0f;
    model->slot1_kind_bias[0] = 0.5f * (model->policy_bias[14] + model->policy_bias[18]);
    model->slot1_kind_bias[1] = (model->policy_bias[22] + model->policy_bias[23] + model->policy_bias[24] + model->policy_bias[25] + model->policy_bias[26] + model->policy_bias[27]) / 6.0f;
    model->slot0_tera_bias[0] = 0.0f;
    model->slot0_tera_bias[1] = 0.0f;
    model->slot1_tera_bias[0] = 0.0f;
    model->slot1_tera_bias[1] = 0.0f;
}

GruModel* gru_model_create(size_t input_dim, size_t hidden_dim, size_t num_actions) {
    GruModel* model;
    float x_scale;
    float h_scale;
    float p_scale;

    if (input_dim == 0 || hidden_dim == 0 || num_actions == 0) {
        return NULL;
    }

    model = (GruModel*)calloc(1, sizeof(*model));
    if (!model) {
        return NULL;
    }

    model->input_dim = input_dim;
    model->hidden_dim = hidden_dim;
    model->num_actions = num_actions;
    model->entity_encoder_enabled = input_dim == OBSERVATION_FLAT_SIZE ? 1 : 0;

    x_scale = 1.0f / sqrtf((float)input_dim);
    h_scale = 1.0f / sqrtf((float)hidden_dim);
    p_scale = 1.0f / sqrtf((float)hidden_dim);

    model->wzx = matrix_make(hidden_dim, input_dim, x_scale);
    model->wzh = matrix_make(hidden_dim, hidden_dim, h_scale);
    model->bz = vector_make(hidden_dim);
    model->wrx = matrix_make(hidden_dim, input_dim, x_scale);
    model->wrh = matrix_make(hidden_dim, hidden_dim, h_scale);
    model->br = vector_make(hidden_dim);
    model->wnx = matrix_make(hidden_dim, input_dim, x_scale);
    model->wnh = matrix_make(hidden_dim, hidden_dim, h_scale);
    model->bn = vector_make(hidden_dim);
    model->policy_head = matrix_make(num_actions, hidden_dim, p_scale);
    model->policy_bias = vector_make(num_actions);
    model->slot0_kind_head = matrix_make(FACTORIZED_KIND_DIM, hidden_dim, p_scale);
    model->slot0_kind_bias = vector_make(FACTORIZED_KIND_DIM);
    model->slot0_move_head = matrix_make(FACTORIZED_MOVE_DIM, hidden_dim, p_scale);
    model->slot0_move_bias = vector_make(FACTORIZED_MOVE_DIM);
    model->slot0_switch_head = matrix_make(FACTORIZED_SWITCH_DIM, hidden_dim, p_scale);
    model->slot0_switch_bias = vector_make(FACTORIZED_SWITCH_DIM);
    model->slot0_tera_head = matrix_make(FACTORIZED_TERA_DIM, hidden_dim, p_scale);
    model->slot0_tera_bias = vector_make(FACTORIZED_TERA_DIM);
    model->slot0_target_head = matrix_make(FACTORIZED_TARGET_DIM, hidden_dim, p_scale);
    model->slot0_target_bias = vector_make(FACTORIZED_TARGET_DIM);
    model->slot1_kind_head = matrix_make(FACTORIZED_KIND_DIM, hidden_dim, p_scale);
    model->slot1_kind_bias = vector_make(FACTORIZED_KIND_DIM);
    model->slot1_move_head = matrix_make(FACTORIZED_MOVE_DIM, hidden_dim, p_scale);
    model->slot1_move_bias = vector_make(FACTORIZED_MOVE_DIM);
    model->slot1_switch_head = matrix_make(FACTORIZED_SWITCH_DIM, hidden_dim, p_scale);
    model->slot1_switch_bias = vector_make(FACTORIZED_SWITCH_DIM);
    model->slot1_tera_head = matrix_make(FACTORIZED_TERA_DIM, hidden_dim, p_scale);
    model->slot1_tera_bias = vector_make(FACTORIZED_TERA_DIM);
    model->slot1_target_head = matrix_make(FACTORIZED_TARGET_DIM, hidden_dim, p_scale);
    model->slot1_target_bias = vector_make(FACTORIZED_TARGET_DIM);
    model->joint_pair_head = matrix_make(FACTORIZED_PAIR_DIM, hidden_dim, p_scale);
    model->joint_pair_bias = vector_make(FACTORIZED_PAIR_DIM);
    if (model->entity_encoder_enabled) {
        model->entity_encoder = matrix_make(GRU_ENTITY_EMBED_DIM, OBS_POKEMON_FEATURES,
            1.0f / sqrtf((float)OBS_POKEMON_FEATURES));
        model->entity_bias = vector_make(GRU_ENTITY_EMBED_DIM);
        model->entity_decoder = matrix_make(OBS_POKEMON_FEATURES, GRU_ENTITY_EMBED_DIM,
            0.05f / sqrtf((float)GRU_ENTITY_EMBED_DIM));
    }
    model->value_head = (float*)malloc(hidden_dim * sizeof(float));

    if (!model->wzx.data || !model->wzh.data || !model->bz ||
        !model->wrx.data || !model->wrh.data || !model->br ||
        !model->wnx.data || !model->wnh.data || !model->bn ||
        !model->policy_head.data || !model->policy_bias ||
        !model->slot0_kind_head.data || !model->slot0_kind_bias || !model->slot0_move_head.data || !model->slot0_move_bias ||
        !model->slot0_switch_head.data || !model->slot0_switch_bias || !model->slot0_tera_head.data || !model->slot0_tera_bias ||
        !model->slot0_target_head.data || !model->slot0_target_bias ||
        !model->slot1_kind_head.data || !model->slot1_kind_bias || !model->slot1_move_head.data || !model->slot1_move_bias ||
        !model->slot1_switch_head.data || !model->slot1_switch_bias || !model->slot1_tera_head.data || !model->slot1_tera_bias ||
        !model->slot1_target_head.data || !model->slot1_target_bias ||
        !model->joint_pair_head.data || !model->joint_pair_bias ||
        (model->entity_encoder_enabled && (!model->entity_encoder.data || !model->entity_bias || !model->entity_decoder.data)) ||
        !model->value_head) {
        gru_model_destroy(model);
        return NULL;
    }

    {
        size_t i;
        for (i = 0; i < hidden_dim; ++i) {
            model->value_head[i] = rand_uniform() * p_scale;
        }
    }
    return model;
}

void gru_model_destroy(GruModel* model) {
    if (!model) {
        return;
    }

    matrix_free(&model->wzx);
    matrix_free(&model->wzh);
    free(model->bz);
    matrix_free(&model->wrx);
    matrix_free(&model->wrh);
    free(model->br);
    matrix_free(&model->wnx);
    matrix_free(&model->wnh);
    free(model->bn);
    matrix_free(&model->policy_head);
    free(model->policy_bias);
    matrix_free(&model->slot0_kind_head); free(model->slot0_kind_bias);
    matrix_free(&model->slot0_move_head); free(model->slot0_move_bias);
    matrix_free(&model->slot0_switch_head); free(model->slot0_switch_bias);
    matrix_free(&model->slot0_tera_head); free(model->slot0_tera_bias);
    matrix_free(&model->slot0_target_head); free(model->slot0_target_bias);
    matrix_free(&model->slot1_kind_head); free(model->slot1_kind_bias);
    matrix_free(&model->slot1_move_head); free(model->slot1_move_bias);
    matrix_free(&model->slot1_switch_head); free(model->slot1_switch_bias);
    matrix_free(&model->slot1_tera_head); free(model->slot1_tera_bias);
    matrix_free(&model->slot1_target_head); free(model->slot1_target_bias);
    matrix_free(&model->joint_pair_head); free(model->joint_pair_bias);
    matrix_free(&model->entity_encoder); free(model->entity_bias); matrix_free(&model->entity_decoder);
    free(model->value_head);
    gru_gradient_accum_free(&model->grad_accum);
    gru_adam_state_free(&model->adam_state);
    gru_forward_scratch_free(&model->forward_scratch);
    gru_recurrent_scratch_free(&model->recurrent_scratch);
    free(model);
}

size_t gru_model_input_dim(const GruModel* model) {
    return model ? model->input_dim : 0;
}

size_t gru_model_hidden_dim(const GruModel* model) {
    return model ? model->hidden_dim : 0;
}

size_t gru_model_num_actions(const GruModel* model) {
    return model ? model->num_actions : 0;
}

void gru_model_zero_state(const GruModel* model, float* hidden_state_out) {
    if (!model || !hidden_state_out) {
        return;
    }
    memset(hidden_state_out, 0, model->hidden_dim * sizeof(float));
}

static int gru_model_forward_step_with_buffers(
    const GruModel* model,
    const float* input,
    const float* hidden_state_in,
    float* hidden_state_out,
    float* policy_out,
    float* value_out,
    float* z,
    float* r,
    float* n,
    float* gated_hidden,
    float* logits
) {
    size_t h;

    if (!model || !input || !hidden_state_in || !hidden_state_out ||
        !z || !r || !n || !gated_hidden || !logits) {
        return 0;
    }

    memcpy(z, model->bz, model->hidden_dim * sizeof(float));
    memcpy(r, model->br, model->hidden_dim * sizeof(float));
    matrix_vec_mul_accum(&model->wzx, input, z);
    matrix_vec_mul_accum(&model->wzh, hidden_state_in, z);
    matrix_vec_mul_accum(&model->wrx, input, r);
    matrix_vec_mul_accum(&model->wrh, hidden_state_in, r);

    for (h = 0; h < model->hidden_dim; ++h) {
        z[h] = sigmoidf_approx(z[h]);
        r[h] = sigmoidf_approx(r[h]);
        gated_hidden[h] = r[h] * hidden_state_in[h];
    }

    memcpy(n, model->bn, model->hidden_dim * sizeof(float));
    matrix_vec_mul_accum(&model->wnx, input, n);
    matrix_vec_mul_accum(&model->wnh, gated_hidden, n);
    for (h = 0; h < model->hidden_dim; ++h) {
        n[h] = tanhf(n[h]);
        hidden_state_out[h] = (1.0f - z[h]) * n[h] + z[h] * hidden_state_in[h];
    }

    evaluate_hidden_internal(model, hidden_state_out, NULL, logits, policy_out, value_out);
    return 1;
}

void gru_model_forward_step(
    const GruModel* model,
    const float* input,
    const float* hidden_state_in,
    float* hidden_state_out,
    float* policy_out,
    float* value_out
) {
    GruForwardScratch* scratch;

    if (!model || !input || !hidden_state_in || !hidden_state_out) {
        return;
    }
    if (!gru_forward_scratch_ensure((GruModel*)model)) {
        return;
    }
    scratch = &((GruModel*)model)->forward_scratch;
    transform_observation_input(model, input, scratch->transformed_input, scratch->entity_hidden);
    gru_model_forward_step_with_buffers(model, scratch->transformed_input, hidden_state_in, hidden_state_out, policy_out, value_out,
        scratch->z, scratch->r, scratch->n, scratch->gated_hidden, scratch->logits);
}

void gru_model_forward_sequence(
    const GruModel* model,
    const float* sequence,
    size_t steps,
    float* hidden_state_io,
    float* policy_out,
    float* value_out
) {
    size_t t;
    GruForwardScratch* scratch;

    if (!model || !sequence || !hidden_state_io) {
        return;
    }
    if (!gru_forward_scratch_ensure((GruModel*)model)) {
        return;
    }
    scratch = &((GruModel*)model)->forward_scratch;

    for (t = 0; t < steps; ++t) {
        const float* input = sequence + (t * model->input_dim);
        transform_observation_input(model, input, scratch->transformed_input, scratch->entity_hidden);
        if (!gru_model_forward_step_with_buffers(model, scratch->transformed_input, hidden_state_io, scratch->next_hidden, policy_out, value_out,
                scratch->z, scratch->r, scratch->n, scratch->gated_hidden, scratch->logits)) {
            return;
        }
        memcpy(hidden_state_io, scratch->next_hidden, model->hidden_dim * sizeof(float));
    }
}

int gru_model_evaluate_sequence_step(
    const GruModel* model,
    const float* sequence,
    size_t steps,
    float* policy_out,
    float* value_out
) {
    float* hidden_state;
    if (!model || !sequence || steps == 0 || !policy_out || !value_out) {
        return 0;
    }
    hidden_state = (float*)malloc(model->hidden_dim * sizeof(float));
    if (!hidden_state) {
        return 0;
    }
    gru_model_zero_state(model, hidden_state);
    gru_model_forward_sequence(model, sequence, steps, hidden_state, policy_out, value_out);
    free(hidden_state);
    return 1;
}

int gru_model_select_action(const float* policy, const unsigned char* legal_mask, size_t num_actions) {
    size_t i;
    int best_index = -1;
    float best_value = -1.0f;

    if (!policy || num_actions == 0) {
        return -1;
    }

    for (i = 0; i < num_actions; ++i) {
        if (legal_mask && !legal_mask[i]) {
            continue;
        }
        if (best_index < 0 || policy[i] > best_value) {
            best_index = (int)i;
            best_value = policy[i];
        }
    }
    return best_index;
}

int gru_model_select_action_range(
    const float* policy,
    const unsigned char* legal_mask,
    size_t start_index,
    size_t end_index,
    size_t num_actions
) {
    size_t i;
    int best_index = -1;
    float best_value = -1.0f;

    if (!policy || start_index >= num_actions || end_index >= num_actions || start_index > end_index) {
        return -1;
    }

    for (i = start_index; i <= end_index; ++i) {
        if (legal_mask && !legal_mask[i]) {
            continue;
        }
        if (best_index < 0 || policy[i] > best_value) {
            best_index = (int)i;
            best_value = policy[i];
        }
    }
    return best_index;
}

int gru_model_sample_action(
    const float* policy,
    const unsigned char* legal_mask,
    size_t num_actions
) {
    size_t i;
    float total = 0.0f;
    float draw;
    if (!policy || num_actions == 0) {
        return -1;
    }
    for (i = 0; i < num_actions; ++i) {
        if (legal_mask && !legal_mask[i]) {
            continue;
        }
        total += policy[i];
    }
    if (total <= 0.0f) {
        return gru_model_select_action(policy, legal_mask, num_actions);
    }
    draw = ((float)rand() / (float)RAND_MAX) * total;
    for (i = 0; i < num_actions; ++i) {
        if (legal_mask && !legal_mask[i]) {
            continue;
        }
        draw -= policy[i];
        if (draw <= 0.0f) {
            return (int)i;
        }
    }
    return gru_model_select_action(policy, legal_mask, num_actions);
}

int gru_model_sample_action_range(
    const float* policy,
    const unsigned char* legal_mask,
    size_t start_index,
    size_t end_index,
    size_t num_actions
) {
    size_t i;
    float total = 0.0f;
    float draw;
    if (!policy || start_index >= num_actions || end_index >= num_actions || start_index > end_index) {
        return -1;
    }
    for (i = start_index; i <= end_index; ++i) {
        if (legal_mask && !legal_mask[i]) {
            continue;
        }
        total += policy[i];
    }
    if (total <= 0.0f) {
        return gru_model_select_action_range(policy, legal_mask, start_index, end_index, num_actions);
    }
    draw = ((float)rand() / (float)RAND_MAX) * total;
    for (i = start_index; i <= end_index; ++i) {
        if (legal_mask && !legal_mask[i]) {
            continue;
        }
        draw -= policy[i];
        if (draw <= 0.0f) {
            return (int)i;
        }
    }
    return gru_model_select_action_range(policy, legal_mask, start_index, end_index, num_actions);
}

void gru_model_evaluate_hidden(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    float* policy_out,
    float* value_out
) {
    float* logits;
    if (!model || !hidden_state || !policy_out || !value_out) {
        return;
    }
    logits = (float*)malloc(model->num_actions * sizeof(float));
    if (!logits) {
        return;
    }
    evaluate_hidden_internal(model, hidden_state, legal_mask, logits, policy_out, value_out);
    free(logits);
}

int gru_model_evaluate_factorized_hidden(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    float* slot0_kind_policy,
    float* slot0_move_policy,
    float* slot0_switch_policy,
    float* slot0_tera_policy,
    float* slot0_target_policy,
    float* slot1_kind_policy,
    float* slot1_move_policy,
    float* slot1_switch_policy,
    float* slot1_tera_policy,
    float* slot1_target_policy,
    float* value_out
) {
    unsigned char slot0_kind_mask[FACTORIZED_KIND_DIM];
    unsigned char slot0_move_mask[FACTORIZED_MOVE_DIM];
    unsigned char slot0_switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char slot0_tera_mask[FACTORIZED_TERA_DIM];
    unsigned char slot1_kind_mask[FACTORIZED_KIND_DIM];
    unsigned char slot1_move_mask[FACTORIZED_MOVE_DIM];
    unsigned char slot1_switch_mask[FACTORIZED_SWITCH_DIM];
    unsigned char slot1_tera_mask[FACTORIZED_TERA_DIM];
    size_t i;
    if (!model || !hidden_state) {
        return 0;
    }
    build_factorized_masks(legal_mask, 0, slot0_kind_mask, slot0_move_mask, slot0_switch_mask, slot0_tera_mask);
    build_factorized_masks(legal_mask, 1, slot1_kind_mask, slot1_move_mask, slot1_switch_mask, slot1_tera_mask);
    if (slot0_kind_policy) evaluate_small_head(&model->slot0_kind_head, model->slot0_kind_bias, hidden_state, slot0_kind_mask, FACTORIZED_KIND_DIM, slot0_kind_policy);
    if (slot0_move_policy) evaluate_small_head(&model->slot0_move_head, model->slot0_move_bias, hidden_state, slot0_move_mask, FACTORIZED_MOVE_DIM, slot0_move_policy);
    if (slot0_switch_policy) evaluate_small_head(&model->slot0_switch_head, model->slot0_switch_bias, hidden_state, slot0_switch_mask, FACTORIZED_SWITCH_DIM, slot0_switch_policy);
    if (slot0_tera_policy) evaluate_small_head(&model->slot0_tera_head, model->slot0_tera_bias, hidden_state, slot0_tera_mask, FACTORIZED_TERA_DIM, slot0_tera_policy);
    if (slot0_target_policy) evaluate_small_head(&model->slot0_target_head, model->slot0_target_bias, hidden_state, NULL, FACTORIZED_TARGET_DIM, slot0_target_policy);
    if (slot1_kind_policy) evaluate_small_head(&model->slot1_kind_head, model->slot1_kind_bias, hidden_state, slot1_kind_mask, FACTORIZED_KIND_DIM, slot1_kind_policy);
    if (slot1_move_policy) evaluate_small_head(&model->slot1_move_head, model->slot1_move_bias, hidden_state, slot1_move_mask, FACTORIZED_MOVE_DIM, slot1_move_policy);
    if (slot1_switch_policy) evaluate_small_head(&model->slot1_switch_head, model->slot1_switch_bias, hidden_state, slot1_switch_mask, FACTORIZED_SWITCH_DIM, slot1_switch_policy);
    if (slot1_tera_policy) evaluate_small_head(&model->slot1_tera_head, model->slot1_tera_bias, hidden_state, slot1_tera_mask, FACTORIZED_TERA_DIM, slot1_tera_policy);
    if (slot1_target_policy) evaluate_small_head(&model->slot1_target_head, model->slot1_target_bias, hidden_state, NULL, FACTORIZED_TARGET_DIM, slot1_target_policy);
    if (value_out) {
        *value_out = model->value_bias;
        for (i = 0; i < model->hidden_dim; ++i) {
            *value_out += model->value_head[i] * hidden_state[i];
        }
    }
    return 1;
}

int gru_model_evaluate_joint_hidden(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    float* joint_policy,
    float* value_out
) {
    float logits[FACTORIZED_JOINT_DIM];
    int action0;
    int action1;
    int legal_count = 0;
    if (!model || !hidden_state || !legal_mask || !joint_policy) {
        return 0;
    }
    for (action0 = 0; action0 < FACTORIZED_LOCAL_ACTION_DIM; ++action0) {
        for (action1 = 0; action1 < FACTORIZED_LOCAL_ACTION_DIM; ++action1) {
            size_t index = (size_t)action0 * FACTORIZED_LOCAL_ACTION_DIM + (size_t)action1;
            int legal = legal_mask[action0] && legal_mask[14 + action1] &&
                joint_local_pair_legal(action0, action1);
            if (legal) {
                size_t pair = joint_pair_index(action0, action1);
                logits[index] = factorized_local_action_logit(model, hidden_state, 0, action0) +
                    factorized_local_action_logit(model, hidden_state, 1, action1) +
                    linear_row_logit(&model->joint_pair_head, model->joint_pair_bias, pair, hidden_state);
                ++legal_count;
            } else {
                logits[index] = -1.0e9f;
            }
        }
    }
    if (legal_count == 0) {
        memset(joint_policy, 0, FACTORIZED_JOINT_DIM * sizeof(float));
        return 0;
    }
    softmax(logits, FACTORIZED_JOINT_DIM, joint_policy);
    if (value_out) {
        size_t h;
        *value_out = model->value_bias;
        for (h = 0; h < model->hidden_dim; ++h) {
            *value_out += model->value_head[h] * hidden_state[h];
        }
    }
    return 1;
}

static int recurrent_update_sequence(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_secondary,
    int target_action_secondary,
    const unsigned char* legal_mask,
    int target_action,
    float policy_scale,
    float target_value,
    float entropy_coef,
    const float* anchor_policy,
    float anchor_kl_coef,
    float learning_rate,
    int accumulate_only,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out,
    const FactorizedActionChoice* factorized_choice
) {
    size_t hdim;
    size_t xdim;
    size_t adim;
    size_t t;
    size_t h;
    size_t a;
    GruRecurrentScratch* scratch = NULL;
    float* h_states = NULL;
    float* z_cache = NULL;
    float* r_cache = NULL;
    float* n_cache = NULL;
    float* gated_cache = NULL;
    float* zero_hidden = NULL;
    float* logits = NULL;
    float* policy = NULL;
    float* grad_h = NULL;
    float* next_grad_h = NULL;
    float* grad_logits = NULL;
    float* d_gated = NULL;
    float* d_pre_z = NULL;
    float* d_pre_r = NULL;
    float* d_pre_n = NULL;
    float* grad_wzx = NULL; float* grad_wzh = NULL; float* grad_bz = NULL;
    float* grad_wrx = NULL; float* grad_wrh = NULL; float* grad_br = NULL;
    float* grad_wnx = NULL; float* grad_wnh = NULL; float* grad_bn = NULL;
    float* grad_policy_head = NULL; float* grad_policy_bias = NULL;
    float* grad_value_head = NULL;
    float* transformed_inputs = NULL;
    float* entity_hidden_cache = NULL;
    float* grad_input = NULL;
    float* grad_entity_encoder = NULL;
    float* grad_entity_bias = NULL;
    float* grad_entity_decoder = NULL;
    float* grad_slot0_kind_head = NULL; float* grad_slot0_kind_bias = NULL;
    float* grad_slot0_move_head = NULL; float* grad_slot0_move_bias = NULL;
    float* grad_slot0_switch_head = NULL; float* grad_slot0_switch_bias = NULL;
    float* grad_slot0_tera_head = NULL; float* grad_slot0_tera_bias = NULL;
    float* grad_slot0_target_head = NULL; float* grad_slot0_target_bias = NULL;
    float* grad_slot1_kind_head = NULL; float* grad_slot1_kind_bias = NULL;
    float* grad_slot1_move_head = NULL; float* grad_slot1_move_bias = NULL;
    float* grad_slot1_switch_head = NULL; float* grad_slot1_switch_bias = NULL;
    float* grad_slot1_tera_head = NULL; float* grad_slot1_tera_bias = NULL;
    float* grad_slot1_target_head = NULL; float* grad_slot1_target_bias = NULL;
    float* grad_joint_pair_head = NULL; float* grad_joint_pair_bias = NULL;
    float grad_value_bias = 0.0f;
    float value = 0.0f;
    float dv;
    size_t target_count = 1;
    float accuracy_sum = 0.0f;
    float action_loss_sum = 0.0f;
    const float* start_hidden;
    GruGradientAccum* accum = NULL;
    unsigned char combined_legal[OBS_NUM_ACTIONS];
    FactorizedActionChoice target_choice;
    JointHeadGradients joint_grads;

    if (!model || !sequence || steps == 0 || target_action < 0 || (size_t)target_action >= model->num_actions) {
        return 0;
    }
    hdim = model->hidden_dim;
    xdim = model->input_dim;
    adim = model->num_actions;
    if (accumulate_only) {
        if (!gru_gradient_accum_ensure(model)) {
            return 0;
        }
        accum = &model->grad_accum;
    }

    if (!gru_recurrent_scratch_ensure(model, steps)) {
        return 0;
    }
    scratch = &model->recurrent_scratch;
    h_states = scratch->h_states;
    z_cache = scratch->z_cache;
    r_cache = scratch->r_cache;
    n_cache = scratch->n_cache;
    gated_cache = scratch->gated_cache;
    zero_hidden = scratch->zero_hidden;
    logits = scratch->logits;
    policy = scratch->policy;
    grad_h = scratch->grad_h;
    next_grad_h = scratch->next_grad_h;
    grad_logits = scratch->grad_logits;
    d_gated = scratch->d_gated;
    d_pre_z = scratch->d_pre_z;
    d_pre_r = scratch->d_pre_r;
    d_pre_n = scratch->d_pre_n;
    grad_wzx = scratch->grad_wzx;
    grad_wzh = scratch->grad_wzh;
    grad_bz = scratch->grad_bz;
    grad_wrx = scratch->grad_wrx;
    grad_wrh = scratch->grad_wrh;
    grad_br = scratch->grad_br;
    grad_wnx = scratch->grad_wnx;
    grad_wnh = scratch->grad_wnh;
    grad_bn = scratch->grad_bn;
    grad_policy_head = scratch->grad_policy_head;
    grad_policy_bias = scratch->grad_policy_bias;
    grad_value_head = scratch->grad_value_head;
    transformed_inputs = scratch->transformed_inputs;
    entity_hidden_cache = scratch->entity_hidden;
    grad_input = scratch->grad_input;
    grad_entity_encoder = scratch->grad_entity_encoder;
    grad_entity_bias = scratch->grad_entity_bias;
    grad_entity_decoder = scratch->grad_entity_decoder;
    memset(h_states, 0, steps * hdim * sizeof(float));
    memset(z_cache, 0, steps * hdim * sizeof(float));
    memset(r_cache, 0, steps * hdim * sizeof(float));
    memset(n_cache, 0, steps * hdim * sizeof(float));
    memset(gated_cache, 0, steps * hdim * sizeof(float));
    memset(zero_hidden, 0, hdim * sizeof(float));
    memset(logits, 0, adim * sizeof(float));
    memset(policy, 0, adim * sizeof(float));
    memset(grad_h, 0, hdim * sizeof(float));
    memset(next_grad_h, 0, hdim * sizeof(float));
    memset(grad_logits, 0, adim * sizeof(float));
    memset(d_gated, 0, hdim * sizeof(float));
    memset(d_pre_z, 0, hdim * sizeof(float));
    memset(d_pre_r, 0, hdim * sizeof(float));
    memset(d_pre_n, 0, hdim * sizeof(float));
    memset(grad_wzx, 0, model->wzx.rows * model->wzx.cols * sizeof(float));
    memset(grad_wzh, 0, model->wzh.rows * model->wzh.cols * sizeof(float));
    memset(grad_bz, 0, hdim * sizeof(float));
    memset(grad_wrx, 0, model->wrx.rows * model->wrx.cols * sizeof(float));
    memset(grad_wrh, 0, model->wrh.rows * model->wrh.cols * sizeof(float));
    memset(grad_br, 0, hdim * sizeof(float));
    memset(grad_wnx, 0, model->wnx.rows * model->wnx.cols * sizeof(float));
    memset(grad_wnh, 0, model->wnh.rows * model->wnh.cols * sizeof(float));
    memset(grad_bn, 0, hdim * sizeof(float));
    memset(grad_policy_head, 0, model->policy_head.rows * model->policy_head.cols * sizeof(float));
    memset(grad_policy_bias, 0, adim * sizeof(float));
    memset(grad_value_head, 0, hdim * sizeof(float));
    memset(transformed_inputs, 0, steps * xdim * sizeof(float));
    memset(entity_hidden_cache, 0, steps * 2u * OBS_TEAM_SIZE * GRU_ENTITY_EMBED_DIM * sizeof(float));
    memset(grad_input, 0, xdim * sizeof(float));
    if (model->entity_encoder_enabled) {
        memset(grad_entity_encoder, 0, model->entity_encoder.rows * model->entity_encoder.cols * sizeof(float));
        memset(grad_entity_bias, 0, GRU_ENTITY_EMBED_DIM * sizeof(float));
        memset(grad_entity_decoder, 0, model->entity_decoder.rows * model->entity_decoder.cols * sizeof(float));
    }
    grad_slot0_kind_head = (float*)calloc(model->slot0_kind_head.rows * model->slot0_kind_head.cols, sizeof(float));
    grad_slot0_kind_bias = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    grad_slot0_move_head = (float*)calloc(model->slot0_move_head.rows * model->slot0_move_head.cols, sizeof(float));
    grad_slot0_move_bias = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    grad_slot0_switch_head = (float*)calloc(model->slot0_switch_head.rows * model->slot0_switch_head.cols, sizeof(float));
    grad_slot0_switch_bias = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    grad_slot0_tera_head = (float*)calloc(model->slot0_tera_head.rows * model->slot0_tera_head.cols, sizeof(float));
    grad_slot0_tera_bias = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    grad_slot0_target_head = (float*)calloc(model->slot0_target_head.rows * model->slot0_target_head.cols, sizeof(float));
    grad_slot0_target_bias = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    grad_slot1_kind_head = (float*)calloc(model->slot1_kind_head.rows * model->slot1_kind_head.cols, sizeof(float));
    grad_slot1_kind_bias = (float*)calloc(FACTORIZED_KIND_DIM, sizeof(float));
    grad_slot1_move_head = (float*)calloc(model->slot1_move_head.rows * model->slot1_move_head.cols, sizeof(float));
    grad_slot1_move_bias = (float*)calloc(FACTORIZED_MOVE_DIM, sizeof(float));
    grad_slot1_switch_head = (float*)calloc(model->slot1_switch_head.rows * model->slot1_switch_head.cols, sizeof(float));
    grad_slot1_switch_bias = (float*)calloc(FACTORIZED_SWITCH_DIM, sizeof(float));
    grad_slot1_tera_head = (float*)calloc(model->slot1_tera_head.rows * model->slot1_tera_head.cols, sizeof(float));
    grad_slot1_tera_bias = (float*)calloc(FACTORIZED_TERA_DIM, sizeof(float));
    grad_slot1_target_head = (float*)calloc(model->slot1_target_head.rows * model->slot1_target_head.cols, sizeof(float));
    grad_slot1_target_bias = (float*)calloc(FACTORIZED_TARGET_DIM, sizeof(float));
    grad_joint_pair_head = (float*)calloc(model->joint_pair_head.rows * model->joint_pair_head.cols, sizeof(float));
    grad_joint_pair_bias = (float*)calloc(FACTORIZED_PAIR_DIM, sizeof(float));
    if (!grad_slot0_kind_head || !grad_slot0_kind_bias || !grad_slot0_move_head || !grad_slot0_move_bias || !grad_slot0_switch_head || !grad_slot0_switch_bias || !grad_slot0_tera_head || !grad_slot0_tera_bias || !grad_slot0_target_head || !grad_slot0_target_bias ||
            !grad_slot1_kind_head || !grad_slot1_kind_bias || !grad_slot1_move_head || !grad_slot1_move_bias || !grad_slot1_switch_head || !grad_slot1_switch_bias || !grad_slot1_tera_head || !grad_slot1_tera_bias || !grad_slot1_target_head || !grad_slot1_target_bias ||
            !grad_joint_pair_head || !grad_joint_pair_bias) {
        free(grad_slot0_kind_head); free(grad_slot0_kind_bias); free(grad_slot0_move_head); free(grad_slot0_move_bias); free(grad_slot0_switch_head); free(grad_slot0_switch_bias); free(grad_slot0_tera_head); free(grad_slot0_tera_bias); free(grad_slot0_target_head); free(grad_slot0_target_bias);
        free(grad_slot1_kind_head); free(grad_slot1_kind_bias); free(grad_slot1_move_head); free(grad_slot1_move_bias); free(grad_slot1_switch_head); free(grad_slot1_switch_bias); free(grad_slot1_tera_head); free(grad_slot1_tera_bias); free(grad_slot1_target_head); free(grad_slot1_target_bias);
        free(grad_joint_pair_head); free(grad_joint_pair_bias);
        return 0;
    }
    memset(&joint_grads, 0, sizeof(joint_grads));
    joint_grads.slot0_kind_head = grad_slot0_kind_head; joint_grads.slot0_kind_bias = grad_slot0_kind_bias;
    joint_grads.slot0_move_head = grad_slot0_move_head; joint_grads.slot0_move_bias = grad_slot0_move_bias;
    joint_grads.slot0_switch_head = grad_slot0_switch_head; joint_grads.slot0_switch_bias = grad_slot0_switch_bias;
    joint_grads.slot0_tera_head = grad_slot0_tera_head; joint_grads.slot0_tera_bias = grad_slot0_tera_bias;
    joint_grads.slot1_kind_head = grad_slot1_kind_head; joint_grads.slot1_kind_bias = grad_slot1_kind_bias;
    joint_grads.slot1_move_head = grad_slot1_move_head; joint_grads.slot1_move_bias = grad_slot1_move_bias;
    joint_grads.slot1_switch_head = grad_slot1_switch_head; joint_grads.slot1_switch_bias = grad_slot1_switch_bias;
    joint_grads.slot1_tera_head = grad_slot1_tera_head; joint_grads.slot1_tera_bias = grad_slot1_tera_bias;
    joint_grads.pair_head = grad_joint_pair_head; joint_grads.pair_bias = grad_joint_pair_bias;

    start_hidden = initial_hidden_state ? initial_hidden_state : zero_hidden;
    if (target_action_secondary >= 0) {
        target_count = 2;
    }
    memset(combined_legal, 0, sizeof(combined_legal));
    if (legal_mask) memcpy(combined_legal, legal_mask, OBS_NUM_ACTIONS * sizeof(unsigned char));
    if (legal_mask_secondary) {
        for (a = 0; a < OBS_NUM_ACTIONS; ++a) {
            if (legal_mask_secondary[a]) combined_legal[a] = 1;
        }
    }
    if (factorized_choice) {
        target_choice = *factorized_choice;
    } else {
        factorized_action_choice_from_flat_actions(&target_choice, target_action, target_action_secondary >= 0 ? target_action_secondary : -1);
    }

    for (t = 0; t < steps; ++t) {
        const float* raw_input = sequence + (t * xdim);
        float* input = transformed_inputs + (t * xdim);
        const float* prev_h = (t > 0) ? (h_states + ((t - 1) * hdim)) : start_hidden;
        float* z_t = z_cache + (t * hdim);
        float* r_t = r_cache + (t * hdim);
        float* n_t = n_cache + (t * hdim);
        float* g_t = gated_cache + (t * hdim);
        float* h_t = h_states + (t * hdim);
        transform_observation_input(model, raw_input, input,
            entity_hidden_cache + t * 2u * OBS_TEAM_SIZE * GRU_ENTITY_EMBED_DIM);
        memcpy(z_t, model->bz, hdim * sizeof(float));
        memcpy(r_t, model->br, hdim * sizeof(float));
        matrix_vec_mul_accum(&model->wzx, input, z_t);
        matrix_vec_mul_accum(&model->wzh, prev_h, z_t);
        matrix_vec_mul_accum(&model->wrx, input, r_t);
        matrix_vec_mul_accum(&model->wrh, prev_h, r_t);
        for (h = 0; h < hdim; ++h) {
            z_t[h] = sigmoidf_approx(z_t[h]);
            r_t[h] = sigmoidf_approx(r_t[h]);
            g_t[h] = r_t[h] * prev_h[h];
        }
        memcpy(n_t, model->bn, hdim * sizeof(float));
        matrix_vec_mul_accum(&model->wnx, input, n_t);
        matrix_vec_mul_accum(&model->wnh, g_t, n_t);
        for (h = 0; h < hdim; ++h) {
            n_t[h] = tanhf(n_t[h]);
            h_t[h] = (1.0f - z_t[h]) * n_t[h] + z_t[h] * prev_h[h];
        }
    }

    value = model->value_bias;
    for (h = 0; h < hdim; ++h) {
        value += model->value_head[h] * h_states[(steps - 1) * hdim + h];
    }

    if (anchor_policy) {
        evaluate_hidden_internal(model, h_states + ((steps - 1) * hdim), legal_mask, logits, policy, &value);
        action_loss_sum += -logf(policy[target_action] > 1.0e-8f ? policy[target_action] : 1.0e-8f);
        accuracy_sum += (gru_model_select_action(policy, legal_mask, adim) == target_action) ? 1.0f : 0.0f;
        {
        float neg_entropy = 0.0f;
        for (a = 0; a < adim; ++a) {
            float p;
            if (legal_mask && !legal_mask[a]) {
                continue;
            }
            p = policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f;
            neg_entropy += p * logf(p);
        }
        for (a = 0; a < adim; ++a) {
                grad_logits[a] = (policy[a] - ((int)a == target_action ? 1.0f : 0.0f)) * policy_scale;
                if (entropy_coef != 0.0f && (!legal_mask || legal_mask[a])) {
                    float p = policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f;
                    grad_logits[a] += entropy_coef * p * (logf(p) - neg_entropy);
                }
                if (anchor_policy && anchor_kl_coef > 0.0f && (!legal_mask || legal_mask[a])) {
                    grad_logits[a] += anchor_kl_coef * (policy[a] - anchor_policy[a]);
                }
                grad_policy_bias[a] += grad_logits[a];
                for (h = 0; h < hdim; ++h) {
                    grad_policy_head[a * hdim + h] += grad_logits[a] * h_states[(steps - 1) * hdim + h];
                }
            }
        }
        matrix_transpose_vec_mul_accum(&model->policy_head, grad_logits, grad_h);

        if (target_action_secondary >= 0) {
        evaluate_hidden_internal(model, h_states + ((steps - 1) * hdim), legal_mask_secondary, logits, policy, &value);
        action_loss_sum += -logf(policy[target_action_secondary] > 1.0e-8f ? policy[target_action_secondary] : 1.0e-8f);
        accuracy_sum += (gru_model_select_action(policy, legal_mask_secondary, adim) == target_action_secondary) ? 1.0f : 0.0f;
        {
            float neg_entropy = 0.0f;
            for (a = 0; a < adim; ++a) {
                float p;
                if (legal_mask_secondary && !legal_mask_secondary[a]) {
                    continue;
                }
                p = policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f;
                neg_entropy += p * logf(p);
            }
            for (a = 0; a < adim; ++a) {
                grad_logits[a] = (policy[a] - ((int)a == target_action_secondary ? 1.0f : 0.0f)) * policy_scale;
                if (entropy_coef != 0.0f && (!legal_mask_secondary || legal_mask_secondary[a])) {
                    float p = policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f;
                    grad_logits[a] += entropy_coef * p * (logf(p) - neg_entropy);
                }
                if (anchor_policy && anchor_kl_coef > 0.0f && (!legal_mask_secondary || legal_mask_secondary[a])) {
                    grad_logits[a] += anchor_kl_coef * (policy[a] - anchor_policy[a]);
                }
                grad_policy_bias[a] += grad_logits[a];
                for (h = 0; h < hdim; ++h) {
                    grad_policy_head[a * hdim + h] += grad_logits[a] * h_states[(steps - 1) * hdim + h];
                }
            }
        }
            matrix_transpose_vec_mul_accum(&model->policy_head, grad_logits, grad_h);
        }
    } else if (target_choice.slot0_has_action && target_choice.slot1_has_action) {
        if (!joint_policy_gradients(model, h_states + ((steps - 1) * hdim), combined_legal,
                &target_choice, policy_scale, entropy_coef, &action_loss_sum, &accuracy_sum,
                grad_h, &joint_grads)) {
            free(grad_slot0_kind_head); free(grad_slot0_kind_bias); free(grad_slot0_move_head); free(grad_slot0_move_bias); free(grad_slot0_switch_head); free(grad_slot0_switch_bias); free(grad_slot0_tera_head); free(grad_slot0_tera_bias); free(grad_slot0_target_head); free(grad_slot0_target_bias);
            free(grad_slot1_kind_head); free(grad_slot1_kind_bias); free(grad_slot1_move_head); free(grad_slot1_move_bias); free(grad_slot1_switch_head); free(grad_slot1_switch_bias); free(grad_slot1_tera_head); free(grad_slot1_tera_bias); free(grad_slot1_target_head); free(grad_slot1_target_bias);
            free(grad_joint_pair_head); free(grad_joint_pair_bias);
            return 0;
        }
        if (target_choice.slot0_kind == FACTORIZED_ACTION_MOVE) {
            factorized_target_policy_gradients(model, h_states + ((steps - 1) * hdim), 0,
                &target_choice, policy_scale, entropy_coef, &action_loss_sum, grad_h,
                grad_slot0_target_head, grad_slot0_target_bias);
        }
        if (target_choice.slot1_kind == FACTORIZED_ACTION_MOVE) {
            factorized_target_policy_gradients(model, h_states + ((steps - 1) * hdim), 1,
                &target_choice, policy_scale, entropy_coef, &action_loss_sum, grad_h,
                grad_slot1_target_head, grad_slot1_target_bias);
        }
    } else {
        if (target_choice.slot0_has_action) {
            factorized_slot_policy_gradients(model, h_states + ((steps - 1) * hdim), combined_legal, 0, &target_choice, policy_scale, entropy_coef,
                &action_loss_sum, &accuracy_sum, grad_h,
                grad_slot0_kind_head, grad_slot0_kind_bias,
                grad_slot0_move_head, grad_slot0_move_bias,
                grad_slot0_switch_head, grad_slot0_switch_bias,
                grad_slot0_tera_head, grad_slot0_tera_bias,
                grad_slot0_target_head, grad_slot0_target_bias);
        }
        if (target_choice.slot1_has_action) {
            factorized_slot_policy_gradients(model, h_states + ((steps - 1) * hdim), combined_legal, 1, &target_choice, policy_scale, entropy_coef,
                &action_loss_sum, &accuracy_sum, grad_h,
                grad_slot1_kind_head, grad_slot1_kind_bias,
                grad_slot1_move_head, grad_slot1_move_bias,
                grad_slot1_switch_head, grad_slot1_switch_bias,
                grad_slot1_tera_head, grad_slot1_tera_bias,
                grad_slot1_target_head, grad_slot1_target_bias);
        }
    }

    dv = value - target_value;
    grad_value_bias += dv * (float)target_count;
    for (h = 0; h < hdim; ++h) {
        grad_value_head[h] += dv * (float)target_count * h_states[(steps - 1) * hdim + h];
        grad_h[h] += model->value_head[h] * dv * (float)target_count;
    }
    if (action_loss_out) {
        *action_loss_out = action_loss_sum / (float)target_count;
    }
    if (value_loss_out) {
        *value_loss_out = 0.5f * dv * dv;
    }
    if (accuracy_out) {
        *accuracy_out = accuracy_sum / (float)target_count;
    }

    for (t = steps; t-- > 0;) {
        const float* raw_input = sequence + (t * xdim);
        const float* input = transformed_inputs + (t * xdim);
        const float* prev_h = (t > 0) ? (h_states + ((t - 1) * hdim)) : start_hidden;
        const float* z_t = z_cache + (t * hdim);
        const float* r_t = r_cache + (t * hdim);
        const float* n_t = n_cache + (t * hdim);
        const float* g_t = gated_cache + (t * hdim);
        memset(next_grad_h, 0, hdim * sizeof(float));
        memset(d_gated, 0, hdim * sizeof(float));
        for (h = 0; h < hdim; ++h) {
            float d_n = grad_h[h] * (1.0f - z_t[h]);
            float d_z = grad_h[h] * (prev_h[h] - n_t[h]);
            next_grad_h[h] += grad_h[h] * z_t[h];
            d_pre_n[h] = d_n * (1.0f - n_t[h] * n_t[h]);
            d_pre_z[h] = d_z * z_t[h] * (1.0f - z_t[h]);
        }
        outer_product_accum(grad_wnx, hdim, xdim, d_pre_n, input);
        outer_product_accum(grad_wnh, hdim, hdim, d_pre_n, g_t);
        for (h = 0; h < hdim; ++h) {
            grad_bn[h] += d_pre_n[h];
        }
        matrix_transpose_vec_mul_accum(&model->wnh, d_pre_n, d_gated);
        for (h = 0; h < hdim; ++h) {
            float d_r = d_gated[h] * prev_h[h];
            next_grad_h[h] += d_gated[h] * r_t[h];
            d_pre_r[h] = d_r * r_t[h] * (1.0f - r_t[h]);
        }
        outer_product_accum(grad_wrx, hdim, xdim, d_pre_r, input);
        outer_product_accum(grad_wrh, hdim, hdim, d_pre_r, prev_h);
        for (h = 0; h < hdim; ++h) {
            grad_br[h] += d_pre_r[h];
            grad_bz[h] += d_pre_z[h];
        }
        outer_product_accum(grad_wzx, hdim, xdim, d_pre_z, input);
        outer_product_accum(grad_wzh, hdim, hdim, d_pre_z, prev_h);
        matrix_transpose_vec_mul_accum(&model->wrh, d_pre_r, next_grad_h);
        matrix_transpose_vec_mul_accum(&model->wzh, d_pre_z, next_grad_h);
        if (model->entity_encoder_enabled) {
            memset(grad_input, 0, xdim * sizeof(float));
            matrix_transpose_vec_mul_accum(&model->wzx, d_pre_z, grad_input);
            matrix_transpose_vec_mul_accum(&model->wrx, d_pre_r, grad_input);
            matrix_transpose_vec_mul_accum(&model->wnx, d_pre_n, grad_input);
            backprop_entity_encoder(model, raw_input,
                entity_hidden_cache + t * 2u * OBS_TEAM_SIZE * GRU_ENTITY_EMBED_DIM,
                grad_input, grad_entity_encoder, grad_entity_bias, grad_entity_decoder);
        }
        memcpy(grad_h, next_grad_h, hdim * sizeof(float));
    }

    if (accumulate_only) {
#ifdef _OPENMP
        #pragma omp parallel for if(model->wzx.rows * model->wzx.cols >= 512)
#endif
        for (h = 0; h < model->wzx.rows * model->wzx.cols; ++h) accum->wzx[h] += grad_wzx[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wzh.rows * model->wzh.cols >= 512)
#endif
        for (h = 0; h < model->wzh.rows * model->wzh.cols; ++h) accum->wzh[h] += grad_wzh[h];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) accum->bz[h] += grad_bz[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wrx.rows * model->wrx.cols >= 512)
#endif
        for (h = 0; h < model->wrx.rows * model->wrx.cols; ++h) accum->wrx[h] += grad_wrx[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wrh.rows * model->wrh.cols >= 512)
#endif
        for (h = 0; h < model->wrh.rows * model->wrh.cols; ++h) accum->wrh[h] += grad_wrh[h];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) accum->br[h] += grad_br[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wnx.rows * model->wnx.cols >= 512)
#endif
        for (h = 0; h < model->wnx.rows * model->wnx.cols; ++h) accum->wnx[h] += grad_wnx[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wnh.rows * model->wnh.cols >= 512)
#endif
        for (h = 0; h < model->wnh.rows * model->wnh.cols; ++h) accum->wnh[h] += grad_wnh[h];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) accum->bn[h] += grad_bn[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->policy_head.rows * model->policy_head.cols >= 512)
#endif
        for (a = 0; a < model->policy_head.rows * model->policy_head.cols; ++a) accum->policy_head[a] += grad_policy_head[a];
#ifdef _OPENMP
        #pragma omp parallel for if(adim >= 16)
#endif
        for (a = 0; a < adim; ++a) accum->policy_bias[a] += grad_policy_bias[a];
        for (a = 0; a < model->slot0_kind_head.rows * model->slot0_kind_head.cols; ++a) accum->slot0_kind_head[a] += grad_slot0_kind_head[a];
        for (a = 0; a < FACTORIZED_KIND_DIM; ++a) accum->slot0_kind_bias[a] += grad_slot0_kind_bias[a];
        for (a = 0; a < model->slot0_move_head.rows * model->slot0_move_head.cols; ++a) accum->slot0_move_head[a] += grad_slot0_move_head[a];
        for (a = 0; a < FACTORIZED_MOVE_DIM; ++a) accum->slot0_move_bias[a] += grad_slot0_move_bias[a];
        for (a = 0; a < model->slot0_switch_head.rows * model->slot0_switch_head.cols; ++a) accum->slot0_switch_head[a] += grad_slot0_switch_head[a];
        for (a = 0; a < FACTORIZED_SWITCH_DIM; ++a) accum->slot0_switch_bias[a] += grad_slot0_switch_bias[a];
        for (a = 0; a < model->slot0_tera_head.rows * model->slot0_tera_head.cols; ++a) accum->slot0_tera_head[a] += grad_slot0_tera_head[a];
        for (a = 0; a < FACTORIZED_TERA_DIM; ++a) accum->slot0_tera_bias[a] += grad_slot0_tera_bias[a];
        for (a = 0; a < model->slot0_target_head.rows * model->slot0_target_head.cols; ++a) accum->slot0_target_head[a] += grad_slot0_target_head[a];
        for (a = 0; a < FACTORIZED_TARGET_DIM; ++a) accum->slot0_target_bias[a] += grad_slot0_target_bias[a];
        for (a = 0; a < model->slot1_kind_head.rows * model->slot1_kind_head.cols; ++a) accum->slot1_kind_head[a] += grad_slot1_kind_head[a];
        for (a = 0; a < FACTORIZED_KIND_DIM; ++a) accum->slot1_kind_bias[a] += grad_slot1_kind_bias[a];
        for (a = 0; a < model->slot1_move_head.rows * model->slot1_move_head.cols; ++a) accum->slot1_move_head[a] += grad_slot1_move_head[a];
        for (a = 0; a < FACTORIZED_MOVE_DIM; ++a) accum->slot1_move_bias[a] += grad_slot1_move_bias[a];
        for (a = 0; a < model->slot1_switch_head.rows * model->slot1_switch_head.cols; ++a) accum->slot1_switch_head[a] += grad_slot1_switch_head[a];
        for (a = 0; a < FACTORIZED_SWITCH_DIM; ++a) accum->slot1_switch_bias[a] += grad_slot1_switch_bias[a];
        for (a = 0; a < model->slot1_tera_head.rows * model->slot1_tera_head.cols; ++a) accum->slot1_tera_head[a] += grad_slot1_tera_head[a];
        for (a = 0; a < FACTORIZED_TERA_DIM; ++a) accum->slot1_tera_bias[a] += grad_slot1_tera_bias[a];
        for (a = 0; a < model->slot1_target_head.rows * model->slot1_target_head.cols; ++a) accum->slot1_target_head[a] += grad_slot1_target_head[a];
        for (a = 0; a < FACTORIZED_TARGET_DIM; ++a) accum->slot1_target_bias[a] += grad_slot1_target_bias[a];
        for (a = 0; a < model->joint_pair_head.rows * model->joint_pair_head.cols; ++a) accum->joint_pair_head[a] += grad_joint_pair_head[a];
        for (a = 0; a < FACTORIZED_PAIR_DIM; ++a) accum->joint_pair_bias[a] += grad_joint_pair_bias[a];
        for (a = 0; a < model->entity_encoder.rows * model->entity_encoder.cols; ++a) accum->entity_encoder[a] += grad_entity_encoder[a];
        for (a = 0; a < (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u); ++a) accum->entity_bias[a] += grad_entity_bias[a];
        for (a = 0; a < model->entity_decoder.rows * model->entity_decoder.cols; ++a) accum->entity_decoder[a] += grad_entity_decoder[a];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) accum->value_head[h] += grad_value_head[h];
        accum->value_bias += grad_value_bias;
        accum->count += 1;
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(model->wzx.rows * model->wzx.cols >= 512)
#endif
        for (h = 0; h < model->wzx.rows * model->wzx.cols; ++h) model->wzx.data[h] -= learning_rate * grad_wzx[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wzh.rows * model->wzh.cols >= 512)
#endif
        for (h = 0; h < model->wzh.rows * model->wzh.cols; ++h) model->wzh.data[h] -= learning_rate * grad_wzh[h];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) model->bz[h] -= learning_rate * grad_bz[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wrx.rows * model->wrx.cols >= 512)
#endif
        for (h = 0; h < model->wrx.rows * model->wrx.cols; ++h) model->wrx.data[h] -= learning_rate * grad_wrx[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wrh.rows * model->wrh.cols >= 512)
#endif
        for (h = 0; h < model->wrh.rows * model->wrh.cols; ++h) model->wrh.data[h] -= learning_rate * grad_wrh[h];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) model->br[h] -= learning_rate * grad_br[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wnx.rows * model->wnx.cols >= 512)
#endif
        for (h = 0; h < model->wnx.rows * model->wnx.cols; ++h) model->wnx.data[h] -= learning_rate * grad_wnx[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->wnh.rows * model->wnh.cols >= 512)
#endif
        for (h = 0; h < model->wnh.rows * model->wnh.cols; ++h) model->wnh.data[h] -= learning_rate * grad_wnh[h];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) model->bn[h] -= learning_rate * grad_bn[h];
#ifdef _OPENMP
        #pragma omp parallel for if(model->policy_head.rows * model->policy_head.cols >= 512)
#endif
        for (a = 0; a < model->policy_head.rows * model->policy_head.cols; ++a) model->policy_head.data[a] -= learning_rate * grad_policy_head[a];
#ifdef _OPENMP
        #pragma omp parallel for if(adim >= 16)
#endif
        for (a = 0; a < adim; ++a) model->policy_bias[a] -= learning_rate * grad_policy_bias[a];
        for (a = 0; a < model->slot0_kind_head.rows * model->slot0_kind_head.cols; ++a) model->slot0_kind_head.data[a] -= learning_rate * grad_slot0_kind_head[a];
        for (a = 0; a < FACTORIZED_KIND_DIM; ++a) model->slot0_kind_bias[a] -= learning_rate * grad_slot0_kind_bias[a];
        for (a = 0; a < model->slot0_move_head.rows * model->slot0_move_head.cols; ++a) model->slot0_move_head.data[a] -= learning_rate * grad_slot0_move_head[a];
        for (a = 0; a < FACTORIZED_MOVE_DIM; ++a) model->slot0_move_bias[a] -= learning_rate * grad_slot0_move_bias[a];
        for (a = 0; a < model->slot0_switch_head.rows * model->slot0_switch_head.cols; ++a) model->slot0_switch_head.data[a] -= learning_rate * grad_slot0_switch_head[a];
        for (a = 0; a < FACTORIZED_SWITCH_DIM; ++a) model->slot0_switch_bias[a] -= learning_rate * grad_slot0_switch_bias[a];
        for (a = 0; a < model->slot0_tera_head.rows * model->slot0_tera_head.cols; ++a) model->slot0_tera_head.data[a] -= learning_rate * grad_slot0_tera_head[a];
        for (a = 0; a < FACTORIZED_TERA_DIM; ++a) model->slot0_tera_bias[a] -= learning_rate * grad_slot0_tera_bias[a];
        for (a = 0; a < model->slot0_target_head.rows * model->slot0_target_head.cols; ++a) model->slot0_target_head.data[a] -= learning_rate * grad_slot0_target_head[a];
        for (a = 0; a < FACTORIZED_TARGET_DIM; ++a) model->slot0_target_bias[a] -= learning_rate * grad_slot0_target_bias[a];
        for (a = 0; a < model->slot1_kind_head.rows * model->slot1_kind_head.cols; ++a) model->slot1_kind_head.data[a] -= learning_rate * grad_slot1_kind_head[a];
        for (a = 0; a < FACTORIZED_KIND_DIM; ++a) model->slot1_kind_bias[a] -= learning_rate * grad_slot1_kind_bias[a];
        for (a = 0; a < model->slot1_move_head.rows * model->slot1_move_head.cols; ++a) model->slot1_move_head.data[a] -= learning_rate * grad_slot1_move_head[a];
        for (a = 0; a < FACTORIZED_MOVE_DIM; ++a) model->slot1_move_bias[a] -= learning_rate * grad_slot1_move_bias[a];
        for (a = 0; a < model->slot1_switch_head.rows * model->slot1_switch_head.cols; ++a) model->slot1_switch_head.data[a] -= learning_rate * grad_slot1_switch_head[a];
        for (a = 0; a < FACTORIZED_SWITCH_DIM; ++a) model->slot1_switch_bias[a] -= learning_rate * grad_slot1_switch_bias[a];
        for (a = 0; a < model->slot1_tera_head.rows * model->slot1_tera_head.cols; ++a) model->slot1_tera_head.data[a] -= learning_rate * grad_slot1_tera_head[a];
        for (a = 0; a < FACTORIZED_TERA_DIM; ++a) model->slot1_tera_bias[a] -= learning_rate * grad_slot1_tera_bias[a];
        for (a = 0; a < model->slot1_target_head.rows * model->slot1_target_head.cols; ++a) model->slot1_target_head.data[a] -= learning_rate * grad_slot1_target_head[a];
        for (a = 0; a < FACTORIZED_TARGET_DIM; ++a) model->slot1_target_bias[a] -= learning_rate * grad_slot1_target_bias[a];
        for (a = 0; a < model->joint_pair_head.rows * model->joint_pair_head.cols; ++a) model->joint_pair_head.data[a] -= learning_rate * grad_joint_pair_head[a];
        for (a = 0; a < FACTORIZED_PAIR_DIM; ++a) model->joint_pair_bias[a] -= learning_rate * grad_joint_pair_bias[a];
        for (a = 0; a < model->entity_encoder.rows * model->entity_encoder.cols; ++a) model->entity_encoder.data[a] -= learning_rate * grad_entity_encoder[a];
        for (a = 0; a < (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u); ++a) model->entity_bias[a] -= learning_rate * grad_entity_bias[a];
        for (a = 0; a < model->entity_decoder.rows * model->entity_decoder.cols; ++a) model->entity_decoder.data[a] -= learning_rate * grad_entity_decoder[a];
#ifdef _OPENMP
        #pragma omp parallel for if(hdim >= 64)
#endif
        for (h = 0; h < hdim; ++h) model->value_head[h] -= learning_rate * grad_value_head[h];
        model->value_bias -= learning_rate * grad_value_bias;
        sync_flat_heads_from_factorized(model);
    }

    free(grad_slot0_kind_head); free(grad_slot0_kind_bias); free(grad_slot0_move_head); free(grad_slot0_move_bias);
    free(grad_slot0_switch_head); free(grad_slot0_switch_bias); free(grad_slot0_tera_head); free(grad_slot0_tera_bias); free(grad_slot0_target_head); free(grad_slot0_target_bias);
    free(grad_slot1_kind_head); free(grad_slot1_kind_bias); free(grad_slot1_move_head); free(grad_slot1_move_bias);
    free(grad_slot1_switch_head); free(grad_slot1_switch_bias); free(grad_slot1_tera_head); free(grad_slot1_tera_bias); free(grad_slot1_target_head); free(grad_slot1_target_bias);
    free(grad_joint_pair_head); free(grad_joint_pair_bias);

    return 1;
}

int gru_model_supervised_update_sequence(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const unsigned char* legal_mask,
    int target_action,
    float target_value,
    float learning_rate,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
) {
    return recurrent_update_sequence(model, sequence, steps, NULL, NULL, -1, legal_mask, target_action, 1.0f, target_value, 0.0f,
        NULL, 0.0f, learning_rate, 0, action_loss_out, value_loss_out, accuracy_out, NULL);
}

int gru_model_supervised_update_sequence_window(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask,
    int target_action,
    float target_value,
    float learning_rate,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
) {
    return recurrent_update_sequence(model, sequence, steps, initial_hidden_state, NULL, -1, legal_mask, target_action,
        1.0f, target_value, 0.0f, NULL, 0.0f, learning_rate, 0, action_loss_out, value_loss_out, accuracy_out, NULL);
}

int gru_model_supervised_update_sequence_window_dual(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    int target_action_a,
    const unsigned char* legal_mask_b,
    int target_action_b,
    float target_value,
    float learning_rate,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
) {
    return recurrent_update_sequence(model, sequence, steps, initial_hidden_state, legal_mask_b, target_action_b,
        legal_mask_a, target_action_a, 1.0f, target_value, 0.0f, NULL, 0.0f, learning_rate, 0,
        action_loss_out, value_loss_out, accuracy_out, NULL);
}

int gru_model_supervised_accumulate_sequence_window(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask,
    int target_action,
    float target_value,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
) {
    return recurrent_update_sequence(model, sequence, steps, initial_hidden_state, NULL, -1, legal_mask, target_action,
        1.0f, target_value, 0.0f, NULL, 0.0f, 0.0f, 1, action_loss_out, value_loss_out, accuracy_out, NULL);
}

int gru_model_supervised_accumulate_sequence_window_dual(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    int target_action_a,
    const unsigned char* legal_mask_b,
    int target_action_b,
    float target_value,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
) {
    return recurrent_update_sequence(model, sequence, steps, initial_hidden_state, legal_mask_b, target_action_b,
        legal_mask_a, target_action_a, 1.0f, target_value, 0.0f, NULL, 0.0f, 0.0f, 1,
        action_loss_out, value_loss_out, accuracy_out, NULL);
}

int gru_model_supervised_accumulate_sequence_window_factorized(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    const unsigned char* legal_mask_b,
    const FactorizedActionChoice* choice,
    float target_value,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
) {
    int action_a = -1;
    int action_b = -1;
    const unsigned char* primary_mask;
    const unsigned char* secondary_mask = NULL;
    int primary_action;
    int secondary_action = -1;
    if (!choice || !factorized_action_choice_to_flat_actions(choice, &action_a, &action_b)) {
        return 0;
    }
    if (action_a >= 0) {
        primary_mask = legal_mask_a;
        primary_action = action_a;
        if (action_b >= 0) {
            secondary_mask = legal_mask_b;
            secondary_action = action_b;
        }
    } else if (action_b >= 0) {
        primary_mask = legal_mask_b;
        primary_action = action_b;
    } else {
        return 0;
    }
    return recurrent_update_sequence(model, sequence, steps, initial_hidden_state,
        secondary_mask, secondary_action, primary_mask, primary_action,
        1.0f, target_value, 0.0f, NULL, 0.0f, 0.0f, 1,
        action_loss_out, value_loss_out, accuracy_out, choice);
}

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
) {
    float* logits;
    float* policy;
    float value;
    float dv;
    size_t a;
    size_t h;

    if (!model || !hidden_state || target_action < 0 || (size_t)target_action >= model->num_actions) {
        return 0;
    }
    logits = (float*)malloc(model->num_actions * sizeof(float));
    policy = (float*)malloc(model->num_actions * sizeof(float));
    if (!logits || !policy) {
        free(logits);
        free(policy);
        return 0;
    }

    evaluate_hidden_internal(model, hidden_state, legal_mask, logits, policy, &value);
    if (action_loss_out) {
        *action_loss_out = -logf(policy[target_action] > 1.0e-8f ? policy[target_action] : 1.0e-8f);
    }
    if (value_loss_out) {
        float err = value - target_value;
        *value_loss_out = 0.5f * err * err;
    }
    if (accuracy_out) {
        *accuracy_out = (gru_model_select_action(policy, legal_mask, model->num_actions) == target_action) ? 1.0f : 0.0f;
    }

    for (a = 0; a < model->num_actions; ++a) {
        float grad;
        if (legal_mask && !legal_mask[a]) {
            continue;
        }
        grad = policy[a] - ((int)a == target_action ? 1.0f : 0.0f);
        for (h = 0; h < model->hidden_dim; ++h) {
            model->policy_head.data[a * model->hidden_dim + h] -= learning_rate * grad * hidden_state[h];
        }
        model->policy_bias[a] -= learning_rate * grad;
    }

    dv = value - target_value;
    for (h = 0; h < model->hidden_dim; ++h) {
        model->value_head[h] -= learning_rate * dv * hidden_state[h];
    }
    model->value_bias -= learning_rate * dv;

    free(logits);
    free(policy);
    return 1;
}

int gru_model_policy_gradient_update_heads(
    GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate
) {
    float* logits;
    float* policy;
    float value;
    float dv;
    size_t a;
    size_t h;

    if (!model || !hidden_state || action < 0 || (size_t)action >= model->num_actions) {
        return 0;
    }
    logits = (float*)malloc(model->num_actions * sizeof(float));
    policy = (float*)malloc(model->num_actions * sizeof(float));
    if (!logits || !policy) {
        free(logits);
        free(policy);
        return 0;
    }

    evaluate_hidden_internal(model, hidden_state, legal_mask, logits, policy, &value);
    {
        float neg_entropy = 0.0f;
        for (a = 0; a < model->num_actions; ++a) {
            float p;
            if (legal_mask && !legal_mask[a]) {
                continue;
            }
            p = policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f;
            neg_entropy += p * logf(p);
        }
        for (a = 0; a < model->num_actions; ++a) {
            float grad;
            if (legal_mask && !legal_mask[a]) {
                continue;
            }
            grad = (policy[a] - ((int)a == action ? 1.0f : 0.0f)) * advantage;
            if (entropy_coef != 0.0f) {
                float p = policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f;
                grad += entropy_coef * p * (logf(p) - neg_entropy);
            }
            for (h = 0; h < model->hidden_dim; ++h) {
                model->policy_head.data[a * model->hidden_dim + h] -= learning_rate * grad * hidden_state[h];
            }
            model->policy_bias[a] -= learning_rate * grad;
        }
    }
    dv = value - target_value;
    for (h = 0; h < model->hidden_dim; ++h) {
        model->value_head[h] -= learning_rate * dv * hidden_state[h];
    }
    model->value_bias -= learning_rate * dv;
    bootstrap_factorized_heads_from_flat(model);
    free(logits);
    free(policy);
    return 1;
}

int gru_model_policy_gradient_update_sequence(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate
) {
    return recurrent_update_sequence(model, sequence, steps, NULL, NULL, -1, legal_mask, action, advantage, target_value, entropy_coef,
        NULL, 0.0f, learning_rate, 0, NULL, NULL, NULL, NULL);
}

int gru_model_policy_gradient_update_sequence_window(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate
) {
    return recurrent_update_sequence(
        model,
        sequence,
        steps,
        initial_hidden_state,
        NULL,
        -1,
        legal_mask,
        action,
        advantage,
        target_value,
        entropy_coef,
        NULL,
        0.0f,
        learning_rate,
        0,
        NULL,
        NULL,
        NULL,
        NULL);
}

int gru_model_policy_gradient_accumulate_sequence_window(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef
) {
    return recurrent_update_sequence(
        model,
        sequence,
        steps,
        initial_hidden_state,
        NULL,
        -1,
        legal_mask,
        action,
        advantage,
        target_value,
        entropy_coef,
        NULL,
        0.0f,
        0.0f,
        1,
        NULL,
        NULL,
        NULL,
        NULL);
}

int gru_model_policy_gradient_update_sequence_window_anchored(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate,
    const float* anchor_policy,
    float anchor_kl_coef
) {
    return recurrent_update_sequence(
        model,
        sequence,
        steps,
        initial_hidden_state,
        NULL,
        -1,
        legal_mask,
        action,
        advantage,
        target_value,
        entropy_coef,
        anchor_policy,
        anchor_kl_coef,
        learning_rate,
        0,
        NULL,
        NULL,
        NULL,
        NULL);
}

int gru_model_policy_gradient_accumulate_sequence_window_anchored(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask,
    int action,
    float advantage,
    float target_value,
    float entropy_coef,
    const float* anchor_policy,
    float anchor_kl_coef
) {
    return recurrent_update_sequence(
        model,
        sequence,
        steps,
        initial_hidden_state,
        NULL,
        -1,
        legal_mask,
        action,
        advantage,
        target_value,
        entropy_coef,
        anchor_policy,
        anchor_kl_coef,
        0.0f,
        1,
        NULL,
        NULL,
        NULL,
        NULL);
}

int gru_model_policy_gradient_update_sequence_window_dual(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    int action_a,
    const unsigned char* legal_mask_b,
    int action_b,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate
) {
    return recurrent_update_sequence(
        model,
        sequence,
        steps,
        initial_hidden_state,
        legal_mask_b,
        action_b,
        legal_mask_a,
        action_a,
        advantage,
        target_value,
        entropy_coef,
        NULL,
        0.0f,
        learning_rate,
        0,
        NULL,
        NULL,
        NULL,
        NULL);
}

int gru_model_policy_gradient_accumulate_sequence_window_dual(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    int action_a,
    const unsigned char* legal_mask_b,
    int action_b,
    float advantage,
    float target_value,
    float entropy_coef
) {
    return recurrent_update_sequence(
        model,
        sequence,
        steps,
        initial_hidden_state,
        legal_mask_b,
        action_b,
        legal_mask_a,
        action_a,
        advantage,
        target_value,
        entropy_coef,
        NULL,
        0.0f,
        0.0f,
        1,
        NULL,
        NULL,
        NULL,
        NULL);
}

int gru_model_policy_gradient_accumulate_sequence_window_factorized(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    const unsigned char* legal_mask_b,
    const FactorizedActionChoice* choice,
    float advantage,
    float target_value,
    float entropy_coef
) {
    int action_a = -1;
    int action_b = -1;
    const unsigned char* primary_mask;
    const unsigned char* secondary_mask = NULL;
    int primary_action;
    int secondary_action = -1;
    if (!choice || !factorized_action_choice_to_flat_actions(choice, &action_a, &action_b)) {
        return 0;
    }
    if (action_a >= 0) {
        primary_mask = legal_mask_a;
        primary_action = action_a;
        if (action_b >= 0) {
            secondary_mask = legal_mask_b;
            secondary_action = action_b;
        }
    } else if (action_b >= 0) {
        primary_mask = legal_mask_b;
        primary_action = action_b;
    } else {
        return 0;
    }
    return recurrent_update_sequence(model, sequence, steps, initial_hidden_state,
        secondary_mask, secondary_action, primary_mask, primary_action,
        advantage, target_value, entropy_coef, NULL, 0.0f, 0.0f, 1,
        NULL, NULL, NULL, choice);
}

int gru_model_policy_gradient_update_sequence_window_dual_anchored(
    GruModel* model,
    const float* sequence,
    size_t steps,
    const float* initial_hidden_state,
    const unsigned char* legal_mask_a,
    int action_a,
    const unsigned char* legal_mask_b,
    int action_b,
    float advantage,
    float target_value,
    float entropy_coef,
    float learning_rate,
    const float* anchor_policy_a,
    const float* anchor_policy_b,
    float anchor_kl_coef
) {
    if (!gru_model_policy_gradient_update_sequence_window_anchored(
            model,
            sequence,
            steps,
            initial_hidden_state,
            legal_mask_a,
            action_a,
            advantage,
            target_value,
            entropy_coef,
            learning_rate,
            anchor_policy_a,
            anchor_kl_coef)) {
        return 0;
    }
    return gru_model_policy_gradient_update_sequence_window_anchored(
        model,
        sequence,
        steps,
        initial_hidden_state,
        legal_mask_b,
        action_b,
        advantage,
        target_value,
        entropy_coef,
        learning_rate,
        anchor_policy_b,
        anchor_kl_coef);
}

size_t gru_model_parameter_count(const GruModel* model) {
    if (!model) {
        return 0;
    }
    return
        model->wzx.rows * model->wzx.cols +
        model->wzh.rows * model->wzh.cols +
        model->hidden_dim +
        model->wrx.rows * model->wrx.cols +
        model->wrh.rows * model->wrh.cols +
        model->hidden_dim +
        model->wnx.rows * model->wnx.cols +
        model->wnh.rows * model->wnh.cols +
        model->hidden_dim +
        model->policy_head.rows * model->policy_head.cols +
        model->num_actions +
        model->slot0_kind_head.rows * model->slot0_kind_head.cols + FACTORIZED_KIND_DIM +
        model->slot0_move_head.rows * model->slot0_move_head.cols + FACTORIZED_MOVE_DIM +
        model->slot0_switch_head.rows * model->slot0_switch_head.cols + FACTORIZED_SWITCH_DIM +
        model->slot0_tera_head.rows * model->slot0_tera_head.cols + FACTORIZED_TERA_DIM +
        model->slot0_target_head.rows * model->slot0_target_head.cols + FACTORIZED_TARGET_DIM +
        model->slot1_kind_head.rows * model->slot1_kind_head.cols + FACTORIZED_KIND_DIM +
        model->slot1_move_head.rows * model->slot1_move_head.cols + FACTORIZED_MOVE_DIM +
        model->slot1_switch_head.rows * model->slot1_switch_head.cols + FACTORIZED_SWITCH_DIM +
        model->slot1_tera_head.rows * model->slot1_tera_head.cols + FACTORIZED_TERA_DIM +
        model->slot1_target_head.rows * model->slot1_target_head.cols + FACTORIZED_TARGET_DIM +
        model->joint_pair_head.rows * model->joint_pair_head.cols + FACTORIZED_PAIR_DIM +
        model->entity_encoder.rows * model->entity_encoder.cols +
        (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u) +
        model->entity_decoder.rows * model->entity_decoder.cols +
        model->hidden_dim +
        1;
}

size_t gru_model_pre_entity_parameter_count(const GruModel* model) {
    if (!model) {
        return 0;
    }
    return gru_model_parameter_count(model) -
        model->entity_encoder.rows * model->entity_encoder.cols -
        (model->entity_encoder_enabled ? GRU_ENTITY_EMBED_DIM : 0u) -
        model->entity_decoder.rows * model->entity_decoder.cols;
}

size_t gru_model_pre_joint_parameter_count(const GruModel* model) {
    if (!model) {
        return 0;
    }
    return gru_model_pre_entity_parameter_count(model) -
        (model->joint_pair_head.rows * model->joint_pair_head.cols + FACTORIZED_PAIR_DIM);
}

size_t gru_model_pre_target_parameter_count(const GruModel* model) {
    if (!model) {
        return 0;
    }
    return gru_model_pre_joint_parameter_count(model) -
        (model->slot0_target_head.rows * model->slot0_target_head.cols + FACTORIZED_TARGET_DIM) -
        (model->slot1_target_head.rows * model->slot1_target_head.cols + FACTORIZED_TARGET_DIM);
}

size_t gru_model_legacy_parameter_count(const GruModel* model) {
    if (!model) {
        return 0;
    }
    return
        model->wzx.rows * model->wzx.cols +
        model->wzh.rows * model->wzh.cols + model->hidden_dim +
        model->wrx.rows * model->wrx.cols +
        model->wrh.rows * model->wrh.cols + model->hidden_dim +
        model->wnx.rows * model->wnx.cols +
        model->wnh.rows * model->wnh.cols + model->hidden_dim +
        model->policy_head.rows * model->policy_head.cols + model->num_actions +
        model->hidden_dim + 1;
}

static void copy_out(float* out, size_t* idx, const float* src, size_t count) {
    memcpy(out + *idx, src, count * sizeof(float));
    *idx += count;
}

static void copy_in(float* dst, const float* in, size_t* idx, size_t count) {
    memcpy(dst, in + *idx, count * sizeof(float));
    *idx += count;
}

int gru_model_export_parameters(const GruModel* model, float* out, size_t count) {
    size_t idx = 0;
    if (!model || !out || count < gru_model_parameter_count(model)) {
        return 0;
    }
    copy_out(out, &idx, model->wzx.data, model->wzx.rows * model->wzx.cols);
    copy_out(out, &idx, model->wzh.data, model->wzh.rows * model->wzh.cols);
    copy_out(out, &idx, model->bz, model->hidden_dim);
    copy_out(out, &idx, model->wrx.data, model->wrx.rows * model->wrx.cols);
    copy_out(out, &idx, model->wrh.data, model->wrh.rows * model->wrh.cols);
    copy_out(out, &idx, model->br, model->hidden_dim);
    copy_out(out, &idx, model->wnx.data, model->wnx.rows * model->wnx.cols);
    copy_out(out, &idx, model->wnh.data, model->wnh.rows * model->wnh.cols);
    copy_out(out, &idx, model->bn, model->hidden_dim);
    copy_out(out, &idx, model->policy_head.data, model->policy_head.rows * model->policy_head.cols);
    copy_out(out, &idx, model->policy_bias, model->num_actions);
    copy_out(out, &idx, model->slot0_kind_head.data, model->slot0_kind_head.rows * model->slot0_kind_head.cols);
    copy_out(out, &idx, model->slot0_kind_bias, FACTORIZED_KIND_DIM);
    copy_out(out, &idx, model->slot0_move_head.data, model->slot0_move_head.rows * model->slot0_move_head.cols);
    copy_out(out, &idx, model->slot0_move_bias, FACTORIZED_MOVE_DIM);
    copy_out(out, &idx, model->slot0_switch_head.data, model->slot0_switch_head.rows * model->slot0_switch_head.cols);
    copy_out(out, &idx, model->slot0_switch_bias, FACTORIZED_SWITCH_DIM);
    copy_out(out, &idx, model->slot0_tera_head.data, model->slot0_tera_head.rows * model->slot0_tera_head.cols);
    copy_out(out, &idx, model->slot0_tera_bias, FACTORIZED_TERA_DIM);
    copy_out(out, &idx, model->slot0_target_head.data, model->slot0_target_head.rows * model->slot0_target_head.cols);
    copy_out(out, &idx, model->slot0_target_bias, FACTORIZED_TARGET_DIM);
    copy_out(out, &idx, model->slot1_kind_head.data, model->slot1_kind_head.rows * model->slot1_kind_head.cols);
    copy_out(out, &idx, model->slot1_kind_bias, FACTORIZED_KIND_DIM);
    copy_out(out, &idx, model->slot1_move_head.data, model->slot1_move_head.rows * model->slot1_move_head.cols);
    copy_out(out, &idx, model->slot1_move_bias, FACTORIZED_MOVE_DIM);
    copy_out(out, &idx, model->slot1_switch_head.data, model->slot1_switch_head.rows * model->slot1_switch_head.cols);
    copy_out(out, &idx, model->slot1_switch_bias, FACTORIZED_SWITCH_DIM);
    copy_out(out, &idx, model->slot1_tera_head.data, model->slot1_tera_head.rows * model->slot1_tera_head.cols);
    copy_out(out, &idx, model->slot1_tera_bias, FACTORIZED_TERA_DIM);
    copy_out(out, &idx, model->slot1_target_head.data, model->slot1_target_head.rows * model->slot1_target_head.cols);
    copy_out(out, &idx, model->slot1_target_bias, FACTORIZED_TARGET_DIM);
    copy_out(out, &idx, model->joint_pair_head.data, model->joint_pair_head.rows * model->joint_pair_head.cols);
    copy_out(out, &idx, model->joint_pair_bias, FACTORIZED_PAIR_DIM);
    if (model->entity_encoder_enabled) {
        copy_out(out, &idx, model->entity_encoder.data, model->entity_encoder.rows * model->entity_encoder.cols);
        copy_out(out, &idx, model->entity_bias, GRU_ENTITY_EMBED_DIM);
        copy_out(out, &idx, model->entity_decoder.data, model->entity_decoder.rows * model->entity_decoder.cols);
    }
    copy_out(out, &idx, model->value_head, model->hidden_dim);
    out[idx++] = model->value_bias;
    return idx == gru_model_parameter_count(model);
}

int gru_model_import_parameters(GruModel* model, const float* in, size_t count) {
    size_t idx = 0;
    size_t legacy_count;
    size_t pre_target_count;
    size_t pre_joint_count;
    size_t pre_entity_count;
    int has_entity_encoder;
    int has_joint_head;
    int has_target_heads;
    int has_factorized_heads;
    if (!model || !in) {
        return 0;
    }
    legacy_count = gru_model_legacy_parameter_count(model);
    pre_entity_count = gru_model_pre_entity_parameter_count(model);
    pre_joint_count = gru_model_pre_joint_parameter_count(model);
    pre_target_count = gru_model_pre_target_parameter_count(model);
    if (count < legacy_count) {
        return 0;
    }
    copy_in(model->wzx.data, in, &idx, model->wzx.rows * model->wzx.cols);
    copy_in(model->wzh.data, in, &idx, model->wzh.rows * model->wzh.cols);
    copy_in(model->bz, in, &idx, model->hidden_dim);
    copy_in(model->wrx.data, in, &idx, model->wrx.rows * model->wrx.cols);
    copy_in(model->wrh.data, in, &idx, model->wrh.rows * model->wrh.cols);
    copy_in(model->br, in, &idx, model->hidden_dim);
    copy_in(model->wnx.data, in, &idx, model->wnx.rows * model->wnx.cols);
    copy_in(model->wnh.data, in, &idx, model->wnh.rows * model->wnh.cols);
    copy_in(model->bn, in, &idx, model->hidden_dim);
    copy_in(model->policy_head.data, in, &idx, model->policy_head.rows * model->policy_head.cols);
    copy_in(model->policy_bias, in, &idx, model->num_actions);
    has_entity_encoder = model->entity_encoder_enabled && count == gru_model_parameter_count(model);
    has_joint_head = has_entity_encoder || count == pre_entity_count;
    has_target_heads = has_joint_head || count == pre_joint_count;
    has_factorized_heads = has_target_heads || count == pre_target_count;
    if (has_factorized_heads) {
        copy_in(model->slot0_kind_head.data, in, &idx, model->slot0_kind_head.rows * model->slot0_kind_head.cols);
        copy_in(model->slot0_kind_bias, in, &idx, FACTORIZED_KIND_DIM);
        copy_in(model->slot0_move_head.data, in, &idx, model->slot0_move_head.rows * model->slot0_move_head.cols);
        copy_in(model->slot0_move_bias, in, &idx, FACTORIZED_MOVE_DIM);
        copy_in(model->slot0_switch_head.data, in, &idx, model->slot0_switch_head.rows * model->slot0_switch_head.cols);
        copy_in(model->slot0_switch_bias, in, &idx, FACTORIZED_SWITCH_DIM);
        copy_in(model->slot0_tera_head.data, in, &idx, model->slot0_tera_head.rows * model->slot0_tera_head.cols);
        copy_in(model->slot0_tera_bias, in, &idx, FACTORIZED_TERA_DIM);
        if (has_target_heads) {
            copy_in(model->slot0_target_head.data, in, &idx, model->slot0_target_head.rows * model->slot0_target_head.cols);
            copy_in(model->slot0_target_bias, in, &idx, FACTORIZED_TARGET_DIM);
        }
        copy_in(model->slot1_kind_head.data, in, &idx, model->slot1_kind_head.rows * model->slot1_kind_head.cols);
        copy_in(model->slot1_kind_bias, in, &idx, FACTORIZED_KIND_DIM);
        copy_in(model->slot1_move_head.data, in, &idx, model->slot1_move_head.rows * model->slot1_move_head.cols);
        copy_in(model->slot1_move_bias, in, &idx, FACTORIZED_MOVE_DIM);
        copy_in(model->slot1_switch_head.data, in, &idx, model->slot1_switch_head.rows * model->slot1_switch_head.cols);
        copy_in(model->slot1_switch_bias, in, &idx, FACTORIZED_SWITCH_DIM);
        copy_in(model->slot1_tera_head.data, in, &idx, model->slot1_tera_head.rows * model->slot1_tera_head.cols);
        copy_in(model->slot1_tera_bias, in, &idx, FACTORIZED_TERA_DIM);
        if (has_target_heads) {
            copy_in(model->slot1_target_head.data, in, &idx, model->slot1_target_head.rows * model->slot1_target_head.cols);
            copy_in(model->slot1_target_bias, in, &idx, FACTORIZED_TARGET_DIM);
        }
    } else {
        bootstrap_factorized_heads_from_flat(model);
    }
    if (!has_target_heads) {
        memset(model->slot0_target_head.data, 0, model->slot0_target_head.rows * model->slot0_target_head.cols * sizeof(float));
        memset(model->slot0_target_bias, 0, FACTORIZED_TARGET_DIM * sizeof(float));
        memset(model->slot1_target_head.data, 0, model->slot1_target_head.rows * model->slot1_target_head.cols * sizeof(float));
        memset(model->slot1_target_bias, 0, FACTORIZED_TARGET_DIM * sizeof(float));
    }
    if (has_joint_head) {
        copy_in(model->joint_pair_head.data, in, &idx, model->joint_pair_head.rows * model->joint_pair_head.cols);
        copy_in(model->joint_pair_bias, in, &idx, FACTORIZED_PAIR_DIM);
    } else {
        memset(model->joint_pair_head.data, 0, model->joint_pair_head.rows * model->joint_pair_head.cols * sizeof(float));
        memset(model->joint_pair_bias, 0, FACTORIZED_PAIR_DIM * sizeof(float));
    }
    if (has_entity_encoder) {
        copy_in(model->entity_encoder.data, in, &idx, model->entity_encoder.rows * model->entity_encoder.cols);
        copy_in(model->entity_bias, in, &idx, GRU_ENTITY_EMBED_DIM);
        copy_in(model->entity_decoder.data, in, &idx, model->entity_decoder.rows * model->entity_decoder.cols);
    } else if (model->entity_encoder_enabled) {
        memset(model->entity_encoder.data, 0, model->entity_encoder.rows * model->entity_encoder.cols * sizeof(float));
        memset(model->entity_bias, 0, GRU_ENTITY_EMBED_DIM * sizeof(float));
        memset(model->entity_decoder.data, 0, model->entity_decoder.rows * model->entity_decoder.cols * sizeof(float));
    }
    copy_in(model->value_head, in, &idx, model->hidden_dim);
    model->value_bias = in[idx++];
    return idx == count;
}
