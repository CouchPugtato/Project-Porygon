#include "gru_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
} GruRecurrentScratch;

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

    float* value_head;
    float value_bias;
    GruForwardScratch forward_scratch;
    GruRecurrentScratch recurrent_scratch;
};

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
    memset(scratch, 0, sizeof(*scratch));
}

static int gru_forward_scratch_ensure(GruModel* model) {
    GruForwardScratch* scratch;
    if (!model) {
        return 0;
    }
    scratch = &model->forward_scratch;
    if (scratch->hidden_capacity == model->hidden_dim && scratch->action_capacity == model->num_actions &&
            scratch->z && scratch->r && scratch->n && scratch->gated_hidden && scratch->logits && scratch->next_hidden) {
        return 1;
    }
    gru_forward_scratch_free(scratch);
    scratch->z = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->r = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->n = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->gated_hidden = (float*)calloc(model->hidden_dim, sizeof(float));
    scratch->logits = (float*)calloc(model->num_actions, sizeof(float));
    scratch->next_hidden = (float*)calloc(model->hidden_dim, sizeof(float));
    if (!scratch->z || !scratch->r || !scratch->n || !scratch->gated_hidden || !scratch->logits || !scratch->next_hidden) {
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
            scratch->grad_policy_head && scratch->grad_policy_bias && scratch->grad_value_head) {
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
    if (!scratch->h_states || !scratch->z_cache || !scratch->r_cache || !scratch->n_cache || !scratch->gated_cache ||
            !scratch->zero_hidden || !scratch->logits || !scratch->policy || !scratch->grad_h ||
            !scratch->next_grad_h || !scratch->grad_logits || !scratch->d_gated || !scratch->d_pre_z ||
            !scratch->d_pre_r || !scratch->d_pre_n || !scratch->grad_wzx || !scratch->grad_wzh ||
            !scratch->grad_bz || !scratch->grad_wrx || !scratch->grad_wrh || !scratch->grad_br ||
            !scratch->grad_wnx || !scratch->grad_wnh || !scratch->grad_bn || !scratch->grad_policy_head ||
            !scratch->grad_policy_bias || !scratch->grad_value_head) {
        gru_recurrent_scratch_free(scratch);
        return 0;
    }
    scratch->step_capacity = steps;
    scratch->hidden_capacity = model->hidden_dim;
    scratch->input_capacity = model->input_dim;
    scratch->action_capacity = model->num_actions;
    return 1;
}

static void matrix_vec_mul_accum(const Matrix* matrix, const float* vec, float* out) {
    size_t r;
    size_t c;
    for (r = 0; r < matrix->rows; ++r) {
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
    size_t c;
    for (r = 0; r < matrix->rows; ++r) {
        const float* row = matrix->data + (r * matrix->cols);
        for (c = 0; c < matrix->cols; ++c) {
            out[c] += row[c] * vec[r];
        }
    }
}

static void outer_product_accum(float* dst, size_t rows, size_t cols, const float* lhs, const float* rhs) {
    size_t r;
    size_t c;
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            dst[r * cols + c] += lhs[r] * rhs[c];
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

static void evaluate_hidden_internal(
    const GruModel* model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    float* logits,
    float* policy_out,
    float* value_out
) {
    size_t i;
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
    *value_out = model->value_bias;
    for (i = 0; i < model->hidden_dim; ++i) {
        *value_out += model->value_head[i] * hidden_state[i];
    }
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
    model->value_head = (float*)malloc(hidden_dim * sizeof(float));

    if (!model->wzx.data || !model->wzh.data || !model->bz ||
        !model->wrx.data || !model->wrh.data || !model->br ||
        !model->wnx.data || !model->wnh.data || !model->bn ||
        !model->policy_head.data || !model->policy_bias || !model->value_head) {
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
    free(model->value_head);
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

    if (!model || !input || !hidden_state_in || !hidden_state_out || !policy_out || !value_out ||
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

    if (!model || !input || !hidden_state_in || !hidden_state_out || !policy_out || !value_out) {
        return;
    }
    if (!gru_forward_scratch_ensure((GruModel*)model)) {
        return;
    }
    scratch = &((GruModel*)model)->forward_scratch;
    gru_model_forward_step_with_buffers(model, input, hidden_state_in, hidden_state_out, policy_out, value_out,
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

    if (!model || !sequence || !hidden_state_io || !policy_out || !value_out) {
        return;
    }
    if (!gru_forward_scratch_ensure((GruModel*)model)) {
        return;
    }
    scratch = &((GruModel*)model)->forward_scratch;

    for (t = 0; t < steps; ++t) {
        const float* input = sequence + (t * model->input_dim);
        if (!gru_model_forward_step_with_buffers(model, input, hidden_state_io, scratch->next_hidden, policy_out, value_out,
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
    float learning_rate,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out
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
    float grad_value_bias = 0.0f;
    float value = 0.0f;
    float dv;
    size_t target_count = 1;
    float accuracy_sum = 0.0f;
    float action_loss_sum = 0.0f;
    const float* start_hidden;

    if (!model || !sequence || steps == 0 || target_action < 0 || (size_t)target_action >= model->num_actions) {
        return 0;
    }
    hdim = model->hidden_dim;
    xdim = model->input_dim;
    adim = model->num_actions;

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

    start_hidden = initial_hidden_state ? initial_hidden_state : zero_hidden;
    if (target_action_secondary >= 0) {
        target_count = 2;
    }

    for (t = 0; t < steps; ++t) {
        const float* input = sequence + (t * xdim);
        const float* prev_h = (t > 0) ? (h_states + ((t - 1) * hdim)) : start_hidden;
        float* z_t = z_cache + (t * hdim);
        float* r_t = r_cache + (t * hdim);
        float* n_t = n_cache + (t * hdim);
        float* g_t = gated_cache + (t * hdim);
        float* h_t = h_states + (t * hdim);
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

    evaluate_hidden_internal(model, h_states + ((steps - 1) * hdim), legal_mask, logits, policy, &value);
    action_loss_sum += -logf(policy[target_action] > 1.0e-8f ? policy[target_action] : 1.0e-8f);
    accuracy_sum += (gru_model_select_action(policy, legal_mask, adim) == target_action) ? 1.0f : 0.0f;
    for (a = 0; a < adim; ++a) {
        grad_logits[a] = (policy[a] - ((int)a == target_action ? 1.0f : 0.0f)) * policy_scale;
        if (entropy_coef != 0.0f) {
            grad_logits[a] += entropy_coef * policy[a] * logf(policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f);
        }
        grad_policy_bias[a] += grad_logits[a];
        for (h = 0; h < hdim; ++h) {
            grad_policy_head[a * hdim + h] += grad_logits[a] * h_states[(steps - 1) * hdim + h];
        }
    }
    matrix_transpose_vec_mul_accum(&model->policy_head, grad_logits, grad_h);

    if (target_action_secondary >= 0) {
        evaluate_hidden_internal(model, h_states + ((steps - 1) * hdim), legal_mask_secondary, logits, policy, &value);
        action_loss_sum += -logf(policy[target_action_secondary] > 1.0e-8f ? policy[target_action_secondary] : 1.0e-8f);
        accuracy_sum += (gru_model_select_action(policy, legal_mask_secondary, adim) == target_action_secondary) ? 1.0f : 0.0f;
        for (a = 0; a < adim; ++a) {
            grad_logits[a] = (policy[a] - ((int)a == target_action_secondary ? 1.0f : 0.0f)) * policy_scale;
            if (entropy_coef != 0.0f) {
                grad_logits[a] += entropy_coef * policy[a] * logf(policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f);
            }
            grad_policy_bias[a] += grad_logits[a];
            for (h = 0; h < hdim; ++h) {
                grad_policy_head[a * hdim + h] += grad_logits[a] * h_states[(steps - 1) * hdim + h];
            }
        }
        matrix_transpose_vec_mul_accum(&model->policy_head, grad_logits, grad_h);
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
        const float* input = sequence + (t * xdim);
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
        memcpy(grad_h, next_grad_h, hdim * sizeof(float));
    }

    for (h = 0; h < model->wzx.rows * model->wzx.cols; ++h) model->wzx.data[h] -= learning_rate * grad_wzx[h];
    for (h = 0; h < model->wzh.rows * model->wzh.cols; ++h) model->wzh.data[h] -= learning_rate * grad_wzh[h];
    for (h = 0; h < hdim; ++h) model->bz[h] -= learning_rate * grad_bz[h];
    for (h = 0; h < model->wrx.rows * model->wrx.cols; ++h) model->wrx.data[h] -= learning_rate * grad_wrx[h];
    for (h = 0; h < model->wrh.rows * model->wrh.cols; ++h) model->wrh.data[h] -= learning_rate * grad_wrh[h];
    for (h = 0; h < hdim; ++h) model->br[h] -= learning_rate * grad_br[h];
    for (h = 0; h < model->wnx.rows * model->wnx.cols; ++h) model->wnx.data[h] -= learning_rate * grad_wnx[h];
    for (h = 0; h < model->wnh.rows * model->wnh.cols; ++h) model->wnh.data[h] -= learning_rate * grad_wnh[h];
    for (h = 0; h < hdim; ++h) model->bn[h] -= learning_rate * grad_bn[h];
    for (a = 0; a < model->policy_head.rows * model->policy_head.cols; ++a) model->policy_head.data[a] -= learning_rate * grad_policy_head[a];
    for (a = 0; a < adim; ++a) model->policy_bias[a] -= learning_rate * grad_policy_bias[a];
    for (h = 0; h < hdim; ++h) model->value_head[h] -= learning_rate * grad_value_head[h];
    model->value_bias -= learning_rate * grad_value_bias;

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
        learning_rate, action_loss_out, value_loss_out, accuracy_out);
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
        1.0f, target_value, 0.0f, learning_rate, action_loss_out, value_loss_out, accuracy_out);
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
        legal_mask_a, target_action_a, 1.0f, target_value, 0.0f, learning_rate,
        action_loss_out, value_loss_out, accuracy_out);
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
    for (a = 0; a < model->num_actions; ++a) {
        float grad;
        if (legal_mask && !legal_mask[a]) {
            continue;
        }
        grad = (policy[a] - ((int)a == action ? 1.0f : 0.0f)) * advantage;
        if (entropy_coef != 0.0f) {
            grad += entropy_coef * policy[a] * logf(policy[a] > 1.0e-8f ? policy[a] : 1.0e-8f);
        }
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
        learning_rate, NULL, NULL, NULL);
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
        model->hidden_dim +
        1;
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
    copy_out(out, &idx, model->value_head, model->hidden_dim);
    out[idx++] = model->value_bias;
    return idx == gru_model_parameter_count(model);
}

int gru_model_import_parameters(GruModel* model, const float* in, size_t count) {
    size_t idx = 0;
    if (!model || !in || count < gru_model_parameter_count(model)) {
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
    copy_in(model->value_head, in, &idx, model->hidden_dim);
    model->value_bias = in[idx++];
    return idx == gru_model_parameter_count(model);
}
