#include "gru_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

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

void gru_model_forward_step(
    const GruModel* model,
    const float* input,
    const float* hidden_state_in,
    float* hidden_state_out,
    float* policy_out,
    float* value_out
) {
    size_t h;
    float* z;
    float* r;
    float* n;
    float* gated_hidden;
    float* logits;

    if (!model || !input || !hidden_state_in || !hidden_state_out || !policy_out || !value_out) {
        return;
    }

    z = (float*)calloc(model->hidden_dim, sizeof(float));
    r = (float*)calloc(model->hidden_dim, sizeof(float));
    n = (float*)calloc(model->hidden_dim, sizeof(float));
    gated_hidden = (float*)calloc(model->hidden_dim, sizeof(float));
    logits = (float*)calloc(model->num_actions, sizeof(float));
    if (!z || !r || !n || !gated_hidden || !logits) {
        free(z);
        free(r);
        free(n);
        free(gated_hidden);
        free(logits);
        return;
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

    free(z);
    free(r);
    free(n);
    free(gated_hidden);
    free(logits);
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
    float* next_hidden;

    if (!model || !sequence || !hidden_state_io || !policy_out || !value_out) {
        return;
    }

    next_hidden = (float*)malloc(model->hidden_dim * sizeof(float));
    if (!next_hidden) {
        return;
    }

    for (t = 0; t < steps; ++t) {
        const float* input = sequence + (t * model->input_dim);
        gru_model_forward_step(model, input, hidden_state_io, next_hidden, policy_out, value_out);
        memcpy(hidden_state_io, next_hidden, model->hidden_dim * sizeof(float));
    }
    free(next_hidden);
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
