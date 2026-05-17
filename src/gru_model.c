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

    memcpy(logits, model->policy_bias, model->num_actions * sizeof(float));
    matrix_vec_mul_accum(&model->policy_head, hidden_state_out, logits);
    softmax(logits, model->num_actions, policy_out);

    *value_out = model->value_bias;
    for (h = 0; h < model->hidden_dim; ++h) {
        *value_out += model->value_head[h] * hidden_state_out[h];
    }

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
