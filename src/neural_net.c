#include "neural_net.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int in_dim;
    int out_dim;
    double* W;      // [out_dim * in_dim]
    double* b;      // [out_dim]
    double* dW;     // gradient accumulator
    double* db;     // gradient accumulator
} DenseLayer;

struct NeuralNetwork {
    int input_dim;
    int num_actions;
    int hidden_count;
    DenseLayer* hidden;       // array of hidden layers
    DenseLayer policy_head;   // maps last hidden -> logits
    DenseLayer value_head;    // maps last hidden -> scalar value
};

static double randn() {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void he_init(double* W, int out_dim, int in_dim) {
    double std = sqrt(2.0 / (double)in_dim);
    for (int o = 0; o < out_dim; ++o) {
        for (int i = 0; i < in_dim; ++i) {
            W[o * in_dim + i] = randn() * std;
        }
    }
}

static void zeros(double* arr, size_t n) {
    for (size_t i = 0; i < n; ++i) arr[i] = 0.0;
}

static DenseLayer make_layer(int in_dim, int out_dim) {
    DenseLayer L;
    L.in_dim = in_dim;
    L.out_dim = out_dim;
    size_t Wn = (size_t)out_dim * (size_t)in_dim;
    L.W = (double*)malloc(sizeof(double) * Wn);
    L.b = (double*)malloc(sizeof(double) * out_dim);
    L.dW = (double*)malloc(sizeof(double) * Wn);
    L.db = (double*)malloc(sizeof(double) * out_dim);
    if (!L.W || !L.b || !L.dW || !L.db) return (DenseLayer){0};
    he_init(L.W, out_dim, in_dim);
    zeros(L.b, out_dim);
    zeros(L.dW, Wn);
    zeros(L.db, out_dim);
    return L;
}

static void free_layer(DenseLayer* L) {
    if (!L) return;
    free(L->W); free(L->b); free(L->dW); free(L->db);
    L->W = L->b = L->dW = L->db = NULL;
    L->in_dim = L->out_dim = 0;
}

NeuralNetwork* nn_create(int input_dim, const int* hidden_dims, int hidden_count, int num_actions) {
    if (input_dim <= 0 || hidden_count < 0 || num_actions <= 0) return NULL;
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) return NULL;
    nn->input_dim = input_dim;
    nn->num_actions = num_actions;
    nn->hidden_count = hidden_count;
    nn->hidden = NULL;

    int prev = input_dim;
    if (hidden_count > 0) {
        nn->hidden = (DenseLayer*)malloc(sizeof(DenseLayer) * hidden_count);
        if (!nn->hidden) { free(nn); return NULL; }
        for (int h = 0; h < hidden_count; ++h) {
            nn->hidden[h] = make_layer(prev, hidden_dims[h]);
            if (nn->hidden[h].W == NULL) { nn_destroy(nn); return NULL; }
            prev = hidden_dims[h];
        }
    }

    nn->policy_head = make_layer(prev, num_actions);
    nn->value_head  = make_layer(prev, 1);
    if (nn->policy_head.W == NULL || nn->value_head.W == NULL) { nn_destroy(nn); return NULL; }
    return nn;
}

void nn_destroy(NeuralNetwork* nn) {
    if (!nn) return;
    if (nn->hidden) {
        for (int h = 0; h < nn->hidden_count; ++h) free_layer(&nn->hidden[h]);
        free(nn->hidden);
    }
    free_layer(&nn->policy_head);
    free_layer(&nn->value_head);
    free(nn);
}

static inline double relu(double x) { return x > 0.0 ? x : 0.0; }
static inline double relu_grad(double x) { return x > 0.0 ? 1.0 : 0.0; }

static void softmax(const double* logits, int n, double* out) {
    double maxlog = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > maxlog) maxlog = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        out[i] = exp(logits[i] - maxlog);
        sum += out[i];
    }
    double inv = 1.0 / sum;
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

void nn_forward(NeuralNetwork* nn, const double* input, double* policy_out, double* value_out) {
    if (!nn || !input || !policy_out || !value_out) return;

    // forward through hidden layers
    const double* prev_act = input;
    double* acts_buf = NULL; // reused buffer per layer
    double* last_acts = NULL;

    for (int h = 0; h < nn->hidden_count; ++h) {
        DenseLayer* L = &nn->hidden[h];
        if (!acts_buf) acts_buf = (double*)malloc(sizeof(double) * L->out_dim);
        else acts_buf = (double*)realloc(acts_buf, sizeof(double) * L->out_dim);
        if (!acts_buf) return;
        for (int o = 0; o < L->out_dim; ++o) {
            double z = L->b[o];
            const double* Wrow = &L->W[o * L->in_dim];
            for (int i = 0; i < L->in_dim; ++i) z += Wrow[i] * prev_act[i];
            acts_buf[o] = relu(z);
        }
        prev_act = acts_buf;
        last_acts = acts_buf;
    }

    const double* features = (nn->hidden_count > 0) ? last_acts : input;

    // policy logits
    double* logits = (double*)malloc(sizeof(double) * nn->num_actions);
    for (int a = 0; a < nn->num_actions; ++a) {
        double z = nn->policy_head.b[a];
        const double* Wrow = &nn->policy_head.W[a * nn->policy_head.in_dim];
        for (int i = 0; i < nn->policy_head.in_dim; ++i) z += Wrow[i] * features[i];
        logits[a] = z;
    }
    softmax(logits, nn->num_actions, policy_out);

    // value
    double zv = nn->value_head.b[0];
    for (int i = 0; i < nn->value_head.in_dim; ++i) zv += nn->value_head.W[i] * features[i];
    *value_out = zv;

    free(logits);
    free(acts_buf);
}

// Backprop for one step using (policy gradient with advantage) + (value MSE)
void nn_backward(NeuralNetwork* nn, const double* input, int action_index, double advantage, double target_value) {
    if (!nn || !input) return;

    // forward caches
    const double* prev_act = input;
    double** acts = NULL; // store activations per hidden layer
    if (nn->hidden_count > 0) {
        acts = (double**)malloc(sizeof(double*) * nn->hidden_count);
    }
    for (int h = 0; h < nn->hidden_count; ++h) {
        DenseLayer* L = &nn->hidden[h];
        acts[h] = (double*)malloc(sizeof(double) * L->out_dim);
        for (int o = 0; o < L->out_dim; ++o) {
            double z = L->b[o];
            const double* Wrow = &L->W[o * L->in_dim];
            for (int i = 0; i < L->in_dim; ++i) z += Wrow[i] * prev_act[i];
            acts[h][o] = relu(z);
        }
        prev_act = acts[h];
    }
    const double* features = (nn->hidden_count > 0) ? acts[nn->hidden_count - 1] : input;

    // policy forward
    double* logits = (double*)malloc(sizeof(double) * nn->num_actions);
    for (int a = 0; a < nn->num_actions; ++a) {
        double z = nn->policy_head.b[a];
        const double* Wrow = &nn->policy_head.W[a * nn->policy_head.in_dim];
        for (int i = 0; i < nn->policy_head.in_dim; ++i) z += Wrow[i] * features[i];
        logits[a] = z;
    }
    double* policy = (double*)malloc(sizeof(double) * nn->num_actions);
    softmax(logits, nn->num_actions, policy);

    // value forward
    double value = nn->value_head.b[0];
    for (int i = 0; i < nn->value_head.in_dim; ++i) value += nn->value_head.W[i] * features[i];

    // gradients: policy head (logits gradient = policy - one_hot(action)) * advantage
    double* dlogits = (double*)malloc(sizeof(double) * nn->num_actions);
    for (int a = 0; a < nn->num_actions; ++a) {
        double y = (a == action_index) ? 1.0 : 0.0;
        dlogits[a] = (policy[a] - y) * advantage;
    }

    // accumulate gradients for policy head
    for (int a = 0; a < nn->num_actions; ++a) {
        double* Wrow = &nn->policy_head.W[a * nn->policy_head.in_dim];
        double* dWrow = &nn->policy_head.dW[a * nn->policy_head.in_dim];
        for (int i = 0; i < nn->policy_head.in_dim; ++i) dWrow[i] += dlogits[a] * features[i];
        nn->policy_head.db[a] += dlogits[a];
    }

    // value head gradient: dL/dz = (value - target)
    double dv = (value - target_value);
    for (int i = 0; i < nn->value_head.in_dim; ++i) nn->value_head.dW[i] += dv * features[i];
    nn->value_head.db[0] += dv;

    // backprop into last hidden features
    int last_dim = (nn->hidden_count > 0) ? nn->hidden[nn->hidden_count - 1].out_dim : nn->input_dim;
    double* dfeatures = (double*)calloc((size_t)last_dim, sizeof(double));
    // from policy head
    for (int a = 0; a < nn->num_actions; ++a) {
        const double* Wrow = &nn->policy_head.W[a * nn->policy_head.in_dim];
        for (int i = 0; i < nn->policy_head.in_dim; ++i) dfeatures[i] += dlogits[a] * Wrow[i];
    }
    // from value head
    for (int i = 0; i < nn->value_head.in_dim; ++i) dfeatures[i] += dv * nn->value_head.W[i];

    // propagate through hidden layers
    for (int h = nn->hidden_count - 1; h >= 0; --h) {
        DenseLayer* L = &nn->hidden[h];
        const double* prev = (h > 0) ? acts[h - 1] : input;
        // apply relu gradient to dfeatures
        for (int o = 0; o < L->out_dim; ++o) dfeatures[o] *= relu_grad(acts[h][o]);
        // accumulate dW, db
        for (int o = 0; o < L->out_dim; ++o) {
            double* Wrow = &L->W[o * L->in_dim];
            double* dWrow = &L->dW[o * L->in_dim];
            double g = dfeatures[o];
            for (int i = 0; i < L->in_dim; ++i) dWrow[i] += g * prev[i];
            L->db[o] += g;
        }
        // compute new dfeatures for previous layer
        if (h > 0) {
            double* prev_d = (double*)calloc((size_t)L->in_dim, sizeof(double));
            for (int o = 0; o < L->out_dim; ++o) {
                const double* Wrow = &L->W[o * L->in_dim];
                double g = dfeatures[o];
                for (int i = 0; i < L->in_dim; ++i) prev_d[i] += g * Wrow[i];
            }
            free(dfeatures);
            dfeatures = prev_d;
        }
    }

    // cleanup
    for (int h = 0; h < nn->hidden_count; ++h) free(acts[h]);
    free(acts);
    free(dfeatures);
    free(policy);
    free(logits);
    free(dlogits);
}

void nn_update(NeuralNetwork* nn, double learning_rate) {
    if (!nn) return;
    // hidden layers
    for (int h = 0; h < nn->hidden_count; ++h) {
        DenseLayer* L = &nn->hidden[h];
        size_t Wn = (size_t)L->out_dim * (size_t)L->in_dim;
        for (size_t k = 0; k < Wn; ++k) { L->W[k] -= learning_rate * L->dW[k]; L->dW[k] = 0.0; }
        for (int o = 0; o < L->out_dim; ++o) { L->b[o] -= learning_rate * L->db[o]; L->db[o] = 0.0; }
    }
    // policy head
    size_t Wnp = (size_t)nn->policy_head.out_dim * (size_t)nn->policy_head.in_dim;
    for (size_t k = 0; k < Wnp; ++k) { nn->policy_head.W[k] -= learning_rate * nn->policy_head.dW[k]; nn->policy_head.dW[k] = 0.0; }
    for (int o = 0; o < nn->policy_head.out_dim; ++o) { nn->policy_head.b[o] -= learning_rate * nn->policy_head.db[o]; nn->policy_head.db[o] = 0.0; }
    // value head
    size_t Wnv = (size_t)nn->value_head.out_dim * (size_t)nn->value_head.in_dim;
    for (size_t k = 0; k < Wnv; ++k) { nn->value_head.W[k] -= learning_rate * nn->value_head.dW[k]; nn->value_head.dW[k] = 0.0; }
    for (int o = 0; o < nn->value_head.out_dim; ++o) { nn->value_head.b[o] -= learning_rate * nn->value_head.db[o]; nn->value_head.db[o] = 0.0; }
}

void one_hot(int index, int length, double* out) {
    for (int i = 0; i < length; ++i) out[i] = (i == index) ? 1.0 : 0.0;
}