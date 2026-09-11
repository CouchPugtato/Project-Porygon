#include "action_value.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define ACTION_VALUE_JOINT_ROWS FACTORIZED_JOINT_DIM
#define ACTION_VALUE_SINGLE_ROWS OBS_NUM_ACTIONS
#define ACTION_VALUE_TARGET_ROWS (2 * FACTORIZED_TARGET_DIM)
#define ACTION_VALUE_HEAD_ROWS \
    (ACTION_VALUE_JOINT_ROWS + ACTION_VALUE_SINGLE_ROWS + ACTION_VALUE_TARGET_ROWS)
#define ACTION_VALUE_MAX_LATENT_DIM 64

struct ActionValueModel {
    size_t hidden_dim;
    size_t latent_dim;
    size_t projection_offset;
    size_t projection_bias_offset;
    size_t head_offset;
    size_t head_bias_offset;
    size_t parameter_count;
    float* parameters;
    float* gradients;
    float* adam_mean;
    float* adam_variance;
    size_t gradient_samples;
    size_t adam_step;
};

typedef struct {
    float latent[ACTION_VALUE_MAX_LATENT_DIM];
    float head_scores[ACTION_VALUE_HEAD_ROWS];
    float probabilities[ACTION_VALUE_JOINT_ROWS];
    size_t probability_count;
    size_t selected_row;
    float selected_raw;
    float expected_raw;
    float spread;
    size_t legal_count;
    FactorizedPolicySnapshot policy;
} ActionValueWork;

static unsigned int action_value_random_next(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float action_value_random_signed(unsigned int* state) {
    return (float)((double)action_value_random_next(state) / 4294967295.0 * 2.0 - 1.0);
}

static size_t projection_index(const ActionValueModel* model, size_t latent, size_t hidden) {
    return model->projection_offset + latent * model->hidden_dim + hidden;
}

static size_t head_index(const ActionValueModel* model, size_t row, size_t latent) {
    return model->head_offset + row * model->latent_dim + latent;
}

static float head_score(const ActionValueModel* model, size_t row, const float* latent) {
    float value = model->parameters[model->head_bias_offset + row];
    size_t i;
    for (i = 0; i < model->latent_dim; ++i) {
        value += model->parameters[head_index(model, row, i)] * latent[i];
    }
    return value;
}

static void build_latent(
    const ActionValueModel* model,
    const float* hidden_state,
    float* latent
) {
    size_t i;
    size_t h;
    for (i = 0; i < model->latent_dim; ++i) {
        float value = model->parameters[model->projection_bias_offset + i];
        for (h = 0; h < model->hidden_dim; ++h) {
            value += model->parameters[projection_index(model, i, h)] * hidden_state[h];
        }
        latent[i] = tanhf(value);
    }
}

static void build_local_policy(
    const FactorizedPolicySnapshot* policy,
    const unsigned char* legal_mask,
    int slot,
    float* local_policy
) {
    const float* kind = slot == 0 ? policy->slot0_kind_policy : policy->slot1_kind_policy;
    const float* move = slot == 0 ? policy->slot0_move_policy : policy->slot1_move_policy;
    const float* sw = slot == 0 ? policy->slot0_switch_policy : policy->slot1_switch_policy;
    const float* tera = slot == 0 ? policy->slot0_tera_policy : policy->slot1_tera_policy;
    int base = slot == 0 ? 0 : FACTORIZED_LOCAL_ACTION_DIM;
    float total = 0.0f;
    int i;

    memset(local_policy, 0, FACTORIZED_LOCAL_ACTION_DIM * sizeof(*local_policy));
    for (i = 0; i < FACTORIZED_MOVE_DIM; ++i) {
        float tera_total = 0.0f;
        if (legal_mask[base + i]) tera_total += tera[0];
        if (legal_mask[base + FACTORIZED_MOVE_DIM + i]) tera_total += tera[1];
        if (tera_total > 0.0f) {
            if (legal_mask[base + i]) {
                local_policy[i] = kind[0] * move[i] * tera[0] / tera_total;
            }
            if (legal_mask[base + FACTORIZED_MOVE_DIM + i]) {
                local_policy[FACTORIZED_MOVE_DIM + i] =
                    kind[0] * move[i] * tera[1] / tera_total;
            }
        }
    }
    for (i = 0; i < FACTORIZED_SWITCH_DIM; ++i) {
        if (legal_mask[base + 8 + i]) local_policy[8 + i] = kind[1] * sw[i];
    }
    for (i = 0; i < FACTORIZED_LOCAL_ACTION_DIM; ++i) total += local_policy[i];
    if (total > 0.0f) {
        for (i = 0; i < FACTORIZED_LOCAL_ACTION_DIM; ++i) local_policy[i] /= total;
    }
}

static int add_target_component(
    ActionValueWork* work,
    const FactorizedActionChoice* choice,
    int slot,
    float* selected_raw,
    float* expected_raw
) {
    unsigned char target_bits;
    int target_index;
    const float* target_policy;
    float probability_sum = 0.0f;
    size_t row_base;
    int i;

    if (slot == 0) {
        if (!choice->slot0_has_action || choice->slot0_kind != FACTORIZED_ACTION_MOVE) return 1;
        target_bits = choice->slot0_target_mask;
        target_index = choice->slot0_target_index;
        target_policy = work->policy.slot0_target_policy;
    } else {
        if (!choice->slot1_has_action || choice->slot1_kind != FACTORIZED_ACTION_MOVE) return 1;
        target_bits = choice->slot1_target_mask;
        target_index = choice->slot1_target_index;
        target_policy = work->policy.slot1_target_policy;
    }
    if (target_bits == 0u) return 1;
    if (target_index < 0 || target_index >= FACTORIZED_TARGET_DIM ||
            !(target_bits & FACTORIZED_TARGET_BIT(target_index))) return 0;
    row_base = ACTION_VALUE_JOINT_ROWS + ACTION_VALUE_SINGLE_ROWS +
        (size_t)slot * FACTORIZED_TARGET_DIM;
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) {
        if (target_bits & FACTORIZED_TARGET_BIT(i)) probability_sum += target_policy[i];
    }
    if (probability_sum <= 0.0f) return 0;
    *selected_raw += work->head_scores[row_base + (size_t)target_index];
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) {
        if (target_bits & FACTORIZED_TARGET_BIT(i)) {
            *expected_raw += target_policy[i] / probability_sum *
                work->head_scores[row_base + (size_t)i];
        }
    }
    return 1;
}

static int prepare_work(
    const ActionValueModel* model,
    const GruModel* policy_model,
    const float* hidden_state,
    const unsigned char* legal_mask,
    const FactorizedActionChoice* choice,
    int action0,
    int action1,
    ActionValueWork* work
) {
    int dual;
    size_t row;
    float minimum = 0.0f;
    float maximum = 0.0f;
    int have_score = 0;

    memset(work, 0, sizeof(*work));
    build_latent(model, hidden_state, work->latent);
    for (row = 0; row < ACTION_VALUE_HEAD_ROWS; ++row) {
        work->head_scores[row] = head_score(model, row, work->latent);
    }
    dual = choice->slot0_has_action && choice->slot1_has_action;
    if (!gru_model_evaluate_policy_snapshot(
            policy_model, hidden_state, legal_mask, dual, &work->policy, NULL)) return 0;
    if (dual) {
        size_t index;
        if (action0 < 0 || action0 >= FACTORIZED_LOCAL_ACTION_DIM ||
                action1 < FACTORIZED_LOCAL_ACTION_DIM || action1 >= OBS_NUM_ACTIONS) return 0;
        work->selected_row = (size_t)action0 * FACTORIZED_LOCAL_ACTION_DIM +
            (size_t)(action1 - FACTORIZED_LOCAL_ACTION_DIM);
        work->probability_count = ACTION_VALUE_JOINT_ROWS;
        for (index = 0; index < ACTION_VALUE_JOINT_ROWS; ++index) {
            float probability = work->policy.joint_policy[index];
            work->probabilities[index] = probability;
            if (probability <= 0.0f) continue;
            work->expected_raw += probability * work->head_scores[index];
            if (!have_score || work->head_scores[index] < minimum) minimum = work->head_scores[index];
            if (!have_score || work->head_scores[index] > maximum) maximum = work->head_scores[index];
            have_score = 1;
            ++work->legal_count;
        }
    } else {
        float local_policy[FACTORIZED_LOCAL_ACTION_DIM];
        int slot = choice->slot0_has_action ? 0 : 1;
        int local_action = slot == 0 ? action0 : action1 - FACTORIZED_LOCAL_ACTION_DIM;
        int base = slot * FACTORIZED_LOCAL_ACTION_DIM;
        int i;
        if ((!choice->slot0_has_action && !choice->slot1_has_action) ||
                local_action < 0 || local_action >= FACTORIZED_LOCAL_ACTION_DIM) return 0;
        build_local_policy(&work->policy, legal_mask, slot, local_policy);
        work->selected_row = ACTION_VALUE_JOINT_ROWS + (size_t)base + (size_t)local_action;
        work->probability_count = FACTORIZED_LOCAL_ACTION_DIM;
        for (i = 0; i < FACTORIZED_LOCAL_ACTION_DIM; ++i) {
            size_t single_row = ACTION_VALUE_JOINT_ROWS + (size_t)base + (size_t)i;
            work->probabilities[i] = local_policy[i];
            if (local_policy[i] <= 0.0f) continue;
            work->expected_raw += local_policy[i] * work->head_scores[single_row];
            if (!have_score || work->head_scores[single_row] < minimum) minimum = work->head_scores[single_row];
            if (!have_score || work->head_scores[single_row] > maximum) maximum = work->head_scores[single_row];
            have_score = 1;
            ++work->legal_count;
        }
    }
    if (!have_score || work->selected_row >= ACTION_VALUE_HEAD_ROWS) return 0;
    if (dual) {
        if (work->probabilities[work->selected_row] <= 0.0f) return 0;
    } else {
        size_t selected_local = work->selected_row - ACTION_VALUE_JOINT_ROWS -
            (size_t)(choice->slot0_has_action ? 0 : FACTORIZED_LOCAL_ACTION_DIM);
        if (selected_local >= work->probability_count ||
                work->probabilities[selected_local] <= 0.0f) return 0;
    }
    work->selected_raw = work->head_scores[work->selected_row];
    if (!add_target_component(work, choice, 0, &work->selected_raw, &work->expected_raw) ||
            !add_target_component(work, choice, 1, &work->selected_raw, &work->expected_raw)) return 0;
    work->spread = maximum - minimum;
    return 1;
}

static void free_work(ActionValueWork* work) {
    if (!work) return;
    memset(work, 0, sizeof(*work));
}

ActionValueModel* action_value_model_create(
    size_t hidden_dim,
    size_t latent_dim,
    unsigned int seed
) {
    ActionValueModel* model;
    size_t i;
    float scale;
    if (hidden_dim == 0 || latent_dim == 0 || latent_dim > ACTION_VALUE_MAX_LATENT_DIM) return NULL;
    model = (ActionValueModel*)calloc(1, sizeof(*model));
    if (!model) return NULL;
    model->hidden_dim = hidden_dim;
    model->latent_dim = latent_dim;
    model->projection_offset = 0;
    model->projection_bias_offset = latent_dim * hidden_dim;
    model->head_offset = model->projection_bias_offset + latent_dim;
    model->head_bias_offset = model->head_offset + ACTION_VALUE_HEAD_ROWS * latent_dim;
    model->parameter_count = model->head_bias_offset + ACTION_VALUE_HEAD_ROWS;
    model->parameters = (float*)calloc(model->parameter_count, sizeof(float));
    model->gradients = (float*)calloc(model->parameter_count, sizeof(float));
    model->adam_mean = (float*)calloc(model->parameter_count, sizeof(float));
    model->adam_variance = (float*)calloc(model->parameter_count, sizeof(float));
    if (!model->parameters || !model->gradients || !model->adam_mean || !model->adam_variance) {
        action_value_model_destroy(model);
        return NULL;
    }
    scale = sqrtf(2.0f / (float)(hidden_dim + latent_dim));
    for (i = 0; i < latent_dim * hidden_dim; ++i) {
        model->parameters[i] = action_value_random_signed(&seed) * scale;
    }
    return model;
}

void action_value_model_destroy(ActionValueModel* model) {
    if (!model) return;
    free(model->parameters);
    free(model->gradients);
    free(model->adam_mean);
    free(model->adam_variance);
    free(model);
}

size_t action_value_model_hidden_dim(const ActionValueModel* model) {
    return model ? model->hidden_dim : 0;
}

size_t action_value_model_latent_dim(const ActionValueModel* model) {
    return model ? model->latent_dim : 0;
}

size_t action_value_model_parameter_count(const ActionValueModel* model) {
    return model ? model->parameter_count : 0;
}

int action_value_model_export_parameters(
    const ActionValueModel* model,
    float* parameters,
    size_t count
) {
    if (!model || !parameters || count != model->parameter_count) return 0;
    memcpy(parameters, model->parameters, count * sizeof(*parameters));
    return 1;
}

int action_value_model_import_parameters(
    ActionValueModel* model,
    const float* parameters,
    size_t count
) {
    if (!model || !parameters || count != model->parameter_count) return 0;
    memcpy(model->parameters, parameters, count * sizeof(*parameters));
    return 1;
}

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
) {
    ActionValueWork work;
    if (!model || !policy_model || !hidden_state || !legal_mask || !choice || !prediction) return 0;
    if (!prepare_work(model, policy_model, hidden_state, legal_mask, choice, action0, action1, &work)) {
        free_work(&work);
        return 0;
    }
    prediction->baseline_value = baseline_value;
    prediction->advantage = work.selected_raw - work.expected_raw;
    prediction->q_value = baseline_value + prediction->advantage;
    prediction->legal_action_spread = work.spread;
    prediction->legal_action_count = work.legal_count;
    free_work(&work);
    return isfinite(prediction->q_value) && isfinite(prediction->advantage);
}

void action_value_model_clear_gradients(ActionValueModel* model) {
    if (!model) return;
    memset(model->gradients, 0, model->parameter_count * sizeof(*model->gradients));
    model->gradient_samples = 0;
}

static void accumulate_head_row(
    ActionValueModel* model,
    const ActionValueWork* work,
    size_t row,
    float coefficient,
    float* latent_gradient
) {
    size_t i;
    if (coefficient == 0.0f) return;
    for (i = 0; i < model->latent_dim; ++i) {
        size_t index = head_index(model, row, i);
        model->gradients[index] += coefficient * work->latent[i];
        latent_gradient[i] += coefficient * model->parameters[index];
    }
    model->gradients[model->head_bias_offset + row] += coefficient;
}

static int accumulate_target_rows(
    ActionValueModel* model,
    const ActionValueWork* work,
    const FactorizedActionChoice* choice,
    int slot,
    float error,
    float* latent_gradient
) {
    unsigned char bits;
    int selected;
    const float* policy;
    float sum = 0.0f;
    size_t row_base;
    int i;
    if (slot == 0) {
        if (!choice->slot0_has_action || choice->slot0_kind != FACTORIZED_ACTION_MOVE) return 1;
        bits = choice->slot0_target_mask;
        selected = choice->slot0_target_index;
        policy = work->policy.slot0_target_policy;
    } else {
        if (!choice->slot1_has_action || choice->slot1_kind != FACTORIZED_ACTION_MOVE) return 1;
        bits = choice->slot1_target_mask;
        selected = choice->slot1_target_index;
        policy = work->policy.slot1_target_policy;
    }
    if (bits == 0u) return 1;
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) {
        if (bits & FACTORIZED_TARGET_BIT(i)) sum += policy[i];
    }
    if (sum <= 0.0f) return 0;
    row_base = ACTION_VALUE_JOINT_ROWS + ACTION_VALUE_SINGLE_ROWS +
        (size_t)slot * FACTORIZED_TARGET_DIM;
    for (i = 0; i < FACTORIZED_TARGET_DIM; ++i) {
        float coefficient;
        if (!(bits & FACTORIZED_TARGET_BIT(i))) continue;
        coefficient = ((i == selected) ? 1.0f : 0.0f) - policy[i] / sum;
        accumulate_head_row(model, work, row_base + (size_t)i,
            error * coefficient, latent_gradient);
    }
    return 1;
}

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
) {
    ActionValueWork work;
    float latent_gradient[ACTION_VALUE_MAX_LATENT_DIM] = {0};
    float advantage;
    float error;
    size_t i;
    int dual;
    int ok = 0;
    if (!model || !policy_model || !hidden_state || !legal_mask || !choice) return 0;
    if (!prepare_work(model, policy_model, hidden_state, legal_mask, choice, action0, action1, &work)) {
        free_work(&work);
        return 0;
    }
    advantage = work.selected_raw - work.expected_raw;
    error = baseline_value + advantage - target_value;
    if (loss_out) *loss_out = 0.5f * error * error;
    dual = choice->slot0_has_action && choice->slot1_has_action;
    for (i = 0; i < work.probability_count; ++i) {
        size_t row = dual ? i : ACTION_VALUE_JOINT_ROWS +
            (size_t)(choice->slot0_has_action ? 0 : FACTORIZED_LOCAL_ACTION_DIM) + i;
        float coefficient = ((row == work.selected_row) ? 1.0f : 0.0f) -
            work.probabilities[i];
        accumulate_head_row(model, &work, row, error * coefficient, latent_gradient);
    }
    if (!accumulate_target_rows(model, &work, choice, 0, error, latent_gradient) ||
            !accumulate_target_rows(model, &work, choice, 1, error, latent_gradient)) goto cleanup;
    for (i = 0; i < model->latent_dim; ++i) {
        float projection_gradient = latent_gradient[i] *
            (1.0f - work.latent[i] * work.latent[i]);
        size_t h;
        model->gradients[model->projection_bias_offset + i] += projection_gradient;
        for (h = 0; h < model->hidden_dim; ++h) {
            model->gradients[projection_index(model, i, h)] +=
                projection_gradient * hidden_state[h];
        }
    }
    ++model->gradient_samples;
    ok = 1;

cleanup:
    free_work(&work);
    return ok;
}

int action_value_model_apply_adam(
    ActionValueModel* model,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float gradient_clip,
    float l2_coefficient
) {
    double norm_square = 0.0;
    float scale = 1.0f;
    float bias_correction1;
    float bias_correction2;
    size_t i;
    if (!model || model->gradient_samples == 0 || learning_rate <= 0.0f ||
            beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f ||
            epsilon <= 0.0f || gradient_clip <= 0.0f || l2_coefficient < 0.0f) return 0;
    for (i = 0; i < model->parameter_count; ++i) {
        model->gradients[i] /= (float)model->gradient_samples;
        model->gradients[i] += l2_coefficient * model->parameters[i];
        norm_square += (double)model->gradients[i] * model->gradients[i];
    }
    if (norm_square > (double)gradient_clip * gradient_clip) {
        scale = gradient_clip / (float)sqrt(norm_square);
    }
    ++model->adam_step;
    bias_correction1 = 1.0f - powf(beta1, (float)model->adam_step);
    bias_correction2 = 1.0f - powf(beta2, (float)model->adam_step);
    for (i = 0; i < model->parameter_count; ++i) {
        float gradient = model->gradients[i] * scale;
        float mean_hat;
        float variance_hat;
        model->adam_mean[i] = beta1 * model->adam_mean[i] + (1.0f - beta1) * gradient;
        model->adam_variance[i] = beta2 * model->adam_variance[i] +
            (1.0f - beta2) * gradient * gradient;
        mean_hat = model->adam_mean[i] / bias_correction1;
        variance_hat = model->adam_variance[i] / bias_correction2;
        model->parameters[i] -= learning_rate * mean_hat / (sqrtf(variance_hat) + epsilon);
    }
    action_value_model_clear_gradients(model);
    return 1;
}

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t hidden_dim;
    uint64_t latent_dim;
    uint64_t parameter_count;
    uint64_t parameter_checksum;
} ActionValueFileHeader;

static uint64_t action_value_checksum(const float* parameters, size_t count) {
    const unsigned char* bytes = (const unsigned char*)parameters;
    size_t byte_count = count * sizeof(*parameters);
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0; i < byte_count; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int action_value_model_save(const char* path, const ActionValueModel* model) {
    ActionValueFileHeader header;
    char* temporary_path;
    FILE* out;
    size_t path_length;
    int ok = 0;
    if (!path || !*path || !model) return 0;
    path_length = strlen(path);
    temporary_path = (char*)malloc(path_length + 5u);
    if (!temporary_path) return 0;
    memcpy(temporary_path, path, path_length);
    memcpy(temporary_path + path_length, ".tmp", 5u);
    out = fopen(temporary_path, "wb");
    if (!out) goto cleanup;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "PORYQ01", 7u);
    header.version = 1u;
    header.hidden_dim = (uint64_t)model->hidden_dim;
    header.latent_dim = (uint64_t)model->latent_dim;
    header.parameter_count = (uint64_t)model->parameter_count;
    header.parameter_checksum = action_value_checksum(
        model->parameters, model->parameter_count);
    if (fwrite(&header, sizeof(header), 1u, out) != 1u ||
            fwrite(model->parameters, sizeof(float), model->parameter_count, out) !=
                model->parameter_count ||
            fflush(out) != 0 || fclose(out) != 0) {
        out = NULL;
        goto cleanup;
    }
    out = NULL;
#ifdef _WIN32
    if (!MoveFileExA(temporary_path, path,
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) goto cleanup;
#else
    if (rename(temporary_path, path) != 0) goto cleanup;
#endif
    ok = 1;

cleanup:
    if (out) fclose(out);
    if (!ok) remove(temporary_path);
    free(temporary_path);
    return ok;
}

ActionValueModel* action_value_model_load(const char* path, size_t expected_hidden_dim) {
    ActionValueFileHeader header;
    ActionValueModel* model = NULL;
    FILE* in;
    int trailing;
    if (!path || !*path || expected_hidden_dim == 0) return NULL;
    in = fopen(path, "rb");
    if (!in) return NULL;
    if (fread(&header, sizeof(header), 1u, in) != 1u ||
            memcmp(header.magic, "PORYQ01", 7u) != 0 ||
            header.version != 1u || header.hidden_dim != expected_hidden_dim ||
            header.latent_dim == 0 || header.latent_dim > 1024u) goto failure;
    model = action_value_model_create(
        (size_t)header.hidden_dim, (size_t)header.latent_dim, 0u);
    if (!model || header.parameter_count != model->parameter_count ||
            fread(model->parameters, sizeof(float), model->parameter_count, in) !=
                model->parameter_count ||
            action_value_checksum(model->parameters, model->parameter_count) !=
                header.parameter_checksum) goto failure;
    trailing = fgetc(in);
    if (trailing != EOF || ferror(in)) goto failure;
    fclose(in);
    return model;

failure:
    fclose(in);
    action_value_model_destroy(model);
    return NULL;
}
