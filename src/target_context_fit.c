#include "target_context_fit.h"
#include "target_context.h"
#include "target_sidecar.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char validation, slot, move, legal, best; TargetContext context; } Example;
typedef struct { size_t count, correct; double nll, preferred_mass; } Metrics;
typedef struct {
    TargetSidecar* sidecar;
    size_t parameters, base_parameters, updates;
    float *gradient, *first, *second, *best;
    int contextual, best_epoch, epochs;
    Metrics before, train, validation;
} Fit;

static int predict(const Fit* fit, const Example* e, const float* hidden, float policy[4]) {
    float residual[4] = {0};
    int target;
    if (fit->contextual) for (target = 0; target < 4; ++target) {
        size_t indices[2], k, count = target_context_feature_indices(&e->context, target, indices);
        for (k = 0; k < count; ++k) residual[target] += fit->sidecar->weights[fit->base_parameters + indices[k]];
    }
    return target_sidecar_predict_residual(fit->sidecar, hidden, e->slot, e->move, e->legal, residual, policy);
}

static Metrics evaluate(const Fit* fit, const Example* examples, const float* hidden, size_t count, int validation) {
    Metrics m = {0};
    size_t i;
    for (i = 0; i < count; ++i) if (examples[i].validation == validation) {
        const Example* e = &examples[i];
        float policy[4];
        int target, selected = -1;
        if (!predict(fit, e, hidden + i * fit->sidecar->hidden_dim, policy)) { m.nll = NAN; return m; }
        m.nll += target_sidecar_backward(fit->sidecar, hidden + i * fit->sidecar->hidden_dim,
            e->slot, e->move, e->legal, e->best, policy, NULL);
        for (target = 0; target < 4; ++target) {
            if ((e->legal & (1 << target)) && (selected < 0 || policy[target] > policy[selected])) selected = target;
            if (e->best & (1 << target)) m.preferred_mass += policy[target];
        }
        m.correct += !!(e->best & (1 << selected));
        ++m.count;
    }
    if (!m.count) { m.nll = NAN; return m; }
    m.nll /= m.count; m.preferred_mass /= m.count;
    return m;
}

static void free_fit(Fit* fit) {
    target_sidecar_free(fit->sidecar); free(fit->gradient); free(fit->first); free(fit->second); free(fit->best);
}

static int fit_model(Fit* fit, const GruModel* parent, const Example* examples,
    const float* hidden, size_t count, int contextual
) {
    size_t i;
    int epoch, stale = 0;
    memset(fit, 0, sizeof(*fit));
    fit->sidecar = target_sidecar_create(parent);
    if (!fit->sidecar) return 0;
    fit->contextual = contextual;
    fit->base_parameters = target_sidecar_parameter_count(fit->sidecar);
    fit->parameters = fit->base_parameters + (contextual ? TARGET_CONTEXT_PARAMETER_COUNT : 0);
    float* weights = realloc(fit->sidecar->weights, fit->parameters * sizeof(float));
    if (!weights) return 0;
    fit->sidecar->weights = weights;
    if (contextual) memset(weights + fit->base_parameters, 0, TARGET_CONTEXT_PARAMETER_COUNT * sizeof(float));
    fit->gradient = calloc(fit->parameters, sizeof(float)); fit->first = calloc(fit->parameters, sizeof(float));
    fit->second = calloc(fit->parameters, sizeof(float)); fit->best = malloc(fit->parameters * sizeof(float));
    if (!fit->gradient || !fit->first || !fit->second || !fit->best) return 0;
    fit->before = fit->validation = evaluate(fit, examples, hidden, count, 1);
    if (fit->before.count < 100 || !isfinite(fit->before.nll)) return 0;
    memcpy(fit->best, weights, fit->parameters * sizeof(float));
    for (epoch = 1; epoch <= 10; ++epoch) {
        size_t batch = 0;
        for (i = 0; i < count; ++i) if (!examples[i].validation) {
            const Example* e = &examples[i];
            const float* state = hidden + i * fit->sidecar->hidden_dim;
            float policy[4];
            int target, positives = 0;
            if (!predict(fit, e, state, policy) || !isfinite(target_sidecar_backward(fit->sidecar, state,
                    e->slot, e->move, e->legal, e->best, policy, fit->gradient))) return 0;
            for (target = 0; target < 4; ++target) positives += !!(e->best & (1 << target));
            if (contextual) for (target = 0; target < 4; ++target) {
                size_t indices[2], k, n = target_context_feature_indices(&e->context, target, indices);
                float difference = policy[target] - ((e->best & (1 << target)) ? 1.0f / positives : 0);
                for (k = 0; k < n; ++k) fit->gradient[fit->base_parameters + indices[k]] += difference;
            }
            if (++batch == 64) {
                target_fit_adam_batch(weights, fit->parameters, fit->gradient, fit->first, fit->second, batch, ++fit->updates);
                batch = 0;
            }
        }
        if (batch) target_fit_adam_batch(weights, fit->parameters, fit->gradient, fit->first, fit->second, batch, ++fit->updates);
        Metrics current = evaluate(fit, examples, hidden, count, 1);
        if (!isfinite(current.nll)) return 0;
        fit->epochs = epoch;
        if (current.nll < fit->validation.nll - 1e-6) {
            fit->validation = current; fit->best_epoch = epoch; stale = 0;
            memcpy(fit->best, weights, fit->parameters * sizeof(float));
        } else ++stale;
        fprintf(stderr, "[target-context] mode=%s epoch=%d/10 validation_nll=%.6f accuracy=%.4f best_epoch=%d\n",
            contextual ? "type_pair_residual" : "baseline", epoch, current.nll, (double)current.correct / current.count, fit->best_epoch);
        if (stale >= 3) break;
    }
    memcpy(weights, fit->best, fit->parameters * sizeof(float));
    fit->train = evaluate(fit, examples, hidden, count, 0);
    return fit->train.count >= 200 && isfinite(fit->train.nll);
}

static void write_metrics(FILE* out, const Metrics* m) {
    fprintf(out, "{\"examples\":%zu,\"nll\":%.9g,\"accuracy\":%.9g,\"preferred_mass\":%.9g}",
        m->count, m->nll, (double)m->correct / m->count, m->preferred_mass);
}

int target_context_compare(const char* dataset, const GruModel* parent, FILE* report) {
    FILE* file = fopen(dataset, "rb");
    uint32_t dimensions, count;
    uint64_t fingerprint = target_parent_fingerprint(parent), stored;
    char magic[9];
    Example* examples = NULL;
    float* hidden = NULL;
    Fit fits[2] = {{0}};
    size_t i, j;
    int ok = 0, mode;
    if (!file || !fingerprint || fread(magic, 1, 9, file) != 9 || memcmp(magic, "PGTCONTX1", 9) ||
            fread(&dimensions, 4, 1, file) != 1 || dimensions != gru_model_hidden_dim(parent) ||
            fread(&count, 4, 1, file) != 1 || count < 300 || count > 100000 ||
            (size_t)count * dimensions > 16 * 1024 * 1024 || fread(&stored, 8, 1, file) != 1 || stored != fingerprint) goto cleanup;
    examples = calloc(count, sizeof(*examples)); hidden = calloc((size_t)count * dimensions, sizeof(float));
    if (!examples || !hidden) goto cleanup;
    for (i = 0; i < count; ++i) {
        Example* e = &examples[i];
        if (fread(e, 1, 5, file) != 5 || e->validation > 1 || e->slot > 1 || e->move > 3 ||
                !e->best || e->legal > 15 || (e->best & ~e->legal) || e->best == e->legal ||
                fread(hidden + i * dimensions, sizeof(float), dimensions, file) != dimensions ||
                fread(&e->context, 1, sizeof(e->context), file) != sizeof(e->context) || !target_context_valid(&e->context)) goto cleanup;
        for (j = 0; j < dimensions; ++j) if (!isfinite(hidden[i * dimensions + j])) goto cleanup;
    }
    if (fgetc(file) != EOF) goto cleanup;
    for (mode = 0; mode < 2; ++mode) if (!fit_model(&fits[mode], parent, examples, hidden, count, mode)) goto cleanup;
    if (target_parent_fingerprint(parent) != fingerprint || fits[0].before.nll != fits[1].before.nll) goto cleanup;
    fprintf(report, "{\"diagnostic\":\"target_context_comparison\",\"parent_unchanged\":true,\"weights_published\":false,"
        "\"gameplay_authorized\":false,\"optimizer\":\"adam\",\"learning_rate\":0.001,\"minibatch_examples\":64,"
        "\"max_epochs\":10,\"early_stop_patience\":3,\"context_parameters\":%d,\"before\":", TARGET_CONTEXT_PARAMETER_COUNT);
    write_metrics(report, &fits[0].before);
    for (mode = 0; mode < 2; ++mode) {
        fprintf(report, ",\"%s\":{\"best_epoch\":%d,\"epochs_completed\":%d,\"parameters\":%zu,\"train\":",
            mode ? "type_pair_residual" : "baseline", fits[mode].best_epoch, fits[mode].epochs, fits[mode].parameters);
        write_metrics(report, &fits[mode].train); fputs(",\"validation\":", report); write_metrics(report, &fits[mode].validation); fputc('}', report);
    }
    fprintf(report, ",\"context_lower_validation_nll\":%s}\n", fits[1].validation.nll < fits[0].validation.nll ? "true" : "false");
    ok = !ferror(report) && fflush(report) == 0;
cleanup:
    if (file) fclose(file);
    for (mode = 0; mode < 2; ++mode) free_fit(&fits[mode]);
    free(examples); free(hidden);
    if (!ok) fprintf(stderr, "[target-context] comparison failed; no weights published\n");
    return ok;
}
