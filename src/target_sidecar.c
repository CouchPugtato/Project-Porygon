#include "target_sidecar.h"
#include "id_tables.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

static uint64_t fingerprint_bytes(uint64_t hash, const void* data, size_t count) {
    const unsigned char* bytes = data;
    size_t i;
    for (i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
    return hash;
}

uint64_t target_parent_fingerprint(const GruModel* model) {
    if (!model) return 0;
    size_t count = gru_model_parameter_count(model);
    float* weights = malloc(count * sizeof(float));
    uint64_t hash = UINT64_C(14695981039346656037);
    uint64_t dimensions[3] = {gru_model_input_dim(model), gru_model_hidden_dim(model), gru_model_num_actions(model)};
    if (!weights || !gru_model_export_parameters(model, weights, count)) { free(weights); return 0; }
    hash = fingerprint_bytes(hash, dimensions, sizeof(dimensions));
    hash = fingerprint_bytes(hash, weights, count * sizeof(float));
    free(weights);
    return hash;
}

size_t target_sidecar_parameter_count(const TargetSidecar* sidecar) {
    return sidecar ? 32 * (sidecar->hidden_dim + 1) +
        (sidecar->contextual ? TARGET_CONTEXT_PARAMETER_COUNT : 0) : 0;
}

uint64_t target_sidecar_fingerprint(const TargetSidecar* sidecar) {
    if (!sidecar || !sidecar->weights) return 0;
    uint64_t metadata[3] = {sidecar->parent_fingerprint, sidecar->hidden_dim, sidecar->contextual != 0};
    uint64_t hash = fingerprint_bytes(UINT64_C(14695981039346656037), metadata, sizeof(metadata));
    return fingerprint_bytes(hash, sidecar->weights, target_sidecar_parameter_count(sidecar) * sizeof(float));
}

TargetSidecar* target_sidecar_create(const GruModel* parent) {
    if (!parent) return NULL;
    TargetSidecar* s = calloc(1, sizeof(*s));
    int slot, move;
    if (!s) return NULL;
    s->hidden_dim = gru_model_hidden_dim(parent);
    s->parent_fingerprint = target_parent_fingerprint(parent);
    if (!s->hidden_dim || s->hidden_dim > 4096 || !s->parent_fingerprint) { free(s); return NULL; }
    s->weights = calloc(target_sidecar_parameter_count(s), sizeof(float));
    if (!s->weights) { free(s); return NULL; }
    for (slot = 0; slot < 2; ++slot) for (move = 0; move < 4; ++move) {
        if (!gru_model_export_target_head(parent, slot, s->weights + (slot * 4 + move) * 4 * (s->hidden_dim + 1),
                4 * (s->hidden_dim + 1))) { target_sidecar_free(s); return NULL; }
    }
    return s;
}

void target_sidecar_free(TargetSidecar* s) { if (s) { free(s->weights); free(s); } }

int target_sidecar_predict_residual(const TargetSidecar* s, const float* hidden,
    int slot, int move, unsigned char legal, const float residual[4], float* policy
) {
    float largest = -FLT_MAX;
    float sum = 0;
    size_t h;
    int target;
    if (!s || !hidden || !policy || slot < 0 || slot > 1 || move < 0 || move > 3 || !legal || (legal & ~15)) return 0;
    for (h = 0; h < s->hidden_dim; ++h) if (!isfinite(hidden[h])) return 0;
    for (target = 0; target < 4; ++target) {
        const float* row = s->weights + ((slot * 4 + move) * 4 + target) * (s->hidden_dim + 1);
        float logit = row[s->hidden_dim];
        for (h = 0; h < s->hidden_dim; ++h) logit += row[h] * hidden[h];
        if (residual) logit += residual[target];
        if (!isfinite(logit)) return 0;
        policy[target] = logit;
        if ((legal & (1 << target)) && logit > largest) largest = logit;
    }
    for (target = 0; target < 4; ++target) {
        policy[target] = legal & (1 << target) ? expf(policy[target] - largest) : 0;
        sum += policy[target];
    }
    if (!(sum > 0) || !isfinite(sum)) return 0;
    for (target = 0; target < 4; ++target) policy[target] /= (float)sum;
    return 1;
}

int target_sidecar_predict(const TargetSidecar* s, const float* hidden,
    int slot, int move, unsigned char legal, float* policy
) {
    return target_sidecar_predict_residual(s, hidden, slot, move, legal, NULL, policy);
}

int target_sidecar_predict_context(const TargetSidecar* s, const float* hidden,
    int slot, int move, unsigned char legal, const TargetContext* context, float* policy
) {
    float residual[4] = {0};
    int target;
    if (!s || !s->contextual || !context || !target_context_valid(context)) return 0;
    size_t base = 32 * (s->hidden_dim + 1);
    for (target = 0; target < 4; ++target) {
        size_t indices[2], k, n = target_context_feature_indices(context, target, indices);
        for (k = 0; k < n; ++k) residual[target] += s->weights[base + indices[k]];
    }
    return target_sidecar_predict_residual(s, hidden, slot, move, legal, residual, policy);
}

double target_sidecar_backward(const TargetSidecar* s, const float* hidden,
    int slot, int move, unsigned char legal, unsigned char best, const float policy[4], float* gradient
) {
    double loss = 0;
    int target, count = 0;
    size_t h;
    if (!best || (best & ~legal)) return NAN;
    for (target = 0; target < 4; ++target) count += !!(best & (1 << target));
    for (target = 0; target < 4; ++target) {
        float label = best & (1 << target) ? 1.0f / count : 0;
        float difference = policy[target] - label;
        if (label) loss -= label * log(fmax(policy[target], 1e-30));
        if (gradient) {
            size_t offset = ((slot * 4 + move) * 4 + target) * (s->hidden_dim + 1);
            for (h = 0; h < s->hidden_dim; ++h) gradient[offset + h] += difference * hidden[h];
            gradient[offset + s->hidden_dim] += difference;
        }
    }
    return loss;
}

double target_sidecar_gradient(const TargetSidecar* s, const float* hidden,
    int slot, int move, unsigned char legal, unsigned char best, float* gradient
) {
    float policy[4];
    if (!target_sidecar_predict(s, hidden, slot, move, legal, policy)) return NAN;
    return target_sidecar_backward(s, hidden, slot, move, legal, best, policy, gradient);
}

static const char* visibility_reason(const Observation* observation, const ParsedRequest* request) {
    int i;
    if (request->forced_switch_any) return "forced_switch";
    for (i = 0; i < OBS_TEAM_SIZE; ++i) {
        const ObsPokemon* teams[2] = {&observation->self_team[i], &observation->opp_team[i]};
        int team;
        for (team = 0; team < 2; ++team) {
            const ObsPokemon* p = teams[team];
            if (p->active && !p->fainted && (!p->type1_known_mode || !p->type2_known_mode ||
                    p->type1_id <= 0 || p->type1_id >= OBS_NUM_TYPES || p->type2_id < 0 || p->type2_id >= OBS_NUM_TYPES)) return "unknown_visible_types";
        }
    }
    return NULL;
}

static const char* move_support_reason(const ParsedRequest* request, const FactorizedActionChoice* choice, int slot) {
    int has_action = slot ? choice->slot1_has_action : choice->slot0_has_action;
    int kind = slot ? choice->slot1_kind : choice->slot0_kind;
    int move = slot ? choice->slot1_move_index : choice->slot0_move_index;
    int tera = slot ? choice->slot1_use_tera : choice->slot0_use_tera;
    MoveSemanticMetadata metadata;
    if (!has_action || kind != FACTORIZED_ACTION_MOVE) return "not_move";
    int id = request->active[slot].move_id[move];
    if (tera || (request->active_team_idx_known[slot] && request->side_tera_used[request->active_team_idx[slot]])) return "tera";
    if (!move_semantics_from_id(id, &metadata)) return "unknown_move";
    if (!metadata.damaging) return "status_move";
    if (metadata.base_power <= 0 ||
            (metadata.category != 2 && metadata.category != 3) || metadata.accuracy <= 0 || metadata.accuracy > 100 ||
            move_type_from_id(id) <= 0) return "unsupported_move_metadata";
    if (!build_move_target_mask(request, slot, move) &&
            request->active[slot].move_target[move] != REQUEST_TARGET_ALL_ADJACENT &&
            request->active[slot].move_target[move] != REQUEST_TARGET_ALL_ADJACENT_FOES) return "unsupported_target_kind";
    return NULL;
}

const char* target_sidecar_support_reason(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice
) {
    const char* reason = visibility_reason(observation, request);
    int slot;
    if (reason) return reason;
    for (slot = 0; slot < 2; ++slot) {
        int has = slot ? choice->slot1_has_action : choice->slot0_has_action;
        int kind = slot ? choice->slot1_kind : choice->slot0_kind;
        if (has && kind == FACTORIZED_ACTION_MOVE && (reason = move_support_reason(request, choice, slot))) return reason;
    }
    return NULL;
}

const char* target_sidecar_slot_support_reason(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int slot
) {
    const char* reason = visibility_reason(observation, request);
    int other;
    if (reason) return reason;
    /* Tera can change an ally target's typing. Keep that restriction across
     * both selected moves even though status/metadata eligibility is local. */
    for (other = 0; other < 2; ++other) {
        int has = other ? choice->slot1_has_action : choice->slot0_has_action;
        int kind = other ? choice->slot1_kind : choice->slot0_kind;
        int tera = other ? choice->slot1_use_tera : choice->slot0_use_tera;
        if (has && kind == FACTORIZED_ACTION_MOVE && (tera || (request->active_team_idx_known[other] &&
                request->side_tera_used[request->active_team_idx[other]]))) return "tera";
    }
    return move_support_reason(request, choice, slot);
}

int target_sidecar_supports(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice
) {
    return target_sidecar_support_reason(observation, request, choice) == NULL;
}

static int publish_file(const char* temporary, const char* path) {
#ifdef _WIN32
    return MoveFileExA(temporary, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temporary, path) == 0;
#endif
}

static int finish_file(FILE* file) {
    int ok = fflush(file) == 0;
#ifdef _WIN32
    if (ok) ok = _commit(_fileno(file)) == 0;
#else
    if (ok) ok = fsync(fileno(file)) == 0;
#endif
    if (fclose(file) != 0) ok = 0;
    return ok;
}

int target_sidecar_save(const TargetSidecar* s, const char* path) {
    if (!s || !s->weights || !path || !*path) return 0;
    char temporary[4096];
    uint32_t dimensions = (uint32_t)s->hidden_dim;
    uint64_t checksum = fingerprint_bytes(UINT64_C(14695981039346656037), s->weights,
        target_sidecar_parameter_count(s) * sizeof(float));
    FILE* file;
    int ok;
    if (snprintf(temporary, sizeof(temporary), "%s.tmp", path) >= (int)sizeof(temporary)) return 0;
    file = fopen(temporary, "wb");
    if (!file) return 0;
    ok = fwrite(s->contextual ? "PGTARGTC1" : "PGTARGET1", 1, 9, file) == 9 && fwrite(&dimensions, 4, 1, file) == 1 &&
        fwrite(&s->parent_fingerprint, 8, 1, file) == 1 && fwrite(&checksum, 8, 1, file) == 1 &&
        fwrite(s->weights, sizeof(float), target_sidecar_parameter_count(s), file) == target_sidecar_parameter_count(s);
    if (!finish_file(file)) ok = 0;
    return ok && publish_file(temporary, path);
}

TargetSidecar* target_sidecar_load(const char* path, const GruModel* parent) {
    TargetSidecar* s = target_sidecar_create(parent);
    FILE* file = fopen(path, "rb");
    char magic[9];
    uint32_t dimensions;
    uint64_t fingerprint, checksum;
    size_t i;
    int ok = file && s && fread(magic, 1, 9, file) == 9 &&
        (!memcmp(magic, "PGTARGET1", 9) || !memcmp(magic, "PGTARGTC1", 9));
    if (ok && !memcmp(magic, "PGTARGTC1", 9)) {
        s->contextual = 1;
        float* weights = realloc(s->weights, target_sidecar_parameter_count(s) * sizeof(float));
        if (!weights) ok = 0;
        else s->weights = weights;
    }
    ok = ok &&
        fread(&dimensions, 4, 1, file) == 1 && dimensions == s->hidden_dim &&
        fread(&fingerprint, 8, 1, file) == 1 && fingerprint == s->parent_fingerprint &&
        fread(&checksum, 8, 1, file) == 1 &&
        fread(s->weights, sizeof(float), target_sidecar_parameter_count(s), file) == target_sidecar_parameter_count(s) &&
        fgetc(file) == EOF;
    if (file) fclose(file);
    if (ok) ok = checksum == fingerprint_bytes(UINT64_C(14695981039346656037), s->weights,
        target_sidecar_parameter_count(s) * sizeof(float));
    if (ok) for (i = 0; i < target_sidecar_parameter_count(s); ++i) if (!isfinite(s->weights[i])) ok = 0;
    if (!ok) { target_sidecar_free(s); return NULL; }
    return s;
}

typedef struct { unsigned char validation, slot, move, legal, best; } Example;
typedef struct { double nll; size_t count, correct; } Metrics;

void target_fit_adam_batch(float* weights, size_t count, float* gradient, float* first, float* second, size_t batch, size_t update) {
    size_t j;
    double correction1 = 1 - pow(.9, (double)update), correction2 = 1 - pow(.999, (double)update);
    for (j = 0; j < count; ++j) {
        float g = gradient[j] / batch;
        first[j] = .9f * first[j] + .1f * g;
        second[j] = .999f * second[j] + .001f * g * g;
        weights[j] -= .001f * (first[j] / correction1) / (sqrt(second[j] / correction2) + 1e-8);
        gradient[j] = 0;
    }
}

static Metrics evaluate(const TargetSidecar* s, const Example* examples, const float* hidden, size_t count, int validation, int group) {
    Metrics m = {0};
    size_t i;
    for (i = 0; i < count; ++i) if (examples[i].validation == validation &&
            (group < 0 || examples[i].slot * 4 + examples[i].move == group)) {
        float policy[4];
        int t, selected = -1;
        const Example* e = &examples[i];
        double loss = target_sidecar_gradient(s, hidden + i * s->hidden_dim, e->slot, e->move, e->legal, e->best, NULL);
        if (!isfinite(loss) || !target_sidecar_predict(s, hidden + i * s->hidden_dim, e->slot, e->move, e->legal, policy)) {
            m.nll = NAN; return m;
        }
        for (t = 0; t < 4; ++t) if ((e->legal & (1 << t)) && (selected < 0 || policy[t] > policy[selected])) selected = t;
        m.correct += !!(e->best & (1 << selected));
        m.nll += loss;
        ++m.count;
    }
    m.nll = m.count ? m.nll / m.count : NAN;
    return m;
}

static int fit_target_sidecar(const char* dataset, const GruModel* parent, const char* output, const char* report, int exploratory) {
    TargetSidecar* s = target_sidecar_create(parent);
    FILE* file = fopen(dataset, "rb");
    char magic[9], temporary[4096];
    uint32_t dimensions, count;
    uint64_t fingerprint;
    Example* examples = NULL;
    float *hidden = NULL, *gradient = NULL, *first = NULL, *second = NULL, *best_weights = NULL;
    Metrics before = {0}, after = {0}, train_before = {0}, train_after = {0};
    Metrics groups_before[8];
    size_t i, j, parameters = target_sidecar_parameter_count(s), updates = 0, batch = 0;
    int epoch, epochs_completed = 0, best_epoch = 0, stale = 0, ok = 0, passed = 0;
    if (!s || !file || fread(magic, 1, 9, file) != 9 || memcmp(magic, "PGTLABEL1", 9) ||
            fread(&dimensions, 4, 1, file) != 1 || dimensions != s->hidden_dim ||
            fread(&count, 4, 1, file) != 1 || count < 300 || count > 100000 ||
            (size_t)count * dimensions > 16 * 1024 * 1024 ||
            fread(&fingerprint, 8, 1, file) != 1 || fingerprint != s->parent_fingerprint) goto cleanup;
    examples = calloc(count, sizeof(*examples));
    hidden = calloc((size_t)count * dimensions, sizeof(float));
    gradient = calloc(parameters, sizeof(float)); first = calloc(parameters, sizeof(float)); second = calloc(parameters, sizeof(float));
    best_weights = malloc(parameters * sizeof(float));
    if (!examples || !hidden || !gradient || !first || !second || !best_weights) goto cleanup;
    for (i = 0; i < count; ++i) {
        Example* e = &examples[i];
        if (fread(e, 1, 5, file) != 5 || e->validation > 1 || e->slot > 1 || e->move > 3 ||
                !e->best || (e->best & ~e->legal) || e->legal > 15 || e->best == e->legal ||
                fread(hidden + i * dimensions, sizeof(float), dimensions, file) != dimensions) goto cleanup;
        for (j = 0; j < dimensions; ++j) if (!isfinite(hidden[i * dimensions + j])) goto cleanup;
    }
    if (fgetc(file) != EOF) goto cleanup;
    before = evaluate(s, examples, hidden, count, 1, -1); train_before = evaluate(s, examples, hidden, count, 0, -1);
    for (i = 0; i < 8; ++i) groups_before[i] = evaluate(s, examples, hidden, count, 1, (int)i);
    if (before.count < 100 || train_before.count < 200 || !isfinite(before.nll) || !isfinite(train_before.nll)) goto cleanup;
    after = before;
    memcpy(best_weights, s->weights, parameters * sizeof(float));
    for (epoch = 1; epoch <= 10; ++epoch) {
        for (i = 0; i < count; ++i) if (!examples[i].validation) {
            Example* e = &examples[i];
            if (!isfinite(target_sidecar_gradient(s, hidden + i * dimensions, e->slot, e->move, e->legal, e->best, gradient))) goto cleanup;
            ++batch;
            if (batch == 64) {
                target_fit_adam_batch(s->weights, parameters, gradient, first, second, batch, ++updates);
                batch = 0;
            }
        }
        /* A trailing validation row must not strand the last training batch. */
        if (batch) {
            target_fit_adam_batch(s->weights, parameters, gradient, first, second, batch, ++updates);
            batch = 0;
        }
        Metrics current = evaluate(s, examples, hidden, count, 1, -1);
        epochs_completed = epoch;
        if (!isfinite(current.nll)) goto cleanup;
        if (current.nll < after.nll - 1e-6) {
            after = current; best_epoch = epoch; stale = 0;
            memcpy(best_weights, s->weights, parameters * sizeof(float));
        } else ++stale;
        printf("[target-fit] epoch=%d/10 holdout_nll=%.6f accuracy=%.4f best_epoch=%d\n",
            epoch, current.nll, (double)current.correct / current.count, best_epoch);
        fflush(stdout);
        if (stale >= 3) break;
    }
    memcpy(s->weights, best_weights, parameters * sizeof(float));
    train_after = evaluate(s, examples, hidden, count, 0, -1);
    passed = best_epoch > 0 && after.nll < before.nll && after.correct >= before.correct &&
        isfinite(train_after.nll) && target_parent_fingerprint(parent) == s->parent_fingerprint;
    int exploratory_exception = !passed && exploratory && best_epoch > 0 && after.nll < before.nll &&
        isfinite(train_after.nll) && target_parent_fingerprint(parent) == s->parent_fingerprint;
    int published = passed || exploratory_exception;
    if (published && !target_sidecar_save(s, output)) goto cleanup;
    if (snprintf(temporary, sizeof(temporary), "%s.tmp", report) >= (int)sizeof(temporary)) goto cleanup;
    FILE* json = fopen(temporary, "w");
    if (!json) goto cleanup;
    fprintf(json, "{\"passed\":%s,\"sidecar_published\":%s,\"exploratory_exception\":%s,\"evaluation_only\":true,\"best_epoch\":%d,\"epochs_completed\":%d,\"max_epochs\":10,"
        "\"optimizer\":\"adam\",\"learning_rate\":0.001,\"minibatch_examples\":64,"
        "\"train_examples\":%zu,\"holdout_examples\":%zu,\"train_nll_before\":%.9g,\"train_nll_after\":%.9g,"
        "\"holdout_nll_before\":%.9g,\"holdout_nll_after\":%.9g,\"accuracy_before\":%.9g,\"accuracy_after\":%.9g,"
        "\"parent_fingerprint\":\"%016llx\",\"parent_unchanged\":true,\"holdout_groups\":[",
        passed ? "true" : "false", published ? "true" : "false", exploratory_exception ? "true" : "false",
        best_epoch, epochs_completed, train_before.count, before.count,
        train_before.nll, train_after.nll, before.nll, after.nll,
        (double)before.correct / before.count, (double)after.correct / after.count,
        (unsigned long long)s->parent_fingerprint);
    for (i = 0; i < 8; ++i) {
        Metrics group = evaluate(s, examples, hidden, count, 1, (int)i);
        fprintf(json, "%s{\"slot\":%zu,\"move\":%zu,\"examples\":%zu", i ? "," : "", i / 4, i % 4, group.count);
        if (group.count) fprintf(json, ",\"nll_before\":%.9g,\"nll_after\":%.9g,\"correct_before\":%zu,\"correct_after\":%zu",
            groups_before[i].nll, group.nll, groups_before[i].correct, group.correct);
        fputc('}', json);
    }
    fprintf(json, "],\"checks\":{\"holdout_nll_improved\":%s,\"holdout_accuracy_not_worse\":%s},\"failure_reasons\":[%s%s%s]}\n",
        after.nll < before.nll ? "true" : "false", after.correct >= before.correct ? "true" : "false",
        after.nll < before.nll ? "" : "\"holdout_nll_did_not_improve\"",
        !(after.nll < before.nll) && after.correct < before.correct ? "," : "",
        after.correct >= before.correct ? "" : "\"holdout_accuracy_decreased\"");
    ok = finish_file(json) && publish_file(temporary, report);
cleanup:
    if (file) fclose(file);
    free(examples); free(hidden); free(gradient); free(first); free(second); free(best_weights);
    target_sidecar_free(s);
    if (!ok) fprintf(stderr, "[target-fit] failed to read, fit, or publish the bounded target dataset\n");
    return ok;
}

int target_sidecar_fit(const char* dataset, const GruModel* parent, const char* output, const char* report) {
    return fit_target_sidecar(dataset, parent, output, report, 0);
}

int target_sidecar_fit_for_evaluation(const char* dataset, const GruModel* parent, const char* output, const char* report) {
    return fit_target_sidecar(dataset, parent, output, report, 1);
}
