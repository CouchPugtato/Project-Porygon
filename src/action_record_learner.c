#include "action_record_learner.h"
#include "policy_decoding.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int action_log(const GruModel* model, const ActionLearningRecord* row, double* result) {
    FactorizedActionChoice action = row->choice;
    FactorizedPolicySnapshot policy;
    action.slot0_target_mask = action.slot1_target_mask = 0;
    return gru_model_evaluate_policy_snapshot(model, row->hidden, row->legal,
        action.slot0_has_action && action.slot1_has_action, &policy, NULL) &&
        policy_choice_log_probability(&policy, row->legal, &action, result) && isfinite(*result);
}

static int measure(const GruModel* actor, const ActionLearningRecord* rows, size_t count, double* maximum) {
    *maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        double current;
        if (!action_log(actor, rows + i, &current)) return 0;
        current += rows[i].conditional_target_log_probability;
        double difference = fabs(current - rows[i].behavior_log_probability);
        if (!isfinite(difference)) return 0;
        if (difference > *maximum) *maximum = difference;
    }
    return 1;
}

int action_record_learn(const GruModel* reference, const GruModel* behavior, GruModel* actor,
    GruActionHeadOptimizer** optimizer, const ActionLearningRecord* rows, size_t count,
    const ActionLearningOptions* options, ActionLearningProgress* progress
) {
    if (!options || !progress || !rows || !count || count > 128 || !optimizer || !*optimizer ||
        gru_action_head_optimizer_model(*optimizer) != actor || !options->minibatch || options->minibatch > 8 ||
        !options->max_updates || options->max_updates > 8 || !options->invocation_updates || options->invocation_updates > 8 ||
        !isfinite(options->learning_rate) || options->learning_rate <= 0 || options->learning_rate > .0001f ||
        !isfinite(options->max_abs_log_ratio) || options->max_abs_log_ratio <= 0 || options->max_abs_log_ratio > .05f ||
        progress->guard_stopped || progress->updates > options->max_updates ||
        progress->cursor != (progress->updates * options->minibatch < count ? progress->updates * options->minibatch : count) ||
        gru_model_action_head_compatibility_reason(reference, behavior) ||
        gru_model_action_head_compatibility_reason(reference, actor)) return 0;
    for (size_t i = 0; i < count; ++i) {
        double before;
        if (!rows[i].hidden || !isfinite(rows[i].terminal_return) || fabsf(rows[i].terminal_return) > 1 ||
            !isfinite(rows[i].conditional_target_log_probability) || rows[i].conditional_target_log_probability > 0 ||
            !isfinite(rows[i].behavior_log_probability) || rows[i].behavior_log_probability > 0) return 0;
        for (size_t h = 0; h < gru_model_hidden_dim(actor); ++h) if (!isfinite(rows[i].hidden[h])) return 0;
        for (size_t a = 0; a < 28; ++a) if (rows[i].legal[a] > 1) return 0;
        if (!action_log(behavior, rows + i, &before) ||
            fabs(before + rows[i].conditional_target_log_probability - rows[i].behavior_log_probability) > 1e-6) return 0;
    }
    size_t parameters = gru_model_parameter_count(actor), moments = gru_action_head_optimizer_state_count(*optimizer);
    float* saved = malloc((parameters + 2 * moments) * sizeof(float));
    if (!saved) return 0;
    uint64_t step;
    int ok = gru_action_head_optimizer_export_state(*optimizer, saved + parameters, saved + parameters + moments, moments, &step) &&
        step == progress->updates && measure(actor, rows, count, &progress->max_abs_log_ratio) &&
        progress->max_abs_log_ratio <= options->max_abs_log_ratio;
    for (size_t update = 0; ok && update < options->invocation_updates &&
            progress->updates < options->max_updates && progress->cursor < count; ++update) {
        ok = gru_model_export_parameters(actor, saved, parameters) &&
            gru_action_head_optimizer_export_state(*optimizer, saved + parameters, saved + parameters + moments, moments, &step);
        GruActionHeadOptimizer* rollback = gru_action_head_optimizer_create(actor);
        if (!ok || !rollback || !gru_action_head_optimizer_import_state(rollback, saved + parameters,
                saved + parameters + moments, moments, step)) {
            gru_action_head_optimizer_free(rollback); ok = 0; break;
        }
        size_t end = progress->cursor + options->minibatch;
        if (end > count) end = count;
        for (size_t i = progress->cursor; ok && i < end; ++i)
            ok = gru_action_head_accumulate_hidden(*optimizer, rows[i].hidden, rows[i].legal, &rows[i].choice, rows[i].terminal_return);
        if (ok) ok = gru_action_head_apply_adam(*optimizer, options->learning_rate, .9f, .999f, 1e-8f, 1);
        double proposed = 0;
        if (ok) ok = !gru_model_action_head_compatibility_reason(reference, actor) && measure(actor, rows, count, &proposed);
        if (!ok || proposed > options->max_abs_log_ratio) {
            int restored = gru_model_import_parameters(actor, saved, parameters);
            gru_action_head_optimizer_free(*optimizer); *optimizer = rollback;
            if (ok && restored) { progress->guard_stopped = 1; progress->rejected_abs_log_ratio = proposed; }
            else ok = 0;
            break;
        }
        gru_action_head_optimizer_free(rollback);
        progress->cursor = end;
        ++progress->updates;
        progress->max_abs_log_ratio = proposed;
    }
    free(saved);
    return ok;
}
