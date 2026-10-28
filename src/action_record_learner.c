#include "action_record_learner.h"
#include "policy_decoding.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int action_record_ppo_surrogate(double current_log, double behavior_log, double advantage,
    double epsilon, ActionPpoSurrogate* result
) {
    if (!result || !isfinite(current_log) || current_log > 0 || !isfinite(behavior_log) || behavior_log > 0 ||
        !isfinite(advantage) || !isfinite(epsilon) || epsilon <= 0 || epsilon > (double).2f) return 0;
    double ratio = exp(current_log - behavior_log);
    if (!isfinite(ratio) || ratio <= 0) return 0;
    double clipped_ratio = fmax(1 - epsilon, fmin(1 + epsilon, ratio));
    ActionPpoSurrogate proposed = {ratio, fmin(ratio * advantage, clipped_ratio * advantage), 0, 0};
    proposed.clipped = (advantage > 0 && ratio > 1 + epsilon) || (advantage < 0 && ratio < 1 - epsilon);
    proposed.log_probability_gradient = proposed.clipped ? 0 : ratio * advantage;
    if (!isfinite(proposed.surrogate) || !isfinite(proposed.log_probability_gradient)) return 0;
    *result = proposed;
    return 1;
}

static int action_log(const GruModel* model, const ActionLearningRecord* row, double* result, float* value) {
    FactorizedActionChoice action = row->choice;
    FactorizedPolicySnapshot policy;
    action.slot0_target_mask = action.slot1_target_mask = 0;
    return gru_model_evaluate_policy_snapshot(model, row->hidden, row->legal,
        action.slot0_has_action && action.slot1_has_action, &policy, value) &&
        policy_choice_log_probability(&policy, row->legal, &action, result) && isfinite(*result);
}

static int measure(const GruModel* actor, const ActionLearningRecord* rows, size_t count,
    const double* advantages, const ActionLearningOptions* options, ActionLearningProgress* progress
) {
    progress->max_abs_log_ratio = 0;
    progress->ratio_min = progress->ratio_max = progress->surrogate_mean = progress->clipped_fraction = 0;
    size_t clipped = 0;
    for (size_t i = 0; i < count; ++i) {
        double current;
        if (!action_log(actor, rows + i, &current, NULL)) return 0;
        current += rows[i].conditional_target_log_probability;
        double difference = fabs(current - rows[i].behavior_log_probability);
        if (!isfinite(difference)) return 0;
        if (difference > progress->max_abs_log_ratio) progress->max_abs_log_ratio = difference;
        if (options->objective == ACTION_PPO_FROZEN_VALUE) {
            ActionPpoSurrogate term;
            if (!action_record_ppo_surrogate(current, rows[i].behavior_log_probability, advantages[i], options->ppo_clip_epsilon, &term)) return 0;
            if (!i || term.ratio < progress->ratio_min) progress->ratio_min = term.ratio;
            if (!i || term.ratio > progress->ratio_max) progress->ratio_max = term.ratio;
            progress->surrogate_mean += term.surrogate / count;
            clipped += term.clipped;
        }
    }
    progress->clipped_fraction = (double)clipped / count;
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
        (options->objective != ACTION_RETURN_DIAGNOSTIC && options->objective != ACTION_PPO_FROZEN_VALUE) ||
        (options->objective == ACTION_PPO_FROZEN_VALUE && (!isfinite(options->ppo_clip_epsilon) ||
            options->ppo_clip_epsilon <= 0 || options->ppo_clip_epsilon > .2f)) ||
        progress->guard_stopped || progress->updates > options->max_updates ||
        progress->cursor != (progress->updates * options->minibatch < count ? progress->updates * options->minibatch : count) ||
        gru_model_action_head_compatibility_reason(reference, behavior) ||
        gru_model_action_head_compatibility_reason(reference, actor)) return 0;
    double advantages[128];
    progress->mean_value = progress->mean_advantage = 0;
    for (size_t i = 0; i < count; ++i) {
        double before;
        float value = 0;
        if (!rows[i].hidden || !isfinite(rows[i].terminal_return) || fabsf(rows[i].terminal_return) > 1 ||
            !isfinite(rows[i].conditional_target_log_probability) || rows[i].conditional_target_log_probability > 0 ||
            !isfinite(rows[i].behavior_log_probability) || rows[i].behavior_log_probability > 0) return 0;
        for (size_t h = 0; h < gru_model_hidden_dim(actor); ++h) if (!isfinite(rows[i].hidden[h])) return 0;
        for (size_t a = 0; a < 28; ++a) if (rows[i].legal[a] > 1) return 0;
        if (!action_log(behavior, rows + i, &before, options->objective == ACTION_PPO_FROZEN_VALUE ? &value : NULL) || !isfinite(value) ||
            fabs(before + rows[i].conditional_target_log_probability - rows[i].behavior_log_probability) > 1e-6) return 0;
        advantages[i] = (double)rows[i].terminal_return - value;
        progress->mean_value += (double)value / count;
        progress->mean_advantage += advantages[i] / count;
    }
    size_t parameters = gru_model_parameter_count(actor), moments = gru_action_head_optimizer_state_count(*optimizer);
    float* saved = malloc((parameters + 2 * moments) * sizeof(float));
    if (!saved) return 0;
    uint64_t step;
    int ok = gru_action_head_optimizer_export_state(*optimizer, saved + parameters, saved + parameters + moments, moments, &step) &&
        step == progress->updates && measure(actor, rows, count, advantages, options, progress) &&
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
        for (size_t i = progress->cursor; ok && i < end; ++i) {
            double weight = advantages[i];
            if (options->objective == ACTION_PPO_FROZEN_VALUE) {
                double current;
                ActionPpoSurrogate term;
                ok = action_log(actor, rows + i, &current, NULL) && action_record_ppo_surrogate(
                    current + rows[i].conditional_target_log_probability, rows[i].behavior_log_probability,
                    advantages[i], options->ppo_clip_epsilon, &term);
                if (ok) weight = term.log_probability_gradient;
            }
            if (ok) ok = gru_action_head_accumulate_hidden(*optimizer, rows[i].hidden, rows[i].legal, &rows[i].choice, (float)weight);
        }
        if (ok) ok = gru_action_head_apply_adam(*optimizer, options->learning_rate, .9f, .999f, 1e-8f, 1);
        ActionLearningProgress proposed = *progress;
        if (ok) ok = !gru_model_action_head_compatibility_reason(reference, actor) && measure(actor, rows, count, advantages, options, &proposed);
        if (!ok || proposed.max_abs_log_ratio > options->max_abs_log_ratio) {
            int restored = gru_model_import_parameters(actor, saved, parameters);
            gru_action_head_optimizer_free(*optimizer); *optimizer = rollback;
            if (ok && restored) { progress->guard_stopped = 1; progress->rejected_abs_log_ratio = proposed.max_abs_log_ratio; }
            else ok = 0;
            break;
        }
        gru_action_head_optimizer_free(rollback);
        *progress = proposed;
        progress->cursor = end;
        ++progress->updates;
    }
    free(saved);
    return ok;
}
