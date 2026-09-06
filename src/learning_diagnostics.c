#include "learning_diagnostics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

static int evaluate_episodes(
    const GruTrainer* trainer,
    const GruModel* model,
    const Episode* const* episodes,
    size_t episode_count,
    PolicyEvaluationMetrics* total
) {
    size_t i;
    policy_evaluation_init(total);
    for (i = 0; i < episode_count; ++i) {
        PolicyEvaluationMetrics part;
        policy_evaluation_init(&part);
        if (!policy_evaluation_add_episode(model, trainer->bptt_window, episodes[i], &part)) {
            return 0;
        }
        policy_evaluation_merge(total, &part);
    }
    return 1;
}

int learning_diagnostic_run_supervised_overfit(
    GruTrainer* trainer,
    GruModel* model,
    const Episode* const* episodes,
    size_t episode_count,
    size_t epochs,
    SupervisedOverfitResult* result
) {
    double before_action_loss;
    double after_action_loss;
    size_t epoch;
    size_t i;

    if (!trainer || !model || !episodes || episode_count == 0 || epochs == 0 || !result) {
        return 0;
    }
    memset(result, 0, sizeof(*result));
    if (!evaluate_episodes(trainer, model, episodes, episode_count, &result->before)) {
        return 0;
    }

    result->training_completed = 1;
    for (epoch = 0; epoch < epochs && result->training_completed; ++epoch) {
        for (i = 0; i < episode_count; ++i) {
            if (!gru_trainer_supervised_episode(trainer, model, episodes[i])) {
                result->training_completed = 0;
                break;
            }
        }
    }
    if (!evaluate_episodes(trainer, model, episodes, episode_count, &result->after)) {
        return 0;
    }

    before_action_loss = policy_evaluation_action_nll(&result->before);
    after_action_loss = policy_evaluation_action_nll(&result->after);
    if (before_action_loss > 0.0) {
        result->action_loss_reduction = 1.0 - after_action_loss / before_action_loss;
    }
    result->action_loss_reduced = before_action_loss > 0.0 &&
        after_action_loss <= before_action_loss * 0.5;
    result->full_turn_accuracy_reached =
        policy_evaluation_full_turn_accuracy(&result->after) >= 0.8;
    result->action_probability_increased =
        policy_evaluation_action_probability(&result->after) >
        policy_evaluation_action_probability(&result->before);
    result->target_probability_increased = result->before.target_labels > 0 &&
        result->after.target_labels == result->before.target_labels &&
        policy_evaluation_target_probability(&result->after) >
        policy_evaluation_target_probability(&result->before);
    result->value_error_decreased =
        policy_evaluation_value_loss(&result->after) <
        policy_evaluation_value_loss(&result->before);
    result->outputs_finite = result->before.nonfinite_values == 0 &&
        result->after.nonfinite_values == 0;
    result->predictions_legal = result->before.illegal_predictions == 0 &&
        result->after.illegal_predictions == 0;
    result->passed = result->training_completed && result->action_loss_reduced &&
        result->full_turn_accuracy_reached && result->action_probability_increased &&
        result->target_probability_increased && result->value_error_decreased &&
        result->outputs_finite && result->predictions_legal;
    return 1;
}

typedef struct {
    size_t episode_count;
    size_t sample_count;
    size_t nonfinite_count;
    double return_sum;
    double return_square_sum;
    double value_sum;
    double value_square_sum;
    double error_sum;
    double error_square_sum;
    double return_value_product_sum;
} CriticMetricAccumulator;

static void critic_metric_add(
    CriticMetricAccumulator* accumulator,
    double target,
    double value
) {
    double error;
    if (!accumulator) return;
    if (!isfinite(target) || !isfinite(value)) {
        ++accumulator->nonfinite_count;
        return;
    }
    error = target - value;
    accumulator->return_sum += target;
    accumulator->return_square_sum += target * target;
    accumulator->value_sum += value;
    accumulator->value_square_sum += value * value;
    accumulator->error_sum += error;
    accumulator->error_square_sum += error * error;
    accumulator->return_value_product_sum += target * value;
    ++accumulator->sample_count;
}

static void critic_metric_finish(
    const CriticMetricAccumulator* accumulator,
    CriticFitMetrics* metrics
) {
    double count;
    double return_variance;
    double value_variance;
    double error_variance;
    double covariance;
    memset(metrics, 0, sizeof(*metrics));
    metrics->episode_count = accumulator->episode_count;
    metrics->sample_count = accumulator->sample_count;
    metrics->nonfinite_count = accumulator->nonfinite_count;
    if (accumulator->sample_count == 0) return;
    count = (double)accumulator->sample_count;
    metrics->mean_return = accumulator->return_sum / count;
    metrics->mean_value = accumulator->value_sum / count;
    metrics->value_bias = metrics->mean_value - metrics->mean_return;
    metrics->value_loss = 0.5 * accumulator->error_square_sum / count;
    return_variance = accumulator->return_square_sum / count -
        metrics->mean_return * metrics->mean_return;
    value_variance = accumulator->value_square_sum / count -
        metrics->mean_value * metrics->mean_value;
    error_variance = accumulator->error_square_sum / count -
        (accumulator->error_sum / count) * (accumulator->error_sum / count);
    if (return_variance < 0.0) return_variance = 0.0;
    if (value_variance < 0.0) value_variance = 0.0;
    if (error_variance < 0.0) error_variance = 0.0;
    if (return_variance > 1.0e-12) {
        metrics->explained_variance = 1.0 - error_variance / return_variance;
    }
    if (return_variance > 1.0e-12 && value_variance > 1.0e-12) {
        covariance = accumulator->return_value_product_sum / count -
            metrics->mean_return * metrics->mean_value;
        metrics->return_value_correlation = covariance /
            sqrt(return_variance * value_variance);
    }
}

static int evaluate_critic(
    const GruTrainer* trainer,
    const GruModel* model,
    const Episode* const* episodes,
    size_t episode_count,
    CriticFitEvaluation* evaluation
) {
    CriticMetricAccumulator overall = {0};
    CriticMetricAccumulator wins = {0};
    CriticMetricAccumulator losses = {0};
    size_t hidden_dim;
    size_t i;

    if (!trainer || !model || !episodes || !evaluation) return 0;
    hidden_dim = gru_model_hidden_dim(model);
    for (i = 0; i < episode_count; ++i) {
        const Episode* episode = episodes[i];
        CriticMetricAccumulator* outcome = NULL;
        float* returns;
        float* hidden;
        float* next_hidden;
        float running_return = 0.0f;
        size_t t;
        if (!episode || episode->count == 0) continue;
        if (episode->rewards[episode->count - 1u] > 0.0f) outcome = &wins;
        else if (episode->rewards[episode->count - 1u] < 0.0f) outcome = &losses;
        ++overall.episode_count;
        if (outcome) ++outcome->episode_count;
        returns = (float*)calloc(episode->count, sizeof(float));
        hidden = (float*)calloc(hidden_dim, sizeof(float));
        next_hidden = (float*)malloc(hidden_dim * sizeof(float));
        if (!returns || !hidden || !next_hidden) {
            free(returns);
            free(hidden);
            free(next_hidden);
            return 0;
        }
        for (t = episode->count; t > 0; --t) {
            size_t index = t - 1u;
            if (episode->dones[index]) running_return = 0.0f;
            running_return = episode->rewards[index] + trainer->gamma * running_return;
            returns[index] = running_return;
        }
        gru_model_zero_state(model, hidden);
        for (t = 0; t < episode->count; ++t) {
            float value;
            gru_model_forward_step(
                model,
                episode->observations + t * episode->obs_dim,
                hidden,
                next_hidden,
                NULL,
                &value);
            memcpy(hidden, next_hidden, hidden_dim * sizeof(float));
            if (episode->actions[t] < 0 && episode->actions2[t] < 0) continue;
            critic_metric_add(&overall, returns[t], value);
            if (outcome) critic_metric_add(outcome, returns[t], value);
        }
        free(returns);
        free(hidden);
        free(next_hidden);
    }
    critic_metric_finish(&overall, &evaluation->overall);
    critic_metric_finish(&wins, &evaluation->wins);
    critic_metric_finish(&losses, &evaluation->losses);
    return 1;
}

static unsigned int critic_shuffle_next(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static double critic_wall_seconds(void) {
#ifdef _WIN32
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0;
#endif
}

static void critic_shuffle(size_t* order, size_t count, unsigned int* state) {
    size_t i;
    for (i = count; i > 1u; --i) {
        size_t other = (size_t)(critic_shuffle_next(state) % (unsigned int)i);
        size_t temporary = order[i - 1u];
        order[i - 1u] = order[other];
        order[other] = temporary;
    }
}

static int train_critic_branch(
    const char* name,
    GruTrainer* trainer,
    GruModel* model,
    const Episode* const* episodes,
    size_t episode_count,
    size_t epochs,
    size_t minibatch_episodes,
    unsigned int shuffle_seed,
    int update_recurrent
) {
    const Episode** minibatch;
    size_t* order;
    unsigned int shuffle_state = shuffle_seed;
    double started_at = critic_wall_seconds();
    size_t epoch;
    size_t i;

    order = (size_t*)malloc(episode_count * sizeof(size_t));
    minibatch = (const Episode**)malloc(minibatch_episodes * sizeof(*minibatch));
    if (!order || !minibatch) {
        free(order);
        free(minibatch);
        return 0;
    }
    for (i = 0; i < episode_count; ++i) order[i] = i;
    for (epoch = 0; epoch < epochs; ++epoch) {
        critic_shuffle(order, episode_count, &shuffle_state);
        for (i = 0; i < episode_count; i += minibatch_episodes) {
            size_t batch_count = episode_count - i;
            size_t j;
            if (batch_count > minibatch_episodes) batch_count = minibatch_episodes;
            for (j = 0; j < batch_count; ++j) minibatch[j] = episodes[order[i + j]];
            if (!gru_trainer_critic_minibatch(
                    trainer, model, minibatch, batch_count, update_recurrent)) {
                free(order);
                free(minibatch);
                return 0;
            }
        }
        {
            double elapsed = critic_wall_seconds() - started_at;
            double eta = elapsed / (double)(epoch + 1u) * (double)(epochs - epoch - 1u);
            printf("[critic-fit] mode=%s epoch=%zu/%zu elapsed=%.1fs eta=%.1fs\n",
                name, epoch + 1u, epochs, elapsed, eta);
        }
    }
    free(order);
    free(minibatch);
    return 1;
}

static int critic_metrics_generalize(
    const CriticFitMetrics* before,
    const CriticFitMetrics* after
) {
    return after->sample_count == before->sample_count &&
        after->sample_count > 0 &&
        after->nonfinite_count == 0 &&
        after->explained_variance >= 0.05 &&
        after->explained_variance >= before->explained_variance + 0.02 &&
        after->return_value_correlation >= 0.20;
}

static int critic_subgroup_is_consistent(const CriticFitMetrics* metrics) {
    return metrics->sample_count < 50u || metrics->return_value_correlation >= 0.0;
}

void learning_diagnostic_assess_critic_fit(CriticFitResult* result) {
    if (!result) return;
    result->head_generalizes = result->head_training_completed &&
        critic_metrics_generalize(
            &result->before_holdout.overall,
            &result->head_after_holdout.overall);
    result->recurrent_aggregate_generalizes = result->recurrent_training_completed &&
        critic_metrics_generalize(
            &result->before_holdout.overall,
            &result->recurrent_after_holdout.overall);
    result->recurrent_policy_drift_acceptable =
        fabs(result->recurrent_policy_probability_delta) <= 0.01;
    result->recurrent_outcome_consistent =
        critic_subgroup_is_consistent(&result->recurrent_after_holdout.wins) &&
        critic_subgroup_is_consistent(&result->recurrent_after_holdout.losses);
    result->recurrent_generalization_gap =
        result->recurrent_after_train.overall.explained_variance -
        result->recurrent_after_holdout.overall.explained_variance;
    result->recurrent_overfit = result->recurrent_generalization_gap > 0.25;
    result->recurrent_generalizes =
        result->recurrent_aggregate_generalizes &&
        result->recurrent_policy_drift_acceptable &&
        result->recurrent_outcome_consistent &&
        !result->recurrent_overfit;
    result->critic_learnable = result->head_generalizes || result->recurrent_generalizes;
}

int learning_diagnostic_run_critic_fit(
    GruTrainer* head_trainer,
    GruModel* head_model,
    GruTrainer* recurrent_trainer,
    GruModel* recurrent_model,
    const Episode* const* train_episodes,
    size_t train_count,
    const Episode* const* holdout_episodes,
    size_t holdout_count,
    size_t epochs,
    size_t minibatch_episodes,
    unsigned int shuffle_seed,
    CriticFitResult* result
) {
    PolicyEvaluationMetrics policy_before;
    PolicyEvaluationMetrics head_policy_after;
    PolicyEvaluationMetrics recurrent_policy_after;
    double policy_probability_before;

    if (!head_trainer || !head_model || !recurrent_trainer || !recurrent_model ||
            !train_episodes || train_count == 0 || !holdout_episodes || holdout_count == 0 ||
            epochs == 0 || minibatch_episodes == 0 || !result) {
        return 0;
    }
    memset(result, 0, sizeof(*result));
    if (!evaluate_critic(head_trainer, head_model, train_episodes, train_count, &result->before_train) ||
            !evaluate_critic(head_trainer, head_model, holdout_episodes, holdout_count, &result->before_holdout) ||
            !evaluate_episodes(head_trainer, head_model, holdout_episodes, holdout_count, &policy_before)) {
        return 0;
    }
    policy_probability_before = policy_evaluation_action_probability(&policy_before);

    result->head_training_completed = train_critic_branch(
        "head", head_trainer, head_model, train_episodes, train_count,
        epochs, minibatch_episodes, shuffle_seed, 0);
    if (result->head_training_completed &&
            (!evaluate_critic(head_trainer, head_model, train_episodes, train_count, &result->head_after_train) ||
             !evaluate_critic(head_trainer, head_model, holdout_episodes, holdout_count, &result->head_after_holdout) ||
             !evaluate_episodes(head_trainer, head_model, holdout_episodes, holdout_count, &head_policy_after))) {
        return 0;
    }
    if (result->head_training_completed) {
        result->head_policy_probability_delta =
            policy_evaluation_action_probability(&head_policy_after) - policy_probability_before;
        result->head_policy_unchanged = fabs(result->head_policy_probability_delta) < 1.0e-12 &&
            policy_evaluation_full_turn_accuracy(&head_policy_after) ==
                policy_evaluation_full_turn_accuracy(&policy_before);
    }

    result->recurrent_training_completed = train_critic_branch(
        "recurrent", recurrent_trainer, recurrent_model, train_episodes, train_count,
        epochs, minibatch_episodes, shuffle_seed, 1);
    if (result->recurrent_training_completed &&
            (!evaluate_critic(recurrent_trainer, recurrent_model, train_episodes, train_count, &result->recurrent_after_train) ||
             !evaluate_critic(recurrent_trainer, recurrent_model, holdout_episodes, holdout_count, &result->recurrent_after_holdout) ||
             !evaluate_episodes(recurrent_trainer, recurrent_model, holdout_episodes, holdout_count, &recurrent_policy_after))) {
        return 0;
    }
    if (result->recurrent_training_completed) {
        result->recurrent_policy_probability_delta =
            policy_evaluation_action_probability(&recurrent_policy_after) - policy_probability_before;
    }
    learning_diagnostic_assess_critic_fit(result);
    return 1;
}

static void write_json_string(FILE* out, const char* text) {
    const unsigned char* p = (const unsigned char*)(text ? text : "");
    fputc('"', out);
    while (*p) {
        switch (*p) {
            case '"': fputs("\\\"", out); break;
            case '\\': fputs("\\\\", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (*p < 0x20u) fprintf(out, "\\u%04x", (unsigned int)*p);
                else fputc((int)*p, out);
                break;
        }
        ++p;
    }
    fputc('"', out);
}

static void write_metrics(FILE* out, const PolicyEvaluationMetrics* metrics, const char* indent) {
    fprintf(out, "%s{\n", indent);
    fprintf(out, "%s  \"action_nll\": %.9g,\n", indent, policy_evaluation_action_nll(metrics));
    fprintf(out, "%s  \"target_nll\": %.9g,\n", indent, policy_evaluation_target_nll(metrics));
    fprintf(out, "%s  \"full_turn_nll\": %.9g,\n", indent, policy_evaluation_full_turn_nll(metrics));
    fprintf(out, "%s  \"value_loss\": %.9g,\n", indent, policy_evaluation_value_loss(metrics));
    fprintf(out, "%s  \"full_turn_accuracy\": %.9g,\n", indent, policy_evaluation_full_turn_accuracy(metrics));
    fprintf(out, "%s  \"joint_pair_accuracy\": %.9g,\n", indent, policy_evaluation_joint_pair_accuracy(metrics));
    fprintf(out, "%s  \"action_probability\": %.9g,\n", indent, policy_evaluation_action_probability(metrics));
    fprintf(out, "%s  \"target_probability\": %.9g,\n", indent, policy_evaluation_target_probability(metrics));
    fprintf(out, "%s  \"decision_turns\": %zu,\n", indent, metrics->decision_turns);
    fprintf(out, "%s  \"action_labels\": %zu,\n", indent, metrics->action_labels);
    fprintf(out, "%s  \"target_labels\": %zu,\n", indent, metrics->target_labels);
    fprintf(out, "%s  \"illegal_predictions\": %zu,\n", indent, metrics->illegal_predictions);
    fprintf(out, "%s  \"nonfinite_values\": %zu\n", indent, metrics->nonfinite_values);
    fprintf(out, "%s}", indent);
}

static void write_critic_metrics(FILE* out, const CriticFitMetrics* metrics, const char* indent) {
    fprintf(out, "%s{\n", indent);
    fprintf(out, "%s  \"episodes\": %zu,\n", indent, metrics->episode_count);
    fprintf(out, "%s  \"samples\": %zu,\n", indent, metrics->sample_count);
    fprintf(out, "%s  \"nonfinite_values\": %zu,\n", indent, metrics->nonfinite_count);
    fprintf(out, "%s  \"value_loss\": %.9g,\n", indent, metrics->value_loss);
    fprintf(out, "%s  \"mean_return\": %.9g,\n", indent, metrics->mean_return);
    fprintf(out, "%s  \"mean_value\": %.9g,\n", indent, metrics->mean_value);
    fprintf(out, "%s  \"value_bias\": %.9g,\n", indent, metrics->value_bias);
    fprintf(out, "%s  \"explained_variance\": %.9g,\n", indent, metrics->explained_variance);
    fprintf(out, "%s  \"return_value_correlation\": %.9g\n", indent,
        metrics->return_value_correlation);
    fprintf(out, "%s}", indent);
}

static void write_critic_evaluation(
    FILE* out,
    const CriticFitEvaluation* evaluation,
    const char* indent
) {
    fprintf(out, "%s{\n%s  \"overall\": ", indent, indent);
    write_critic_metrics(out, &evaluation->overall, "    ");
    fprintf(out, ",\n%s  \"wins\": ", indent);
    write_critic_metrics(out, &evaluation->wins, "    ");
    fprintf(out, ",\n%s  \"losses\": ", indent);
    write_critic_metrics(out, &evaluation->losses, "    ");
    fprintf(out, "\n%s}", indent);
}

static void write_failure_reason(FILE* out, int* first, const char* reason) {
    if (!*first) fputs(", ", out);
    write_json_string(out, reason);
    *first = 0;
}

static void write_failure_reasons(FILE* out, const SupervisedOverfitResult* result) {
    int first = 1;
    fputc('[', out);
    if (!result->training_completed) write_failure_reason(out, &first, "training_update_failed");
    if (!result->action_loss_reduced) write_failure_reason(out, &first, "action_loss_reduction_below_50_percent");
    if (!result->full_turn_accuracy_reached) write_failure_reason(out, &first, "full_turn_accuracy_below_80_percent");
    if (!result->action_probability_increased) write_failure_reason(out, &first, "selected_action_probability_did_not_increase");
    if (!result->target_probability_increased) {
        write_failure_reason(out, &first,
            result->before.target_labels == 0 ? "no_target_labels" : "selected_target_probability_did_not_increase");
    }
    if (!result->value_error_decreased) write_failure_reason(out, &first, "value_error_did_not_decrease");
    if (!result->outputs_finite) write_failure_reason(out, &first, "nonfinite_model_output");
    if (!result->predictions_legal) write_failure_reason(out, &first, "illegal_prediction");
    fputc(']', out);
}

int learning_diagnostic_write_supervised_report(
    const char* report_path,
    const char* source_path,
    const char* first_battle_id,
    const char* second_battle_id,
    unsigned int seed,
    size_t epochs,
    const GruTrainer* trainer,
    const SupervisedOverfitResult* result
) {
    FILE* out;
    if (!report_path || !*report_path || !trainer || !result) return 0;
    out = fopen(report_path, "w");
    if (!out) return 0;

    fputs("{\n  \"diagnostic\": \"supervised_overfit\",\n", out);
    fprintf(out, "  \"metrics_version\": %d,\n", POLICY_EVALUATION_METRICS_VERSION);
    fprintf(out, "  \"passed\": %s,\n", result->passed ? "true" : "false");
    fputs("  \"source_replay\": ", out);
    write_json_string(out, source_path);
    fputs(",\n  \"battle_ids\": [", out);
    write_json_string(out, first_battle_id);
    fputs(", ", out);
    write_json_string(out, second_battle_id);
    fputs("],\n", out);
    fprintf(out, "  \"session_count\": 2,\n  \"seed\": %u,\n  \"epochs\": %zu,\n",
        seed, epochs);
    fputs("  \"optimizer\": ", out);
    write_json_string(out, gru_supervised_optimizer_name(trainer->supervised_optimizer));
    fprintf(out,
        ",\n  \"learning_rate\": %.9g,\n  \"bptt_window\": %zu,\n  \"gradient_clip\": %.9g,\n"
        "  \"adam_beta1\": %.9g,\n  \"adam_beta2\": %.9g,\n  \"adam_epsilon\": %.9g,\n",
        trainer->learning_rate,
        trainer->bptt_window,
        trainer->gradient_clip,
        trainer->adam_beta1,
        trainer->adam_beta2,
        trainer->adam_epsilon);
    fputs("  \"before\": ", out);
    write_metrics(out, &result->before, "  ");
    fputs(",\n  \"after\": ", out);
    write_metrics(out, &result->after, "  ");
    fprintf(out,
        ",\n  \"checks\": {\n"
        "    \"action_loss_reduction\": %.9g,\n"
        "    \"action_loss_reduced_50_percent\": %s,\n"
        "    \"full_turn_accuracy_at_least_80_percent\": %s,\n"
        "    \"selected_action_probability_increased\": %s,\n"
        "    \"selected_target_probability_increased\": %s,\n"
        "    \"value_error_decreased\": %s,\n"
        "    \"outputs_finite\": %s,\n"
        "    \"predictions_legal\": %s\n"
        "  },\n  \"failure_reasons\": ",
        result->action_loss_reduction,
        result->action_loss_reduced ? "true" : "false",
        result->full_turn_accuracy_reached ? "true" : "false",
        result->action_probability_increased ? "true" : "false",
        result->target_probability_increased ? "true" : "false",
        result->value_error_decreased ? "true" : "false",
        result->outputs_finite ? "true" : "false",
        result->predictions_legal ? "true" : "false");
    write_failure_reasons(out, result);
    fputs("\n}\n", out);
    if (fclose(out) != 0) return 0;
    return 1;
}

int learning_diagnostic_write_critic_report(
    const char* report_path,
    const char* source_path,
    const char* checkpoint_path,
    unsigned int validation_seed,
    unsigned int shuffle_seed,
    size_t epochs,
    size_t minibatch_episodes,
    const GruTrainer* trainer,
    const CriticFitResult* result
) {
    const char* recommendation;
    FILE* out;
    if (!report_path || !*report_path || !trainer || !result) return 0;
    if (result->head_generalizes) recommendation = "value_head_warmup";
    else if (result->recurrent_generalizes) recommendation = "recurrent_critic_fit";
    else if (result->recurrent_aggregate_generalizes) recommendation =
        "recurrent_critic_overfits_or_changes_policy";
    else recommendation = "critic_signal_not_generalizing";
    out = fopen(report_path, "w");
    if (!out) return 0;

    fputs("{\n  \"diagnostic\": \"critic_fit\",\n  \"metrics_version\": 2,\n", out);
    fputs("  \"source_episode_batch\": ", out);
    write_json_string(out, source_path);
    fputs(",\n  \"checkpoint\": ", out);
    write_json_string(out, checkpoint_path);
    fprintf(out,
        ",\n  \"validation_seed\": %u,\n"
        "  \"shuffle_seed\": %u,\n"
        "  \"epochs\": %zu,\n"
        "  \"minibatch_episodes\": %zu,\n"
        "  \"learning_rate\": %.9g,\n"
        "  \"gamma\": %.9g,\n"
        "  \"bptt_window\": %zu,\n"
        "  \"optimizer\": \"adam\",\n"
        "  \"return_target\": \"discounted_monte_carlo\",\n"
        "  \"published_checkpoint\": false,\n",
        validation_seed,
        shuffle_seed,
        epochs,
        minibatch_episodes,
        trainer->learning_rate,
        trainer->gamma,
        trainer->bptt_window);
    fputs("  \"before\": {\n    \"train\": ", out);
    write_critic_evaluation(out, &result->before_train, "    ");
    fputs(",\n    \"holdout\": ", out);
    write_critic_evaluation(out, &result->before_holdout, "    ");
    fputs("\n  },\n  \"value_head_only\": {\n    \"training_completed\": ", out);
    fputs(result->head_training_completed ? "true" : "false", out);
    fputs(",\n    \"policy_outputs_unchanged\": ", out);
    fputs(result->head_policy_unchanged ? "true" : "false", out);
    fprintf(out, ",\n    \"policy_action_probability_delta\": %.9g,\n    \"train\": ",
        result->head_policy_probability_delta);
    write_critic_evaluation(out, &result->head_after_train, "    ");
    fputs(",\n    \"holdout\": ", out);
    write_critic_evaluation(out, &result->head_after_holdout, "    ");
    fputs("\n  },\n  \"recurrent_critic\": {\n    \"training_completed\": ", out);
    fputs(result->recurrent_training_completed ? "true" : "false", out);
    fputs(",\n    \"policy_head_parameters_frozen\": true,\n", out);
    fprintf(out, "    \"policy_action_probability_delta\": %.9g,\n    \"train\": ",
        result->recurrent_policy_probability_delta);
    write_critic_evaluation(out, &result->recurrent_after_train, "    ");
    fputs(",\n    \"holdout\": ", out);
    write_critic_evaluation(out, &result->recurrent_after_holdout, "    ");
    fputs("\n  },\n  \"assessment\": {\n", out);
    fprintf(out,
        "    \"head_generalizes\": %s,\n"
        "    \"recurrent_aggregate_generalizes\": %s,\n"
        "    \"recurrent_policy_drift_acceptable\": %s,\n"
        "    \"recurrent_outcome_consistent\": %s,\n"
        "    \"recurrent_overfit\": %s,\n"
        "    \"recurrent_generalization_gap\": %.9g,\n"
        "    \"recurrent_generalizes\": %s,\n"
        "    \"critic_learnable\": %s,\n",
        result->head_generalizes ? "true" : "false",
        result->recurrent_aggregate_generalizes ? "true" : "false",
        result->recurrent_policy_drift_acceptable ? "true" : "false",
        result->recurrent_outcome_consistent ? "true" : "false",
        result->recurrent_overfit ? "true" : "false",
        result->recurrent_generalization_gap,
        result->recurrent_generalizes ? "true" : "false",
        result->critic_learnable ? "true" : "false");
    fputs("    \"aggregate_rule\": \"holdout explained variance >= 0.05, improvement >= 0.02, correlation >= 0.20\",\n", out);
    fputs("    \"recurrent_safety_rule\": \"absolute policy-probability drift <= 0.01, train-holdout explained-variance gap <= 0.25, and non-negative win/loss correlation when a subgroup has at least 50 samples\",\n", out);
    fputs("    \"recommended_path\": ", out);
    write_json_string(out, recommendation);
    fputs("\n  }\n}\n", out);
    return fclose(out) == 0;
}
