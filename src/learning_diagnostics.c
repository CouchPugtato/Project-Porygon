#include "learning_diagnostics.h"

#include <stdio.h>
#include <string.h>

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
