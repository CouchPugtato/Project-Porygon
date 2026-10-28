#include "action_record_learner.h"
#include "action_optimizer_state.h"
#include "checkpoint.h"
#include "observation.h"
#include "policy_decoding.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int check(int condition, const char* message) {
    if (!condition) fprintf(stderr, "test failed: %s\n", message);
    return condition;
}

static double record_log(const GruModel* actor, const ActionLearningRecord* row) {
    FactorizedPolicySnapshot policy;
    FactorizedActionChoice choice = row->choice;
    choice.slot0_target_mask = choice.slot1_target_mask = 0;
    double result = NAN;
    if (!gru_model_evaluate_policy_snapshot(actor, row->hidden, row->legal,
            choice.slot0_has_action && choice.slot1_has_action, &policy, NULL) ||
            !policy_choice_log_probability(&policy, row->legal, &choice, &result)) return NAN;
    return result + row->conditional_target_log_probability;
}

static GruModel* copy_model(const GruModel* model) {
    GruModel* copy = gru_model_create(gru_model_input_dim(model), gru_model_hidden_dim(model), gru_model_num_actions(model));
    size_t count = gru_model_parameter_count(model);
    float* parameters = malloc(count * sizeof(float));
    if (!copy || !parameters || !gru_model_export_parameters(model, parameters, count) ||
            !gru_model_import_parameters(copy, parameters, count)) { gru_model_destroy(copy); copy = NULL; }
    free(parameters); return copy;
}

static int write_pack(const ActionLearningRecord* rows, size_t cursor, size_t updates, size_t invocation, int ppo) {
    FILE* file = fopen("test_action_learner_input.tmp", "wb");
    if (!file) return 0;
    uint32_t header[12] = {0x01020304, 8, 6, (uint32_t)cursor, (uint32_t)updates, 3, (uint32_t)invocation, 2};
    float learning_rate = .00001f, limit = .02f;
    memcpy(header + 8, &learning_rate, 4); memcpy(header + 9, &limit, 4);
    header[10] = ppo ? ACTION_PPO_FROZEN_VALUE : 0;
    float epsilon = .01f; memcpy(header + 11, &epsilon, 4);
    char pin[64]; memset(pin, 'a', 64);
    size_t words = ppo ? 12 : 10;
    int ok = fwrite(ppo ? "PGACTREC2" : "PGACTREC1", 1, 9, file) == 9 && fwrite(header, 4, words, file) == words && fwrite(pin, 1, 64, file) == 64;
    for (size_t i = 0; ok && i < 6; ++i) {
        const FactorizedActionChoice* c = &rows[i].choice;
        unsigned char choice[14] = {c->slot0_has_action, c->slot0_kind, c->slot0_move_index, c->slot0_switch_index,
            c->slot0_use_tera, c->slot0_target_index, c->slot0_target_mask, c->slot1_has_action, c->slot1_kind,
            c->slot1_move_index, c->slot1_switch_index, c->slot1_use_tera, c->slot1_target_index, c->slot1_target_mask};
        ok = fwrite(rows[i].hidden, 4, 8, file) == 8 && fwrite(rows[i].legal, 1, 28, file) == 28 &&
            fwrite(choice, 1, 14, file) == 14 && fwrite(&rows[i].terminal_return, 4, 1, file) == 1 &&
            fwrite(&rows[i].behavior_log_probability, 8, 1, file) == 1 && fwrite(&rows[i].conditional_target_log_probability, 8, 1, file) == 1;
    }
    if (fclose(file)) ok = 0;
    return ok;
}

static int test_action_record_learner_mode(int ppo) {
    srand(32001);
    GruModel* behavior = gru_model_create(observation_flat_size(), 8, OBS_NUM_ACTIONS);
    GruModel *whole = NULL, *partial = NULL, *loaded = NULL;
    GruActionHeadOptimizer *whole_opt = NULL, *partial_opt = NULL, *loaded_opt = NULL;
    float hidden[6][8] = {{0}};
    ActionLearningRecord rows[6] = {{0}};
    int ok = behavior != NULL;
    if (!ok) return 0;
    for (size_t i = 0; i < 6; ++i) {
        hidden[i][0] = .2f; hidden[i][1] = -.3f; hidden[i][i + 2] = .1f;
        rows[i].hidden = hidden[i]; memset(rows[i].legal, 1, sizeof(rows[i].legal));
        factorized_action_choice_from_flat_actions(&rows[i].choice, i % 3 == 2 ? -1 : i % 3 == 1 ? 8 : 0,
            i % 3 == 1 ? -1 : i % 3 == 2 ? 18 : 15);
        rows[i].terminal_return = i % 2 ? -1 : 1;
        if (rows[i].choice.slot0_has_action && rows[i].choice.slot0_kind == FACTORIZED_ACTION_MOVE) {
            rows[i].choice.slot0_target_mask = 12; rows[i].choice.slot0_target_index = 2;
        }
        if (rows[i].choice.slot1_has_action && rows[i].choice.slot1_kind == FACTORIZED_ACTION_MOVE) {
            rows[i].choice.slot1_target_mask = 12; rows[i].choice.slot1_target_index = 2;
        }
        rows[i].conditional_target_log_probability = rows[i].choice.slot0_target_mask || rows[i].choice.slot1_target_mask ? -.7 : 0;
        rows[i].behavior_log_probability = record_log(behavior, rows + i);
    }
    ActionLearningOptions options = {1, 1, 1, .0001f, .05f};
    options.objective = ppo ? ACTION_PPO_FROZEN_VALUE : ACTION_RETURN_DIAGNOSTIC; options.ppo_clip_epsilon = .01f;
    for (size_t i = 0; ok && i < 6; ++i) {
        GruModel* actor = copy_model(behavior);
        GruActionHeadOptimizer* optimizer = gru_action_head_optimizer_create(actor);
        ActionLearningProgress progress = {0};
        if (ppo) {
            ActionPpoSurrogate initial;
            ok &= check(action_record_ppo_surrogate(record_log(actor, rows + i), rows[i].behavior_log_probability, 1, .01, &initial) &&
                initial.ratio == 1, "Joint/single-slot PPO initialization replays full-turn likelihoods at ratio one");
        }
        ok &= actor && optimizer && action_record_learn(behavior, behavior, actor, &optimizer, rows + i, 1, &options, &progress);
        double after = record_log(actor, rows + i);
        ok &= check(progress.cursor == 1 && progress.updates == 1 && !progress.guard_stopped &&
            (rows[i].terminal_return > 0 ? after > rows[i].behavior_log_probability : after < rows[i].behavior_log_probability),
            "Diagnostic joint/single-slot updates follow signed terminal returns under fixed target likelihoods");
        ok &= check(!gru_model_action_head_compatibility_reason(behavior, actor), "Diagnostic learner preserves every frozen parameter region");
        gru_action_head_optimizer_free(optimizer); gru_model_destroy(actor);
    }
    whole = copy_model(behavior); partial = copy_model(behavior);
    whole_opt = gru_action_head_optimizer_create(whole); partial_opt = gru_action_head_optimizer_create(partial);
    ActionLearningProgress full = {0}, paused = {0};
    options = (ActionLearningOptions){2, 3, 3, .00001f, .02f};
    options.objective = ppo ? ACTION_PPO_FROZEN_VALUE : ACTION_RETURN_DIAGNOSTIC; options.ppo_clip_epsilon = .01f;
    ok &= whole && partial && whole_opt && partial_opt &&
        action_record_learn(behavior, behavior, whole, &whole_opt, rows, 6, &options, &full);
    options.invocation_updates = 1;
    ok &= action_record_learn(behavior, behavior, partial, &partial_opt, rows, 6, &options, &paused);
    char pin[65]; memset(pin, 'a', 64); pin[64] = 0;
    TrainerCheckpointState state = {0};
    ok &= check(paused.cursor == 2 && paused.updates == 1 &&
        checkpoint_save("test_action_learner_actor.tmp", partial, &state) &&
        action_optimizer_state_save("test_action_learner_opt.tmp", partial_opt, behavior, pin, NULL), "Diagnostic checkpoint pauses at an exact record boundary");
    loaded = checkpoint_load("test_action_learner_actor.tmp", NULL);
    loaded_opt = action_optimizer_state_load("test_action_learner_opt.tmp", loaded, behavior, pin, NULL);
    options.invocation_updates = 2;
    ok &= loaded && loaded_opt && action_record_learn(behavior, behavior, loaded, &loaded_opt, rows, 6, &options, &paused);
    size_t count = gru_model_parameter_count(behavior), moments = gru_action_head_optimizer_state_count(whole_opt);
    float *a = malloc((count + 2 * moments) * sizeof(float)), *b = malloc((count + 2 * moments) * sizeof(float));
    uint64_t step_a = 0, step_b = 0;
    if (!a || !b) ok = 0;
    if (ok) {
        ok &= gru_model_export_parameters(whole, a, count) && gru_model_export_parameters(loaded, b, count) &&
            gru_action_head_optimizer_export_state(whole_opt, a + count, a + count + moments, moments, &step_a) &&
            gru_action_head_optimizer_export_state(loaded_opt, b + count, b + count + moments, moments, &step_b);
        ok &= check(!memcmp(a, b, (count + 2 * moments) * sizeof(float)) && step_a == 3 && step_b == 3 &&
            paused.cursor == full.cursor && paused.updates == full.updates && paused.max_abs_log_ratio == full.max_abs_log_ratio,
            "Saved actor/optimizer plus record cursor reproduce uninterrupted diagnostic updates exactly");
        ok &= check(!gru_model_action_head_compatibility_reason(behavior, loaded), "Resumed diagnostic leaves representation, targets and value frozen");
        ok &= check(paused.ratio_min == full.ratio_min && paused.ratio_max == full.ratio_max &&
            paused.mean_advantage == full.mean_advantage && paused.mean_value == full.mean_value &&
            paused.surrogate_mean == full.surrogate_mean && paused.clipped_fraction == full.clipped_fraction,
            "PPO baseline, ratio, surrogate and clipping metrics match exactly after resume");
    }
    free(a); free(b);
    {
        GruModel* actor = copy_model(behavior);
        GruActionHeadOptimizer* optimizer = gru_action_head_optimizer_create(actor);
        ActionLearningProgress progress = {0};
        options = (ActionLearningOptions){1, 1, 1, .0001f, 1e-10f};
        options.objective = ppo ? ACTION_PPO_FROZEN_VALUE : ACTION_RETURN_DIAGNOSTIC; options.ppo_clip_epsilon = .01f;
        ok &= action_record_learn(behavior, behavior, actor, &optimizer, rows, 6, &options, &progress);
        size_t n = gru_model_parameter_count(actor);
        float *before = malloc(n * sizeof(float)), *after = malloc(n * sizeof(float));
        ok &= before && after && gru_model_export_parameters(behavior, before, n) && gru_model_export_parameters(actor, after, n);
        uint64_t step = 1;
        float* state_moments = malloc(2 * moments * sizeof(float));
        ok &= state_moments && gru_action_head_optimizer_export_state(optimizer, state_moments, state_moments + moments, moments, &step);
        ok &= check(progress.guard_stopped && progress.rejected_abs_log_ratio > options.max_abs_log_ratio &&
            !progress.cursor && !progress.updates && step == 0 && !memcmp(before, after, n * sizeof(float)),
            "Oversized diagnostic update rolls back parameters, moments and cursor before stopping");
        free(state_moments); free(before); free(after);
        gru_action_head_optimizer_free(optimizer); gru_model_destroy(actor);
    }
    {
        ActionLearningRecord invalid[6]; memcpy(invalid, rows, sizeof(rows));
        invalid[5].behavior_log_probability += .01;
        GruModel* actor = copy_model(behavior);
        GruActionHeadOptimizer* optimizer = gru_action_head_optimizer_create(actor);
        ActionLearningProgress progress = {0};
        options = (ActionLearningOptions){1, 1, 1, .00001f, .02f};
        options.objective = ppo ? ACTION_PPO_FROZEN_VALUE : ACTION_RETURN_DIAGNOSTIC; options.ppo_clip_epsilon = .01f;
        ok &= check(!action_record_learn(behavior, behavior, actor, &optimizer, invalid, 6, &options, &progress),
            "Every behavior likelihood is checked before a diagnostic update, including records beyond the first minibatch");
        gru_action_head_optimizer_free(optimizer); gru_model_destroy(actor);
    }
    if (ok) {
        ok &= checkpoint_save("test_action_learner_behavior.tmp", behavior, &state) && write_pack(rows, 0, 0, 1, ppo);
        char* args[] = {"test", "--diagnose-action-records", "test_action_learner_input.tmp", "test_action_learner_behavior.tmp",
            "test_action_learner_behavior.tmp", "-", "-", "test_action_learner_cli_partial"};
        ok &= check(action_record_learner_cli(8, args) == 0, "Packed diagnostic CLI runs only the requested bounded update");
        ok &= write_pack(rows, 2, 1, 2, ppo);
        args[5] = "test_action_learner_cli_partial.chk"; args[6] = "test_action_learner_cli_partial.opt";
        args[7] = "test_action_learner_cli_resumed";
        ok &= check(action_record_learner_cli(8, args) == 0, "Packed diagnostic CLI restores the actor, optimizer and record cursor");
        GruModel* cli_actor = checkpoint_load("test_action_learner_cli_resumed.chk", NULL);
        float *expected = malloc(count * sizeof(float)), *actual = malloc(count * sizeof(float));
        ok &= cli_actor && expected && actual && gru_model_export_parameters(whole, expected, count) &&
            gru_model_export_parameters(cli_actor, actual, count) && !memcmp(expected, actual, count * sizeof(float));
        ok &= check(ok, "Packed native CLI resume matches uninterrupted library updates byte-for-byte");
        free(expected); free(actual); gru_model_destroy(cli_actor);
    }
    remove("test_action_learner_actor.tmp"); remove("test_action_learner_opt.tmp"); remove("test_action_learner_opt.tmp.tmp");
    remove("test_action_learner_input.tmp"); remove("test_action_learner_behavior.tmp");
    remove("test_action_learner_cli_partial.chk"); remove("test_action_learner_cli_partial.opt");
    remove("test_action_learner_cli_resumed.chk"); remove("test_action_learner_cli_resumed.opt");
    gru_action_head_optimizer_free(whole_opt); gru_action_head_optimizer_free(partial_opt); gru_action_head_optimizer_free(loaded_opt);
    gru_model_destroy(behavior); gru_model_destroy(whole); gru_model_destroy(partial); gru_model_destroy(loaded);
    return ok;
}

static int test_ppo_surrogate(void) {
    int ok = 1;
    const double ratios[] = {.5, .8, 1, 1.2, 2};
    for (int sign = -1; sign <= 1; ++sign) for (size_t i = 0; i < 5; ++i) {
        const double advantage = 2 * sign, old_log = -2, current = old_log + log(ratios[i]), h = 1e-6;
        ActionPpoSurrogate term, plus, minus;
        ok &= action_record_ppo_surrogate(current, old_log, advantage, .1, &term) &&
            action_record_ppo_surrogate(current + h, old_log, advantage, .1, &plus) &&
            action_record_ppo_surrogate(current - h, old_log, advantage, .1, &minus);
        int clipped = (sign > 0 && ratios[i] > 1.1) || (sign < 0 && ratios[i] < .9);
        double expected = clipped ? 0 : ratios[i] * advantage;
        ok &= check(term.clipped == clipped && fabs(term.ratio - ratios[i]) < 1e-12 &&
            fabs(term.log_probability_gradient - expected) < 1e-12 &&
            fabs((plus.surrogate - minus.surrogate) / (2 * h) - expected) < 1e-7,
            "PPO clipped surrogate derivative matches finite differences for both signs and corrective branches");
        if (i == 2) ok &= check(term.ratio == 1 && term.log_probability_gradient == advantage, "Unchanged full-turn behavior starts at ratio one");
    }
    ActionPpoSurrogate term;
    ok &= check(!action_record_ppo_surrogate(NAN, -2, 1, .1, &term) &&
        !action_record_ppo_surrogate(0, -1000, 1, .1, &term) &&
        !action_record_ppo_surrogate(-1000, 0, 1, .1, &term) &&
        !action_record_ppo_surrogate(-2, -2, 1, 0, &term), "Invalid PPO ratios are rejected rather than floored or clamped");
    return ok;
}

static int test_ppo_baseline_and_clipping(void) {
    int ok = 1;
    srand(32101);
    GruModel* original = gru_model_create(observation_flat_size(), 8, OBS_NUM_ACTIONS);
    size_t count = gru_model_parameter_count(original);
    float* parameters = malloc(count * sizeof(float));
    float* after = malloc(count * sizeof(float));
    if (!original || !parameters || !after) { gru_model_destroy(original); free(parameters); free(after); return 0; }
    for (int sign = -1; ok && sign <= 1; sign += 2) {
        GruModel* behavior = copy_model(original);
        ok &= gru_model_export_parameters(behavior, parameters, count);
        parameters[count - 1] = sign * .5f;
        ok &= gru_model_import_parameters(behavior, parameters, count);
        GruModel* actor = copy_model(behavior);
        GruActionHeadOptimizer* optimizer = gru_action_head_optimizer_create(actor);
        float hidden[8] = {0};
        ActionLearningRecord row = {0}; row.hidden = hidden; memset(row.legal, 1, 28);
        factorized_action_choice_from_flat_actions(&row.choice, 0, 15);
        row.behavior_log_probability = record_log(behavior, &row); // Draw return is zero.
        ActionLearningOptions options = {1, 1, 1, .0001f, .05f, ACTION_PPO_FROZEN_VALUE, .01f};
        ActionLearningProgress progress = {0};
        ok &= action_record_learn(behavior, behavior, actor, &optimizer, &row, 1, &options, &progress);
        double current = record_log(actor, &row);
        ok &= check(progress.mean_value == sign * .5 && progress.mean_advantage == -sign * .5 &&
            (sign > 0 ? current < row.behavior_log_probability : current > row.behavior_log_probability) &&
            !gru_model_action_head_compatibility_reason(behavior, actor),
            "Frozen value baseline, not reward sign alone, determines PPO direction and remains unchanged");
        gru_action_head_optimizer_free(optimizer); gru_model_destroy(actor); gru_model_destroy(behavior);
    }
    // With a fresh ratio-one first update, the next identical record hits the
    // plateau. Compare to an explicit zero-gradient update: Adam momentum may
    // still move weights, so unchanged parameters would be the wrong assertion.
    for (int mode = 0; ok && mode < 3; ++mode) {
        GruModel* actor = copy_model(original);
        GruActionHeadOptimizer* optimizer = gru_action_head_optimizer_create(actor);
        float hidden[8] = {0};
        ActionLearningRecord rows[2] = {{0}};
        for (size_t i = 0; i < 2; ++i) {
            rows[i].hidden = hidden; memset(rows[i].legal, 1, 28);
            factorized_action_choice_from_flat_actions(&rows[i].choice, 0, 15);
            rows[i].terminal_return = mode == 1 || (mode == 2 && i == 1) ? -1 : 1;
            rows[i].behavior_log_probability = record_log(original, rows + i);
        }
        ActionLearningOptions options = {1, 2, 1, .0001f, .05f, ACTION_PPO_FROZEN_VALUE, .000001f};
        ActionLearningProgress progress = {0};
        ok &= action_record_learn(original, original, actor, &optimizer, rows, 2, &options, &progress);
        GruModel* control = copy_model(actor);
        GruActionHeadOptimizer* control_opt = gru_action_head_optimizer_create(control);
        size_t moments = gru_action_head_optimizer_state_count(optimizer);
        float* state = malloc(4 * moments * sizeof(float)); uint64_t step = 0, control_step = 0;
        if (!state) { ok = 0; gru_action_head_optimizer_free(control_opt); gru_model_destroy(control); gru_action_head_optimizer_free(optimizer); gru_model_destroy(actor); break; }
        ok &= gru_action_head_optimizer_export_state(optimizer, state, state + moments, moments, &step) &&
            gru_action_head_optimizer_import_state(control_opt, state, state + moments, moments, step);
        double current = record_log(actor, rows + 1);
        ActionPpoSurrogate term;
        ok &= action_record_ppo_surrogate(current, rows[1].behavior_log_probability, rows[1].terminal_return, options.ppo_clip_epsilon, &term);
        ok &= check(mode == 2 ? !term.clipped && fabs(term.log_probability_gradient) > 1 : term.clipped && term.log_probability_gradient == 0,
            "Only improvement-side PPO branches clip; corrective updates retain the importance ratio");
        ok &= gru_action_head_accumulate_hidden(control_opt, hidden, rows[1].legal, &rows[1].choice, (float)term.log_probability_gradient) &&
            gru_action_head_apply_adam(control_opt, options.learning_rate, .9f, .999f, 1e-8f, 1) &&
            action_record_learn(original, original, actor, &optimizer, rows, 2, &options, &progress) &&
            gru_model_export_parameters(actor, parameters, count) && gru_model_export_parameters(control, after, count) &&
            gru_action_head_optimizer_export_state(optimizer, state, state + moments, moments, &step) &&
            gru_action_head_optimizer_export_state(control_opt, state + 2 * moments, state + 3 * moments, moments, &control_step);
        ok &= check(!memcmp(parameters, after, count * sizeof(float)) && !memcmp(state, state + 2 * moments, 2 * moments * sizeof(float)) &&
            step == 2 && control_step == 2, "Native PPO clipping/importance weighting matches the expected gradients and Adam moments exactly");
        free(state); gru_action_head_optimizer_free(control_opt); gru_model_destroy(control);
        gru_action_head_optimizer_free(optimizer); gru_model_destroy(actor);
    }
    free(parameters); free(after); gru_model_destroy(original);
    return ok;
}

int test_action_record_learner(void) {
    return test_ppo_surrogate() && test_action_record_learner_mode(0) &&
        test_action_record_learner_mode(1) && test_ppo_baseline_and_clipping();
}
