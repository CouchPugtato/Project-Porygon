#include "action_optimizer_state.h"
#include "artifact_io.h"
#include "checkpoint.h"
#include "observation.h"
#include "target_sidecar.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int check(int condition, const char* message) {
    if (!condition) fprintf(stderr, "test failed: %s\n", message);
    return condition;
}

static int update(GruActionHeadOptimizer* optimizer, int index) {
    unsigned char legal[OBS_NUM_ACTIONS];
    memset(legal, 1, sizeof(legal));
    float hidden[8] = {.2f, -.3f, .1f, .4f};
    hidden[index % 8] += .07f;
    FactorizedActionChoice choice;
    int mode = index % 3;
    factorized_action_choice_from_flat_actions(&choice, mode == 2 ? -1 : mode == 1 ? 8 : 0,
        mode == 1 ? -1 : mode == 2 ? 18 : 15);
    float advantage = index == 6 ? 0 : index % 2 ? -.7f : .8f;
    return gru_action_head_accumulate_hidden(optimizer, hidden, legal, &choice, advantage) &&
        gru_action_head_accumulate_hidden(optimizer, hidden, legal, &choice, advantage * .3f) &&
        gru_action_head_apply_adam(optimizer, .001f, .9f, .999f, 1e-8f, 1);
}

static unsigned char* read_file(const char* path, size_t* count) {
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END)) { fclose(file); return NULL; }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET)) { fclose(file); return NULL; }
    unsigned char* bytes = malloc((size_t)length + 1);
    int ok = bytes && fread(bytes, 1, (size_t)length, file) == (size_t)length;
    fclose(file);
    if (!ok) { free(bytes); return NULL; }
    *count = (size_t)length;
    return bytes;
}

static int write_file(const char* path, const void* bytes, size_t count) {
    FILE* file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(bytes, 1, count, file) == count;
    if (fclose(file)) ok = 0;
    return ok;
}

int test_action_optimizer_state(void) {
    const char* state_path = "test_action_optimizer_state.tmp";
    const char* bad_path = "test_action_optimizer_state_bad.tmp";
    const char* checkpoint_path = "test_action_optimizer_actor.tmp";
    char pin[65], wrong_pin[65];
    memset(pin, 'a', 64); pin[64] = 0;
    memset(wrong_pin, 'b', 64); wrong_pin[64] = 0;
    srand(31901);
    GruModel* reference = gru_model_create(observation_flat_size(), 8, OBS_NUM_ACTIONS);
    GruModel* actor = gru_model_create(observation_flat_size(), 8, OBS_NUM_ACTIONS);
    GruModel* resumed = NULL;
    GruActionHeadOptimizer* uninterrupted = NULL, *restored = NULL, *pending = NULL;
    float *before = NULL, *after = NULL, *first = NULL, *second = NULL;
    unsigned char *original = NULL, *bad = NULL;
    const char* reason = NULL;
    int ok = reference && actor;
    size_t parameter_count = gru_model_parameter_count(reference), file_size = 0, count = 0;
    if (!ok) goto done;
    before = malloc(parameter_count * sizeof(float)); after = malloc(parameter_count * sizeof(float));
    ok = before && after && gru_model_export_parameters(reference, before, parameter_count) &&
        gru_model_import_parameters(actor, before, parameter_count);
    if (!ok) goto done;
    const uint64_t reference_fingerprint = target_parent_fingerprint(reference);
    uninterrupted = gru_action_head_optimizer_create(actor);
    if (!uninterrupted) { ok = 0; goto done; }
    count = gru_action_head_optimizer_state_count(uninterrupted);
    first = malloc(2 * count * sizeof(float)); second = malloc(2 * count * sizeof(float));
    if (!first || !second) { ok = 0; goto done; }
    ok &= check(action_optimizer_state_save(state_path, uninterrupted, reference, pin, &reason) && !reason,
        "Fresh zero-step action optimizer state can be saved");
    restored = action_optimizer_state_load(state_path, actor, reference, pin, &reason);
    uint64_t step = UINT64_MAX;
    ok &= check(restored && !reason && gru_action_head_optimizer_export_state(restored, first, first + count, count, &step) && step == 0,
        "Fresh optimizer state loads with zero moments and zero updates");
    gru_action_head_optimizer_free(restored); restored = NULL;
    for (int i = 0; ok && i < 3; ++i) ok &= update(uninterrupted, i);
    ok &= gru_model_export_parameters(actor, before, parameter_count) &&
        action_optimizer_state_save(state_path, uninterrupted, reference, pin, &reason);
    TrainerCheckpointState trainer = {0};
    ok &= checkpoint_save(checkpoint_path, actor, &trainer);
    resumed = checkpoint_load(checkpoint_path, NULL);
    if (!ok || !resumed) { ok = 0; goto done; }
    srand(31902); int next_random = rand(); srand(31902);
    restored = action_optimizer_state_load(state_path, resumed, reference, pin, &reason);
    ok &= check(restored && !reason && rand() == next_random &&
        gru_model_export_parameters(resumed, after, parameter_count) && !memcmp(before, after, parameter_count * sizeof(float)),
        "Restoring action optimizer state changes neither actor weights nor RNG");
    original = read_file(state_path, &file_size);
    bad = malloc(file_size + 1);
    if (!ok || !original || !bad) { ok = 0; goto done; }

    GruActionHeadOptimizer* rejected = action_optimizer_state_load(state_path, resumed, reference, wrong_pin, &reason);
    ok &= check(!rejected && reason && !strcmp(reason, "bundle_pin"), "Wrong behavior bundle pin cannot restore moments");
    gru_action_head_optimizer_free(rejected);
    rejected = action_optimizer_state_load(state_path, resumed, resumed, pin, &reason);
    ok &= check(!rejected && reason && !strcmp(reason, "parameter_fingerprint"), "Changed compatible reference cannot restore old state");
    gru_action_head_optimizer_free(rejected);
    pending = gru_action_head_optimizer_create(resumed);
    unsigned char legal[OBS_NUM_ACTIONS]; memset(legal, 1, sizeof(legal));
    float hidden[8] = {.2f};
    FactorizedActionChoice choice; factorized_action_choice_from_flat_actions(&choice, 0, 15);
    ok &= gru_action_head_accumulate_hidden(pending, hidden, legal, &choice, 1);
    ok &= check(!action_optimizer_state_save(state_path, pending, reference, pin, &reason) &&
        reason && !strcmp(reason, "unfinished_or_invalid_update"), "Pending minibatch cannot overwrite a completed optimizer state");
    size_t saved_size = 0;
    unsigned char* saved_bytes = read_file(state_path, &saved_size);
    ok &= check(saved_bytes && saved_size == file_size && !memcmp(original, saved_bytes, file_size),
        "Refused partial-update save preserves the old file byte-for-byte");
    free(saved_bytes);
    gru_action_head_optimizer_free(pending); pending = NULL;

    // Corruption cases include valid-checksum invalid moments: the loader must
    // validate Adam state, not merely trust a correctly sized byte stream.
    for (int mutation = 0; ok && mutation < 11; ++mutation) {
        memcpy(bad, original, file_size);
        size_t length = file_size;
        if (mutation == 0) bad[0] ^= 1;
        if (mutation == 1) bad[file_size - 9] ^= 1;
        if (mutation == 2) length -= 3;
        if (mutation == 3) bad[length++] = 0;
        if (mutation >= 4) {
            size_t moments_offset = 9 + 9 * sizeof(uint64_t) + 64;
            if (mutation == 4) { float value = NAN; memcpy(bad + moments_offset, &value, sizeof(value)); }
            if (mutation == 5) { float value = -1; memcpy(bad + moments_offset + count * sizeof(float), &value, sizeof(value)); }
            if (mutation == 6) { uint64_t value = UINT64_MAX; memcpy(bad + 9 + 8 * sizeof(uint64_t), &value, sizeof(value)); }
            if (mutation == 7) { uint64_t value = 0; memcpy(bad + 9 + 8 * sizeof(uint64_t), &value, sizeof(value)); }
            if (mutation == 8) { uint64_t value = UINT64_MAX; memcpy(bad + 9 + 5 * sizeof(uint64_t), &value, sizeof(value)); }
            if (mutation == 9) { uint64_t value = 2; memcpy(bad + 9, &value, sizeof(value)); }
            if (mutation == 10) { uint64_t value = 0; memcpy(bad + 9 + sizeof(uint64_t), &value, sizeof(value)); }
            uint64_t checksum = artifact_fingerprint_bytes(UINT64_C(14695981039346656037), bad, file_size - sizeof(uint64_t));
            memcpy(bad + file_size - sizeof(uint64_t), &checksum, sizeof(checksum));
        }
        ok &= write_file(bad_path, bad, length);
        rejected = action_optimizer_state_load(bad_path, resumed, reference, pin, &reason);
        ok &= check(!rejected && reason, "Corrupt/truncated/trailing/invalid action optimizer state is rejected");
        gru_action_head_optimizer_free(rejected);
    }
    ok &= gru_model_export_parameters(resumed, after, parameter_count) &&
        check(!memcmp(before, after, parameter_count * sizeof(float)), "Failed restores never mutate the actor");
    ok &= gru_action_head_optimizer_export_state(restored, first, first + count, count, &step);
    ok &= check(!gru_action_head_optimizer_import_state(restored, first, first + count, count, step),
        "Live optimizer momentum cannot be overwritten through the fresh-state import API");

    for (int i = 3; ok && i < 9; ++i) {
        ok &= update(uninterrupted, i) && update(restored, i) &&
            gru_model_export_parameters(actor, before, parameter_count) && gru_model_export_parameters(resumed, after, parameter_count);
        ok &= check(!memcmp(before, after, parameter_count * sizeof(float)),
            "Uninterrupted and save/checkpoint/reload/resume updates produce byte-identical weights");
        ok &= check(!gru_model_action_head_compatibility_reason(reference, resumed) &&
            target_parent_fingerprint(reference) == reference_fingerprint,
            "Every resumed update preserves frozen recurrent, entity, target and value parameters");
    }
    uint64_t other_step = 0;
    ok &= gru_action_head_optimizer_export_state(uninterrupted, first, first + count, count, &step) &&
        gru_action_head_optimizer_export_state(restored, second, second + count, count, &other_step);
    ok &= check(step == 9 && other_step == step && !memcmp(first, second, 2 * count * sizeof(float)),
        "Resumed Adam update counter and both moment arrays exactly match uninterrupted state");
    rejected = action_optimizer_state_load(state_path, resumed, reference, pin, &reason);
    ok &= check(!rejected && reason && !strcmp(reason, "parameter_fingerprint"), "A newer compatible actor cannot load stale moments");
    gru_action_head_optimizer_free(rejected);
    // An interrupted write leaves a sibling temporary file, never a new state.
    ok &= action_optimizer_state_save(state_path, restored, reference, pin, &reason) &&
        write_file("test_action_optimizer_state.tmp.tmp", "partial", 7);
    rejected = action_optimizer_state_load(state_path, resumed, reference, pin, &reason);
    ok &= check(rejected && !reason, "Uncommitted temporary output is ignored when loading a completed primary");
    gru_action_head_optimizer_free(rejected);
    ok &= action_optimizer_state_save(state_path, restored, reference, pin, &reason);
    rejected = action_optimizer_state_load(state_path, resumed, reference, pin, &reason);
    ok &= check(rejected && !reason, "Atomic publication replaces interrupted temporary output with a complete state");
    gru_action_head_optimizer_free(rejected);
done:
    free(before); free(after); free(first); free(second); free(original); free(bad);
    gru_action_head_optimizer_free(uninterrupted); gru_action_head_optimizer_free(restored); gru_action_head_optimizer_free(pending);
    gru_model_destroy(reference); gru_model_destroy(actor); gru_model_destroy(resumed);
    remove(state_path); remove("test_action_optimizer_state.tmp.tmp"); remove(bad_path); remove(checkpoint_path);
    return ok;
}
