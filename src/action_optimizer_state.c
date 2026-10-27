#include "action_optimizer_state.h"
#include "artifact_io.h"
#include "target_sidecar.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

static const char MAGIC[9] = "PGACTOPT1";
enum { VERSION, ABI, INPUT, HIDDEN, ACTIONS, COUNT, REFERENCE, ACTOR, STEP, HEADER_WORDS };

static const char* state_context(const GruModel* reference, const GruModel* actor, const char* pin, uint64_t* header) {
    _Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24, "state requires float32");
    if (!pin || strlen(pin) != 64 || strspn(pin, "0123456789abcdef") != 64) return "bundle_pin";
    const char* reason = gru_model_action_head_compatibility_reason(reference, actor);
    if (reason) return reason;
    if (gru_model_hidden_dim(actor) > 4096) return "model_dimensions";
    header[VERSION] = 1;
    header[ABI] = UINT64_C(0x010203043f800000);
    header[INPUT] = gru_model_input_dim(actor);
    header[HIDDEN] = gru_model_hidden_dim(actor);
    header[ACTIONS] = gru_model_num_actions(actor);
    header[REFERENCE] = target_parent_fingerprint(reference);
    header[ACTOR] = target_parent_fingerprint(actor);
    return header[REFERENCE] && header[ACTOR] ? NULL : "parameter_fingerprint";
}

static uint64_t state_checksum(const uint64_t* header, const char* pin, const float* moments) {
    uint64_t hash = artifact_fingerprint_bytes(UINT64_C(14695981039346656037), MAGIC, sizeof(MAGIC));
    hash = artifact_fingerprint_bytes(hash, header, HEADER_WORDS * sizeof(uint64_t));
    hash = artifact_fingerprint_bytes(hash, pin, 64);
    return artifact_fingerprint_bytes(hash, moments, 2 * (size_t)header[COUNT] * sizeof(float));
}

int action_optimizer_state_save(const char* path, const GruActionHeadOptimizer* optimizer,
    const GruModel* reference, const char* bundle_sha256, const char** failure_reason
) {
    const char* reason = NULL;
    uint64_t header[HEADER_WORDS] = {0};
    float* moments = NULL;
    if (!path || !*path || !optimizer) { reason = "invalid_argument"; goto done; }
    reason = state_context(reference, gru_action_head_optimizer_model(optimizer), bundle_sha256, header);
    if (reason) goto done;
    size_t count = gru_action_head_optimizer_state_count(optimizer);
    if (!count || count > SIZE_MAX / (2 * sizeof(float))) { reason = "state_dimensions"; goto done; }
    header[COUNT] = count;
    moments = malloc(2 * count * sizeof(float));
    if (!moments) { reason = "allocation_failure"; goto done; }
    if (!gru_action_head_optimizer_export_state(optimizer, moments, moments + count, count, &header[STEP])) {
        reason = "unfinished_or_invalid_update"; goto done;
    }
    char temporary[4096];
    if (snprintf(temporary, sizeof(temporary), "%s.tmp", path) >= (int)sizeof(temporary)) { reason = "path_length"; goto done; }
    uint64_t checksum = state_checksum(header, bundle_sha256, moments);
    FILE* file = fopen(temporary, "wb");
    if (!file) { reason = "open_failed"; goto done; }
    int written = fwrite(MAGIC, 1, sizeof(MAGIC), file) == sizeof(MAGIC) &&
        fwrite(header, sizeof(uint64_t), HEADER_WORDS, file) == HEADER_WORDS &&
        fwrite(bundle_sha256, 1, 64, file) == 64 &&
        fwrite(moments, sizeof(float), 2 * count, file) == 2 * count &&
        fwrite(&checksum, sizeof(checksum), 1, file) == 1;
    if (!artifact_finish_file(file)) written = 0;
    if (!written) reason = "write_failed";
    else if (!artifact_publish_file(temporary, path)) reason = "publish_failed";
done:
    free(moments);
    if (failure_reason) *failure_reason = reason;
    return reason == NULL;
}

GruActionHeadOptimizer* action_optimizer_state_load(const char* path, GruModel* actor,
    const GruModel* reference, const char* bundle_sha256, const char** failure_reason
) {
    const char* reason = NULL;
    uint64_t expected[HEADER_WORDS] = {0}, header[HEADER_WORDS], checksum;
    char magic[sizeof(MAGIC)], pin[64];
    FILE* file = NULL;
    float* moments = NULL;
    GruActionHeadOptimizer* optimizer = NULL;
    if (!path || !*path) { reason = "invalid_argument"; goto done; }
    reason = state_context(reference, actor, bundle_sha256, expected);
    if (reason) goto done;
    optimizer = gru_action_head_optimizer_create(actor);
    if (!optimizer) { reason = "allocation_failure"; goto done; }
    expected[COUNT] = gru_action_head_optimizer_state_count(optimizer);
    file = fopen(path, "rb");
    if (!file) { reason = "open_failed"; goto done; }
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        fread(header, sizeof(uint64_t), HEADER_WORDS, file) != HEADER_WORDS || fread(pin, 1, 64, file) != 64) {
        reason = "truncated_header"; goto done;
    }
    if (memcmp(magic, MAGIC, sizeof(magic)) || header[VERSION] != expected[VERSION] || header[ABI] != expected[ABI]) {
        reason = "state_format"; goto done;
    }
    if (memcmp(pin, bundle_sha256, sizeof(pin))) { reason = "bundle_pin"; goto done; }
    if (header[INPUT] != expected[INPUT] || header[HIDDEN] != expected[HIDDEN] || header[ACTIONS] != expected[ACTIONS] ||
        header[COUNT] != expected[COUNT]) { reason = "state_dimensions"; goto done; }
    if (header[REFERENCE] != expected[REFERENCE] || header[ACTOR] != expected[ACTOR]) { reason = "parameter_fingerprint"; goto done; }
    size_t count = (size_t)expected[COUNT];
    if (count > SIZE_MAX / (2 * sizeof(float))) { reason = "state_dimensions"; goto done; }
    moments = malloc(2 * count * sizeof(float));
    if (!moments) { reason = "allocation_failure"; goto done; }
    if (fread(moments, sizeof(float), 2 * count, file) != 2 * count || fread(&checksum, sizeof(checksum), 1, file) != 1) {
        reason = "truncated_state"; goto done;
    }
    if (fgetc(file) != EOF || ferror(file)) { reason = "trailing_or_read_error"; goto done; }
    if (checksum != state_checksum(header, pin, moments)) { reason = "checksum"; goto done; }
    if (!gru_action_head_optimizer_import_state(optimizer, moments, moments + count, count, header[STEP])) reason = "invalid_moments";
done:
    if (file && fclose(file) != 0 && !reason) reason = "read_close_failed";
    free(moments);
    if (reason) { gru_action_head_optimizer_free(optimizer); optimizer = NULL; }
    if (failure_reason) *failure_reason = reason;
    return optimizer;
}
