#include "checkpoint.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECKPOINT_MAX_INPUT_DIM 1000000u
#define CHECKPOINT_MAX_HIDDEN_DIM 4096u
#define CHECKPOINT_MAX_ACTIONS 100000u

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define MKDIR(path) mkdir(path, 0777)
#endif

typedef struct {
    char magic[8];
    unsigned int version;
    size_t input_dim;
    size_t hidden_dim;
    size_t num_actions;
    size_t parameter_count;
    TrainerCheckpointState trainer;
} CheckpointHeader;

static int ensure_parent_directory(const char* path) {
    char buffer[1024];
    size_t len;
    size_t i;

    if (!path) {
        return 0;
    }
    len = strlen(path);
    if (len == 0 || len >= sizeof(buffer)) {
        return 0;
    }
    memcpy(buffer, path, len + 1);
    for (i = 0; i < len; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\\') {
            char saved = buffer[i];
            if (i > 0) {
                buffer[i] = '\0';
                MKDIR(buffer);
                buffer[i] = saved;
            }
        }
    }
    return 1;
}

static int flush_checkpoint_file(FILE* file) {
    if (!file || fflush(file) != 0) {
        return 0;
    }
#ifdef _WIN32
    return _commit(_fileno(file)) == 0;
#else
    return fsync(fileno(file)) == 0;
#endif
}

static int replace_checkpoint_file(const char* temporary_path, const char* path) {
#ifdef _WIN32
    return MoveFileExA(
        temporary_path,
        path,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    ) != 0;
#else
    return rename(temporary_path, path) == 0;
#endif
}

int checkpoint_save(const char* path, const GruModel* model, const TrainerCheckpointState* state) {
    static const char temporary_suffix[] = ".tmp";
    FILE* f = NULL;
    CheckpointHeader header;
    float* params = NULL;
    char* temporary_path = NULL;
    size_t count;
    size_t path_length;
    int saved = 0;

    if (!path || !*path || !model || !state) {
        return 0;
    }
    if (!ensure_parent_directory(path)) {
        return 0;
    }
    path_length = strlen(path);
    if (path_length > SIZE_MAX - sizeof(temporary_suffix)) {
        return 0;
    }

    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "PORYCHK", 7);
    header.version = CHECKPOINT_FORMAT_VERSION;
    header.input_dim = gru_model_input_dim(model);
    header.hidden_dim = gru_model_hidden_dim(model);
    header.num_actions = gru_model_num_actions(model);
    header.parameter_count = gru_model_parameter_count(model);
    header.trainer = *state;

    count = header.parameter_count;
    if (count == 0 || count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    params = (float*)malloc(count * sizeof(float));
    if (!params) {
        return 0;
    }
    if (!gru_model_export_parameters(model, params, count)) {
        goto cleanup;
    }

    temporary_path = (char*)malloc(path_length + sizeof(temporary_suffix));
    if (!temporary_path) {
        goto cleanup;
    }
    memcpy(temporary_path, path, path_length);
    memcpy(temporary_path + path_length, temporary_suffix, sizeof(temporary_suffix));
    f = fopen(temporary_path, "wb");
    if (!f) {
        goto cleanup;
    }

    if (fwrite(&header, sizeof(header), 1, f) != 1 ||
            fwrite(params, sizeof(float), count, f) != count) {
        goto cleanup;
    }
    if (!flush_checkpoint_file(f)) {
        goto cleanup;
    }
    if (fclose(f) != 0) {
        f = NULL;
        goto cleanup;
    }
    f = NULL;
    if (!replace_checkpoint_file(temporary_path, path)) {
        goto cleanup;
    }
    saved = 1;

cleanup:
    if (f) {
        fclose(f);
    }
    if (!saved && temporary_path) {
        remove(temporary_path);
    }
    free(params);
    free(temporary_path);
    return saved;
}

static void checkpoint_load_result_init(
    CheckpointLoadResult* result,
    size_t expected_input_dim,
    size_t expected_num_actions
) {
    if (!result) {
        return;
    }
    memset(result, 0, sizeof(*result));
    result->status = CHECKPOINT_LOAD_INVALID_ARGUMENT;
    result->expected_input_dim = expected_input_dim;
    result->expected_num_actions = expected_num_actions;
}

static void checkpoint_load_result_header(CheckpointLoadResult* result, const CheckpointHeader* header) {
    if (!result || !header) {
        return;
    }
    result->stored_version = header->version;
    result->stored_input_dim = header->input_dim;
    result->stored_hidden_dim = header->hidden_dim;
    result->stored_num_actions = header->num_actions;
    result->stored_parameter_count = header->parameter_count;
}

const char* checkpoint_load_status_string(CheckpointLoadStatus status) {
    switch (status) {
        case CHECKPOINT_LOAD_OK: return "ok";
        case CHECKPOINT_LOAD_INVALID_ARGUMENT: return "invalid argument";
        case CHECKPOINT_LOAD_NOT_FOUND: return "checkpoint not found";
        case CHECKPOINT_LOAD_OPEN_FAILED: return "checkpoint open failed";
        case CHECKPOINT_LOAD_TRUNCATED_HEADER: return "truncated checkpoint header";
        case CHECKPOINT_LOAD_BAD_MAGIC: return "invalid checkpoint magic";
        case CHECKPOINT_LOAD_UNSUPPORTED_VERSION: return "unsupported checkpoint version";
        case CHECKPOINT_LOAD_INVALID_DIMENSIONS: return "invalid checkpoint dimensions";
        case CHECKPOINT_LOAD_INPUT_DIM_MISMATCH: return "observation input dimension mismatch";
        case CHECKPOINT_LOAD_ACTION_COUNT_MISMATCH: return "action count mismatch";
        case CHECKPOINT_LOAD_MODEL_CREATE_FAILED: return "failed to create checkpoint model";
        case CHECKPOINT_LOAD_PARAMETER_LAYOUT_MISMATCH: return "unsupported checkpoint parameter layout";
        case CHECKPOINT_LOAD_TRUNCATED_PARAMETERS: return "truncated checkpoint parameter payload";
        case CHECKPOINT_LOAD_TRAILING_DATA: return "unexpected trailing checkpoint data";
        case CHECKPOINT_LOAD_FILE_SIZE_FAILED: return "failed to determine checkpoint file size";
        case CHECKPOINT_LOAD_ALLOCATION_FAILED: return "checkpoint allocation failed";
        case CHECKPOINT_LOAD_PARAMETER_IMPORT_FAILED: return "checkpoint parameter import failed";
        default: return "unknown checkpoint load error";
    }
}

GruModel* checkpoint_load_compatible(
    const char* path,
    TrainerCheckpointState* state_out,
    size_t expected_input_dim,
    size_t expected_num_actions,
    CheckpointLoadResult* result_out
) {
    FILE* f;
    CheckpointHeader header;
    GruModel* model = NULL;
    float* params = NULL;
    size_t current_parameter_count;
    size_t legacy_parameter_count;
    size_t expected_file_size;
    long actual_file_size;
    CheckpointLoadResult local_result;
    CheckpointLoadResult* result = result_out ? result_out : &local_result;

    checkpoint_load_result_init(result, expected_input_dim, expected_num_actions);
    if (state_out) {
        memset(state_out, 0, sizeof(*state_out));
    }
    if (!path || !*path) {
        return NULL;
    }
    errno = 0;
    f = fopen(path, "rb");
    if (!f) {
        result->status = errno == ENOENT ? CHECKPOINT_LOAD_NOT_FOUND : CHECKPOINT_LOAD_OPEN_FAILED;
        return NULL;
    }
    memset(&header, 0, sizeof(header));
    if (fread(&header, sizeof(header), 1, f) != 1) {
        result->status = CHECKPOINT_LOAD_TRUNCATED_HEADER;
        fclose(f);
        return NULL;
    }
    if (memcmp(header.magic, "PORYCHK", 7) != 0) {
        result->status = CHECKPOINT_LOAD_BAD_MAGIC;
        fclose(f);
        return NULL;
    }
    checkpoint_load_result_header(result, &header);
    if (header.version != CHECKPOINT_FORMAT_VERSION) {
        result->status = CHECKPOINT_LOAD_UNSUPPORTED_VERSION;
        fclose(f);
        return NULL;
    }
    if (header.input_dim == 0 || header.input_dim > CHECKPOINT_MAX_INPUT_DIM ||
            header.hidden_dim == 0 || header.hidden_dim > CHECKPOINT_MAX_HIDDEN_DIM ||
            header.num_actions == 0 || header.num_actions > CHECKPOINT_MAX_ACTIONS) {
        result->status = CHECKPOINT_LOAD_INVALID_DIMENSIONS;
        fclose(f);
        return NULL;
    }
    if (expected_input_dim > 0 && header.input_dim != expected_input_dim) {
        result->status = CHECKPOINT_LOAD_INPUT_DIM_MISMATCH;
        fclose(f);
        return NULL;
    }
    if (expected_num_actions > 0 && header.num_actions != expected_num_actions) {
        result->status = CHECKPOINT_LOAD_ACTION_COUNT_MISMATCH;
        fclose(f);
        return NULL;
    }
    model = gru_model_create(header.input_dim, header.hidden_dim, header.num_actions);
    if (!model) {
        result->status = CHECKPOINT_LOAD_MODEL_CREATE_FAILED;
        fclose(f);
        return NULL;
    }
    current_parameter_count = gru_model_parameter_count(model);
    legacy_parameter_count = gru_model_legacy_parameter_count(model);
    if (header.parameter_count == current_parameter_count) {
        result->parameter_layout = CHECKPOINT_LAYOUT_FACTORIZED;
    } else if (header.parameter_count == legacy_parameter_count) {
        result->parameter_layout = CHECKPOINT_LAYOUT_LEGACY_FLAT;
    } else {
        result->status = CHECKPOINT_LOAD_PARAMETER_LAYOUT_MISMATCH;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if (header.parameter_count > (SIZE_MAX - sizeof(header)) / sizeof(float)) {
        result->status = CHECKPOINT_LOAD_FILE_SIZE_FAILED;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    expected_file_size = sizeof(header) + (header.parameter_count * sizeof(float));
    if (expected_file_size > (size_t)LONG_MAX || fseek(f, 0, SEEK_END) != 0) {
        result->status = CHECKPOINT_LOAD_FILE_SIZE_FAILED;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    actual_file_size = ftell(f);
    if (actual_file_size < 0) {
        result->status = CHECKPOINT_LOAD_FILE_SIZE_FAILED;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if ((size_t)actual_file_size < expected_file_size) {
        result->status = CHECKPOINT_LOAD_TRUNCATED_PARAMETERS;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if ((size_t)actual_file_size > expected_file_size) {
        result->status = CHECKPOINT_LOAD_TRAILING_DATA;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if (fseek(f, (long)sizeof(header), SEEK_SET) != 0) {
        result->status = CHECKPOINT_LOAD_FILE_SIZE_FAILED;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    params = (float*)malloc(header.parameter_count * sizeof(float));
    if (!params) {
        result->status = CHECKPOINT_LOAD_ALLOCATION_FAILED;
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if (fread(params, sizeof(float), header.parameter_count, f) != header.parameter_count ||
        !gru_model_import_parameters(model, params, header.parameter_count)) {
        result->status = CHECKPOINT_LOAD_PARAMETER_IMPORT_FAILED;
        free(params);
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if (state_out) {
        *state_out = header.trainer;
    }
    free(params);
    fclose(f);
    result->status = CHECKPOINT_LOAD_OK;
    result->migrated_legacy_heads = result->parameter_layout == CHECKPOINT_LAYOUT_LEGACY_FLAT;
    return model;
}

GruModel* checkpoint_load(const char* path, TrainerCheckpointState* state_out) {
    return checkpoint_load_compatible(path, state_out, 0, 0, NULL);
}
