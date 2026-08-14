#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "gru_model.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t step;
    float learning_rate;
    size_t bptt_window;
    float gradient_clip;
    unsigned int seed;
} TrainerCheckpointState;

#define CHECKPOINT_MIN_SUPPORTED_VERSION 1u
#define CHECKPOINT_FORMAT_VERSION 2u

typedef enum {
    CHECKPOINT_LOAD_OK = 0,
    CHECKPOINT_LOAD_INVALID_ARGUMENT,
    CHECKPOINT_LOAD_NOT_FOUND,
    CHECKPOINT_LOAD_OPEN_FAILED,
    CHECKPOINT_LOAD_TRUNCATED_HEADER,
    CHECKPOINT_LOAD_BAD_MAGIC,
    CHECKPOINT_LOAD_UNSUPPORTED_VERSION,
    CHECKPOINT_LOAD_INVALID_DIMENSIONS,
    CHECKPOINT_LOAD_INPUT_DIM_MISMATCH,
    CHECKPOINT_LOAD_ACTION_COUNT_MISMATCH,
    CHECKPOINT_LOAD_MODEL_CREATE_FAILED,
    CHECKPOINT_LOAD_PARAMETER_LAYOUT_MISMATCH,
    CHECKPOINT_LOAD_TRUNCATED_PARAMETERS,
    CHECKPOINT_LOAD_TRUNCATED_CHECKSUM,
    CHECKPOINT_LOAD_TRAILING_DATA,
    CHECKPOINT_LOAD_CHECKSUM_MISMATCH,
    CHECKPOINT_LOAD_FILE_SIZE_FAILED,
    CHECKPOINT_LOAD_ALLOCATION_FAILED,
    CHECKPOINT_LOAD_PARAMETER_IMPORT_FAILED
} CheckpointLoadStatus;

typedef enum {
    CHECKPOINT_LAYOUT_UNKNOWN = 0,
    CHECKPOINT_LAYOUT_LEGACY_FLAT = 1,
    CHECKPOINT_LAYOUT_FACTORIZED = 2
} CheckpointParameterLayout;

typedef struct {
    CheckpointLoadStatus status;
    unsigned int stored_version;
    size_t stored_input_dim;
    size_t stored_hidden_dim;
    size_t stored_num_actions;
    size_t stored_parameter_count;
    size_t expected_input_dim;
    size_t expected_num_actions;
    CheckpointParameterLayout parameter_layout;
    int migrated_legacy_heads;
    int checksum_verified;
    uint32_t stored_checksum;
    uint32_t computed_checksum;
} CheckpointLoadResult;

int checkpoint_save(const char* path, const GruModel* model, const TrainerCheckpointState* state);
GruModel* checkpoint_load(const char* path, TrainerCheckpointState* state_out);
GruModel* checkpoint_load_compatible(
    const char* path,
    TrainerCheckpointState* state_out,
    size_t expected_input_dim,
    size_t expected_num_actions,
    CheckpointLoadResult* result_out
);
const char* checkpoint_load_status_string(CheckpointLoadStatus status);

#endif
