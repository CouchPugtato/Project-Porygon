#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
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

int checkpoint_save(const char* path, const GruModel* model, const TrainerCheckpointState* state) {
    FILE* f;
    CheckpointHeader header;
    float* params;
    size_t count;

    if (!path || !model || !state) {
        return 0;
    }
    if (!ensure_parent_directory(path)) {
        return 0;
    }
    f = fopen(path, "wb");
    if (!f) {
        return 0;
    }

    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "PORYCHK", 7);
    header.version = 1;
    header.input_dim = gru_model_input_dim(model);
    header.hidden_dim = gru_model_hidden_dim(model);
    header.num_actions = gru_model_num_actions(model);
    header.parameter_count = gru_model_parameter_count(model);
    header.trainer = *state;

    count = header.parameter_count;
    params = (float*)malloc(count * sizeof(float));
    if (!params) {
        fclose(f);
        return 0;
    }
    if (!gru_model_export_parameters(model, params, count)) {
        free(params);
        fclose(f);
        return 0;
    }
    fwrite(&header, sizeof(header), 1, f);
    fwrite(params, sizeof(float), count, f);
    free(params);
    fclose(f);
    return 1;
}

GruModel* checkpoint_load(const char* path, TrainerCheckpointState* state_out) {
    FILE* f;
    CheckpointHeader header;
    GruModel* model;
    float* params;
    if (!path) {
        return NULL;
    }
    f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fread(&header, sizeof(header), 1, f) != 1 || memcmp(header.magic, "PORYCHK", 7) != 0) {
        fclose(f);
        return NULL;
    }
    model = gru_model_create(header.input_dim, header.hidden_dim, header.num_actions);
    if (!model) {
        fclose(f);
        return NULL;
    }
    params = (float*)malloc(header.parameter_count * sizeof(float));
    if (!params) {
        gru_model_destroy(model);
        fclose(f);
        return NULL;
    }
    if (fread(params, sizeof(float), header.parameter_count, f) != header.parameter_count ||
        !gru_model_import_parameters(model, params, header.parameter_count)) {
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
    return model;
}
