#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "gru_model.h"

#include <stddef.h>

typedef struct {
    size_t step;
    float learning_rate;
    size_t bptt_window;
    float gradient_clip;
    unsigned int seed;
} TrainerCheckpointState;

int checkpoint_save(const char* path, const GruModel* model, const TrainerCheckpointState* state);
GruModel* checkpoint_load(const char* path, TrainerCheckpointState* state_out);

#endif
