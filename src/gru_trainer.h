#ifndef GRU_TRAINER_H
#define GRU_TRAINER_H

#include "checkpoint.h"
#include "episode.h"
#include "gru_model.h"

typedef struct {
    size_t step;
    float learning_rate;
    size_t bptt_window;
    float gradient_clip;
    unsigned int seed;
    float last_action_loss;
    float last_value_loss;
    float last_accuracy;
} GruTrainer;

void gru_trainer_init(GruTrainer* trainer, float learning_rate, size_t bptt_window, float gradient_clip, unsigned int seed);
TrainerCheckpointState gru_trainer_checkpoint_state(const GruTrainer* trainer);
int gru_trainer_supervised_episode(GruTrainer* trainer, GruModel* model, const Episode* episode);
int gru_trainer_policy_gradient_episode(GruTrainer* trainer, GruModel* model, const Episode* episode);

#endif
