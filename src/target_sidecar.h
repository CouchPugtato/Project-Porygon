#ifndef TARGET_SIDECAR_H
#define TARGET_SIDECAR_H

#include <stdint.h>
#include "gru_model.h"

typedef struct {
    size_t hidden_dim;
    uint64_t parent_fingerprint;
    float* weights;
} TargetSidecar;

uint64_t target_parent_fingerprint(const GruModel* model);
TargetSidecar* target_sidecar_create(const GruModel* parent);
void target_sidecar_free(TargetSidecar* sidecar);
size_t target_sidecar_parameter_count(const TargetSidecar* sidecar);
int target_sidecar_predict(const TargetSidecar* sidecar, const float* hidden,
    int slot, int move, unsigned char legal, float* policy);
/* Metadata/visibility eligibility only: no teacher score or target ranking. */
int target_sidecar_supports(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice);
double target_sidecar_gradient(const TargetSidecar* sidecar, const float* hidden,
    int slot, int move, unsigned char legal, unsigned char best, float* gradient);
int target_sidecar_save(const TargetSidecar* sidecar, const char* path);
TargetSidecar* target_sidecar_load(const char* path, const GruModel* parent);

/* The dataset is a bounded binary cache produced by the accepted-label
 * collector. No base-model parameters are updated by this command. */
int target_sidecar_fit(const char* dataset, const GruModel* parent,
    const char* output_path, const char* report_path);

#endif
