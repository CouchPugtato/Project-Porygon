#ifndef TARGET_LABELS_H
#define TARGET_LABELS_H

#include <stdint.h>
#include <stdio.h>
#include "tactical_policy.h"

typedef struct { unsigned char legal[2], best[2]; } TargetLabels;

/* Labels only the chosen actions; enumeration never changes play or RNG. */
int target_labels_collect(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TargetLabels* labels);
void target_labels_write(FILE* out, const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, const float* hidden, size_t hidden_dim, uint64_t fingerprint);

#endif
