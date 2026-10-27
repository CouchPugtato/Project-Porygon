#ifndef TARGET_CONTEXT_H
#define TARGET_CONTEXT_H

#include <stdint.h>
#include <stdio.h>
#include "action_mapper.h"

typedef struct {
    unsigned char position, present, friendly, type1_id, type1_known_mode, type2_id, type2_known_mode;
} TargetContextPokemon;

/* A 32-byte, little-endian cache record; no scores or hidden opponent facts. */
typedef struct {
    uint16_t move_id;
    unsigned char move_type_id, move_type_known;
    TargetContextPokemon targets[4];
} TargetContext;

void target_context_collect(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int slot, TargetContext* context);
void target_context_write(FILE* out, const TargetContext* context);
int target_context_valid(const TargetContext* context);
#define TARGET_CONTEXT_PARAMETER_COUNT (2 * OBS_NUM_TYPES * OBS_NUM_TYPES)
size_t target_context_feature_indices(const TargetContext* context, int target, size_t indices[2]);

#endif
