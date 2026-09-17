#ifndef ACTION_SEMANTICS_H
#define ACTION_SEMANTICS_H

#include <stddef.h>

#include "episode.h"
#include "observation.h"
#include "request_parser.h"

#define ACTION_SEMANTIC_SLOT_DIM 101u
#define ACTION_SEMANTIC_DIM (2u * ACTION_SEMANTIC_SLOT_DIM)

/*
 * Builds a compact description of the selected doubles action. The recurrent
 * state still describes the battle; these features identify what the selected
 * move or switch means in that state.
 */
int action_semantics_build(
    float* output,
    size_t output_count,
    const Observation* observation,
    const ParsedRequest* request,
    const FactorizedActionChoice* choice
);

#endif
