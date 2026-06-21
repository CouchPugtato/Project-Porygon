#ifndef OBSERVATION_BUILDER_H
#define OBSERVATION_BUILDER_H

#include "action_mapper.h"
#include "observation.h"
#include "raw_battle_state.h"

void observation_from_raw_state(
    Observation* out,
    const RawBattleState* state,
    const ParsedRequest* req,
    const ActionMask* mask
);

#endif
