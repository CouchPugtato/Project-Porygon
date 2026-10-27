#ifndef TARGET_BEHAVIOR_H
#define TARGET_BEHAVIOR_H

#include "action_mapper.h"
#include <stdint.h>
#include <stdio.h>

/* Parameter fingerprints identify loaded weights; experiment manifests retain
 * file SHA256 and executable hashes separately. This is not a PPO schema. */
void target_behavior_identity_write(FILE* out, uint64_t parent, uint64_t sidecar,
    int contextual, int argmax, int slot_local);
void target_behavior_write(FILE* out, const FactorizedActionChoice* choice,
    unsigned char learned_slots, const float policies[2][4]);

#endif
