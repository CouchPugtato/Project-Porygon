#ifndef TARGET_BEHAVIOR_H
#define TARGET_BEHAVIOR_H

#include "action_mapper.h"
#include "target_sidecar.h"
#include <stdint.h>
#include <stdio.h>

/* Parameter fingerprints identify loaded weights; experiment manifests retain
 * file SHA256 and executable hashes separately. This is not a PPO schema. */
void target_behavior_identity_write(FILE* out, uint64_t parent, uint64_t sidecar,
    int contextual, int argmax, int slot_local);
void target_action_bundle_identity_write(FILE* out, uint64_t reference, uint64_t actor,
    uint64_t sidecar, int argmax, int slot_local, const char* manifest_sha256);
void target_behavior_write(FILE* out, const FactorizedActionChoice* choice,
    unsigned char learned_slots, const float policies[2][4], const double* full_turn_log_probability);

/* Read-only replay at the post-forward hidden state. Caller must verify the
 * bundle binding and reject any decision remapped after sampling. */
int target_behavior_log_probability(const TargetSidecar* sidecar, int argmax, int slot_local,
    const FactorizedPolicySnapshot* actor_policy, const float* hidden,
    const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, double* log_probability, unsigned char* learned_slots);

#endif
