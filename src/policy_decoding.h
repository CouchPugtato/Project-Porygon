#ifndef POLICY_DECODING_H
#define POLICY_DECODING_H

#include "gru_model.h"

/* First legal maximum on ties; no RNG or policy mutation. */
int policy_decode_target_argmax(const float policy[4], unsigned char legal);

/* Maximizes the full legal turn probability, including conditional targets.
 * Does not advance the model, consume RNG, or change the supplied snapshot. */
int policy_decode_greedy(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedPolicySnapshot* policy, FactorizedActionChoice* choice
);

#endif
