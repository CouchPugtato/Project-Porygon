#ifndef POLICY_DECODING_H
#define POLICY_DECODING_H

#include "gru_model.h"

/* First legal maximum on ties; no RNG or policy mutation. */
int policy_decode_target_argmax(const float policy[4], unsigned char legal);

/* Conditional behavior after masking/decoding, not the pre-decoder softmax. */
int policy_target_distribution(const float policy[4], unsigned char legal, int argmax, float distribution[4]);

/* Unfloored likelihood under the supplied decoded distributions. A legal
 * zero-probability choice returns -INFINITY rather than an artificial floor. */
int policy_choice_log_probability(const FactorizedPolicySnapshot* policy, const unsigned char* legal,
    const FactorizedActionChoice* choice, double* log_probability);

/* Maximizes the full legal turn probability, including conditional targets.
 * Does not advance the model, consume RNG, or change the supplied snapshot. */
int policy_decode_greedy(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedPolicySnapshot* policy, FactorizedActionChoice* choice
);

#endif
