#ifndef ACTION_OPTIMIZER_STATE_H
#define ACTION_OPTIMIZER_STATE_H

#include "gru_model.h"

/* Separate native-format state; existing .chk layouts are unchanged. The
 * caller supplies the trusted behavior-bundle SHA256 pin and keeps the actor
 * checkpoint paired with this file. Only completed minibatches can be saved.
 * Failure reasons are stable string literals; load never mutates the actor. */
int action_optimizer_state_save(const char* path, const GruActionHeadOptimizer* optimizer,
    const GruModel* reference, const char* bundle_sha256, const char** failure_reason);
GruActionHeadOptimizer* action_optimizer_state_load(const char* path, GruModel* actor,
    const GruModel* reference, const char* bundle_sha256, const char** failure_reason);

#endif
