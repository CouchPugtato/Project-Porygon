#ifndef TARGET_AUDIT_H
#define TARGET_AUDIT_H
#include <stdio.h>
#include "tactical_audit.h"

void target_audit_write(FILE* out, const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int turn, unsigned char learned_slots, int slot_local);
#endif
