#ifndef TACTICAL_AUDIT_H
#define TACTICAL_AUDIT_H

#include <stdio.h>
#include "tactical_policy.h"

typedef struct {
    TacticalChoiceEvaluation selected;
    TacticalDecision teacher;
    TacticalDecision retargeted;
    int teacher_available;
    int retargeted_available;
} TacticalAudit;

/* Passive comparison: no model step, RNG use, or choice replacement. */
int tactical_audit_choice(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit);
/* Returns one only for a fully scored strict improvement; changes target
 * indices only. Unsupported, forced, and tied choices remain byte-identical. */
int tactical_retarget_choice(const Observation* observation, const ParsedRequest* request,
    FactorizedActionChoice* choice);
void tactical_audit_write_json(FILE* out, const Observation* observation,
    const ParsedRequest* request, const FactorizedActionChoice* choice);

#endif
