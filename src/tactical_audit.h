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
int tactical_audit_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit);
/* Diagnostic-only current-board retargeting; never used to choose live actions. */
int tactical_diagnostic_power(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context);
int tactical_diagnostic_slot_power(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int slot, TacticalChoiceEvaluation* evaluation);
int tactical_diagnostic_targets(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalAudit* audit);
/* Returns one only for a fully scored strict improvement; changes target
 * indices only. Unsupported, forced, and tied choices remain byte-identical. */
int tactical_retarget_choice(const Observation* observation, const ParsedRequest* request,
    FactorizedActionChoice* choice);
void tactical_audit_write_json(FILE* out, const Observation* observation,
    const ParsedRequest* request, const FactorizedActionChoice* choice);

#endif
