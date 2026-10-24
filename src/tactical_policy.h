#ifndef TACTICAL_POLICY_H
#define TACTICAL_POLICY_H

#include "action_mapper.h"

typedef struct {
    float score;
    float opponent_power;
    float friendly_power;
    size_t unscored_moves;
} TacticalChoiceEvaluation;

typedef struct {
    FactorizedActionChoice choice;
    TacticalChoiceEvaluation evaluation;
    size_t evaluated_choices;
} TacticalDecision;

typedef int (*TacticalChoiceEvaluator)(
    const Observation* observation,
    const ParsedRequest* request,
    const FactorizedActionChoice* choice,
    TacticalChoiceEvaluation* evaluation,
    void* context
);

/* Enumerates slot actions and explicit targets, then applies shared pair
 * constraints. Equal scores keep the first choice for reproducibility.
 * Returns zero for wait/preview requests or when no choice can be evaluated. */
int tactical_select_choice(
    const Observation* observation,
    const ParsedRequest* request,
    TacticalChoiceEvaluator evaluate,
    void* context,
    TacticalDecision* decision
);

/* Evaluates choices emitted by tactical_select_choice. Unmodeled moves
 * contribute zero and are counted explicitly. This initial heuristic omits
 * status value, switch value, abilities, protection, and Tera attack scoring.
 * It is not a calibrated damage or battle-value estimate. */
int tactical_evaluate_power(
    const Observation* observation,
    const ParsedRequest* request,
    const FactorizedActionChoice* choice,
    TacticalChoiceEvaluation* evaluation,
    void* context
);

#endif
