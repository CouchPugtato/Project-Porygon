#ifndef TACTICAL_SCORING_H
#define TACTICAL_SCORING_H

#include "id_tables.h"
#include "observation.h"

typedef struct {
    float expected_power;
    float hit_probability;
    float stab_multiplier;
    float type_multiplier;
} TacticalAttackScore;

/* Ordinary fixed-power attacks only. This score is in effective base-power
 * units, not HP damage. Callers must handle move exceptions, Tera, abilities,
 * stats, field effects, targets, and legality before using it to choose actions.
 * Returns zero when the move or the supplied type information is unsupported.
 */
int tactical_score_attack(
    const MoveSemanticMetadata* move,
    int move_type_id,
    const ObsPokemon* attacker,
    const ObsPokemon* target,
    TacticalAttackScore* score
);

#endif
