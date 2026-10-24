#include "tactical_scoring.h"

#include <string.h>

typedef struct {
    const char* attack;
    const char* super_effective[6];
    const char* resisted[8];
    const char* immune[2];
} TypeMatchup;

/* Generation 9 ordinary type matchups, from Pokemon Showdown data/typechart.ts.
 * Names avoid coupling this table to the vocabulary's numeric type order.
 */
static const TypeMatchup type_matchups[] = {
    {"normal", {NULL}, {"rock", "steel"}, {"ghost"}},
    {"fire", {"grass", "ice", "bug", "steel"},
        {"fire", "water", "rock", "dragon"}, {NULL}},
    {"water", {"fire", "ground", "rock"}, {"water", "grass", "dragon"}, {NULL}},
    {"electric", {"water", "flying"}, {"electric", "grass", "dragon"}, {"ground"}},
    {"grass", {"water", "ground", "rock"},
        {"fire", "grass", "poison", "flying", "bug", "dragon", "steel"}, {NULL}},
    {"ice", {"grass", "ground", "flying", "dragon"},
        {"fire", "water", "ice", "steel"}, {NULL}},
    {"fighting", {"normal", "ice", "rock", "dark", "steel"},
        {"poison", "flying", "psychic", "bug", "fairy"}, {"ghost"}},
    {"poison", {"grass", "fairy"}, {"poison", "ground", "rock", "ghost"}, {"steel"}},
    {"ground", {"fire", "electric", "poison", "rock", "steel"},
        {"grass", "bug"}, {"flying"}},
    {"flying", {"grass", "fighting", "bug"}, {"electric", "rock", "steel"}, {NULL}},
    {"psychic", {"fighting", "poison"}, {"psychic", "steel"}, {"dark"}},
    {"bug", {"grass", "psychic", "dark"},
        {"fire", "fighting", "poison", "flying", "ghost", "steel", "fairy"}, {NULL}},
    {"rock", {"fire", "ice", "flying", "bug"}, {"fighting", "ground", "steel"}, {NULL}},
    {"ghost", {"psychic", "ghost"}, {"dark"}, {"normal"}},
    {"dragon", {"dragon"}, {"steel"}, {"fairy"}},
    {"dark", {"psychic", "ghost"}, {"fighting", "dark", "fairy"}, {NULL}},
    {"steel", {"ice", "rock", "fairy"}, {"fire", "water", "electric", "steel"}, {NULL}},
    {"fairy", {"fighting", "dragon", "dark"}, {"fire", "poison", "steel"}, {NULL}}
};

static int contains_type(const char* const* types, const char* type) {
    size_t i;
    for (i = 0; types[i]; ++i) {
        if (strcmp(types[i], type) == 0) return 1;
    }
    return 0;
}

static const TypeMatchup* find_matchup(int type_id) {
    const char* name = type_name_from_id(type_id);
    size_t i;
    if (!name) return NULL;
    for (i = 0; i < sizeof(type_matchups) / sizeof(type_matchups[0]); ++i) {
        if (strcmp(type_matchups[i].attack, name) == 0) return &type_matchups[i];
    }
    return NULL;
}

static int has_known_types(const ObsPokemon* pokemon) {
    return pokemon->type1_known_mode > 0 && pokemon->type2_known_mode > 0 &&
        find_matchup(pokemon->type1_id) &&
        (pokemon->type2_id == 0 || find_matchup(pokemon->type2_id));
}

static float single_type_multiplier(const TypeMatchup* matchup, int defense_type) {
    const char* name = type_name_from_id(defense_type);
    if (contains_type(matchup->immune, name)) return 0.0f;
    if (contains_type(matchup->super_effective, name)) return 2.0f;
    if (contains_type(matchup->resisted, name)) return 0.5f;
    return 1.0f;
}

int tactical_score_attack(
    const MoveSemanticMetadata* move,
    int move_type_id,
    const ObsPokemon* attacker,
    const ObsPokemon* target,
    TacticalAttackScore* score
) {
    const TypeMatchup* matchup;
    if (!score) return 0;
    memset(score, 0, sizeof(*score));
    if (!move || !attacker || !target || !move->damaging ||
            (move->category != 2 && move->category != 3) ||
            move->base_power <= 0 || move->accuracy <= 0 || move->accuracy > 100 ||
            !has_known_types(attacker) || !has_known_types(target)) return 0;
    matchup = find_matchup(move_type_id);
    if (!matchup) return 0;
    score->hit_probability = (float)move->accuracy / 100.0f;
    score->stab_multiplier =
        attacker->type1_id == move_type_id || attacker->type2_id == move_type_id
            ? 1.5f : 1.0f;
    score->type_multiplier = single_type_multiplier(matchup, target->type1_id);
    if (target->type2_id != 0 && target->type2_id != target->type1_id) {
        score->type_multiplier *= single_type_multiplier(matchup, target->type2_id);
    }
    score->expected_power = (float)move->base_power * score->hit_probability *
        score->stab_multiplier * score->type_multiplier;
    return 1;
}
