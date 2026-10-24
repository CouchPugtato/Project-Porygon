#include "tactical_scoring.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int check(int condition, const char* message) {
    if (!condition) fprintf(stderr, "test failed: %s\n", message);
    return condition;
}

static ObsPokemon pokemon(const char* first, const char* second) {
    ObsPokemon result;
    memset(&result, 0, sizeof(result));
    result.type1_id = type_id_from_name(first);
    result.type2_id = second ? type_id_from_name(second) : 0;
    result.type1_known_mode = result.type2_known_mode = 2;
    return result;
}

static int score_named_move(
    const char* name, const ObsPokemon* attacker, const ObsPokemon* target,
    TacticalAttackScore* score
) {
    MoveSemanticMetadata move;
    int id = move_id_from_name(name);
    return move_semantics_from_id(id, &move) && tactical_score_attack(
        &move, move_type_from_id(id), attacker, target, score);
}

static int test_target_matchups(void) {
    ObsPokemon attacker = pokemon("fire", NULL);
    ObsPokemon weak = pokemon("grass", "steel");
    ObsPokemon resistant = pokemon("water", "dragon");
    ObsPokemon ground = pokemon("ground", "flying");
    ObsPokemon ghost = pokemon("ghost", NULL);
    TacticalAttackScore strong_score, weak_score, score;
    int ok = 1;
    ok &= check(score_named_move("flamethrower", &attacker, &weak, &strong_score) &&
        score_named_move("flamethrower", &attacker, &resistant, &weak_score) &&
        strong_score.type_multiplier == 4.0f && weak_score.type_multiplier == 0.25f &&
        strong_score.expected_power > weak_score.expected_power,
        "Fire attack ranks the Grass/Steel target above the Water/Dragon target");
    ok &= check(score_named_move("thunderbolt", &attacker, &ground, &score) &&
        score.type_multiplier == 0.0f && score.expected_power == 0.0f,
        "Ground immunity overrides Flying weakness to Electric");
    ok &= check(score_named_move("tackle", &attacker, &ghost, &score) &&
        score.expected_power == 0.0f, "Normal attacks cannot damage Ghost targets");
    return ok;
}

static int test_accuracy_and_stab(void) {
    ObsPokemon attacker = pokemon("fire", NULL);
    ObsPokemon target = pokemon("normal", NULL);
    TacticalAttackScore score;
    int ok = 1;
    ok &= check(score_named_move("fireblast", &attacker, &target, &score) &&
        fabsf(score.hit_probability - 0.85f) < 1e-6f &&
        score.stab_multiplier == 1.5f && fabsf(score.expected_power - 140.25f) < 1e-4f,
        "Fire Blast combines base power, accuracy, and STAB");
    ok &= check(score_named_move("surf", &attacker, &target, &score) &&
        score.stab_multiplier == 1.0f, "Coverage attacks receive no unrelated STAB");
    attacker.type2_id = attacker.type1_id;
    ok &= check(score_named_move("flamethrower", &attacker, &target, &score) &&
        score.stab_multiplier == 1.5f, "Duplicate types do not multiply STAB twice");
    return ok;
}

static int test_unknown_information_and_unsupported_moves(void) {
    ObsPokemon attacker = pokemon("fire", NULL);
    ObsPokemon target = pokemon("grass", NULL);
    TacticalAttackScore score = {123.0f, 1.0f, 1.0f, 1.0f};
    int ok = 1;
    target.type1_known_mode = 0;
    ok &= check(!score_named_move("flamethrower", &attacker, &target, &score) &&
        score.expected_power == 0.0f,
        "A populated but unknown type is not treated as visible information");
    target.type1_known_mode = 2;
    target.type2_known_mode = 0;
    ok &= check(!score_named_move("flamethrower", &attacker, &target, &score),
        "Unknown secondary typing is not silently treated as monotype");
    target.type2_known_mode = 2;
    ok &= check(!score_named_move("protect", &attacker, &target, &score),
        "Status moves are outside the ordinary attack score");
    ok &= check(!score_named_move("seismictoss", &attacker, &target, &score),
        "Fixed-damage moves are not treated as fixed-base-power attacks");
    return ok;
}

static int test_visible_type_changes_and_non_mutation(void) {
    ObsPokemon attacker = pokemon("fire", NULL);
    ObsPokemon target = pokemon("water", NULL);
    ObsPokemon original_attacker = attacker;
    ObsPokemon original_target = target;
    TacticalAttackScore score;
    int ok = 1;
    target.species_id = species_id_from_name("ferrothorn");
    original_target = target;
    ok &= check(score_named_move("flamethrower", &attacker, &target, &score) &&
        score.type_multiplier == 0.5f,
        "Visible defensive typing overrides the species' usual typing");
    ok &= check(memcmp(&attacker, &original_attacker, sizeof(attacker)) == 0 &&
        memcmp(&target, &original_target, sizeof(target)) == 0,
        "Tactical scoring preserves the supplied battle observations");
    return ok;
}

int main(void) {
    if (!check(id_tables_init(), "load vocabulary and move metadata")) return 1;
    if (!test_target_matchups() || !test_accuracy_and_stab() ||
            !test_unknown_information_and_unsupported_moves() ||
            !test_visible_type_changes_and_non_mutation()) return 1;
    puts("tactical scoring tests passed");
    return 0;
}
