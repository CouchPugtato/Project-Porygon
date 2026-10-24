#include "tactical_scoring.h"
#include "tactical_policy.h"
#include "action_semantics.h"

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

static int decision_fixture(Observation* observation, ParsedRequest* request) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"flamethrower\",\"pp\":10,\"maxpp\":16,\"target\":\"normal\"},"
        "{\"id\":\"protect\",\"pp\":10,\"maxpp\":16,\"target\":\"self\"}],\"canTerastallize\":\"Fire\"},"
        "{\"moves\":[{\"id\":\"tackle\",\"pp\":10,\"maxpp\":16,\"target\":\"normal\"},"
        "{\"id\":\"helpinghand\",\"pp\":10,\"maxpp\":16,\"target\":\"adjacentAlly\"}],\"canTerastallize\":\"Normal\"}],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Arcanine, L80\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Snorlax, L80\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Jolteon, L80\",\"condition\":\"100/100\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Tornadus, L80\",\"condition\":\"100/100\",\"active\":false}]}}";
    memset(observation, 0, sizeof(*observation));
    parsed_request_init(request);
    if (!parse_request_payload(request, json, 1, 1)) return 0;
    observation->self_team[0] = pokemon("fire", NULL);
    observation->self_team[1] = pokemon("normal", NULL);
    observation->self_team[2] = pokemon("electric", NULL);
    observation->self_team[3] = pokemon("flying", NULL);
    observation->opp_team[4] = pokemon("grass", "steel");
    observation->opp_team[5] = pokemon("water", "dragon");
    observation->self_team[0].active = observation->self_team[1].active = 1;
    observation->self_team[0].active_slot = 1;
    observation->self_team[1].active_slot = 2;
    observation->opp_team[4].active = observation->opp_team[5].active = 1;
    observation->opp_team[4].active_slot = 1;
    observation->opp_team[5].active_slot = 2;
    return 1;
}

static int test_joint_targets_and_non_mutation(void) {
    Observation observation, before;
    ParsedRequest request, original;
    TacticalDecision decision;
    char command[256];
    int ok = check(decision_fixture(&observation, &request), "parse tactical decision fixture");
    if (!ok) return 0;
    before = observation;
    original = request;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_move_index == 0 && decision.choice.slot1_move_index == 0 &&
        decision.choice.slot0_target_index == FACTORIZED_TARGET_FOE_LEFT &&
        decision.choice.slot1_target_index == FACTORIZED_TARGET_FOE_RIGHT &&
        !decision.choice.slot0_use_tera && !decision.choice.slot1_use_tera &&
        factorized_choice_to_command(&request, &decision.choice, command, sizeof(command)) &&
        strcmp(command, "/choose move 1 1, move 1 2") == 0,
        "Joint selection chooses useful foe targets using active positions, not roster indices");
    ok &= check(memcmp(&before, &observation, sizeof(observation)) == 0 &&
        memcmp(&original, &request, sizeof(request)) == 0,
        "Joint enumeration preserves the battle observation and request");
    observation.opp_team[4].fainted = 1;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_target_index == FACTORIZED_TARGET_FOE_RIGHT &&
        decision.choice.slot1_target_index == FACTORIZED_TARGET_FOE_RIGHT,
        "Explicit targeting excludes a fainted foe");
    return ok;
}

static int reward_tera(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    (void)observation;
    (void)request;
    (void)context;
    memset(evaluation, 0, sizeof(*evaluation));
    evaluation->score = 10.5f * choice->slot0_use_tera + 10.0f * choice->slot1_use_tera;
    return 1;
}

static int test_shared_tera_and_required_ally_target(void) {
    Observation observation;
    ParsedRequest request;
    TacticalDecision decision;
    char command[256];
    int ok = check(decision_fixture(&observation, &request), "parse shared-Tera fixture");
    if (!ok) return 0;
    ok &= check(tactical_select_choice(&observation, &request, reward_tera, NULL, &decision) &&
        decision.choice.slot0_use_tera && !decision.choice.slot1_use_tera &&
        decision.evaluation.score == 10.5f,
        "Pair constraints reject double Tera even when the evaluator rewards it");
    request.active[0].move_disabled[0] = 1;
    request.active[1].move_disabled[0] = 1;
    request.slot_can_switch[0] = request.slot_can_switch[1] = 0;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot1_target_index == FACTORIZED_TARGET_ALLY &&
        factorized_choice_to_command(&request, &decision.choice, command, sizeof(command)) &&
        strcmp(command, "/choose move 2, move 2 -1") == 0 &&
        decision.evaluation.unscored_moves == 2,
        "Helping Hand retains its required ally target and reports unmodeled status value");
    return ok;
}

static int test_spread_power_and_switching_partner(void) {
    Observation observation;
    ParsedRequest request;
    TacticalDecision decision;
    int ok = check(decision_fixture(&observation, &request), "parse spread fixture");
    if (!ok) return 0;
    observation.self_team[0] = pokemon("ground", NULL);
    observation.self_team[0].active = observation.self_team[0].active_slot = 1;
    observation.self_team[1] = pokemon("electric", NULL);
    observation.self_team[1].active = 1;
    observation.self_team[1].active_slot = 2;
    observation.opp_team[4] = pokemon("electric", NULL);
    observation.opp_team[5] = pokemon("electric", NULL);
    observation.opp_team[4].active = observation.opp_team[5].active = 1;
    observation.opp_team[4].active_slot = 1;
    observation.opp_team[5].active_slot = 2;
    request.active[0].move_id[0] = move_id_from_name("earthquake");
    request.active[0].move_target[0] = REQUEST_TARGET_ALL_ADJACENT;
    request.active[0].move_id[1] = move_id_from_name("earthpower");
    request.active[0].move_target[1] = REQUEST_TARGET_NORMAL;
    request.active[1].move_disabled[0] = 1;
    request.slot_can_switch[0] = request.slot_can_switch[1] = 0;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_move_index == 1,
        "Partner damage makes Earth Power preferable to Earthquake beside an Electric ally");
    request.slot_can_move[1] = 0;
    request.slot_can_switch[1] = 1;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_move_index == 0 &&
        decision.choice.slot1_kind == FACTORIZED_ACTION_SWITCH && decision.choice.slot1_switch_index == 3 &&
        decision.evaluation.friendly_power == 0 && decision.evaluation.opponent_power == 450.0f,
        "Joint evaluation accounts for the incoming partner's Ground immunity and spread reduction");
    return ok;
}

static int test_forced_switches_and_passes(void) {
    Observation observation;
    ParsedRequest request;
    TacticalDecision decision;
    char command[256];
    int ok = check(decision_fixture(&observation, &request), "parse replacement fixture");
    if (!ok) return 0;
    request.forced_switch_any = 1;
    request.force_switch[0] = request.force_switch[1] = 1;
    request.choice_kind[0] = request.choice_kind[1] = REQUEST_SLOT_FORCE_SWITCH;
    request.slot_can_move[0] = request.slot_can_move[1] = 0;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_has_action && decision.choice.slot1_has_action &&
        decision.choice.slot0_switch_index != decision.choice.slot1_switch_index,
        "Two replacements use distinct bench Pokemon");
    request.switch_available[3] = 0;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_has_action + decision.choice.slot1_has_action == 1 &&
        factorized_choice_to_command(&request, &decision.choice, command, sizeof(command)) &&
        strcmp(command, "/choose switch 3, pass") == 0,
        "One remaining bench Pokemon fills one forced slot without a duplicate switch");
    request.force_switch[0] = request.slot_needs_choice[0] = 0;
    request.choice_kind[0] = REQUEST_SLOT_NONE;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        !decision.choice.slot0_has_action && decision.choice.slot1_has_action &&
        factorized_choice_to_command(&request, &decision.choice, command, sizeof(command)) &&
        strcmp(command, "/choose pass, switch 3") == 0,
        "A right-slot-only replacement preserves the left pass");
    return ok;
}

static int nonfinite_evaluator(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    (void)observation;
    (void)request;
    (void)choice;
    (void)context;
    evaluation->score = NAN;
    return 1;
}

static int test_selector_respects_traps_wait_and_finite_scores(void) {
    Observation observation;
    ParsedRequest request;
    TacticalDecision decision;
    int ok = check(decision_fixture(&observation, &request), "parse constrained request fixture");
    if (!ok) return 0;
    /* slot_can_switch is the parser's resolved trap restriction. */
    request.slot_can_switch[0] = request.slot_can_switch[1] = 0;
    request.active[0].trapped = request.active[1].trapped = 1;
    request.active[0].move_disabled[0] = request.active[1].move_disabled[0] = 1;
    ok &= check(tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision) &&
        decision.choice.slot0_kind == FACTORIZED_ACTION_MOVE && decision.choice.slot0_move_index == 1 &&
        decision.choice.slot1_kind == FACTORIZED_ACTION_MOVE && decision.choice.slot1_move_index == 1,
        "Trapped Pokemon use the remaining enabled moves instead of switching");
    ok &= check(!tactical_select_choice(&observation, &request, nonfinite_evaluator, NULL, &decision) &&
        !decision.choice.slot0_has_action && !decision.choice.slot1_has_action,
        "Non-finite evaluations cannot become selected actions");
    request.wait = 1;
    ok &= check(!tactical_select_choice(&observation, &request, tactical_evaluate_power, NULL, &decision),
        "Wait requests do not produce tactical choices");
    return ok;
}

static int test_all_adjacent_parser_and_legacy_encoding(void) {
    Observation observation;
    ParsedRequest request;
    FactorizedActionChoice choice;
    float current[ACTION_SEMANTIC_DIM], legacy[ACTION_SEMANTIC_DIM];
    const char* json = "{\"active\":[{\"moves\":[{\"id\":\"earthquake\",\"pp\":10,\"target\":\"allAdjacent\"}]}]}";
    int ok = check(decision_fixture(&observation, &request), "parse semantic compatibility fixture");
    if (!ok) return 0;
    ok &= check(parse_request_payload(&request, json, 2, 1) &&
        request.active[0].move_target[0] == REQUEST_TARGET_ALL_ADJACENT,
        "The parser distinguishes allAdjacent from unknown targeting");
    factorized_action_choice_from_flat_actions(&choice, OBS_A1_MOVE1, -1);
    ok &= check(action_semantics_build(current, ACTION_SEMANTIC_DIM, &observation, &request, &choice),
        "Build semantics with the new parsed spread target");
    request.active[0].move_target[0] = REQUEST_TARGET_UNKNOWN;
    ok &= check(action_semantics_build(legacy, ACTION_SEMANTIC_DIM, &observation, &request, &choice) &&
        memcmp(current, legacy, sizeof(current)) == 0,
        "Parsing allAdjacent preserves the previous neural input encoding");
    return ok;
}

int main(void) {
    if (!check(id_tables_init(), "load vocabulary and move metadata")) return 1;
    if (!test_target_matchups() || !test_accuracy_and_stab() ||
            !test_unknown_information_and_unsupported_moves() ||
            !test_visible_type_changes_and_non_mutation() ||
            !test_joint_targets_and_non_mutation() ||
            !test_shared_tera_and_required_ally_target() ||
            !test_spread_power_and_switching_partner() ||
            !test_forced_switches_and_passes() ||
            !test_selector_respects_traps_wait_and_finite_scores() ||
            !test_all_adjacent_parser_and_legacy_encoding()) return 1;
    puts("tactical scoring tests passed");
    return 0;
}
