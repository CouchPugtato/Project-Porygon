#include "action_mapper.h"
#include "id_tables.h"
#include "request_parser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int assert_true(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "test failed: %s\n", message);
        return 0;
    }
    return 1;
}

static int test_normal_doubles_roundtrip(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\"},{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\"}],\"canTerastallize\":\"Grass\",\"trapped\":false},"
        "{\"moves\":[{\"id\":\"helpinghand\",\"pp\":32,\"maxpp\":32,\"target\":\"adjacentAllyOrSelf\"},{\"id\":\"suckerpunch\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\"}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Armarouge, L80, M\",\"condition\":\"100/100\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Dodrio, L85, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    ActionMask mask;
    char command[256];
    int slot0_has_action = 0;
    int slot1_has_action = 0;
    enum ObsAction action0 = OBS_A1_MOVE1;
    enum ObsAction action1 = OBS_A2_MOVE1;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 25, 1), "parse normal doubles request")) return 0;
    if (!assert_true(parsed_request_slot_choice_kind(&req, 0) == REQUEST_SLOT_MOVE_OR_SWITCH, "slot0 move/switch kind")) return 0;
    if (!assert_true(parsed_request_slot_choice_kind(&req, 1) == REQUEST_SLOT_MOVE_OR_SWITCH, "slot1 move/switch kind")) return 0;
    if (!assert_true(build_action_mask_from_request(&mask, &req), "build action mask")) return 0;
    if (!assert_true(mask.legal[OBS_A1_MOVE1] == 1, "slot0 protect legal")) return 0;
    if (!assert_true(mask.legal[OBS_A2_MOVE2] == 1, "slot1 sucker punch legal")) return 0;
    if (!assert_true(request_choice_to_command(&req, 1, OBS_A1_MOVE2, 1, OBS_A2_MOVE1, command, sizeof(command)), "map request choice to command")) return 0;
    if (!assert_true(strcmp(command, "/choose move 2 1, move 1 -2") == 0, "expected command shape")) return 0;
    if (!assert_true(command_to_request_choice(command, &req, &slot0_has_action, &action0, &slot1_has_action, &action1), "roundtrip command to actions")) return 0;
    if (!assert_true(slot0_has_action == 1 && action0 == OBS_A1_MOVE2, "roundtrip slot0")) return 0;
    if (!assert_true(slot1_has_action == 1 && action1 == OBS_A2_MOVE1, "roundtrip slot1")) return 0;
    return 1;
}

static int test_partial_forced_switch_roundtrip(void) {
    const char* json =
        "{\"forceSwitch\":[false,true],"
        "\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\"}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Armarouge, L80, M\",\"condition\":\"100/100\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Dodrio, L85, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    ActionMask mask;
    char command[256];
    int slot0_has_action = 0;
    int slot1_has_action = 0;
    enum ObsAction action0 = OBS_A1_MOVE1;
    enum ObsAction action1 = OBS_A2_MOVE1;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 42, 1), "parse forced-switch request")) return 0;
    if (!assert_true(parsed_request_slot_choice_kind(&req, 0) == REQUEST_SLOT_NONE, "slot0 no choice during partial forced switch")) return 0;
    if (!assert_true(parsed_request_slot_choice_kind(&req, 1) == REQUEST_SLOT_FORCE_SWITCH, "slot1 force switch kind")) return 0;
    if (!assert_true(build_action_mask_from_request(&mask, &req), "build forced-switch action mask")) return 0;
    if (!assert_true(mask.legal[OBS_A2_SWITCH3] == 1, "slot1 switch3 legal")) return 0;
    if (!assert_true(mask.legal[OBS_A2_MOVE1] == 0, "slot1 moves illegal during force switch")) return 0;
    if (!assert_true(request_choice_to_command(&req, 0, OBS_A1_MOVE1, 1, OBS_A2_SWITCH3, command, sizeof(command)), "map partial forced-switch command")) return 0;
    if (!assert_true(strcmp(command, "/choose pass, switch 3") == 0, "expected partial forced-switch command")) return 0;
    if (!assert_true(command_to_request_choice(command, &req, &slot0_has_action, &action0, &slot1_has_action, &action1), "roundtrip forced-switch command")) return 0;
    if (!assert_true(slot0_has_action == 0, "forced-switch slot0 absent")) return 0;
    if (!assert_true(slot1_has_action == 1 && action1 == OBS_A2_SWITCH3, "forced-switch slot1 reconstructed")) return 0;
    return 1;
}

static int test_switch_choices_are_request_relative(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"playrough\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Fairy\"},"
        "{\"moves\":[{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Poison\"}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Mabosstiff, L84, F\",\"condition\":\"272/272\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Rotom-Mow, L86\",\"condition\":\"226/226\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Slither Wing, L83\",\"condition\":\"277/277\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Samurott-Hisui, L83, M\",\"condition\":\"285/285\",\"active\":false},"
        "{\"ident\":\"p1: E\",\"details\":\"Grafaiai, L88, M\",\"condition\":\"254/254\",\"active\":false},"
        "{\"ident\":\"p1: F\",\"details\":\"Persian, L92, M\",\"condition\":\"269/269\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    ActionMask mask;
    char command[256];
    int slot0_has_action = 0;
    int slot1_has_action = 0;
    enum ObsAction action0 = OBS_A1_MOVE1;
    enum ObsAction action1 = OBS_A2_MOVE1;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 29, 1), "parse request-relative switch request")) return 0;
    if (!assert_true(build_action_mask_from_request(&mask, &req), "build request-relative switch mask")) return 0;
    if (!assert_true(mask.legal[OBS_A2_SWITCH2] == 0, "active slot switch2 illegal for slot1")) return 0;
    if (!assert_true(mask.legal[OBS_A2_SWITCH4] == 1, "absolute switch4 legal for slot1")) return 0;
    if (!assert_true(mask.legal[OBS_A2_SWITCH6] == 1, "absolute switch6 legal for slot1")) return 0;
    if (!assert_true(request_choice_to_command(&req, 1, OBS_A1_MOVE1, 1, OBS_A2_SWITCH4, command, sizeof(command)), "map move plus absolute switch")) return 0;
    if (!assert_true(strcmp(command, "/choose move 1 1, switch 4") == 0, "absolute switch command text")) return 0;
    if (!assert_true(command_to_request_choice(command, &req, &slot0_has_action, &action0, &slot1_has_action, &action1), "roundtrip request-relative switch command")) return 0;
    if (!assert_true(slot1_has_action == 1 && action1 == OBS_A2_SWITCH4, "reconstruct absolute switch action")) return 0;
    return 1;
}

static int test_single_living_active_uses_single_choice(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"dazzlinggleam\",\"pp\":16,\"maxpp\":16,\"target\":\"allAdjacentFoes\",\"disabled\":false},{\"id\":\"fireblast\",\"pp\":5,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Fire\"},"
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},{\"id\":\"iceshard\",\"pp\":47,\"maxpp\":48,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Water\"}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: Weezing\",\"details\":\"Weezing-Galar, L89, F\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p1: Mamoswine\",\"details\":\"Mamoswine, L82, F\",\"condition\":\"27/315\",\"active\":true},"
        "{\"ident\":\"p1: Regidrago\",\"details\":\"Regidrago, L74\",\"condition\":\"0 fnt\",\"active\":false},"
        "{\"ident\":\"p1: Krookodile\",\"details\":\"Krookodile, L80, F\",\"condition\":\"0 fnt\",\"active\":false},"
        "{\"ident\":\"p1: Vikavolt\",\"details\":\"Vikavolt, L84, F\",\"condition\":\"0 fnt\",\"active\":false},"
        "{\"ident\":\"p1: Flamigo\",\"details\":\"Flamigo, L84, M\",\"condition\":\"0 fnt\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    ActionMask mask;
    char command[256];

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 609, 1), "parse stale-active request")) return 0;
    if (!assert_true(req.active[0].fainted == 1, "slot0 inferred fainted from side state")) return 0;
    if (!assert_true(req.active[1].fainted == 0, "slot1 inferred alive from side state")) return 0;
    if (!assert_true(parsed_request_slot_needs_choice(&req, 0) == 0, "slot0 no choice when fainted")) return 0;
    if (!assert_true(parsed_request_slot_needs_choice(&req, 1) == 1, "slot1 still needs choice")) return 0;
    if (!assert_true(build_action_mask_from_request(&mask, &req), "build single-living-active mask")) return 0;
    if (!assert_true(mask.legal[OBS_A1_MOVE1] == 0, "slot0 moves illegal")) return 0;
    if (!assert_true(mask.legal[OBS_A2_MOVE1] == 1, "slot1 move legal")) return 0;
    if (!assert_true(request_choice_to_command(&req, 0, OBS_A1_MOVE1, 1, OBS_A2_MOVE2, command, sizeof(command)), "map single-living-active command")) return 0;
    if (!assert_true(strcmp(command, "/choose move 2 1") == 0, "single-living-active command shape")) return 0;
    return 1;
}

static int test_wait_request_needs_no_choice(void) {
    const char* json =
        "{\"wait\":true,"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Lucario, L86, M\",\"condition\":\"131/261\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Gengar, L84, M\",\"condition\":\"3/238 par\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Zamazenta, L72\",\"condition\":\"251/251\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Latias, L80, F\",\"condition\":\"89/259\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    ActionMask mask;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 169, 1), "parse wait request")) return 0;
    if (!assert_true(req.wait == 1, "wait flag parsed")) return 0;
    if (!assert_true(build_action_mask_from_request(&mask, &req), "build wait action mask")) return 0;
    if (!assert_true(parsed_request_slot_needs_choice(&req, 0) == 0, "wait slot0 no choice")) return 0;
    if (!assert_true(parsed_request_slot_needs_choice(&req, 1) == 0, "wait slot1 no choice")) return 0;
    return 1;
}

static int test_double_force_switch_one_bench_degrades_to_pass(void) {
    const char* json =
        "{\"forceSwitch\":[true,true],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Lucario, L86, M\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Zamazenta, L72\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Gouging Fire, L75\",\"condition\":\"221/281 par\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Latias, L80, F\",\"condition\":\"0 fnt\",\"active\":false},"
        "{\"ident\":\"p1: E\",\"details\":\"Gengar, L84, M\",\"condition\":\"0 fnt\",\"active\":false},"
        "{\"ident\":\"p1: F\",\"details\":\"Forretress, L90, F\",\"condition\":\"0 fnt\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    ActionMask mask;
    float policy[OBS_NUM_ACTIONS] = {0};
    ValidatedRequestChoice validated;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 251, 1), "parse one-bench double force-switch request")) return 0;
    if (!assert_true(build_action_mask_from_request(&mask, &req), "build one-bench force-switch mask")) return 0;
    if (!assert_true(mask.legal[OBS_A1_SWITCH3] == 1, "slot0 only live bench legal")) return 0;
    if (!assert_true(mask.legal[OBS_A2_SWITCH3] == 1, "slot1 only live bench legal")) return 0;
    policy[OBS_A1_SWITCH3] = 1.0f;
    policy[OBS_A2_SWITCH3] = 1.0f;
    if (!assert_true(validate_or_resample_request_choice(&req, &mask, policy, 1, OBS_A1_SWITCH3, 1, OBS_A2_SWITCH3, &validated), "degrade one-bench force-switch")) return 0;
    if (!assert_true(
            (strcmp(validated.command, "/choose switch 3, pass") == 0) ||
            (strcmp(validated.command, "/choose pass, switch 3") == 0),
            "one-bench force-switch command shape")) return 0;
    return 1;
}

int main(void) {
    if (!id_tables_init()) {
        fprintf(stderr, "failed to initialize id tables\n");
        return 1;
    }
    if (!test_normal_doubles_roundtrip()) {
        return 1;
    }
    if (!test_partial_forced_switch_roundtrip()) {
        return 1;
    }
    if (!test_switch_choices_are_request_relative()) {
        return 1;
    }
    if (!test_single_living_active_uses_single_choice()) {
        return 1;
    }
    if (!test_wait_request_needs_no_choice()) {
        return 1;
    }
    if (!test_double_force_switch_one_bench_degrades_to_pass()) {
        return 1;
    }
    printf("legality tests passed\n");
    return 0;
}
