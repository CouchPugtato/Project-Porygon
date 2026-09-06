#include "env_session.h"
#include "id_tables.h"
#include "observation_builder.h"
#include "checkpoint.h"
#include "gru_trainer.h"
#include "learning_diagnostics.h"
#include "policy_evaluation.h"
#include "validation_split.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define TEST_MKDIR(path) _mkdir(path)
#define TEST_RMDIR(path) _rmdir(path)
#else
#include <sys/stat.h>
#include <unistd.h>
#define TEST_MKDIR(path) mkdir(path, 0777)
#define TEST_RMDIR(path) rmdir(path)
#endif

#define TEST_JSONL_LINE_MAX 65536
#define TEST_CAPTURE_PATH "matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl"
#define TEST_RANDOM_CAPTURE_PATH "matches/runs/run_0011_new_random/run_0011_new_random_raw.jsonl"

static int assert_true(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "test failed: %s\n", message);
        return 0;
    }
    return 1;
}

static int nearly_equal(double actual, double expected, double tolerance) {
    return fabs(actual - expected) <= tolerance;
}

static RawPokemon* find_self(RawBattleState* state, const char* ident) {
    int i;
    const char* target_name = strstr(ident, ": ");
    target_name = target_name ? (target_name + 2) : ident;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        const char* current_name = strstr(state->self_team[i].ident, ": ");
        current_name = current_name ? (current_name + 2) : state->self_team[i].ident;
        if (strcmp(current_name, target_name) == 0) {
            return &state->self_team[i];
        }
    }
    return NULL;
}

static RawPokemon* find_opp(RawBattleState* state, const char* ident) {
    int i;
    const char* target_name = strstr(ident, ": ");
    target_name = target_name ? (target_name + 2) : ident;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        const char* current_name = strstr(state->opp_team[i].ident, ": ");
        current_name = current_name ? (current_name + 2) : state->opp_team[i].ident;
        if (strcmp(current_name, target_name) == 0) {
            return &state->opp_team[i];
        }
    }
    return NULL;
}

static void apply_event_lines(RawBattleState* state, const char* const* lines, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) {
        raw_battle_state_update_from_event_line(state, lines[i]);
    }
}

typedef struct {
    RawBattleState state;
    int saw_battle_start;
    int saw_terminal;
    int terminal_is_win;
    int saw_disconnect_loss;
    int saw_forfeit;
    int event_count;
    int request_count;
} CaptureReplayResult;

static int extract_json_string_value(const char* line, const char* key, char* out, size_t out_len) {
    char needle[64];
    const char* p;
    size_t i = 0;
    if (!line || !key || !out || out_len == 0) {
        return 0;
    }
    snprintf(needle, sizeof(needle), "\"%s\":\"", key);
    p = strstr(line, needle);
    if (!p) {
        return 0;
    }
    p += strlen(needle);
    while (*p && *p != '"' && i + 1 < out_len) {
        if (*p == '\\') {
            ++p;
            if (*p == 'u') {
                out[i++] = '?';
                if (p[1]) ++p;
                if (p[1]) ++p;
                if (p[1]) ++p;
                if (p[1]) ++p;
                ++p;
                continue;
            }
            if (*p == 'n') out[i++] = '\n';
            else if (*p == 'r') out[i++] = '\r';
            else if (*p == 't') out[i++] = '\t';
            else if (*p == '"' || *p == '\\' || *p == '/') out[i++] = *p;
            else out[i++] = *p ? *p : '\\';
            if (*p) {
                ++p;
            }
            continue;
        }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return 1;
}

static int extract_json_int_value(const char* line, const char* key, int* out) {
    char needle[64];
    const char* p;
    if (!line || !key || !out) {
        return 0;
    }
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    p = strstr(line, needle);
    if (!p) {
        return 0;
    }
    p += strlen(needle);
    *out = atoi(p);
    return 1;
}

static int extract_json_object_value(const char* line, const char* key, char* out, size_t out_len) {
    char needle[64];
    const char* p;
    const char* start;
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    size_t len;
    if (!line || !key || !out || out_len == 0) {
        return 0;
    }
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    p = strstr(line, needle);
    if (!p) {
        return 0;
    }
    p += strlen(needle);
    while (*p == ' ' || *p == '\t') {
        ++p;
    }
    if (*p != '{') {
        return 0;
    }
    start = p;
    while (*p) {
        if (escaped) {
            escaped = 0;
        } else if (*p == '\\' && in_string) {
            escaped = 1;
        } else if (*p == '"') {
            in_string = !in_string;
        } else if (!in_string) {
            if (*p == '{') {
                ++depth;
            } else if (*p == '}') {
                --depth;
                if (depth == 0) {
                    ++p;
                    break;
                }
            }
        }
        ++p;
    }
    if (depth != 0) {
        return 0;
    }
    len = (size_t)(p - start);
    if (len + 1 > out_len) {
        return 0;
    }
    memcpy(out, start, len);
    out[len] = '\0';
    return 1;
}

static int replay_capture_battle_from_path(const char* path, const char* battle_id, int stop_after_turn_request, CaptureReplayResult* out) {
    FILE* fp;
    char line[TEST_JSONL_LINE_MAX];
    char line_battle_id[256];
    char type[64];
    if (!path || !battle_id || !out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    raw_battle_state_init(&out->state, 1);
    fp = fopen(path, "r");
    if (!fp) {
        return 0;
    }
    while (fgets(line, sizeof(line), fp)) {
        char event_line[4096];
        char payload[32768];
        char result[32];
        ParsedRequest req;
        int request_id = 0;
        if (!extract_json_string_value(line, "battle_id", line_battle_id, sizeof(line_battle_id))) {
            continue;
        }
        if (strcmp(line_battle_id, battle_id) != 0) {
            continue;
        }
        if (!extract_json_string_value(line, "type", type, sizeof(type))) {
            continue;
        }
        if (strcmp(type, "battle_start") == 0) {
            out->saw_battle_start = 1;
            continue;
        }
        if (strcmp(type, "event") == 0) {
            if (!extract_json_string_value(line, "line", event_line, sizeof(event_line))) {
                fclose(fp);
                return 0;
            }
            if (strstr(event_line, "lost due to inactivity") != NULL) {
                out->saw_disconnect_loss = 1;
            }
            if (strstr(event_line, "forfeited") != NULL) {
                out->saw_forfeit = 1;
            }
            raw_battle_state_update_from_event_line(&out->state, event_line);
            out->event_count += 1;
            continue;
        }
        if (strcmp(type, "request") == 0) {
            if (!extract_json_int_value(line, "request_id", &request_id)) {
                fclose(fp);
                return 0;
            }
            if (!extract_json_object_value(line, "payload", payload, sizeof(payload))) {
                fclose(fp);
                return 0;
            }
            parsed_request_init(&req);
            if (!parse_request_payload(&req, payload, request_id, 1)) {
                fclose(fp);
                return 0;
            }
            if (!raw_battle_state_update_from_request(&out->state, &req)) {
                fclose(fp);
                return 0;
            }
            out->request_count += 1;
            if (stop_after_turn_request > 0 && out->state.turn_number >= stop_after_turn_request) {
                fclose(fp);
                return 1;
            }
            continue;
        }
        if (strcmp(type, "terminal") == 0) {
            out->saw_terminal = 1;
            if (!extract_json_string_value(line, "result", result, sizeof(result))) {
                fclose(fp);
                return 0;
            }
            out->terminal_is_win = strcmp(result, "win") == 0;
            fclose(fp);
            return 1;
        }
    }
    fclose(fp);
    return out->saw_battle_start != 0;
}

static int replay_capture_battle(const char* battle_id, int stop_after_turn_request, CaptureReplayResult* out) {
    return replay_capture_battle_from_path(TEST_CAPTURE_PATH, battle_id, stop_after_turn_request, out);
}

static int state_slot_maps_consistent(const RawBattleState* state) {
    int slot;
    if (!state) {
        return 0;
    }
    for (slot = 0; slot < 2; ++slot) {
        int self_idx = state->self_active_slot_to_team_index[slot];
        int opp_idx = state->opp_active_slot_to_team_index[slot];
        if (self_idx >= 0) {
            if (!state->self_team[self_idx].active || state->self_team[self_idx].active_slot != slot + 1) {
                return 0;
            }
        }
        if (opp_idx >= 0) {
            if (!state->opp_team[opp_idx].active || state->opp_team[opp_idx].active_slot != slot + 1) {
                return 0;
            }
        }
    }
    return 1;
}

static int team_faint_hp_consistent(const RawPokemon* team) {
    int i;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].fainted && team[i].current_hp != 0) {
            return 0;
        }
        if (team[i].active && team[i].fainted && team[i].current_hp > 0) {
            return 0;
        }
    }
    return 1;
}

static int team_remaining_matches(const RawPokemon* team, int remaining) {
    int i;
    int count = 0;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (!team[i].fainted) {
            count += 1;
        }
    }
    return count == remaining;
}

static int assert_capture_replay_sane(const CaptureReplayResult* replay, const char* label) {
    char message[160];
    if (!replay) {
        return 0;
    }
    snprintf(message, sizeof(message), "%s keeps slot maps consistent", label);
    if (!assert_true(state_slot_maps_consistent(&replay->state), message)) return 0;
    snprintf(message, sizeof(message), "%s keeps self faint hp consistent", label);
    if (!assert_true(team_faint_hp_consistent(replay->state.self_team), message)) return 0;
    snprintf(message, sizeof(message), "%s keeps opponent faint hp consistent", label);
    if (!assert_true(team_faint_hp_consistent(replay->state.opp_team), message)) return 0;
    snprintf(message, sizeof(message), "%s self remaining count stays synchronized", label);
    if (!assert_true(team_remaining_matches(replay->state.self_team, replay->state.self_side.remaining_pokemon), message)) return 0;
    snprintf(message, sizeof(message), "%s opponent remaining count stays synchronized", label);
    if (!assert_true(team_remaining_matches(replay->state.opp_team, replay->state.opp_side.remaining_pokemon), message)) return 0;
    return 1;
}

static int battle_id_seen(char ids[][256], int count, const char* battle_id) {
    int i;
    for (i = 0; i < count; ++i) {
        if (strcmp(ids[i], battle_id) == 0) {
            return 1;
        }
    }
    return 0;
}

static int run_batch_replay_mode(const char* path) {
    FILE* fp;
    char line[TEST_JSONL_LINE_MAX];
    char battle_id[256];
    char type[64];
    char result[32];
    char seen_ids[512][256];
    int seen_count = 0;
    int terminal_count = 0;
    int clean_count = 0;
    int pass_count = 0;
    int skipped_disconnect = 0;

    if (!path) {
        fprintf(stderr, "batch replay requires a jsonl path\n");
        return 1;
    }
    fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "failed to open batch replay file: %s\n", path);
        return 1;
    }
    while (fgets(line, sizeof(line), fp)) {
        CaptureReplayResult replay;
        char label[320];
        if (!extract_json_string_value(line, "type", type, sizeof(type))) {
            continue;
        }
        if (strcmp(type, "terminal") != 0) {
            continue;
        }
        if (!extract_json_string_value(line, "battle_id", battle_id, sizeof(battle_id))) {
            continue;
        }
        if (battle_id_seen(seen_ids, seen_count, battle_id)) {
            continue;
        }
        if (!extract_json_string_value(line, "result", result, sizeof(result))) {
            continue;
        }
        if (strcmp(result, "win") != 0 && strcmp(result, "loss") != 0) {
            continue;
        }
        if (seen_count < (int)(sizeof(seen_ids) / sizeof(seen_ids[0]))) {
            strncpy(seen_ids[seen_count], battle_id, sizeof(seen_ids[seen_count]) - 1);
            seen_ids[seen_count][sizeof(seen_ids[seen_count]) - 1] = '\0';
            ++seen_count;
        }
        ++terminal_count;
        if (!replay_capture_battle_from_path(path, battle_id, 0, &replay)) {
            fclose(fp);
            fprintf(stderr, "batch replay failed to parse battle: %s\n", battle_id);
            return 1;
        }
        if (replay.saw_disconnect_loss) {
            ++skipped_disconnect;
            continue;
        }
        ++clean_count;
        snprintf(label, sizeof(label), "batch replay %s", battle_id);
        if (!assert_capture_replay_sane(&replay, label)) {
            fclose(fp);
            return 1;
        }
        ++pass_count;
    }
    fclose(fp);
    printf("batch replay passed: %d clean battles (%d terminal seen, %d skipped disconnect) from %s\n",
        pass_count, terminal_count, skipped_disconnect, path);
    return clean_count == pass_count ? 0 : 1;
}

static int test_request_reconciliation_preserves_identity(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false},"
        "{\"moves\":[{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: C\",\"details\":\"Armarouge, L80, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":false},"
        "{\"ident\":\"p1: D\",\"details\":\"Dodrio, L85, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* a;
    RawPokemon* b;
    RawPokemon* c;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: B|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: C|Armarouge, L80, M|100/100");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 7, 1), "parse reconciliation request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "reconcile request onto prior event state")) return 0;

    a = find_self(&state, "p1: A");
    b = find_self(&state, "p1: B");
    c = find_self(&state, "p1: C");
    if (!assert_true(a != NULL && b != NULL && c != NULL, "all self pokemon retained by ident")) return 0;
    if (!assert_true(a->active == 0, "benched A stays inactive")) return 0;
    if (!assert_true(c->active == 1 && c->active_slot == 1, "C remains slot1 active")) return 0;
    if (!assert_true(b->active == 1 && b->active_slot == 2, "B remains slot2 active")) return 0;
    if (!assert_true(strcmp(c->ident, "p1: C") != 0, "request ident does not overwrite event ident format")) return 0;
    if (!assert_true(c->move_ids[0].value == move_id_from_name("protect"), "slot1 moves attached to C")) return 0;
    if (!assert_true(b->move_ids[0].value == move_id_from_name("leafstorm"), "slot2 moves attached to B")) return 0;
    return 1;
}

static int test_observation_request_flags_and_side_features(void) {
    RawBattleState state;
    ParsedRequest req;
    ActionMask mask;
    Observation obs;
    float* flat;
    size_t flat_size;

    raw_battle_state_init(&state, 1);
    parsed_request_init(&req);
    action_mask_init(&mask);
    req.forced_switch_any = 1;
    req.team_preview = 1;
    state.self_side.safeguard = 1;
    state.self_side.safeguard_turns = 4;
    state.self_side.mist = 1;
    state.self_side.mist_turns = 3;
    state.self_side.lucky_chant = 1;
    state.self_side.lucky_chant_turns = 2;
    state.self_side.quick_guard = 1;
    state.self_side.wide_guard = 1;
    state.self_side.crafty_shield = 1;
    state.self_side.mat_block = 1;

    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(obs.forced_switch == 1, "forced switch propagated")) return 0;
    if (!assert_true(obs.team_preview == 1, "team preview propagated")) return 0;
    if (!assert_true(obs.self_side.safeguard == 1 && obs.self_side.safeguard_turns == 4.0f, "safeguard exported")) return 0;
    if (!assert_true(obs.self_side.quick_guard == 1 && obs.self_side.mat_block == 1, "single-turn side effects exported")) return 0;

    flat_size = observation_flat_size();
    flat = (float*)malloc(flat_size * sizeof(float));
    if (!assert_true(flat != NULL, "allocate flattened observation")) return 0;
    if (!assert_true(observation_flatten(flat, flat_size, &obs) == flat_size, "flatten expanded observation")) {
        free(flat);
        return 0;
    }
    free(flat);
    return 1;
}

static int test_observation_exports_active_slot_identity(void) {
    RawBattleState state;
    Observation obs;
    float* flat = NULL;
    size_t entity_offset = OBS_GLOBAL_FEATURES + 2u * OBS_SIDE_FEATURES;
    int self_left;
    int self_right;
    int opp_left;
    int opp_right;
    int replacement;
    int ok = 1;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Left|Pikachu, L80|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Right|Raichu, L80|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: Foe Left|Eevee, L80|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Foe Right|Vaporeon, L80|100/100");
    self_left = state.self_active_slot_to_team_index[0];
    self_right = state.self_active_slot_to_team_index[1];
    opp_left = state.opp_active_slot_to_team_index[0];
    opp_right = state.opp_active_slot_to_team_index[1];
    ok &= assert_true(self_left >= 0 && self_right >= 0 && opp_left >= 0 && opp_right >= 0,
        "active-slot observation test reconstructs all four board positions");
    observation_from_raw_state(&obs, &state, NULL, NULL);
    ok &= assert_true(obs.self_team[self_left].active_slot == 1 && obs.self_team[self_right].active_slot == 2,
        "structured self observation distinguishes left and right active slots");
    ok &= assert_true(obs.opp_team[opp_left].active_slot == 1 && obs.opp_team[opp_right].active_slot == 2,
        "structured opponent observation distinguishes left and right active slots");
    flat = (float*)malloc(observation_flat_size() * sizeof(float));
    ok &= assert_true(flat != NULL &&
            observation_flatten(flat, observation_flat_size(), &obs) == observation_flat_size(),
        "active-slot observation test flattens features");
    if (flat) {
        size_t self_left_base = entity_offset + (size_t)self_left * OBS_POKEMON_FEATURES;
        size_t self_right_base = entity_offset + (size_t)self_right * OBS_POKEMON_FEATURES;
        size_t opp_left_base = entity_offset + (OBS_TEAM_SIZE + (size_t)opp_left) * OBS_POKEMON_FEATURES;
        size_t opp_right_base = entity_offset + (OBS_TEAM_SIZE + (size_t)opp_right) * OBS_POKEMON_FEATURES;
        ok &= assert_true(flat[self_left_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 1u] == 1.0f &&
                flat[self_left_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 2u] == 0.0f,
            "flattened self-left entity selects left-slot class");
        ok &= assert_true(flat[self_right_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 2u] == 1.0f &&
                flat[self_right_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 1u] == 0.0f,
            "flattened self-right entity selects right-slot class");
        ok &= assert_true(flat[opp_left_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 1u] == 1.0f &&
                flat[opp_right_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 2u] == 1.0f,
            "flattened opponent entities preserve board-side slot classes");
    }
    free(flat);
    flat = NULL;

    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Replacement|Jolteon, L80|100/100");
    replacement = state.self_active_slot_to_team_index[0];
    observation_from_raw_state(&obs, &state, NULL, NULL);
    ok &= assert_true(replacement >= 0 && obs.self_team[replacement].active_slot == 1,
        "replacement inherits the vacated left-slot identity");
    ok &= assert_true(obs.self_team[self_left].active_slot == 0 && !obs.self_team[self_left].active,
        "benched Pokemon returns to the non-active slot class");
    flat = (float*)malloc(observation_flat_size() * sizeof(float));
    ok &= assert_true(flat != NULL &&
            observation_flatten(flat, observation_flat_size(), &obs) == observation_flat_size(),
        "post-switch active-slot observation flattens");
    if (flat) {
        size_t benched_base = entity_offset + (size_t)self_left * OBS_POKEMON_FEATURES;
        size_t replacement_base = entity_offset + (size_t)replacement * OBS_POKEMON_FEATURES;
        ok &= assert_true(flat[benched_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET] == 1.0f &&
                flat[benched_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 1u] == 0.0f,
            "flattened benched entity selects non-active class");
        ok &= assert_true(flat[replacement_base + OBS_POKEMON_ACTIVE_SLOT_OFFSET + 1u] == 1.0f,
            "flattened replacement selects left-slot class");
    }
    free(flat);
    return ok;
}

static int test_find_or_make_does_not_overwrite_full_team(void) {
    RawBattleState state;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: B|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1: C|Armarouge, L80, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1: D|Dodrio, L85, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1: E|Grafaiai, L88, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1: F|Persian, L92, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-status|p1: G|brn");

    if (!assert_true(strcmp(state.self_team[0].ident, "p1a: A") == 0, "slot0 ident preserved when team full")) return 0;
    return 1;
}

static int test_condition_status_without_hp_preserves_hp(void) {
    RawBattleState state;
    RawPokemon* pokemon;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    pokemon = find_self(&state, "p1a: A");
    if (!assert_true(pokemon != NULL, "find switched pokemon")) return 0;
    pokemon->current_hp = 75;
    pokemon->max_hp = 100;
    raw_battle_state_update_from_event_line(&state, "|-damage|p1a: A|tox");
    if (!assert_true(pokemon->current_hp == 75 && pokemon->max_hp == 100, "hp preserved when absent from condition")) return 0;
    if (!assert_true(pokemon->status_id.value == 4, "toxic status parsed without hp text")) return 0;
    return 1;
}

static int test_turn_number_not_overwritten_by_request(void) {
    const char* json =
        "{\"active\":[{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false}],"
        "\"side\":{\"pokemon\":[{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true}]}}";
    ParsedRequest req;
    RawBattleState state;
    Observation obs;
    ActionMask mask;

    raw_battle_state_init(&state, 0);
    parsed_request_init(&req);
    action_mask_init(&mask);
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    if (!assert_true(parse_request_payload(&req, json, 77, 0), "parse turn regression request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "update state from request")) return 0;
    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(fabs((double)obs.turn_norm - 0.02) < 0.0001, "turn norm uses battle turn not request id")) return 0;
    return 1;
}

static int test_runtime_request_session_not_forced_doubles(void) {
    GruModel* model;
    EnvRuntime runtime;
    RuntimeMessage msg;
    int ok;

    model = gru_model_create(observation_flat_size(), 8, OBS_NUM_ACTIONS);
    if (!assert_true(model != NULL, "create minimal gru model")) return 0;
    if (!assert_true(env_runtime_init(&runtime, model, NULL, 1, ENV_REWARD_TERMINAL, NULL, NULL), "init runtime")) {
        gru_model_destroy(model);
        return 0;
    }
    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_REQUEST;
    strncpy(msg.battle_id, "battle-test", sizeof(msg.battle_id) - 1);
    strncpy(msg.payload, "{\"wait\":true}", sizeof(msg.payload) - 1);
    msg.request_id = 3;
    msg.is_doubles = 0;
    ok = env_runtime_handle_message(&runtime, &msg, NULL);
    if (!assert_true(ok, "queue request before battle_start")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(runtime.count == 1, "session created for pre-start buffering")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(runtime.sessions[0].format_known == 0 && runtime.sessions[0].pending_prestart_count == 1,
            "request buffered before battle_start")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_BATTLE_START;
    strncpy(msg.battle_id, "battle-test", sizeof(msg.battle_id) - 1);
    msg.is_doubles = 0;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "handle singles battle_start")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(runtime.count == 1 && runtime.sessions[0].raw_state.is_doubles == 0, "battle_start sets singles format")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(runtime.sessions[0].pending_prestart_count == 0, "pre-start queue drained after battle_start")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    env_runtime_free(&runtime);
    gru_model_destroy(model);
    return 1;
}

static int test_runtime_dense_additive_rewards(void) {
    const char* request_payload =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false},"
        "{\"moves\":[{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":true}"
        "]}}";
    GruModel* model;
    EnvRuntime runtime;
    RuntimeMessage msg;
    EnvSession* session;
    const EnvDenseRewardConfig dense_reward_config = {0.10f, 0.25f, 0.40f};
    static const char* const setup_events[] = {
        "|switch|p1a: A|Sawsbuck, L91, M|100/100",
        "|switch|p1b: B|Kingambit, L77, M|100/100",
        "|switch|p2a: X|Armarouge, L80, M|100/100",
        "|switch|p2b: Y|Dodrio, L85, M|100/100"
    };
    size_t i;

    model = gru_model_create(observation_flat_size(), 8, OBS_NUM_ACTIONS);
    if (!assert_true(model != NULL, "create minimal gru model for dense rewards")) return 0;
    if (!assert_true(env_runtime_init(&runtime, model, NULL, 1, ENV_REWARD_DENSE_ADDITIVE, &dense_reward_config, NULL), "init dense runtime")) {
        gru_model_destroy(model);
        return 0;
    }

    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_BATTLE_START;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    msg.is_doubles = 1;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "dense battle_start")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }

    for (i = 0; i < sizeof(setup_events) / sizeof(setup_events[0]); ++i) {
        runtime_message_init(&msg);
        msg.type = RUNTIME_MSG_EVENT;
        strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
        strncpy(msg.line, setup_events[i], sizeof(msg.line) - 1);
        msg.is_doubles = 1;
        if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "dense setup event")) {
            env_runtime_free(&runtime);
            gru_model_destroy(model);
            return 0;
        }
    }

    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_REQUEST;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    strncpy(msg.payload, request_payload, sizeof(msg.payload) - 1);
    msg.request_id = 1;
    msg.is_doubles = 1;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "first dense request")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    session = &runtime.sessions[0];
    if (!assert_true(strcmp(session->episode.reward_mode, "dense_additive") == 0 &&
            session->episode.reward_config_present,
            "dense runtime records reward provenance")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(session->episode.count == 1, "first dense request appends one step")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(fabs((double)session->episode.rewards[0]) < 0.0001, "first dense request reward is zero")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }

    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_EVENT;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    strncpy(msg.line, "|-damage|p2a: X|50/100", sizeof(msg.line) - 1);
    msg.is_doubles = 1;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "dense opponent chip event")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_EVENT;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    strncpy(msg.line, "|faint|p2b: Y", sizeof(msg.line) - 1);
    msg.is_doubles = 1;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "dense opponent faint event")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }

    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_REQUEST;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    strncpy(msg.payload, request_payload, sizeof(msg.payload) - 1);
    msg.request_id = 2;
    msg.is_doubles = 1;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "second dense request")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(session->episode.count == 2, "second dense request appends second step")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(fabs((double)(session->episode.rewards[1] - 0.40f)) < 0.0001, "dense reward clips hp and faint swing at configured maximum")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }

    session->reward_snapshot_valid = 1;
    session->prev_self_hp_frac_sum = 6.0f;
    session->prev_opp_hp_frac_sum = 6.0f;
    session->prev_self_fainted_count = 0;
    session->prev_opp_fainted_count = 0;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        session->raw_state.self_team[i].max_hp = 100;
        session->raw_state.self_team[i].current_hp = 100;
        session->raw_state.self_team[i].fainted = 0;
        session->raw_state.opp_team[i].max_hp = 100;
        session->raw_state.opp_team[i].current_hp = 0;
        session->raw_state.opp_team[i].fainted = 1;
    }

    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_REQUEST;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    strncpy(msg.payload, request_payload, sizeof(msg.payload) - 1);
    msg.request_id = 3;
    msg.is_doubles = 1;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "third dense request")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(session->episode.count == 3, "third dense request appends third step")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(fabs((double)(session->episode.rewards[2] - 0.40f)) < 0.0001, "dense reward clips at configured maximum")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }

    runtime_message_init(&msg);
    msg.type = RUNTIME_MSG_TERMINAL;
    strncpy(msg.battle_id, "battle-dense", sizeof(msg.battle_id) - 1);
    msg.reward = 1.0f;
    if (!assert_true(env_runtime_handle_message(&runtime, &msg, NULL), "dense terminal")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(fabs((double)(session->episode.rewards[2] - 1.40f)) < 0.0001, "terminal reward adds onto dense reward")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }
    if (!assert_true(session->episode.dones[2] == 1, "dense terminal marks final step done")) {
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 0;
    }

    env_runtime_free(&runtime);
    gru_model_destroy(model);
    return 1;
}

static int test_single_turn_side_guards_reconstructed(void) {
    RawBattleState state;
    ParsedRequest req;
    ActionMask mask;
    Observation obs;

    raw_battle_state_init(&state, 1);
    parsed_request_init(&req);
    action_mask_init(&mask);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Quick Guard");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Wide Guard");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Crafty Shield");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Mat Block");
    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(obs.self_side.quick_guard == 1, "quick guard reconstructed")) return 0;
    if (!assert_true(obs.self_side.wide_guard == 1, "wide guard reconstructed")) return 0;
    if (!assert_true(obs.self_side.crafty_shield == 1, "crafty shield reconstructed")) return 0;
    if (!assert_true(obs.self_side.mat_block == 1, "mat block reconstructed")) return 0;
    return 1;
}

static int test_switch_clears_volatile_state(void) {
    RawBattleState state;
    RawPokemon* outgoing;
    RawPokemon* incoming;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-start|p1a: A|Encore");
    raw_battle_state_update_from_event_line(&state, "|-start|p1a: A|Substitute");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: B|Kingambit, L77, M|100/100");
    outgoing = find_self(&state, "p1a: A");
    incoming = find_self(&state, "p1a: B");
    if (!assert_true(outgoing != NULL && incoming != NULL, "find outgoing and incoming pokemon")) return 0;
    if (!assert_true(outgoing->active == 0, "outgoing pokemon benched")) return 0;
    if (!assert_true(outgoing->encore_active == 0, "outgoing encore cleared on switch")) return 0;
    if (!assert_true(outgoing->substitute_active == 0, "outgoing substitute cleared on switch")) return 0;
    if (!assert_true(incoming->active == 1 && incoming->active_slot == 1, "incoming pokemon occupies slot")) return 0;
    return 1;
}

static int test_request_active_data_respects_board_slots(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false},"
        "{\"moves\":[{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Armarouge, L80, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* slot1;
    RawPokemon* slot2;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: B|Kingambit, L77, M|100/100");
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 9, 1), "parse board-slot request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "update board-slot state from request")) return 0;
    slot1 = find_self(&state, "p1a: A");
    slot2 = find_self(&state, "p1b: B");
    if (!assert_true(slot1 != NULL && slot2 != NULL, "find slot-mapped actives")) return 0;
    if (!assert_true(slot1->active_slot == 1 && slot2->active_slot == 2, "board slots preserved from prior event state")) return 0;
    if (!assert_true(slot1->move_ids[0].value == move_id_from_name("protect"), "slot1 private data bound to slot1")) return 0;
    if (!assert_true(slot2->move_ids[0].value == move_id_from_name("leafstorm"), "slot2 private data bound to slot2")) return 0;
    return 1;
}

static int test_hazard_layers_are_capped(void) {
    RawBattleState state;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Toxic Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Toxic Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p2: foe|Toxic Spikes");
    if (!assert_true(state.opp_side.spikes == 3, "spikes capped at 3")) return 0;
    if (!assert_true(state.opp_side.toxic_spikes == 2, "toxic spikes capped at 2")) return 0;
    return 1;
}

static int test_protect_chain_resets_when_not_used(void) {
    RawBattleState state;
    RawPokemon* pokemon;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Protect");
    pokemon = find_self(&state, "p1a: A");
    if (!assert_true(pokemon != NULL, "find protect user")) return 0;
    if (!assert_true(pokemon->protect_chain_count == 1, "protect chain increments")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    if (!assert_true(pokemon->protect_chain_count == 1, "protect chain survives into next turn after use")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|3");
    if (!assert_true(pokemon->protect_chain_count == 0, "protect chain resets after a turn without protect")) return 0;
    return 1;
}

static int test_toxic_counter_progresses_and_resets_on_switch(void) {
    RawBattleState state;
    RawPokemon* toxic_target;
    RawPokemon* benched;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-status|p1a: A|tox");
    toxic_target = find_self(&state, "p1a: A");
    if (!assert_true(toxic_target != NULL, "find toxic target")) return 0;
    if (!assert_true(toxic_target->toxic_counter == 1, "toxic counter initialized")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    if (!assert_true(toxic_target->toxic_counter == 2, "toxic counter increments each turn while active")) return 0;
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: B|Kingambit, L77, M|100/100");
    benched = find_self(&state, "p1a: A");
    if (!assert_true(benched != NULL, "find benched toxic pokemon")) return 0;
    if (!assert_true(benched->toxic_counter == 0, "toxic counter resets on switch")) return 0;
    return 1;
}

static int test_sleep_turns_elapsed_over_time(void) {
    RawBattleState state;
    RawPokemon* sleeper;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-status|p1a: A|slp");
    sleeper = find_self(&state, "p1a: A");
    if (!assert_true(sleeper != NULL, "find sleeping pokemon")) return 0;
    if (!assert_true(sleeper->sleep_turns_elapsed == 0, "sleep elapsed initialized")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    if (!assert_true(sleeper->sleep_turns_elapsed == 1, "sleep elapsed increments after one turn")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|3");
    if (!assert_true(sleeper->sleep_turns_elapsed == 2, "sleep elapsed increments again")) return 0;
    return 1;
}

static int test_remaining_pokemon_tracks_faints(void) {
    RawBattleState state;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Armarouge, L80, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|faint|p1a: A");
    raw_battle_state_update_from_event_line(&state, "|faint|p2a: X");
    if (!assert_true(state.self_side.remaining_pokemon == 5, "self remaining pokemon decremented on faint")) return 0;
    if (!assert_true(state.opp_side.remaining_pokemon == 5, "opp remaining pokemon decremented on faint")) return 0;
    return 1;
}

static int test_transform_effective_state_and_switch_cleanup(void) {
    RawBattleState state;
    ParsedRequest req;
    ActionMask mask;
    Observation obs;
    RawPokemon* ditto_like;

    raw_battle_state_init(&state, 1);
    parsed_request_init(&req);
    action_mask_init(&mask);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|move|p2a: X|protect");
    raw_battle_state_update_from_event_line(&state, "|-transform|p1a: A|p2a: X");
    ditto_like = find_self(&state, "p1a: A");
    if (!assert_true(ditto_like != NULL, "find transformed pokemon")) return 0;
    if (!assert_true(ditto_like->transformed == 1, "transform flag set")) return 0;
    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(obs.self_team[0].species_id == species_id_from_name("kingambit"), "effective transformed species exported")) return 0;
    if (!assert_true(obs.self_team[0].move_id[0] == move_id_from_name("protect"), "effective transformed move exported")) return 0;
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: B|Dodrio, L85, M|100/100");
    if (!assert_true(ditto_like->transformed == 0, "transform cleared on switch")) return 0;
    if (!assert_true(ditto_like->effective_species_id.value == species_id_from_name("sawsbuck"), "effective species reset on switch")) return 0;
    return 1;
}

static int test_first_doubles_request_bootstrap_succeeds(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false},"
        "{\"moves\":[{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":true}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;

    raw_battle_state_init(&state, 1);
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 1, 1), "parse first doubles request")) return 0;
    if (!assert_true(req.bootstrap_slot_binding_ambiguous == 0, "first doubles request accepted as authoritative bootstrap")) return 0;
    if (!assert_true(req.active_team_idx_known[0] == 1 && req.active_team_idx_known[1] == 1, "active team indices marked known")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "first doubles request bootstrap succeeds")) return 0;
    if (!assert_true(state.self_active_slot_to_team_index[0] >= 0 && state.self_active_slot_to_team_index[1] >= 0, "active slots established from bootstrap")) return 0;
    return 1;
}

static int test_switch_identity_prefers_exact_or_empty_over_same_name_reuse(void) {
    RawBattleState state;
    int known_count = 0;
    int i;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: X|Dodrio, L85, M|100/100");
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (state.opp_team[i].ident[0]) {
            ++known_count;
        }
    }
    if (!assert_true(known_count == 2, "same-name switch creates distinct team objects when slot identity differs")) return 0;
    return 1;
}

static int test_p2_request_sets_self_perspective_and_reuses_event_state(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"judgment\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Electric\"},"
        "{\"moves\":[{\"id\":\"crabhammer\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Water\"}"
        "],"
        "\"side\":{\"name\":\"Guest\",\"id\":\"p2\",\"pokemon\":["
        "{\"ident\":\"p2: Arceus\",\"details\":\"Arceus-Ice, L73\",\"condition\":\"296/296\",\"active\":true},"
        "{\"ident\":\"p2: Crawdaunt\",\"details\":\"Crawdaunt, L87, F\",\"condition\":\"251/251\",\"active\":true},"
        "{\"ident\":\"p2: Falinks\",\"details\":\"Falinks, L87\",\"condition\":\"255/255\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: X|Weezing-Galar, L89, F|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Y|Meowscarada, L82, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: Arceus|Arceus-Ice, L73|296/296");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Crawdaunt|Crawdaunt, L87, F|251/251");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 3, 1), "parse p2 request")) return 0;
    if (!assert_true(req.side_player == 2, "request side identifies as p2")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "reconcile p2 request")) return 0;
    if (!assert_true(state.perspective_known == 1 && state.self_side_player == 2, "state perspective switches to p2 self")) return 0;
    if (!assert_true(state.self_active_slot_to_team_index[0] >= 0 && state.self_active_slot_to_team_index[1] >= 0, "self active slots set for p2 request")) return 0;
    if (!assert_true(strcmp(state.self_team[state.self_active_slot_to_team_index[0]].canonical_ident, "p2a: Arceus") == 0, "p2a event mon becomes self slot1")) return 0;
    if (!assert_true(strcmp(state.self_team[state.self_active_slot_to_team_index[1]].canonical_ident, "p2b: Crawdaunt") == 0, "p2b event mon becomes self slot2")) return 0;
    if (!assert_true(state.self_team[state.self_active_slot_to_team_index[0]].current_hp == 296, "p2 self slot1 keeps event hp")) return 0;
    if (!assert_true(state.self_team[state.self_active_slot_to_team_index[1]].current_hp == 251, "p2 self slot2 keeps event hp")) return 0;
    if (!assert_true(strcmp(state.opp_team[state.opp_active_slot_to_team_index[0]].canonical_ident, "p1a: X") == 0, "p1a becomes opponent after perspective swap")) return 0;
    return 1;
}

static int test_transform_uses_battle_effective_pp(void) {
    RawBattleState state;
    ParsedRequest req;
    ActionMask mask;
    Observation obs;
    RawPokemon* ditto_like;

    raw_battle_state_init(&state, 1);
    parsed_request_init(&req);
    action_mask_init(&mask);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|move|p2a: X|protect");
    {
        RawPokemon* target = NULL;
        int i;
        for (i = 0; i < RAW_TEAM_SIZE; ++i) {
            if (strcmp(state.opp_team[i].ident, "p2a: X") == 0) {
                target = &state.opp_team[i];
                break;
            }
        }
        if (!assert_true(target != NULL, "find transform target")) return 0;
        target->effective_move_pp[0] = 2;
        target->effective_move_max_pp[0] = 8;
    }
    raw_battle_state_update_from_event_line(&state, "|-transform|p1a: A|p2a: X");
    ditto_like = find_self(&state, "p1a: A");
    if (!assert_true(ditto_like != NULL, "find transformed pokemon for pp test")) return 0;
    if (!assert_true(ditto_like->effective_move_pp[0] == 5 && ditto_like->effective_move_max_pp[0] == 5, "transform uses 5/5 effective pp")) return 0;
    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(fabs((double)obs.self_team[0].move_pp_frac[0] - 1.0) < 0.0001, "transform observation pp fraction uses copied move pp")) return 0;
    return 1;
}

static int test_transform_survives_request_reconciliation(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":3,\"maxpp\":5,\"target\":\"self\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true}"
        "]}}";
    RawBattleState state;
    ParsedRequest req;
    ActionMask mask;
    Observation obs;
    RawPokemon* transformed_mon;

    raw_battle_state_init(&state, 0);
    parsed_request_init(&req);
    action_mask_init(&mask);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|move|p2a: X|protect");
    raw_battle_state_update_from_event_line(&state, "|-transform|p1a: A|p2a: X");
    if (!assert_true(parse_request_payload(&req, json, 2, 0), "parse transformed request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "reconcile transformed request")) return 0;
    transformed_mon = find_self(&state, "p1a: A");
    if (!assert_true(transformed_mon != NULL, "find transformed pokemon after request")) return 0;
    if (!assert_true(transformed_mon->transformed == 1, "transform flag preserved across request")) return 0;
    if (!assert_true(transformed_mon->effective_species_id.value == species_id_from_name("kingambit"), "effective species preserved across request")) return 0;
    if (!assert_true(transformed_mon->effective_move_ids[0].value == move_id_from_name("protect"), "effective move preserved across request")) return 0;
    if (!assert_true(transformed_mon->effective_move_type_ids[0].value == type_id_from_name("normal"), "effective move type preserved across request")) return 0;
    if (!assert_true(transformed_mon->effective_move_pp[0] == 3 && transformed_mon->effective_move_max_pp[0] == 5, "request updates transformed effective pp only")) return 0;
    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(obs.self_team[0].species_id == species_id_from_name("kingambit"), "observation keeps transformed species after request")) return 0;
    if (!assert_true(obs.self_team[0].move_id[0] == move_id_from_name("protect"), "observation keeps transformed move after request")) return 0;
    if (!assert_true(transformed_mon->move_ids[0].value == 0, "base movepool not contaminated by transformed request state")) return 0;
    return 1;
}

static int test_same_name_same_species_does_not_merge(void) {
    RawBattleState state;
    int known_count = 0;
    int i;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: X|Kingambit, L77, M|100/100");
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (state.opp_team[i].ident[0]) {
            ++known_count;
        }
    }
    if (!assert_true(known_count == 2, "same-name same-species switch creates distinct objects when ambiguous")) return 0;
    return 1;
}

static int test_transformed_move_event_does_not_contaminate_base_moves(void) {
    RawBattleState state;
    RawPokemon* transformed_mon;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|move|p2a: X|protect");
    raw_battle_state_update_from_event_line(&state, "|-transform|p1a: A|p2a: X");
    raw_battle_state_update_from_event_line(&state, "|move|p1a: A|protect");
    transformed_mon = find_self(&state, "p1a: A");
    if (!assert_true(transformed_mon != NULL, "find transformed pokemon after move")) return 0;
    if (!assert_true(transformed_mon->effective_move_ids[0].value == move_id_from_name("protect"), "effective move learned while transformed")) return 0;
    if (!assert_true(transformed_mon->move_ids[0].value == 0, "base move state unchanged by transformed move event")) return 0;
    return 1;
}

static int test_ambiguous_non_slot_event_fails_closed(void) {
    RawBattleState state;
    int known_count = 0;
    int i;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: X|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-damage|p2: X|50/100");
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (state.opp_team[i].ident[0]) {
            ++known_count;
        }
    }
    if (!assert_true(known_count == 2, "ambiguous non-slot event does not allocate a new object")) return 0;
    return 1;
}

static int test_event_switch_rebuilds_slot_maps_for_followup_events(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false},"
        "{\"moves\":[{\"id\":\"leafstorm\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: C\",\"details\":\"Armarouge, L80, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* a;
    RawPokemon* c;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: B|Kingambit, L77, M|100/100");
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 1, 1), "parse slot-map seed request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "seed slot-map state from request")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p1a: C|Armarouge, L80, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-damage|p1a: C|50/100");
    a = find_self(&state, "p1: A");
    c = find_self(&state, "p1: C");
    if (!assert_true(a != NULL && c != NULL, "find old and new slot1 pokemon")) return 0;
    if (!assert_true(a->current_hp == 100, "old slot1 pokemon not damaged after replacement")) return 0;
    if (!assert_true(c->current_hp == 50 && c->active == 1 && c->active_slot == 1, "follow-up event resolves against current slot occupant")) return 0;
    return 1;
}

static int test_faint_event_zeros_hp_in_state_and_observation(void) {
    RawBattleState state;
    ParsedRequest req;
    ActionMask mask;
    Observation obs;
    RawPokemon* pokemon;

    raw_battle_state_init(&state, 0);
    parsed_request_init(&req);
    action_mask_init(&mask);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|75/100");
    pokemon = find_self(&state, "p1a: A");
    if (!assert_true(pokemon != NULL, "find pokemon before faint")) return 0;
    raw_battle_state_update_from_event_line(&state, "|faint|p1a: A");
    if (!assert_true(pokemon->fainted == 1 && pokemon->current_hp == 0, "faint event zeros hp")) return 0;
    observation_from_raw_state(&obs, &state, &req, &mask);
    if (!assert_true(obs.self_team[0].hp_frac == 0.0f && obs.self_team[0].fainted == 1, "observation exports zero hp for fainted mon")) return 0;
    return 1;
}

static int test_faint_condition_without_slash_zeros_hp(void) {
    RawBattleState state;
    RawPokemon* pokemon;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|turn|1");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|75/100");
    pokemon = find_self(&state, "p1a: A");
    if (!assert_true(pokemon != NULL, "find pokemon before faint condition")) return 0;
    raw_battle_state_update_from_event_line(&state, "|-damage|p1a: A|0 fnt");
    if (!assert_true(pokemon->fainted == 1 && pokemon->current_hp == 0 && pokemon->max_hp == 100, "0 fnt condition zeros hp without losing max hp")) return 0;
    return 1;
}

static int test_request_parser_reads_private_tera_type_even_when_moves_disabled(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":0,\"maxpp\":16,\"target\":\"self\",\"disabled\":true}],\"canTerastallize\":\"Grass\",\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true}"
        "]}}";
    ParsedRequest req;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 5, 0), "parse tera-type request")) return 0;
    if (!assert_true(req.can_tera == 1, "global tera availability parsed even when moves are disabled")) return 0;
    if (!assert_true(req.active[0].can_tera == 1, "slot tera availability parsed")) return 0;
    if (!assert_true(req.active[0].tera_type_id == type_id_from_name("Grass"), "private tera type parsed from request")) return 0;
    return 1;
}

static int test_request_parser_reads_private_side_item_ability_tera_and_moves(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"canTerastallize\":\"Grass\"}"
        "],"
        "\"side\":{\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true,\"moves\":[\"protect\",\"doubleedge\",\"jumpkick\",\"trailblaze\"],\"baseAbility\":\"chlorophyll\",\"ability\":\"chlorophyll\",\"item\":\"lifeorb\",\"teraType\":\"Grass\",\"terastallized\":\"\"},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":false,\"moves\":[\"kowtowcleave\",\"ironhead\",\"protect\",\"suckerpunch\"],\"baseAbility\":\"supremeoverlord\",\"item\":\"leftovers\",\"teraType\":\"Dark\",\"terastallized\":\"Dark\"}"
        "]}}";
    ParsedRequest req;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 6, 0), "parse private side metadata request")) return 0;
    if (!assert_true(req.side_item_id[0] == item_id_from_name("lifeorb"), "private side item parsed")) return 0;
    if (!assert_true(req.side_ability_id[0] == ability_id_from_name("chlorophyll"), "private side ability parsed")) return 0;
    if (!assert_true(req.side_tera_type_id[0] == type_id_from_name("Grass"), "private side tera type parsed")) return 0;
    if (!assert_true(req.side_move_id[0][1] == move_id_from_name("doubleedge"), "private side move list parsed")) return 0;
    if (!assert_true(req.side_tera_used[1] == 1, "private side terastallized flag parsed")) return 0;
    return 1;
}

static int test_request_parser_reads_private_side_stats_and_flags(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":true,\"maybeTrapped\":true}"
        "],"
        "\"side\":{\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/321\",\"active\":true,\"stats\":{\"atk\":222,\"def\":180,\"spa\":140,\"spd\":160,\"spe\":199},\"moves\":[\"protect\",\"doubleedge\"],\"baseAbility\":\"chlorophyll\",\"ability\":\"\",\"item\":\"\",\"commanding\":true,\"reviving\":true,\"teraType\":\"Grass\",\"terastallized\":\"\"}"
        "]}}";
    ParsedRequest req;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 61, 0), "parse private side stats and flags request")) return 0;
    if (!assert_true(req.side_ability_id[0] == ability_id_from_name("chlorophyll"), "baseAbility fallback parsed")) return 0;
    if (!assert_true(req.side_base_ability_id[0] == ability_id_from_name("chlorophyll"), "baseAbility stored separately")) return 0;
    if (!assert_true(req.side_item_id[0] == 0, "empty item parsed as confirmed none")) return 0;
    if (!assert_true(req.side_stats_hp[0] == 321, "private hp stat parsed from condition")) return 0;
    if (!assert_true(req.side_stats_atk[0] == 222 && req.side_stats_spe[0] == 199, "private stats parsed")) return 0;
    if (!assert_true(req.side_commanding[0] == 1 && req.side_reviving[0] == 1, "private commanding and reviving parsed")) return 0;
    if (!assert_true(req.active[0].trapped == 1 && req.active[0].maybe_trapped == 1, "active trapped flags parsed")) return 0;
    return 1;
}

static int test_request_reconciliation_imports_private_side_metadata(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"canTerastallize\":\"Grass\"}"
        "],"
        "\"side\":{\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true,\"moves\":[\"protect\",\"doubleedge\",\"jumpkick\",\"trailblaze\"],\"ability\":\"chlorophyll\",\"item\":\"lifeorb\",\"teraType\":\"Grass\",\"terastallized\":\"\"},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":false,\"moves\":[\"kowtowcleave\",\"ironhead\",\"protect\",\"suckerpunch\"],\"ability\":\"supremeoverlord\",\"item\":\"leftovers\",\"teraType\":\"Dark\",\"terastallized\":\"Dark\"}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* active;
    RawPokemon* bench;

    raw_battle_state_init(&state, 0);
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 7, 0), "parse private side metadata reconciliation request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply private side metadata reconciliation request")) return 0;
    active = find_self(&state, "p1: A");
    bench = find_self(&state, "p1: B");
    if (!assert_true(active != NULL && bench != NULL, "find self pokemon after private metadata reconciliation")) return 0;
    if (!assert_true(active->item_id.value == item_id_from_name("lifeorb"), "active item imported from request side data")) return 0;
    if (!assert_true(active->ability_id.value == ability_id_from_name("chlorophyll"), "active ability imported from request side data")) return 0;
    if (!assert_true(active->tera_type_id.value == type_id_from_name("Grass"), "active tera type imported from request side data")) return 0;
    if (!assert_true(active->move_ids[1].value == move_id_from_name("doubleedge"), "active full move list imported from request side data")) return 0;
    if (!assert_true(active->move_type_ids[1].value == type_id_from_name("normal"), "active move type imported from request side data")) return 0;
    if (!assert_true(bench->item_id.value == item_id_from_name("leftovers"), "bench item imported from request side data")) return 0;
    if (!assert_true(bench->ability_id.value == ability_id_from_name("supremeoverlord"), "bench ability imported from request side data")) return 0;
    if (!assert_true(bench->tera_used == 1 && bench->tera_type_id.value == type_id_from_name("Dark"), "bench tera metadata imported from request side data")) return 0;
    if (!assert_true(bench->move_ids[0].value == move_id_from_name("kowtowcleave"), "bench move list imported from request side data")) return 0;
    if (!assert_true(bench->move_type_ids[0].value == type_id_from_name("dark"), "bench move type imported from request side data")) return 0;
    return 1;
}

static int test_request_reconciliation_imports_stats_flags_and_trapped_state(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":true,\"maybeTrapped\":true}"
        "],"
        "\"side\":{\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/321\",\"active\":true,\"stats\":{\"atk\":222,\"def\":180,\"spa\":140,\"spd\":160,\"spe\":199},\"moves\":[\"protect\",\"doubleedge\"],\"ability\":\"\",\"baseAbility\":\"chlorophyll\",\"item\":\"\",\"commanding\":true,\"reviving\":true,\"teraType\":\"Grass\",\"terastallized\":\"\"},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/303\",\"active\":false,\"stats\":{\"atk\":240,\"def\":201,\"spa\":120,\"spd\":155,\"spe\":111},\"moves\":[\"kowtowcleave\",\"ironhead\"],\"ability\":\"supremeoverlord\",\"item\":\"leftovers\",\"commanding\":false,\"reviving\":false,\"teraType\":\"Dark\",\"terastallized\":\"Dark\"}"
        "]}}";
    const char* untrapped_json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":15,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false,\"maybeTrapped\":false}"
        "],"
        "\"side\":{\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/321\",\"active\":true,\"stats\":{\"atk\":222,\"def\":180,\"spa\":140,\"spd\":160,\"spe\":199},\"moves\":[\"protect\",\"doubleedge\"],\"ability\":\"chlorophyll\",\"item\":\"\",\"commanding\":false,\"reviving\":false,\"teraType\":\"Grass\",\"terastallized\":\"\"},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/303\",\"active\":false,\"stats\":{\"atk\":240,\"def\":201,\"spa\":120,\"spd\":155,\"spe\":111},\"moves\":[\"kowtowcleave\",\"ironhead\"],\"ability\":\"supremeoverlord\",\"item\":\"leftovers\",\"commanding\":false,\"reviving\":false,\"teraType\":\"Dark\",\"terastallized\":\"Dark\"}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* active;
    RawPokemon* bench;
    Observation obs;

    raw_battle_state_init(&state, 0);
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 62, 0), "parse stats and trapped reconciliation request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply stats and trapped reconciliation request")) return 0;
    active = find_self(&state, "p1: A");
    bench = find_self(&state, "p1: B");
    if (!assert_true(active != NULL && bench != NULL, "find self pokemon after stats reconciliation")) return 0;
    if (!assert_true(active->ability_id.value == ability_id_from_name("chlorophyll"), "ability fallback imported from request")) return 0;
    if (!assert_true(active->item_id.value == 0, "empty item imported as confirmed none")) return 0;
    if (!assert_true(active->base_hp_stat == 321 && active->base_atk_stat == 222 && active->base_spe_stat == 199, "active base stats imported")) return 0;
    if (!assert_true(active->commanding_active == 1 && active->reviving == 1, "active commanding and reviving imported")) return 0;
    if (!assert_true(active->trapped == 1 && active->maybe_trapped == 1, "active trapped state persisted")) return 0;
    if (!assert_true(bench->base_hp_stat == 303 && bench->base_def_stat == 201, "bench base stats imported")) return 0;
    if (!assert_true(bench->tera_used == 1, "bench tera used imported")) return 0;
    observation_from_raw_state(&obs, &state, &req, NULL);
    if (!assert_true(obs.self_team[0].trapped == 1 && obs.self_team[0].maybe_trapped == 1, "observation exports trapped state")) return 0;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, untrapped_json, 63, 0), "parse untrapped followup request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply untrapped followup request")) return 0;
    if (!assert_true(active->trapped == 0 && active->maybe_trapped == 0, "later request clears trapped state")) return 0;
    if (!assert_true(active->commanding_active == 0 && active->reviving == 0, "later request clears transient request booleans")) return 0;
    return 1;
}

static int test_request_reconciliation_infers_encore_move_slot(void) {
    const char* json =
        "{\"active\":["
        "{\"moves\":["
        "{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},"
        "{\"id\":\"doubleedge\",\"pp\":15,\"maxpp\":24,\"target\":\"normal\",\"disabled\":true},"
        "{\"id\":\"jumpkick\",\"pp\":15,\"maxpp\":24,\"target\":\"normal\",\"disabled\":true},"
        "{\"id\":\"trailblaze\",\"pp\":15,\"maxpp\":24,\"target\":\"normal\",\"disabled\":true}"
        "],\"trapped\":false}"
        "],"
        "\"side\":{\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true,\"moves\":[\"protect\",\"doubleedge\",\"jumpkick\",\"trailblaze\"],\"ability\":\"chlorophyll\"}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* active;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-start|p1a: A|Encore");
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, json, 64, 0), "parse encore inference request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply encore inference request")) return 0;
    active = find_self(&state, "p1: A");
    if (!assert_true(active != NULL, "find active encore pokemon")) return 0;
    return assert_true(active->encore_move_slot == 0, "request inference resolves encore move slot");
}

static int test_event_parser_sets_flinch_and_disable_slot(void) {
    RawBattleState state;
    RawPokemon* active;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    active = find_self(&state, "p1: A");
    if (!assert_true(active != NULL, "find active pokemon for event parser transient test")) return 0;
    tracked_int_set_confirmed(&active->move_ids[0], move_id_from_name("protect"));
    tracked_int_set_confirmed(&active->move_ids[1], move_id_from_name("doubleedge"));
    tracked_int_set_confirmed(&active->move_ids[2], move_id_from_name("jumpkick"));
    tracked_int_set_confirmed(&active->move_ids[3], move_id_from_name("trailblaze"));
    raw_pokemon_refresh_effective_state(active);

    raw_battle_state_update_from_event_line(&state, "|cant|p1a: A|flinch");
    if (!assert_true(active->flinch_active == 1, "cant flinch sets flinch_active")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    if (!assert_true(active->flinch_active == 0, "turn boundary clears flinch_active")) return 0;

    raw_battle_state_update_from_event_line(&state, "|-start|p1a: A|Disable|Jump Kick");
    if (!assert_true(active->disable_active == 1, "disable event sets disable_active")) return 0;
    if (!assert_true(active->disable_move_slot == 2, "disable event resolves disabled move slot")) return 0;
    raw_battle_state_update_from_event_line(&state, "|-end|p1a: A|Disable");
    return assert_true(active->disable_move_slot == -1, "disable end clears disabled move slot");
}

static int test_switch_and_faint_clear_boosts(void) {
    RawBattleState state;
    RawPokemon* active;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    active = find_self(&state, "p1: A");
    if (!assert_true(active != NULL, "find boosted pokemon")) return 0;
    raw_battle_state_update_from_event_line(&state, "|-boost|p1a: A|atk|2");
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: B|Kingambit, L77, M|100/100");
    if (!assert_true(active->boosts[0] == 0, "switch out clears boosts")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-boost|p1a: A|atk|2");
    raw_battle_state_update_from_event_line(&state, "|faint|p1a: A");
    return assert_true(active->boosts[0] == 0 && active->active == 0 && active->fainted == 1, "faint clears boosts and active state");
}

static int test_observation_hides_inferred_weather_and_exports_more_transients(void) {
    RawBattleState state;
    Observation obs;
    RawPokemon* active;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-weather|RainDance");
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|Electric Terrain");
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|move: Magic Room");
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|move: Wonder Room");
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|move: Gravity");
    active = find_self(&state, "p1: A");
    if (!assert_true(active != NULL, "find active pokemon for observation transient export test")) return 0;
    tracked_int_set_confirmed(&active->move_ids[0], move_id_from_name("protect"));
    tracked_int_set_confirmed(&active->move_ids[1], move_id_from_name("trailblaze"));
    active->move_known[0] = 1;
    active->move_known[1] = 1;
    active->encore_move_slot = 1;
    active->disable_move_slot = 0;
    active->torment_active = 1;
    active->torment_turns = 2;
    active->heal_block_active = 1;
    active->heal_block_turns = 4;
    active->embargo_active = 1;
    active->embargo_turns = 3;
    active->yawn_active = 1;
    active->yawn_turns = 2;
    active->helping_hand_active = 1;
    active->flinch_active = 1;
    active->seed_active = 1;
    active->charge_active = 1;
    active->charge_turns = 2;
    raw_pokemon_refresh_effective_state(active);

    observation_from_raw_state(&obs, &state, NULL, NULL);
    if (!assert_true(obs.weather_turns == 0.0f && obs.weather_turns_known_mode == 1, "observation hides inferred weather turns but keeps knowledge mode")) return 0;
    if (!assert_true(obs.terrain_turns == 0.0f && obs.terrain_turns_known_mode == 1, "observation hides inferred terrain turns but keeps knowledge mode")) return 0;
    if (!assert_true(obs.magic_room == 1 && obs.magic_room_turns == 5.0f, "observation exports Magic Room state")) return 0;
    if (!assert_true(obs.wonder_room == 1 && obs.wonder_room_turns == 5.0f, "observation exports Wonder Room state")) return 0;
    if (!assert_true(obs.gravity == 1 && obs.gravity_turns == 5.0f, "observation exports Gravity state")) return 0;
    if (!assert_true(obs.self_team[0].torment_active == 1 && obs.self_team[0].torment_turns == 2.0f, "observation exports torment state")) return 0;
    if (!assert_true(obs.self_team[0].heal_block_active == 1 && obs.self_team[0].heal_block_turns == 4.0f, "observation exports heal block state")) return 0;
    if (!assert_true(obs.self_team[0].embargo_active == 1 && obs.self_team[0].embargo_turns == 3.0f, "observation exports embargo state")) return 0;
    if (!assert_true(obs.self_team[0].yawn_active == 1 && obs.self_team[0].yawn_turns == 2.0f, "observation exports yawn state")) return 0;
    if (!assert_true(obs.self_team[0].encore_move_slot == 1 && obs.self_team[0].disable_move_slot == 0, "observation exports encore and disable move slots")) return 0;
    if (!assert_true(obs.self_team[0].helping_hand_active == 1 && obs.self_team[0].flinch_active == 1, "observation exports single-turn flags")) return 0;
    if (!assert_true(obs.self_team[0].move_type_id[0] == type_id_from_name("normal") && obs.self_team[0].move_type_id[1] == type_id_from_name("grass"), "observation exports move types")) return 0;
    return assert_true(obs.self_team[0].seed_active == 1 && obs.self_team[0].charge_active == 1 && obs.self_team[0].charge_turns == 2.0f,
        "observation exports seed and charge state");
}

static int test_event_parser_reveals_public_abilities_typechange_and_cant_status(void) {
    RawBattleState state;
    RawPokemon* cinderace;
    RawPokemon* farigiraf;
    RawPokemon* lumineon;
    RawPokemon* vaporeon;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Cinderace|Cinderace, L82, F|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Farigiraf|Farigiraf, L80|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: Lumineon|Lumineon, L79, F|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Vaporeon|Vaporeon, L80|100/100");

    raw_battle_state_update_from_event_line(&state, "|-start|p1a: Cinderace|typechange|Fighting|[from] ability: Libero");
    raw_battle_state_update_from_event_line(&state, "|-start|p1a: Cinderace|move: No Retreat");
    raw_battle_state_update_from_event_line(&state, "|cant|p1b: Farigiraf|ability: Armor Tail|Fake Out|[of] p2a: Persian");
    raw_battle_state_update_from_event_line(&state, "|-activate|p2a: Lumineon|ability: Storm Drain");
    raw_battle_state_update_from_event_line(&state, "|cant|p2b: Vaporeon|frz");

    cinderace = find_self(&state, "p1: Cinderace");
    farigiraf = find_self(&state, "p1: Farigiraf");
    lumineon = find_opp(&state, "p2: Lumineon");
    vaporeon = find_opp(&state, "p2: Vaporeon");

    if (!assert_true(cinderace != NULL && farigiraf != NULL && lumineon != NULL && vaporeon != NULL, "find public reveal test pokemon")) return 0;
    if (!assert_true(cinderace->ability_id.value == ability_id_from_name("Libero"), "typechange reveal confirms subject ability")) return 0;
    if (!assert_true(cinderace->effective_type1_id.value == type_id_from_name("Fighting") && cinderace->effective_type2_id.value == 0, "typechange updates effective types")) return 0;
    if (!assert_true(cinderace->trapped == 1, "No Retreat start marks trapped state")) return 0;
    if (!assert_true(farigiraf->ability_id.value == ability_id_from_name("Armor Tail"), "cant ability reveal confirms subject ability")) return 0;
    if (!assert_true(lumineon->ability_id.value == ability_id_from_name("Storm Drain"), "activate ability reveal confirms opponent ability")) return 0;
    return assert_true(vaporeon->status_id.value == 5, "cant frz confirms frozen status");
}

static int test_event_parser_formechange_updates_species_and_reveals_ability(void) {
    RawBattleState state;
    RawPokemon* minior;

    raw_battle_state_init(&state, 0);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Minior|Minior, L75|100/100");
    raw_battle_state_update_from_event_line(&state, "|-formechange|p1a: Minior|Minior-Meteor||[from] ability: Shields Down");
    minior = find_self(&state, "p1: Minior");
    if (!assert_true(minior != NULL, "find formechange pokemon")) return 0;
    if (!assert_true(minior->species_id.value == species_id_from_name("Minior-Meteor"), "formechange updates species")) return 0;
    if (!assert_true(minior->effective_species_id.value == species_id_from_name("Minior-Meteor"), "formechange updates effective species")) return 0;
    return assert_true(minior->ability_id.value == ability_id_from_name("Shields Down"), "formechange reveal confirms ability");
}

static int test_request_reconciliation_preserves_can_tera_through_wait_and_forced_switch(void) {
    const char* actionable_json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"canTerastallize\":\"Grass\",\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    const char* wait_json =
        "{\"wait\":true,"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    const char* forced_switch_json =
        "{\"forceSwitch\":[true],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    const char* spent_json =
        "{\"active\":["
        "{\"moves\":[{\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"trapped\":false}"
        "],"
        "\"side\":{\"pokemon\":["
        "{\"ident\":\"p1: A\",\"details\":\"Sawsbuck, L91, M\",\"condition\":\"100/100\",\"active\":true},"
        "{\"ident\":\"p1: B\",\"details\":\"Kingambit, L77, M\",\"condition\":\"100/100\",\"active\":false}"
        "]}}";
    ParsedRequest req;
    RawBattleState state;

    raw_battle_state_init(&state, 0);
    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, actionable_json, 10, 0), "parse actionable tera request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply actionable tera request")) return 0;
    if (!assert_true(state.can_tera == 1, "actionable request sets can_tera")) return 0;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, wait_json, 11, 0), "parse wait request for tera persistence")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply wait request")) return 0;
    if (!assert_true(state.can_tera == 1, "wait request preserves can_tera")) return 0;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, forced_switch_json, 12, 0), "parse forced-switch request for tera persistence")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply forced-switch request")) return 0;
    if (!assert_true(state.can_tera == 1, "forced-switch request preserves can_tera")) return 0;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, spent_json, 13, 0), "parse non-tera actionable request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply non-tera actionable request")) return 0;
    if (!assert_true(state.can_tera == 0, "normal actionable request with no tera clears can_tera")) return 0;
    return 1;
}

static int test_real_battle_2632274530_faint_force_switch_and_tera_request(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632274530. Clean terminal loss, no disconnect/forfeit. */
    const char* opening_request =
        "{\"active\":["
        "{\"moves\":[{\"move\":\"Crunch\",\"id\":\"crunch\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Play Rough\",\"id\":\"playrough\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Wild Charge\",\"id\":\"wildcharge\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Psychic Fangs\",\"id\":\"psychicfangs\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Fairy\"},"
        "{\"moves\":[{\"move\":\"Shell Smash\",\"id\":\"shellsmash\",\"pp\":24,\"maxpp\":24,\"target\":\"self\",\"disabled\":false},{\"move\":\"Heat Wave\",\"id\":\"heatwave\",\"pp\":16,\"maxpp\":16,\"target\":\"allAdjacentFoes\",\"disabled\":false},{\"move\":\"Power Gem\",\"id\":\"powergem\",\"pp\":32,\"maxpp\":32,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Protect\",\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false}],\"canTerastallize\":\"Fairy\"}"
        "],"
        "\"side\":{\"name\":\"Guest 26657556\",\"id\":\"p2\",\"pokemon\":["
        "{\"ident\":\"p2: Mabosstiff\",\"details\":\"Mabosstiff, L84, F\",\"condition\":\"272/272\",\"active\":true},"
        "{\"ident\":\"p2: Magcargo\",\"details\":\"Magcargo, L93, F\",\"condition\":\"262/262\",\"active\":true},"
        "{\"ident\":\"p2: Tentacruel\",\"details\":\"Tentacruel, L85, F\",\"condition\":\"275/275\",\"active\":false},"
        "{\"ident\":\"p2: Galvantula\",\"details\":\"Galvantula, L85, F\",\"condition\":\"258/258\",\"active\":false},"
        "{\"ident\":\"p2: Suicune\",\"details\":\"Suicune, L79\",\"condition\":\"288/288\",\"active\":false},"
        "{\"ident\":\"p2: Cinderace\",\"details\":\"Cinderace, L82, F\",\"condition\":\"265/265\",\"active\":false}"
        "]},\"rqid\":3}";
    const char* forced_switch_request =
        "{\"forceSwitch\":[false,true],"
        "\"side\":{\"name\":\"Guest 26657556\",\"id\":\"p2\",\"pokemon\":["
        "{\"ident\":\"p2: Suicune\",\"details\":\"Suicune, L79\",\"condition\":\"288/288\",\"active\":true},"
        "{\"ident\":\"p2: Galvantula\",\"details\":\"Galvantula, L85, F\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p2: Tentacruel\",\"details\":\"Tentacruel, L85, F\",\"condition\":\"275/275\",\"active\":false},"
        "{\"ident\":\"p2: Magcargo\",\"details\":\"Magcargo, L93, F\",\"condition\":\"262/262\",\"active\":false},"
        "{\"ident\":\"p2: Mabosstiff\",\"details\":\"Mabosstiff, L84, F\",\"condition\":\"272/272\",\"active\":false},"
        "{\"ident\":\"p2: Cinderace\",\"details\":\"Cinderace, L82, F\",\"condition\":\"265/265\",\"active\":false}"
        "]},\"noCancel\":true,\"rqid\":5}";
    const char* post_switch_request =
        "{\"active\":["
        "{\"moves\":[{\"move\":\"Scald\",\"id\":\"scald\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Ice Beam\",\"id\":\"icebeam\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Protect\",\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},{\"move\":\"Calm Mind\",\"id\":\"calmmind\",\"pp\":32,\"maxpp\":32,\"target\":\"self\",\"disabled\":false}],\"canTerastallize\":\"Grass\"},"
        "{\"moves\":[{\"move\":\"U-turn\",\"id\":\"uturn\",\"pp\":32,\"maxpp\":32,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Protect\",\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},{\"move\":\"High Jump Kick\",\"id\":\"highjumpkick\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Pyro Ball\",\"id\":\"pyroball\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Fighting\"}"
        "],"
        "\"side\":{\"name\":\"Guest 26657556\",\"id\":\"p2\",\"pokemon\":["
        "{\"ident\":\"p2: Suicune\",\"details\":\"Suicune, L79\",\"condition\":\"288/288\",\"active\":true},"
        "{\"ident\":\"p2: Cinderace\",\"details\":\"Cinderace, L82, F\",\"condition\":\"265/265\",\"active\":true},"
        "{\"ident\":\"p2: Tentacruel\",\"details\":\"Tentacruel, L85, F\",\"condition\":\"275/275\",\"active\":false},"
        "{\"ident\":\"p2: Magcargo\",\"details\":\"Magcargo, L93, F\",\"condition\":\"262/262\",\"active\":false},"
        "{\"ident\":\"p2: Mabosstiff\",\"details\":\"Mabosstiff, L84, F\",\"condition\":\"272/272\",\"active\":false},"
        "{\"ident\":\"p2: Galvantula\",\"details\":\"Galvantula, L85, F\",\"condition\":\"0 fnt\",\"active\":false}"
        "]},\"rqid\":7}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* suicune;
    RawPokemon* cinderace;
    RawPokemon* galvantula;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Rayquaza|Rayquaza, L75|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Alomomola|Alomomola, L96, F|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: Mabosstiff|Mabosstiff, L84, F|272/272");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Magcargo|Magcargo, L93, F|262/262");
    raw_battle_state_update_from_event_line(&state, "|turn|1");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, opening_request, 28, 1), "parse real battle 2632274530 opening request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632274530 opening request")) return 0;
    if (!assert_true(state.can_tera == 1, "opening real request exposes tera availability")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p2a: Suicune|Suicune, L79|288/288");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Galvantula|Galvantula, L85, F|258/258");
    raw_battle_state_update_from_event_line(&state, "|move|p1a: Rayquaza|Dragon Ascent|p2b: Galvantula");
    raw_battle_state_update_from_event_line(&state, "|-crit|p2b: Galvantula");
    raw_battle_state_update_from_event_line(&state, "|-damage|p2b: Galvantula|0 fnt");
    raw_battle_state_update_from_event_line(&state, "|faint|p2b: Galvantula");

    galvantula = find_self(&state, "p2: Galvantula");
    if (!assert_true(galvantula != NULL, "find Galvantula from real battle 2632274530")) return 0;
    if (!assert_true(galvantula->fainted == 1 && galvantula->current_hp == 0, "real battle faint path zeroes hp")) return 0;

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, forced_switch_request, 50, 1), "parse real battle 2632274530 forced switch request")) return 0;
    if (!assert_true(req.forced_switch_any == 1 && req.force_switch[1] == 1, "real battle forced switch flags parsed")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632274530 forced switch request")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Cinderace|Cinderace, L82, F|265/265");
    raw_battle_state_update_from_event_line(&state, "|turn|2");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, post_switch_request, 56, 1), "parse real battle 2632274530 post-switch request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632274530 post-switch request")) return 0;

    suicune = find_self(&state, "p2: Suicune");
    cinderace = find_self(&state, "p2: Cinderace");
    if (!assert_true(suicune != NULL && cinderace != NULL, "find current actives from real battle 2632274530")) return 0;
    if (!assert_true(suicune->active == 1 && suicune->active_slot == 1, "Suicune occupies slot1 after real forced switch")) return 0;
    if (!assert_true(cinderace->active == 1 && cinderace->active_slot == 2, "Cinderace occupies slot2 after real forced switch")) return 0;
    if (!assert_true(cinderace->move_ids[0].value == move_id_from_name("uturn"), "real post-switch private moves bind to Cinderace")) return 0;
    if (!assert_true(suicune->tera_type_id.value == type_id_from_name("Grass"), "real post-switch request preserves private tera type")) return 0;
    return 1;
}

static int test_real_battle_2632287191_switch_tera_and_force_switch(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632287191. Clean terminal loss, no disconnect/forfeit. */
    const char* opening_request =
        "{\"active\":["
        "{\"moves\":[{\"move\":\"Heat Wave\",\"id\":\"heatwave\",\"pp\":16,\"maxpp\":16,\"target\":\"allAdjacentFoes\",\"disabled\":false},{\"move\":\"Tailwind\",\"id\":\"tailwind\",\"pp\":24,\"maxpp\":24,\"target\":\"allySide\",\"disabled\":false},{\"move\":\"Blue Flare\",\"id\":\"blueflare\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Draco Meteor\",\"id\":\"dracometeor\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Fire\"},"
        "{\"moves\":[{\"move\":\"Sleep Powder\",\"id\":\"sleeppowder\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Sludge Bomb\",\"id\":\"sludgebomb\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Quiver Dance\",\"id\":\"quiverdance\",\"pp\":32,\"maxpp\":32,\"target\":\"self\",\"disabled\":false},{\"move\":\"Bug Buzz\",\"id\":\"bugbuzz\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Water\"}"
        "],"
        "\"side\":{\"name\":\"Guest 26670187\",\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: Reshiram\",\"details\":\"Reshiram, L72\",\"condition\":\"263/263\",\"active\":true},"
        "{\"ident\":\"p1: Venomoth\",\"details\":\"Venomoth, L90, F\",\"condition\":\"272/272\",\"active\":true},"
        "{\"ident\":\"p1: Sawsbuck\",\"details\":\"Sawsbuck-Summer, L91, M\",\"condition\":\"293/293\",\"active\":false},"
        "{\"ident\":\"p1: Miraidon\",\"details\":\"Miraidon, L65\",\"condition\":\"238/238\",\"active\":false},"
        "{\"ident\":\"p1: Palafin\",\"details\":\"Palafin, L79, M\",\"condition\":\"288/288\",\"active\":false},"
        "{\"ident\":\"p1: Quagsire\",\"details\":\"Quagsire, L91, M\",\"condition\":\"321/321\",\"active\":false}"
        "]},\"rqid\":2}";
    const char* tera_request =
        "{\"active\":["
        "{\"moves\":[{\"move\":\"Protect\",\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},{\"move\":\"Dragon Pulse\",\"id\":\"dragonpulse\",\"pp\":16,\"maxpp\":16,\"target\":\"any\",\"disabled\":false},{\"move\":\"Overheat\",\"id\":\"overheat\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Electro Drift\",\"id\":\"electrodrift\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Electric\"},"
        "{\"moves\":[{\"move\":\"Sleep Powder\",\"id\":\"sleeppowder\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Sludge Bomb\",\"id\":\"sludgebomb\",\"pp\":15,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Quiver Dance\",\"id\":\"quiverdance\",\"pp\":32,\"maxpp\":32,\"target\":\"self\",\"disabled\":false},{\"move\":\"Bug Buzz\",\"id\":\"bugbuzz\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}],\"canTerastallize\":\"Water\"}"
        "],"
        "\"side\":{\"name\":\"Guest 26670187\",\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: Miraidon\",\"details\":\"Miraidon, L65\",\"condition\":\"238/238\",\"active\":true},"
        "{\"ident\":\"p1: Venomoth\",\"details\":\"Venomoth, L90, F\",\"condition\":\"245/272\",\"active\":true},"
        "{\"ident\":\"p1: Sawsbuck\",\"details\":\"Sawsbuck-Summer, L91, M\",\"condition\":\"293/293\",\"active\":false},"
        "{\"ident\":\"p1: Reshiram\",\"details\":\"Reshiram, L72\",\"condition\":\"263/263\",\"active\":false},"
        "{\"ident\":\"p1: Palafin\",\"details\":\"Palafin, L79, M\",\"condition\":\"288/288\",\"active\":false},"
        "{\"ident\":\"p1: Quagsire\",\"details\":\"Quagsire, L91, M\",\"condition\":\"321/321\",\"active\":false}"
        "]},\"rqid\":4}";
    const char* post_tera_request =
        "{\"active\":["
        "{\"moves\":[{\"move\":\"Protect\",\"id\":\"protect\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},{\"move\":\"Dragon Pulse\",\"id\":\"dragonpulse\",\"pp\":15,\"maxpp\":16,\"target\":\"any\",\"disabled\":false},{\"move\":\"Overheat\",\"id\":\"overheat\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Electro Drift\",\"id\":\"electrodrift\",\"pp\":8,\"maxpp\":8,\"target\":\"normal\",\"disabled\":false}]},"
        "{\"moves\":[{\"move\":\"High Horsepower\",\"id\":\"highhorsepower\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Liquidation\",\"id\":\"liquidation\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},{\"move\":\"Icy Wind\",\"id\":\"icywind\",\"pp\":24,\"maxpp\":24,\"target\":\"allAdjacentFoes\",\"disabled\":false},{\"move\":\"Yawn\",\"id\":\"yawn\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}]}"
        "],"
        "\"side\":{\"name\":\"Guest 26670187\",\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: Miraidon\",\"details\":\"Miraidon, L65\",\"condition\":\"131/238\",\"active\":true},"
        "{\"ident\":\"p1: Quagsire\",\"details\":\"Quagsire, L91, M\",\"condition\":\"321/321\",\"active\":true},"
        "{\"ident\":\"p1: Sawsbuck\",\"details\":\"Sawsbuck-Summer, L91, M\",\"condition\":\"293/293\",\"active\":false},"
        "{\"ident\":\"p1: Reshiram\",\"details\":\"Reshiram, L72\",\"condition\":\"263/263\",\"active\":false},"
        "{\"ident\":\"p1: Palafin\",\"details\":\"Palafin, L79, M\",\"condition\":\"288/288\",\"active\":false},"
        "{\"ident\":\"p1: Venomoth\",\"details\":\"Venomoth, L90, F\",\"condition\":\"245/272\",\"active\":false}"
        "]},\"rqid\":6}";
    const char* forced_switch_request =
        "{\"forceSwitch\":[false,true],"
        "\"side\":{\"name\":\"Guest 26670187\",\"id\":\"p1\",\"pokemon\":["
        "{\"ident\":\"p1: Sawsbuck\",\"details\":\"Sawsbuck-Summer, L91, M\",\"condition\":\"184/293\",\"active\":true},"
        "{\"ident\":\"p1: Quagsire\",\"details\":\"Quagsire, L91, M\",\"condition\":\"0 fnt\",\"active\":true},"
        "{\"ident\":\"p1: Reshiram\",\"details\":\"Reshiram, L72\",\"condition\":\"198/263\",\"active\":false},"
        "{\"ident\":\"p1: Venomoth\",\"details\":\"Venomoth, L90, F\",\"condition\":\"0 fnt\",\"active\":false},"
        "{\"ident\":\"p1: Palafin\",\"details\":\"Palafin, L79, M\",\"condition\":\"288/288\",\"active\":false},"
        "{\"ident\":\"p1: Miraidon\",\"details\":\"Miraidon, L65\",\"condition\":\"0 fnt\",\"active\":false}"
        "]},\"noCancel\":true,\"rqid\":20}";
    ParsedRequest req;
    RawBattleState state;
    RawPokemon* miraidon;
    RawPokemon* quagsire;
    RawPokemon* reshiram;

    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Reshiram|Reshiram, L72|263/263");
    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Venomoth|Venomoth, L90, F|272/272");
    raw_battle_state_update_from_event_line(&state, "|switch|p2a: Iron Hands|Iron Hands, L77|100/100");
    raw_battle_state_update_from_event_line(&state, "|switch|p2b: Blastoise|Blastoise, L83, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|turn|1");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, opening_request, 662, 1), "parse real battle 2632287191 opening request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632287191 opening request")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p1a: Miraidon|Miraidon, L65|238/238");
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|move: Electric Terrain|[from] ability: Hadron Engine|[of] p1a: Miraidon");
    raw_battle_state_update_from_event_line(&state, "|move|p1b: Venomoth|Sludge Bomb|p2a: Iron Hands");
    raw_battle_state_update_from_event_line(&state, "|-damage|p2a: Iron Hands|54/100");
    raw_battle_state_update_from_event_line(&state, "|-damage|p1b: Venomoth|245/272|[from] item: Life Orb");
    raw_battle_state_update_from_event_line(&state, "|turn|2");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, tera_request, 689, 1), "parse real battle 2632287191 tera request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632287191 tera request")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Quagsire|Quagsire, L91, M|321/321");
    raw_battle_state_update_from_event_line(&state, "|-terastallize|p1a: Miraidon|Electric");
    raw_battle_state_update_from_event_line(&state, "|move|p2a: Iron Hands|Protect|p2a: Iron Hands");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p2a: Iron Hands|Protect");
    raw_battle_state_update_from_event_line(&state, "|move|p2b: Blastoise|Dragon Pulse|p1a: Miraidon");
    raw_battle_state_update_from_event_line(&state, "|-damage|p1a: Miraidon|131/238");
    raw_battle_state_update_from_event_line(&state, "|turn|3");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, post_tera_request, 705, 1), "parse real battle 2632287191 post-tera request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632287191 post-tera request")) return 0;

    miraidon = find_self(&state, "p1a: Miraidon");
    quagsire = find_self(&state, "p1b: Quagsire");
    if (!assert_true(miraidon != NULL && quagsire != NULL, "find Miraidon and Quagsire after real tera turn")) return 0;
    if (!assert_true(miraidon->tera_used == 1 && miraidon->effective_type1_id.value == type_id_from_name("Electric"), "real tera event updates effective type")) return 0;
    if (!assert_true(quagsire->active == 1 && quagsire->active_slot == 2, "real switch keeps Quagsire in slot2")) return 0;
    if (!assert_true(miraidon->current_hp == 131, "real post-tera request keeps current hp")) return 0;

    raw_battle_state_update_from_event_line(&state, "|-damage|p1b: Quagsire|0 fnt");
    raw_battle_state_update_from_event_line(&state, "|faint|p1b: Quagsire");

    parsed_request_init(&req);
    if (!assert_true(parse_request_payload(&req, forced_switch_request, 812, 1), "parse real battle 2632287191 forced-switch request")) return 0;
    if (!assert_true(raw_battle_state_update_from_request(&state, &req), "apply real battle 2632287191 forced-switch request")) return 0;
    if (!assert_true(parsed_request_slot_needs_choice(&req, 1) == 1 && parsed_request_slot_needs_choice(&req, 0) == 0, "real forced-switch request only requires slot2 choice")) return 0;

    raw_battle_state_update_from_event_line(&state, "|switch|p1b: Reshiram|Reshiram, L72|198/263");
    raw_battle_state_update_from_event_line(&state, "|-damage|p1b: Reshiram|133/263|[from] Stealth Rock");
    reshiram = find_self(&state, "p1b: Reshiram");
    if (!assert_true(reshiram != NULL, "find Reshiram after real forced switch")) return 0;
    if (!assert_true(reshiram->current_hp == 133 && reshiram->active_slot == 2, "real forced switch and follow-up damage bind to new slot2 occupant")) return 0;
    return 1;
}

static int test_real_battle_2632276902_houndoom_tera_and_tailwind(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632276902. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1a: Swalot|Swalot, L90, M|326/326",
        "|switch|p1b: Houndoom|Houndoom, L86, M|269/269",
        "|switch|p2a: Illumise|Illumise, L83, F|100/100",
        "|switch|p2b: Lapras|Lapras, L83, F|100/100",
        "|move|p2a: Illumise|Tailwind|p2a: Illumise",
        "|-sidestart|p2: Beyonces Wig|move: Tailwind",
        "|switch|p1b: Probopass|Probopass, L90, M|254/254",
        "|switch|p1a: Houndoom|Houndoom, L86, M|269/269",
        "|-damage|p1a: Houndoom|249/269",
        "|switch|p1b: Articuno|Articuno, L82|227/282",
        "|-terastallize|p1a: Houndoom|Grass",
        "|-damage|p1a: Houndoom|218/269"
    };
    RawBattleState state;
    RawPokemon* houndoom;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    houndoom = find_self(&state, "p1: Houndoom");
    if (!assert_true(houndoom != NULL, "find Houndoom from real battle 2632276902")) return 0;
    if (!assert_true(state.opp_side.tailwind == 1 && state.opp_side.tailwind_turns == 4, "real battle 2632276902 sets opponent tailwind")) return 0;
    if (!assert_true(houndoom->tera_used == 1 && houndoom->effective_type1_id.value == type_id_from_name("Grass"), "real battle 2632276902 tera updates Houndoom effective type")) return 0;
    if (!assert_true(houndoom->current_hp == 218 && houndoom->active_slot == 1, "real battle 2632276902 preserves Houndoom slot and hp")) return 0;
    return 1;
}

static int test_real_battle_2632278612_salamence_tera_faint_and_whiscash_sleep(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632278612. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1a: Salamence|Salamence, L80, M|283/283",
        "|switch|p1b: Zebstrika|Zebstrika, L87, M|272/272",
        "|switch|p2a: Cryogonal|Cryogonal, L88|100/100",
        "|switch|p2b: Toedscruel|Toedscruel, L87, F|100/100",
        "|-terastallize|p1a: Salamence|Dragon",
        "|-status|p2a: Cryogonal|brn",
        "|-damage|p1a: Salamence|0 fnt",
        "|faint|p1a: Salamence",
        "|switch|p1a: Whiscash|Whiscash, L88, M|337/337",
        "|-status|p1a: Whiscash|slp|[from] move: Spore"
    };
    RawBattleState state;
    RawPokemon* salamence;
    RawPokemon* cryogonal;
    RawPokemon* whiscash;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    salamence = find_self(&state, "p1: Salamence");
    cryogonal = find_opp(&state, "p2: Cryogonal");
    whiscash = find_self(&state, "p1: Whiscash");
    if (!assert_true(salamence != NULL && cryogonal != NULL && whiscash != NULL, "find real battle 2632278612 pokemon")) return 0;
    if (!assert_true(salamence->tera_used == 1 && salamence->fainted == 1 && salamence->current_hp == 0, "real battle 2632278612 tera user later faints cleanly")) return 0;
    if (!assert_true(cryogonal->status_id.value == 1, "real battle 2632278612 burn status parsed")) return 0;
    if (!assert_true(whiscash->status_id.value == 6 && whiscash->active_slot == 1, "real battle 2632278612 switched-in Whiscash is asleep in slot1")) return 0;
    return 1;
}

static int test_real_battle_2632283886_magmortar_tera_and_perrserker_burn_faint(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632283886. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1a: Magmortar|Magmortar, L84, F|263/263",
        "|switch|p1b: Altaria|Altaria, L90, M|281/281",
        "|switch|p2a: Ho-Oh|Ho-Oh, L70|100/100",
        "|switch|p2b: Eelektross|Eelektross, L86, M|100/100",
        "|-terastallize|p1a: Magmortar|Fire",
        "|switch|p1a: Perrserker|Perrserker, L88, M|266/266",
        "|switch|p1b: Ninetales|Ninetales, L79, F|245/245",
        "|-status|p1a: Perrserker|brn",
        "|-damage|p1a: Perrserker|0 fnt",
        "|faint|p1a: Perrserker",
        "|switch|p1a: Magmortar|Magmortar, L84, F, tera:Fire|220/263"
    };
    RawBattleState state;
    RawPokemon* magmortar;
    RawPokemon* perrserker;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    magmortar = find_self(&state, "p1: Magmortar");
    perrserker = find_self(&state, "p1: Perrserker");
    if (!assert_true(magmortar != NULL && perrserker != NULL, "find real battle 2632283886 pokemon")) return 0;
    if (!assert_true(magmortar->tera_used == 1 && magmortar->active_slot == 1 && magmortar->current_hp == 220, "real battle 2632283886 tera Magmortar returns in slot1")) return 0;
    if (!assert_true(perrserker->status_id.value == 1 && perrserker->fainted == 1 && perrserker->current_hp == 0, "real battle 2632283886 burned Perrserker later faints")) return 0;
    return 1;
}

static int test_real_battle_2632285682_dusknoir_paralyzed_dondozo_and_frosmoth_faint(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632285682-0uf4qwm3klbj4p0hlmk79tr9846kke2pw. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1a: Chimecho|Chimecho, L94, F|293/293",
        "|switch|p1b: Dusknoir|Dusknoir, L89, F|225/225",
        "|switch|p2a: Thundurus|Thundurus-Therian, L78, M|100/100",
        "|switch|p2b: Sandaconda|Sandaconda, L88, M|100/100",
        "|switch|p1a: Frosmoth|Frosmoth, L86, M|261/261",
        "|switch|p1b: Dondozo|Dondozo, L85, F|394/394",
        "|-damage|p1b: Dondozo|189/394",
        "|-status|p1b: Dondozo|par",
        "|switch|p1b: Weavile|Weavile, L81, F|246/246",
        "|-damage|p1a: Frosmoth|0 fnt",
        "|faint|p1a: Frosmoth"
    };
    RawBattleState state;
    RawPokemon* dusknoir;
    RawPokemon* dondozo;
    RawPokemon* frosmoth;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    dusknoir = find_self(&state, "p1: Dusknoir");
    dondozo = find_self(&state, "p1: Dondozo");
    frosmoth = find_self(&state, "p1: Frosmoth");
    if (!assert_true(dusknoir != NULL && dondozo != NULL && frosmoth != NULL, "find real battle 2632285682 pokemon")) return 0;
    if (!assert_true(dusknoir->active == 0, "real battle 2632285682 benches Dusknoir after opening switches")) return 0;
    if (!assert_true(dondozo->status_id.value == 2 && dondozo->current_hp == 189, "real battle 2632285682 preserves Dondozo paralysis and hp")) return 0;
    if (!assert_true(frosmoth->fainted == 1 && frosmoth->current_hp == 0, "real battle 2632285682 Frosmoth faint path zeros hp")) return 0;
    return 1;
}

static int test_real_battle_2632291348_wait_replacement_tera_and_faint(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632291348. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1a: Sceptile|Sceptile, L88, M|266/266",
        "|switch|p1b: Shiftry|Shiftry, L83, M|285/285",
        "|switch|p2a: Arboliva|Arboliva, L88, M|100/100",
        "|switch|p2b: Electrode|Electrode, L91|100/100",
        "|switch|p2b: Espathra|Espathra, L83, F|100/100|[from] Volt Switch",
        "|switch|p1a: Necrozma|Necrozma-Dusk-Mane, L70|252/252",
        "|-terastallize|p1b: Shiftry|Ghost",
        "|-damage|p2a: Arboliva|0 fnt",
        "|faint|p2a: Arboliva",
        "|-damage|p1b: Shiftry|0 fnt",
        "|faint|p1b: Shiftry",
        "|switch|p1b: Ambipom|Ambipom, L87, M|272/272"
    };
    RawBattleState state;
    RawPokemon* shiftry;
    RawPokemon* arboliva;
    RawPokemon* ambipom;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    shiftry = find_self(&state, "p1: Shiftry");
    arboliva = find_opp(&state, "p2: Arboliva");
    ambipom = find_self(&state, "p1: Ambipom");
    if (!assert_true(shiftry != NULL && arboliva != NULL && ambipom != NULL, "find real battle 2632291348 pokemon")) return 0;
    if (!assert_true(shiftry->tera_used == 1 && shiftry->fainted == 1 && shiftry->current_hp == 0, "real battle 2632291348 Shiftry teras then faints")) return 0;
    if (!assert_true(arboliva->fainted == 1 && arboliva->current_hp == 0, "real battle 2632291348 Arboliva faint path zeroes hp")) return 0;
    if (!assert_true(ambipom->active == 1 && ambipom->active_slot == 2, "real battle 2632291348 Ambipom replaces slot2 cleanly")) return 0;
    return 1;
}

static int test_real_battle_2632295968_toxic_spikes_and_glalie_poison(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632295968. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1b: Glalie|Glalie, L94, M|303/303",
        "|-status|p1b: Glalie|psn",
        "|-sidestart|p1: Guest 26670187|move: Toxic Spikes"
    };
    RawBattleState state;
    RawPokemon* glalie;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    glalie = find_self(&state, "p1: Glalie");
    if (!assert_true(glalie != NULL, "find Glalie from real battle 2632295968")) return 0;
    if (!assert_true(glalie->status_id.value == 3 && glalie->active_slot == 2, "real battle 2632295968 Glalie enters poisoned in slot2")) return 0;
    if (!assert_true(state.self_side.toxic_spikes == 1, "real battle 2632295968 toxic spikes tracked on self side")) return 0;
    return 1;
}

static int test_real_battle_2632300182_weavile_tera_and_opponent_tailwind(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632300182. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p1a: Weavile|Weavile, L81, M|246/246",
        "|switch|p1b: Kyogre|Kyogre, L65|238/238",
        "|switch|p2a: Drifblim|Drifblim, L85, F|100/100",
        "|switch|p2b: Pelipper|Pelipper, L83, F|100/100",
        "|switch|p1b: Brambleghast|Brambleghast, L86, F|235/235",
        "|-terastallize|p1a: Weavile|Ghost",
        "|-damage|p1b: Brambleghast|25/235",
        "|-heal|p1b: Brambleghast|83/235|[from] item: Sitrus Berry",
        "|-sidestart|p2: qihang14|move: Tailwind"
    };
    RawBattleState state;
    RawPokemon* weavile;
    RawPokemon* brambleghast;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    weavile = find_self(&state, "p1: Weavile");
    brambleghast = find_self(&state, "p1: Brambleghast");
    if (!assert_true(weavile != NULL && brambleghast != NULL, "find real battle 2632300182 pokemon")) return 0;
    if (!assert_true(weavile->tera_used == 1 && weavile->effective_type1_id.value == type_id_from_name("Ghost"), "real battle 2632300182 Weavile tera updates effective type")) return 0;
    if (!assert_true(brambleghast->current_hp == 83, "real battle 2632300182 Brambleghast sitrus heal applied")) return 0;
    if (!assert_true(state.opp_side.tailwind == 1 && state.opp_side.tailwind_turns == 4, "real battle 2632300182 opponent tailwind tracked")) return 0;
    return 1;
}

static int test_real_battle_2632302019_greninja_faint_and_trick_room(void) {
    /* Captured from matches/runs/run_0010_postfix_smoke/run_0010_postfix_smoke_raw.jsonl
       battle-gen9randomdoublesbattle-2632302019. Clean terminal loss, no disconnect/forfeit. */
    const char* lines[] = {
        "|switch|p2a: Hatterene|Hatterene, L84, F|182/233",
        "|switch|p2b: Greninja|Greninja, L83, M|255/255",
        "|-damage|p2b: Greninja|230/255|[from] item: Life Orb",
        "|-damage|p2b: Greninja|0 fnt",
        "|faint|p2b: Greninja",
        "|-damage|p2a: Hatterene|116/233",
        "|-fieldstart|move: Trick Room|[of] p2a: Hatterene"
    };
    RawBattleState state;
    RawPokemon* greninja;
    RawPokemon* hatterene;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    greninja = find_opp(&state, "p2: Greninja");
    hatterene = find_opp(&state, "p2: Hatterene");
    if (!assert_true(greninja != NULL && hatterene != NULL, "find real battle 2632302019 pokemon")) return 0;
    if (!assert_true(greninja->fainted == 1 && greninja->current_hp == 0, "real battle 2632302019 Greninja faint path zeroes hp")) return 0;
    if (!assert_true(hatterene->current_hp == 116, "real battle 2632302019 Hatterene hp updated before Trick Room")) return 0;
    if (!assert_true(state.trick_room == 1 && state.trick_room_turns_remaining == 5, "real battle 2632302019 Trick Room field state tracked")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632274530_three_turn_progression(void) {
    const char* lines[] = {
        "|switch|p1a: Rayquaza|Rayquaza, L75|100/100",
        "|switch|p1b: Alomomola|Alomomola, L96, F|100/100",
        "|switch|p2a: Mabosstiff|Mabosstiff, L84, F|272/272",
        "|switch|p2b: Magcargo|Magcargo, L93, F|262/262",
        "|turn|1",
        "|switch|p2a: Suicune|Suicune, L79|288/288",
        "|switch|p2b: Galvantula|Galvantula, L85, F|258/258",
        "|-damage|p2b: Galvantula|0 fnt",
        "|faint|p2b: Galvantula",
        "|-damage|p2a: Suicune|275/288",
        "|-heal|p2a: Suicune|288/288|[from] item: Leftovers",
        "|switch|p2b: Cinderace|Cinderace, L82, F|265/265",
        "|turn|2",
        "|switch|p2b: Tentacruel|Tentacruel, L85, F|275/275",
        "|switch|p1a: Drednaw|Drednaw, L83, M|100/100",
        "|-terastallize|p2a: Suicune|Grass",
        "|-damage|p2a: Suicune|252/288",
        "|-damage|p2b: Tentacruel|267/275",
        "|-damage|p1a: Drednaw|69/100",
        "|-heal|p2a: Suicune|270/288|[from] item: Leftovers",
        "|turn|3"
    };
    RawBattleState state;
    RawPokemon* suicune;
    RawPokemon* tentacruel;
    RawPokemon* galvantula;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    suicune = find_opp(&state, "p2: Suicune");
    tentacruel = find_opp(&state, "p2: Tentacruel");
    galvantula = find_opp(&state, "p2: Galvantula");
    if (!assert_true(suicune != NULL && tentacruel != NULL && galvantula != NULL, "find real battle 2632274530 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 3, "real battle 2632274530 reaches turn 3")) return 0;
    if (!assert_true(suicune->tera_used == 1 && suicune->current_hp == 270 && suicune->active_slot == 1, "real battle 2632274530 keeps tera Suicune in slot1 across turns")) return 0;
    if (!assert_true(tentacruel->current_hp == 267 && tentacruel->active_slot == 2, "real battle 2632274530 binds turn2 damage to Tentacruel")) return 0;
    if (!assert_true(galvantula->fainted == 1 && galvantula->current_hp == 0, "real battle 2632274530 keeps earlier fainted Galvantula benched and dead")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632276902_houndoom_across_three_turns(void) {
    const char* lines[] = {
        "|switch|p1a: Swalot|Swalot, L90, M|326/326",
        "|switch|p1b: Houndoom|Houndoom, L86, M|269/269",
        "|switch|p2a: Illumise|Illumise, L83, F|100/100",
        "|switch|p2b: Lapras|Lapras, L83, F|100/100",
        "|turn|1",
        "|switch|p1b: Articuno|Articuno, L82|282/282",
        "|switch|p1a: Dragonite|Dragonite, L82, M|283/283",
        "|-sidestart|p2: Beyonces Wig|move: Tailwind",
        "|-damage|p1a: Dragonite|254/283",
        "|-damage|p1b: Articuno|227/282",
        "|turn|2",
        "|switch|p2a: Luvdisc|Luvdisc, M|100/100",
        "|switch|p1b: Probopass|Probopass, L90, M|254/254",
        "|switch|p1a: Houndoom|Houndoom, L86, M|269/269",
        "|-damage|p1a: Houndoom|249/269",
        "|-damage|p1b: Probopass|243/254",
        "|turn|3",
        "|switch|p1b: Articuno|Articuno, L82|227/282",
        "|-terastallize|p1a: Houndoom|Grass",
        "|-damage|p1b: Articuno|152/282",
        "|-damage|p1a: Houndoom|218/269",
        "|-damage|p1b: Articuno|106/282"
    };
    RawBattleState state;
    RawPokemon* houndoom;
    RawPokemon* articuno;
    RawPokemon* probopass;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    houndoom = find_self(&state, "p1: Houndoom");
    articuno = find_self(&state, "p1: Articuno");
    probopass = find_self(&state, "p1: Probopass");
    if (!assert_true(houndoom != NULL && articuno != NULL && probopass != NULL, "find real battle 2632276902 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 3, "real battle 2632276902 reaches turn 3")) return 0;
    if (!assert_true(state.opp_side.tailwind == 1 && state.opp_side.tailwind_turns == 2, "real battle 2632276902 carries opponent tailwind across later turn boundaries")) return 0;
    if (!assert_true(houndoom->tera_used == 1 && houndoom->current_hp == 218 && houndoom->active_slot == 1, "real battle 2632276902 Houndoom survives across turns and teras")) return 0;
    if (!assert_true(articuno->current_hp == 106 && articuno->active_slot == 2, "real battle 2632276902 Articuno re-enters slot2 and takes follow-up damage")) return 0;
    if (!assert_true(probopass->current_hp == 243 && probopass->active == 0, "real battle 2632276902 earlier slot2 damage stays on benched Probopass")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632278612_tera_then_forced_switch_sequence(void) {
    const char* lines[] = {
        "|switch|p1a: Salamence|Salamence, L80, M|283/283",
        "|switch|p1b: Zebstrika|Zebstrika, L87, M|272/272",
        "|switch|p2a: Cryogonal|Cryogonal, L88|100/100",
        "|switch|p2b: Toedscruel|Toedscruel, L87, F|100/100",
        "|turn|1",
        "|-terastallize|p1a: Salamence|Dragon",
        "|-damage|p1a: Salamence|181/283",
        "|-status|p2a: Cryogonal|brn",
        "|-heal|p2a: Cryogonal|51/100 brn|[from] item: Sitrus Berry",
        "|-damage|p1a: Salamence|153/283|[from] item: Life Orb",
        "|-damage|p2a: Cryogonal|45/100 brn|[from] brn",
        "|turn|2",
        "|-damage|p1a: Salamence|0 fnt",
        "|faint|p1a: Salamence",
        "|-damage|p1b: Zebstrika|54/272",
        "|switch|p1a: Whiscash|Whiscash, L88, M|337/337",
        "|turn|3"
    };
    RawBattleState state;
    RawPokemon* salamence;
    RawPokemon* cryogonal;
    RawPokemon* whiscash;
    RawPokemon* zebstrika;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    salamence = find_self(&state, "p1: Salamence");
    cryogonal = find_opp(&state, "p2: Cryogonal");
    whiscash = find_self(&state, "p1: Whiscash");
    zebstrika = find_self(&state, "p1: Zebstrika");
    if (!assert_true(salamence != NULL && cryogonal != NULL && whiscash != NULL && zebstrika != NULL, "find real battle 2632278612 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 3, "real battle 2632278612 reaches turn 3")) return 0;
    if (!assert_true(salamence->tera_used == 1 && salamence->fainted == 1 && salamence->current_hp == 0, "real battle 2632278612 tera Salamence later faints")) return 0;
    if (!assert_true(cryogonal->status_id.value == 1 && cryogonal->current_hp == 45, "real battle 2632278612 Cryogonal remains burned with chip damage")) return 0;
    if (!assert_true(whiscash->active_slot == 1 && whiscash->current_hp == 337, "real battle 2632278612 Whiscash replaces slot1 next turn")) return 0;
    if (!assert_true(zebstrika->current_hp == 54 && zebstrika->active_slot == 2, "real battle 2632278612 Zebstrika stays in slot2 with carried damage")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632283886_tera_tailwind_and_switch_chain(void) {
    const char* lines[] = {
        "|switch|p1a: Magmortar|Magmortar, L84, F|263/263",
        "|switch|p1b: Altaria|Altaria, L90, M|281/281",
        "|switch|p2a: Ho-Oh|Ho-Oh, L70|100/100",
        "|switch|p2b: Eelektross|Eelektross, L86, M|100/100",
        "|turn|1",
        "|-terastallize|p1a: Magmortar|Fire",
        "|-damage|p2a: Ho-Oh|31/100",
        "|-heal|p2a: Ho-Oh|56/100|[from] item: Sitrus Berry",
        "|-sidestart|p2: Woodyf27|move: Tailwind",
        "|-damage|p1a: Magmortar|220/263",
        "|-damage|p1b: Altaria|239/281",
        "|turn|2",
        "|switch|p1a: Arceus|Arceus, L71|288/288",
        "|-heal|p2a: Ho-Oh|100/100",
        "|-damage|p1a: Arceus|240/288",
        "|-damage|p1b: Altaria|199/281",
        "|turn|3"
    };
    RawBattleState state;
    RawPokemon* magmortar;
    RawPokemon* arceus;
    RawPokemon* altaria;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    magmortar = find_self(&state, "p1: Magmortar");
    arceus = find_self(&state, "p1: Arceus");
    altaria = find_self(&state, "p1: Altaria");
    if (!assert_true(magmortar != NULL && arceus != NULL && altaria != NULL, "find real battle 2632283886 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 3, "real battle 2632283886 reaches turn 3")) return 0;
    if (!assert_true(magmortar->tera_used == 1 && magmortar->current_hp == 220 && magmortar->active == 0, "real battle 2632283886 keeps tera state on benched Magmortar")) return 0;
    if (!assert_true(arceus->current_hp == 240 && arceus->active_slot == 1, "real battle 2632283886 turn2 switch puts Arceus in slot1")) return 0;
    if (!assert_true(altaria->current_hp == 199 && altaria->active_slot == 2, "real battle 2632283886 Altaria remains active in slot2 across turns")) return 0;
    if (!assert_true(state.opp_side.tailwind == 1, "real battle 2632283886 opponent tailwind persists during slice")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632285682_opening_replacements_over_two_turns(void) {
    const char* lines[] = {
        "|switch|p1a: Chimecho|Chimecho, L94, F|293/293",
        "|switch|p1b: Dusknoir|Dusknoir, L89, F|225/225",
        "|switch|p2a: Thundurus|Thundurus-Therian, L78, M|100/100",
        "|switch|p2b: Sandaconda|Sandaconda, L88, M|100/100",
        "|turn|1",
        "|switch|p1a: Frosmoth|Frosmoth, L86, M|261/261",
        "|switch|p1b: Dondozo|Dondozo, L85, F|394/394",
        "|-damage|p1a: Frosmoth|204/261",
        "|-damage|p1b: Dondozo|189/394",
        "|-status|p1b: Dondozo|par",
        "|turn|2",
        "|switch|p1b: Weavile|Weavile, L81, F|246/246",
        "|-damage|p1a: Frosmoth|0 fnt",
        "|faint|p1a: Frosmoth"
    };
    RawBattleState state;
    RawPokemon* dusknoir;
    RawPokemon* frosmoth;
    RawPokemon* dondozo;
    RawPokemon* weavile;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    dusknoir = find_self(&state, "p1: Dusknoir");
    frosmoth = find_self(&state, "p1: Frosmoth");
    dondozo = find_self(&state, "p1: Dondozo");
    weavile = find_self(&state, "p1: Weavile");
    if (!assert_true(dusknoir != NULL && frosmoth != NULL && dondozo != NULL && weavile != NULL, "find real battle 2632285682 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 2, "real battle 2632285682 reaches turn 2")) return 0;
    if (!assert_true(dusknoir->active == 0, "real battle 2632285682 initial Dusknoir stays benched after replacement")) return 0;
    if (!assert_true(dondozo->current_hp == 189 && dondozo->status_id.value == 2, "real battle 2632285682 Dondozo keeps turn1 hp and paralysis")) return 0;
    if (!assert_true(frosmoth->fainted == 1 && frosmoth->current_hp == 0, "real battle 2632285682 Frosmoth dies on turn 2")) return 0;
    if (!assert_true(weavile->active_slot == 2 && weavile->current_hp == 246, "real battle 2632285682 Weavile occupies slot2 after replacement")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632287191_miraidon_to_quagsire_sequence(void) {
    const char* lines[] = {
        "|switch|p1a: Reshiram|Reshiram, L72|263/263",
        "|switch|p1b: Venomoth|Venomoth, L90, F|272/272",
        "|switch|p2a: Iron Hands|Iron Hands, L77|100/100",
        "|switch|p2b: Blastoise|Blastoise, L83, M|100/100",
        "|turn|1",
        "|switch|p1a: Miraidon|Miraidon, L65|238/238",
        "|-fieldstart|move: Electric Terrain|[from] ability: Hadron Engine|[of] p1a: Miraidon",
        "|-damage|p2a: Iron Hands|54/100",
        "|-damage|p1b: Venomoth|245/272|[from] item: Life Orb",
        "|turn|2",
        "|switch|p1b: Quagsire|Quagsire, L91, M|321/321",
        "|-terastallize|p1a: Miraidon|Electric",
        "|-damage|p1a: Miraidon|131/238",
        "|turn|3"
    };
    RawBattleState state;
    RawPokemon* miraidon;
    RawPokemon* quagsire;
    RawPokemon* venomoth;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    miraidon = find_self(&state, "p1: Miraidon");
    quagsire = find_self(&state, "p1: Quagsire");
    venomoth = find_self(&state, "p1: Venomoth");
    if (!assert_true(miraidon != NULL && quagsire != NULL && venomoth != NULL, "find real battle 2632287191 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 3, "real battle 2632287191 reaches turn 3")) return 0;
    if (!assert_true(miraidon->tera_used == 1 && miraidon->current_hp == 131 && miraidon->active_slot == 1, "real battle 2632287191 Miraidon teras and stays in slot1")) return 0;
    if (!assert_true(quagsire->active_slot == 2 && quagsire->current_hp == 321, "real battle 2632287191 Quagsire takes over slot2")) return 0;
    if (!assert_true(venomoth->current_hp == 245 && venomoth->active == 0, "real battle 2632287191 Venomoth retains turn1 recoil after being benched")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632291348_replacement_chain_over_three_turns(void) {
    const char* lines[] = {
        "|switch|p1a: Sceptile|Sceptile, L88, M|266/266",
        "|switch|p1b: Shiftry|Shiftry, L83, M|285/285",
        "|switch|p2a: Arboliva|Arboliva, L88, M|100/100",
        "|switch|p2b: Electrode|Electrode, L91|100/100",
        "|turn|1",
        "|switch|p2b: Espathra|Espathra, L83, F|100/100|[from] Volt Switch",
        "|turn|2",
        "|switch|p1a: Necrozma|Necrozma-Dusk-Mane, L70|252/252",
        "|-terastallize|p1b: Shiftry|Ghost",
        "|-damage|p2a: Arboliva|0 fnt",
        "|faint|p2a: Arboliva",
        "|turn|3",
        "|-damage|p1b: Shiftry|0 fnt",
        "|faint|p1b: Shiftry",
        "|switch|p1b: Ambipom|Ambipom, L87, M|272/272"
    };
    RawBattleState state;
    RawPokemon* necrozma;
    RawPokemon* shiftry;
    RawPokemon* ambipom;
    RawPokemon* arboliva;
    RawPokemon* espathra;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    necrozma = find_self(&state, "p1: Necrozma");
    shiftry = find_self(&state, "p1: Shiftry");
    ambipom = find_self(&state, "p1: Ambipom");
    arboliva = find_opp(&state, "p2: Arboliva");
    espathra = find_opp(&state, "p2: Espathra");
    if (!assert_true(necrozma != NULL && shiftry != NULL && ambipom != NULL && arboliva != NULL && espathra != NULL, "find real battle 2632291348 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 3, "real battle 2632291348 reaches turn 3")) return 0;
    if (!assert_true(necrozma->active_slot == 1 && necrozma->current_hp == 252, "real battle 2632291348 Necrozma claims slot1 cleanly")) return 0;
    if (!assert_true(shiftry->tera_used == 1 && shiftry->fainted == 1, "real battle 2632291348 Shiftry teras before fainting")) return 0;
    if (!assert_true(arboliva->fainted == 1 && arboliva->current_hp == 0, "real battle 2632291348 Arboliva dies earlier in the chain")) return 0;
    if (!assert_true(ambipom->active_slot == 2 && espathra->active_slot == 2, "real battle 2632291348 both sides keep slot2 replacements aligned")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632295968_turn_one_and_two_continuity(void) {
    const char* lines[] = {
        "|switch|p1a: Lurantis|Lurantis, L85, M|258/258",
        "|switch|p1b: Minior|Minior, L83|232/232",
        "|switch|p2a: Meowscarada|Meowscarada, L79, F|100/100",
        "|switch|p2b: Qwilfish|Qwilfish-Hisui, L82, F|100/100",
        "|-formechange|p1b: Minior|Minior-Meteor||[from] ability: Shields Down",
        "|turn|1",
        "|switch|p1b: Flutter Mane|Flutter Mane, L73|201/201",
        "|-terastallize|p1a: Lurantis|Fighting",
        "|-damage|p1a: Lurantis|123/258",
        "|-damage|p1b: Flutter Mane|186/201",
        "|-damage|p2a: Meowscarada|0 fnt",
        "|faint|p2a: Meowscarada",
        "|turn|2",
        "|switch|p2a: Hawlucha|Hawlucha, L84, F|100/100",
        "|-damage|p1a: Lurantis|0 fnt",
        "|faint|p1a: Lurantis",
        "|-sidestart|p1: Guest 26670187|move: Toxic Spikes"
    };
    RawBattleState state;
    RawPokemon* lurantis;
    RawPokemon* flutter_mane;
    RawPokemon* meowscarada;
    RawPokemon* hawlucha;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    lurantis = find_self(&state, "p1: Lurantis");
    flutter_mane = find_self(&state, "p1: Flutter Mane");
    meowscarada = find_opp(&state, "p2: Meowscarada");
    hawlucha = find_opp(&state, "p2: Hawlucha");
    if (!assert_true(lurantis != NULL && flutter_mane != NULL && meowscarada != NULL && hawlucha != NULL, "find real battle 2632295968 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 2, "real battle 2632295968 reaches turn 2")) return 0;
    if (!assert_true(lurantis->tera_used == 1 && lurantis->fainted == 1 && lurantis->current_hp == 0, "real battle 2632295968 tera Lurantis later faints")) return 0;
    if (!assert_true(flutter_mane->current_hp == 186 && flutter_mane->active_slot == 2, "real battle 2632295968 Flutter Mane keeps slot2 damage across turn boundary")) return 0;
    if (!assert_true(meowscarada->fainted == 1 && hawlucha->active_slot == 1, "real battle 2632295968 opponent slot1 replacement follows the faint")) return 0;
    if (!assert_true(state.self_side.toxic_spikes == 1, "real battle 2632295968 toxic spikes appear later in the sequence")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632300182_weather_tera_tailwind_sequence(void) {
    const char* lines[] = {
        "|switch|p1a: Weavile|Weavile, L81, M|246/246",
        "|switch|p1b: Kyogre|Kyogre, L65|238/238",
        "|switch|p2a: Drifblim|Drifblim, L85, F|100/100",
        "|switch|p2b: Pelipper|Pelipper, L83, F|100/100",
        "|-weather|RainDance|[from] ability: Drizzle|[of] p1b: Kyogre",
        "|turn|1",
        "|switch|p1b: Brambleghast|Brambleghast, L86, F|235/235",
        "|-terastallize|p1a: Weavile|Ghost",
        "|-damage|p1b: Brambleghast|25/235",
        "|-heal|p1b: Brambleghast|83/235|[from] item: Sitrus Berry",
        "|-sidestart|p2: qihang14|move: Tailwind",
        "|turn|2"
    };
    RawBattleState state;
    RawPokemon* weavile;
    RawPokemon* brambleghast;
    RawPokemon* kyogre;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    weavile = find_self(&state, "p1: Weavile");
    brambleghast = find_self(&state, "p1: Brambleghast");
    kyogre = find_self(&state, "p1: Kyogre");
    if (!assert_true(weavile != NULL && brambleghast != NULL && kyogre != NULL, "find real battle 2632300182 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 2, "real battle 2632300182 reaches turn 2")) return 0;
    if (!assert_true(state.weather_id != 0, "real battle 2632300182 rain starts from opening ability")) return 0;
    if (!assert_true(weavile->tera_used == 1 && weavile->active_slot == 1, "real battle 2632300182 Weavile remains tera slot1 attacker")) return 0;
    if (!assert_true(brambleghast->current_hp == 83 && brambleghast->active_slot == 2, "real battle 2632300182 Brambleghast survives and heals in slot2")) return 0;
    if (!assert_true(kyogre->active == 0, "real battle 2632300182 Kyogre is benched after turn1 replacement")) return 0;
    return 1;
}

static int test_multiturn_real_battle_2632302019_trick_room_carries_to_turn_ten(void) {
    const char* lines[] = {
        "|switch|p2a: Hatterene|Hatterene, L84, F|182/233",
        "|switch|p2b: Greninja|Greninja, L83, M|255/255",
        "|turn|8",
        "|-damage|p2b: Greninja|230/255|[from] item: Life Orb",
        "|-damage|p2b: Greninja|0 fnt",
        "|faint|p2b: Greninja",
        "|-damage|p2a: Hatterene|116/233",
        "|-fieldstart|move: Trick Room|[of] p2a: Hatterene",
        "|turn|9",
        "|switch|p1a: Leafeon|Leafeon, L90, M|100/100",
        "|move|p2a: Hatterene|Protect|p2a: Hatterene",
        "|-singleturn|p2a: Hatterene|Protect",
        "|turn|10"
    };
    RawBattleState state;
    RawPokemon* greninja;
    RawPokemon* hatterene;

    raw_battle_state_init(&state, 1);
    apply_event_lines(&state, lines, sizeof(lines) / sizeof(lines[0]));

    greninja = find_opp(&state, "p2: Greninja");
    hatterene = find_opp(&state, "p2: Hatterene");
    if (!assert_true(greninja != NULL && hatterene != NULL, "find real battle 2632302019 multi-turn pokemon")) return 0;
    if (!assert_true(state.turn_number == 10, "real battle 2632302019 reaches turn 10")) return 0;
    if (!assert_true(state.trick_room == 1 && state.trick_room_turns_remaining == 3, "real battle 2632302019 Trick Room remains tracked after later turn markers")) return 0;
    if (!assert_true(greninja->fainted == 1 && greninja->current_hp == 0, "real battle 2632302019 keeps Greninja fainted on later turns")) return 0;
    if (!assert_true(hatterene->current_hp == 116 && hatterene->active_slot == 1, "real battle 2632302019 Hatterene stays in slot1 with preserved hp")) return 0;
    return 1;
}

static int test_multiturn_capture_replay_2632274530_turn2_request_state(void) {
    CaptureReplayResult replay;
    RawPokemon* suicune;
    RawPokemon* cinderace;
    RawPokemon* galvantula;
    if (!assert_true(replay_capture_battle("battle-gen9randomdoublesbattle-2632274530", 2, &replay), "replay battle 2632274530 to turn2 request")) return 0;
    if (!assert_capture_replay_sane(&replay, "turn2 replay 2632274530")) return 0;
    suicune = find_self(&replay.state, "p2: Suicune");
    cinderace = find_self(&replay.state, "p2: Cinderace");
    galvantula = find_self(&replay.state, "p2: Galvantula");
    if (!assert_true(suicune != NULL && cinderace != NULL && galvantula != NULL, "find battle 2632274530 turn2 replay mons")) return 0;
    if (!assert_true(replay.state.turn_number == 2, "battle 2632274530 replay reaches turn 2 request")) return 0;
    if (!assert_true(suicune->active_slot == 1 && cinderace->active_slot == 2, "battle 2632274530 replay keeps Suicune and Cinderace active")) return 0;
    if (!assert_true(galvantula->fainted == 1 && galvantula->current_hp == 0, "battle 2632274530 replay keeps Galvantula fainted")) return 0;
    return 1;
}

static int test_multiturn_capture_replay_2632288269_turn2_request_state(void) {
    CaptureReplayResult replay;
    RawPokemon* koraidon;
    RawPokemon* heracross;
    RawPokemon* espeon;
    RawPokemon* walking_wake;
    if (!assert_true(replay_capture_battle("battle-gen9randomdoublesbattle-2632288269-9llqfqt4j5c1nkpaincla1sfxa3rsknpw", 2, &replay), "replay battle 2632288269 to turn2 request")) return 0;
    if (!assert_capture_replay_sane(&replay, "turn2 replay 2632288269")) return 0;
    koraidon = find_self(&replay.state, "p1: Koraidon");
    heracross = find_self(&replay.state, "p1: Heracross");
    espeon = find_opp(&replay.state, "p2: Espeon");
    walking_wake = find_opp(&replay.state, "p2: Walking Wake");
    if (!assert_true(koraidon != NULL && heracross != NULL && espeon != NULL && walking_wake != NULL, "find battle 2632288269 turn2 replay mons")) return 0;
    if (!assert_true(replay.state.turn_number == 2, "battle 2632288269 replay reaches turn 2 request")) return 0;
    if (!assert_true(replay.state.weather_id == 1, "battle 2632288269 replay keeps sun active")) return 0;
    if (!assert_true(replay.state.opp_side.tailwind == 1 && replay.state.opp_side.tailwind_turns == 3, "battle 2632288269 replay keeps opponent tailwind into turn 2")) return 0;
    if (!assert_true(espeon->fainted == 1 && espeon->current_hp == 0, "battle 2632288269 replay keeps Espeon fainted")) return 0;
    if (!assert_true(koraidon->active_slot == 1 && heracross->active_slot == 2 && walking_wake->active_slot == 1, "battle 2632288269 replay keeps slot ownership consistent")) return 0;
    return 1;
}

static int test_multiturn_capture_replay_2632290515_turn2_request_state(void) {
    CaptureReplayResult replay;
    RawPokemon* magmortar;
    RawPokemon* tyranitar;
    RawPokemon* hoopa;
    if (!assert_true(replay_capture_battle("battle-gen9randomdoublesbattle-2632290515", 2, &replay), "replay battle 2632290515 to turn2 request")) return 0;
    if (!assert_capture_replay_sane(&replay, "turn2 replay 2632290515")) return 0;
    magmortar = find_self(&replay.state, "p1: Magmortar");
    tyranitar = find_self(&replay.state, "p1: Tyranitar");
    hoopa = find_self(&replay.state, "p1: Hoopa");
    if (!assert_true(magmortar != NULL && tyranitar != NULL && hoopa != NULL, "find battle 2632290515 turn2 replay mons")) return 0;
    if (!assert_true(replay.state.turn_number == 2, "battle 2632290515 replay reaches turn 2 request")) return 0;
    if (!assert_true(replay.state.weather_id == 3, "battle 2632290515 replay tracks sand after Tyranitar switch")) return 0;
    if (!assert_true(magmortar->tera_used == 1 && magmortar->active_slot == 1, "battle 2632290515 replay keeps Magmortar tera in slot1")) return 0;
    if (!assert_true(tyranitar->active_slot == 2 && hoopa->active == 0, "battle 2632290515 replay keeps Tyranitar replacing Hoopa")) return 0;
    return 1;
}

static int test_multiturn_capture_replay_2632293423_turn2_request_state(void) {
    CaptureReplayResult replay;
    RawPokemon* gastrodon;
    RawPokemon* yanmega;
    RawPokemon* lilligant;
    RawPokemon* squawkabilly;
    if (!assert_true(replay_capture_battle("battle-gen9randomdoublesbattle-2632293423-qlib5swp8plrnl8yyns4hl0yu31h2kjpw", 2, &replay), "replay battle 2632293423 to turn2 request")) return 0;
    if (!assert_capture_replay_sane(&replay, "turn2 replay 2632293423")) return 0;
    gastrodon = find_self(&replay.state, "p1: Gastrodon");
    yanmega = find_self(&replay.state, "p1: Yanmega");
    lilligant = find_opp(&replay.state, "p2: Lilligant");
    squawkabilly = find_self(&replay.state, "p1: Squawkabilly");
    if (!assert_true(gastrodon != NULL && yanmega != NULL && lilligant != NULL && squawkabilly != NULL, "find battle 2632293423 turn2 replay mons")) return 0;
    if (!assert_true(replay.state.turn_number == 2, "battle 2632293423 replay reaches turn 2 request")) return 0;
    if (!assert_true(gastrodon->active_slot == 1 && yanmega->active_slot == 2 && squawkabilly->active == 0, "battle 2632293423 replay keeps U-turn replacement aligned")) return 0;
    if (!assert_true(lilligant->yawn_active == 1, "battle 2632293423 replay keeps Lilligant yawned")) return 0;
    return 1;
}

static int test_multiturn_capture_replay_2632310612_turn2_request_state(void) {
    CaptureReplayResult replay;
    RawPokemon* amoonguss;
    RawPokemon* lumineon;
    RawPokemon* clefairy;
    if (!assert_true(replay_capture_battle("battle-gen9randomdoublesbattle-2632310612", 2, &replay), "replay battle 2632310612 to turn2 request")) return 0;
    if (!assert_capture_replay_sane(&replay, "turn2 replay 2632310612")) return 0;
    amoonguss = find_self(&replay.state, "p2: Amoonguss");
    lumineon = find_self(&replay.state, "p2: Lumineon");
    clefairy = find_self(&replay.state, "p2: Clefairy");
    if (!assert_true(amoonguss != NULL && lumineon != NULL && clefairy != NULL, "find battle 2632310612 turn2 replay mons")) return 0;
    if (!assert_true(replay.state.turn_number == 2, "battle 2632310612 replay reaches turn 2 request")) return 0;
    if (!assert_true(amoonguss->active_slot == 1 && lumineon->active_slot == 2, "battle 2632310612 replay keeps Amoonguss and Lumineon active")) return 0;
    if (!assert_true(clefairy->fainted == 1 && clefairy->current_hp == 0, "battle 2632310612 replay keeps Clefairy fainted")) return 0;
    return 1;
}

static int assert_full_battle_replay(const char* battle_id, int expected_win, int min_turn, const char* label) {
    CaptureReplayResult replay;
    char message[160];
    if (!assert_true(replay_capture_battle(battle_id, 0, &replay), label)) return 0;
    if (!assert_capture_replay_sane(&replay, label)) return 0;
    snprintf(message, sizeof(message), "%s reaches terminal", label);
    if (!assert_true(replay.saw_terminal == 1, message)) return 0;
    snprintf(message, sizeof(message), "%s matches expected result", label);
    if (!assert_true(replay.terminal_is_win == expected_win, message)) return 0;
    snprintf(message, sizeof(message), "%s avoids disconnect terminal", label);
    if (!assert_true(replay.saw_disconnect_loss == 0, message)) return 0;
    snprintf(message, sizeof(message), "%s includes battle start", label);
    if (!assert_true(replay.saw_battle_start == 1, message)) return 0;
    snprintf(message, sizeof(message), "%s includes requests", label);
    if (!assert_true(replay.request_count > 0, message)) return 0;
    snprintf(message, sizeof(message), "%s reaches a meaningful turn count", label);
    if (!assert_true(replay.state.turn_number >= min_turn, message)) return 0;
    return 1;
}

static int test_full_battle_replay_2632274530(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632274530", 0, 7, "full replay 2632274530");
}

static int test_full_battle_replay_2632276902(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632276902", 0, 6, "full replay 2632276902");
}

static int test_full_battle_replay_2632278612(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632278612", 0, 5, "full replay 2632278612");
}

static int test_full_battle_replay_2632283886(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632283886", 0, 10, "full replay 2632283886");
}

static int test_full_battle_replay_2632285682(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632285682-0uf4qwm3klbj4p0hlmk79tr9846kke2pw", 0, 4, "full replay 2632285682");
}

static int test_full_battle_replay_2632287191(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632287191", 0, 4, "full replay 2632287191");
}

static int test_full_battle_replay_2632288269(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632288269-9llqfqt4j5c1nkpaincla1sfxa3rsknpw", 0, 5, "full replay 2632288269");
}

static int test_full_battle_replay_2632290515(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632290515", 0, 5, "full replay 2632290515");
}

static int test_full_battle_replay_2632293423(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632293423-qlib5swp8plrnl8yyns4hl0yu31h2kjpw", 1, 3, "full replay 2632293423");
}

static int test_full_battle_replay_2632310612(void) {
    return assert_full_battle_replay("battle-gen9randomdoublesbattle-2632310612", 1, 4, "full replay 2632310612");
}

static int test_full_battle_replay_2636632844_gliscor_terminal_state(void) {
    CaptureReplayResult replay;
    RawPokemon* gliscor;
    int self_remaining;
    int opp_remaining;
    if (!assert_true(replay_capture_battle_from_path(TEST_RANDOM_CAPTURE_PATH, "battle-gen9randomdoublesbattle-2636632844", 0, &replay), "full replay 2636632844")) return 0;
    if (!assert_capture_replay_sane(&replay, "full replay 2636632844")) return 0;
    if (!assert_true(replay.saw_terminal == 1 && replay.terminal_is_win == 1, "full replay 2636632844 reaches clean win terminal")) return 0;
    gliscor = find_opp(&replay.state, "p1: Gliscor");
    if (!assert_true(gliscor != NULL, "find Gliscor in full replay 2636632844")) return 0;
    if (!assert_true(gliscor->fainted == 1 && gliscor->current_hp == 0, "full replay 2636632844 keeps Gliscor fainted at terminal")) return 0;
    self_remaining = replay.state.self_side.remaining_pokemon;
    opp_remaining = replay.state.opp_side.remaining_pokemon;
    if (!assert_true(self_remaining == 2, "full replay 2636632844 keeps two self mons alive at terminal")) return 0;
    if (!assert_true(opp_remaining == 0, "full replay 2636632844 fully clears the losing side at terminal")) return 0;
    return 1;
}

static int test_full_battle_replay_2637505742_zeroes_fainted_reserve_hp(void) {
    CaptureReplayResult replay;
    RawPokemon* murkrow;
    RawPokemon* krookodile;
    RawPokemon* hitmonchan;
    RawPokemon* iron_leaves;
    if (!assert_true(replay_capture_battle_from_path(TEST_RANDOM_CAPTURE_PATH, "battle-gen9randomdoublesbattle-2637505742-hksiymfrrflfie2b6yqnvo8sh0w06gupw", 0, &replay), "full replay 2637505742")) return 0;
    if (!assert_capture_replay_sane(&replay, "full replay 2637505742")) return 0;
    if (!assert_true(replay.saw_terminal == 1 && replay.terminal_is_win == 1, "full replay 2637505742 reaches clean win terminal")) return 0;
    murkrow = find_self(&replay.state, "p1: Murkrow");
    krookodile = find_self(&replay.state, "p1: Krookodile");
    hitmonchan = find_self(&replay.state, "p1: Hitmonchan");
    iron_leaves = find_self(&replay.state, "p1: Iron Leaves");
    if (!assert_true(murkrow != NULL && krookodile != NULL && hitmonchan != NULL && iron_leaves != NULL, "find fainted reserves in full replay 2637505742")) return 0;
    if (!assert_true(murkrow->fainted == 1 && murkrow->current_hp == 0, "full replay 2637505742 zeroes Murkrow reserve hp")) return 0;
    if (!assert_true(krookodile->fainted == 1 && krookodile->current_hp == 0, "full replay 2637505742 zeroes Krookodile reserve hp")) return 0;
    if (!assert_true(hitmonchan->fainted == 1 && hitmonchan->current_hp == 0, "full replay 2637505742 zeroes Hitmonchan reserve hp")) return 0;
    if (!assert_true(iron_leaves->fainted == 1 && iron_leaves->current_hp == 0, "full replay 2637505742 zeroes Iron Leaves reserve hp")) return 0;
    return 1;
}

static int test_synthetic_sideeffect_prefix_tailwind(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Tailwind");
    return assert_true(state.self_side.tailwind == 1 && state.self_side.tailwind_turns == 4, "synthetic move: Tailwind side effect parses");
}

static int test_synthetic_sideeffect_prefix_reflect(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Reflect");
    return assert_true(state.self_side.reflect == 1 && state.self_side.reflect_turns == 5, "synthetic move: Reflect side effect parses");
}

static int test_synthetic_sideeffect_prefix_light_screen(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Light Screen");
    return assert_true(state.self_side.light_screen == 1 && state.self_side.light_screen_turns == 5, "synthetic move: Light Screen side effect parses");
}

static int test_synthetic_sideeffect_prefix_aurora_veil(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Aurora Veil");
    return assert_true(state.self_side.aurora_veil == 1 && state.self_side.aurora_veil_turns == 5, "synthetic move: Aurora Veil side effect parses");
}

static int test_synthetic_sideeffect_prefix_safeguard(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Safeguard");
    return assert_true(state.self_side.safeguard == 1 && state.self_side.safeguard_turns == 5, "synthetic move: Safeguard side effect parses");
}

static int test_synthetic_sideeffect_prefix_mist(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Mist");
    return assert_true(state.self_side.mist == 1 && state.self_side.mist_turns == 5, "synthetic move: Mist side effect parses");
}

static int test_synthetic_sideeffect_prefix_lucky_chant(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Lucky Chant");
    return assert_true(state.self_side.lucky_chant == 1 && state.self_side.lucky_chant_turns == 5, "synthetic move: Lucky Chant side effect parses");
}

static int test_synthetic_sideeffect_prefix_toxic_spikes(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Toxic Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Toxic Spikes");
    return assert_true(state.self_side.toxic_spikes == 2, "synthetic move: Toxic Spikes layers parse and cap");
}

static int test_synthetic_sideeffect_prefix_sticky_web(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Sticky Web");
    return assert_true(state.self_side.sticky_web == 1, "synthetic move: Sticky Web side effect parses");
}

static int test_synthetic_sideeffect_prefix_spikes_cap(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Spikes");
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Spikes");
    return assert_true(state.self_side.spikes == 3, "synthetic move: Spikes caps at 3");
}

static int test_synthetic_weather_overwrite_resets_counter(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-weather|RainDance");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    raw_battle_state_update_from_event_line(&state, "|-weather|SunnyDay");
    return assert_true(state.weather_id == 1 && state.weather_turns_remaining.value == 5, "synthetic weather overwrite resets duration");
}

static int test_synthetic_terrain_overwrite_resets_counter(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|Electric Terrain");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|Grassy Terrain");
    return assert_true(state.terrain_id == 2 && state.terrain_turns_remaining.value == 5, "synthetic terrain overwrite resets duration");
}

static int test_synthetic_field_end_clears_trick_room(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|move: Trick Room");
    raw_battle_state_update_from_event_line(&state, "|-fieldend|move: Trick Room");
    return assert_true(state.trick_room == 0 && state.trick_room_turns_remaining == 0, "synthetic Trick Room field end clears state");
}

static int test_synthetic_field_end_clears_gravity(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-fieldstart|move: Gravity");
    raw_battle_state_update_from_event_line(&state, "|-fieldend|move: Gravity");
    return assert_true(state.gravity == 0 && state.gravity_turns_remaining == 0, "synthetic Gravity field end clears state");
}

static int test_synthetic_tailwind_expires_after_four_turns(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Tailwind");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    raw_battle_state_update_from_event_line(&state, "|turn|3");
    raw_battle_state_update_from_event_line(&state, "|turn|4");
    raw_battle_state_update_from_event_line(&state, "|turn|5");
    return assert_true(state.self_side.tailwind == 0 && state.self_side.tailwind_turns == 0, "synthetic Tailwind expires after four turn markers");
}

static int test_synthetic_reflect_expires_after_five_turns(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Reflect");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    raw_battle_state_update_from_event_line(&state, "|turn|3");
    raw_battle_state_update_from_event_line(&state, "|turn|4");
    raw_battle_state_update_from_event_line(&state, "|turn|5");
    raw_battle_state_update_from_event_line(&state, "|turn|6");
    return assert_true(state.self_side.reflect == 0 && state.self_side.reflect_turns == 0, "synthetic Reflect expires after five turn markers");
}

static int test_synthetic_sleep_does_not_progress_while_benched(void) {
    RawBattleState state;
    RawPokemon* pokemon;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-status|p1a: A|slp");
    pokemon = find_self(&state, "p1: A");
    if (!assert_true(pokemon != NULL, "find sleeping pokemon")) return 0;
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: B|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    return assert_true(pokemon->sleep_turns_elapsed == 0, "synthetic benching stops sleep turn progression");
}

static int test_synthetic_toxic_does_not_progress_while_benched(void) {
    RawBattleState state;
    RawPokemon* pokemon;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-status|p1a: A|tox");
    pokemon = find_self(&state, "p1: A");
    if (!assert_true(pokemon != NULL, "find toxic pokemon")) return 0;
    pokemon->toxic_counter = 2;
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: B|Kingambit, L77, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    return assert_true(pokemon->toxic_counter == 0, "synthetic benching clears and stops toxic progression");
}

static int test_synthetic_singleturn_flags_clear_on_next_turn(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Protect");
    raw_battle_state_update_from_event_line(&state, "|-singleturn|p1a: A|Helping Hand");
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    return assert_true(find_self(&state, "p1: A")->protect_active == 0 && find_self(&state, "p1: A")->helping_hand_active == 0,
        "synthetic single-turn flags clear on next turn");
}

static int test_synthetic_yawn_duration_expires_across_turns(void) {
    RawBattleState state;
    RawPokemon* pokemon;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|switch|p1a: A|Sawsbuck, L91, M|100/100");
    raw_battle_state_update_from_event_line(&state, "|-start|p1a: A|move: Yawn");
    pokemon = find_self(&state, "p1: A");
    if (!assert_true(pokemon != NULL, "find yawned pokemon")) return 0;
    raw_battle_state_update_from_event_line(&state, "|turn|2");
    raw_battle_state_update_from_event_line(&state, "|turn|3");
    return assert_true(pokemon->yawn_active == 0 && pokemon->yawn_turns == 0, "synthetic Yawn duration expires over turn boundaries");
}

static int test_synthetic_sideend_clears_reflect(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Reflect");
    raw_battle_state_update_from_event_line(&state, "|-sideend|p1: Tester|Reflect");
    return assert_true(state.self_side.reflect == 0 && state.self_side.reflect_turns == 0, "synthetic sideend clears Reflect");
}

static int test_synthetic_sideend_clears_tailwind(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Tailwind");
    raw_battle_state_update_from_event_line(&state, "|-sideend|p1: Tester|Tailwind");
    return assert_true(state.self_side.tailwind == 0 && state.self_side.tailwind_turns == 0, "synthetic sideend clears Tailwind");
}

static int test_synthetic_sideend_clears_sticky_web(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-sidestart|p1: Tester|move: Sticky Web");
    raw_battle_state_update_from_event_line(&state, "|-sideend|p1: Tester|Sticky Web");
    return assert_true(state.self_side.sticky_web == 0, "synthetic sideend clears Sticky Web");
}

static int test_synthetic_weather_clear_sets_unknown_duration(void) {
    RawBattleState state;
    raw_battle_state_init(&state, 1);
    raw_battle_state_update_from_event_line(&state, "|-weather|RainDance");
    raw_battle_state_update_from_event_line(&state, "|-weather|none");
    return assert_true(state.weather_id == 0 && state.weather_turns_remaining.knowledge == KNOW_UNKNOWN, "synthetic weather clear resets id and duration knowledge");
}

typedef struct {
    char magic[8];
    unsigned int version;
    size_t input_dim;
    size_t hidden_dim;
    size_t num_actions;
    size_t parameter_count;
    TrainerCheckpointState trainer;
} TestCheckpointHeader;

static int write_test_checkpoint(
    const char* path,
    const TestCheckpointHeader* header,
    const float* parameters
) {
    FILE* file = fopen(path, "wb");
    int ok;
    if (!file || !header || (!parameters && header->parameter_count > 0)) {
        if (file) fclose(file);
        return 0;
    }
    ok = fwrite(header, sizeof(*header), 1, file) == 1 &&
        fwrite(parameters, sizeof(float), header->parameter_count, file) == header->parameter_count;
    fclose(file);
    return ok;
}

static int flip_test_checkpoint_byte(const char* path, long offset) {
    FILE* file = fopen(path, "r+b");
    int value;
    int ok = 0;
    if (!file || offset < 0 || fseek(file, offset, SEEK_SET) != 0) {
        if (file) fclose(file);
        return 0;
    }
    value = fgetc(file);
    if (value != EOF && fseek(file, offset, SEEK_SET) == 0 &&
            fputc(value ^ 0x01, file) != EOF) {
        ok = 1;
    }
    if (fclose(file) != 0) {
        ok = 0;
    }
    return ok;
}

static int test_episode_target_roundtrip(void) {
    const char* replay_path = "target_replay_test.jsonl";
    Episode episode;
    Episode parsed;
    float observation[4] = {0};
    unsigned char legal_mask[OBS_NUM_ACTIONS] = {0};
    FILE* file = NULL;
    char json[8192];
    char battle_id[64] = {0};
    char policy_tag[64] = {0};
    int ok = 1;
    memset(&episode, 0, sizeof(episode));
    memset(&parsed, 0, sizeof(parsed));
    legal_mask[OBS_A1_MOVE1] = 1;
    ok &= assert_true(episode_init(&episode, 1u, 4u), "initialize target replay episode");
    ok &= assert_true(episode_append(&episode, observation, legal_mask, OBS_A1_MOVE1, 1.0f, 1),
        "append target replay step");
    episode.factorized_actions[0].slot0_has_action = 1;
    episode.factorized_actions[0].slot0_kind = FACTORIZED_ACTION_MOVE;
    episode.factorized_actions[0].slot0_move_index = 0;
    episode.factorized_actions[0].slot0_target_index = FACTORIZED_TARGET_FOE_RIGHT;
    episode.factorized_actions[0].slot0_target_mask =
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    strncpy(episode.reward_mode, "dense_additive", sizeof(episode.reward_mode) - 1);
    episode.dense_hp_swing_weight = 0.10f;
    episode.dense_faint_swing_weight = 0.25f;
    episode.dense_reward_clip = 0.40f;
    episode.reward_config_present = 1;
    strncpy(episode.reward_mode, "dense_additive", sizeof(episode.reward_mode) - 1);
    episode.dense_hp_swing_weight = 0.10f;
    episode.dense_faint_swing_weight = 0.25f;
    episode.dense_reward_clip = 0.40f;
    episode.reward_config_present = 1;
    remove(replay_path);
    file = fopen(replay_path, "w+b");
    ok &= assert_true(file != NULL, "open target replay temporary file");
    if (file) {
        ok &= assert_true(episode_write_json_record(file, &episode, "battle-target", "policy-target"),
            "write target replay JSON");
        rewind(file);
        ok &= assert_true(fgets(json, sizeof(json), file) != NULL, "read target replay JSON");
        fclose(file);
        file = NULL;
        ok &= assert_true(episode_parse_json_record(json, &parsed, battle_id, sizeof(battle_id), policy_tag, sizeof(policy_tag)),
            "parse target replay JSON");
        ok &= assert_true(parsed.factorized_actions[0].slot0_target_index == FACTORIZED_TARGET_FOE_RIGHT,
            "target replay preserves selected target");
        ok &= assert_true(parsed.factorized_actions[0].slot0_target_mask == episode.factorized_actions[0].slot0_target_mask,
            "target replay preserves legal target mask");
        ok &= assert_true(strcmp(parsed.reward_mode, "dense_additive") == 0,
            "episode replay preserves reward mode");
        ok &= assert_true(parsed.reward_config_present &&
                fabs((double)(parsed.dense_hp_swing_weight - 0.10f)) < 0.0001 &&
                fabs((double)(parsed.dense_faint_swing_weight - 0.25f)) < 0.0001 &&
                fabs((double)(parsed.dense_reward_clip - 0.40f)) < 0.0001,
            "episode replay preserves dense reward configuration");
        ok &= assert_true(strcmp(parsed.reward_mode, "dense_additive") == 0,
            "episode replay preserves reward mode");
        ok &= assert_true(parsed.reward_config_present &&
                fabs((double)(parsed.dense_hp_swing_weight - 0.10f)) < 0.0001 &&
                fabs((double)(parsed.dense_faint_swing_weight - 0.25f)) < 0.0001 &&
                fabs((double)(parsed.dense_reward_clip - 0.40f)) < 0.0001,
            "episode replay preserves dense reward configuration");
    }
    if (file) fclose(file);
    remove(replay_path);
    episode_free(&episode);
    episode_free(&parsed);
    return ok;
}

static int test_factorized_target_head_training(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    float sequence[4] = {0};
    float hidden[8] = {0};
    unsigned char legal_mask[OBS_NUM_ACTIONS] = {0};
    FactorizedActionChoice choice;
    float before[FACTORIZED_TARGET_DIM] = {0};
    float after[FACTORIZED_TARGET_DIM] = {0};
    float value = 0.0f;
    int ok = 1;
    if (!assert_true(model != NULL, "create target-head training model")) return 0;
    legal_mask[OBS_A1_MOVE1] = 1;
    factorized_action_choice_init(&choice);
    choice.slot0_has_action = 1;
    choice.slot0_kind = FACTORIZED_ACTION_MOVE;
    choice.slot0_move_index = 0;
    choice.slot0_target_mask = FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    choice.slot0_target_index = FACTORIZED_TARGET_FOE_RIGHT;
    ok &= assert_true(gru_model_evaluate_factorized_hidden(
        model, hidden, legal_mask,
        NULL, NULL, NULL, NULL, before,
        NULL, NULL, NULL, NULL, NULL, &value), "evaluate target head before update");
    gru_model_clear_accumulated_supervised_updates(model);
    ok &= assert_true(gru_model_supervised_accumulate_sequence_window_factorized(
        model, sequence, 1u, NULL, legal_mask, NULL, &choice, 1.0f,
        NULL, NULL, NULL), "accumulate target-head supervised update");
    ok &= assert_true(gru_model_apply_accumulated_supervised_updates(model, 0.1f), "apply target-head supervised update");
    ok &= assert_true(gru_model_evaluate_factorized_hidden(
        model, hidden, legal_mask,
        NULL, NULL, NULL, NULL, after,
        NULL, NULL, NULL, NULL, NULL, &value), "evaluate target head after update");
    ok &= assert_true(after[FACTORIZED_TARGET_FOE_RIGHT] > before[FACTORIZED_TARGET_FOE_RIGHT],
        "target-head update increases labeled target probability");
    gru_model_destroy(model);
    return ok;
}

static int test_factorized_ppo_anchor_regularization(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    GruModel* anchor = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    Episode episode;
    GruTrainer trainer;
    float sequence[4] = {0};
    float hidden[8] = {0};
    unsigned char legal0[OBS_NUM_ACTIONS] = {0};
    unsigned char legal1[OBS_NUM_ACTIONS] = {0};
    unsigned char combined_legal[OBS_NUM_ACTIONS] = {0};
    FactorizedActionChoice drift_choice;
    float joint_before[FACTORIZED_JOINT_DIM] = {0};
    float joint_after[FACTORIZED_JOINT_DIM] = {0};
    float* zero_parameters = NULL;
    size_t parameter_count;
    float before_distance;
    float after_distance = 0.0f;
    int legal_joint_indices[4] = {0, 1, FACTORIZED_LOCAL_ACTION_DIM, FACTORIZED_LOCAL_ACTION_DIM + 1};
    const Episode* minibatch[2];
    int i;
    int ok = 1;
    memset(&episode, 0, sizeof(episode));
    ok &= assert_true(model != NULL && anchor != NULL, "PPO anchor test creates matching models");
    if (!model || !anchor) {
        gru_model_destroy(model);
        gru_model_destroy(anchor);
        return 0;
    }
    parameter_count = gru_model_parameter_count(model);
    zero_parameters = (float*)calloc(parameter_count, sizeof(float));
    ok &= assert_true(zero_parameters != NULL &&
        gru_model_import_parameters(model, zero_parameters, parameter_count) &&
        gru_model_import_parameters(anchor, zero_parameters, parameter_count),
        "PPO anchor test starts from identical models");
    if (!zero_parameters) {
        gru_model_destroy(model);
        gru_model_destroy(anchor);
        return 0;
    }
    legal0[OBS_A1_MOVE1] = 1;
    legal0[OBS_A1_MOVE2] = 1;
    legal1[OBS_A2_MOVE1] = 1;
    legal1[OBS_A2_MOVE2] = 1;
    memcpy(combined_legal, legal0, sizeof(combined_legal));
    combined_legal[OBS_A2_MOVE1] = 1;
    combined_legal[OBS_A2_MOVE2] = 1;
    factorized_action_choice_init(&drift_choice);
    drift_choice.slot0_has_action = 1;
    drift_choice.slot0_kind = FACTORIZED_ACTION_MOVE;
    drift_choice.slot0_move_index = 1;
    drift_choice.slot1_has_action = 1;
    drift_choice.slot1_kind = FACTORIZED_ACTION_MOVE;
    drift_choice.slot1_move_index = 1;
    gru_model_clear_accumulated_supervised_updates(model);
    ok &= assert_true(gru_model_supervised_accumulate_sequence_window_factorized(
        model, sequence, 1u, NULL, legal0, legal1, &drift_choice, 0.0f,
        NULL, NULL, NULL), "PPO anchor test creates policy drift");
    ok &= assert_true(gru_model_apply_accumulated_supervised_updates(model, 1.0f),
        "PPO anchor test applies policy drift");
    ok &= assert_true(gru_model_evaluate_joint_hidden(
        model, hidden, combined_legal, joint_before, NULL),
        "PPO anchor test evaluates drifted policy");
    before_distance = 0.0f;
    for (i = 0; i < 4; ++i) {
        float delta = joint_before[legal_joint_indices[i]] - 0.25f;
        before_distance += delta * delta;
    }
    ok &= assert_true(before_distance > 0.01f, "PPO anchor test has measurable drift from anchor");

    ok &= assert_true(episode_init(&episode, 1u, 4u), "PPO anchor test initializes episode");
    ok &= assert_true(episode_append(&episode, sequence, combined_legal, OBS_A1_MOVE1, 0.0f, 1),
        "PPO anchor test appends zero-advantage step");
    episode.actions2[0] = OBS_A2_MOVE1;
    episode.factorized_actions[0].slot0_has_action = 1;
    episode.factorized_actions[0].slot0_kind = FACTORIZED_ACTION_MOVE;
    episode.factorized_actions[0].slot0_move_index = 0;
    episode.factorized_actions[0].slot1_has_action = 1;
    episode.factorized_actions[0].slot1_kind = FACTORIZED_ACTION_MOVE;
    episode.factorized_actions[0].slot1_move_index = 0;
    episode.old_log_probs[0] = logf(joint_before[0] > 1.0e-8f ? joint_before[0] : 1.0e-8f);
    episode.old_values[0] = 0.0f;

    gru_trainer_init(&trainer, 0.01f, 1u, 1.0f, 7u);
    trainer.advantage_norm = 0;
    trainer.entropy_coef = 0.0f;
    trainer.anchor_model = anchor;
    trainer.anchor_kl_coef = 0.1f;
    minibatch[0] = &episode;
    minibatch[1] = &episode;
    ok &= assert_true(gru_trainer_ppo_minibatch(&trainer, model, minibatch, 2u),
        "factorized PPO applies anchored minibatch update");
    ok &= assert_true(trainer.last_rl_labels == 2u,
        "factorized PPO minibatch reports all labels");
    ok &= assert_true(trainer.last_anchor_kl_mean > 1.0e-5f &&
        trainer.last_anchor_kl_max >= trainer.last_anchor_kl_mean &&
        trainer.last_anchor_loss > 0.0f,
        "factorized PPO reports nonzero anchor KL and loss");
    ok &= assert_true(gru_model_evaluate_joint_hidden(
        model, hidden, combined_legal, joint_after, NULL),
        "PPO anchor test evaluates regularized policy");
    for (i = 0; i < 4; ++i) {
        float delta = joint_after[legal_joint_indices[i]] - 0.25f;
        after_distance += delta * delta;
    }
    ok &= assert_true(after_distance < before_distance,
        "factorized PPO anchor moves policy back toward reference");

    episode_free(&episode);
    free(zero_parameters);
    gru_model_destroy(model);
    gru_model_destroy(anchor);
    return ok;
}

static int test_ppo_hard_kl_requires_consecutive_breaches(void) {
    int breaches = 0;
    int ok = 1;
    ok &= assert_true(!gru_trainer_ppo_hard_kl_stop_update(0.10f, 0.02f, 4.0f, 2, &breaches) && breaches == 1,
        "first extreme PPO KL minibatch is recorded without stopping");
    ok &= assert_true(!gru_trainer_ppo_hard_kl_stop_update(0.01f, 0.02f, 4.0f, 2, &breaches) && breaches == 0,
        "ordinary PPO KL minibatch resets consecutive breach count");
    ok &= assert_true(!gru_trainer_ppo_hard_kl_stop_update(0.09f, 0.02f, 4.0f, 2, &breaches) && breaches == 1,
        "new extreme PPO KL sequence starts at one breach");
    ok &= assert_true(gru_trainer_ppo_hard_kl_stop_update(0.11f, 0.02f, 4.0f, 2, &breaches) && breaches == 2,
        "second consecutive extreme PPO KL minibatch triggers emergency stop");
    return ok;
}

static int test_symmetric_joint_action_training(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    float sequence[4] = {0};
    float hidden[8] = {0};
    unsigned char legal0[OBS_NUM_ACTIONS] = {0};
    unsigned char legal1[OBS_NUM_ACTIONS] = {0};
    unsigned char combined[OBS_NUM_ACTIONS] = {0};
    float before[FACTORIZED_JOINT_DIM] = {0};
    float after[FACTORIZED_JOINT_DIM] = {0};
    float* parameters = NULL;
    size_t parameter_count;
    FactorizedActionChoice choice;
    float value = 0.0f;
    float probability_sum = 0.0f;
    int selected = 1;
    int i;
    int ok = 1;
    ok &= assert_true(model != NULL, "joint scorer test creates model");
    if (!model) return 0;
    parameter_count = gru_model_parameter_count(model);
    parameters = (float*)calloc(parameter_count, sizeof(float));
    ok &= assert_true(parameters != NULL && gru_model_import_parameters(model, parameters, parameter_count),
        "joint scorer test starts from neutral parameters");
    if (!parameters) {
        gru_model_destroy(model);
        return 0;
    }
    legal0[OBS_A1_MOVE1] = 1;
    legal0[OBS_A1_MOVE2] = 1;
    legal1[OBS_A2_MOVE1] = 1;
    legal1[OBS_A2_MOVE2] = 1;
    memcpy(combined, legal0, sizeof(combined));
    combined[OBS_A2_MOVE1] = 1;
    combined[OBS_A2_MOVE2] = 1;
    factorized_action_choice_from_flat_actions(&choice, OBS_A1_MOVE1, OBS_A2_MOVE2);
    ok &= assert_true(gru_model_evaluate_joint_hidden(model, hidden, combined, before, &value),
        "joint scorer evaluates legal action pairs");
    gru_model_clear_accumulated_supervised_updates(model);
    ok &= assert_true(gru_model_supervised_accumulate_sequence_window_factorized(
        model, sequence, 1u, NULL, legal0, legal1, &choice, 0.0f,
        NULL, NULL, NULL), "joint scorer accumulates supervised pair update");
    ok &= assert_true(gru_model_apply_accumulated_supervised_updates(model, 0.1f),
        "joint scorer applies supervised pair update");
    ok &= assert_true(gru_model_evaluate_joint_hidden(model, hidden, combined, after, &value),
        "joint scorer evaluates after training");
    ok &= assert_true(after[selected] > before[selected],
        "joint scorer raises the labeled action-pair probability");
    for (i = 0; i < FACTORIZED_JOINT_DIM; ++i) probability_sum += after[i];
    ok &= assert_true(fabsf(probability_sum - 1.0f) < 1.0e-5f,
        "joint scorer normalizes over legal pairs");

    memset(combined, 0, sizeof(combined));
    combined[OBS_A1_MOVE1] = 1;
    combined[OBS_A1_MOVE1_TERA] = 1;
    combined[OBS_A1_SWITCH1] = 1;
    combined[OBS_A2_MOVE1] = 1;
    combined[OBS_A2_MOVE1_TERA] = 1;
    combined[OBS_A2_SWITCH1] = 1;
    ok &= assert_true(gru_model_evaluate_joint_hidden(model, hidden, combined, after, &value),
        "joint scorer evaluates constrained pairs");
    ok &= assert_true(after[4 * FACTORIZED_LOCAL_ACTION_DIM + 4] == 0.0f,
        "joint scorer masks two simultaneous Terastallizations");
    ok &= assert_true(after[8 * FACTORIZED_LOCAL_ACTION_DIM + 8] == 0.0f,
        "joint scorer masks duplicate switch destinations");
    free(parameters);
    gru_model_destroy(model);
    return ok;
}

static int test_shared_entity_encoder_training_and_migration(void) {
    const char* pre_entity_path = "checkpoint_test_pre_entity.bin";
    GruModel* model = gru_model_create(observation_flat_size(), 4u, OBS_NUM_ACTIONS);
    GruModel* migrated = NULL;
    GruModel* loaded = NULL;
    Observation observation;
    FactorizedActionChoice choice;
    float* sequence = NULL;
    float* before = NULL;
    float* after = NULL;
    float* pre_entity = NULL;
    float* migrated_parameters = NULL;
    float hidden_state[4] = {0};
    float forward_value = 0.0f;
    unsigned char legal_mask[OBS_NUM_ACTIONS] = {0};
    size_t current_count;
    size_t pre_entity_count;
    size_t value_count = 5u;
    size_t entity_chunk =
        (size_t)GRU_ENTITY_EMBED_DIM * OBS_POKEMON_FEATURES +
        GRU_ENTITY_EMBED_DIM +
        (size_t)OBS_POKEMON_FEATURES * GRU_ENTITY_EMBED_DIM;
    size_t entity_offset;
    size_t i;
    TestCheckpointHeader header;
    TrainerCheckpointState state;
    CheckpointLoadResult result;
    int encoder_changed = 0;
    int adam_changed = 0;
    int ok = 1;
    ok &= assert_true(model != NULL, "shared entity encoder test creates observation-sized model");
    if (!model) return 0;
    current_count = gru_model_parameter_count(model);
    pre_entity_count = gru_model_pre_entity_parameter_count(model);
    ok &= assert_true(current_count - pre_entity_count == entity_chunk,
        "shared entity encoder is serialized once for all Pokemon slots");
    sequence = (float*)calloc(observation_flat_size(), sizeof(float));
    before = (float*)malloc(current_count * sizeof(float));
    after = (float*)malloc(current_count * sizeof(float));
    pre_entity = (float*)malloc(pre_entity_count * sizeof(float));
    migrated_parameters = (float*)malloc(current_count * sizeof(float));
    ok &= assert_true(sequence && before && after && pre_entity && migrated_parameters,
        "shared entity encoder test allocates buffers");
    if (!sequence || !before || !after || !pre_entity || !migrated_parameters) goto cleanup;
    observation_init(&observation);
    observation.self_team[0].known = 1;
    observation.self_team[0].revealed = 1;
    observation.self_team[0].active = 1;
    observation.self_team[0].hp_frac = 1.0f;
    observation.self_team[0].species_id = 25;
    observation.self_team[0].species_known_mode = 2;
    observation.legal_mask[OBS_A1_MOVE1] = 1;
    ok &= assert_true(observation_flatten(sequence, observation_flat_size(), &observation) == observation_flat_size(),
        "shared entity encoder test flattens observation");
    gru_model_forward_step(model, sequence, hidden_state, hidden_state, NULL, &forward_value);
    ok &= assert_true(hidden_state[0] != 0.0f || hidden_state[1] != 0.0f ||
            hidden_state[2] != 0.0f || hidden_state[3] != 0.0f,
        "value-only runtime forward call advances recurrent state");
    legal_mask[OBS_A1_MOVE1] = 1;
    factorized_action_choice_from_flat_actions(&choice, OBS_A1_MOVE1, -1);
    ok &= assert_true(gru_model_export_parameters(model, before, current_count),
        "shared entity encoder test exports initial parameters");
    gru_model_clear_accumulated_supervised_updates(model);
    ok &= assert_true(gru_model_supervised_accumulate_sequence_window_factorized(
        model, sequence, 1u, NULL, legal_mask, NULL, &choice, 0.0f,
        NULL, NULL, NULL), "shared entity encoder receives recurrent gradients");
    ok &= assert_true(gru_model_apply_accumulated_supervised_updates(model, 0.05f),
        "shared entity encoder applies recurrent update");
    ok &= assert_true(gru_model_export_parameters(model, after, current_count),
        "shared entity encoder test exports updated parameters");
    entity_offset = current_count - value_count - entity_chunk;
    for (i = 0; i < entity_chunk; ++i) {
        if (before[entity_offset + i] != after[entity_offset + i]) {
            encoder_changed = 1;
            break;
        }
    }
    ok &= assert_true(encoder_changed, "shared entity encoder learns from a Pokemon in any team slot");
    memcpy(before, after, current_count * sizeof(float));
    gru_model_clear_accumulated_supervised_updates(model);
    ok &= assert_true(gru_model_supervised_accumulate_sequence_window_factorized(
        model, sequence, 1u, NULL, legal_mask, NULL, &choice, 0.0f,
        NULL, NULL, NULL), "shared entity encoder accumulates an Adam update");
    ok &= assert_true(gru_model_apply_accumulated_adam_updates(
        model, 0.001f, 0.9f, 0.999f, 1.0e-8f, 1.0f),
        "shared entity encoder participates in Adam optimization");
    ok &= assert_true(gru_model_export_parameters(model, after, current_count),
        "shared entity encoder exports after Adam optimization");
    for (i = 0; i < entity_chunk; ++i) {
        if (before[entity_offset + i] != after[entity_offset + i]) {
            adam_changed = 1;
            break;
        }
    }
    ok &= assert_true(adam_changed, "Adam updates the shared entity encoder parameters");

    memcpy(pre_entity, after, entity_offset * sizeof(float));
    memcpy(pre_entity + entity_offset, after + entity_offset + entity_chunk, value_count * sizeof(float));
    memset(&header, 0, sizeof(header));
    memset(&state, 0, sizeof(state));
    memcpy(header.magic, "PORYCHK", 7);
    header.version = CHECKPOINT_MIN_SUPPORTED_VERSION;
    header.input_dim = observation_flat_size();
    header.hidden_dim = 4u;
    header.num_actions = OBS_NUM_ACTIONS;
    header.parameter_count = pre_entity_count;
    header.trainer = state;
    remove(pre_entity_path);
    ok &= assert_true(write_test_checkpoint(pre_entity_path, &header, pre_entity),
        "shared entity encoder test writes pre-entity checkpoint fixture");
    loaded = checkpoint_load_compatible(pre_entity_path, NULL,
        observation_flat_size(), OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && result.status == CHECKPOINT_LOAD_OK &&
            result.parameter_layout == CHECKPOINT_LAYOUT_FACTORIZED,
        "checkpoint loader accepts pre-entity factorized layout");
    migrated = gru_model_create(observation_flat_size(), 4u, OBS_NUM_ACTIONS);
    ok &= assert_true(migrated != NULL &&
            gru_model_import_parameters(migrated, pre_entity, pre_entity_count) &&
            gru_model_export_parameters(migrated, migrated_parameters, current_count),
        "pre-entity checkpoint parameters migrate successfully");
    if (migrated) {
        int neutral = 1;
        for (i = 0; i < entity_chunk; ++i) {
            if (migrated_parameters[entity_offset + i] != 0.0f) {
                neutral = 0;
                break;
            }
        }
        ok &= assert_true(neutral, "pre-entity migration initializes the shared residual path neutrally");
    }

cleanup:
    remove(pre_entity_path);
    gru_model_destroy(loaded);
    gru_model_destroy(migrated);
    gru_model_destroy(model);
    free(sequence);
    free(before);
    free(after);
    free(pre_entity);
    free(migrated_parameters);
    return ok;
}

static int test_active_slot_schema_migrates_legacy_checkpoint(void) {
    const char* path = "checkpoint_test_pre_active_slot_schema.bin";
    size_t old_input_dim = observation_flat_size() -
        2u * OBS_TEAM_SIZE * OBS_ACTIVE_SLOT_CLASSES;
    GruModel* old_model = gru_model_create(old_input_dim, 4u, OBS_NUM_ACTIONS);
    GruModel* loaded = NULL;
    TrainerCheckpointState state;
    CheckpointLoadResult result;
    TestCheckpointHeader header;
    float* full_parameters = NULL;
    float* legacy_parameters = NULL;
    float* old_input = NULL;
    float* current_input = NULL;
    float old_policy[OBS_NUM_ACTIONS] = {0};
    float current_policy[OBS_NUM_ACTIONS] = {0};
    float old_hidden[4] = {0};
    float current_hidden[4] = {0};
    float old_value = 0.0f;
    float current_value = 0.0f;
    size_t full_count = 0;
    size_t legacy_count = 0;
    size_t value_count = 5u;
    size_t prefix_count = 0;
    size_t old_index;
    size_t current_index;
    size_t pokemon;
    size_t i;
    int ok = 1;
    memset(&state, 0, sizeof(state));
    memset(&header, 0, sizeof(header));
    remove(path);
    ok &= assert_true(old_model != NULL, "active-slot schema test creates prior-dimension model");
    if (old_model) {
        full_count = gru_model_parameter_count(old_model);
        legacy_count = gru_model_legacy_parameter_count(old_model);
        prefix_count = legacy_count - value_count;
        full_parameters = (float*)malloc(full_count * sizeof(float));
        legacy_parameters = (float*)malloc(legacy_count * sizeof(float));
        old_input = (float*)malloc(old_input_dim * sizeof(float));
        current_input = (float*)calloc(observation_flat_size(), sizeof(float));
        ok &= assert_true(full_parameters && legacy_parameters && old_input && current_input,
            "active-slot schema test allocates migration buffers");
        if (!full_parameters || !legacy_parameters || !old_input || !current_input) goto cleanup;
        ok &= assert_true(gru_model_export_parameters(old_model, full_parameters, full_count),
            "active-slot schema test exports prior parameters");
        memcpy(legacy_parameters, full_parameters, prefix_count * sizeof(float));
        memcpy(legacy_parameters + prefix_count, full_parameters + full_count - value_count,
            value_count * sizeof(float));
        ok &= assert_true(gru_model_import_parameters(old_model, legacy_parameters, legacy_count),
            "active-slot schema test restores the exact legacy model layout");
        memcpy(header.magic, "PORYCHK", 7);
        header.version = CHECKPOINT_MIN_SUPPORTED_VERSION;
        header.input_dim = old_input_dim;
        header.hidden_dim = 4u;
        header.num_actions = OBS_NUM_ACTIONS;
        header.parameter_count = legacy_count;
        header.trainer = state;
        ok &= assert_true(write_test_checkpoint(path, &header, legacy_parameters),
            "active-slot schema test writes legacy prior-dimension checkpoint");
        loaded = checkpoint_load_compatible(path, NULL,
            observation_flat_size(), OBS_NUM_ACTIONS, &result);
        ok &= assert_true(loaded != NULL && result.status == CHECKPOINT_LOAD_OK &&
                result.stored_input_dim == old_input_dim &&
                result.parameter_layout == CHECKPOINT_LAYOUT_LEGACY_FLAT &&
                result.migrated_legacy_heads && result.migrated_active_slot_inputs &&
                gru_model_input_dim(loaded) == observation_flat_size(),
            "checkpoint loader safely migrates the known active-slot schema transition");
        for (i = 0; i < old_input_dim; ++i) {
            old_input[i] = (float)((int)(i % 19u) - 9) / 10.0f;
        }
        old_index = 0;
        current_index = 0;
        i = OBS_GLOBAL_FEATURES + 2u * OBS_SIDE_FEATURES;
        memcpy(current_input, old_input, i * sizeof(float));
        old_index += i;
        current_index += i;
        for (pokemon = 0; pokemon < 2u * OBS_TEAM_SIZE; ++pokemon) {
            size_t remaining = OBS_POKEMON_FEATURES - OBS_ACTIVE_SLOT_CLASSES -
                OBS_POKEMON_ACTIVE_SLOT_OFFSET;
            memcpy(current_input + current_index, old_input + old_index,
                OBS_POKEMON_ACTIVE_SLOT_OFFSET * sizeof(float));
            old_index += OBS_POKEMON_ACTIVE_SLOT_OFFSET;
            current_index += OBS_POKEMON_ACTIVE_SLOT_OFFSET + OBS_ACTIVE_SLOT_CLASSES;
            memcpy(current_input + current_index, old_input + old_index,
                remaining * sizeof(float));
            old_index += remaining;
            current_index += remaining;
        }
        memcpy(current_input + current_index, old_input + old_index,
            (old_input_dim - old_index) * sizeof(float));
        if (loaded) {
            gru_model_forward_step(old_model, old_input, old_hidden, old_hidden, old_policy, &old_value);
            gru_model_forward_step(loaded, current_input, current_hidden, current_hidden,
                current_policy, &current_value);
            for (i = 0; i < 4u; ++i) {
                ok &= assert_true(nearly_equal(old_hidden[i], current_hidden[i], 1.0e-6),
                    "active-slot migration preserves recurrent output with neutral new inputs");
            }
            for (i = 0; i < OBS_NUM_ACTIONS; ++i) {
                ok &= assert_true(nearly_equal(old_policy[i], current_policy[i], 1.0e-6),
                    "active-slot migration preserves flat policy output");
            }
            ok &= assert_true(nearly_equal(old_value, current_value, 1.0e-6),
                "active-slot migration preserves value output");
        }
    }
cleanup:
    gru_model_destroy(loaded);
    gru_model_destroy(old_model);
    free(full_parameters);
    free(legacy_parameters);
    free(old_input);
    free(current_input);
    remove(path);
    return ok;
}

static int test_checkpoint_compatibility_validation(void) {
    const char* current_path = "checkpoint_test_current.bin";
    const char* current_temporary_path = "checkpoint_test_current.bin.tmp";
    const char* replacement_failure_path = "checkpoint_test_replace_failure";
    const char* replacement_failure_temporary_path = "checkpoint_test_replace_failure.tmp";
    const char* legacy_path = "checkpoint_test_legacy.bin";
    const char* pre_joint_path = "checkpoint_test_pre_joint.bin";
    const char* pre_target_path = "checkpoint_test_pre_target.bin";
    const char* version_path = "checkpoint_test_version.bin";
    const char* header_corrupt_path = "checkpoint_test_header_corrupt.bin";
    const char* parameter_corrupt_path = "checkpoint_test_parameter_corrupt.bin";
    const char* truncated_path = "checkpoint_test_truncated.bin";
    const char* truncated_parameters_path = "checkpoint_test_truncated_parameters.bin";
    const char* truncated_checksum_path = "checkpoint_test_truncated_checksum.bin";
    const char* trailing_path = "checkpoint_test_trailing.bin";
    GruModel* model = NULL;
    GruModel* loaded = NULL;
    TrainerCheckpointState state;
    TrainerCheckpointState replacement_state;
    TrainerCheckpointState loaded_state;
    CheckpointLoadResult result;
    TestCheckpointHeader header;
    float* current_parameters = NULL;
    float* legacy_parameters = NULL;
    float* pre_joint_parameters = NULL;
    float* pre_target_parameters = NULL;
    size_t current_count;
    size_t pre_joint_count;
    size_t pre_target_count;
    size_t legacy_count;
    size_t value_count;
    size_t prefix_count;
    int ok = 1;
    FILE* file = NULL;

    remove(current_path);
    remove(current_temporary_path);
    remove(replacement_failure_temporary_path);
    TEST_RMDIR(replacement_failure_path);
    remove(legacy_path);
    remove(pre_joint_path);
    remove(pre_target_path);
    remove(version_path);
    remove(header_corrupt_path);
    remove(parameter_corrupt_path);
    remove(truncated_path);
    remove(truncated_parameters_path);
    remove(truncated_checksum_path);
    remove(trailing_path);
    memset(&state, 0, sizeof(state));
    state.step = 42u;
    state.learning_rate = 0.001f;
    state.bptt_window = 16u;
    state.gradient_clip = 1.0f;
    state.seed = 7u;
    model = gru_model_create(8u, 4u, OBS_NUM_ACTIONS);
    ok &= assert_true(model != NULL, "checkpoint test creates model");
    if (!model) goto cleanup;
    ok &= assert_true(checkpoint_save(current_path, model, &state), "checkpoint test saves current layout");

    memset(&loaded_state, 0, sizeof(loaded_state));
    loaded = checkpoint_load_compatible(current_path, &loaded_state, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && result.status == CHECKPOINT_LOAD_OK, "current checkpoint loads successfully");
    ok &= assert_true(result.parameter_layout == CHECKPOINT_LAYOUT_FACTORIZED && !result.migrated_legacy_heads,
        "current checkpoint reports factorized layout");
    ok &= assert_true(result.stored_version == CHECKPOINT_FORMAT_VERSION && result.checksum_verified &&
            result.stored_checksum == result.computed_checksum,
        "current checkpoint reports verified v2 checksum");
    ok &= assert_true(loaded_state.step == state.step, "current checkpoint restores trainer state");
    gru_model_destroy(loaded);
    loaded = NULL;

    file = fopen(current_temporary_path, "wb");
    ok &= assert_true(file != NULL, "checkpoint test creates interrupted temporary write");
    if (file) {
        fwrite("partial", 1, 7u, file);
        fclose(file);
        file = NULL;
    }
    loaded = checkpoint_load_compatible(current_path, &loaded_state, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && loaded_state.step == state.step,
        "interrupted temporary write preserves existing checkpoint");
    gru_model_destroy(loaded);
    loaded = NULL;

    replacement_state = state;
    replacement_state.step = 84u;
    ok &= assert_true(checkpoint_save(current_path, model, &replacement_state),
        "checkpoint save recovers from interrupted temporary write");
    file = fopen(current_temporary_path, "rb");
    ok &= assert_true(file == NULL, "successful checkpoint replacement leaves no temporary file");
    if (file) {
        fclose(file);
        file = NULL;
    }
    loaded = checkpoint_load_compatible(current_path, &loaded_state, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && loaded_state.step == replacement_state.step,
        "atomic checkpoint replacement publishes new trainer state");
    gru_model_destroy(loaded);
    loaded = NULL;

    ok &= assert_true(TEST_MKDIR(replacement_failure_path) == 0,
        "checkpoint test creates replacement failure target");
    ok &= assert_true(!checkpoint_save(replacement_failure_path, model, &state),
        "checkpoint save reports atomic replacement failure");
    file = fopen(replacement_failure_temporary_path, "rb");
    ok &= assert_true(file == NULL, "failed checkpoint replacement removes temporary file");
    if (file) {
        fclose(file);
        file = NULL;
    }
    TEST_RMDIR(replacement_failure_path);

    loaded = checkpoint_load_compatible(current_path, NULL, 9u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_INPUT_DIM_MISMATCH,
        "checkpoint rejects observation dimension mismatch");
    loaded = checkpoint_load_compatible(current_path, NULL, 8u, OBS_NUM_ACTIONS + 1u, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_ACTION_COUNT_MISMATCH,
        "checkpoint rejects action count mismatch");

    current_count = gru_model_parameter_count(model);
    pre_joint_count = gru_model_pre_joint_parameter_count(model);
    pre_target_count = gru_model_pre_target_parameter_count(model);
    legacy_count = gru_model_legacy_parameter_count(model);
    value_count = gru_model_hidden_dim(model) + 1u;
    prefix_count = legacy_count - value_count;
    current_parameters = (float*)malloc(current_count * sizeof(float));
    legacy_parameters = (float*)malloc(legacy_count * sizeof(float));
    pre_joint_parameters = (float*)malloc(pre_joint_count * sizeof(float));
    pre_target_parameters = (float*)malloc(pre_target_count * sizeof(float));
    ok &= assert_true(current_parameters != NULL && legacy_parameters != NULL && pre_joint_parameters != NULL && pre_target_parameters != NULL,
        "checkpoint test allocates legacy fixture parameters");
    if (!current_parameters || !legacy_parameters || !pre_joint_parameters || !pre_target_parameters) goto cleanup;
    ok &= assert_true(gru_model_export_parameters(model, current_parameters, current_count),
        "checkpoint test exports current parameters");
    {
        size_t joint_chunk = (gru_model_hidden_dim(model) * FACTORIZED_PAIR_DIM) + FACTORIZED_PAIR_DIM;
        size_t joint_offset = current_count - value_count - joint_chunk;
        memcpy(pre_joint_parameters, current_parameters, joint_offset * sizeof(float));
        memcpy(pre_joint_parameters + joint_offset,
            current_parameters + joint_offset + joint_chunk, value_count * sizeof(float));
        ok &= assert_true(joint_offset + value_count == pre_joint_count,
            "checkpoint test builds pre-joint fixture layout");
    }
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "PORYCHK", 7);
    header.version = CHECKPOINT_MIN_SUPPORTED_VERSION;
    header.input_dim = gru_model_input_dim(model);
    header.hidden_dim = gru_model_hidden_dim(model);
    header.num_actions = gru_model_num_actions(model);
    header.parameter_count = pre_joint_count;
    header.trainer = state;
    ok &= assert_true(write_test_checkpoint(pre_joint_path, &header, pre_joint_parameters),
        "checkpoint test writes pre-joint factorized layout");
    loaded = checkpoint_load_compatible(pre_joint_path, &loaded_state, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && result.status == CHECKPOINT_LOAD_OK &&
            result.parameter_layout == CHECKPOINT_LAYOUT_FACTORIZED,
        "pre-joint checkpoint loads with a neutral compatibility scorer");
    gru_model_destroy(loaded);
    loaded = NULL;
    {
        size_t target_chunk = (gru_model_hidden_dim(model) * FACTORIZED_TARGET_DIM) + FACTORIZED_TARGET_DIM;
        size_t slot1_factorized_chunk =
            (gru_model_hidden_dim(model) * FACTORIZED_KIND_DIM) + FACTORIZED_KIND_DIM +
            (gru_model_hidden_dim(model) * FACTORIZED_MOVE_DIM) + FACTORIZED_MOVE_DIM +
            (gru_model_hidden_dim(model) * FACTORIZED_SWITCH_DIM) + FACTORIZED_SWITCH_DIM +
            (gru_model_hidden_dim(model) * FACTORIZED_TERA_DIM) + FACTORIZED_TERA_DIM;
        size_t target1_offset = pre_joint_count - value_count - target_chunk;
        size_t target0_offset = target1_offset - slot1_factorized_chunk - target_chunk;
        size_t out_index = 0;
        memcpy(pre_target_parameters, pre_joint_parameters, target0_offset * sizeof(float));
        out_index += target0_offset;
        memcpy(pre_target_parameters + out_index,
            pre_joint_parameters + target0_offset + target_chunk,
            slot1_factorized_chunk * sizeof(float));
        out_index += slot1_factorized_chunk;
        memcpy(pre_target_parameters + out_index,
            pre_joint_parameters + target1_offset + target_chunk,
            value_count * sizeof(float));
        out_index += value_count;
        ok &= assert_true(out_index == pre_target_count, "checkpoint test builds pre-target fixture layout");
    }
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "PORYCHK", 7);
    header.version = CHECKPOINT_MIN_SUPPORTED_VERSION;
    header.input_dim = gru_model_input_dim(model);
    header.hidden_dim = gru_model_hidden_dim(model);
    header.num_actions = gru_model_num_actions(model);
    header.parameter_count = pre_target_count;
    header.trainer = state;
    ok &= assert_true(write_test_checkpoint(pre_target_path, &header, pre_target_parameters),
        "checkpoint test writes pre-target factorized layout");
    loaded = checkpoint_load_compatible(pre_target_path, &loaded_state, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && result.status == CHECKPOINT_LOAD_OK &&
            result.parameter_layout == CHECKPOINT_LAYOUT_FACTORIZED,
        "pre-target factorized checkpoint loads with neutral target heads");
    gru_model_destroy(loaded);
    loaded = NULL;

    memcpy(legacy_parameters, current_parameters, prefix_count * sizeof(float));
    memcpy(legacy_parameters + prefix_count, current_parameters + current_count - value_count, value_count * sizeof(float));
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "PORYCHK", 7);
    header.version = CHECKPOINT_MIN_SUPPORTED_VERSION;
    header.input_dim = gru_model_input_dim(model);
    header.hidden_dim = gru_model_hidden_dim(model);
    header.num_actions = gru_model_num_actions(model);
    header.parameter_count = legacy_count;
    header.trainer = state;
    ok &= assert_true(write_test_checkpoint(legacy_path, &header, legacy_parameters),
        "checkpoint test writes legacy flat layout");
    loaded = checkpoint_load_compatible(legacy_path, &loaded_state, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded != NULL && result.status == CHECKPOINT_LOAD_OK,
        "legacy flat checkpoint loads successfully");
    ok &= assert_true(result.parameter_layout == CHECKPOINT_LAYOUT_LEGACY_FLAT && result.migrated_legacy_heads,
        "legacy checkpoint reports factorized-head migration");
    ok &= assert_true(result.stored_version == CHECKPOINT_MIN_SUPPORTED_VERSION && !result.checksum_verified,
        "v1 checkpoint loads as checksum-unverified");
    gru_model_destroy(loaded);
    loaded = NULL;

    header.version = CHECKPOINT_FORMAT_VERSION + 1u;
    header.parameter_count = current_count;
    ok &= assert_true(write_test_checkpoint(version_path, &header, current_parameters),
        "checkpoint test writes unsupported version fixture");
    loaded = checkpoint_load_compatible(version_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_UNSUPPORTED_VERSION,
        "checkpoint rejects unsupported format version");

    ok &= assert_true(checkpoint_save(header_corrupt_path, model, &state),
        "checkpoint test saves header corruption fixture");
    ok &= assert_true(flip_test_checkpoint_byte(
            header_corrupt_path,
            (long)(offsetof(TestCheckpointHeader, trainer) + offsetof(TrainerCheckpointState, seed))),
        "checkpoint test corrupts checksummed trainer state");
    loaded = checkpoint_load_compatible(header_corrupt_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_CHECKSUM_MISMATCH &&
            result.stored_checksum != result.computed_checksum,
        "checkpoint rejects checksummed header corruption");

    ok &= assert_true(checkpoint_save(parameter_corrupt_path, model, &state),
        "checkpoint test saves parameter corruption fixture");
    ok &= assert_true(flip_test_checkpoint_byte(
            parameter_corrupt_path,
            (long)sizeof(TestCheckpointHeader)),
        "checkpoint test corrupts checksummed parameter payload");
    loaded = checkpoint_load_compatible(parameter_corrupt_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_CHECKSUM_MISMATCH &&
            result.stored_checksum != result.computed_checksum,
        "checkpoint rejects checksummed parameter corruption");

    file = fopen(truncated_path, "wb");
    ok &= assert_true(file != NULL, "checkpoint test opens truncated fixture");
    if (file) {
        fwrite(&header, 1, sizeof(header) / 2u, file);
        fclose(file);
        file = NULL;
    }
    loaded = checkpoint_load_compatible(truncated_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_TRUNCATED_HEADER,
        "checkpoint rejects truncated header");

    header.version = CHECKPOINT_FORMAT_VERSION;
    header.parameter_count = current_count;
    file = fopen(truncated_parameters_path, "wb");
    ok &= assert_true(file != NULL, "checkpoint test opens truncated parameter fixture");
    if (file) {
        fwrite(&header, sizeof(header), 1, file);
        fclose(file);
        file = NULL;
    }
    loaded = checkpoint_load_compatible(truncated_parameters_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_TRUNCATED_PARAMETERS,
        "checkpoint rejects truncated parameter payload");

    ok &= assert_true(write_test_checkpoint(truncated_checksum_path, &header, current_parameters),
        "checkpoint test writes missing-checksum fixture");
    loaded = checkpoint_load_compatible(truncated_checksum_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_TRUNCATED_CHECKSUM,
        "checkpoint rejects truncated checksum footer");

    ok &= assert_true(checkpoint_save(trailing_path, model, &state),
        "checkpoint test saves trailing-data fixture");
    file = fopen(trailing_path, "ab");
    ok &= assert_true(file != NULL, "checkpoint test opens trailing-data fixture");
    if (file) {
        fputc(0, file);
        fclose(file);
        file = NULL;
    }
    loaded = checkpoint_load_compatible(trailing_path, NULL, 8u, OBS_NUM_ACTIONS, &result);
    ok &= assert_true(loaded == NULL && result.status == CHECKPOINT_LOAD_TRAILING_DATA,
        "checkpoint rejects unexpected trailing data");

cleanup:
    if (file) fclose(file);
    gru_model_destroy(loaded);
    gru_model_destroy(model);
    free(current_parameters);
    free(legacy_parameters);
    free(pre_joint_parameters);
    free(pre_target_parameters);
    remove(current_path);
    remove(current_temporary_path);
    remove(replacement_failure_temporary_path);
    TEST_RMDIR(replacement_failure_path);
    remove(legacy_path);
    remove(pre_joint_path);
    remove(pre_target_path);
    remove(version_path);
    remove(header_corrupt_path);
    remove(parameter_corrupt_path);
    remove(truncated_path);
    remove(truncated_parameters_path);
    remove(truncated_checksum_path);
    remove(trailing_path);
    return ok;
}

static int test_policy_evaluation_matches_legal_runtime_policy(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    Episode dual_episode;
    Episode single_episode;
    PolicyEvaluationMetrics dual_metrics;
    PolicyEvaluationMetrics single_metrics;
    PolicyEvaluationMetrics combined_metrics;
    FactorizedPolicySnapshot snapshot;
    float observation[4] = {0.25f, -0.5f, 0.75f, 0.1f};
    float hidden[8] = {0};
    uint8_t dual_legal[OBS_NUM_ACTIONS] = {0};
    uint8_t single_legal[OBS_NUM_ACTIONS] = {0};
    float* parameters = NULL;
    float* after = NULL;
    size_t parameter_count;
    float value = 0.0f;
    float joint_sum = 0.0f;
    int ok = 1;
    int i;

    memset(&dual_episode, 0, sizeof(dual_episode));
    memset(&single_episode, 0, sizeof(single_episode));
    if (!assert_true(model != NULL, "create model for policy evaluation")) return 0;
    parameter_count = gru_model_parameter_count(model);
    parameters = (float*)calloc(parameter_count, sizeof(float));
    after = (float*)malloc(parameter_count * sizeof(float));
    if (!parameters || !after ||
            !gru_model_import_parameters(model, parameters, parameter_count) ||
            !episode_init(&dual_episode, 2u, 4u) ||
            !episode_init(&single_episode, 1u, 4u)) {
        ok = assert_true(0, "initialize policy evaluation fixture");
        goto cleanup;
    }

    dual_legal[OBS_A1_MOVE1] = 1;
    dual_legal[OBS_A1_MOVE2] = 1;
    dual_legal[OBS_A2_MOVE1] = 1;
    dual_legal[OBS_A2_MOVE2] = 1;
    ok &= assert_true(gru_model_evaluate_policy_snapshot(
        model, hidden, dual_legal, 1, &snapshot, &value), "evaluate shared joint policy snapshot");
    ok &= assert_true(snapshot.has_joint_policy, "joint policy snapshot records joint mode");
    for (i = 0; i < FACTORIZED_JOINT_DIM; ++i) joint_sum += snapshot.joint_policy[i];
    ok &= assert_true(nearly_equal(joint_sum, 1.0, 1.0e-5), "joint policy is normalized");
    ok &= assert_true(snapshot.joint_policy[0] > 0.0f && snapshot.joint_policy[1] > 0.0f,
        "legal joint pairs receive probability");
    ok &= assert_true(snapshot.joint_policy[2] == 0.0f &&
        snapshot.joint_policy[FACTORIZED_LOCAL_ACTION_DIM * 2] == 0.0f,
        "illegal joint pairs have zero probability");

    ok &= assert_true(episode_append(&dual_episode, observation, dual_legal, OBS_A1_MOVE1, 0.0f, 0),
        "append dual policy evaluation turn");
    dual_episode.actions2[0] = OBS_A2_MOVE1;
    factorized_action_choice_from_flat_actions(
        &dual_episode.factorized_actions[0], OBS_A1_MOVE1, OBS_A2_MOVE1);
    dual_episode.factorized_actions[0].slot0_target_mask =
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    dual_episode.factorized_actions[0].slot1_target_mask =
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    dual_episode.factorized_actions[0].slot0_target_index = FACTORIZED_TARGET_FOE_LEFT;
    dual_episode.factorized_actions[0].slot1_target_index = FACTORIZED_TARGET_FOE_LEFT;
    ok &= assert_true(episode_append(&dual_episode, observation, dual_legal, OBS_A1_MOVE1, 0.0f, 1),
        "append dual turn with a missed target");
    dual_episode.actions2[1] = OBS_A2_MOVE1;
    factorized_action_choice_from_flat_actions(
        &dual_episode.factorized_actions[1], OBS_A1_MOVE1, OBS_A2_MOVE1);
    dual_episode.factorized_actions[1].slot0_target_mask =
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    dual_episode.factorized_actions[1].slot1_target_mask =
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
        FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    dual_episode.factorized_actions[1].slot0_target_index = FACTORIZED_TARGET_FOE_RIGHT;
    dual_episode.factorized_actions[1].slot1_target_index = FACTORIZED_TARGET_FOE_RIGHT;
    policy_evaluation_init(&dual_metrics);
    ok &= assert_true(policy_evaluation_add_episode(model, 16u, &dual_episode, &dual_metrics),
        "evaluate dual action episode");
    ok &= assert_true(dual_metrics.decision_turns == 2u && dual_metrics.action_labels == 4u,
        "dual action metrics count turns separately from slot actions");
    ok &= assert_true(dual_metrics.joint_pair_hits == 2u && dual_metrics.full_turn_hits == 1u,
        "a missed target lowers full-turn accuracy without changing pair accuracy");
    ok &= assert_true(dual_metrics.slot0_hits == 2u && dual_metrics.slot1_hits == 2u,
        "joint marginals score each active slot");
    ok &= assert_true(dual_metrics.target_hits == 2u && dual_metrics.target_labels == 4u,
        "target metrics include only targetable moves");
    ok &= assert_true(dual_metrics.top3_hits == 2u && dual_metrics.kind_hits == 4u &&
        dual_metrics.move_hits == 4u && dual_metrics.tera_hits == 4u,
        "legal top-three and factorized component metrics use their own denominators");
    ok &= assert_true(nearly_equal(policy_evaluation_action_nll(&dual_metrics), log(4.0), 1.0e-5),
        "joint negative log likelihood uses the legal pair distribution");
    ok &= assert_true(dual_metrics.illegal_predictions == 0u && dual_metrics.nonfinite_values == 0u,
        "dual evaluation stays legal and finite");

    single_legal[OBS_A2_SWITCH3] = 1;
    ok &= assert_true(episode_append(&single_episode, observation, single_legal, OBS_A2_SWITCH3, 0.0f, 1),
        "append single slot policy evaluation turn");
    single_episode.actions[0] = -1;
    single_episode.actions2[0] = OBS_A2_SWITCH3;
    factorized_action_choice_from_flat_actions(
        &single_episode.factorized_actions[0], -1, OBS_A2_SWITCH3);
    policy_evaluation_init(&single_metrics);
    ok &= assert_true(policy_evaluation_add_episode(model, 16u, &single_episode, &single_metrics),
        "evaluate single slot episode");
    ok &= assert_true(single_metrics.joint_pair_labels == 0u && single_metrics.slot1_hits == 1u,
        "single slot evaluation does not invent a joint decision");
    ok &= assert_true(single_metrics.full_turn_hits == 1u && single_metrics.switch_hits == 1u,
        "single legal switch scores as a correct full turn");

    policy_evaluation_init(&combined_metrics);
    policy_evaluation_merge(&combined_metrics, &dual_metrics);
    policy_evaluation_merge(&combined_metrics, &single_metrics);
    ok &= assert_true(combined_metrics.sessions == 2u && combined_metrics.decision_turns == 3u &&
        combined_metrics.action_labels == 5u, "policy evaluation metrics merge without losing denominators");
    ok &= assert_true(gru_model_export_parameters(model, after, parameter_count) &&
        memcmp(parameters, after, parameter_count * sizeof(float)) == 0,
        "policy evaluation does not mutate model parameters");

cleanup:
    episode_free(&single_episode);
    episode_free(&dual_episode);
    free(after);
    free(parameters);
    gru_model_destroy(model);
    return ok;
}

static int initialize_learning_episode(Episode* episode, float observation_scale, float reward, int include_targets) {
    float observation[4] = {
        0.25f * observation_scale,
        -0.5f * observation_scale,
        0.75f * observation_scale,
        0.1f * observation_scale
    };
    uint8_t legal[OBS_NUM_ACTIONS] = {0};
    FactorizedActionChoice* choice;

    legal[OBS_A1_MOVE1] = 1;
    legal[OBS_A1_MOVE2] = 1;
    legal[OBS_A2_MOVE1] = 1;
    legal[OBS_A2_MOVE2] = 1;
    if (!episode_init(episode, 1u, 4u) ||
            !episode_append(episode, observation, legal, OBS_A1_MOVE2, reward, 1)) {
        return 0;
    }
    episode->actions2[0] = OBS_A2_MOVE2;
    choice = &episode->factorized_actions[0];
    factorized_action_choice_from_flat_actions(choice, OBS_A1_MOVE2, OBS_A2_MOVE2);
    if (include_targets) {
        choice->slot0_target_mask = FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
            FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
        choice->slot1_target_mask = FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT) |
            FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
        choice->slot0_target_index = FACTORIZED_TARGET_FOE_RIGHT;
        choice->slot1_target_index = FACTORIZED_TARGET_FOE_RIGHT;
    }
    return 1;
}

static int zero_model_parameters(GruModel* model) {
    size_t count;
    float* parameters;
    int ok;
    if (!model) return 0;
    count = gru_model_parameter_count(model);
    parameters = (float*)calloc(count, sizeof(float));
    if (!parameters) return 0;
    ok = gru_model_import_parameters(model, parameters, count);
    free(parameters);
    return ok;
}

static int test_supervised_overfit_diagnostic_learns(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    GruTrainer trainer;
    Episode first;
    Episode second;
    const Episode* episodes[2];
    SupervisedOverfitResult result;
    int ok = 1;

    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    if (!assert_true(model != NULL && zero_model_parameters(model) &&
            initialize_learning_episode(&first, 1.0f, 1.0f, 1) &&
            initialize_learning_episode(&second, -1.0f, 1.0f, 1),
            "initialize deterministic supervised overfit fixture")) {
        episode_free(&second);
        episode_free(&first);
        gru_model_destroy(model);
        return 0;
    }
    gru_trainer_init(&trainer, 0.03f, 16u, 1.0f, 17u);
    trainer.supervised_optimizer = GRU_SUPERVISED_OPTIMIZER_ADAM;
    trainer.supervised_profile_enabled = 0;
    episodes[0] = &first;
    episodes[1] = &second;
    ok &= assert_true(learning_diagnostic_run_supervised_overfit(
        &trainer, model, episodes, 2u, 40u, &result), "run deterministic supervised overfit check");
    ok &= assert_true(result.passed, "supervised optimizer satisfies all learnability criteria");
    ok &= assert_true(result.action_loss_reduction >= 0.5,
        "supervised demonstrated-action loss falls by at least half");
    ok &= assert_true(policy_evaluation_full_turn_accuracy(&result.after) >= 0.8,
        "supervised full-turn accuracy reaches the diagnostic threshold");

    episode_free(&second);
    episode_free(&first);
    gru_model_destroy(model);
    return ok;
}

static int selected_joint_probability_and_value(
    const GruModel* model,
    const Episode* episode,
    float* probability_out,
    float* value_out
) {
    float hidden[8] = {0};
    float flat_policy[OBS_NUM_ACTIONS];
    FactorizedPolicySnapshot snapshot;
    int action0;
    int action1;
    size_t pair_index;
    if (!model || !episode || episode->count != 1u || !probability_out || !value_out) return 0;
    gru_model_forward_sequence(model, episode->observations, 1u, hidden, flat_policy, value_out);
    if (!gru_model_evaluate_policy_snapshot(
            model, hidden, episode->legal_masks, 1, &snapshot, value_out) ||
            !factorized_action_choice_to_flat_actions(
                &episode->factorized_actions[0], &action0, &action1)) {
        return 0;
    }
    pair_index = (size_t)action0 * FACTORIZED_LOCAL_ACTION_DIM +
        (size_t)(action1 - FACTORIZED_LOCAL_ACTION_DIM);
    *probability_out = snapshot.joint_policy[pair_index];
    return 1;
}

static int initialize_two_step_ppo_episode(Episode* episode, int move_slot, float terminal_reward) {
    float first_observation[4] = {0.25f, -0.5f, 0.75f, 0.1f};
    float second_observation[4] = {0.3f, -0.4f, 0.7f, 0.2f};
    uint8_t legal[OBS_NUM_ACTIONS] = {0};
    int action0 = move_slot == 1 ? OBS_A1_MOVE1 : OBS_A1_MOVE2;
    int action1 = move_slot == 1 ? OBS_A2_MOVE1 : OBS_A2_MOVE2;
    size_t t;

    legal[OBS_A1_MOVE1] = 1;
    legal[OBS_A1_MOVE2] = 1;
    legal[OBS_A2_MOVE1] = 1;
    legal[OBS_A2_MOVE2] = 1;
    if (!episode_init(episode, 2u, 4u) ||
            !episode_append(episode, first_observation, legal, action0, 0.0f, 0) ||
            !episode_append(episode, second_observation, legal, action0, terminal_reward, 1)) {
        return 0;
    }
    for (t = 0; t < episode->count; ++t) {
        episode->actions2[t] = action1;
        factorized_action_choice_from_flat_actions(&episode->factorized_actions[t], action0, action1);
    }
    return 1;
}

static int selected_joint_probability_and_value_at(
    const GruModel* model,
    const Episode* episode,
    size_t step,
    float* probability_out,
    float* value_out
) {
    float hidden[8] = {0};
    float flat_policy[OBS_NUM_ACTIONS];
    FactorizedPolicySnapshot snapshot;
    int action0;
    int action1;
    size_t pair_index;
    if (!model || !episode || step >= episode->count || !probability_out || !value_out) return 0;
    gru_model_forward_sequence(model, episode->observations, step + 1u, hidden, flat_policy, value_out);
    if (!gru_model_evaluate_policy_snapshot(
            model, hidden, episode->legal_masks + step * OBS_NUM_ACTIONS, 1, &snapshot, value_out) ||
            !factorized_action_choice_to_flat_actions(
                &episode->factorized_actions[step], &action0, &action1)) {
        return 0;
    }
    pair_index = (size_t)action0 * FACTORIZED_LOCAL_ACTION_DIM +
        (size_t)(action1 - FACTORIZED_LOCAL_ACTION_DIM);
    *probability_out = snapshot.joint_policy[pair_index];
    return 1;
}

static int populate_ppo_behavior_predictions(GruModel* model, Episode* episode) {
    size_t t;
    for (t = 0; t < episode->count; ++t) {
        float probability;
        float value;
        if (!selected_joint_probability_and_value_at(model, episode, t, &probability, &value) ||
                probability <= 0.0f) {
            return 0;
        }
        episode->old_log_probs[t] = logf(probability);
        episode->old_values[t] = value;
    }
    return 1;
}

static int test_ppo_normalizes_advantages_across_minibatch(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    Episode winning_episode;
    Episode losing_episode;
    const Episode* minibatch[2];
    GruTrainer trainer;
    float winning_probability_before;
    float winning_probability_after;
    float losing_probability_before;
    float losing_probability_after;
    float ignored_value;
    int ok = 1;

    memset(&winning_episode, 0, sizeof(winning_episode));
    memset(&losing_episode, 0, sizeof(losing_episode));
    if (!assert_true(model && zero_model_parameters(model) &&
            initialize_two_step_ppo_episode(&winning_episode, 2, 1.0f) &&
            initialize_two_step_ppo_episode(&losing_episode, 1, -1.0f) &&
            populate_ppo_behavior_predictions(model, &winning_episode) &&
            populate_ppo_behavior_predictions(model, &losing_episode),
            "initialize mixed-outcome PPO minibatch fixture")) {
        ok = 0;
        goto cleanup;
    }
    ok &= assert_true(selected_joint_probability_and_value_at(
        model, &winning_episode, 0u, &winning_probability_before, &ignored_value),
        "evaluate winning episode before normalized PPO minibatch");
    ok &= assert_true(selected_joint_probability_and_value_at(
        model, &losing_episode, 0u, &losing_probability_before, &ignored_value),
        "evaluate losing episode before normalized PPO minibatch");

    gru_trainer_init(&trainer, 0.01f, 16u, 1.0f, 29u);
    trainer.advantage_norm = 1;
    trainer.entropy_coef = 0.0f;
    minibatch[0] = &winning_episode;
    minibatch[1] = &losing_episode;
    ok &= assert_true(gru_trainer_ppo_minibatch(&trainer, model, minibatch, 2u),
        "apply production normalized PPO minibatch");
    ok &= assert_true(fabsf(trainer.last_mean_advantage) < 1.0e-5f,
        "minibatch-normalized advantages have a shared zero mean");
    ok &= assert_true(selected_joint_probability_and_value_at(
        model, &winning_episode, 0u, &winning_probability_after, &ignored_value),
        "evaluate winning episode after normalized PPO minibatch");
    ok &= assert_true(selected_joint_probability_and_value_at(
        model, &losing_episode, 0u, &losing_probability_after, &ignored_value),
        "evaluate losing episode after normalized PPO minibatch");
    ok &= assert_true(winning_probability_after > winning_probability_before,
        "shared normalization reinforces actions from the winning episode");
    ok &= assert_true(losing_probability_after < losing_probability_before,
        "shared normalization suppresses actions from the losing episode");

cleanup:
    episode_free(&losing_episode);
    episode_free(&winning_episode);
    gru_model_destroy(model);
    return ok;
}

static int test_ppo_clipped_policy_still_updates_value(void) {
    GruModel* model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    Episode episode;
    GruTrainer trainer;
    float probability;
    float value_before;
    float value_after;
    int ok = 1;

    memset(&episode, 0, sizeof(episode));
    if (!assert_true(model && zero_model_parameters(model) &&
            initialize_learning_episode(&episode, 1.0f, 1.0f, 0) &&
            selected_joint_probability_and_value(model, &episode, &probability, &value_before),
            "initialize clipped-policy value fixture")) {
        ok = 0;
        goto cleanup;
    }
    episode.old_values[0] = value_before;
    episode.old_log_probs[0] = logf(probability) - logf(2.0f);
    gru_trainer_init(&trainer, 0.01f, 16u, 1.0f, 31u);
    trainer.advantage_norm = 0;
    trainer.entropy_coef = 0.0f;
    ok &= assert_true(gru_trainer_ppo_episode(&trainer, model, &episode),
        "apply PPO update with clipped positive policy advantage");
    ok &= assert_true(selected_joint_probability_and_value(
        model, &episode, &probability, &value_after),
        "evaluate critic after clipped policy update");
    ok &= assert_true(value_after > value_before,
        "critic still learns when PPO clips the policy term");

cleanup:
    episode_free(&episode);
    gru_model_destroy(model);
    return ok;
}

static int test_dual_action_turn_has_one_value_target(void) {
    GruModel* single_model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    GruModel* dual_model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    Episode single_episode;
    Episode dual_episode;
    GruTrainer single_trainer;
    GruTrainer dual_trainer;
    float hidden[8] = {0};
    float policy[OBS_NUM_ACTIONS];
    float single_value;
    float dual_value;
    int ok = 1;

    memset(&single_episode, 0, sizeof(single_episode));
    memset(&dual_episode, 0, sizeof(dual_episode));
    if (!assert_true(single_model && dual_model &&
            zero_model_parameters(single_model) && zero_model_parameters(dual_model) &&
            initialize_learning_episode(&single_episode, 1.0f, 1.0f, 0) &&
            initialize_learning_episode(&dual_episode, 1.0f, 1.0f, 0),
            "initialize single- and dual-action value fixtures")) {
        ok = 0;
        goto cleanup;
    }
    single_episode.actions2[0] = -1;
    factorized_action_choice_from_flat_actions(
        &single_episode.factorized_actions[0], single_episode.actions[0], -1);
    gru_trainer_init(&single_trainer, 0.01f, 16u, 100.0f, 37u);
    gru_trainer_init(&dual_trainer, 0.01f, 16u, 100.0f, 37u);
    ok &= assert_true(gru_trainer_supervised_episode(
        &single_trainer, single_model, &single_episode), "train single-action value fixture");
    ok &= assert_true(gru_trainer_supervised_episode(
        &dual_trainer, dual_model, &dual_episode), "train dual-action value fixture");
    gru_model_forward_sequence(single_model, single_episode.observations, 1u, hidden, policy, &single_value);
    memset(hidden, 0, sizeof(hidden));
    gru_model_forward_sequence(dual_model, dual_episode.observations, 1u, hidden, policy, &dual_value);
    ok &= assert_true(fabsf(single_value - dual_value) < 1.0e-6f,
        "dual-action turn applies the turn-level value target once");

cleanup:
    episode_free(&dual_episode);
    episode_free(&single_episode);
    gru_model_destroy(dual_model);
    gru_model_destroy(single_model);
    return ok;
}

static int test_ppo_critic_diagnostics(void) {
    GruTrainer trainer;
    memset(&trainer, 0, sizeof(trainer));
    trainer.last_critic_samples = 2u;
    trainer.last_return_sum = 0.0;
    trainer.last_return_square_sum = 2.0;
    trainer.last_value_sum = 0.0;
    trainer.last_value_square_sum = 0.5;
    trainer.last_value_error_sum = 0.0;
    trainer.last_value_error_square_sum = 0.5;
    trainer.last_return_value_product_sum = 1.0;
    if (!assert_true(fabs(gru_trainer_critic_explained_variance(&trainer) - 0.75) < 1.0e-9,
            "critic explained variance uses return and residual variance")) return 0;
    return assert_true(fabs(gru_trainer_return_value_correlation(&trainer) - 1.0) < 1.0e-9,
        "critic diagnostic reports return/value correlation");
}

static int test_ppo_update_moves_policy_and_value_in_expected_directions(void) {
    GruModel* positive_model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    GruModel* negative_model = gru_model_create(4u, 8u, OBS_NUM_ACTIONS);
    Episode positive_episode;
    Episode negative_episode;
    GruTrainer positive_trainer;
    GruTrainer negative_trainer;
    float positive_probability_before;
    float positive_probability_after;
    float negative_probability_before;
    float negative_probability_after;
    float positive_value_before;
    float positive_value_after;
    float negative_value_before;
    float negative_value_after;
    int ok = 1;

    memset(&positive_episode, 0, sizeof(positive_episode));
    memset(&negative_episode, 0, sizeof(negative_episode));
    if (!assert_true(positive_model && negative_model &&
            zero_model_parameters(positive_model) && zero_model_parameters(negative_model) &&
            initialize_learning_episode(&positive_episode, 1.0f, 1.0f, 0) &&
            initialize_learning_episode(&negative_episode, 1.0f, -1.0f, 0),
            "initialize deterministic PPO direction fixtures")) {
        ok = 0;
        goto cleanup;
    }
    ok &= assert_true(selected_joint_probability_and_value(
        positive_model, &positive_episode, &positive_probability_before, &positive_value_before),
        "evaluate positive-advantage PPO fixture before update");
    ok &= assert_true(selected_joint_probability_and_value(
        negative_model, &negative_episode, &negative_probability_before, &negative_value_before),
        "evaluate negative-advantage PPO fixture before update");
    positive_episode.old_log_probs[0] = logf(positive_probability_before);
    positive_episode.old_values[0] = positive_value_before;
    negative_episode.old_log_probs[0] = logf(negative_probability_before);
    negative_episode.old_values[0] = negative_value_before;

    gru_trainer_init(&positive_trainer, 0.01f, 16u, 0.0f, 23u);
    gru_trainer_init(&negative_trainer, 0.01f, 16u, 0.0f, 23u);
    positive_trainer.advantage_norm = 0;
    negative_trainer.advantage_norm = 0;
    positive_trainer.entropy_coef = 0.0f;
    negative_trainer.entropy_coef = 0.0f;
    ok &= assert_true(gru_trainer_ppo_episode(&positive_trainer, positive_model, &positive_episode),
        "apply positive-advantage PPO update");
    ok &= assert_true(gru_trainer_ppo_episode(&negative_trainer, negative_model, &negative_episode),
        "apply negative-advantage PPO update");
    ok &= assert_true(selected_joint_probability_and_value(
        positive_model, &positive_episode, &positive_probability_after, &positive_value_after),
        "evaluate positive-advantage PPO fixture after update");
    ok &= assert_true(selected_joint_probability_and_value(
        negative_model, &negative_episode, &negative_probability_after, &negative_value_after),
        "evaluate negative-advantage PPO fixture after update");
    ok &= assert_true(positive_probability_after > positive_probability_before,
        "positive advantage raises the demonstrated joint-action probability");
    ok &= assert_true(negative_probability_after < negative_probability_before,
        "negative advantage lowers the demonstrated joint-action probability");
    ok &= assert_true(fabsf(1.0f - positive_value_after) < fabsf(1.0f - positive_value_before) &&
        fabsf(-1.0f - negative_value_after) < fabsf(-1.0f - negative_value_before),
        "PPO value predictions move toward their returns");

cleanup:
    episode_free(&negative_episode);
    episode_free(&positive_episode);
    gru_model_destroy(negative_model);
    gru_model_destroy(positive_model);
    return ok;
}

static int test_validation_split_is_stable_and_seeded(void) {
    uint64_t expected = UINT64_C(5406646322143240280);
    if (!assert_true(validation_split_hash("battle-4", 1337u) == expected,
            "validation split hash stays stable across builds")) return 0;
    if (!assert_true(validation_split_contains("battle-4", 1337u),
            "stable hash assigns matching battle to validation")) return 0;
    if (!assert_true(!validation_split_contains("battle-alpha", 1337u),
            "stable hash leaves nonmatching battle in training")) return 0;
    return assert_true(
        validation_split_hash("battle-alpha", 1337u) != validation_split_hash("battle-alpha", 1338u),
        "validation seed changes the split hash");
}

int main(int argc, char** argv) {
    if (!id_tables_init()) {
        fprintf(stderr, "failed to initialize id tables\n");
        return 1;
    }
    if (argc >= 3 && strcmp(argv[1], "--batch-replay") == 0) {
        return run_batch_replay_mode(argv[2]);
    }
    if (argc > 1) {
        fprintf(stderr, "unknown arguments\n");
        fprintf(stderr, "usage:\n");
        fprintf(stderr, "  %s\n", argv[0]);
        fprintf(stderr, "  %s --batch-replay <jsonl_path>\n", argv[0]);
        return 1;
    }
    if (!test_episode_target_roundtrip()) return 1;
    if (!test_factorized_target_head_training()) return 1;
    if (!test_factorized_ppo_anchor_regularization()) return 1;
    if (!test_ppo_hard_kl_requires_consecutive_breaches()) return 1;
    if (!test_symmetric_joint_action_training()) return 1;
    if (!test_shared_entity_encoder_training_and_migration()) return 1;
    if (!test_active_slot_schema_migrates_legacy_checkpoint()) return 1;
    if (!test_checkpoint_compatibility_validation()) return 1;
    if (!test_policy_evaluation_matches_legal_runtime_policy()) return 1;
    if (!test_supervised_overfit_diagnostic_learns()) return 1;
    if (!test_ppo_update_moves_policy_and_value_in_expected_directions()) return 1;
    if (!test_ppo_normalizes_advantages_across_minibatch()) return 1;
    if (!test_ppo_clipped_policy_still_updates_value()) return 1;
    if (!test_dual_action_turn_has_one_value_target()) return 1;
    if (!test_ppo_critic_diagnostics()) return 1;
    if (!test_validation_split_is_stable_and_seeded()) return 1;
    if (!test_request_reconciliation_preserves_identity()) return 1;
    if (!test_observation_request_flags_and_side_features()) return 1;
    if (!test_observation_exports_active_slot_identity()) return 1;
    if (!test_find_or_make_does_not_overwrite_full_team()) return 1;
    if (!test_condition_status_without_hp_preserves_hp()) return 1;
    if (!test_turn_number_not_overwritten_by_request()) return 1;
    if (!test_runtime_request_session_not_forced_doubles()) return 1;
    if (!test_runtime_dense_additive_rewards()) return 1;
    if (!test_single_turn_side_guards_reconstructed()) return 1;
    if (!test_switch_clears_volatile_state()) return 1;
    if (!test_hazard_layers_are_capped()) return 1;
    if (!test_request_active_data_respects_board_slots()) return 1;
    if (!test_protect_chain_resets_when_not_used()) return 1;
    if (!test_toxic_counter_progresses_and_resets_on_switch()) return 1;
    if (!test_sleep_turns_elapsed_over_time()) return 1;
    if (!test_remaining_pokemon_tracks_faints()) return 1;
    if (!test_transform_effective_state_and_switch_cleanup()) return 1;
    if (!test_first_doubles_request_bootstrap_succeeds()) return 1;
    if (!test_p2_request_sets_self_perspective_and_reuses_event_state()) return 1;
    if (!test_switch_identity_prefers_exact_or_empty_over_same_name_reuse()) return 1;
    if (!test_transform_uses_battle_effective_pp()) return 1;
    if (!test_transform_survives_request_reconciliation()) return 1;
    if (!test_same_name_same_species_does_not_merge()) return 1;
    if (!test_transformed_move_event_does_not_contaminate_base_moves()) return 1;
    if (!test_ambiguous_non_slot_event_fails_closed()) return 1;
    if (!test_event_switch_rebuilds_slot_maps_for_followup_events()) return 1;
    if (!test_faint_event_zeros_hp_in_state_and_observation()) return 1;
    if (!test_faint_condition_without_slash_zeros_hp()) return 1;
    if (!test_request_parser_reads_private_tera_type_even_when_moves_disabled()) return 1;
    if (!test_request_parser_reads_private_side_item_ability_tera_and_moves()) return 1;
    if (!test_request_parser_reads_private_side_stats_and_flags()) return 1;
    if (!test_request_reconciliation_imports_private_side_metadata()) return 1;
    if (!test_request_reconciliation_imports_stats_flags_and_trapped_state()) return 1;
    if (!test_request_reconciliation_infers_encore_move_slot()) return 1;
    if (!test_event_parser_sets_flinch_and_disable_slot()) return 1;
    if (!test_switch_and_faint_clear_boosts()) return 1;
    if (!test_observation_hides_inferred_weather_and_exports_more_transients()) return 1;
    if (!test_event_parser_reveals_public_abilities_typechange_and_cant_status()) return 1;
    if (!test_event_parser_formechange_updates_species_and_reveals_ability()) return 1;
    if (!test_request_reconciliation_preserves_can_tera_through_wait_and_forced_switch()) return 1;
    if (!test_real_battle_2632274530_faint_force_switch_and_tera_request()) return 1;
    if (!test_real_battle_2632287191_switch_tera_and_force_switch()) return 1;
    if (!test_real_battle_2632276902_houndoom_tera_and_tailwind()) return 1;
    if (!test_real_battle_2632278612_salamence_tera_faint_and_whiscash_sleep()) return 1;
    if (!test_real_battle_2632283886_magmortar_tera_and_perrserker_burn_faint()) return 1;
    if (!test_real_battle_2632285682_dusknoir_paralyzed_dondozo_and_frosmoth_faint()) return 1;
    if (!test_real_battle_2632291348_wait_replacement_tera_and_faint()) return 1;
    if (!test_real_battle_2632295968_toxic_spikes_and_glalie_poison()) return 1;
    if (!test_real_battle_2632300182_weavile_tera_and_opponent_tailwind()) return 1;
    if (!test_real_battle_2632302019_greninja_faint_and_trick_room()) return 1;
    if (!test_multiturn_real_battle_2632274530_three_turn_progression()) return 1;
    if (!test_multiturn_real_battle_2632276902_houndoom_across_three_turns()) return 1;
    if (!test_multiturn_real_battle_2632278612_tera_then_forced_switch_sequence()) return 1;
    if (!test_multiturn_real_battle_2632283886_tera_tailwind_and_switch_chain()) return 1;
    if (!test_multiturn_real_battle_2632285682_opening_replacements_over_two_turns()) return 1;
    if (!test_multiturn_real_battle_2632287191_miraidon_to_quagsire_sequence()) return 1;
    if (!test_multiturn_real_battle_2632291348_replacement_chain_over_three_turns()) return 1;
    if (!test_multiturn_real_battle_2632295968_turn_one_and_two_continuity()) return 1;
    if (!test_multiturn_real_battle_2632300182_weather_tera_tailwind_sequence()) return 1;
    if (!test_multiturn_real_battle_2632302019_trick_room_carries_to_turn_ten()) return 1;
    if (!test_multiturn_capture_replay_2632274530_turn2_request_state()) return 1;
    if (!test_multiturn_capture_replay_2632288269_turn2_request_state()) return 1;
    if (!test_multiturn_capture_replay_2632290515_turn2_request_state()) return 1;
    if (!test_multiturn_capture_replay_2632293423_turn2_request_state()) return 1;
    if (!test_multiturn_capture_replay_2632310612_turn2_request_state()) return 1;
    if (!test_full_battle_replay_2632274530()) return 1;
    if (!test_full_battle_replay_2632276902()) return 1;
    if (!test_full_battle_replay_2632278612()) return 1;
    if (!test_full_battle_replay_2632283886()) return 1;
    if (!test_full_battle_replay_2632285682()) return 1;
    if (!test_full_battle_replay_2632287191()) return 1;
    if (!test_full_battle_replay_2632288269()) return 1;
    if (!test_full_battle_replay_2632290515()) return 1;
    if (!test_full_battle_replay_2632293423()) return 1;
    if (!test_full_battle_replay_2632310612()) return 1;
    if (!test_full_battle_replay_2636632844_gliscor_terminal_state()) return 1;
    if (!test_full_battle_replay_2637505742_zeroes_fainted_reserve_hp()) return 1;
    if (!test_synthetic_sideeffect_prefix_tailwind()) return 1;
    if (!test_synthetic_sideeffect_prefix_reflect()) return 1;
    if (!test_synthetic_sideeffect_prefix_light_screen()) return 1;
    if (!test_synthetic_sideeffect_prefix_aurora_veil()) return 1;
    if (!test_synthetic_sideeffect_prefix_safeguard()) return 1;
    if (!test_synthetic_sideeffect_prefix_mist()) return 1;
    if (!test_synthetic_sideeffect_prefix_lucky_chant()) return 1;
    if (!test_synthetic_sideeffect_prefix_toxic_spikes()) return 1;
    if (!test_synthetic_sideeffect_prefix_sticky_web()) return 1;
    if (!test_synthetic_sideeffect_prefix_spikes_cap()) return 1;
    if (!test_synthetic_weather_overwrite_resets_counter()) return 1;
    if (!test_synthetic_terrain_overwrite_resets_counter()) return 1;
    if (!test_synthetic_field_end_clears_trick_room()) return 1;
    if (!test_synthetic_field_end_clears_gravity()) return 1;
    if (!test_synthetic_tailwind_expires_after_four_turns()) return 1;
    if (!test_synthetic_reflect_expires_after_five_turns()) return 1;
    if (!test_synthetic_sleep_does_not_progress_while_benched()) return 1;
    if (!test_synthetic_toxic_does_not_progress_while_benched()) return 1;
    if (!test_synthetic_singleturn_flags_clear_on_next_turn()) return 1;
    if (!test_synthetic_yawn_duration_expires_across_turns()) return 1;
    if (!test_synthetic_sideend_clears_reflect()) return 1;
    if (!test_synthetic_sideend_clears_tailwind()) return 1;
    if (!test_synthetic_sideend_clears_sticky_web()) return 1;
    if (!test_synthetic_weather_clear_sets_unknown_duration()) return 1;
    printf("reconstruction tests passed\n");
    return 0;
}
