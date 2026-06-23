#include "request_parser.h"
#include "id_tables.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static const char* skip_ws(const char* p) {
    while (p && *p && isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static const char* find_after_key(const char* json, const char* key) {
    char pattern[64];
    size_t n;

    if (!json || !key) {
        return NULL;
    }
    n = strlen(key);
    if (n + 4 >= sizeof(pattern)) {
        return NULL;
    }
    pattern[0] = '"';
    memcpy(pattern + 1, key, n);
    pattern[n + 1] = '"';
    pattern[n + 2] = ':';
    pattern[n + 3] = '\0';
    return strstr(json, pattern);
}

static int parse_int_after(const char* json, const char* key, int default_value) {
    const char* p = find_after_key(json, key);
    if (!p) {
        return default_value;
    }
    p = strchr(p, ':');
    if (!p) {
        return default_value;
    }
    p = skip_ws(p + 1);
    return atoi(p);
}

static int parse_bool_after(const char* json, const char* key, int default_value) {
    const char* p = find_after_key(json, key);
    if (!p) {
        return default_value;
    }
    p = strchr(p, ':');
    if (!p) {
        return default_value;
    }
    p = skip_ws(p + 1);
    if (strncmp(p, "true", 4) == 0) {
        return 1;
    }
    if (strncmp(p, "false", 5) == 0) {
        return 0;
    }
    return default_value;
}

static int extract_json_array(const char* start, char* out, size_t out_len) {
    int depth = 0;
    int in_string = 0;
    size_t i = 0;
    const char* p = start;

    if (!p || *p != '[' || out_len == 0) {
        return 0;
    }
    while (*p && i + 1 < out_len) {
        char ch = *p;
        out[i++] = ch;
        if (ch == '"' && (p == start || p[-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (ch == '[') {
                depth++;
            } else if (ch == ']') {
                depth--;
                if (depth == 0) {
                    out[i] = '\0';
                    return 1;
                }
            }
        }
        ++p;
    }
    if (i < out_len) {
        out[i] = '\0';
    }
    return 0;
}

static int extract_json_object(const char* start, char* out, size_t out_len) {
    int depth = 0;
    int in_string = 0;
    size_t i = 0;
    const char* p = start;

    if (!p || *p != '{' || out_len == 0) {
        return 0;
    }
    while (*p && i + 1 < out_len) {
        char ch = *p;
        out[i++] = ch;
        if (ch == '"' && (p == start || p[-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (ch == '{') {
                depth++;
            } else if (ch == '}') {
                depth--;
                if (depth == 0) {
                    out[i] = '\0';
                    return 1;
                }
            }
        }
        ++p;
    }
    if (i < out_len) {
        out[i] = '\0';
    }
    return 0;
}

static const char* next_top_level_object_in_array(const char* array_json, const char* cursor) {
    int array_depth = (cursor && cursor != array_json) ? 1 : 0;
    int in_string = 0;
    const char* p;

    if (!array_json) {
        return NULL;
    }
    p = cursor ? cursor : array_json;
    while (*p) {
        char ch = *p;
        if (ch == '"' && (p == array_json || p[-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (ch == '[') {
                ++array_depth;
            } else if (ch == ']') {
                if (array_depth == 1) {
                    return NULL;
                }
                if (array_depth > 0) {
                    --array_depth;
                }
            } else if (ch == '{' && array_depth == 1) {
                return p;
            }
        }
        ++p;
    }
    return NULL;
}

static const char* skip_json_object_text(const char* start) {
    int depth = 0;
    int in_string = 0;
    const char* p = start;

    if (!p || *p != '{') {
        return NULL;
    }
    while (*p) {
        char ch = *p;
        if (ch == '"' && (p == start || p[-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (ch == '{') {
                ++depth;
            } else if (ch == '}') {
                --depth;
                if (depth == 0) {
                    return p + 1;
                }
            }
        }
        ++p;
    }
    return NULL;
}

static int parse_move_id_from_object(const char* obj) {
    const char* p = strstr(obj, "\"id\":");
    char token[64];
    size_t i = 0;
    if (!p) {
        return 0;
    }
    p = strchr(p, '"');
    if (!p) {
        return 0;
    }
    p = strchr(p + 1, '"');
    if (!p) {
        return 0;
    }
    p = strchr(p + 1, '"');
    if (!p) {
        return 0;
    }
    ++p;
    while (*p && *p != '"' && i + 1 < sizeof(token)) {
        token[i++] = *p++;
    }
    token[i] = '\0';
    return move_id_from_name(token);
}

static int parse_move_id_array_from_string(const char* text, int out_ids[PARSED_REQUEST_MOVE_SLOTS]) {
    const char* p;
    int count = 0;
    if (!text || !out_ids) {
        return 0;
    }
    p = strchr(text, '[');
    if (!p) {
        return 0;
    }
    ++p;
    while (*p && count < PARSED_REQUEST_MOVE_SLOTS) {
        while (*p && *p != '"' && *p != ']') {
            ++p;
        }
        if (*p == ']') {
            break;
        }
        if (*p != '"') {
            break;
        }
        {
            char token[64];
            size_t i = 0;
            ++p;
            while (*p && *p != '"' && i + 1 < sizeof(token)) {
                token[i++] = *p++;
            }
            token[i] = '\0';
            out_ids[count++] = move_id_from_name(token);
        }
        while (*p && *p != ',' && *p != ']') {
            ++p;
        }
        if (*p == ',') {
            ++p;
        }
    }
    return count;
}

static int parse_string_id_after(const char* json, const char* key, int (*fn)(const char*)) {
    const char* p = find_after_key(json, key);
    char token[64];
    size_t i = 0;
    if (!p || !fn) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    p = skip_ws(p + 1);
    if (*p != '"') {
        return 0;
    }
    ++p;
    while (*p && *p != '"' && i + 1 < sizeof(token)) {
        token[i++] = *p++;
    }
    token[i] = '\0';
    return fn(token);
}

static ParsedMoveTarget parse_move_target_value(const char* obj) {
    const char* p = find_after_key(obj, "target");
    char token[64];
    size_t i = 0;
    if (!p) {
        return REQUEST_TARGET_UNKNOWN;
    }
    p = strchr(p, ':');
    if (!p) {
        return REQUEST_TARGET_UNKNOWN;
    }
    p = skip_ws(p + 1);
    if (*p != '"') {
        return REQUEST_TARGET_UNKNOWN;
    }
    ++p;
    while (*p && *p != '"' && i + 1 < sizeof(token)) {
        token[i++] = *p++;
    }
    token[i] = '\0';

    if (strcmp(token, "normal") == 0) return REQUEST_TARGET_NORMAL;
    if (strcmp(token, "adjacentFoe") == 0) return REQUEST_TARGET_ADJACENT_FOE;
    if (strcmp(token, "any") == 0) return REQUEST_TARGET_ANY;
    if (strcmp(token, "adjacentAlly") == 0) return REQUEST_TARGET_ADJACENT_ALLY;
    if (strcmp(token, "adjacentAllyOrSelf") == 0) return REQUEST_TARGET_ADJACENT_ALLY_OR_SELF;
    if (strcmp(token, "self") == 0) return REQUEST_TARGET_SELF;
    if (strcmp(token, "allAdjacentFoes") == 0) return REQUEST_TARGET_ALL_ADJACENT_FOES;
    if (strcmp(token, "all") == 0) return REQUEST_TARGET_ALL;
    if (strcmp(token, "allySide") == 0) return REQUEST_TARGET_ALLY_SIDE;
    if (strcmp(token, "foeSide") == 0) return REQUEST_TARGET_FOE_SIDE;
    return REQUEST_TARGET_UNKNOWN;
}

static int parse_json_string_field(const char* obj, const char* key, char* out, size_t out_len) {
    const char* p = find_after_key(obj, key);
    size_t i = 0;
    if (!p || !out || out_len == 0) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    p = skip_ws(p + 1);
    if (*p != '"') {
        return 0;
    }
    ++p;
    while (*p && *p != '"' && i + 1 < out_len) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return 1;
}

void parsed_request_init(ParsedRequest* req) {
    int i;
    if (!req) {
        return;
    }
    memset(req, 0, sizeof(*req));
    req->max_chosen_team_size = 2;
    req->side_player = 0;
    for (i = 0; i < PARSED_REQUEST_ACTIVE_SLOTS; ++i) {
        req->active_team_idx[i] = -1;
        req->active_team_idx_known[i] = 0;
    }
}

static void finalize_slot_semantics(ParsedRequest* req) {
    int slot;
    if (!req) {
        return;
    }
    for (slot = 0; slot < PARSED_REQUEST_ACTIVE_SLOTS; ++slot) {
        int bench_switch_exists = 0;
        int move_exists = 0;
        int i;
        req->slot_present[slot] = (slot < req->active_count) || req->force_switch[slot];
        req->slot_needs_choice[slot] = 0;
        req->slot_can_move[slot] = 0;
        req->slot_can_switch[slot] = 0;
        req->choice_kind[slot] = REQUEST_SLOT_NONE;

        for (i = 0; i < PARSED_REQUEST_TEAM_SIZE; ++i) {
            if (req->switch_available[i] && !req->switch_fainted[i] && !req->switch_active[i]) {
                bench_switch_exists = 1;
                break;
            }
        }
        for (i = 0; i < PARSED_REQUEST_MOVE_SLOTS; ++i) {
            if (slot < req->active_count &&
                    req->active[slot].move_id[i] > 0 &&
                    !req->active[slot].move_disabled[i]) {
                move_exists = 1;
                break;
            }
        }

        if (req->team_preview) {
            if (slot == 0) {
                req->slot_present[slot] = 1;
                req->slot_needs_choice[slot] = 1;
                req->slot_can_switch[slot] = 1;
                req->choice_kind[slot] = REQUEST_SLOT_TEAM_PREVIEW;
            }
            continue;
        }

        if (!req->slot_present[slot]) {
            continue;
        }
        if (req->force_switch[slot]) {
            req->slot_needs_choice[slot] = 1;
            req->slot_can_switch[slot] = bench_switch_exists;
            req->choice_kind[slot] = REQUEST_SLOT_FORCE_SWITCH;
            continue;
        }
        if (req->forced_switch_any) {
            continue;
        }
        if (slot >= req->active_count || req->active[slot].fainted) {
            continue;
        }
        req->slot_needs_choice[slot] = 1;
        req->slot_can_move[slot] = move_exists;
        req->slot_can_switch[slot] = !req->active[slot].trapped && bench_switch_exists;
        if (req->slot_can_move[slot] || req->slot_can_switch[slot]) {
            req->choice_kind[slot] = REQUEST_SLOT_MOVE_OR_SWITCH;
        }
    }
}

static void sync_active_fainted_state_from_side(ParsedRequest* req) {
    int team_idx;
    int active_slot = 0;
    int active_found = 0;

    if (!req) {
        return;
    }
    for (team_idx = 0; team_idx < PARSED_REQUEST_TEAM_SIZE && active_slot < PARSED_REQUEST_ACTIVE_SLOTS; ++team_idx) {
        if (!req->switch_active[team_idx]) {
            continue;
        }
        if (active_found < PARSED_REQUEST_ACTIVE_SLOTS) {
            ++active_found;
        }
        req->active_team_idx[active_slot] = team_idx;
        req->active[active_slot].fainted = req->switch_fainted[team_idx] ? 1 : 0;
        ++active_slot;
    }
    if (active_found == req->active_count && active_found > 0) {
        int i;
        for (i = 0; i < active_found && i < PARSED_REQUEST_ACTIVE_SLOTS; ++i) {
            req->active_team_idx_known[i] = (req->active_team_idx[i] >= 0) ? 1u : 0u;
        }
        req->bootstrap_slot_binding_ambiguous = 0;
    } else if (active_found > 0 || req->active_count > 0) {
        req->bootstrap_slot_binding_ambiguous = 1;
    }
    for (; active_slot < PARSED_REQUEST_ACTIVE_SLOTS; ++active_slot) {
        req->active_team_idx[active_slot] = -1;
        req->active_team_idx_known[active_slot] = 0;
    }
    if (active_found != req->active_count && req->active_count > 0) {
        req->bootstrap_slot_binding_ambiguous = 1;
    }
}

int parse_request_payload(ParsedRequest* req, const char* json, int request_id, int is_doubles) {
    const char* active_key;
    const char* side_key;
    char active_array[PARSED_REQUEST_MAX_JSON];
    char side_obj[PARSED_REQUEST_MAX_JSON];
    const char* p;
    int slot = 0;

    if (!req || !json) {
        return 0;
    }

    parsed_request_init(req);
    req->request_id = request_id;
    req->is_doubles = is_doubles ? 1 : 0;
    strncpy(req->raw_json, json, sizeof(req->raw_json) - 1);
    req->wait = parse_bool_after(json, "wait", 0);
    req->team_preview = parse_bool_after(json, "teamPreview", 0);
    req->max_chosen_team_size = parse_int_after(json, "maxChosenTeamSize", req->is_doubles ? 2 : 1);

    {
        const char* force_switch_key = find_after_key(json, "forceSwitch");
        if (force_switch_key) {
            const char* force_arr = strchr(force_switch_key, '[');
            if (force_arr) {
                char force_array[128];
                const char* cursor;
                int idx = 0;
                if (extract_json_array(force_arr, force_array, sizeof(force_array))) {
                    cursor = force_array;
                    while (*cursor && idx < PARSED_REQUEST_ACTIVE_SLOTS) {
                        if (strncmp(cursor, "true", 4) == 0) {
                            req->force_switch[idx++] = 1;
                            req->forced_switch_any = 1;
                            cursor += 4;
                        } else if (strncmp(cursor, "false", 5) == 0) {
                            req->force_switch[idx++] = 0;
                            cursor += 5;
                        } else {
                            ++cursor;
                        }
                    }
                    if (idx > req->active_count) {
                        req->active_count = idx;
                    }
                }
            }
        }
    }

    active_key = find_after_key(json, "active");
    if (active_key) {
        p = strchr(active_key, ':');
        if (!p) {
            return 0;
        }
        p = skip_ws(p + 1);
        if (*p == '[') {
            if (!extract_json_array(p, active_array, sizeof(active_array))) {
                return 0;
            }
        } else {
            p = NULL;
        }
        if (p) {
            const char* cursor = next_top_level_object_in_array(active_array, NULL);
            while (cursor && slot < PARSED_REQUEST_ACTIVE_SLOTS) {
                char obj[4096];
                const char* moves_key;
                const char* move_cursor;
                int move_slot = 0;
                const char* next_cursor;

                if (!extract_json_object(cursor, obj, sizeof(obj))) {
                    return 0;
                }
                req->active[slot].can_tera = strstr(obj, "\"canTerastallize\"") != NULL;
                req->active[slot].tera_type_id = parse_string_id_after(obj, "canTerastallize", type_id_from_name);
                req->active[slot].trapped = parse_bool_after(obj, "trapped", 0);
                req->active[slot].maybe_trapped = parse_bool_after(obj, "maybeTrapped", 0);
                req->active[slot].fainted = parse_bool_after(obj, "fainted", 0);
                req->active[slot].has_force_switch = parse_bool_after(obj, "forceSwitch", 0);
                req->force_switch[slot] = req->active[slot].has_force_switch;
                if (req->active[slot].has_force_switch) {
                    req->forced_switch_any = 1;
                }
                if (req->active[slot].can_tera) {
                    req->can_tera = 1;
                }

                moves_key = find_after_key(obj, "moves");
                if (moves_key) {
                    move_cursor = strchr(moves_key, '{');
                    while (move_cursor && move_slot < PARSED_REQUEST_MOVE_SLOTS) {
                        char move_obj[1024];
                        if (!extract_json_object(move_cursor, move_obj, sizeof(move_obj))) {
                            return 0;
                        }
                        req->active[slot].move_id[move_slot] = parse_move_id_from_object(move_obj);
                        req->active[slot].move_disabled[move_slot] = parse_bool_after(move_obj, "disabled", 0);
                        req->active[slot].move_maybe_disabled[move_slot] = parse_bool_after(move_obj, "maybeDisabled", 0);
                        req->active[slot].move_pp[move_slot] = parse_int_after(move_obj, "pp", 0);
                        req->active[slot].move_max_pp[move_slot] = parse_int_after(move_obj, "maxpp", req->active[slot].move_pp[move_slot]);
                        req->active[slot].move_target[move_slot] = parse_move_target_value(move_obj);
                        move_cursor = strchr(move_cursor + 1, '{');
                        ++move_slot;
                    }
                }
                req->active_count = slot + 1;
                next_cursor = skip_json_object_text(cursor);
                cursor = next_top_level_object_in_array(active_array, next_cursor);
                ++slot;
            }
        }
    }

    side_key = find_after_key(json, "side");
    if (side_key) {
        p = strchr(side_key, '{');
        if (!p || !extract_json_object(p, side_obj, sizeof(side_obj))) {
            return 0;
        }
        {
            const char* pokemon_key = find_after_key(side_obj, "pokemon");
            int team_idx = 0;
            if (pokemon_key) {
                char pokemon_array[PARSED_REQUEST_MAX_JSON];
                const char* poke_cursor;
                const char* arr = strchr(pokemon_key, '[');
                char cond[64];
                char details[64];
                if (!arr || !extract_json_array(arr, pokemon_array, sizeof(pokemon_array))) {
                    return 0;
                }
                poke_cursor = next_top_level_object_in_array(pokemon_array, NULL);
                while (poke_cursor && team_idx < PARSED_REQUEST_TEAM_SIZE) {
                    char poke_obj[4096];
                    int is_active = 0;
                    int fainted = 0;
                    const char* next_poke_cursor;
                    if (!extract_json_object(poke_cursor, poke_obj, sizeof(poke_obj))) {
                        return 0;
                    }

                    req->switch_available[team_idx] = 1;
                    req->switch_fainted[team_idx] = 0;
                    req->switch_active[team_idx] = 0;
                    req->side_species_id[team_idx] = 0;
                    req->side_item_id[team_idx] = 0;
                    req->side_ability_id[team_idx] = 0;
                    req->side_base_ability_id[team_idx] = 0;
                    req->side_tera_type_id[team_idx] = 0;
                    req->side_tera_used[team_idx] = 0;
                    req->side_stats_hp[team_idx] = 0;
                    req->side_stats_atk[team_idx] = 0;
                    req->side_stats_def[team_idx] = 0;
                    req->side_stats_spa[team_idx] = 0;
                    req->side_stats_spd[team_idx] = 0;
                    req->side_stats_spe[team_idx] = 0;
                    req->side_commanding[team_idx] = 0;
                    req->side_reviving[team_idx] = 0;
                    req->side_ident[team_idx][0] = '\0';
                    memset(req->side_move_id[team_idx], 0, sizeof(req->side_move_id[team_idx]));

                    if (parse_json_string_field(poke_obj, "condition", cond, sizeof(cond))) {
                        const char* slash = strchr(cond, '/');
                        fainted = strstr(cond, "fnt") ? 1 : 0;
                        if (slash) {
                            req->side_stats_hp[team_idx] = atoi(slash + 1);
                        }
                    } else {
                        fainted = 0;
                    }
                    req->switch_available[team_idx] = fainted ? 0 : 1;
                    req->switch_fainted[team_idx] = fainted;

                    parse_json_string_field(poke_obj, "ident", req->side_ident[team_idx], sizeof(req->side_ident[team_idx]));
                    if (parse_json_string_field(poke_obj, "details", details, sizeof(details))) {
                        char* comma = strchr(details, ',');
                        if (comma) {
                            *comma = '\0';
                        }
                        req->side_species_id[team_idx] = species_id_from_name(details);
                    }
                    req->side_item_id[team_idx] = parse_string_id_after(poke_obj, "item", item_id_from_name);
                    req->side_ability_id[team_idx] = parse_string_id_after(poke_obj, "ability", ability_id_from_name);
                    req->side_base_ability_id[team_idx] = parse_string_id_after(poke_obj, "baseAbility", ability_id_from_name);
                    if (req->side_ability_id[team_idx] <= 0) {
                        req->side_ability_id[team_idx] = req->side_base_ability_id[team_idx];
                    }
                    req->side_tera_type_id[team_idx] = parse_string_id_after(poke_obj, "teraType", type_id_from_name);
                    if (parse_json_string_field(poke_obj, "terastallized", details, sizeof(details)) && details[0]) {
                        req->side_tera_used[team_idx] = 1;
                    }
                    req->side_stats_atk[team_idx] = parse_int_after(poke_obj, "atk", 0);
                    req->side_stats_def[team_idx] = parse_int_after(poke_obj, "def", 0);
                    req->side_stats_spa[team_idx] = parse_int_after(poke_obj, "spa", 0);
                    req->side_stats_spd[team_idx] = parse_int_after(poke_obj, "spd", 0);
                    req->side_stats_spe[team_idx] = parse_int_after(poke_obj, "spe", 0);
                    req->side_commanding[team_idx] = parse_bool_after(poke_obj, "commanding", 0);
                    req->side_reviving[team_idx] = parse_bool_after(poke_obj, "reviving", 0);
                    {
                        const char* moves_key = find_after_key(poke_obj, "moves");
                        const char* moves_arr = moves_key ? strchr(moves_key, '[') : NULL;
                        char moves_array[512];
                        if (moves_arr && extract_json_array(moves_arr, moves_array, sizeof(moves_array))) {
                            parse_move_id_array_from_string(moves_array, req->side_move_id[team_idx]);
                        }
                    }

                    is_active = parse_bool_after(poke_obj, "active", 0);
                    req->switch_active[team_idx] = is_active;
                    if (is_active) {
                        if (!fainted) {
                            req->living_active_count += 1;
                        }
                        req->switch_available[team_idx] = 0;
                    }

                    next_poke_cursor = skip_json_object_text(poke_cursor);
                    poke_cursor = next_top_level_object_in_array(pokemon_array, next_poke_cursor);
                    ++team_idx;
                }
            }
            parse_json_string_field(side_obj, "id", req->side_id, sizeof(req->side_id));
            if (strcmp(req->side_id, "p1") == 0) {
                req->side_player = 1;
            } else if (strcmp(req->side_id, "p2") == 0) {
                req->side_player = 2;
            }
        }
    }

    if (req->force_switch[1] && req->active_count < 2) {
        req->active_count = 2;
    } else if (req->force_switch[0] && req->active_count < 1) {
        req->active_count = 1;
    }

    sync_active_fainted_state_from_side(req);
    finalize_slot_semantics(req);

    return 1;
}

int parsed_request_slot_needs_choice(const ParsedRequest* req, int slot) {
    if (!req || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return 0;
    }
    return req->slot_needs_choice[slot];
}

int parsed_request_slot_can_move(const ParsedRequest* req, int slot) {
    if (!req || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return 0;
    }
    return req->slot_can_move[slot];
}

int parsed_request_slot_can_switch(const ParsedRequest* req, int slot) {
    if (!req || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return 0;
    }
    return req->slot_can_switch[slot];
}

ParsedSlotChoiceKind parsed_request_slot_choice_kind(const ParsedRequest* req, int slot) {
    if (!req || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return REQUEST_SLOT_NONE;
    }
    return req->choice_kind[slot];
}
