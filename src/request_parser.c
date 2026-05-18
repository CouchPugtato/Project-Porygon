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
    if (!req) {
        return;
    }
    memset(req, 0, sizeof(*req));
    req->max_chosen_team_size = 2;
}

int parse_request_payload(ParsedRequest* req, const char* json, int request_id, int is_doubles) {
    const char* active_key;
    const char* side_key;
    char active_array[4096];
    char side_obj[4096];
    const char* p;
    int slot = 0;

    if (!req || !json) {
        return 0;
    }

    parsed_request_init(req);
    req->request_id = request_id;
    req->is_doubles = is_doubles ? 1 : 0;
    strncpy(req->raw_json, json, sizeof(req->raw_json) - 1);
    req->team_preview = parse_bool_after(json, "teamPreview", 0);
    req->max_chosen_team_size = parse_int_after(json, "maxChosenTeamSize", req->is_doubles ? 2 : 1);

    active_key = find_after_key(json, "active");
    if (active_key) {
        p = strchr(active_key, '[');
        if (p && extract_json_array(p, active_array, sizeof(active_array))) {
            const char* cursor = active_array;
            while ((cursor = strchr(cursor, '{')) != NULL && slot < PARSED_REQUEST_ACTIVE_SLOTS) {
                char obj[1024];
                const char* moves_key;
                const char* move_cursor;
                int move_slot = 0;

                if (!extract_json_object(cursor, obj, sizeof(obj))) {
                    break;
                }
                req->active[slot].can_tera = strstr(obj, "\"canTerastallize\"") != NULL;
                req->active[slot].tera_type_id = 0;
                req->active[slot].trapped = parse_bool_after(obj, "trapped", 0);
                req->active[slot].maybe_trapped = parse_bool_after(obj, "maybeTrapped", 0);
                req->active[slot].fainted = parse_bool_after(obj, "fainted", 0);
                req->active[slot].has_force_switch = parse_bool_after(obj, "forceSwitch", 0);
                req->force_switch[slot] = req->active[slot].has_force_switch;
                if (req->active[slot].has_force_switch) {
                    req->forced_switch_any = 1;
                }

                moves_key = find_after_key(obj, "moves");
                if (moves_key) {
                    move_cursor = strchr(moves_key, '{');
                    while (move_cursor && move_slot < PARSED_REQUEST_MOVE_SLOTS) {
                        char move_obj[512];
                        if (!extract_json_object(move_cursor, move_obj, sizeof(move_obj))) {
                            break;
                        }
                        req->active[slot].move_id[move_slot] = parse_move_id_from_object(move_obj);
                        req->active[slot].move_disabled[move_slot] = parse_bool_after(move_obj, "disabled", 0);
                        req->active[slot].move_maybe_disabled[move_slot] = parse_bool_after(move_obj, "maybeDisabled", 0);
                        req->active[slot].move_pp[move_slot] = parse_int_after(move_obj, "pp", 0);
                        req->active[slot].move_max_pp[move_slot] = parse_int_after(move_obj, "maxpp", req->active[slot].move_pp[move_slot]);
                        req->active[slot].move_target[move_slot] = parse_move_target_value(move_obj);
                        if (req->active[slot].can_tera && !req->active[slot].move_disabled[move_slot]) {
                            req->can_tera = 1;
                        }
                        move_cursor = strchr(move_cursor + 1, '{');
                        ++move_slot;
                    }
                }
                req->active_count = slot + 1;
                cursor = strchr(cursor + 1, '{');
                ++slot;
            }
        }
    }

    side_key = find_after_key(json, "side");
    if (side_key) {
        p = strchr(side_key, '{');
        if (p && extract_json_object(p, side_obj, sizeof(side_obj))) {
            const char* pokemon_key = find_after_key(side_obj, "pokemon");
            int team_idx = 0;
            if (pokemon_key) {
                char pokemon_array[4096];
                const char* poke_cursor;
                const char* arr = strchr(pokemon_key, '[');
                char cond[64];
                char details[64];
                if (!arr || !extract_json_array(arr, pokemon_array, sizeof(pokemon_array))) {
                    return 1;
                }
                poke_cursor = pokemon_array;
                while ((poke_cursor = strchr(poke_cursor, '{')) != NULL && team_idx < PARSED_REQUEST_TEAM_SIZE) {
                    char poke_obj[1024];
                    int is_active = 0;
                    int fainted = 0;
                    if (!extract_json_object(poke_cursor, poke_obj, sizeof(poke_obj))) {
                        break;
                    }

                    req->switch_available[team_idx] = 1;
                    req->switch_fainted[team_idx] = 0;
                    req->switch_active[team_idx] = 0;
                    req->side_species_id[team_idx] = 0;
                    req->side_ident[team_idx][0] = '\0';

                    if (parse_json_string_field(poke_obj, "condition", cond, sizeof(cond)) && strstr(cond, "fnt")) {
                        fainted = 1;
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

                    is_active = parse_bool_after(poke_obj, "active", 0);
                    req->switch_active[team_idx] = is_active;
                    if (is_active) {
                        req->switch_available[team_idx] = 0;
                    }

                    poke_cursor = strchr(poke_cursor + 1, '{');
                    ++team_idx;
                }
            }
        }
    }

    return 1;
}
