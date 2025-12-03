#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include "game_state.h"


static int parse_hp_condition(const char* cond, int* cur, int* max, int* status_enum) {
    // condition format examples: "173/300 tox", "0 fnt", "270/270"
    if (!cond) return 0;
    const char* slash = strchr(cond, '/');
    if (!slash) return 0;
    int a = 0, b = 0;
    a = atoi(cond);
    b = atoi(slash + 1);
    *cur = a;
    *max = b;
    *status_enum = -1;
    const char* tail = slash;
    while (*tail && !isspace((unsigned char)*tail)) tail++;
    while (*tail && isspace((unsigned char)*tail)) tail++;
    if (*tail) {
        if (strncmp(tail, "brn", 3) == 0) *status_enum = 0;        // BURN
        else if (strncmp(tail, "par", 3) == 0) *status_enum = 1;   // PARALYZE
        else if (strncmp(tail, "psn", 3) == 0) *status_enum = 2;   // POISONED
        else if (strncmp(tail, "tox", 3) == 0) *status_enum = 3;   // TOXIC
        else if (strncmp(tail, "frz", 3) == 0) *status_enum = 4;   // FREEZE
        else if (strncmp(tail, "slp", 3) == 0) *status_enum = 5;   // SLEEP
    }
    return 1;
}

static const char* find_key_str(const char* json, const char* key) {
    // naive search for a key occurrence, returns pointer after key":
    if (!json || !key) return NULL;
    char pattern[64];
    snprintf(pattern, sizeof pattern, "\"%s\":", key);
    return strstr(json, pattern);
}

static int extract_string_value(const char* start, char* out, size_t outsz) {
    // expects start to point to '"' beginning a string value
    const char* p = start;
    while (p && *p && *p != '"') p++;
    if (!p || *p != '"') return 0;
    p++; // after opening quote
    size_t i = 0;
    while (*p && *p != '"' && i + 1 < outsz) { out[i++] = *p++; }
    out[i] = '\0';
    return (*p == '"');
}

void battle_state_init(struct BattleState* bs) {
    if (!bs) return;
    memset(bs, 0, sizeof *bs);
    bs->weather = 0; // NONE
    bs->terrain = 0; // NORMAL_TERRAIN
}

void battle_state_update_from_request(struct BattleState* bs, const char* request_json) {
    if (!bs || !request_json) return;
    // Reset per-turn flags relevant to request
    // Parse side.pokemon array and fill friendly team HP/status
    const char* side = strstr(request_json, "\"side\":");
    if (!side) return;
    const char* pokes = strstr(side, "\"pokemon\":");
    if (!pokes) return;
    const char* iter = pokes;
    int idx = 0;
    while (idx < 6) {
        // find next "condition":"..."
        const char* condKey = strstr(iter, "\"condition\":");
        if (!condKey) break;
        const char* q = strchr(condKey, '"');
        if (!q) break;
        q = strchr(q + 1, '"'); // move past "condition"
        if (!q) break;
        // next " should start the value
        const char* valStart = strchr(q + 1, '"');
        if (!valStart) break;
        char condBuf[64];
        if (!extract_string_value(valStart, condBuf, sizeof condBuf)) break;
        int cur = 0, max = 0, status = -1;
        if (parse_hp_condition(condBuf, &cur, &max, &status)) {
            bs->friendly_pokemon[idx].current_hp = cur;
            bs->friendly_pokemon[idx].max_hp = max;
            if (status >= 0) bs->friendly_pokemon[idx].status_condition = status;
        }
        // mark on_field unknown here; will be set by stream events
        bs->friendly_pokemon[idx].on_field = 0;
        // advance
        iter = valStart + strlen(condBuf) + 2; // crude advance
        idx++;
    }
}

void battle_state_update_from_line(struct BattleState* bs, const char* line) {
    if (!bs || !line || !*line) return;
    if (strncmp(line, "|-weather|", 10) == 0) {
        const char* w = line + 10;
        if (strncmp(w, "None", 4) == 0 || strncmp(w, "none", 4) == 0) {
            bs->weather = 0; // NONE
            bs->min_turns_of_weather = 0;
        } else if (strncmp(w, "Sun", 3) == 0 || strncmp(w, "Drought", 8) == 0) {
            bs->weather = 1; // DROUGHT
        } else if (strncmp(w, "Rain", 4) == 0) {
            bs->weather = 2; // RAIN
        } else if (strncmp(w, "Sandstorm", 9) == 0) {
            bs->weather = 3; // SANDSTORM
        } else if (strncmp(w, "Snow", 4) == 0) {
            bs->weather = 4; // SNOW
        } else if (strncmp(w, "Strong Winds", 12) == 0) {
            bs->weather = 5; // STRONG_WINDS
        }
        return;
    }
    if (strncmp(line, "|-fieldstart|", 13) == 0) {
        const char* t = line + 13;
        if (strncmp(t, "Electric Terrain", 16) == 0) bs->terrain = 1;
        else if (strncmp(t, "Grassy Terrain", 14) == 0) bs->terrain = 2;
        else if (strncmp(t, "Misty Terrain", 13) == 0) bs->terrain = 3;
        else if (strncmp(t, "Psychic Terrain", 15) == 0) bs->terrain = 4;
        bs->min_turns_of_terrain = 5; // default duration
        return;
    }
    if (strncmp(line, "|-fieldend|", 11) == 0) {
        bs->terrain = 0;
        bs->min_turns_of_terrain = 0;
        return;
    }
    if (strncmp(line, "|-sidestart|", 12) == 0) {
        const char* p = line + 12;
        int is_p1 = (strncmp(p, "p1|", 3) == 0);
        const char* eff = strstr(p, "|");
        if (!eff) return;
        eff++;
        if (strncmp(eff, "Stealth Rock", 12) == 0) {
            if (is_p1) bs->friendly_hazards.stealth_rock = 1;
            else bs->opponent_hazards.stealth_rock = 1;
        } else if (strncmp(eff, "Spikes", 6) == 0) {
            if (is_p1) bs->friendly_hazards.spikes++;
            else bs->opponent_hazards.spikes++;
        } else if (strncmp(eff, "Toxic Spikes", 12) == 0) {
            if (is_p1) bs->friendly_hazards.toxic_spikes++;
            else bs->opponent_hazards.toxic_spikes++;
        } else if (strncmp(eff, "Sticky Web", 10) == 0) {
            if (is_p1) bs->friendly_hazards.sticky_web = 1;
            else bs->opponent_hazards.sticky_web = 1;
        }
        return;
    }
    if (strncmp(line, "|-sideend|", 10) == 0) {
        const char* p = line + 10;
        int is_p1 = (strncmp(p, "p1|", 3) == 0);
        const char* eff = strstr(p, "|");
        if (!eff) return;
        eff++;
        if (strncmp(eff, "Stealth Rock", 12) == 0) {
            if (is_p1) bs->friendly_hazards.stealth_rock = 0;
            else bs->opponent_hazards.stealth_rock = 0;
        } else if (strncmp(eff, "Spikes", 6) == 0) {
            if (is_p1) bs->friendly_hazards.spikes = 0;
            else bs->opponent_hazards.spikes = 0;
        } else if (strncmp(eff, "Toxic Spikes", 12) == 0) {
            if (is_p1) bs->friendly_hazards.toxic_spikes = 0;
            else bs->opponent_hazards.toxic_spikes = 0;
        } else if (strncmp(eff, "Sticky Web", 10) == 0) {
            if (is_p1) bs->friendly_hazards.sticky_web = 0;
            else bs->opponent_hazards.sticky_web = 0;
        }
        return;
    }
}