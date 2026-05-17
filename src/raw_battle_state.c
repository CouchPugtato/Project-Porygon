#include "raw_battle_state.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static int status_from_text(const char* text) {
    if (!text || !*text) return 0;
    if (strncmp(text, "brn", 3) == 0) return 1;
    if (strncmp(text, "par", 3) == 0) return 2;
    if (strncmp(text, "psn", 3) == 0) return 3;
    if (strncmp(text, "tox", 3) == 0) return 4;
    if (strncmp(text, "frz", 3) == 0) return 5;
    if (strncmp(text, "slp", 3) == 0) return 6;
    return 0;
}

static void parse_condition(const char* cond, int* hp, int* max_hp, int* status_id, int* fainted) {
    const char* slash;
    *hp = 0;
    *max_hp = 0;
    *status_id = 0;
    *fainted = 0;
    if (!cond) {
        return;
    }
    if (strstr(cond, "fnt")) {
        *fainted = 1;
    }
    slash = strchr(cond, '/');
    if (slash) {
        *hp = atoi(cond);
        *max_hp = atoi(slash + 1);
        while (*slash && !isspace((unsigned char)*slash)) ++slash;
        while (*slash && isspace((unsigned char)*slash)) ++slash;
        *status_id = status_from_text(slash);
    }
}

static void clear_active_flags(RawPokemon* team) {
    int i;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        team[i].active = 0;
        team[i].active_slot = 0;
    }
}

static RawPokemon* ensure_team_slot(RawPokemon* team, int slot) {
    if (slot < 0 || slot >= RAW_TEAM_SIZE) {
        return NULL;
    }
    team[slot].known = 1;
    return &team[slot];
}

static RawPokemon* find_or_allocate_by_ident(RawPokemon* team, const char* ident) {
    int i;
    int free_slot = -1;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].ident[0] && strcmp(team[i].ident, ident) == 0) {
            return &team[i];
        }
        if (free_slot < 0 && !team[i].ident[0]) {
            free_slot = i;
        }
    }
    if (free_slot >= 0) {
        strncpy(team[free_slot].ident, ident, RAW_IDENT_LEN - 1);
        team[free_slot].known = 1;
        return &team[free_slot];
    }
    return &team[0];
}

void raw_battle_state_init(RawBattleState* state, int is_doubles) {
    if (!state) {
        return;
    }
    memset(state, 0, sizeof(*state));
    state->is_doubles = is_doubles ? 1 : 0;
    state->self_side.remaining_pokemon = RAW_TEAM_SIZE;
    state->opp_side.remaining_pokemon = RAW_TEAM_SIZE;
}

void raw_battle_state_reset(RawBattleState* state) {
    int is_doubles = state ? state->is_doubles : 0;
    raw_battle_state_init(state, is_doubles);
}

void raw_battle_state_update_from_request(RawBattleState* state, const ParsedRequest* req) {
    int i;
    if (!state || !req) {
        return;
    }

    state->turn_number = req->request_id > 0 ? req->request_id : state->turn_number;
    state->can_tera = req->can_tera;
    clear_active_flags(state->self_team);

    for (i = 0; i < req->active_count && i < RAW_TEAM_SIZE; ++i) {
        RawPokemon* pokemon = ensure_team_slot(state->self_team, i);
        int m;
        if (!pokemon) {
            continue;
        }
        pokemon->active = 1;
        pokemon->active_slot = i + 1;
        pokemon->revealed = 1;
        pokemon->can_tera = req->active[i].can_tera;
        pokemon->fainted = req->active[i].fainted;
        for (m = 0; m < RAW_MOVE_SLOTS; ++m) {
            if (req->active[i].move_id[m] > 0) {
                pokemon->move_ids[m] = req->active[i].move_id[m];
                pokemon->move_known[m] = 1;
                pokemon->move_pp[m] = req->active[i].move_pp[m];
                pokemon->move_max_pp[m] = req->active[i].move_max_pp[m];
                pokemon->move_disabled[m] = req->active[i].move_disabled[m];
            }
        }
    }

    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        state->self_team[i].fainted = req->switch_fainted[i];
    }
}

void raw_battle_state_update_from_event_line(RawBattleState* state, const char* line) {
    if (!state || !line || !*line) {
        return;
    }
    if (strncmp(line, "|turn|", 6) == 0) {
        state->turn_number = atoi(line + 6);
        return;
    }
    if (strncmp(line, "|-weather|", 10) == 0) {
        const char* w = line + 10;
        if (strncmp(w, "Rain", 4) == 0) state->weather_id = 2;
        else if (strncmp(w, "Sun", 3) == 0 || strncmp(w, "Drought", 8) == 0) state->weather_id = 1;
        else if (strncmp(w, "Sandstorm", 9) == 0) state->weather_id = 3;
        else if (strncmp(w, "Snow", 4) == 0) state->weather_id = 4;
        else state->weather_id = 0;
        return;
    }
    if (strncmp(line, "|-fieldstart|", 13) == 0) {
        const char* t = line + 13;
        if (strncmp(t, "Electric Terrain", 16) == 0) state->terrain_id = 1;
        else if (strncmp(t, "Grassy Terrain", 14) == 0) state->terrain_id = 2;
        else if (strncmp(t, "Misty Terrain", 13) == 0) state->terrain_id = 3;
        else if (strncmp(t, "Psychic Terrain", 15) == 0) state->terrain_id = 4;
        else if (strncmp(t, "move: Trick Room", 16) == 0) state->trick_room = 1;
        return;
    }
    if (strncmp(line, "|-fieldend|move: Trick Room", 27) == 0) {
        state->trick_room = 0;
        return;
    }
    if (strncmp(line, "|-sidestart|", 12) == 0 || strncmp(line, "|-sideend|", 10) == 0) {
        int starting = line[6] == 's';
        const char* p = strchr(line + (starting ? 12 : 10), '|');
        int is_self = strstr(line, "p1|") != NULL;
        RawSideState* side = is_self ? &state->self_side : &state->opp_side;
        if (!p) {
            return;
        }
        ++p;
        if (strncmp(p, "Stealth Rock", 12) == 0) side->stealth_rock = starting ? 1 : 0;
        else if (strncmp(p, "Spikes", 6) == 0) side->spikes = starting ? side->spikes + 1 : 0;
        else if (strncmp(p, "Toxic Spikes", 12) == 0) side->toxic_spikes = starting ? side->toxic_spikes + 1 : 0;
        else if (strncmp(p, "Sticky Web", 10) == 0) side->sticky_web = starting ? 1 : 0;
        else if (strncmp(p, "Reflect", 7) == 0) side->reflect = starting ? 1 : 0;
        else if (strncmp(p, "Light Screen", 12) == 0) side->light_screen = starting ? 1 : 0;
        else if (strncmp(p, "Aurora Veil", 11) == 0) side->aurora_veil = starting ? 1 : 0;
        else if (strncmp(p, "Tailwind", 8) == 0) side->tailwind = starting ? 1 : 0;
        return;
    }
}
