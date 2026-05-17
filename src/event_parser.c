#include "event_parser.h"

#include "id_tables.h"
#include "inference_engine.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* parts[16];
    int count;
} LineParts;

static void split_line(const char* line, LineParts* out) {
    const char* p = line;
    int idx = 0;
    memset(out, 0, sizeof(*out));
    if (!line || !*line) {
        return;
    }
    if (*p == '|') {
        ++p;
    }
    out->parts[idx++] = p;
    while (*p && idx < 16) {
        if (*p == '|') {
            *((char*)p) = '\0';
            out->parts[idx++] = p + 1;
        }
        ++p;
    }
    out->count = idx;
}

static RawPokemon* find_or_make(RawPokemon* team, const char* ident) {
    int i;
    int free_idx = 0;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].ident[0] && strcmp(team[i].ident, ident) == 0) {
            return &team[i];
        }
        if (!team[i].ident[0]) {
            free_idx = i;
            break;
        }
    }
    strncpy(team[free_idx].ident, ident, RAW_IDENT_LEN - 1);
    team[free_idx].known = 1;
    return &team[free_idx];
}

static RawPokemon* pokemon_for_ident(RawBattleState* state, const char* ident) {
    if (!state || !ident) {
        return NULL;
    }
    return strstr(ident, "p1") == ident ? find_or_make(state->self_team, ident) : find_or_make(state->opp_team, ident);
}

static RawSideState* side_for_ident(RawBattleState* state, const char* ident) {
    if (!state || !ident) {
        return NULL;
    }
    return strstr(ident, "p1") == ident ? &state->self_side : &state->opp_side;
}

static void parse_condition(const char* cond, int* hp, int* max_hp, int* status_id, int* fainted) {
    const char* slash;
    *hp = 0;
    *max_hp = 0;
    *status_id = 0;
    *fainted = 0;
    if (!cond) return;
    if (strstr(cond, "fnt")) *fainted = 1;
    slash = strchr(cond, '/');
    if (slash) {
        *hp = atoi(cond);
        *max_hp = atoi(slash + 1);
        if (strstr(cond, "brn")) *status_id = 1;
        else if (strstr(cond, "par")) *status_id = 2;
        else if (strstr(cond, "psn")) *status_id = 3;
        else if (strstr(cond, "tox")) *status_id = 4;
        else if (strstr(cond, "frz")) *status_id = 5;
        else if (strstr(cond, "slp")) *status_id = 6;
    }
}

static void handle_turn(RawBattleState* state, const char* value) {
    raw_battle_state_end_turn(state);
    raw_battle_state_begin_turn(state, atoi(value));
}

static void handle_weather(RawBattleState* state, const char* value) {
    if (strncmp(value, "Rain", 4) == 0) state->weather_id = 2;
    else if (strncmp(value, "Sun", 3) == 0 || strncmp(value, "Drought", 8) == 0) state->weather_id = 1;
    else if (strncmp(value, "Sandstorm", 9) == 0) state->weather_id = 3;
    else if (strncmp(value, "Snow", 4) == 0) state->weather_id = 4;
    else state->weather_id = 0;
    state->weather_turns_remaining = state->weather_id ? 5 : 0;
}

static void handle_field_start(RawBattleState* state, const char* value) {
    if (strncmp(value, "Electric Terrain", 16) == 0) {
        state->terrain_id = 1;
        state->terrain_turns_remaining = 5;
    } else if (strncmp(value, "Grassy Terrain", 14) == 0) {
        state->terrain_id = 2;
        state->terrain_turns_remaining = 5;
    } else if (strncmp(value, "Misty Terrain", 13) == 0) {
        state->terrain_id = 3;
        state->terrain_turns_remaining = 5;
    } else if (strncmp(value, "Psychic Terrain", 15) == 0) {
        state->terrain_id = 4;
        state->terrain_turns_remaining = 5;
    } else if (strncmp(value, "move: Trick Room", 16) == 0) {
        state->trick_room = 1;
        state->trick_room_turns_remaining = 5;
    } else if (strncmp(value, "move: Magic Room", 16) == 0) {
        state->magic_room = 1;
        state->magic_room_turns_remaining = 5;
    } else if (strncmp(value, "move: Wonder Room", 17) == 0) {
        state->wonder_room = 1;
        state->wonder_room_turns_remaining = 5;
    } else if (strncmp(value, "move: Gravity", 13) == 0) {
        state->gravity = 1;
        state->gravity_turns_remaining = 5;
    }
}

static void handle_field_end(RawBattleState* state, const char* value) {
    if (strncmp(value, "Electric Terrain", 16) == 0 ||
        strncmp(value, "Grassy Terrain", 14) == 0 ||
        strncmp(value, "Misty Terrain", 13) == 0 ||
        strncmp(value, "Psychic Terrain", 15) == 0) {
        state->terrain_id = 0;
        state->terrain_turns_remaining = 0;
    } else if (strncmp(value, "move: Trick Room", 16) == 0) {
        state->trick_room = 0;
        state->trick_room_turns_remaining = 0;
    } else if (strncmp(value, "move: Magic Room", 16) == 0) {
        state->magic_room = 0;
        state->magic_room_turns_remaining = 0;
    } else if (strncmp(value, "move: Wonder Room", 17) == 0) {
        state->wonder_room = 0;
        state->wonder_room_turns_remaining = 0;
    } else if (strncmp(value, "move: Gravity", 13) == 0) {
        state->gravity = 0;
        state->gravity_turns_remaining = 0;
    }
}

static void handle_side_effect(RawBattleState* state, const char* who, const char* effect, int start) {
    RawSideState* side = side_for_ident(state, who);
    if (!side) return;
    if (strncmp(effect, "Stealth Rock", 12) == 0) side->stealth_rock = start ? 1 : 0;
    else if (strncmp(effect, "Spikes", 6) == 0) side->spikes = start ? side->spikes + 1 : 0;
    else if (strncmp(effect, "Toxic Spikes", 12) == 0) side->toxic_spikes = start ? side->toxic_spikes + 1 : 0;
    else if (strncmp(effect, "Sticky Web", 10) == 0) side->sticky_web = start ? 1 : 0;
    else if (strncmp(effect, "Reflect", 7) == 0) { side->reflect = start ? 1 : 0; side->reflect_turns = start ? 5 : 0; }
    else if (strncmp(effect, "Light Screen", 12) == 0) { side->light_screen = start ? 1 : 0; side->light_screen_turns = start ? 5 : 0; }
    else if (strncmp(effect, "Aurora Veil", 11) == 0) { side->aurora_veil = start ? 1 : 0; side->aurora_veil_turns = start ? 5 : 0; }
    else if (strncmp(effect, "Tailwind", 8) == 0) { side->tailwind = start ? 1 : 0; side->tailwind_turns = start ? 4 : 0; }
    else if (strncmp(effect, "Safeguard", 9) == 0) { side->safeguard = start ? 1 : 0; side->safeguard_turns = start ? 5 : 0; }
    else if (strncmp(effect, "Mist", 4) == 0) { side->mist = start ? 1 : 0; side->mist_turns = start ? 5 : 0; }
    else if (strncmp(effect, "Lucky Chant", 11) == 0) { side->lucky_chant = start ? 1 : 0; side->lucky_chant_turns = start ? 5 : 0; }
}

static void handle_switch_like(RawBattleState* state, const char* ident, const char* details, const char* cond) {
    RawPokemon* pokemon = pokemon_for_ident(state, ident);
    int hp, max_hp, status_id, fainted;
    char species[64];
    size_t i = 0;
    if (!pokemon) return;
    while (details[i] && details[i] != ',' && i + 1 < sizeof(species)) {
        species[i] = details[i];
        ++i;
    }
    species[i] = '\0';
    tracked_int_promote_confirmed(&pokemon->species_id, species_id_from_name(species));
    parse_condition(cond, &hp, &max_hp, &status_id, &fainted);
    pokemon->current_hp = hp;
    pokemon->max_hp = max_hp;
    tracked_int_promote_confirmed(&pokemon->status_id, status_id);
    pokemon->fainted = fainted;
    pokemon->active = 1;
    pokemon->revealed = 1;
    pokemon->switched_in_turn = state->turn_number;
    pokemon->first_turn_on_field = 1;
    pokemon->active_slot = strstr(ident, "b:") ? 2 : 1;
}

static void handle_status_line(RawBattleState* state, const char* ident, const char* value) {
    RawPokemon* pokemon = pokemon_for_ident(state, ident);
    if (!pokemon) return;
    if (strncmp(value, "brn", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 1);
    else if (strncmp(value, "par", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 2);
    else if (strncmp(value, "psn", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 3);
    else if (strncmp(value, "tox", 3) == 0) { tracked_int_promote_confirmed(&pokemon->status_id, 4); pokemon->toxic_counter = 1; }
    else if (strncmp(value, "frz", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 5);
    else if (strncmp(value, "slp", 3) == 0) { tracked_int_promote_confirmed(&pokemon->status_id, 6); pokemon->sleep_turns = 3; }
}

static int boost_index(const char* stat) {
    if (strcmp(stat, "atk") == 0) return 0;
    if (strcmp(stat, "def") == 0) return 1;
    if (strcmp(stat, "spa") == 0) return 2;
    if (strcmp(stat, "spd") == 0) return 3;
    if (strcmp(stat, "spe") == 0) return 4;
    if (strcmp(stat, "accuracy") == 0) return 5;
    if (strcmp(stat, "evasion") == 0) return 6;
    return -1;
}

static void handle_boost(RawBattleState* state, const char* ident, const char* stat, int amount, int mode) {
    RawPokemon* pokemon = pokemon_for_ident(state, ident);
    int idx = boost_index(stat);
    if (!pokemon || idx < 0) return;
    if (mode == 0) pokemon->boosts[idx] += amount;
    else if (mode == 1) pokemon->boosts[idx] -= amount;
    else if (mode == 2) pokemon->boosts[idx] = amount;
}

static void handle_start(RawBattleState* state, const char* ident, const char* effect) {
    RawPokemon* p = pokemon_for_ident(state, ident);
    if (!p) return;
    if (strncmp(effect, "Encore", 6) == 0) { p->encore_active = 1; p->encore_turns = 3; }
    else if (strncmp(effect, "Disable", 7) == 0) { p->disable_active = 1; p->disable_turns = 4; }
    else if (strncmp(effect, "Taunt", 5) == 0) { p->taunt_active = 1; p->taunt_turns = 3; }
    else if (strncmp(effect, "Torment", 7) == 0) { p->torment_active = 1; p->torment_turns = 3; }
    else if (strncmp(effect, "Heal Block", 10) == 0) { p->heal_block_active = 1; p->heal_block_turns = 5; }
    else if (strncmp(effect, "Embargo", 7) == 0) { p->embargo_active = 1; p->embargo_turns = 5; }
    else if (strncmp(effect, "Yawn", 4) == 0) { p->yawn_active = 1; p->yawn_turns = 2; }
    else if (strncmp(effect, "confusion", 9) == 0) { p->confusion_active = 1; p->confusion_turns = 3; }
    else if (strncmp(effect, "Substitute", 10) == 0) p->substitute_active = 1;
    else if (strncmp(effect, "Leech Seed", 10) == 0) p->seed_active = 1;
    else if (strncmp(effect, "Perish Song", 11) == 0) p->perish_song_counter = 4;
    else if (strncmp(effect, "Charge", 6) == 0) { p->charge_active = 1; p->charge_turns = 2; }
}

static void handle_end(RawBattleState* state, const char* ident, const char* effect) {
    RawPokemon* p = pokemon_for_ident(state, ident);
    if (!p) return;
    if (strncmp(effect, "Encore", 6) == 0) { p->encore_active = 0; p->encore_turns = 0; }
    else if (strncmp(effect, "Disable", 7) == 0) { p->disable_active = 0; p->disable_turns = 0; }
    else if (strncmp(effect, "Taunt", 5) == 0) { p->taunt_active = 0; p->taunt_turns = 0; }
    else if (strncmp(effect, "Torment", 7) == 0) { p->torment_active = 0; p->torment_turns = 0; }
    else if (strncmp(effect, "Heal Block", 10) == 0) { p->heal_block_active = 0; p->heal_block_turns = 0; }
    else if (strncmp(effect, "Embargo", 7) == 0) { p->embargo_active = 0; p->embargo_turns = 0; }
    else if (strncmp(effect, "Yawn", 4) == 0) { p->yawn_active = 0; p->yawn_turns = 0; }
    else if (strncmp(effect, "confusion", 9) == 0) { p->confusion_active = 0; p->confusion_turns = 0; }
    else if (strncmp(effect, "Substitute", 10) == 0) p->substitute_active = 0;
    else if (strncmp(effect, "Leech Seed", 10) == 0) p->seed_active = 0;
    else if (strncmp(effect, "Charge", 6) == 0) { p->charge_active = 0; p->charge_turns = 0; }
}

void event_parser_apply_line(RawBattleState* state, const char* line) {
    char buffer[2048];
    LineParts parts;
    if (!state || !line || !*line) return;
    strncpy(buffer, line, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    split_line(buffer, &parts);
    if (parts.count <= 0 || !parts.parts[0]) return;

    if (strcmp(parts.parts[0], "turn") == 0 && parts.count > 1) handle_turn(state, parts.parts[1]);
    else if (strcmp(parts.parts[0], "-weather") == 0 && parts.count > 1) handle_weather(state, parts.parts[1]);
    else if (strcmp(parts.parts[0], "-fieldstart") == 0 && parts.count > 1) handle_field_start(state, parts.parts[1]);
    else if (strcmp(parts.parts[0], "-fieldend") == 0 && parts.count > 1) handle_field_end(state, parts.parts[1]);
    else if (strcmp(parts.parts[0], "-sidestart") == 0 && parts.count > 2) handle_side_effect(state, parts.parts[1], parts.parts[2], 1);
    else if (strcmp(parts.parts[0], "-sideend") == 0 && parts.count > 2) handle_side_effect(state, parts.parts[1], parts.parts[2], 0);
    else if ((strcmp(parts.parts[0], "switch") == 0 || strcmp(parts.parts[0], "drag") == 0 || strcmp(parts.parts[0], "replace") == 0) && parts.count > 3) handle_switch_like(state, parts.parts[1], parts.parts[2], parts.parts[3]);
    else if (strcmp(parts.parts[0], "-status") == 0 && parts.count > 2) handle_status_line(state, parts.parts[1], parts.parts[2]);
    else if (strcmp(parts.parts[0], "-curestatus") == 0 && parts.count > 1) tracked_int_set_confirmed(&pokemon_for_ident(state, parts.parts[1])->status_id, 0);
    else if (strcmp(parts.parts[0], "-boost") == 0 && parts.count > 3) handle_boost(state, parts.parts[1], parts.parts[2], atoi(parts.parts[3]), 0);
    else if (strcmp(parts.parts[0], "-unboost") == 0 && parts.count > 3) handle_boost(state, parts.parts[1], parts.parts[2], atoi(parts.parts[3]), 1);
    else if (strcmp(parts.parts[0], "-setboost") == 0 && parts.count > 3) handle_boost(state, parts.parts[1], parts.parts[2], atoi(parts.parts[3]), 2);
    else if (strcmp(parts.parts[0], "-clearallboost") == 0) { int i,j; for (i = 0; i < RAW_TEAM_SIZE; ++i) for (j = 0; j < 7; ++j) { state->self_team[i].boosts[j] = 0; state->opp_team[i].boosts[j] = 0; } }
    else if (strcmp(parts.parts[0], "-clearboost") == 0 && parts.count > 1) { int j; RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) for (j = 0; j < 7; ++j) p->boosts[j] = 0; }
    else if (strcmp(parts.parts[0], "-start") == 0 && parts.count > 2) handle_start(state, parts.parts[1], parts.parts[2]);
    else if (strcmp(parts.parts[0], "-end") == 0 && parts.count > 2) handle_end(state, parts.parts[1], parts.parts[2]);
    else if (strcmp(parts.parts[0], "-singleturn") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]);
        if (p && (strcmp(parts.parts[2], "Protect") == 0 || strcmp(parts.parts[2], "Detect") == 0 || strcmp(parts.parts[2], "Endure") == 0)) {
            p->protect_active = 1;
            p->protect_chain_count += 1;
        }
    }
    else if (strcmp(parts.parts[0], "-singlemove") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]);
        if (p && strcmp(parts.parts[2], "Helping Hand") == 0) p->helping_hand_active = 1;
    }
    else if (strcmp(parts.parts[0], "-damage") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); int hp,max_hp,status,fnt; if (p) { parse_condition(parts.parts[2], &hp,&max_hp,&status,&fnt); p->current_hp=hp; p->max_hp=max_hp; if (status) tracked_int_promote_confirmed(&p->status_id,status); p->fainted=fnt; }
    }
    else if (strcmp(parts.parts[0], "-heal") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); int hp,max_hp,status,fnt; if (p) { parse_condition(parts.parts[2], &hp,&max_hp,&status,&fnt); p->current_hp=hp; p->max_hp=max_hp; if (status) tracked_int_promote_confirmed(&p->status_id,status); p->fainted=fnt; }
    }
    else if (strcmp(parts.parts[0], "faint") == 0 && parts.count > 1) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) p->fainted = 1;
    }
    else if (strcmp(parts.parts[0], "-ability") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) tracked_int_set_confirmed(&p->ability_id, ability_id_from_name(parts.parts[2]));
    }
    else if (strcmp(parts.parts[0], "-item") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) tracked_int_set_confirmed(&p->item_id, item_id_from_name(parts.parts[2]));
    }
    else if (strcmp(parts.parts[0], "-enditem") == 0 && parts.count > 1) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) tracked_int_set_confirmed(&p->item_id, 0);
    }
    else if (strcmp(parts.parts[0], "-terastallize") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) { p->tera_used = 1; tracked_int_set_confirmed(&p->tera_type_id, atoi(parts.parts[2])); }
    }
    else if (strcmp(parts.parts[0], "move") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident(state, parts.parts[1]); if (p) inference_engine_note_move(p, move_id_from_name(parts.parts[2]), state->turn_number);
    }
}
