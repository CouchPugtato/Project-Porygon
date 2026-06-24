#include "event_parser.h"

#include "id_tables.h"
#include "inference_engine.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* parts[16];
    int count;
} LineParts;

static int ident_side_player(const char* ident) {
    if (!ident) {
        return 0;
    }
    if (strncmp(ident, "p1", 2) == 0) {
        return 1;
    }
    if (strncmp(ident, "p2", 2) == 0) {
        return 2;
    }
    return 0;
}

static int ident_is_self_side(const RawBattleState* state, const char* ident) {
    int side_player;
    if (!state || !ident) {
        return 0;
    }
    side_player = ident_side_player(ident);
    if (side_player == 0) {
        return 0;
    }
    return side_player == state->self_side_player;
}

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
    int free_idx = -1;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].ident[0] && strcmp(team[i].ident, ident) == 0) {
            return &team[i];
        }
        if (free_idx < 0 && !team[i].ident[0]) {
            free_idx = i;
        }
    }
    if (free_idx < 0) {
        return NULL;
    }
    strncpy(team[free_idx].ident, ident, RAW_IDENT_LEN - 1);
    team[free_idx].known = 1;
    return &team[free_idx];
}

static int same_identity(const char* a, const char* b);

static RawPokemon* find_by_canonical_or_exact_ident(RawPokemon* team, const char* ident) {
    int i;
    if (!team || !ident) {
        return NULL;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if ((team[i].canonical_ident[0] && strcmp(team[i].canonical_ident, ident) == 0) ||
                (team[i].ident[0] && strcmp(team[i].ident, ident) == 0)) {
            return &team[i];
        }
    }
    return NULL;
}

static RawPokemon* allocate_empty_identity_slot(RawPokemon* team, const char* ident) {
    int i;
    if (!team || !ident) {
        return NULL;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (!team[i].ident[0] && !team[i].canonical_ident[0]) {
            strncpy(team[i].ident, ident, RAW_IDENT_LEN - 1);
            team[i].ident[RAW_IDENT_LEN - 1] = '\0';
            team[i].known = 1;
            return &team[i];
        }
    }
    return NULL;
}

static RawPokemon* find_same_identity_candidate(RawPokemon* team, const char* ident, int species_id, int require_inactive) {
    int i;
    RawPokemon* match = NULL;
    if (!team || !ident) {
        return NULL;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (!team[i].ident[0]) {
            continue;
        }
        if (!same_identity(team[i].ident, ident)) {
            continue;
        }
        if (require_inactive && team[i].active) {
            continue;
        }
        if (species_id > 0 && team[i].species_id.value > 0 && team[i].species_id.value != species_id) {
            continue;
        }
        if (match) {
            return NULL;
        }
        match = &team[i];
    }
    return match;
}

static const char* ident_name_part(const char* ident) {
    const char* sep;
    if (!ident) {
        return "";
    }
    sep = strstr(ident, ": ");
    return sep ? (sep + 2) : ident;
}

static int same_identity(const char* a, const char* b) {
    if (!a || !b || !*a || !*b) {
        return 0;
    }
    if (a[0] != b[0] || a[1] != b[1]) {
        return 0;
    }
    return strcmp(ident_name_part(a), ident_name_part(b)) == 0;
}

static int ident_to_active_slot(const char* ident) {
    if (!ident) {
        return 0;
    }
    if (strstr(ident, "a:")) {
        return 1;
    }
    if (strstr(ident, "b:")) {
        return 2;
    }
    return 0;
}

static RawPokemon* active_slot_pokemon_for_ident(RawBattleState* state, const char* ident) {
    RawPokemon* team;
    int* slot_map;
    int slot;
    int team_index;
    if (!state || !ident) {
        return NULL;
    }
    slot = ident_to_active_slot(ident);
    if (slot <= 0) {
        return NULL;
    }
    if (ident_is_self_side(state, ident)) {
        team = state->self_team;
        slot_map = state->self_active_slot_to_team_index;
    } else if (ident_side_player(ident) != 0) {
        team = state->opp_team;
        slot_map = state->opp_active_slot_to_team_index;
    } else {
        return NULL;
    }
    team_index = slot_map[slot - 1];
    if (team_index < 0 || team_index >= RAW_TEAM_SIZE) {
        return NULL;
    }
    return &team[team_index];
}

static RawPokemon* fallback_pokemon_for_ident(RawBattleState* state, const char* ident) {
    RawPokemon* team;
    RawPokemon* pokemon;
    if (!state || !ident) {
        return NULL;
    }
    team = ident_is_self_side(state, ident) ? state->self_team : state->opp_team;
    pokemon = find_by_canonical_or_exact_ident(team, ident);
    if (pokemon) {
        return pokemon;
    }
    pokemon = find_same_identity_candidate(team, ident, 0, 0);
    if (pokemon) {
        return pokemon;
    }
    /* Fail closed for non-switch identity ambiguity. Creating a fresh object here
       can fork one logical Pokemon into multiple tracked objects. */
    return NULL;
}

static RawPokemon* incoming_switch_pokemon_for_ident(RawBattleState* state, const char* ident, int species_id) {
    RawPokemon* team;
    RawPokemon* pokemon;
    if (!state || !ident) {
        return NULL;
    }
    team = ident_is_self_side(state, ident) ? state->self_team : state->opp_team;
    pokemon = find_by_canonical_or_exact_ident(team, ident);
    if (pokemon) {
        return pokemon;
    }
    pokemon = find_same_identity_candidate(team, ident, species_id, 1);
    if (pokemon) {
        return pokemon;
    }
    pokemon = allocate_empty_identity_slot(team, ident);
    if (pokemon) {
        return pokemon;
    }
    return NULL;
}

static RawPokemon* pokemon_for_ident_via_slot_or_identity(RawBattleState* state, const char* ident) {
    RawPokemon* pokemon;
    if (!state || !ident) {
        return NULL;
    }
    pokemon = active_slot_pokemon_for_ident(state, ident);
    if (pokemon) {
        return pokemon;
    }
    return fallback_pokemon_for_ident(state, ident);
}

static void reveal_ability_for_ident(RawBattleState* state, const char* ident, const char* ability_name) {
    RawPokemon* pokemon;
    int ability_id;
    if (!state || !ident || !ability_name || !*ability_name) {
        return;
    }
    ability_id = ability_id_from_name(ability_name);
    if (ability_id <= 0) {
        return;
    }
    pokemon = pokemon_for_ident_via_slot_or_identity(state, ident);
    if (!pokemon) {
        return;
    }
    tracked_int_set_confirmed(&pokemon->ability_id, ability_id);
}

static const char* direct_ability_name_from_part(const char* part) {
    if (!part) {
        return NULL;
    }
    if (strncmp(part, "ability: ", 9) == 0) {
        return part + 9;
    }
    return NULL;
}

static const char* from_ability_name_from_part(const char* part) {
    if (!part) {
        return NULL;
    }
    if (strncmp(part, "[from] ability: ", 16) == 0) {
        return part + 16;
    }
    return NULL;
}

static const char* of_ident_from_part(const char* part) {
    if (!part) {
        return NULL;
    }
    if (strncmp(part, "[of] ", 5) == 0) {
        return part + 5;
    }
    return NULL;
}

static void reveal_ability_from_parts(RawBattleState* state, const char* subject_ident, const LineParts* parts, int start_index) {
    const char* direct_ability = NULL;
    const char* from_ability = NULL;
    const char* of_ident = NULL;
    int i;
    if (!state || !parts) {
        return;
    }
    for (i = start_index; i < parts->count; ++i) {
        if (!direct_ability) {
            direct_ability = direct_ability_name_from_part(parts->parts[i]);
        }
        if (!from_ability) {
            from_ability = from_ability_name_from_part(parts->parts[i]);
        }
        if (!of_ident) {
            of_ident = of_ident_from_part(parts->parts[i]);
        }
    }
    if (direct_ability && subject_ident && ident_side_player(subject_ident) != 0) {
        reveal_ability_for_ident(state, subject_ident, direct_ability);
        return;
    }
    if (from_ability) {
        if (of_ident) {
            reveal_ability_for_ident(state, of_ident, from_ability);
        } else if (subject_ident && ident_side_player(subject_ident) != 0) {
            reveal_ability_for_ident(state, subject_ident, from_ability);
        }
    }
}

static RawSideState* side_for_ident(RawBattleState* state, const char* ident) {
    if (!state || !ident) {
        return NULL;
    }
    return ident_is_self_side(state, ident) ? &state->self_side : &state->opp_side;
}

static RawPokemon* active_switch_source_for_side(RawBattleState* state, int self_side) {
    RawPokemon* team;
    int i;
    if (!state) {
        return NULL;
    }
    team = self_side ? state->self_team : state->opp_team;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].active && team[i].switched_in_turn == state->turn_number && !team[i].ability_triggered_on_switch_in) {
            return &team[i];
        }
    }
    return NULL;
}

static void clear_on_switch_local(RawPokemon* pokemon) {
    if (!pokemon) {
        return;
    }
    pokemon->active = 0;
    pokemon->active_slot = 0;
    pokemon->encore_active = 0;
    pokemon->encore_turns = 0;
    pokemon->disable_active = 0;
    pokemon->disable_turns = 0;
    pokemon->taunt_active = 0;
    pokemon->taunt_turns = 0;
    pokemon->torment_active = 0;
    pokemon->torment_turns = 0;
    pokemon->heal_block_active = 0;
    pokemon->heal_block_turns = 0;
    pokemon->embargo_active = 0;
    pokemon->embargo_turns = 0;
    pokemon->yawn_active = 0;
    pokemon->yawn_turns = 0;
    pokemon->confusion_active = 0;
    pokemon->confusion_turns = 0;
    pokemon->seed_active = 0;
    pokemon->substitute_active = 0;
    pokemon->charge_active = 0;
    pokemon->charge_turns = 0;
    pokemon->toxic_counter = 0;
    pokemon->protect_chain_count = 0;
    pokemon->sleep_turns_elapsed = 0;
    pokemon->perish_song_counter = 0;
    pokemon->ability_triggered_on_switch_in = 0;
    raw_pokemon_clear_transform(pokemon);
}

static void clear_existing_active_slot(RawBattleState* state, const char* ident) {
    RawPokemon* team;
    int slot;
    int i;
    if (!state || !ident) {
        return;
    }
    team = ident_is_self_side(state, ident) ? state->self_team : state->opp_team;
    slot = strstr(ident, "b:") ? 2 : 1;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].active && team[i].active_slot == slot) {
            clear_on_switch_local(&team[i]);
            team[i].first_turn_on_field = 0;
        }
    }
}

static void parse_condition(const char* cond, int* hp, int* max_hp, int* status_id, int* fainted, int* hp_known) {
    const char* slash;
    *hp = 0;
    *max_hp = 0;
    *status_id = 0;
    *fainted = 0;
    *hp_known = 0;
    if (!cond) return;
    if (strstr(cond, "fnt")) *fainted = 1;
    if (strstr(cond, "brn")) *status_id = 1;
    else if (strstr(cond, "par")) *status_id = 2;
    else if (strstr(cond, "psn")) *status_id = 3;
    else if (strstr(cond, "tox")) *status_id = 4;
    else if (strstr(cond, "frz")) *status_id = 5;
    else if (strstr(cond, "slp")) *status_id = 6;
    slash = strchr(cond, '/');
    if (slash) {
        *hp_known = 1;
        *hp = atoi(cond);
        *max_hp = atoi(slash + 1);
    }
}

static void apply_effective_type_change(RawPokemon* pokemon, const char* detail) {
    const char* slash;
    char type1_name[32];
    char type2_name[32];
    size_t len;
    int type1_id;
    int type2_id = 0;
    if (!pokemon || !detail || !*detail) {
        return;
    }
    slash = strchr(detail, '/');
    if (!slash) {
        type1_id = type_id_from_name(detail);
        if (type1_id <= 0) {
            return;
        }
        tracked_int_set_confirmed(&pokemon->effective_type1_id, type1_id);
        tracked_int_set_confirmed(&pokemon->effective_type2_id, 0);
        return;
    }
    len = (size_t)(slash - detail);
    if (len >= sizeof(type1_name)) {
        len = sizeof(type1_name) - 1;
    }
    memcpy(type1_name, detail, len);
    type1_name[len] = '\0';
    strncpy(type2_name, slash + 1, sizeof(type2_name) - 1);
    type2_name[sizeof(type2_name) - 1] = '\0';
    type1_id = type_id_from_name(type1_name);
    type2_id = type_id_from_name(type2_name);
    if (type1_id <= 0) {
        return;
    }
    tracked_int_set_confirmed(&pokemon->effective_type1_id, type1_id);
    tracked_int_set_confirmed(&pokemon->effective_type2_id, type2_id > 0 ? type2_id : 0);
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
    if (state->weather_id) {
        /* Public protocol usually tells us weather start/clear but not whether an
           extender is in play, so remaining turns are tracked as inferred. */
        tracked_int_set_inferred(&state->weather_turns_remaining, 5);
    } else {
        tracked_int_set_unknown(&state->weather_turns_remaining);
    }
}

static void handle_field_start(RawBattleState* state, const char* value) {
    if (strncmp(value, "Electric Terrain", 16) == 0) {
        state->terrain_id = 1;
        /* Terrain duration is an inferred estimate unless separately proven. */
        tracked_int_set_inferred(&state->terrain_turns_remaining, 5);
    } else if (strncmp(value, "Grassy Terrain", 14) == 0) {
        state->terrain_id = 2;
        tracked_int_set_inferred(&state->terrain_turns_remaining, 5);
    } else if (strncmp(value, "Misty Terrain", 13) == 0) {
        state->terrain_id = 3;
        tracked_int_set_inferred(&state->terrain_turns_remaining, 5);
    } else if (strncmp(value, "Psychic Terrain", 15) == 0) {
        state->terrain_id = 4;
        tracked_int_set_inferred(&state->terrain_turns_remaining, 5);
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
        tracked_int_set_unknown(&state->terrain_turns_remaining);
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
    } else if (strncmp(value, "move: Mud Sport", 15) == 0) {
        state->mud_sport = 0;
    } else if (strncmp(value, "move: Water Sport", 17) == 0) {
        state->water_sport = 0;
    } else if (strncmp(value, "move: Ion Deluge", 16) == 0) {
        state->ion_deluge = 0;
    }
}

static void handle_side_effect(RawBattleState* state, const char* who, const char* effect, int start) {
    RawSideState* side = side_for_ident(state, who);
    const char* normalized = effect;
    if (!side) return;
    if (normalized && strncmp(normalized, "move: ", 6) == 0) {
        normalized += 6;
    }
    if (strncmp(normalized, "Stealth Rock", 12) == 0) side->stealth_rock = start ? 1 : 0;
    else if (strncmp(normalized, "Spikes", 6) == 0) side->spikes = start ? (side->spikes < 3 ? side->spikes + 1 : 3) : 0;
    else if (strncmp(normalized, "Toxic Spikes", 12) == 0) side->toxic_spikes = start ? (side->toxic_spikes < 2 ? side->toxic_spikes + 1 : 2) : 0;
    else if (strncmp(normalized, "Sticky Web", 10) == 0) side->sticky_web = start ? 1 : 0;
    else if (strncmp(normalized, "Reflect", 7) == 0) { side->reflect = start ? 1 : 0; side->reflect_turns = start ? 5 : 0; }
    else if (strncmp(normalized, "Light Screen", 12) == 0) { side->light_screen = start ? 1 : 0; side->light_screen_turns = start ? 5 : 0; }
    else if (strncmp(normalized, "Aurora Veil", 11) == 0) { side->aurora_veil = start ? 1 : 0; side->aurora_veil_turns = start ? 5 : 0; }
    else if (strncmp(normalized, "Tailwind", 8) == 0) { side->tailwind = start ? 1 : 0; side->tailwind_turns = start ? 4 : 0; }
    else if (strncmp(normalized, "Safeguard", 9) == 0) { side->safeguard = start ? 1 : 0; side->safeguard_turns = start ? 5 : 0; }
    else if (strncmp(normalized, "Mist", 4) == 0) { side->mist = start ? 1 : 0; side->mist_turns = start ? 5 : 0; }
    else if (strncmp(normalized, "Lucky Chant", 11) == 0) { side->lucky_chant = start ? 1 : 0; side->lucky_chant_turns = start ? 5 : 0; }
}

static void handle_switch_like(RawBattleState* state, const char* ident, const char* details, const char* cond) {
    int species_id;
    RawPokemon* pokemon;
    int hp, max_hp, status_id, fainted, hp_known;
    char species[64];
    size_t i = 0;
    while (details[i] && details[i] != ',' && i + 1 < sizeof(species)) {
        species[i] = details[i];
        ++i;
    }
    species[i] = '\0';
    species_id = species_id_from_name(species);
    pokemon = incoming_switch_pokemon_for_ident(state, ident, species_id);
    if (!pokemon) return;
    clear_existing_active_slot(state, ident);
    clear_on_switch_local(pokemon);
    tracked_int_promote_confirmed(&pokemon->species_id, species_id);
    raw_pokemon_refresh_types(pokemon);
    raw_pokemon_refresh_effective_state(pokemon);
    parse_condition(cond, &hp, &max_hp, &status_id, &fainted, &hp_known);
    if (hp_known) {
        pokemon->current_hp = hp;
        pokemon->max_hp = max_hp;
    }
    tracked_int_promote_confirmed(&pokemon->status_id, status_id);
    pokemon->fainted = fainted;
    pokemon->active = 1;
    pokemon->revealed = 1;
    pokemon->switched_in_turn = state->turn_number;
    pokemon->first_turn_on_field = 1;
    pokemon->active_slot = strstr(ident, "b:") ? 2 : 1;
    strncpy(pokemon->canonical_ident, ident, RAW_IDENT_LEN - 1);
    pokemon->canonical_ident[RAW_IDENT_LEN - 1] = '\0';
}

static void handle_status_line(RawBattleState* state, const char* ident, const char* value) {
    RawPokemon* pokemon = pokemon_for_ident_via_slot_or_identity(state, ident);
    if (!pokemon) return;
    if (strncmp(value, "brn", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 1);
    else if (strncmp(value, "par", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 2);
    else if (strncmp(value, "psn", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 3);
    else if (strncmp(value, "tox", 3) == 0) { tracked_int_promote_confirmed(&pokemon->status_id, 4); pokemon->toxic_counter = 1; }
    else if (strncmp(value, "frz", 3) == 0) tracked_int_promote_confirmed(&pokemon->status_id, 5);
    else if (strncmp(value, "slp", 3) == 0) { tracked_int_promote_confirmed(&pokemon->status_id, 6); pokemon->sleep_turns_elapsed = 0; }
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
    RawPokemon* pokemon = pokemon_for_ident_via_slot_or_identity(state, ident);
    int idx = boost_index(stat);
    if (!pokemon || idx < 0) return;
    if (mode == 0) pokemon->boosts[idx] += amount;
    else if (mode == 1) pokemon->boosts[idx] -= amount;
    else if (mode == 2) pokemon->boosts[idx] = amount;
}

static int move_slot_for_name(const RawPokemon* pokemon, const char* move_name) {
    int move_id;
    int i;
    if (!pokemon || !move_name || !*move_name) {
        return -1;
    }
    move_id = move_id_from_name(move_name);
    if (move_id <= 0) {
        return -1;
    }
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (pokemon->effective_move_ids[i].value == move_id || pokemon->move_ids[i].value == move_id) {
            return i;
        }
    }
    return -1;
}

static void handle_start(RawBattleState* state, const char* ident, const char* effect, const char* detail) {
    RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, ident);
    const char* normalized = effect;
    if (!p) return;
    if (normalized && strncmp(normalized, "ability: ", 9) == 0) {
        reveal_ability_for_ident(state, ident, normalized + 9);
        return;
    }
    if (normalized && strncmp(normalized, "move: ", 6) == 0) {
        normalized += 6;
    }
    if (strncmp(normalized, "Encore", 6) == 0) { p->encore_active = 1; p->encore_turns = 3; }
    else if (strncmp(normalized, "Disable", 7) == 0) {
        p->disable_active = 1;
        p->disable_turns = 4;
        if (detail && *detail) {
            p->disable_move_slot = move_slot_for_name(p, detail);
        }
    }
    else if (strncmp(normalized, "Taunt", 5) == 0) { p->taunt_active = 1; p->taunt_turns = 3; }
    else if (strncmp(normalized, "Torment", 7) == 0) { p->torment_active = 1; p->torment_turns = 3; }
    else if (strncmp(normalized, "Heal Block", 10) == 0) { p->heal_block_active = 1; p->heal_block_turns = 5; }
    else if (strncmp(normalized, "Embargo", 7) == 0) { p->embargo_active = 1; p->embargo_turns = 5; }
    else if (strncmp(normalized, "Yawn", 4) == 0) { p->yawn_active = 1; p->yawn_turns = 2; }
    else if (strncmp(normalized, "confusion", 9) == 0) { p->confusion_active = 1; p->confusion_turns = 3; }
    else if (strncmp(normalized, "Substitute", 10) == 0) p->substitute_active = 1;
    else if (strncmp(normalized, "Leech Seed", 10) == 0) p->seed_active = 1;
    else if (strncmp(normalized, "Perish Song", 11) == 0) p->perish_song_counter = 4;
    else if (strncmp(normalized, "Charge", 6) == 0) { p->charge_active = 1; p->charge_turns = 2; }
    else if (strncmp(normalized, "No Retreat", 10) == 0) { p->trapped = 1; p->maybe_trapped = 0; }
    else if (strncmp(normalized, "typechange", 10) == 0) apply_effective_type_change(p, detail);
}

static void handle_end(RawBattleState* state, const char* ident, const char* effect) {
    RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, ident);
    const char* normalized = effect;
    if (!p) return;
    if (normalized && strncmp(normalized, "move: ", 6) == 0) {
        normalized += 6;
    }
    if (strncmp(normalized, "Encore", 6) == 0) { p->encore_active = 0; p->encore_turns = 0; p->encore_move_slot = -1; }
    else if (strncmp(normalized, "Disable", 7) == 0) { p->disable_active = 0; p->disable_turns = 0; p->disable_move_slot = -1; }
    else if (strncmp(normalized, "Taunt", 5) == 0) { p->taunt_active = 0; p->taunt_turns = 0; }
    else if (strncmp(normalized, "Torment", 7) == 0) { p->torment_active = 0; p->torment_turns = 0; }
    else if (strncmp(normalized, "Heal Block", 10) == 0) { p->heal_block_active = 0; p->heal_block_turns = 0; }
    else if (strncmp(normalized, "Embargo", 7) == 0) { p->embargo_active = 0; p->embargo_turns = 0; }
    else if (strncmp(normalized, "Yawn", 4) == 0) { p->yawn_active = 0; p->yawn_turns = 0; }
    else if (strncmp(normalized, "confusion", 9) == 0) { p->confusion_active = 0; p->confusion_turns = 0; }
    else if (strncmp(normalized, "Substitute", 10) == 0) p->substitute_active = 0;
    else if (strncmp(normalized, "Leech Seed", 10) == 0) p->seed_active = 0;
    else if (strncmp(normalized, "Charge", 6) == 0) { p->charge_active = 0; p->charge_turns = 0; }
    else if (strncmp(normalized, "typechange", 10) == 0) raw_pokemon_refresh_effective_state(p);
}

static void refresh_active_counts(RawBattleState* state) {
    int i;
    int self_count = 0;
    int opp_count = 0;
    int self_remaining = 0;
    int opp_remaining = 0;
    if (!state) {
        return;
    }
    state->self_active_slot_to_team_index[0] = -1;
    state->self_active_slot_to_team_index[1] = -1;
    state->opp_active_slot_to_team_index[0] = -1;
    state->opp_active_slot_to_team_index[1] = -1;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (state->self_team[i].active && !state->self_team[i].fainted) {
            ++self_count;
        }
        if (state->opp_team[i].active && !state->opp_team[i].fainted) {
            ++opp_count;
        }
        if (state->self_team[i].active &&
                state->self_team[i].active_slot >= 1 &&
                state->self_team[i].active_slot <= 2) {
            state->self_active_slot_to_team_index[state->self_team[i].active_slot - 1] = i;
        }
        if (state->opp_team[i].active &&
                state->opp_team[i].active_slot >= 1 &&
                state->opp_team[i].active_slot <= 2) {
            state->opp_active_slot_to_team_index[state->opp_team[i].active_slot - 1] = i;
        }
        if (!state->self_team[i].fainted) {
            ++self_remaining;
        }
        if (!state->opp_team[i].fainted) {
            ++opp_remaining;
        }
    }
    state->self_active_count = self_count;
    state->opp_active_count = opp_count;
    state->self_side.remaining_pokemon = self_remaining;
    state->opp_side.remaining_pokemon = opp_remaining;
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
    else if (strcmp(parts.parts[0], "-weather") == 0 && parts.count > 1) {
        handle_weather(state, parts.parts[1]);
        reveal_ability_from_parts(state, NULL, &parts, 2);
        if (state->weather_id > 0) {
            RawPokemon* source = active_switch_source_for_side(state, 0);
            if (!source) {
                source = active_switch_source_for_side(state, 1);
            }
            if (source) {
                inference_engine_infer_weather_ability(source, state->weather_id);
                source->ability_triggered_on_switch_in = 1;
            }
        }
    }
    else if (strcmp(parts.parts[0], "-fieldstart") == 0 && parts.count > 1) {
        handle_field_start(state, parts.parts[1]);
        reveal_ability_from_parts(state, NULL, &parts, 2);
        if (strstr(parts.parts[1], "Terrain")) {
            RawPokemon* source = active_switch_source_for_side(state, 0);
            if (!source) {
                source = active_switch_source_for_side(state, 1);
            }
            if (source) {
                inference_engine_infer_terrain_ability(source, state->terrain_id);
                source->ability_triggered_on_switch_in = 1;
            }
        }
    }
    else if (strcmp(parts.parts[0], "-fieldend") == 0 && parts.count > 1) handle_field_end(state, parts.parts[1]);
    else if (strcmp(parts.parts[0], "-sidestart") == 0 && parts.count > 2) handle_side_effect(state, parts.parts[1], parts.parts[2], 1);
    else if (strcmp(parts.parts[0], "-sideend") == 0 && parts.count > 2) handle_side_effect(state, parts.parts[1], parts.parts[2], 0);
    else if ((strcmp(parts.parts[0], "switch") == 0 || strcmp(parts.parts[0], "drag") == 0 || strcmp(parts.parts[0], "replace") == 0) && parts.count > 3) handle_switch_like(state, parts.parts[1], parts.parts[2], parts.parts[3]);
    else if (strcmp(parts.parts[0], "-status") == 0 && parts.count > 2) handle_status_line(state, parts.parts[1], parts.parts[2]);
    else if (strcmp(parts.parts[0], "-curestatus") == 0 && parts.count > 1) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        if (p) {
            tracked_int_set_confirmed(&p->status_id, 0);
            p->sleep_turns_elapsed = 0;
            p->toxic_counter = 0;
        }
    }
    else if (strcmp(parts.parts[0], "-boost") == 0 && parts.count > 3) handle_boost(state, parts.parts[1], parts.parts[2], atoi(parts.parts[3]), 0);
    else if (strcmp(parts.parts[0], "-unboost") == 0 && parts.count > 3) {
        handle_boost(state, parts.parts[1], parts.parts[2], atoi(parts.parts[3]), 1);
        if (strcmp(parts.parts[2], "atk") == 0) {
            RawPokemon* source = ident_is_self_side(state, parts.parts[1])
                ? active_switch_source_for_side(state, 0)
                : active_switch_source_for_side(state, 1);
            if (source && source->ability_id.knowledge != KNOW_CONFIRMED) {
                tracked_int_set_inferred(&source->ability_id, ability_id_from_name("intimidate"));
                source->ability_triggered_on_switch_in = 1;
            }
        }
    }
    else if (strcmp(parts.parts[0], "-setboost") == 0 && parts.count > 3) handle_boost(state, parts.parts[1], parts.parts[2], atoi(parts.parts[3]), 2);
    else if (strcmp(parts.parts[0], "-clearallboost") == 0) { int i,j; for (i = 0; i < RAW_TEAM_SIZE; ++i) for (j = 0; j < 7; ++j) { state->self_team[i].boosts[j] = 0; state->opp_team[i].boosts[j] = 0; } }
    else if (strcmp(parts.parts[0], "-clearboost") == 0 && parts.count > 1) { int j; RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) for (j = 0; j < 7; ++j) p->boosts[j] = 0; }
    else if (strcmp(parts.parts[0], "-start") == 0 && parts.count > 2) {
        handle_start(state, parts.parts[1], parts.parts[2], parts.count > 3 ? parts.parts[3] : NULL);
        reveal_ability_from_parts(state, parts.parts[1], &parts, 2);
    }
    else if (strcmp(parts.parts[0], "-end") == 0 && parts.count > 2) handle_end(state, parts.parts[1], parts.parts[2]);
    else if (strcmp(parts.parts[0], "-formechange") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        int species_id = species_id_from_name(parts.parts[2]);
        if (p && species_id > 0) {
            tracked_int_set_confirmed(&p->species_id, species_id);
            raw_pokemon_refresh_types(p);
            raw_pokemon_refresh_effective_state(p);
        }
        reveal_ability_from_parts(state, parts.parts[1], &parts, 3);
    }
    else if (strcmp(parts.parts[0], "-flinch") == 0 && parts.count > 1) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        if (p) {
            p->flinch_active = 1;
        }
    }
    else if (strcmp(parts.parts[0], "cant") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        if (p && strcmp(parts.parts[2], "flinch") == 0) {
            p->flinch_active = 1;
        }
        if (p && strcmp(parts.parts[2], "par") == 0) {
            tracked_int_promote_confirmed(&p->status_id, 2);
        } else if (p && strcmp(parts.parts[2], "frz") == 0) {
            tracked_int_promote_confirmed(&p->status_id, 5);
        } else if (p && strcmp(parts.parts[2], "slp") == 0) {
            tracked_int_promote_confirmed(&p->status_id, 6);
        }
        reveal_ability_from_parts(state, parts.parts[1], &parts, 2);
    }
    else if (strcmp(parts.parts[0], "-activate") == 0 && parts.count > 2) reveal_ability_from_parts(state, parts.parts[1], &parts, 2);
    else if (strcmp(parts.parts[0], "-singleturn") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        RawSideState* side = side_for_ident(state, parts.parts[1]);
        if (p && (strcmp(parts.parts[2], "Protect") == 0 || strcmp(parts.parts[2], "Detect") == 0 || strcmp(parts.parts[2], "Endure") == 0)) {
            p->protect_active = 1;
            p->protect_chain_count += 1;
        }
        if (side && strcmp(parts.parts[2], "Quick Guard") == 0) side->quick_guard = 1;
        else if (side && strcmp(parts.parts[2], "Wide Guard") == 0) side->wide_guard = 1;
        else if (side && strcmp(parts.parts[2], "Crafty Shield") == 0) side->crafty_shield = 1;
        else if (side && strcmp(parts.parts[2], "Mat Block") == 0) side->mat_block = 1;
    }
    else if (strcmp(parts.parts[0], "-singlemove") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        if (p && strcmp(parts.parts[2], "Helping Hand") == 0) p->helping_hand_active = 1;
    }
    else if (strcmp(parts.parts[0], "-damage") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); int hp,max_hp,status,fnt,hp_known; if (p) { parse_condition(parts.parts[2], &hp,&max_hp,&status,&fnt,&hp_known); if (hp_known) { p->current_hp=hp; p->max_hp=max_hp; } else if (fnt) { p->current_hp = 0; } if (status) tracked_int_promote_confirmed(&p->status_id,status); p->fainted=fnt; }
    }
    else if (strcmp(parts.parts[0], "-heal") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); int hp,max_hp,status,fnt,hp_known; if (p) { parse_condition(parts.parts[2], &hp,&max_hp,&status,&fnt,&hp_known); if (hp_known) { p->current_hp=hp; p->max_hp=max_hp; } else if (fnt) { p->current_hp = 0; } if (status) tracked_int_promote_confirmed(&p->status_id,status); p->fainted=fnt; }
    }
    else if (strcmp(parts.parts[0], "faint") == 0 && parts.count > 1) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) { p->fainted = 1; p->current_hp = 0; }
    }
    else if (strcmp(parts.parts[0], "-ability") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) tracked_int_set_confirmed(&p->ability_id, ability_id_from_name(parts.parts[2]));
    }
    else if (strcmp(parts.parts[0], "-item") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) tracked_int_set_confirmed(&p->item_id, item_id_from_name(parts.parts[2]));
    }
    else if (strcmp(parts.parts[0], "-enditem") == 0 && parts.count > 1) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) tracked_int_set_confirmed(&p->item_id, 0);
    }
    else if (strcmp(parts.parts[0], "-terastallize") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) { p->tera_used = 1; tracked_int_set_confirmed(&p->tera_type_id, type_id_from_name(parts.parts[2])); raw_pokemon_refresh_effective_state(p); }
    }
    else if (strcmp(parts.parts[0], "-transform") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]);
        RawPokemon* target = pokemon_for_ident_via_slot_or_identity(state, parts.parts[2]);
        if (p && target) {
            raw_pokemon_apply_transform(p, target);
        }
    }
    else if (strcmp(parts.parts[0], "-fieldactivate") == 0 && parts.count > 1) {
        if (strcmp(parts.parts[1], "move: Ion Deluge") == 0) state->ion_deluge = 1;
        else if (strcmp(parts.parts[1], "move: Mud Sport") == 0) state->mud_sport = 1;
        else if (strcmp(parts.parts[1], "move: Water Sport") == 0) state->water_sport = 1;
    }
    else if (strcmp(parts.parts[0], "move") == 0 && parts.count > 2) {
        RawPokemon* p = pokemon_for_ident_via_slot_or_identity(state, parts.parts[1]); if (p) { if (p->transformed) inference_engine_note_effective_move(p, move_id_from_name(parts.parts[2]), state->turn_number); else { inference_engine_note_move(p, move_id_from_name(parts.parts[2]), state->turn_number); raw_pokemon_refresh_effective_state(p); } }
    }
    refresh_active_counts(state);
}
