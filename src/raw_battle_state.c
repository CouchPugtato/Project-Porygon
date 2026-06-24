#include "raw_battle_state.h"

#include "event_parser.h"
#include "id_tables.h"

#include <string.h>

static void tracked_int_copy(TrackedInt* out, const TrackedInt* in) {
    if (!out || !in) {
        return;
    }
    *out = *in;
}

static void swap_raw_pokemon(RawPokemon* a, RawPokemon* b) {
    RawPokemon tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void swap_raw_side(RawSideState* a, RawSideState* b) {
    RawSideState tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void raw_pokemon_init(RawPokemon* pokemon) {
    int i;
    if (!pokemon) {
        return;
    }
    memset(pokemon, 0, sizeof(*pokemon));
    tracked_int_reset(&pokemon->species_id);
    tracked_int_reset(&pokemon->item_id);
    tracked_int_reset(&pokemon->ability_id);
    tracked_int_reset(&pokemon->tera_type_id);
    tracked_int_reset(&pokemon->type1_id);
    tracked_int_reset(&pokemon->type2_id);
    tracked_int_reset(&pokemon->effective_species_id);
    tracked_int_reset(&pokemon->effective_type1_id);
    tracked_int_reset(&pokemon->effective_type2_id);
    tracked_int_reset(&pokemon->status_id);
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        tracked_int_reset(&pokemon->move_ids[i]);
        tracked_int_reset(&pokemon->effective_move_ids[i]);
        tracked_int_reset(&pokemon->move_type_ids[i]);
        tracked_int_reset(&pokemon->effective_move_type_ids[i]);
    }
    pokemon->encore_move_slot = -1;
    pokemon->disable_move_slot = -1;
    pokemon->self_request_roster_index = -1;
}

static void refresh_tracked_move_type(TrackedInt* out, const TrackedInt* move_id) {
    int type_id;
    if (!out || !move_id) {
        return;
    }
    type_id = move_type_from_id(move_id->value);
    if (type_id <= 0) {
        tracked_int_set_unknown(out);
        return;
    }
    if (move_id->knowledge == KNOW_CONFIRMED) {
        tracked_int_set_confirmed(out, type_id);
    } else if (move_id->knowledge == KNOW_INFERRED) {
        tracked_int_set_inferred(out, type_id);
    } else {
        tracked_int_set_unknown(out);
    }
}

static void raw_side_init(RawSideState* side) {
    if (!side) {
        return;
    }
    memset(side, 0, sizeof(*side));
    side->remaining_pokemon = RAW_TEAM_SIZE;
}

void raw_pokemon_refresh_types(RawPokemon* pokemon) {
    int base_type1;
    int base_type2;
    if (!pokemon) {
        return;
    }
    base_type1 = species_type1_from_id(pokemon->species_id.value);
    base_type2 = species_type2_from_id(pokemon->species_id.value);
    if (base_type1 <= 0) {
        tracked_int_set_unknown(&pokemon->type1_id);
        tracked_int_set_unknown(&pokemon->type2_id);
    } else if (pokemon->species_id.knowledge == KNOW_CONFIRMED) {
        tracked_int_set_confirmed(&pokemon->type1_id, base_type1);
        tracked_int_set_confirmed(&pokemon->type2_id, base_type2);
    } else if (pokemon->species_id.knowledge == KNOW_INFERRED) {
        tracked_int_set_inferred(&pokemon->type1_id, base_type1);
        tracked_int_set_inferred(&pokemon->type2_id, base_type2);
    } else {
        tracked_int_set_unknown(&pokemon->type1_id);
        tracked_int_set_unknown(&pokemon->type2_id);
    }
}

void raw_pokemon_refresh_effective_state(RawPokemon* pokemon) {
    int i;
    if (!pokemon) {
        return;
    }
    /* Invariant: base->effective refresh owns only non-transformed state.
       Transformed effective state must be maintained by the transform-specific
       paths and must not be rebuilt from base fields. */
    if (pokemon->transformed) {
        return;
    }
    tracked_int_copy(&pokemon->effective_species_id, &pokemon->species_id);
    tracked_int_copy(&pokemon->effective_type1_id, &pokemon->type1_id);
    tracked_int_copy(&pokemon->effective_type2_id, &pokemon->type2_id);
    if (pokemon->tera_used && pokemon->tera_type_id.value > 0) {
        if (pokemon->tera_type_id.knowledge == KNOW_CONFIRMED) {
            tracked_int_set_confirmed(&pokemon->effective_type1_id, pokemon->tera_type_id.value);
            tracked_int_set_confirmed(&pokemon->effective_type2_id, 0);
        } else if (pokemon->tera_type_id.knowledge == KNOW_INFERRED) {
            tracked_int_set_inferred(&pokemon->effective_type1_id, pokemon->tera_type_id.value);
            tracked_int_set_inferred(&pokemon->effective_type2_id, 0);
        } else {
            tracked_int_set_unknown(&pokemon->effective_type1_id);
            tracked_int_set_unknown(&pokemon->effective_type2_id);
        }
    }
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        refresh_tracked_move_type(&pokemon->move_type_ids[i], &pokemon->move_ids[i]);
        tracked_int_copy(&pokemon->effective_move_ids[i], &pokemon->move_ids[i]);
        tracked_int_copy(&pokemon->effective_move_type_ids[i], &pokemon->move_type_ids[i]);
        pokemon->effective_move_known[i] = pokemon->move_known[i];
        pokemon->effective_move_pp[i] = pokemon->move_pp[i];
        pokemon->effective_move_max_pp[i] = pokemon->move_max_pp[i];
        pokemon->effective_move_disabled[i] = pokemon->move_disabled[i];
        pokemon->effective_move_maybe_disabled[i] = pokemon->move_maybe_disabled[i];
    }
}

void raw_pokemon_apply_transform(RawPokemon* pokemon, const RawPokemon* target) {
    int i;
    if (!pokemon || !target) {
        return;
    }
    pokemon->transformed = 1;
    tracked_int_copy(&pokemon->effective_species_id, &target->effective_species_id);
    tracked_int_copy(&pokemon->effective_type1_id, &target->effective_type1_id);
    tracked_int_copy(&pokemon->effective_type2_id, &target->effective_type2_id);
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        tracked_int_copy(&pokemon->effective_move_ids[i], &target->effective_move_ids[i]);
        tracked_int_copy(&pokemon->effective_move_type_ids[i], &target->effective_move_type_ids[i]);
        pokemon->effective_move_known[i] = target->effective_move_known[i];
        if (target->effective_move_ids[i].value > 0 || target->effective_move_known[i]) {
            pokemon->effective_move_pp[i] = 5;
            pokemon->effective_move_max_pp[i] = 5;
        } else {
            pokemon->effective_move_pp[i] = 0;
            pokemon->effective_move_max_pp[i] = 0;
        }
        pokemon->effective_move_disabled[i] = 0;
        pokemon->effective_move_maybe_disabled[i] = 0;
    }
}

void raw_pokemon_clear_transform(RawPokemon* pokemon) {
    if (!pokemon) {
        return;
    }
    pokemon->transformed = 0;
    raw_pokemon_refresh_effective_state(pokemon);
}

static void reset_active_slot_maps(RawBattleState* state) {
    if (!state) {
        return;
    }
    state->self_active_slot_to_team_index[0] = -1;
    state->self_active_slot_to_team_index[1] = -1;
    state->opp_active_slot_to_team_index[0] = -1;
    state->opp_active_slot_to_team_index[1] = -1;
}

static void clear_per_turn_flags(RawPokemon* pokemon) {
    if (!pokemon) {
        return;
    }
    if (!pokemon->protect_active) {
        pokemon->protect_chain_count = 0;
    }
    pokemon->protect_active = 0;
    pokemon->helping_hand_active = 0;
    pokemon->flinch_active = 0;
    pokemon->first_turn_on_field = 0;
}

static void clear_on_switch(RawPokemon* pokemon) {
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
    pokemon->trapped = 0;
    pokemon->maybe_trapped = 0;
    raw_pokemon_clear_transform(pokemon);
}

static int count_living_active(const RawPokemon* team) {
    int i;
    int count = 0;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (team[i].active && !team[i].fainted) {
            ++count;
        }
    }
    return count;
}

static int count_remaining_pokemon(const RawPokemon* team) {
    int i;
    int count = 0;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (!team[i].fainted) {
            ++count;
        }
    }
    return count;
}

static void refresh_active_counts(RawBattleState* state) {
    int i;
    if (!state) {
        return;
    }
    reset_active_slot_maps(state);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (state->self_team[i].active && state->self_team[i].active_slot >= 1 && state->self_team[i].active_slot <= 2) {
            state->self_active_slot_to_team_index[state->self_team[i].active_slot - 1] = i;
        }
        if (state->opp_team[i].active && state->opp_team[i].active_slot >= 1 && state->opp_team[i].active_slot <= 2) {
            state->opp_active_slot_to_team_index[state->opp_team[i].active_slot - 1] = i;
        }
    }
    state->self_active_count = count_living_active(state->self_team);
    state->opp_active_count = count_living_active(state->opp_team);
    state->self_side.remaining_pokemon = count_remaining_pokemon(state->self_team);
    state->opp_side.remaining_pokemon = count_remaining_pokemon(state->opp_team);
}

static int infer_single_disabled_move_slot(const ParsedActive* active) {
    int i;
    int slot = -1;
    if (!active) {
        return -1;
    }
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (!active->move_id[i]) {
            continue;
        }
        if (!active->move_disabled[i]) {
            continue;
        }
        if (slot >= 0) {
            return -1;
        }
        slot = i;
    }
    return slot;
}

static int infer_encore_move_slot(const ParsedActive* active) {
    int i;
    int slot = -1;
    if (!active) {
        return -1;
    }
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (!active->move_id[i]) {
            continue;
        }
        if (active->move_disabled[i] || active->move_maybe_disabled[i]) {
            continue;
        }
        if (slot >= 0) {
            return -1;
        }
        slot = i;
    }
    return slot;
}

static void apply_self_side_perspective(RawBattleState* state, int side_player) {
    int i;
    if (!state || (side_player != 1 && side_player != 2)) {
        return;
    }
    if (!state->perspective_known) {
        state->self_side_player = side_player;
        state->perspective_known = 1;
        if (side_player == 2) {
            for (i = 0; i < RAW_TEAM_SIZE; ++i) {
                swap_raw_pokemon(&state->self_team[i], &state->opp_team[i]);
            }
            swap_raw_side(&state->self_side, &state->opp_side);
        }
        refresh_active_counts(state);
    }
}

static RawPokemon* find_self_by_ident(RawBattleState* state, const char* ident, int* out_index) {
    int i;
    const char* target_name;
    if (!state || !ident || !*ident) {
        return NULL;
    }
    target_name = strstr(ident, ": ");
    target_name = target_name ? (target_name + 2) : ident;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        const char* current_name;
        if (!state->self_team[i].ident[0]) {
            continue;
        }
        current_name = strstr(state->self_team[i].ident, ": ");
        current_name = current_name ? (current_name + 2) : state->self_team[i].ident;
        if (state->self_team[i].ident[0] == ident[0] &&
                state->self_team[i].ident[1] == ident[1] &&
                strcmp(current_name, target_name) == 0) {
            if (out_index) {
                *out_index = i;
            }
            return &state->self_team[i];
        }
    }
    return NULL;
}

static RawPokemon* find_self_by_roster_index(RawBattleState* state, int roster_index, const unsigned char* used, int* out_index) {
    int i;
    if (!state || roster_index < 0) {
        return NULL;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (used && used[i]) {
            continue;
        }
        if (state->self_team[i].self_request_roster_index == roster_index) {
            if (out_index) {
                *out_index = i;
            }
            return &state->self_team[i];
        }
    }
    return NULL;
}

static RawPokemon* find_self_by_species_and_inactive_preference(RawBattleState* state, int species_id, const unsigned char* used, int* out_index) {
    int i;
    if (!state || species_id <= 0) {
        return NULL;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (used && used[i]) {
            continue;
        }
        if (state->self_team[i].species_id.value == species_id && !state->self_team[i].active) {
            if (out_index) {
                *out_index = i;
            }
            return &state->self_team[i];
        }
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (used && used[i]) {
            continue;
        }
        if (state->self_team[i].species_id.value == species_id) {
            if (out_index) {
                *out_index = i;
            }
            return &state->self_team[i];
        }
    }
    return NULL;
}

static RawPokemon* find_empty_self_slot(RawBattleState* state, const unsigned char* used, int* out_index) {
    int i;
    if (!state) {
        return NULL;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (used && used[i]) {
            continue;
        }
        if (!state->self_team[i].ident[0] && state->self_team[i].species_id.value == 0) {
            if (out_index) {
                *out_index = i;
            }
            return &state->self_team[i];
        }
    }
    return NULL;
}

static RawPokemon* resolve_self_request_slot(
    RawBattleState* state,
    const ParsedRequest* req,
    int team_idx,
    const unsigned char* used,
    int* out_index
) {
    RawPokemon* pokemon;
    pokemon = find_self_by_ident(state, req->side_ident[team_idx], out_index);
    if (pokemon && (!used || !used[*out_index])) {
        return pokemon;
    }
    pokemon = find_self_by_roster_index(state, team_idx, used, out_index);
    if (pokemon) {
        return pokemon;
    }
    pokemon = find_self_by_species_and_inactive_preference(state, req->side_species_id[team_idx], used, out_index);
    if (pokemon) {
        return pokemon;
    }
    return find_empty_self_slot(state, used, out_index);
}

static int first_free_active_slot(const unsigned char* slot_used) {
    int slot;
    for (slot = 0; slot < 2; ++slot) {
        if (!slot_used[slot]) {
            return slot + 1;
        }
    }
    return 0;
}

void raw_battle_state_init(RawBattleState* state, int is_doubles) {
    int i;
    if (!state) {
        return;
    }
    memset(state, 0, sizeof(*state));
    state->is_doubles = is_doubles ? 1 : 0;
    state->self_side_player = 1;
    state->perspective_known = 0;
    tracked_int_reset(&state->weather_turns_remaining);
    tracked_int_reset(&state->terrain_turns_remaining);
    raw_side_init(&state->self_side);
    raw_side_init(&state->opp_side);
    reset_active_slot_maps(state);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        raw_pokemon_init(&state->self_team[i]);
        raw_pokemon_init(&state->opp_team[i]);
    }
}

void raw_battle_state_reset(RawBattleState* state) {
    int is_doubles = state ? state->is_doubles : 0;
    raw_battle_state_init(state, is_doubles);
}

void raw_battle_state_begin_turn(RawBattleState* state, int turn_number) {
    int i;
    if (!state) {
        return;
    }
    state->turn_number = turn_number;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        clear_per_turn_flags(&state->self_team[i]);
        clear_per_turn_flags(&state->opp_team[i]);
    }
    state->self_side.quick_guard = 0;
    state->self_side.wide_guard = 0;
    state->self_side.crafty_shield = 0;
    state->self_side.mat_block = 0;
    state->opp_side.quick_guard = 0;
    state->opp_side.wide_guard = 0;
    state->opp_side.crafty_shield = 0;
    state->opp_side.mat_block = 0;
    state->ion_deluge = 0;
}

static void decrement_duration(int* active, int* turns) {
    if (*active && *turns > 0) {
        --(*turns);
        if (*turns <= 0) {
            *active = 0;
            *turns = 0;
        }
    }
}

static void decrement_tracked_duration(int* active, TrackedInt* turns) {
    if (*active && turns->value > 0) {
        --turns->value;
        if (turns->value <= 0) {
            *active = 0;
            turns->value = 0;
        }
    }
}

void raw_battle_state_end_turn(RawBattleState* state) {
    int i;
    if (!state) {
        return;
    }
    decrement_duration(&state->trick_room, &state->trick_room_turns_remaining);
    decrement_duration(&state->magic_room, &state->magic_room_turns_remaining);
    decrement_duration(&state->wonder_room, &state->wonder_room_turns_remaining);
    decrement_duration(&state->gravity, &state->gravity_turns_remaining);
    if (state->weather_turns_remaining.value > 0) {
        --state->weather_turns_remaining.value;
    }
    if (state->terrain_turns_remaining.value > 0) {
        --state->terrain_turns_remaining.value;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (state->self_team[i].status_id.value == 4 &&
                state->self_team[i].active &&
                !state->self_team[i].fainted &&
                state->self_team[i].toxic_counter > 0) {
            state->self_team[i].toxic_counter += 1;
        }
        if (state->opp_team[i].status_id.value == 4 &&
                state->opp_team[i].active &&
                !state->opp_team[i].fainted &&
                state->opp_team[i].toxic_counter > 0) {
            state->opp_team[i].toxic_counter += 1;
        }
        if (state->self_team[i].status_id.value == 6 &&
                state->self_team[i].active &&
                !state->self_team[i].fainted) {
            state->self_team[i].sleep_turns_elapsed += 1;
        }
        if (state->opp_team[i].status_id.value == 6 &&
                state->opp_team[i].active &&
                !state->opp_team[i].fainted) {
            state->opp_team[i].sleep_turns_elapsed += 1;
        }
        if (state->self_team[i].perish_song_counter > 0 &&
                state->self_team[i].active &&
                !state->self_team[i].fainted) {
            state->self_team[i].perish_song_counter -= 1;
        }
        if (state->opp_team[i].perish_song_counter > 0 &&
                state->opp_team[i].active &&
                !state->opp_team[i].fainted) {
            state->opp_team[i].perish_song_counter -= 1;
        }
        decrement_duration(&state->self_team[i].encore_active, &state->self_team[i].encore_turns);
        decrement_duration(&state->self_team[i].disable_active, &state->self_team[i].disable_turns);
        decrement_duration(&state->self_team[i].taunt_active, &state->self_team[i].taunt_turns);
        decrement_duration(&state->self_team[i].torment_active, &state->self_team[i].torment_turns);
        decrement_duration(&state->self_team[i].heal_block_active, &state->self_team[i].heal_block_turns);
        decrement_duration(&state->self_team[i].embargo_active, &state->self_team[i].embargo_turns);
        decrement_duration(&state->self_team[i].yawn_active, &state->self_team[i].yawn_turns);
        decrement_duration(&state->self_team[i].confusion_active, &state->self_team[i].confusion_turns);
        decrement_duration(&state->self_team[i].charge_active, &state->self_team[i].charge_turns);
        decrement_duration(&state->opp_team[i].encore_active, &state->opp_team[i].encore_turns);
        decrement_duration(&state->opp_team[i].disable_active, &state->opp_team[i].disable_turns);
        decrement_duration(&state->opp_team[i].taunt_active, &state->opp_team[i].taunt_turns);
        decrement_duration(&state->opp_team[i].torment_active, &state->opp_team[i].torment_turns);
        decrement_duration(&state->opp_team[i].heal_block_active, &state->opp_team[i].heal_block_turns);
        decrement_duration(&state->opp_team[i].embargo_active, &state->opp_team[i].embargo_turns);
        decrement_duration(&state->opp_team[i].yawn_active, &state->opp_team[i].yawn_turns);
        decrement_duration(&state->opp_team[i].confusion_active, &state->opp_team[i].confusion_turns);
        decrement_duration(&state->opp_team[i].charge_active, &state->opp_team[i].charge_turns);
    }
    decrement_duration(&state->self_side.reflect, &state->self_side.reflect_turns);
    decrement_duration(&state->self_side.light_screen, &state->self_side.light_screen_turns);
    decrement_duration(&state->self_side.aurora_veil, &state->self_side.aurora_veil_turns);
    decrement_duration(&state->self_side.tailwind, &state->self_side.tailwind_turns);
    decrement_duration(&state->self_side.safeguard, &state->self_side.safeguard_turns);
    decrement_duration(&state->self_side.mist, &state->self_side.mist_turns);
    decrement_duration(&state->self_side.lucky_chant, &state->self_side.lucky_chant_turns);
    decrement_duration(&state->opp_side.reflect, &state->opp_side.reflect_turns);
    decrement_duration(&state->opp_side.light_screen, &state->opp_side.light_screen_turns);
    decrement_duration(&state->opp_side.aurora_veil, &state->opp_side.aurora_veil_turns);
    decrement_duration(&state->opp_side.tailwind, &state->opp_side.tailwind_turns);
    decrement_duration(&state->opp_side.safeguard, &state->opp_side.safeguard_turns);
    decrement_duration(&state->opp_side.mist, &state->opp_side.mist_turns);
    decrement_duration(&state->opp_side.lucky_chant, &state->opp_side.lucky_chant_turns);
}

int raw_battle_state_update_from_request(RawBattleState* state, const ParsedRequest* req) {
    int i;
    int matched_index[RAW_TEAM_SIZE];
    unsigned char used[RAW_TEAM_SIZE];
    unsigned char slot_used[2] = {0, 0};
    int prev_active_slot[RAW_TEAM_SIZE];
    int had_prior_active_slot_mapping = 0;
    if (!state || !req) {
        return 0;
    }
    apply_self_side_perspective(state, req->side_player);
    memset(matched_index, -1, sizeof(matched_index));
    memset(used, 0, sizeof(used));
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        prev_active_slot[i] = state->self_team[i].active ? state->self_team[i].active_slot : 0;
        if (prev_active_slot[i] >= 1 && prev_active_slot[i] <= 2) {
            had_prior_active_slot_mapping = 1;
        }
    }
    if (req->can_tera) {
        state->can_tera = 1;
    } else if (!req->wait && !req->team_preview && !req->forced_switch_any && req->active_count > 0) {
        state->can_tera = 0;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        state->self_team[i].active = 0;
        state->self_team[i].active_slot = 0;
        state->self_team[i].can_tera = 0;
        state->self_team[i].trapped = 0;
        state->self_team[i].maybe_trapped = 0;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        RawPokemon* pokemon;
        int resolved_index = -1;
        if (!req->side_ident[i][0] && req->side_species_id[i] <= 0) {
            continue;
        }
        pokemon = resolve_self_request_slot(state, req, i, used, &resolved_index);
        if (!pokemon || resolved_index < 0) {
            continue;
        }
        used[resolved_index] = 1;
        matched_index[i] = resolved_index;
        pokemon->known = 1;
        pokemon->revealed = 1;
        pokemon->self_request_roster_index = i;
        pokemon->fainted = req->switch_fainted[i];
        if (req->side_max_hp[i] > 0) {
            pokemon->current_hp = req->side_current_hp[i];
            pokemon->max_hp = req->side_max_hp[i];
        } else if (req->switch_fainted[i]) {
            pokemon->current_hp = 0;
        }
        if (req->side_ident[i][0] && !pokemon->canonical_ident[0]) {
            strncpy(pokemon->canonical_ident, req->side_ident[i], RAW_IDENT_LEN - 1);
            pokemon->canonical_ident[RAW_IDENT_LEN - 1] = '\0';
        }
        if (req->side_species_id[i] > 0) {
            tracked_int_promote_confirmed(&pokemon->species_id, req->side_species_id[i]);
            raw_pokemon_refresh_types(pokemon);
            raw_pokemon_refresh_effective_state(pokemon);
        }
        if (req->side_ident[i][0]) {
            if (!pokemon->ident[0]) {
                strncpy(pokemon->ident, req->side_ident[i], RAW_IDENT_LEN - 1);
                pokemon->ident[RAW_IDENT_LEN - 1] = '\0';
            }
        }
        if (req->side_item_id[i] >= 0) {
            tracked_int_promote_confirmed(&pokemon->item_id, req->side_item_id[i]);
        }
        if (req->side_ability_id[i] > 0) {
            tracked_int_promote_confirmed(&pokemon->ability_id, req->side_ability_id[i]);
        } else if (req->side_base_ability_id[i] > 0) {
            tracked_int_promote_confirmed(&pokemon->ability_id, req->side_base_ability_id[i]);
        }
        if (req->side_tera_type_id[i] > 0) {
            tracked_int_promote_confirmed(&pokemon->tera_type_id, req->side_tera_type_id[i]);
        }
        if (req->side_tera_used[i]) {
            pokemon->tera_used = 1;
        }
        pokemon->base_hp_stat = req->side_stats_hp[i];
        pokemon->base_atk_stat = req->side_stats_atk[i];
        pokemon->base_def_stat = req->side_stats_def[i];
        pokemon->base_spa_stat = req->side_stats_spa[i];
        pokemon->base_spd_stat = req->side_stats_spd[i];
        pokemon->base_spe_stat = req->side_stats_spe[i];
        pokemon->commanding_active = req->side_commanding[i];
        pokemon->reviving = req->side_reviving[i];
        {
            int m;
            for (m = 0; m < RAW_MOVE_SLOTS; ++m) {
                if (req->side_move_id[i][m] > 0) {
                    tracked_int_promote_confirmed(&pokemon->move_ids[m], req->side_move_id[i][m]);
                    pokemon->move_known[m] = 1;
                }
            }
        }
        raw_pokemon_refresh_types(pokemon);
        if (!pokemon->transformed) {
            raw_pokemon_refresh_effective_state(pokemon);
        }
    }
    if (!had_prior_active_slot_mapping) {
        for (i = 0; i < req->active_count && i < 2; ++i) {
            if (!req->active_team_idx_known[i] || req->active_team_idx[i] < 0) {
                return 0;
            }
        }
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        int team_index = matched_index[i];
        int assigned_slot = 0;
        if (team_index < 0 || !req->switch_active[i]) {
            continue;
        }
        if (prev_active_slot[team_index] >= 1 && prev_active_slot[team_index] <= 2 &&
                !slot_used[prev_active_slot[team_index] - 1]) {
            assigned_slot = prev_active_slot[team_index];
        } else {
            assigned_slot = first_free_active_slot(slot_used);
        }
        if (assigned_slot <= 0) {
            continue;
        }
        state->self_team[team_index].active = 1;
        state->self_team[team_index].active_slot = assigned_slot;
        slot_used[assigned_slot - 1] = 1;
    }
    refresh_active_counts(state);
    for (i = 0; i < req->active_count && i < 2; ++i) {
        int roster_index = req->active_team_idx[i];
        int team_index = had_prior_active_slot_mapping
            ? state->self_active_slot_to_team_index[i]
            : ((roster_index >= 0 && roster_index < RAW_TEAM_SIZE) ? matched_index[roster_index] : state->self_active_slot_to_team_index[i]);
        RawPokemon* pokemon;
        int m;
        if (team_index < 0 || team_index >= RAW_TEAM_SIZE) {
            continue;
        }
        pokemon = &state->self_team[team_index];
        pokemon->can_tera = req->active[i].can_tera;
        pokemon->fainted = req->active[i].fainted;
        pokemon->trapped = req->active[i].trapped;
        pokemon->maybe_trapped = req->active[i].maybe_trapped;
        if (req->active[i].tera_type_id > 0) {
            tracked_int_promote_confirmed(&pokemon->tera_type_id, req->active[i].tera_type_id);
        }
        if (pokemon->disable_active && pokemon->disable_move_slot < 0) {
            pokemon->disable_move_slot = infer_single_disabled_move_slot(&req->active[i]);
        }
        if (pokemon->encore_active && pokemon->encore_move_slot < 0) {
            pokemon->encore_move_slot = infer_encore_move_slot(&req->active[i]);
        }
        for (m = 0; m < RAW_MOVE_SLOTS; ++m) {
            if (pokemon->transformed) {
                if (req->active[i].move_id[m] > 0) {
                    tracked_int_promote_confirmed(&pokemon->effective_move_ids[m], req->active[i].move_id[m]);
                    pokemon->effective_move_known[m] = 1;
                }
                pokemon->effective_move_pp[m] = req->active[i].move_pp[m];
                pokemon->effective_move_max_pp[m] = req->active[i].move_max_pp[m];
                pokemon->effective_move_disabled[m] = req->active[i].move_disabled[m];
                pokemon->effective_move_maybe_disabled[m] = req->active[i].move_maybe_disabled[m];
            } else {
                if (req->active[i].move_id[m] > 0) {
                    tracked_int_promote_confirmed(&pokemon->move_ids[m], req->active[i].move_id[m]);
                    pokemon->move_known[m] = 1;
                }
                pokemon->move_pp[m] = req->active[i].move_pp[m];
                pokemon->move_max_pp[m] = req->active[i].move_max_pp[m];
                pokemon->move_disabled[m] = req->active[i].move_disabled[m];
                pokemon->move_maybe_disabled[m] = req->active[i].move_maybe_disabled[m];
            }
        }
        raw_pokemon_refresh_types(pokemon);
        if (!pokemon->transformed) {
            raw_pokemon_refresh_effective_state(pokemon);
        }
    }
    refresh_active_counts(state);
    return 1;
}

void raw_battle_state_update_from_event_line(RawBattleState* state, const char* line) {
    event_parser_apply_line(state, line);
}
