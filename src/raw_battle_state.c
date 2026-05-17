#include "raw_battle_state.h"

#include "event_parser.h"

#include <string.h>

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
    tracked_int_reset(&pokemon->status_id);
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        tracked_int_reset(&pokemon->move_ids[i]);
    }
    pokemon->encore_move_slot = -1;
    pokemon->disable_move_slot = -1;
}

static void raw_side_init(RawSideState* side) {
    if (!side) {
        return;
    }
    memset(side, 0, sizeof(*side));
    side->remaining_pokemon = RAW_TEAM_SIZE;
}

static void clear_per_turn_flags(RawPokemon* pokemon) {
    if (!pokemon) {
        return;
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
    pokemon->ability_triggered_on_switch_in = 0;
}

void raw_battle_state_init(RawBattleState* state, int is_doubles) {
    int i;
    if (!state) {
        return;
    }
    memset(state, 0, sizeof(*state));
    state->is_doubles = is_doubles ? 1 : 0;
    raw_side_init(&state->self_side);
    raw_side_init(&state->opp_side);
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

void raw_battle_state_end_turn(RawBattleState* state) {
    int i;
    if (!state) {
        return;
    }
    decrement_duration(&state->trick_room, &state->trick_room_turns_remaining);
    decrement_duration(&state->magic_room, &state->magic_room_turns_remaining);
    decrement_duration(&state->wonder_room, &state->wonder_room_turns_remaining);
    decrement_duration(&state->gravity, &state->gravity_turns_remaining);
    if (state->weather_turns_remaining > 0) {
        --state->weather_turns_remaining;
    }
    if (state->terrain_turns_remaining > 0) {
        --state->terrain_turns_remaining;
    }
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
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

void raw_battle_state_update_from_request(RawBattleState* state, const ParsedRequest* req) {
    int i;
    if (!state || !req) {
        return;
    }
    state->turn_number = req->request_id > 0 ? req->request_id : state->turn_number;
    state->can_tera = req->can_tera;
    state->self_active_count = req->active_count;
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        clear_on_switch(&state->self_team[i]);
    }
    for (i = 0; i < req->active_count && i < RAW_TEAM_SIZE; ++i) {
        int m;
        RawPokemon* pokemon = &state->self_team[i];
        pokemon->known = 1;
        pokemon->active = 1;
        pokemon->active_slot = i + 1;
        pokemon->revealed = 1;
        pokemon->can_tera = req->active[i].can_tera;
        pokemon->fainted = req->active[i].fainted;
        if (req->active[i].tera_type_id > 0) {
            tracked_int_promote_confirmed(&pokemon->tera_type_id, req->active[i].tera_type_id);
        }
        for (m = 0; m < RAW_MOVE_SLOTS; ++m) {
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
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        state->self_team[i].fainted = req->switch_fainted[i];
        if (req->side_species_id[i] > 0) {
            tracked_int_promote_confirmed(&state->self_team[i].species_id, req->side_species_id[i]);
        }
        if (req->side_ident[i][0]) {
            strncpy(state->self_team[i].ident, req->side_ident[i], RAW_IDENT_LEN - 1);
        }
    }
}

void raw_battle_state_update_from_event_line(RawBattleState* state, const char* line) {
    event_parser_apply_line(state, line);
}
