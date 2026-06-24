#include "observation_builder.h"

#include <string.h>

static uint8_t knowledge_to_mode(KnowledgeLevel level) {
    if (level == KNOW_CONFIRMED) return 2;
    if (level == KNOW_INFERRED) return 1;
    return 0;
}

static float counter_value_for_observation(const TrackedInt* counter) {
    if (!counter || counter->knowledge != KNOW_CONFIRMED) {
        return 0.0f;
    }
    return (float)counter->value;
}

static void copy_side(ObsSide* out, const RawSideState* in) {
    out->stealth_rock = (unsigned char)in->stealth_rock;
    out->spikes = (unsigned char)in->spikes;
    out->toxic_spikes = (unsigned char)in->toxic_spikes;
    out->sticky_web = (unsigned char)in->sticky_web;
    out->reflect = (unsigned char)in->reflect;
    out->reflect_turns = (float)in->reflect_turns;
    out->light_screen = (unsigned char)in->light_screen;
    out->light_screen_turns = (float)in->light_screen_turns;
    out->aurora_veil = (unsigned char)in->aurora_veil;
    out->aurora_veil_turns = (float)in->aurora_veil_turns;
    out->tailwind = (unsigned char)in->tailwind;
    out->tailwind_turns = (float)in->tailwind_turns;
    out->safeguard = (unsigned char)in->safeguard;
    out->safeguard_turns = (float)in->safeguard_turns;
    out->mist = (unsigned char)in->mist;
    out->mist_turns = (float)in->mist_turns;
    out->lucky_chant = (unsigned char)in->lucky_chant;
    out->lucky_chant_turns = (float)in->lucky_chant_turns;
    out->quick_guard = (unsigned char)in->quick_guard;
    out->wide_guard = (unsigned char)in->wide_guard;
    out->crafty_shield = (unsigned char)in->crafty_shield;
    out->mat_block = (unsigned char)in->mat_block;
}

static void copy_pokemon(ObsPokemon* out, const RawPokemon* in) {
    int i;
    out->known = (unsigned char)in->known;
    out->active = (unsigned char)in->active;
    out->fainted = (unsigned char)in->fainted;
    out->revealed = (unsigned char)in->revealed;
    out->hp_frac = (in->max_hp > 0) ? ((float)in->current_hp / (float)in->max_hp) : 0.0f;
    out->status_id = in->status_id.value;
    out->type1_id = in->effective_type1_id.value;
    out->type1_known_mode = knowledge_to_mode(in->effective_type1_id.knowledge);
    out->type2_id = in->effective_type2_id.value;
    out->type2_known_mode = knowledge_to_mode(in->effective_type2_id.knowledge);
    out->species_id = in->effective_species_id.value;
    out->species_known_mode = knowledge_to_mode(in->effective_species_id.knowledge);
    out->item_id = in->item_id.value;
    out->item_known_mode = knowledge_to_mode(in->item_id.knowledge);
    out->ability_id = in->ability_id.value;
    out->ability_known_mode = knowledge_to_mode(in->ability_id.knowledge);
    out->tera_type_id = in->tera_type_id.value;
    out->tera_type_known_mode = knowledge_to_mode(in->tera_type_id.knowledge);
    for (i = 0; i < OBS_BOOST_SLOTS; ++i) {
        out->boosts[i] = (float)in->boosts[i];
    }
    out->encore_active = (unsigned char)in->encore_active;
    out->encore_turns = (float)in->encore_turns;
    out->disable_active = (unsigned char)in->disable_active;
    out->disable_turns = (float)in->disable_turns;
    out->taunt_active = (unsigned char)in->taunt_active;
    out->taunt_turns = (float)in->taunt_turns;
    out->torment_active = (unsigned char)in->torment_active;
    out->torment_turns = (float)in->torment_turns;
    out->heal_block_active = (unsigned char)in->heal_block_active;
    out->heal_block_turns = (float)in->heal_block_turns;
    out->embargo_active = (unsigned char)in->embargo_active;
    out->embargo_turns = (float)in->embargo_turns;
    out->yawn_active = (unsigned char)in->yawn_active;
    out->yawn_turns = (float)in->yawn_turns;
    out->encore_move_slot = in->encore_move_slot;
    out->disable_move_slot = in->disable_move_slot;
    out->protect_active = (unsigned char)in->protect_active;
    out->protect_chain_count = (float)in->protect_chain_count;
    out->helping_hand_active = (unsigned char)in->helping_hand_active;
    out->flinch_active = (unsigned char)in->flinch_active;
    out->confusion_active = (unsigned char)in->confusion_active;
    out->confusion_turns = (float)in->confusion_turns;
    out->substitute_active = (unsigned char)in->substitute_active;
    out->trapped = (unsigned char)in->trapped;
    out->maybe_trapped = (unsigned char)in->maybe_trapped;
    out->seed_active = (unsigned char)in->seed_active;
    out->toxic_counter = (float)in->toxic_counter;
    out->sleep_turns_elapsed = (float)in->sleep_turns_elapsed;
    out->transformed = (unsigned char)in->transformed;
    out->perish_song_counter = (float)in->perish_song_counter;
    out->charge_active = (unsigned char)in->charge_active;
    out->charge_turns = (float)in->charge_turns;
    for (i = 0; i < OBS_MOVE_SLOTS; ++i) {
        out->move_known[i] = (unsigned char)in->effective_move_known[i];
        out->move_disabled[i] = (unsigned char)in->effective_move_disabled[i];
        out->move_known_mode[i] = knowledge_to_mode(in->effective_move_ids[i].knowledge);
        out->move_pp_frac[i] = in->effective_move_max_pp[i] > 0 ? ((float)in->effective_move_pp[i] / (float)in->effective_move_max_pp[i]) : 0.0f;
        out->move_id[i] = in->effective_move_ids[i].value;
        out->move_type_id[i] = in->effective_move_type_ids[i].value;
    }
}

void observation_from_raw_state(
    Observation* out,
    const RawBattleState* state,
    const ParsedRequest* req,
    const ActionMask* mask
) {
    int i;
    if (!out || !state) {
        return;
    }
    observation_init(out);
    out->weather_id = state->weather_id;
    out->terrain_id = state->terrain_id;
    /* Weather/terrain remaining turns are only surfaced numerically when
       confirmed. Inferred counters stay visible through knowledge mode but do
       not masquerade as exact truth in model features. */
    out->weather_turns = counter_value_for_observation(&state->weather_turns_remaining);
    out->weather_turns_known_mode = knowledge_to_mode(state->weather_turns_remaining.knowledge);
    out->terrain_turns = counter_value_for_observation(&state->terrain_turns_remaining);
    out->terrain_turns_known_mode = knowledge_to_mode(state->terrain_turns_remaining.knowledge);
    out->turn_norm = (float)state->turn_number / 100.0f;
    out->trick_room = (unsigned char)state->trick_room;
    out->trick_room_turns = (float)state->trick_room_turns_remaining;
    out->magic_room = (unsigned char)state->magic_room;
    out->magic_room_turns = (float)state->magic_room_turns_remaining;
    out->wonder_room = (unsigned char)state->wonder_room;
    out->wonder_room_turns = (float)state->wonder_room_turns_remaining;
    out->gravity = (unsigned char)state->gravity;
    out->gravity_turns = (float)state->gravity_turns_remaining;
    out->forced_switch = (unsigned char)((req && req->forced_switch_any) ? 1 : 0);
    out->team_preview = (unsigned char)((req && req->team_preview) ? 1 : 0);
    out->can_tera = (unsigned char)state->can_tera;
    out->is_doubles = (unsigned char)state->is_doubles;
    out->mud_sport = (unsigned char)state->mud_sport;
    out->water_sport = (unsigned char)state->water_sport;
    out->ion_deluge = (unsigned char)state->ion_deluge;

    copy_side(&out->self_side, &state->self_side);
    copy_side(&out->opp_side, &state->opp_side);

    for (i = 0; i < OBS_TEAM_SIZE; ++i) {
        copy_pokemon(&out->self_team[i], &state->self_team[i]);
        copy_pokemon(&out->opp_team[i], &state->opp_team[i]);
    }
    if (mask) {
        memcpy(out->legal_mask, mask->legal, sizeof(out->legal_mask));
    }
}
