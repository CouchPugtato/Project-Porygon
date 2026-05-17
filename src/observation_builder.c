#include "observation_builder.h"

#include <string.h>

static void copy_side(ObsSide* out, const RawSideState* in) {
    out->stealth_rock = (unsigned char)in->stealth_rock;
    out->spikes = (unsigned char)in->spikes;
    out->toxic_spikes = (unsigned char)in->toxic_spikes;
    out->sticky_web = (unsigned char)in->sticky_web;
    out->reflect = (unsigned char)in->reflect;
    out->light_screen = (unsigned char)in->light_screen;
    out->aurora_veil = (unsigned char)in->aurora_veil;
    out->tailwind = (unsigned char)in->tailwind;
}

static void copy_pokemon(ObsPokemon* out, const RawPokemon* in) {
    int i;
    out->known = (unsigned char)in->known;
    out->active = (unsigned char)in->active;
    out->fainted = (unsigned char)in->fainted;
    out->revealed = (unsigned char)in->revealed;
    out->hp_frac = (in->max_hp > 0) ? ((float)in->current_hp / (float)in->max_hp) : 0.0f;
    out->status_id = in->status_id;
    out->type1_id = in->type1_id;
    out->type2_id = in->type2_id;
    out->species_id = in->species_id;
    out->item_id = in->item_id;
    out->ability_id = in->ability_id;
    out->tera_type_id = in->tera_type_id;
    for (i = 0; i < OBS_BOOST_SLOTS; ++i) {
        out->boosts[i] = (float)in->boosts[i];
    }
    for (i = 0; i < OBS_MOVE_SLOTS; ++i) {
        out->move_known[i] = (unsigned char)in->move_known[i];
        out->move_disabled[i] = (unsigned char)in->move_disabled[i];
        out->move_pp_frac[i] = in->move_max_pp[i] > 0 ? ((float)in->move_pp[i] / (float)in->move_max_pp[i]) : 0.0f;
        out->move_id[i] = in->move_ids[i];
    }
}

void observation_from_raw_state(
    Observation* out,
    const RawBattleState* state,
    const ActionMask* mask
) {
    int i;
    if (!out || !state) {
        return;
    }
    observation_init(out);
    out->weather_id = state->weather_id;
    out->terrain_id = state->terrain_id;
    out->weather_turns = (float)state->weather_turns;
    out->terrain_turns = (float)state->terrain_turns;
    out->turn_norm = (float)state->turn_number / 100.0f;
    out->trick_room = (unsigned char)state->trick_room;
    out->forced_switch = 0;
    out->team_preview = 0;
    out->can_tera = (unsigned char)state->can_tera;
    out->is_doubles = (unsigned char)state->is_doubles;

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
