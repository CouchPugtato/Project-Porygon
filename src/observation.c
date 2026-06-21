#include "observation.h"
#include "id_tables.h"

#include <string.h>

static size_t write_flag(float* out, size_t idx, uint8_t value) {
    out[idx] = value ? 1.0f : 0.0f;
    return idx + 1;
}

static size_t write_scalar(float* out, size_t idx, float value) {
    out[idx] = value;
    return idx + 1;
}

static size_t write_one_hot(float* out, size_t idx, int value, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        out[idx + (size_t)i] = 0.0f;
    }
    if (value >= 0 && value < count) {
        out[idx + (size_t)value] = 1.0f;
    }
    return idx + (size_t)count;
}

static size_t write_knowledge(float* out, size_t idx, uint8_t mode) {
    int mapped = (mode <= 2) ? mode : 0;
    return write_one_hot(out, idx, mapped, 3);
}

static size_t flatten_side(float* out, size_t idx, const ObsSide* side) {
    idx = write_flag(out, idx, side->stealth_rock);
    idx = write_scalar(out, idx, (float)side->spikes / 3.0f);
    idx = write_scalar(out, idx, (float)side->toxic_spikes / 2.0f);
    idx = write_flag(out, idx, side->sticky_web);
    idx = write_flag(out, idx, side->reflect);
    idx = write_scalar(out, idx, side->reflect_turns / 8.0f);
    idx = write_flag(out, idx, side->light_screen);
    idx = write_scalar(out, idx, side->light_screen_turns / 8.0f);
    idx = write_flag(out, idx, side->aurora_veil);
    idx = write_scalar(out, idx, side->aurora_veil_turns / 8.0f);
    idx = write_flag(out, idx, side->tailwind);
    idx = write_scalar(out, idx, side->tailwind_turns / 8.0f);
    idx = write_flag(out, idx, side->safeguard);
    idx = write_scalar(out, idx, side->safeguard_turns / 8.0f);
    idx = write_flag(out, idx, side->mist);
    idx = write_scalar(out, idx, side->mist_turns / 8.0f);
    idx = write_flag(out, idx, side->lucky_chant);
    idx = write_scalar(out, idx, side->lucky_chant_turns / 8.0f);
    idx = write_flag(out, idx, side->quick_guard);
    idx = write_flag(out, idx, side->wide_guard);
    idx = write_flag(out, idx, side->crafty_shield);
    idx = write_flag(out, idx, side->mat_block);
    return idx;
}

static size_t flatten_pokemon(float* out, size_t idx, const ObsPokemon* p) {
    int i;

    idx = write_flag(out, idx, p->known);
    idx = write_flag(out, idx, p->active);
    idx = write_flag(out, idx, p->fainted);
    idx = write_flag(out, idx, p->revealed);
    idx = write_scalar(out, idx, p->hp_frac);

    idx = write_one_hot(out, idx, p->status_id, OBS_NUM_STATUS);
    idx = write_one_hot(out, idx, p->type1_id, OBS_NUM_TYPES);
    idx = write_knowledge(out, idx, p->type1_known_mode);
    idx = write_one_hot(out, idx, p->type2_id, OBS_NUM_TYPES);
    idx = write_knowledge(out, idx, p->type2_known_mode);
    idx = write_one_hot(out, idx, p->species_id, OBS_NUM_SPECIES);
    idx = write_knowledge(out, idx, p->species_known_mode);
    idx = write_one_hot(out, idx, p->item_id, OBS_NUM_ITEMS);
    idx = write_knowledge(out, idx, p->item_known_mode);
    idx = write_one_hot(out, idx, p->ability_id, OBS_NUM_ABILITIES);
    idx = write_knowledge(out, idx, p->ability_known_mode);
    idx = write_one_hot(out, idx, p->tera_type_id, OBS_NUM_TYPES);
    idx = write_knowledge(out, idx, p->tera_type_known_mode);

    for (i = 0; i < OBS_BOOST_SLOTS; ++i) {
        idx = write_scalar(out, idx, p->boosts[i] / 6.0f);
    }
    idx = write_flag(out, idx, p->encore_active);
    idx = write_scalar(out, idx, p->encore_turns / 5.0f);
    idx = write_flag(out, idx, p->disable_active);
    idx = write_scalar(out, idx, p->disable_turns / 5.0f);
    idx = write_flag(out, idx, p->taunt_active);
    idx = write_scalar(out, idx, p->taunt_turns / 5.0f);
    idx = write_flag(out, idx, p->protect_active);
    idx = write_scalar(out, idx, p->protect_chain_count / 5.0f);
    idx = write_flag(out, idx, p->confusion_active);
    idx = write_scalar(out, idx, p->confusion_turns / 5.0f);
    idx = write_flag(out, idx, p->substitute_active);
    idx = write_scalar(out, idx, p->toxic_counter / 15.0f);
    idx = write_scalar(out, idx, p->sleep_turns_elapsed / 5.0f);
    idx = write_flag(out, idx, p->transformed);
    idx = write_scalar(out, idx, p->perish_song_counter / 4.0f);

    for (i = 0; i < OBS_MOVE_SLOTS; ++i) {
        idx = write_flag(out, idx, p->move_known[i]);
        idx = write_flag(out, idx, p->move_disabled[i]);
        idx = write_knowledge(out, idx, p->move_known_mode[i]);
        idx = write_scalar(out, idx, p->move_pp_frac[i]);
        idx = write_one_hot(out, idx, p->move_id[i], OBS_NUM_MOVES);
    }

    return idx;
}

void observation_init(Observation* obs) {
    size_t i;

    if (!obs) {
        return;
    }
    memset(obs, 0, sizeof(*obs));
    obs->weather_id = 0;
    obs->terrain_id = 0;

    for (i = 0; i < OBS_TEAM_SIZE; ++i) {
        size_t m;
        obs->self_team[i].status_id = 0;
        obs->self_team[i].type1_id = 0;
        obs->self_team[i].type2_id = 0;
        obs->self_team[i].species_id = 0;
        obs->self_team[i].item_id = 0;
        obs->self_team[i].ability_id = 0;
        obs->self_team[i].tera_type_id = 0;
        obs->opp_team[i].status_id = 0;
        obs->opp_team[i].type1_id = 0;
        obs->opp_team[i].type2_id = 0;
        obs->opp_team[i].species_id = 0;
        obs->opp_team[i].item_id = 0;
        obs->opp_team[i].ability_id = 0;
        obs->opp_team[i].tera_type_id = 0;
        for (m = 0; m < OBS_MOVE_SLOTS; ++m) {
            obs->self_team[i].move_id[m] = 0;
            obs->opp_team[i].move_id[m] = 0;
        }
    }
}

size_t observation_flat_size(void) {
    return OBSERVATION_FLAT_SIZE;
}

size_t observation_flatten(float* out, size_t out_len, const Observation* obs) {
    size_t idx = 0;
    size_t i;

    if (!out || !obs || out_len < OBSERVATION_FLAT_SIZE) {
        return 0;
    }

    idx = write_one_hot(out, idx, obs->weather_id, OBS_NUM_WEATHER);
    idx = write_one_hot(out, idx, obs->terrain_id, OBS_NUM_TERRAIN);
    idx = write_scalar(out, idx, obs->weather_turns);
    idx = write_knowledge(out, idx, obs->weather_turns_known_mode);
    idx = write_scalar(out, idx, obs->terrain_turns);
    idx = write_knowledge(out, idx, obs->terrain_turns_known_mode);
    idx = write_scalar(out, idx, obs->turn_norm);

    idx = write_flag(out, idx, obs->trick_room);
    idx = write_scalar(out, idx, obs->trick_room_turns / 8.0f);
    idx = write_flag(out, idx, obs->forced_switch);
    idx = write_flag(out, idx, obs->team_preview);
    idx = write_flag(out, idx, obs->can_tera);
    idx = write_flag(out, idx, obs->is_doubles);
    idx = write_flag(out, idx, obs->mud_sport);
    idx = write_flag(out, idx, obs->water_sport);
    idx = write_flag(out, idx, obs->ion_deluge);

    idx = flatten_side(out, idx, &obs->self_side);
    idx = flatten_side(out, idx, &obs->opp_side);

    for (i = 0; i < OBS_TEAM_SIZE; ++i) {
        idx = flatten_pokemon(out, idx, &obs->self_team[i]);
    }
    for (i = 0; i < OBS_TEAM_SIZE; ++i) {
        idx = flatten_pokemon(out, idx, &obs->opp_team[i]);
    }
    for (i = 0; i < OBS_NUM_ACTIONS; ++i) {
        idx = write_flag(out, idx, obs->legal_mask[i]);
    }
    return idx == OBSERVATION_FLAT_SIZE ? idx : 0;
}

void observation_fill_demo(Observation* obs) {
    if (!obs) {
        return;
    }

    observation_init(obs);
    obs->weather_id = 2;
    obs->terrain_id = 4;
    obs->weather_turns = 0.6f;
    obs->terrain_turns = 0.4f;
    obs->turn_norm = 0.1f;
    obs->trick_room = 0;
    obs->forced_switch = 0;
    obs->team_preview = 0;
    obs->can_tera = 1;
    obs->is_doubles = 1;

    obs->self_side.stealth_rock = 1;
    obs->opp_side.tailwind = 1;

    obs->self_team[0].known = 1;
    obs->self_team[0].active = 1;
    obs->self_team[0].revealed = 1;
    obs->self_team[0].hp_frac = 0.82f;
    obs->self_team[0].status_id = 0;
    obs->self_team[0].type1_id = 3;
    obs->self_team[0].type2_id = 8;
    obs->self_team[0].species_id = species_id_from_name("garchomp");
    obs->self_team[0].item_id = item_id_from_name("clearamulet");
    obs->self_team[0].ability_id = ability_id_from_name("roughskin");
    obs->self_team[0].tera_type_id = 2;
    obs->self_team[0].move_known[0] = 1;
    obs->self_team[0].move_known[1] = 1;
    obs->self_team[0].move_known[2] = 1;
    obs->self_team[0].move_known[3] = 1;
    obs->self_team[0].move_pp_frac[0] = 0.75f;
    obs->self_team[0].move_pp_frac[1] = 0.80f;
    obs->self_team[0].move_pp_frac[2] = 1.00f;
    obs->self_team[0].move_pp_frac[3] = 0.60f;
    obs->self_team[0].move_id[0] = move_id_from_name("protect");
    obs->self_team[0].move_id[1] = move_id_from_name("earthquake");
    obs->self_team[0].move_id[2] = move_id_from_name("rockslide");
    obs->self_team[0].move_id[3] = move_id_from_name("dragonclaw");

    obs->self_team[1].known = 1;
    obs->self_team[1].revealed = 1;
    obs->self_team[1].hp_frac = 1.00f;
    obs->self_team[1].type1_id = 11;
    obs->self_team[1].species_id = species_id_from_name("amoonguss");

    obs->opp_team[0].known = 1;
    obs->opp_team[0].active = 1;
    obs->opp_team[0].revealed = 1;
    obs->opp_team[0].hp_frac = 0.66f;
    obs->opp_team[0].status_id = 2;
    obs->opp_team[0].type1_id = 10;
    obs->opp_team[0].species_id = species_id_from_name("gyarados");
    obs->opp_team[0].move_known[0] = 1;
    obs->opp_team[0].move_pp_frac[0] = 0.9f;
    obs->opp_team[0].move_id[0] = move_id_from_name("thunderbolt");

    obs->legal_mask[OBS_A1_MOVE1] = 1;
    obs->legal_mask[OBS_A1_MOVE2] = 1;
    obs->legal_mask[OBS_A1_MOVE3] = 1;
    obs->legal_mask[OBS_A1_MOVE4] = 1;
    obs->legal_mask[OBS_A1_MOVE1_TERA] = 1;
    obs->legal_mask[OBS_A1_SWITCH2] = 1;
    obs->legal_mask[OBS_A2_MOVE1] = 1;
    obs->legal_mask[OBS_A2_SWITCH3] = 1;
}
