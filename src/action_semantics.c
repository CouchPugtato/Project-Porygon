#include "action_semantics.h"

#include "id_tables.h"

#include <string.h>

enum {
    SEM_HAS_ACTION = 0,
    SEM_KIND = 1,
    SEM_MOVE_SLOT = 3,
    SEM_MOVE_TYPE = 7,
    SEM_MOVE_TARGET = 26,
    SEM_MOVE_PP = 37,
    SEM_TERA = 38,
    SEM_SELECTED_TARGET = 39,
    SEM_SWITCH_SLOT = 43,
    SEM_SWITCH_TYPE1 = 49,
    SEM_SWITCH_TYPE2 = 68,
    SEM_SWITCH_STATUS = 87,
    SEM_SWITCH_HP = 94,
    SEM_SWITCH_STATS = 95,
    SEM_MOVE_BASE_POWER = 101,
    SEM_MOVE_ACCURACY = 102,
    SEM_MOVE_CATEGORY = 103,
    SEM_MOVE_PRIORITY = 106,
    SEM_MOVE_ROLES = 107
};

static void set_one_hot(float* values, size_t start, size_t count, int index) {
    if (index >= 0 && (size_t)index < count) values[start + (size_t)index] = 1.0f;
}

static float normalized_stat(int value) {
    if (value <= 0) return 0.0f;
    if (value >= 1000) return 1.0f;
    return (float)value / 1000.0f;
}

static float clamp_ratio(int value, int denominator) {
    if (value <= -denominator) return -1.0f;
    if (value >= denominator) return 1.0f;
    return (float)value / (float)denominator;
}

static void write_move_metadata(float* values, int move_id) {
    MoveSemanticMetadata metadata;
    unsigned char roles[15];
    size_t i;
    if (!move_semantics_from_id(move_id, &metadata)) return;
    roles[0] = metadata.spread;
    roles[1] = metadata.healing;
    roles[2] = metadata.protection;
    roles[3] = metadata.setup;
    roles[4] = metadata.redirection;
    roles[5] = metadata.damaging;
    roles[6] = metadata.recoil;
    roles[7] = metadata.drain;
    roles[8] = metadata.self_switch;
    roles[9] = metadata.force_switch;
    roles[10] = metadata.status_effect;
    roles[11] = metadata.weather;
    roles[12] = metadata.terrain;
    roles[13] = metadata.field_effect;
    roles[14] = metadata.ally_support;
    values[SEM_MOVE_BASE_POWER] = clamp_ratio(metadata.base_power, 250);
    values[SEM_MOVE_ACCURACY] = clamp_ratio(metadata.accuracy, 100);
    set_one_hot(values, SEM_MOVE_CATEGORY, 3u, metadata.category - 1);
    values[SEM_MOVE_PRIORITY] = clamp_ratio(metadata.priority, 7);
    for (i = 0; i < sizeof(roles) / sizeof(roles[0]); ++i) {
        values[SEM_MOVE_ROLES + i] = roles[i] ? 1.0f : 0.0f;
    }
}

static void build_slot(
    float* values,
    const Observation* observation,
    const ParsedRequest* request,
    const FactorizedActionChoice* choice,
    int slot
) {
    int has_action = slot == 0 ? choice->slot0_has_action : choice->slot1_has_action;
    int kind = slot == 0 ? choice->slot0_kind : choice->slot1_kind;
    int move_index = slot == 0 ? choice->slot0_move_index : choice->slot1_move_index;
    int switch_index = slot == 0 ? choice->slot0_switch_index : choice->slot1_switch_index;
    int use_tera = slot == 0 ? choice->slot0_use_tera : choice->slot1_use_tera;
    int target_index = slot == 0 ? choice->slot0_target_index : choice->slot1_target_index;

    if (!has_action) return;
    values[SEM_HAS_ACTION] = 1.0f;
    if (kind == FACTORIZED_ACTION_MOVE) {
        const ParsedActive* active;
        int move_id;
        int move_type;
        if (slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS ||
                move_index < 0 || move_index >= PARSED_REQUEST_MOVE_SLOTS) return;
        active = &request->active[slot];
        move_id = active->move_id[move_index];
        move_type = move_type_from_id(move_id);
        values[SEM_KIND] = 1.0f;
        set_one_hot(values, SEM_MOVE_SLOT, PARSED_REQUEST_MOVE_SLOTS, move_index);
        set_one_hot(values, SEM_MOVE_TYPE, OBS_NUM_TYPES, move_type);
        /* allAdjacent was previously parsed as UNKNOWN. Keep its existing
         * neural input encoding while exposing the distinction to the teacher. */
        set_one_hot(values, SEM_MOVE_TARGET, 11u,
            active->move_target[move_index] == REQUEST_TARGET_ALL_ADJACENT
                ? REQUEST_TARGET_UNKNOWN : active->move_target[move_index]);
        if (active->move_max_pp[move_index] > 0) {
            values[SEM_MOVE_PP] =
                (float)active->move_pp[move_index] / (float)active->move_max_pp[move_index];
        }
        values[SEM_TERA] = use_tera ? 1.0f : 0.0f;
        set_one_hot(values, SEM_SELECTED_TARGET, FACTORIZED_TARGET_DIM, target_index);
        write_move_metadata(values, move_id);
        return;
    }
    if (kind == FACTORIZED_ACTION_SWITCH && switch_index >= 0 &&
            switch_index < PARSED_REQUEST_TEAM_SIZE) {
        const ObsPokemon* pokemon = &observation->self_team[switch_index];
        const int stats[6] = {
            request->side_stats_hp[switch_index],
            request->side_stats_atk[switch_index],
            request->side_stats_def[switch_index],
            request->side_stats_spa[switch_index],
            request->side_stats_spd[switch_index],
            request->side_stats_spe[switch_index]
        };
        size_t i;
        values[SEM_KIND + 1u] = 1.0f;
        set_one_hot(values, SEM_SWITCH_SLOT, PARSED_REQUEST_TEAM_SIZE, switch_index);
        set_one_hot(values, SEM_SWITCH_TYPE1, OBS_NUM_TYPES, pokemon->type1_id);
        set_one_hot(values, SEM_SWITCH_TYPE2, OBS_NUM_TYPES, pokemon->type2_id);
        set_one_hot(values, SEM_SWITCH_STATUS, OBS_NUM_STATUS, pokemon->status_id);
        values[SEM_SWITCH_HP] = pokemon->hp_frac;
        for (i = 0; i < 6u; ++i) {
            values[SEM_SWITCH_STATS + i] = normalized_stat(stats[i]);
        }
    }
}

int action_semantics_build(
    float* output,
    size_t output_count,
    const Observation* observation,
    const ParsedRequest* request,
    const FactorizedActionChoice* choice
) {
    if (!output || output_count != ACTION_SEMANTIC_DIM || !observation ||
            !request || !choice) return 0;
    memset(output, 0, output_count * sizeof(*output));
    build_slot(output, observation, request, choice, 0);
    build_slot(output + ACTION_SEMANTIC_SLOT_DIM,
        observation, request, choice, 1);
    return 1;
}
