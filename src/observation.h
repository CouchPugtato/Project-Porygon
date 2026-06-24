#ifndef OBSERVATION_H
#define OBSERVATION_H

#include <stddef.h>
#include <stdint.h>

#define OBS_TEAM_SIZE 6
#define OBS_MOVE_SLOTS 4
#define OBS_BOOST_SLOTS 7

#define OBS_NUM_SPECIES 1025
#define OBS_NUM_MOVES 920
#define OBS_NUM_ITEMS 400
#define OBS_NUM_ABILITIES 310
#define OBS_NUM_TYPES 19
#define OBS_NUM_STATUS 7
#define OBS_NUM_WEATHER 6
#define OBS_NUM_TERRAIN 5

enum ObsAction {
    OBS_A1_MOVE1,
    OBS_A1_MOVE2,
    OBS_A1_MOVE3,
    OBS_A1_MOVE4,
    OBS_A1_MOVE1_TERA,
    OBS_A1_MOVE2_TERA,
    OBS_A1_MOVE3_TERA,
    OBS_A1_MOVE4_TERA,
    OBS_A1_SWITCH1,
    OBS_A1_SWITCH2,
    OBS_A1_SWITCH3,
    OBS_A1_SWITCH4,
    OBS_A1_SWITCH5,
    OBS_A1_SWITCH6,
    OBS_A2_MOVE1,
    OBS_A2_MOVE2,
    OBS_A2_MOVE3,
    OBS_A2_MOVE4,
    OBS_A2_MOVE1_TERA,
    OBS_A2_MOVE2_TERA,
    OBS_A2_MOVE3_TERA,
    OBS_A2_MOVE4_TERA,
    OBS_A2_SWITCH1,
    OBS_A2_SWITCH2,
    OBS_A2_SWITCH3,
    OBS_A2_SWITCH4,
    OBS_A2_SWITCH5,
    OBS_A2_SWITCH6,

    OBS_NUM_ACTIONS
};

typedef struct {
    uint8_t known;
    uint8_t active;
    uint8_t fainted;
    uint8_t revealed;

    float hp_frac;

    int status_id;
    int type1_id;
    uint8_t type1_known_mode;
    int type2_id;
    uint8_t type2_known_mode;
    int species_id;
    uint8_t species_known_mode;
    int item_id;
    uint8_t item_known_mode;
    int ability_id;
    uint8_t ability_known_mode;
    int tera_type_id;
    uint8_t tera_type_known_mode;

    float boosts[OBS_BOOST_SLOTS];

    uint8_t encore_active;
    float encore_turns;
    uint8_t disable_active;
    float disable_turns;
    uint8_t taunt_active;
    float taunt_turns;
    uint8_t torment_active;
    float torment_turns;
    uint8_t heal_block_active;
    float heal_block_turns;
    uint8_t embargo_active;
    float embargo_turns;
    uint8_t yawn_active;
    float yawn_turns;
    uint8_t protect_active;
    float protect_chain_count;
    uint8_t helping_hand_active;
    uint8_t flinch_active;
    uint8_t confusion_active;
    float confusion_turns;
    uint8_t substitute_active;
    uint8_t trapped;
    uint8_t maybe_trapped;
    uint8_t seed_active;
    float toxic_counter;
    float sleep_turns_elapsed;
    uint8_t transformed;
    float perish_song_counter;
    uint8_t charge_active;
    float charge_turns;

    uint8_t move_known[OBS_MOVE_SLOTS];
    uint8_t move_disabled[OBS_MOVE_SLOTS];
    uint8_t move_known_mode[OBS_MOVE_SLOTS];
    float move_pp_frac[OBS_MOVE_SLOTS];
    int move_id[OBS_MOVE_SLOTS];
} ObsPokemon;

typedef struct {
    uint8_t stealth_rock;
    uint8_t spikes;
    uint8_t toxic_spikes;
    uint8_t sticky_web;
    uint8_t reflect;
    float reflect_turns;
    uint8_t light_screen;
    float light_screen_turns;
    uint8_t aurora_veil;
    float aurora_veil_turns;
    uint8_t tailwind;
    float tailwind_turns;
    uint8_t safeguard;
    float safeguard_turns;
    uint8_t mist;
    float mist_turns;
    uint8_t lucky_chant;
    float lucky_chant_turns;
    uint8_t quick_guard;
    uint8_t wide_guard;
    uint8_t crafty_shield;
    uint8_t mat_block;
} ObsSide;

typedef struct {
    int weather_id;
    int terrain_id;
    float weather_turns;
    uint8_t weather_turns_known_mode;
    float terrain_turns;
    uint8_t terrain_turns_known_mode;
    float turn_norm;

    uint8_t trick_room;
    float trick_room_turns;
    uint8_t forced_switch;
    uint8_t team_preview;
    uint8_t can_tera;
    uint8_t is_doubles;
    uint8_t mud_sport;
    uint8_t water_sport;
    uint8_t ion_deluge;

    ObsSide self_side;
    ObsSide opp_side;

    ObsPokemon self_team[OBS_TEAM_SIZE];
    ObsPokemon opp_team[OBS_TEAM_SIZE];

    uint8_t legal_mask[OBS_NUM_ACTIONS];
} Observation;

#define OBS_POKEMON_FEATURES ( \
    4 + \
    1 + \
    OBS_NUM_STATUS + \
    OBS_NUM_TYPES + \
    3 + \
    OBS_NUM_TYPES + \
    3 + \
    OBS_NUM_SPECIES + \
    3 + \
    OBS_NUM_ITEMS + \
    3 + \
    OBS_NUM_ABILITIES + \
    3 + \
    OBS_NUM_TYPES + \
    3 + \
    OBS_BOOST_SLOTS + \
    30 + \
    OBS_MOVE_SLOTS * (2 + 3 + 1 + OBS_NUM_MOVES) \
)

#define OBS_SIDE_FEATURES 22

#define OBS_GLOBAL_FEATURES ( \
    OBS_NUM_WEATHER + \
    OBS_NUM_TERRAIN + \
    1 + 3 + \
    1 + 3 + \
    1 + \
    1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 \
)

#define OBSERVATION_FLAT_SIZE ( \
    OBS_GLOBAL_FEATURES + \
    (2 * OBS_SIDE_FEATURES) + \
    (2 * OBS_TEAM_SIZE * OBS_POKEMON_FEATURES) + \
    OBS_NUM_ACTIONS \
)

void observation_init(Observation* obs);
size_t observation_flat_size(void);
size_t observation_flatten(float* out, size_t out_len, const Observation* obs);
void observation_fill_demo(Observation* obs);

#endif
