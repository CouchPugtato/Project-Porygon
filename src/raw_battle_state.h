#ifndef RAW_BATTLE_STATE_H
#define RAW_BATTLE_STATE_H

#include "request_parser.h"

#define RAW_TEAM_SIZE 6
#define RAW_MOVE_SLOTS 4
#define RAW_IDENT_LEN 32

typedef struct {
    int known;
    int active;
    int active_slot;
    int revealed;
    int fainted;

    int species_id;
    int item_id;
    int ability_id;
    int tera_type_id;
    int tera_used;
    int can_tera;

    int current_hp;
    int max_hp;
    int status_id;
    int type1_id;
    int type2_id;
    int boosts[7];

    int move_ids[RAW_MOVE_SLOTS];
    int move_known[RAW_MOVE_SLOTS];
    int move_pp[RAW_MOVE_SLOTS];
    int move_max_pp[RAW_MOVE_SLOTS];
    int move_disabled[RAW_MOVE_SLOTS];

    char ident[RAW_IDENT_LEN];
} RawPokemon;

typedef struct {
    int stealth_rock;
    int spikes;
    int toxic_spikes;
    int sticky_web;
    int reflect;
    int light_screen;
    int aurora_veil;
    int tailwind;
    int remaining_pokemon;
} RawSideState;

typedef struct {
    RawPokemon self_team[RAW_TEAM_SIZE];
    RawPokemon opp_team[RAW_TEAM_SIZE];
    RawSideState self_side;
    RawSideState opp_side;
    int weather_id;
    int weather_turns;
    int terrain_id;
    int terrain_turns;
    int trick_room;
    int trick_room_turns;
    int turn_number;
    int can_tera;
    int is_doubles;
} RawBattleState;

void raw_battle_state_init(RawBattleState* state, int is_doubles);
void raw_battle_state_reset(RawBattleState* state);
void raw_battle_state_update_from_request(RawBattleState* state, const ParsedRequest* req);
void raw_battle_state_update_from_event_line(RawBattleState* state, const char* line);

#endif
