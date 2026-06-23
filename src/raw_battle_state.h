#ifndef RAW_BATTLE_STATE_H
#define RAW_BATTLE_STATE_H

#include "request_parser.h"
#include "state_value.h"

#define RAW_TEAM_SIZE 6
#define RAW_MOVE_SLOTS 4
#define RAW_IDENT_LEN 32

typedef struct {
    /* Base identity/request-visible state. This must never be mutated by temporary
       battle-only effects like Transform. */
    int known;
    int active;
    int active_slot;
    int revealed;
    int fainted;

    TrackedInt species_id;
    TrackedInt item_id;
    TrackedInt ability_id;
    TrackedInt tera_type_id;
    TrackedInt type1_id;
    TrackedInt type2_id;
    /* Effective battle state. This is what the model sees and may diverge from
       base state during temporary effects like Transform or Terastallization. */
    TrackedInt effective_species_id;
    TrackedInt effective_type1_id;
    TrackedInt effective_type2_id;
    int base_hp_stat;
    int base_atk_stat;
    int base_def_stat;
    int base_spa_stat;
    int base_spd_stat;
    int base_spe_stat;
    int tera_used;
    int can_tera;
    int transformed;
    int substitute_active;
    int trapped;
    int maybe_trapped;
    int commanding_active;
    int reviving;

    int current_hp;
    int max_hp;
    TrackedInt status_id;
    int sleep_turns_elapsed;
    int toxic_counter;
    int boosts[7];

    TrackedInt move_ids[RAW_MOVE_SLOTS];
    TrackedInt effective_move_ids[RAW_MOVE_SLOTS];
    int move_known[RAW_MOVE_SLOTS];
    int effective_move_known[RAW_MOVE_SLOTS];
    int move_pp[RAW_MOVE_SLOTS];
    int effective_move_pp[RAW_MOVE_SLOTS];
    int move_max_pp[RAW_MOVE_SLOTS];
    int effective_move_max_pp[RAW_MOVE_SLOTS];
    int move_disabled[RAW_MOVE_SLOTS];
    int effective_move_disabled[RAW_MOVE_SLOTS];
    int move_maybe_disabled[RAW_MOVE_SLOTS];
    int effective_move_maybe_disabled[RAW_MOVE_SLOTS];

    int encore_active;
    int encore_turns;
    int disable_active;
    int disable_turns;
    int taunt_active;
    int taunt_turns;
    int torment_active;
    int torment_turns;
    int heal_block_active;
    int heal_block_turns;
    int embargo_active;
    int embargo_turns;
    int yawn_active;
    int yawn_turns;
    int encore_move_slot;
    int disable_move_slot;

    int protect_active;
    int protect_chain_count;
    int helping_hand_active;
    int flinch_active;
    int confusion_active;
    int confusion_turns;
    int seed_active;
    int perish_song_counter;
    int charge_active;
    int charge_turns;

    int last_move_id;
    int last_move_turn;
    int switched_in_turn;
    int first_turn_on_field;
    int ability_triggered_on_switch_in;
    int self_request_roster_index;

    char ident[RAW_IDENT_LEN];
    char canonical_ident[RAW_IDENT_LEN];
} RawPokemon;

typedef struct {
    int stealth_rock;
    int spikes;
    int toxic_spikes;
    int sticky_web;

    int reflect;
    int reflect_turns;
    int light_screen;
    int light_screen_turns;
    int aurora_veil;
    int aurora_veil_turns;
    int tailwind;
    int tailwind_turns;
    int safeguard;
    int safeguard_turns;
    int mist;
    int mist_turns;
    int lucky_chant;
    int lucky_chant_turns;
    int quick_guard;
    int wide_guard;
    int crafty_shield;
    int mat_block;

    int remaining_pokemon;
} RawSideState;

typedef struct {
    RawPokemon self_team[RAW_TEAM_SIZE];
    RawPokemon opp_team[RAW_TEAM_SIZE];
    RawSideState self_side;
    RawSideState opp_side;

    int weather_id;
    /* Remaining turn counters for weather/terrain are inferred estimates from
       public protocol starts and ends, not exact hidden-info reconstruction. */
    TrackedInt weather_turns_remaining;
    int terrain_id;
    TrackedInt terrain_turns_remaining;

    int trick_room;
    int trick_room_turns_remaining;
    int magic_room;
    int magic_room_turns_remaining;
    int wonder_room;
    int wonder_room_turns_remaining;
    int gravity;
    int gravity_turns_remaining;

    int mud_sport;
    int water_sport;
    int ion_deluge;

    int turn_number;
    int can_tera;
    int is_doubles;
    int self_side_player;
    int perspective_known;
    int self_active_count;
    int opp_active_count;
    int self_active_slot_to_team_index[2];
    int opp_active_slot_to_team_index[2];
} RawBattleState;

void raw_battle_state_init(RawBattleState* state, int is_doubles);
void raw_battle_state_reset(RawBattleState* state);
int raw_battle_state_update_from_request(RawBattleState* state, const ParsedRequest* req);
void raw_battle_state_update_from_event_line(RawBattleState* state, const char* line);
void raw_battle_state_begin_turn(RawBattleState* state, int turn_number);
void raw_battle_state_end_turn(RawBattleState* state);
void raw_pokemon_refresh_types(RawPokemon* pokemon);
void raw_pokemon_refresh_effective_state(RawPokemon* pokemon);
void raw_pokemon_clear_transform(RawPokemon* pokemon);
void raw_pokemon_apply_transform(RawPokemon* pokemon, const RawPokemon* target);

#endif
