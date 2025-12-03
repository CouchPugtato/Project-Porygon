#ifndef GAME_STATE_H
#define GAME_STATE_H

// Core shared types for battle state representation
enum PokemonType {
    NORMAL,
    FIRE,
    WATER,
    GRASS,
    ELECTRIC,
    ICE,
    FIGHTING,
    POISON,
    GROUND,
    FLYING,
    PSYCHIC,
    BUG,
    ROCK,
    GHOST,
    DRAGON,
    DARK,
    STEEL,
    FAIRY,
    STELLAR,

    NUMBER_OF_TYPES
};

struct Move {
    enum PokemonType type;
    enum {
        PHYSICAL,
        SPECIAL,
        STATUS,

        NUMBER_OF_CATEGORGIES
    } category;
};

struct Pokemon {
    enum PokemonType type[2];
    int current_hp;
    int max_hp;
    int attack;
    int defense;
    int special_attack;
    int special_defense;
    int speed;
    struct Move moves[4];
    int on_field;

    // implement something for held items
    enum {
        BURN,
        PARALYZE,
        POISONED,
        TOXIC,
        FREEZE,
        SLEEP,

        NUMBER_OF_STATUSES
    } status_condition;

    // generation specific mechanics
    enum PokemonType tera_type;
    int is_transformed; // things like dynamax, mega evolution, or terastalization
};

struct BattleState {
    struct Pokemon friendly_pokemon[6];
    struct Pokemon opponent_pokemon[6];

    enum {
        NONE,
        DROUGHT,
        RAIN,
        SANDSTORM,
        SNOW,
        STRONG_WINDS,

        NUMBER_OF_WEATHERS
    } weather;
    int min_turns_of_weather;

    enum {
        NORMAL_TERRAIN,
        ELECTRIC_TERRAIN,
        GRASSY_TERRAIN,
        MISTY_TERRAIN,
        PSYCHIC_TERRAIN,

        NUMBER_OF_TERRAINS
    } terrain;
    int min_turns_of_terrain;

    struct {
        int stealth_rock;
        int spikes;
        int toxic_spikes;
        int sticky_web;
    } friendly_hazards, opponent_hazards;

    int can_generation_mechanic;
};

enum Action {
    ACTIVE_1_MOVE_1,
    ACTIVE_1_MOVE_2,
    ACTIVE_1_MOVE_3,
    ACTIVE_1_MOVE_4,
    ACTIVE_1_SWITCH_1,
    ACTIVE_1_SWITCH_2,
    ACTIVE_1_SWITCH_3,
    ACTIVE_1_SWITCH_4,
    ACTIVE_1_SWITCH_5,
    ACTIVE_1_SWITCH_6,
    ACTIVE_2_MOVE_1,
    ACTIVE_2_MOVE_2,
    ACTIVE_2_MOVE_3,
    ACTIVE_2_MOVE_4,
    ACTIVE_2_SWITCH_1,
    ACTIVE_2_SWITCH_2,
    ACTIVE_2_SWITCH_3,
    ACTIVE_2_SWITCH_4,
    ACTIVE_2_SWITCH_5,
    ACTIVE_2_SWITCH_6,

    NUMBER_OF_ACTIONS
};

// Initialize defaults for BattleState
void battle_state_init(struct BattleState* bs);

// Update friendly side from `|request|{...}` JSON string (minimal parse)
void battle_state_update_from_request(struct BattleState* bs, const char* request_json);

// Update from battle stream line (e.g., "|-weather|", "|-sidestart|", "|-fieldstart|")
void battle_state_update_from_line(struct BattleState* bs, const char* line);

#endif