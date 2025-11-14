#include <math.h>
#include <stdlib.h>

// Pokemon related
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
	
	// Implement something for held items
	enum {
		BURN,
		PARALYZE,
		POISONED,
		TOXIC,
		FREEZE,
		SLEEP,
	
		NUMBER_OF_STATUSES
	} status_condition;

	// Generation specific mechanics
	enum PokemonType tera_type;	
	int is_transformed; // Things like dynamax, mega evolution, or terastalization
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


// Neural network related

struct Neuron {
	double activation;
	double error_delta;
	double* weights;
	double bias;
};

struct Layer {
	int nueron_count;
	struct Neuron* neurons;
};

struct NeuralNetwork {
	int layer_count;
	struct Layer* layers;
};

// Generates normally distributed random numbers
double randn() {
	double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
	double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
	return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void he_initialization() {
	
}

// changes
void encode_enum_as_one_hot(int type, int number_of_types, double* out) {
	for (int i = 0; i < number_of_types; i++) out[i] = i == type ? 1.0 : 0.0;
}



void forward_pass(struct NeuralNetwork* network, double *inputs) {
	
}

void backwards_pass(struct NeuralNetwork* network, double target_outputs) {
	
}

void update_weights(struct NeuralNetwork* network, double learning_rate) {
	
}

double forwards_relu(double x) {
	return x > 0 ? x : 0;
}






