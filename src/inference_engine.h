#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include "raw_battle_state.h"

void inference_engine_note_move(RawPokemon* pokemon, int move_id, int turn_number);
void inference_engine_infer_weather_ability(RawPokemon* pokemon, int weather_id);
void inference_engine_infer_terrain_ability(RawPokemon* pokemon, int terrain_id);

#endif
