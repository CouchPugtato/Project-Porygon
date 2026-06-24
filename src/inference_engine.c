#include "inference_engine.h"

#include "id_tables.h"

static void refresh_tracked_move_type(TrackedInt* out, const TrackedInt* move_id) {
    int type_id;
    if (!out || !move_id) {
        return;
    }
    type_id = move_type_from_id(move_id->value);
    if (type_id <= 0) {
        tracked_int_set_unknown(out);
        return;
    }
    if (move_id->knowledge == KNOW_CONFIRMED) {
        tracked_int_set_confirmed(out, type_id);
    } else if (move_id->knowledge == KNOW_INFERRED) {
        tracked_int_set_inferred(out, type_id);
    } else {
        tracked_int_set_unknown(out);
    }
}

void inference_engine_note_move(RawPokemon* pokemon, int move_id, int turn_number) {
    int i;
    if (!pokemon || move_id <= 0) {
        return;
    }
    pokemon->last_move_id = move_id;
    pokemon->last_move_turn = turn_number;
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (pokemon->move_ids[i].value == move_id) {
            tracked_int_promote_confirmed(&pokemon->move_ids[i], move_id);
            refresh_tracked_move_type(&pokemon->move_type_ids[i], &pokemon->move_ids[i]);
            pokemon->move_known[i] = 1;
            return;
        }
    }
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (!pokemon->move_known[i]) {
            tracked_int_set_inferred(&pokemon->move_ids[i], move_id);
            refresh_tracked_move_type(&pokemon->move_type_ids[i], &pokemon->move_ids[i]);
            pokemon->move_known[i] = 1;
            return;
        }
    }
}

void inference_engine_note_effective_move(RawPokemon* pokemon, int move_id, int turn_number) {
    int i;
    if (!pokemon || move_id <= 0) {
        return;
    }
    pokemon->last_move_id = move_id;
    pokemon->last_move_turn = turn_number;
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (pokemon->effective_move_ids[i].value == move_id) {
            tracked_int_promote_confirmed(&pokemon->effective_move_ids[i], move_id);
            refresh_tracked_move_type(&pokemon->effective_move_type_ids[i], &pokemon->effective_move_ids[i]);
            pokemon->effective_move_known[i] = 1;
            return;
        }
    }
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (!pokemon->effective_move_known[i]) {
            tracked_int_set_inferred(&pokemon->effective_move_ids[i], move_id);
            refresh_tracked_move_type(&pokemon->effective_move_type_ids[i], &pokemon->effective_move_ids[i]);
            pokemon->effective_move_known[i] = 1;
            return;
        }
    }
}

void inference_engine_infer_weather_ability(RawPokemon* pokemon, int weather_id) {
    if (!pokemon || pokemon->ability_id.knowledge == KNOW_CONFIRMED) {
        return;
    }
    switch (weather_id) {
        case 1: tracked_int_set_inferred(&pokemon->ability_id, ability_id_from_name("drought")); break;
        case 2: tracked_int_set_inferred(&pokemon->ability_id, ability_id_from_name("drizzle")); break;
        default: break;
    }
}

void inference_engine_infer_terrain_ability(RawPokemon* pokemon, int terrain_id) {
    if (!pokemon || pokemon->ability_id.knowledge == KNOW_CONFIRMED) {
        return;
    }
    switch (terrain_id) {
        case 2: tracked_int_set_inferred(&pokemon->ability_id, ability_id_from_name("grassysurge")); break;
        default: break;
    }
}
