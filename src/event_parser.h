#ifndef EVENT_PARSER_H
#define EVENT_PARSER_H

#include "raw_battle_state.h"

void event_parser_apply_line(RawBattleState* state, const char* line);

#endif
