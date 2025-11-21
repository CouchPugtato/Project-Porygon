#ifndef GAME_STATE_H
#define GAME_STATE_H

// Initialize defaults for BattleState
void battle_state_init(struct BattleState* bs);

// Update friendly side from `|request|{...}` JSON string (minimal parse)
// Fills HP/max/status for your team.
void battle_state_update_from_request(struct BattleState* bs, const char* request_json);

// Update from battle stream line (e.g., "|-weather|", "|-sidestart|", "|-fieldstart|")
void battle_state_update_from_line(struct BattleState* bs, const char* line);

#endif