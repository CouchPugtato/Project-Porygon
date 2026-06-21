#ifndef REQUEST_PARSER_H
#define REQUEST_PARSER_H

#include <stddef.h>

#define PARSED_REQUEST_MAX_JSON 32768
#define PARSED_REQUEST_ACTIVE_SLOTS 2
#define PARSED_REQUEST_MOVE_SLOTS 4
#define PARSED_REQUEST_TEAM_SIZE 6

typedef enum {
    REQUEST_TARGET_UNKNOWN = 0,
    REQUEST_TARGET_NORMAL,
    REQUEST_TARGET_ADJACENT_FOE,
    REQUEST_TARGET_ANY,
    REQUEST_TARGET_ADJACENT_ALLY,
    REQUEST_TARGET_ADJACENT_ALLY_OR_SELF,
    REQUEST_TARGET_SELF,
    REQUEST_TARGET_ALL_ADJACENT_FOES,
    REQUEST_TARGET_ALL,
    REQUEST_TARGET_ALLY_SIDE,
    REQUEST_TARGET_FOE_SIDE
} ParsedMoveTarget;

typedef enum {
    REQUEST_SLOT_NONE = 0,
    REQUEST_SLOT_MOVE_OR_SWITCH,
    REQUEST_SLOT_FORCE_SWITCH,
    REQUEST_SLOT_TEAM_PREVIEW
} ParsedSlotChoiceKind;

typedef struct {
    int move_id[PARSED_REQUEST_MOVE_SLOTS];
    int move_disabled[PARSED_REQUEST_MOVE_SLOTS];
    int move_maybe_disabled[PARSED_REQUEST_MOVE_SLOTS];
    int move_pp[PARSED_REQUEST_MOVE_SLOTS];
    int move_max_pp[PARSED_REQUEST_MOVE_SLOTS];
    ParsedMoveTarget move_target[PARSED_REQUEST_MOVE_SLOTS];
    int can_tera;
    int tera_type_id;
    int trapped;
    int maybe_trapped;
    int fainted;
    int has_force_switch;
} ParsedActive;

typedef struct {
    int request_id;
    int is_doubles;
    int side_player;
    int wait;
    int team_preview;
    int max_chosen_team_size;
    int active_count;
    int living_active_count;
    int switch_available[PARSED_REQUEST_TEAM_SIZE];
    int switch_fainted[PARSED_REQUEST_TEAM_SIZE];
    int switch_active[PARSED_REQUEST_TEAM_SIZE];
    int force_switch[PARSED_REQUEST_ACTIVE_SLOTS];
    int can_tera;
    int forced_switch_any;
    int slot_present[PARSED_REQUEST_ACTIVE_SLOTS];
    int slot_needs_choice[PARSED_REQUEST_ACTIVE_SLOTS];
    int slot_can_move[PARSED_REQUEST_ACTIVE_SLOTS];
    int slot_can_switch[PARSED_REQUEST_ACTIVE_SLOTS];
    int active_team_idx[PARSED_REQUEST_ACTIVE_SLOTS];
    unsigned char active_team_idx_known[PARSED_REQUEST_ACTIVE_SLOTS];
    unsigned char bootstrap_slot_binding_ambiguous;
    ParsedSlotChoiceKind choice_kind[PARSED_REQUEST_ACTIVE_SLOTS];
    char side_id[8];
    char side_ident[PARSED_REQUEST_TEAM_SIZE][32];
    int side_species_id[PARSED_REQUEST_TEAM_SIZE];
    ParsedActive active[PARSED_REQUEST_ACTIVE_SLOTS];
    char raw_json[PARSED_REQUEST_MAX_JSON];
} ParsedRequest;

void parsed_request_init(ParsedRequest* req);
int parse_request_payload(ParsedRequest* req, const char* json, int request_id, int is_doubles);
int parsed_request_slot_needs_choice(const ParsedRequest* req, int slot);
int parsed_request_slot_can_move(const ParsedRequest* req, int slot);
int parsed_request_slot_can_switch(const ParsedRequest* req, int slot);
ParsedSlotChoiceKind parsed_request_slot_choice_kind(const ParsedRequest* req, int slot);

#endif
