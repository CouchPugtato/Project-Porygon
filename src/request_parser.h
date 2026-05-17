#ifndef REQUEST_PARSER_H
#define REQUEST_PARSER_H

#include <stddef.h>

#define PARSED_REQUEST_MAX_JSON 8192
#define PARSED_REQUEST_ACTIVE_SLOTS 2
#define PARSED_REQUEST_MOVE_SLOTS 4
#define PARSED_REQUEST_TEAM_SIZE 6

typedef struct {
    int move_id[PARSED_REQUEST_MOVE_SLOTS];
    int move_disabled[PARSED_REQUEST_MOVE_SLOTS];
    int move_pp[PARSED_REQUEST_MOVE_SLOTS];
    int move_max_pp[PARSED_REQUEST_MOVE_SLOTS];
    int can_tera;
    int trapped;
    int fainted;
    int has_force_switch;
} ParsedActive;

typedef struct {
    int request_id;
    int is_doubles;
    int team_preview;
    int max_chosen_team_size;
    int active_count;
    int switch_available[PARSED_REQUEST_TEAM_SIZE];
    int switch_fainted[PARSED_REQUEST_TEAM_SIZE];
    int can_tera;
    int forced_switch_any;
    ParsedActive active[PARSED_REQUEST_ACTIVE_SLOTS];
    char raw_json[PARSED_REQUEST_MAX_JSON];
} ParsedRequest;

void parsed_request_init(ParsedRequest* req);
int parse_request_payload(ParsedRequest* req, const char* json, int request_id, int is_doubles);

#endif
