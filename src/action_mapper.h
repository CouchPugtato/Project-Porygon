#ifndef ACTION_MAPPER_H
#define ACTION_MAPPER_H

#include "observation.h"
#include "request_parser.h"

#include <stddef.h>

typedef struct {
    unsigned char legal[OBS_NUM_ACTIONS];
} ActionMask;

void action_mask_init(ActionMask* mask);
int build_action_mask_from_request(ActionMask* out, const ParsedRequest* req);
int action_to_showdown_command(
    char* out,
    size_t out_len,
    enum ObsAction action,
    const ParsedRequest* req
);
int action_to_showdown_part(
    char* out,
    size_t out_len,
    enum ObsAction action,
    const ParsedRequest* req
);
int doubles_actions_to_showdown_command(
    char* out,
    size_t out_len,
    enum ObsAction action1,
    enum ObsAction action2,
    const ParsedRequest* req
);
int request_actions_to_showdown_command(
    char* out,
    size_t out_len,
    const ParsedRequest* req,
    int slot0_has_action,
    enum ObsAction action0,
    int slot1_has_action,
    enum ObsAction action1
);

#endif
