#ifndef ACTION_MAPPER_H
#define ACTION_MAPPER_H

#include "observation.h"
#include "request_parser.h"

#include <stddef.h>

typedef struct {
    unsigned char legal[OBS_NUM_ACTIONS];
} ActionMask;

typedef enum {
    FACTORIZED_ACTION_NONE = 0,
    FACTORIZED_ACTION_MOVE = 1,
    FACTORIZED_ACTION_SWITCH = 2
} FactorizedActionKind;

typedef struct {
    unsigned char slot0_has_action;
    unsigned char slot0_kind;
    unsigned char slot0_move_index;
    unsigned char slot0_switch_index;
    unsigned char slot0_use_tera;
    unsigned char slot1_has_action;
    unsigned char slot1_kind;
    unsigned char slot1_move_index;
    unsigned char slot1_switch_index;
    unsigned char slot1_use_tera;
} FactorizedActionChoice;

typedef struct {
    int slot0_has_action;
    enum ObsAction action0;
    int slot1_has_action;
    enum ObsAction action1;
    char command[256];
} ValidatedRequestChoice;

void action_mask_init(ActionMask* mask);
int build_action_mask_from_request(ActionMask* out, const ParsedRequest* req);
int obs_action_slot(enum ObsAction action);
void factorized_action_choice_init(FactorizedActionChoice* choice);
void factorized_action_choice_from_flat_actions(FactorizedActionChoice* choice, int action0, int action1);
int factorized_action_choice_to_flat_actions(const FactorizedActionChoice* choice, int* action0, int* action1);
void build_slot_legal_mask(const unsigned char* legal_mask, int slot, unsigned char* out);
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
int request_choice_to_command(
    const ParsedRequest* req,
    int slot0_has_action,
    enum ObsAction action0,
    int slot1_has_action,
    enum ObsAction action1,
    char* out,
    size_t out_len
);
int command_to_request_choice(
    const char* command,
    const ParsedRequest* req,
    int* slot0_has_action,
    enum ObsAction* action0,
    int* slot1_has_action,
    enum ObsAction* action1
);
size_t collect_slot_legal_actions(
    const ParsedRequest* req,
    const ActionMask* mask,
    int slot,
    enum ObsAction* out,
    size_t out_cap
);
int validate_or_resample_request_choice(
    const ParsedRequest* req,
    const ActionMask* mask,
    const float* policy,
    int proposed_slot0_has_action,
    enum ObsAction proposed_action0,
    int proposed_slot1_has_action,
    enum ObsAction proposed_action1,
    ValidatedRequestChoice* out
);

#endif
