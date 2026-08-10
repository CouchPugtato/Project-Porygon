#include "action_mapper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int is_move_action(enum ObsAction action, int* active_slot, int* move_slot, int* tera);
static int is_switch_action(enum ObsAction action, int* active_slot, int* switch_slot);

void factorized_action_choice_init(FactorizedActionChoice* choice) {
    if (!choice) {
        return;
    }
    memset(choice, 0, sizeof(*choice));
}

static void factorized_slot_from_action(
    FactorizedActionChoice* choice,
    int slot,
    int action
) {
    int active_slot = -1;
    int move_slot = 0;
    int switch_slot = 0;
    int tera = 0;
    if (!choice || action < 0) {
        return;
    }
    if (is_move_action((enum ObsAction)action, &active_slot, &move_slot, &tera) && active_slot == slot) {
        if (slot == 0) {
            choice->slot0_has_action = 1;
            choice->slot0_kind = FACTORIZED_ACTION_MOVE;
            choice->slot0_move_index = (unsigned char)move_slot;
            choice->slot0_use_tera = (unsigned char)(tera ? 1 : 0);
        } else {
            choice->slot1_has_action = 1;
            choice->slot1_kind = FACTORIZED_ACTION_MOVE;
            choice->slot1_move_index = (unsigned char)move_slot;
            choice->slot1_use_tera = (unsigned char)(tera ? 1 : 0);
        }
        return;
    }
    if (is_switch_action((enum ObsAction)action, &active_slot, &switch_slot) && active_slot == slot) {
        if (slot == 0) {
            choice->slot0_has_action = 1;
            choice->slot0_kind = FACTORIZED_ACTION_SWITCH;
            choice->slot0_switch_index = (unsigned char)switch_slot;
            choice->slot0_use_tera = 0;
        } else {
            choice->slot1_has_action = 1;
            choice->slot1_kind = FACTORIZED_ACTION_SWITCH;
            choice->slot1_switch_index = (unsigned char)switch_slot;
            choice->slot1_use_tera = 0;
        }
    }
}

void factorized_action_choice_from_flat_actions(FactorizedActionChoice* choice, int action0, int action1) {
    factorized_action_choice_init(choice);
    factorized_slot_from_action(choice, 0, action0);
    factorized_slot_from_action(choice, 1, action1);
}

static int factorized_slot_to_action(
    int slot,
    unsigned char has_action,
    unsigned char kind,
    unsigned char move_index,
    unsigned char switch_index,
    unsigned char use_tera,
    int* out_action
) {
    if (!out_action) {
        return 0;
    }
    *out_action = -1;
    if (!has_action || kind == FACTORIZED_ACTION_NONE) {
        return 1;
    }
    if (kind == FACTORIZED_ACTION_MOVE) {
        if (move_index >= 4) {
            return 0;
        }
        if (slot == 0) {
            *out_action = (int)(use_tera ? OBS_A1_MOVE1_TERA : OBS_A1_MOVE1) + (int)move_index;
        } else {
            *out_action = (int)(use_tera ? OBS_A2_MOVE1_TERA : OBS_A2_MOVE1) + (int)move_index;
        }
        return 1;
    }
    if (kind == FACTORIZED_ACTION_SWITCH) {
        if (switch_index >= 6) {
            return 0;
        }
        if (slot == 0) {
            *out_action = (int)OBS_A1_SWITCH1 + (int)switch_index;
        } else {
            *out_action = (int)OBS_A2_SWITCH1 + (int)switch_index;
        }
        return 1;
    }
    return 0;
}

int factorized_action_choice_to_flat_actions(const FactorizedActionChoice* choice, int* action0, int* action1) {
    if (!choice || !action0 || !action1) {
        return 0;
    }
    if (!factorized_slot_to_action(0,
            choice->slot0_has_action,
            choice->slot0_kind,
            choice->slot0_move_index,
            choice->slot0_switch_index,
            choice->slot0_use_tera,
            action0)) {
        return 0;
    }
    if (!factorized_slot_to_action(1,
            choice->slot1_has_action,
            choice->slot1_kind,
            choice->slot1_move_index,
            choice->slot1_switch_index,
            choice->slot1_use_tera,
            action1)) {
        return 0;
    }
    return 1;
}

void action_mask_init(ActionMask* mask) {
    if (!mask) {
        return;
    }
    memset(mask, 0, sizeof(*mask));
}

int obs_action_slot(enum ObsAction action) {
    int active_slot = -1;
    int move_slot = 0;
    int switch_slot = 0;
    int tera = 0;
    if (is_move_action(action, &active_slot, &move_slot, &tera)) {
        return active_slot;
    }
    if (is_switch_action(action, &active_slot, &switch_slot)) {
        return active_slot;
    }
    return -1;
}

void build_slot_legal_mask(const unsigned char* legal_mask, int slot, unsigned char* out) {
    int action;
    if (!out) {
        return;
    }
    memset(out, 0, OBS_NUM_ACTIONS * sizeof(unsigned char));
    if (!legal_mask || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return;
    }
    for (action = 0; action < OBS_NUM_ACTIONS; ++action) {
        if (legal_mask[action] && obs_action_slot((enum ObsAction)action) == slot) {
            out[action] = 1;
        }
    }
}

static int is_move_action(enum ObsAction action, int* active_slot, int* move_slot, int* tera) {
    if (action >= OBS_A1_MOVE1 && action <= OBS_A1_MOVE4) {
        *active_slot = 0;
        *move_slot = (int)action - (int)OBS_A1_MOVE1;
        *tera = 0;
        return 1;
    }
    if (action >= OBS_A1_MOVE1_TERA && action <= OBS_A1_MOVE4_TERA) {
        *active_slot = 0;
        *move_slot = (int)action - (int)OBS_A1_MOVE1_TERA;
        *tera = 1;
        return 1;
    }
    if (action >= OBS_A2_MOVE1 && action <= OBS_A2_MOVE4) {
        *active_slot = 1;
        *move_slot = (int)action - (int)OBS_A2_MOVE1;
        *tera = 0;
        return 1;
    }
    if (action >= OBS_A2_MOVE1_TERA && action <= OBS_A2_MOVE4_TERA) {
        *active_slot = 1;
        *move_slot = (int)action - (int)OBS_A2_MOVE1_TERA;
        *tera = 1;
        return 1;
    }
    return 0;
}

static int is_switch_action(enum ObsAction action, int* active_slot, int* switch_slot) {
    if (action >= OBS_A1_SWITCH1 && action <= OBS_A1_SWITCH6) {
        *active_slot = 0;
        *switch_slot = (int)action - (int)OBS_A1_SWITCH1;
        return 1;
    }
    if (action >= OBS_A2_SWITCH1 && action <= OBS_A2_SWITCH6) {
        *active_slot = 1;
        *switch_slot = (int)action - (int)OBS_A2_SWITCH1;
        return 1;
    }
    return 0;
}

static int move_target_suffix(const ParsedRequest* req, int active_slot, int move_slot, char* out, size_t out_len) {
    ParsedMoveTarget target;
    if (!req || !out || out_len == 0 || active_slot < 0 || active_slot >= PARSED_REQUEST_ACTIVE_SLOTS ||
        move_slot < 0 || move_slot >= PARSED_REQUEST_MOVE_SLOTS) {
        return 0;
    }
    out[0] = '\0';
    target = req->active[active_slot].move_target[move_slot];
    switch (target) {
        case REQUEST_TARGET_NORMAL:
        case REQUEST_TARGET_ADJACENT_FOE:
        case REQUEST_TARGET_ANY:
            snprintf(out, out_len, " 1");
            return 1;
        case REQUEST_TARGET_ADJACENT_ALLY:
            snprintf(out, out_len, active_slot == 0 ? " -2" : " -1");
            return 1;
        case REQUEST_TARGET_ADJACENT_ALLY_OR_SELF:
            snprintf(out, out_len, active_slot == 0 ? " -1" : " -2");
            return 1;
        case REQUEST_TARGET_SELF:
        case REQUEST_TARGET_ALL_ADJACENT_FOES:
        case REQUEST_TARGET_ALL:
        case REQUEST_TARGET_ALLY_SIDE:
        case REQUEST_TARGET_FOE_SIDE:
        case REQUEST_TARGET_UNKNOWN:
        default:
            return 1;
    }
}

static int request_choice_pair_is_valid(
    const ParsedRequest* req,
    int slot0_has_action,
    enum ObsAction action0,
    int slot1_has_action,
    enum ObsAction action1
) {
    int active_slot0 = -1;
    int active_slot1 = -1;
    int move_slot0 = -1;
    int move_slot1 = -1;
    int switch_slot0 = -1;
    int switch_slot1 = -1;
    int tera0 = 0;
    int tera1 = 0;

    if (!req) {
        return 0;
    }
    if (slot0_has_action && slot1_has_action &&
            is_move_action(action0, &active_slot0, &move_slot0, &tera0) &&
            is_move_action(action1, &active_slot1, &move_slot1, &tera1) &&
            tera0 && tera1) {
        return 0;
    }
    if (slot0_has_action && slot1_has_action &&
            is_switch_action(action0, &active_slot0, &switch_slot0) &&
            is_switch_action(action1, &active_slot1, &switch_slot1)) {
        if (switch_slot0 == switch_slot1) {
            return 0;
        }
    }
    return 1;
}

static int action_belongs_to_slot(enum ObsAction action, int slot) {
    int active_slot = -1;
    int move_slot = 0;
    int switch_slot = 0;
    int tera = 0;
    if (is_move_action(action, &active_slot, &move_slot, &tera)) {
        return active_slot == slot;
    }
    if (is_switch_action(action, &active_slot, &switch_slot)) {
        return active_slot == slot;
    }
    return 0;
}

static int action_index_legal_for_request(
    const ParsedRequest* req,
    const ActionMask* mask,
    int slot,
    enum ObsAction action
) {
    if (!req || !mask || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return 0;
    }
    if (!parsed_request_slot_needs_choice(req, slot)) {
        return 0;
    }
    if (!action_belongs_to_slot(action, slot)) {
        return 0;
    }
    if ((int)action < 0 || (int)action >= OBS_NUM_ACTIONS || !mask->legal[action]) {
        return 0;
    }
    if (parsed_request_slot_choice_kind(req, slot) == REQUEST_SLOT_FORCE_SWITCH) {
        int active_slot = -1;
        int switch_slot = 0;
        return is_switch_action(action, &active_slot, &switch_slot);
    }
    if (parsed_request_slot_choice_kind(req, slot) == REQUEST_SLOT_TEAM_PREVIEW) {
        int active_slot = -1;
        int switch_slot = 0;
        return is_switch_action(action, &active_slot, &switch_slot) && active_slot == 0;
    }
    return 1;
}

int build_action_mask_from_request(ActionMask* out, const ParsedRequest* req) {
    int i;
    if (!out || !req) {
        return 0;
    }
    action_mask_init(out);

    if (req->team_preview) {
        out->legal[OBS_A1_SWITCH1] = req->switch_available[0] ? 1 : 0;
        out->legal[OBS_A1_SWITCH2] = req->switch_available[1] ? 1 : 0;
        out->legal[OBS_A1_SWITCH3] = req->switch_available[2] ? 1 : 0;
        out->legal[OBS_A1_SWITCH4] = req->switch_available[3] ? 1 : 0;
        out->legal[OBS_A1_SWITCH5] = req->switch_available[4] ? 1 : 0;
        out->legal[OBS_A1_SWITCH6] = req->switch_available[5] ? 1 : 0;
        return 1;
    }

    for (i = 0; i < req->active_count && i < PARSED_REQUEST_ACTIVE_SLOTS; ++i) {
        int m;
        for (m = 0; m < PARSED_REQUEST_MOVE_SLOTS; ++m) {
            int legal = parsed_request_slot_can_move(req, i) && req->active[i].move_id[m] > 0 && !req->active[i].move_disabled[m];
            if (i == 0) {
                out->legal[OBS_A1_MOVE1 + m] = (unsigned char)legal;
                out->legal[OBS_A1_MOVE1_TERA + m] = (unsigned char)(legal && req->active[i].can_tera);
            } else {
                out->legal[OBS_A2_MOVE1 + m] = (unsigned char)legal;
                out->legal[OBS_A2_MOVE1_TERA + m] = (unsigned char)(legal && req->active[i].can_tera);
            }
        }
    }

    for (i = 0; i < PARSED_REQUEST_TEAM_SIZE; ++i) {
        int bench_switch_legal = req->switch_available[i] && !req->switch_fainted[i] && !req->switch_active[i];
        if (bench_switch_legal) {
            if (parsed_request_slot_can_switch(req, 0)) {
                out->legal[OBS_A1_SWITCH1 + i] = 1;
            }
            if (parsed_request_slot_can_switch(req, 1)) {
                out->legal[OBS_A2_SWITCH1 + i] = 1;
            }
        }
    }

    if (req->forced_switch_any) {
        for (i = 0; i < 8; ++i) {
            out->legal[OBS_A1_MOVE1 + i] = 0;
            out->legal[OBS_A2_MOVE1 + i] = 0;
        }
        for (i = 0; i < PARSED_REQUEST_TEAM_SIZE; ++i) {
            int bench_switch_legal = req->switch_available[i] && !req->switch_fainted[i] && !req->switch_active[i];
            out->legal[OBS_A1_SWITCH1 + i] = (req->force_switch[0] && bench_switch_legal) ? 1 : 0;
            out->legal[OBS_A2_SWITCH1 + i] = (req->force_switch[1] && bench_switch_legal) ? 1 : 0;
        }
    }

    return 1;
}

int action_to_showdown_command(
    char* out,
    size_t out_len,
    enum ObsAction action,
    const ParsedRequest* req
) {
    char part[64];
    if (!out || out_len == 0 || !req) {
        return 0;
    }
    if (req->team_preview) {
        return action_to_showdown_part(out, out_len, action, req);
    }
    if (!action_to_showdown_part(part, sizeof(part), action, req)) {
        return 0;
    }
    snprintf(out, out_len, "/choose %s", part);
    return 1;
}

int action_to_showdown_part(
    char* out,
    size_t out_len,
    enum ObsAction action,
    const ParsedRequest* req
) {
    int active_slot;
    int move_slot;
    int switch_slot;
    int tera;
    char target_suffix[16] = {0};

    if (!out || out_len == 0 || !req) {
        return 0;
    }
    out[0] = '\0';

    if (req->team_preview) {
        if (is_switch_action(action, &active_slot, &switch_slot) && active_slot == 0) {
            snprintf(out, out_len, "/choose team %d", switch_slot + 1);
            return 1;
        }
        return 0;
    }

    if (is_move_action(action, &active_slot, &move_slot, &tera)) {
        if (!move_target_suffix(req, active_slot, move_slot, target_suffix, sizeof(target_suffix))) {
            return 0;
        }
        snprintf(out, out_len, "move %d%s%s",
            move_slot + 1, tera ? " terastallize" : "", target_suffix);
    } else if (is_switch_action(action, &active_slot, &switch_slot)) {
        snprintf(out, out_len, "switch %d", switch_slot + 1);
    } else {
        return 0;
    }
    return 1;
}

int doubles_actions_to_showdown_command(
    char* out,
    size_t out_len,
    enum ObsAction action1,
    enum ObsAction action2,
    const ParsedRequest* req
) {
    char part1[64];
    char part2[64];
    if (!out || out_len == 0 || !req) {
        return 0;
    }
    if (!action_to_showdown_part(part1, sizeof(part1), action1, req)) {
        return 0;
    }
    if (!action_to_showdown_part(part2, sizeof(part2), action2, req)) {
        return 0;
    }
    snprintf(out, out_len, "/choose %s, %s", part1, part2);
    return 1;
}

int request_actions_to_showdown_command(
    char* out,
    size_t out_len,
    const ParsedRequest* req,
    int slot0_has_action,
    enum ObsAction action0,
    int slot1_has_action,
    enum ObsAction action1
) {
    char part0[64];
    char part1[64];

    if (!out || out_len == 0 || !req) {
        return 0;
    }
    out[0] = '\0';

    if (slot0_has_action) {
        if (!action_to_showdown_part(part0, sizeof(part0), action0, req)) {
            return 0;
        }
    }
    if (slot1_has_action) {
        if (!action_to_showdown_part(part1, sizeof(part1), action1, req)) {
            return 0;
        }
    }

    if (slot0_has_action && slot1_has_action) {
        snprintf(out, out_len, "/choose %s, %s", part0, part1);
        return 1;
    }
    if (slot0_has_action) {
        if (req->forced_switch_any && req->force_switch[1]) {
            snprintf(out, out_len, "/choose %s, pass", part0);
        } else {
            snprintf(out, out_len, "/choose %s", part0);
        }
        return 1;
    }
    if (slot1_has_action) {
        if (req->forced_switch_any && req->force_switch[1]) {
            snprintf(out, out_len, "/choose pass, %s", part1);
        } else {
            snprintf(out, out_len, "/choose %s", part1);
        }
        return 1;
    }
    return 0;
}

int request_choice_to_command(
    const ParsedRequest* req,
    int slot0_has_action,
    enum ObsAction action0,
    int slot1_has_action,
    enum ObsAction action1,
    char* out,
    size_t out_len
) {
    return request_actions_to_showdown_command(out, out_len, req, slot0_has_action, action0, slot1_has_action, action1);
}

int command_to_request_choice(
    const char* command,
    const ParsedRequest* req,
    int* slot0_has_action,
    enum ObsAction* action0,
    int* slot1_has_action,
    enum ObsAction* action1
) {
    ActionMask mask;
    char candidate[128];
    int need0;
    int need1;
    int has0_options[2];
    int has1_options[2];
    int i;
    int j;
    int k;
    int l;

    if (!command || !req || !slot0_has_action || !action0 || !slot1_has_action || !action1) {
        return 0;
    }

    need0 = parsed_request_slot_needs_choice(req, 0);
    need1 = parsed_request_slot_needs_choice(req, 1);
    has0_options[0] = need0 ? 1 : 0;
    has0_options[1] = 0;
    has1_options[0] = need1 ? 1 : 0;
    has1_options[1] = 0;
    if (!req->team_preview) {
        if (!need0) {
            has0_options[1] = 1;
        }
        if (!need1) {
            has1_options[1] = 1;
        }
    }

    if (!build_action_mask_from_request(&mask, req)) {
        return 0;
    }

    for (i = 0; i < 2; ++i) {
        int has0 = has0_options[i];
        int start0 = has0 ? 0 : 0;
        int end0 = has0 ? OBS_NUM_ACTIONS : 1;
        for (j = 0; j < 2; ++j) {
            int has1 = has1_options[j];
            int start1 = has1 ? 0 : 0;
            int end1 = has1 ? OBS_NUM_ACTIONS : 1;
            if (!has0 && !has1) {
                continue;
            }
            for (k = start0; k < end0; ++k) {
                    enum ObsAction a0 = has0 ? (enum ObsAction)k : OBS_A1_MOVE1;
                    if (has0 && !action_index_legal_for_request(req, &mask, 0, a0)) {
                        continue;
                    }
                for (l = start1; l < end1; ++l) {
                    enum ObsAction a1 = has1 ? (enum ObsAction)l : OBS_A2_MOVE1;
                    if (has1 && !action_index_legal_for_request(req, &mask, 1, a1)) {
                        continue;
                    }
                    if (!request_actions_to_showdown_command(candidate, sizeof(candidate), req, has0, a0, has1, a1)) {
                        continue;
                    }
                    if (strcmp(candidate, command) == 0) {
                        *slot0_has_action = has0;
                        *action0 = a0;
                        *slot1_has_action = has1;
                        *action1 = a1;
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

size_t collect_slot_legal_actions(
    const ParsedRequest* req,
    const ActionMask* mask,
    int slot,
    enum ObsAction* out,
    size_t out_cap
) {
    size_t count = 0;
    int action;
    if (!req || !mask || !out || out_cap == 0 || slot < 0 || slot >= PARSED_REQUEST_ACTIVE_SLOTS) {
        return 0;
    }
    for (action = 0; action < OBS_NUM_ACTIONS; ++action) {
        if (!action_index_legal_for_request(req, mask, slot, (enum ObsAction)action)) {
            continue;
        }
        if (count < out_cap) {
            out[count] = (enum ObsAction)action;
        }
        ++count;
    }
    return count;
}

static int select_best_legal_action(
    const float* policy,
    const enum ObsAction* legal_actions,
    size_t legal_count,
    enum ObsAction* chosen
) {
    size_t i;
    float best_score;
    size_t ties = 0;
    if (!policy || !legal_actions || legal_count == 0 || !chosen) {
        return 0;
    }
    best_score = policy[legal_actions[0]];
    *chosen = legal_actions[0];
    for (i = 1; i < legal_count; ++i) {
        float score = policy[legal_actions[i]];
        if (score > best_score) {
            best_score = score;
            *chosen = legal_actions[i];
            ties = 0;
        } else if (score == best_score) {
            ++ties;
            if ((rand() % (int)(ties + 2)) == 0) {
                *chosen = legal_actions[i];
            }
        }
    }
    return 1;
}

int validate_or_resample_request_choice(
    const ParsedRequest* req,
    const ActionMask* mask,
    const float* policy,
    int proposed_slot0_has_action,
    enum ObsAction proposed_action0,
    int proposed_slot1_has_action,
    enum ObsAction proposed_action1,
    ValidatedRequestChoice* out
) {
    enum ObsAction slot0_actions[OBS_NUM_ACTIONS];
    enum ObsAction slot1_actions[OBS_NUM_ACTIONS];
    size_t slot0_count;
    size_t slot1_count;
    int need0;
    int need1;

    if (!req || !mask || !policy || !out) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    out->action0 = OBS_A1_MOVE1;
    out->action1 = OBS_A2_MOVE1;

    need0 = parsed_request_slot_needs_choice(req, 0);
    need1 = parsed_request_slot_needs_choice(req, 1);
    slot0_count = need0 ? collect_slot_legal_actions(req, mask, 0, slot0_actions, OBS_NUM_ACTIONS) : 0;
    slot1_count = need1 ? collect_slot_legal_actions(req, mask, 1, slot1_actions, OBS_NUM_ACTIONS) : 0;

    out->slot0_has_action = need0;
    out->slot1_has_action = need1;

    if (need0 && need1) {
        int proposed_valid = proposed_slot0_has_action &&
            proposed_slot1_has_action &&
            action_index_legal_for_request(req, mask, 0, proposed_action0) &&
            action_index_legal_for_request(req, mask, 1, proposed_action1) &&
            request_choice_pair_is_valid(req, 1, proposed_action0, 1, proposed_action1);
        if (proposed_valid) {
            out->action0 = proposed_action0;
            out->action1 = proposed_action1;
        } else {
            size_t i;
            size_t j;
            float best_score = 0.0f;
            int found = 0;
            for (i = 0; i < slot0_count; ++i) {
                for (j = 0; j < slot1_count; ++j) {
                    float score;
                    if (!request_choice_pair_is_valid(req, 1, slot0_actions[i], 1, slot1_actions[j])) {
                        continue;
                    }
                    score = policy[slot0_actions[i]] + policy[slot1_actions[j]];
                    if (!found || score > best_score) {
                        best_score = score;
                        out->action0 = slot0_actions[i];
                        out->action1 = slot1_actions[j];
                        found = 1;
                    }
                }
            }
            if (!found) {
                if (req->forced_switch_any) {
                    if (slot0_count > 0) {
                        out->slot0_has_action = 1;
                        out->slot1_has_action = 0;
                        out->action0 = slot0_actions[0];
                    } else if (slot1_count > 0) {
                        out->slot0_has_action = 0;
                        out->slot1_has_action = 1;
                        out->action1 = slot1_actions[0];
                    } else {
                        return 0;
                    }
                } else {
                    return 0;
                }
            }
        }
    } else if (need0) {
        if (proposed_slot0_has_action && action_index_legal_for_request(req, mask, 0, proposed_action0)) {
            out->action0 = proposed_action0;
        } else if (!select_best_legal_action(policy, slot0_actions, slot0_count, &out->action0)) {
            return 0;
        }
    } else if (need1) {
        if (proposed_slot1_has_action && action_index_legal_for_request(req, mask, 1, proposed_action1)) {
            out->action1 = proposed_action1;
        } else if (!select_best_legal_action(policy, slot1_actions, slot1_count, &out->action1)) {
            return 0;
        }
    }

    if (!request_choice_to_command(req,
            out->slot0_has_action, out->action0,
            out->slot1_has_action, out->action1,
            out->command, sizeof(out->command))) {
        return 0;
    }
    return 1;
}
