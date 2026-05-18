#include "action_mapper.h"

#include <stdio.h>
#include <string.h>

void action_mask_init(ActionMask* mask) {
    if (!mask) {
        return;
    }
    memset(mask, 0, sizeof(*mask));
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
            int legal = req->active[i].move_id[m] > 0 && !req->active[i].move_disabled[m] && !req->active[i].fainted;
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
            if (req->team_preview || !req->active[0].trapped) {
                out->legal[OBS_A1_SWITCH1 + i] = 1;
            }
            if ((req->active_count < 2 && !req->is_doubles) || req->team_preview || !req->active[1].trapped) {
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
            out->legal[OBS_A1_SWITCH1 + i] = bench_switch_legal ? 1 : 0;
            out->legal[OBS_A2_SWITCH1 + i] = bench_switch_legal ? 1 : 0;
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
