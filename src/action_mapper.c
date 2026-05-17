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
        if (req->switch_available[i]) {
            out->legal[OBS_A1_SWITCH1 + i] = 1;
            out->legal[OBS_A2_SWITCH1 + i] = 1;
        }
    }

    if (req->forced_switch_any) {
        for (i = 0; i < 8; ++i) {
            out->legal[OBS_A1_MOVE1 + i] = 0;
            out->legal[OBS_A2_MOVE1 + i] = 0;
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
    int active_slot;
    int move_slot;
    int switch_slot;
    int tera;
    char part1[64] = {0};
    char part2[64] = {0};

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
        snprintf(active_slot == 0 ? part1 : part2, sizeof(part1), "move %d%s", move_slot + 1, tera ? " terastallize" : "");
    } else if (is_switch_action(action, &active_slot, &switch_slot)) {
        snprintf(active_slot == 0 ? part1 : part2, sizeof(part1), "switch %d", switch_slot + 1);
    } else {
        return 0;
    }

    if (!req->is_doubles || req->active_count <= 1) {
        snprintf(out, out_len, "/choose %s", part1[0] ? part1 : part2);
        return 1;
    }

    if (!part1[0]) {
        snprintf(part1, sizeof(part1), "move 1");
    }
    if (!part2[0]) {
        snprintf(part2, sizeof(part2), "move 1");
    }
    snprintf(out, out_len, "/choose %s, %s", part1, part2);
    return 1;
}
