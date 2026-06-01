#include "replay_io.h"

#include <stdio.h>

int replay_write_runtime_message(FILE* out, const RuntimeMessage* msg) {
    if (!out || !msg) {
        return 0;
    }
    switch (msg->type) {
        case RUNTIME_MSG_BATTLE_START:
            fprintf(out, "{\"type\":\"battle_start\",\"battle_id\":\"%s\",\"format\":\"%s\",\"is_doubles\":%s}\n",
                msg->battle_id, msg->format, msg->is_doubles ? "true" : "false");
            return 1;
        case RUNTIME_MSG_REQUEST:
            fprintf(out, "{\"type\":\"request\",\"battle_id\":\"%s\",\"request_id\":%d,\"payload\":%s}\n",
                msg->battle_id, msg->request_id, msg->payload[0] ? msg->payload : "{}");
            return 1;
        case RUNTIME_MSG_EVENT:
            fprintf(out, "{\"type\":\"event\",\"battle_id\":\"%s\",\"seq\":%d,\"line\":\"%s\"}\n",
                msg->battle_id, msg->seq, msg->line);
            return 1;
        case RUNTIME_MSG_TERMINAL:
            fprintf(out, "{\"type\":\"terminal\",\"battle_id\":\"%s\",\"result\":\"%s\",\"reward\":%.3f}\n",
                msg->battle_id, msg->result, msg->reward);
            return 1;
        case RUNTIME_MSG_BATTLE_END:
            fprintf(out, "{\"type\":\"battle_end\",\"battle_id\":\"%s\"}\n", msg->battle_id);
            return 1;
        case RUNTIME_MSG_DECISION:
            if (msg->accepted > 0) {
                fprintf(out, "{\"type\":\"decision_accepted\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"command\":\"%s\"}\n",
                    msg->battle_id, msg->request_id, msg->action, msg->command);
            } else if (msg->accepted == 0) {
                fprintf(out, "{\"type\":\"decision_rejected\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"command\":\"%s\",\"reason\":\"%s\"}\n",
                    msg->battle_id, msg->request_id, msg->action, msg->command, msg->message);
            } else {
                fprintf(out, "{\"type\":\"decision_proposed\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"command\":\"%s\"}\n",
                    msg->battle_id, msg->request_id, msg->action, msg->command);
            }
            return 1;
        default:
            return 0;
    }
}

int replay_write_decision_proposed(FILE* out, const char* battle_id, int request_id, int action, const char* command) {
    if (!out) {
        return 0;
    }
    fprintf(out, "{\"type\":\"decision_proposed\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"command\":\"%s\"}\n",
        battle_id ? battle_id : "", request_id, action, command ? command : "");
    return 1;
}

int replay_write_decision_accepted(FILE* out, const char* battle_id, int request_id, int action, const char* command) {
    if (!out) {
        return 0;
    }
    fprintf(out, "{\"type\":\"decision_accepted\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"command\":\"%s\"}\n",
        battle_id ? battle_id : "", request_id, action, command ? command : "");
    return 1;
}

int replay_write_decision_rejected(FILE* out, const char* battle_id, int request_id, int action, const char* command, const char* reason) {
    if (!out) {
        return 0;
    }
    fprintf(out, "{\"type\":\"decision_rejected\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"command\":\"%s\",\"reason\":\"%s\"}\n",
        battle_id ? battle_id : "", request_id, action, command ? command : "", reason ? reason : "");
    return 1;
}
