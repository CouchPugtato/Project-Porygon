#include "runtime_protocol.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* skip_ws(const char* p) {
    while (p && *p && isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static const char* find_after_key(const char* json, const char* key) {
    char pattern[64];
    size_t n;
    if (!json || !key) {
        return NULL;
    }
    n = strlen(key);
    if (n + 4 >= sizeof(pattern)) {
        return NULL;
    }
    pattern[0] = '"';
    memcpy(pattern + 1, key, n);
    pattern[n + 1] = '"';
    pattern[n + 2] = ':';
    pattern[n + 3] = '\0';
    return strstr(json, pattern);
}

static int extract_json_string(const char* json, const char* key, char* out, size_t out_len) {
    const char* p = find_after_key(json, key);
    size_t i = 0;
    if (!p || out_len == 0) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    p = skip_ws(p + 1);
    if (*p != '"') {
        return 0;
    }
    ++p;
    while (*p && *p != '"' && i + 1 < out_len) {
        if (*p == '\\' && p[1]) {
            ++p;
        }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return *p == '"';
}

static int extract_json_number(const char* json, const char* key, int default_value) {
    const char* p = find_after_key(json, key);
    if (!p) {
        return default_value;
    }
    p = strchr(p, ':');
    if (!p) {
        return default_value;
    }
    p = skip_ws(p + 1);
    return atoi(p);
}

static float extract_json_float(const char* json, const char* key, float default_value) {
    const char* p = find_after_key(json, key);
    if (!p) {
        return default_value;
    }
    p = strchr(p, ':');
    if (!p) {
        return default_value;
    }
    p = skip_ws(p + 1);
    return (float)atof(p);
}

static int extract_json_bool(const char* json, const char* key, int default_value) {
    const char* p = find_after_key(json, key);
    if (!p) {
        return default_value;
    }
    p = strchr(p, ':');
    if (!p) {
        return default_value;
    }
    p = skip_ws(p + 1);
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return default_value;
}

static int extract_json_block(const char* json, const char* key, char open, char close, char* out, size_t out_len) {
    const char* p = find_after_key(json, key);
    int depth = 0;
    int in_string = 0;
    size_t i = 0;
    if (!p || out_len == 0) {
        return 0;
    }
    p = strchr(p, open);
    if (!p) {
        return 0;
    }
    while (*p && i + 1 < out_len) {
        char ch = *p;
        out[i++] = ch;
        if (ch == '"' && (p == json || p[-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (ch == open) ++depth;
            else if (ch == close) {
                --depth;
                if (depth == 0) {
                    out[i] = '\0';
                    return 1;
                }
            }
        }
        ++p;
    }
    out[i < out_len ? i : out_len - 1] = '\0';
    return 0;
}

void runtime_message_init(RuntimeMessage* msg) {
    if (!msg) {
        return;
    }
    memset(msg, 0, sizeof(*msg));
    msg->accepted = -1;
    msg->action = -1;
    msg->action2 = -1;
}

int runtime_message_parse(RuntimeMessage* msg, const char* json_line) {
    char type[32];
    if (!msg || !json_line) {
        return 0;
    }
    runtime_message_init(msg);
    if (!extract_json_string(json_line, "type", type, sizeof(type))) {
        return 0;
    }
    if (strcmp(type, "battle_start") == 0) msg->type = RUNTIME_MSG_BATTLE_START;
    else if (strcmp(type, "request") == 0) msg->type = RUNTIME_MSG_REQUEST;
    else if (strcmp(type, "event") == 0) msg->type = RUNTIME_MSG_EVENT;
    else if (strcmp(type, "terminal") == 0) msg->type = RUNTIME_MSG_TERMINAL;
    else if (strcmp(type, "battle_end") == 0) msg->type = RUNTIME_MSG_BATTLE_END;
    else if (strcmp(type, "error") == 0) msg->type = RUNTIME_MSG_ERROR;
    else if (strcmp(type, "heartbeat") == 0) msg->type = RUNTIME_MSG_HEARTBEAT;
    else if (strcmp(type, "decision") == 0) { msg->type = RUNTIME_MSG_DECISION; msg->accepted = 1; }
    else if (strcmp(type, "decision_proposed") == 0) { msg->type = RUNTIME_MSG_DECISION; msg->accepted = -1; }
    else if (strcmp(type, "decision_accepted") == 0) { msg->type = RUNTIME_MSG_DECISION; msg->accepted = 1; }
    else if (strcmp(type, "decision_rejected") == 0) { msg->type = RUNTIME_MSG_DECISION; msg->accepted = 0; }
    else if (strcmp(type, "action_taken") == 0) { msg->type = RUNTIME_MSG_DECISION; msg->accepted = 1; }
    else if (strcmp(type, "action_rejected") == 0) { msg->type = RUNTIME_MSG_DECISION; msg->accepted = 0; }
    else msg->type = RUNTIME_MSG_UNKNOWN;

    extract_json_string(json_line, "battle_id", msg->battle_id, sizeof(msg->battle_id));
    extract_json_string(json_line, "format", msg->format, sizeof(msg->format));
    msg->is_doubles = extract_json_bool(json_line, "is_doubles", 0);
    msg->request_id = extract_json_number(json_line, "request_id", 0);
    msg->seq = extract_json_number(json_line, "seq", 0);
    msg->reward = extract_json_float(json_line, "reward", 0.0f);
    msg->action = extract_json_number(json_line, "action", -1);
    msg->action2 = extract_json_number(json_line, "action2", -1);
    if (msg->accepted == 0 && strcmp(type, "decision_rejected") == 0) {
        /* keep explicit rejected marker */
    } else if (msg->accepted == 0 && strcmp(type, "action_rejected") == 0) {
        /* keep explicit action rejection marker */
    } else if (msg->accepted == 1 && strcmp(type, "decision_accepted") == 0) {
        /* keep explicit accepted marker */
    } else if (msg->accepted == 1 && strcmp(type, "action_taken") == 0) {
        /* keep explicit action taken marker */
    } else if (msg->accepted == -1 && strcmp(type, "decision_proposed") == 0) {
        /* keep explicit proposed marker */
    } else {
        msg->accepted = extract_json_bool(json_line, "accepted", msg->accepted);
    }
    extract_json_string(json_line, "result", msg->result, sizeof(msg->result));
    extract_json_string(json_line, "line", msg->line, sizeof(msg->line));
    extract_json_string(json_line, "message", msg->message, sizeof(msg->message));
    extract_json_string(json_line, "command", msg->command, sizeof(msg->command));
    extract_json_block(json_line, "payload", '{', '}', msg->payload, sizeof(msg->payload));
    return msg->type != RUNTIME_MSG_UNKNOWN;
}

int runtime_emit_ready_json(char* out, size_t out_len) {
    return snprintf(out, out_len, "{\"type\":\"ready\",\"capabilities\":{\"doubles\":true,\"training\":true}}") > 0;
}

int runtime_emit_action_json(char* out, size_t out_len, const char* battle_id, int request_id, int action, int action2, const char* command) {
    return snprintf(out, out_len, "{\"type\":\"action\",\"battle_id\":\"%s\",\"request_id\":%d,\"action\":%d,\"action2\":%d,\"command\":\"%s\"}",
        battle_id ? battle_id : "", request_id, action, action2, command ? command : "") > 0;
}

int runtime_emit_log_json(char* out, size_t out_len, const char* message) {
    return snprintf(out, out_len, "{\"type\":\"log\",\"message\":\"%s\"}", message ? message : "") > 0;
}

int runtime_emit_error_json(char* out, size_t out_len, const char* battle_id, const char* message) {
    return snprintf(out, out_len, "{\"type\":\"error\",\"battle_id\":\"%s\",\"message\":\"%s\"}",
        battle_id ? battle_id : "", message ? message : "") > 0;
}
