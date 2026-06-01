#ifndef RUNTIME_PROTOCOL_H
#define RUNTIME_PROTOCOL_H

#include <stddef.h>

#define RUNTIME_BATTLE_ID_LEN 192
#define RUNTIME_FORMAT_LEN 64
#define RUNTIME_RESULT_LEN 16
#define RUNTIME_LINE_LEN 2048
#define RUNTIME_PAYLOAD_LEN 8192
#define RUNTIME_MESSAGE_LEN 256
#define RUNTIME_COMMAND_LEN 256

typedef enum {
    RUNTIME_MSG_UNKNOWN = 0,
    RUNTIME_MSG_BATTLE_START,
    RUNTIME_MSG_REQUEST,
    RUNTIME_MSG_EVENT,
    RUNTIME_MSG_TERMINAL,
    RUNTIME_MSG_BATTLE_END,
    RUNTIME_MSG_ERROR,
    RUNTIME_MSG_HEARTBEAT,
    RUNTIME_MSG_DECISION
} RuntimeMessageType;

typedef struct {
    RuntimeMessageType type;
    char battle_id[RUNTIME_BATTLE_ID_LEN];
    char format[RUNTIME_FORMAT_LEN];
    int is_doubles;
    int request_id;
    int seq;
    float reward;
    int action;
    int accepted;
    char result[RUNTIME_RESULT_LEN];
    char line[RUNTIME_LINE_LEN];
    char payload[RUNTIME_PAYLOAD_LEN];
    char message[RUNTIME_MESSAGE_LEN];
    char command[RUNTIME_COMMAND_LEN];
} RuntimeMessage;

void runtime_message_init(RuntimeMessage* msg);
int runtime_message_parse(RuntimeMessage* msg, const char* json_line);
int runtime_emit_ready_json(char* out, size_t out_len);
int runtime_emit_action_json(char* out, size_t out_len, const char* battle_id, int request_id, int action, const char* command);
int runtime_emit_log_json(char* out, size_t out_len, const char* message);
int runtime_emit_error_json(char* out, size_t out_len, const char* battle_id, const char* message);

#endif
