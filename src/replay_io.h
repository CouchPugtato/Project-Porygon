#ifndef REPLAY_IO_H
#define REPLAY_IO_H

#include "runtime_protocol.h"

#include <stdio.h>

int replay_write_runtime_message(FILE* out, const RuntimeMessage* msg);
int replay_write_decision_proposed(FILE* out, const char* battle_id, int request_id, int action, int action2, const char* command);
int replay_write_decision_accepted(FILE* out, const char* battle_id, int request_id, int action, int action2, const char* command);
int replay_write_decision_rejected(FILE* out, const char* battle_id, int request_id, int action, int action2, const char* command, const char* reason);

#endif
