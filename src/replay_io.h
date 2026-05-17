#ifndef REPLAY_IO_H
#define REPLAY_IO_H

#include "runtime_protocol.h"

#include <stdio.h>

int replay_write_runtime_message(FILE* out, const RuntimeMessage* msg);
int replay_write_decision(FILE* out, const char* battle_id, int request_id, int action, const char* command);

#endif
