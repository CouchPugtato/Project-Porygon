#ifndef TARGET_RECORDS_H
#define TARGET_RECORDS_H

#include "env_session.h"

/* Called only after an accepted bundle decision, before pending state clears. */
void target_records_write(FILE* out, const EnvRuntime* runtime, const EnvSession* session, int supported);

#endif
