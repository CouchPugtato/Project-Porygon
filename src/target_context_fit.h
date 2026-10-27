#ifndef TARGET_CONTEXT_FIT_H
#define TARGET_CONTEXT_FIT_H
#include <stdio.h>
#include "gru_model.h"

/* Diagnostic only: prints JSON, never publishes weights or alters the parent. */
int target_context_compare(const char* dataset, const GruModel* parent, FILE* report);
#endif
