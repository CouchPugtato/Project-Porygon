#ifndef TARGET_CONTEXT_FIT_H
#define TARGET_CONTEXT_FIT_H
#include <stdio.h>
#include "gru_model.h"

/* Diagnostic only: prints JSON, never publishes weights or alters the parent. */
int target_context_compare(const char* dataset, const GruModel* parent, FILE* report);
/* Explicit evaluation export; caller must verify reproduction before publication. */
int target_context_compare_for_evaluation(const char* dataset, const GruModel* parent, FILE* report, const char* output);
#endif
