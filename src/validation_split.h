#ifndef VALIDATION_SPLIT_H
#define VALIDATION_SPLIT_H

#include <stdint.h>

uint64_t validation_split_hash(const char* battle_id, unsigned int seed);
int validation_split_contains(const char* battle_id, unsigned int seed);

#endif
