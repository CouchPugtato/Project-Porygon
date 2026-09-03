#include "validation_split.h"

uint64_t validation_split_hash(const char* battle_id, unsigned int seed) {
    uint64_t hash = UINT64_C(14695981039346656037);
    unsigned int seed_byte;
    const unsigned char* p = (const unsigned char*)(battle_id ? battle_id : "");

    for (seed_byte = 0; seed_byte < 4u; ++seed_byte) {
        hash ^= (uint64_t)((seed >> (seed_byte * 8u)) & 0xffu);
        hash *= UINT64_C(1099511628211);
    }
    while (*p) {
        hash ^= (uint64_t)*p++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int validation_split_contains(const char* battle_id, unsigned int seed) {
    return validation_split_hash(battle_id, seed) % UINT64_C(10) == UINT64_C(0);
}
