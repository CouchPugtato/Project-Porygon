#include "target_behavior.h"
#include <inttypes.h>

void target_behavior_identity_write(FILE* out, uint64_t parent, uint64_t sidecar,
    int contextual, int argmax, int slot_local
) {
    fprintf(out, "{\"version\":1,\"parameter_fingerprint_algorithm\":\"fnv1a64-native\","
        "\"checkpoint_parameters\":\"%016" PRIx64 "\",\"sidecar_parameters\":\"%016" PRIx64 "\","
        "\"context_version\":%d,\"action_decoder\":\"sample\",\"target_decoder\":\"%s\","
        "\"eligibility\":\"%s\"}", parent, sidecar, contextual ? 1 : 0,
        argmax ? "argmax" : "sample", slot_local ? "slot-local-v1" : "whole-turn-v1");
}

void target_behavior_write(FILE* out, const FactorizedActionChoice* choice,
    unsigned char learned_slots, const float policies[2][4]
) {
    int slot;
    fputs("{\"version\":1,\"scope\":\"conditional_targets_only\",\"ppo_ready\":false,\"slots\":[", out);
    for (slot = 0; slot < 2; ++slot) {
        unsigned char legal = slot ? choice->slot1_target_mask : choice->slot0_target_mask;
        int selected = slot ? choice->slot1_target_index : choice->slot0_target_index;
        if (slot) fputc(',', out);
        if (!legal) { fputs("null", out); continue; }
        fprintf(out, "{\"source\":\"%s\",\"legal_mask\":%u,\"selected\":%d,"
            "\"probabilities\":[%.9g,%.9g,%.9g,%.9g],\"selected_probability\":%.9g}",
            learned_slots & (1u << slot) ? "sidecar" : "checkpoint", legal, selected,
            policies[slot][0], policies[slot][1], policies[slot][2], policies[slot][3],
            policies[slot][selected]);
    }
    fputs("]}", out);
}
