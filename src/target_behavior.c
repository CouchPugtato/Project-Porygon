#include "target_behavior.h"
#include "policy_decoding.h"
#include "request_parser.h"
#include <inttypes.h>
#include <math.h>
#include <string.h>

int target_behavior_log_probability(const TargetSidecar* sidecar, int argmax, int slot_local,
    const FactorizedPolicySnapshot* actor_policy, const float* hidden,
    const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, double* log_probability, unsigned char* learned_slots
) {
    FactorizedPolicySnapshot policy;
    int actions[2], slot;
    unsigned char applied_slots = 0;
    if (!sidecar || !actor_policy || !hidden || !observation || !request || !choice || !log_probability ||
            !factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1])) return 0;
    if (!!choice->slot0_has_action != !!parsed_request_slot_needs_choice(request, 0) ||
            !!choice->slot1_has_action != !!parsed_request_slot_needs_choice(request, 1) ||
            !request_choice_pair_is_valid(request, choice->slot0_has_action, (enum ObsAction)actions[0],
                choice->slot1_has_action, (enum ObsAction)actions[1])) return 0;
    policy = *actor_policy;
    for (slot = 0; slot < 2; ++slot) {
        int move = slot ? choice->slot1_move_index : choice->slot0_move_index;
        unsigned char legal = slot ? choice->slot1_target_mask : choice->slot0_target_mask;
        int kind = slot ? choice->slot1_kind : choice->slot0_kind;
        if (actions[slot] < 0) continue;
        if (kind != FACTORIZED_ACTION_MOVE) { if (legal) return 0; continue; }
        if (move < 0 || move >= 4 || legal != build_move_target_mask(request, slot, move)) return 0;
        if (!legal) continue;
        float* targets = slot ? policy.slot1_target_policy : policy.slot0_target_policy;
        float effective[4];
        int supported = slot_local ?
            !target_sidecar_slot_support_reason(observation, request, choice, slot) :
            target_sidecar_supports(observation, request, choice);
        int applied = 0;
        if (supported) {
            if (sidecar->contextual) {
                TargetContext context;
                target_context_collect(observation, request, choice, slot, &context);
                applied = target_sidecar_predict_context(sidecar, hidden, slot, move, legal, &context, effective);
            } else applied = target_sidecar_predict(sidecar, hidden, slot, move, 15, effective);
            if (applied) memcpy(targets, effective, sizeof(effective));
        }
        if (!policy_target_distribution(targets, legal, applied && argmax, effective)) return 0;
        memcpy(targets, effective, sizeof(effective));
        if (applied) applied_slots |= (unsigned char)(1u << slot);
    }
    double result;
    if (!policy_choice_log_probability(&policy, observation->legal_mask, choice, &result)) return 0;
    *log_probability = result;
    if (learned_slots) *learned_slots = applied_slots;
    return 1;
}

void target_behavior_identity_write(FILE* out, uint64_t parent, uint64_t sidecar,
    int contextual, int argmax, int slot_local
) {
    fprintf(out, "{\"version\":1,\"parameter_fingerprint_algorithm\":\"fnv1a64-native\","
        "\"checkpoint_parameters\":\"%016" PRIx64 "\",\"sidecar_parameters\":\"%016" PRIx64 "\","
        "\"context_version\":%d,\"action_decoder\":\"sample\",\"target_decoder\":\"%s\","
        "\"eligibility\":\"%s\"}", parent, sidecar, contextual ? 1 : 0,
        argmax ? "argmax" : "sample", slot_local ? "slot-local-v1" : "whole-turn-v1");
}

void target_action_bundle_identity_write(FILE* out, uint64_t reference, uint64_t actor,
    uint64_t sidecar, int argmax, int slot_local, const char* manifest_sha256
) {
    fprintf(out, "{\"version\":2,\"kind\":\"frozen-target-action-actor-v1\","
        "\"parameter_fingerprint_algorithm\":\"fnv1a64-native\",\"reference_parameters\":\"%016" PRIx64 "\","
        "\"actor_parameters\":\"%016" PRIx64 "\",\"sidecar_parameters\":\"%016" PRIx64 "\","
        "\"manifest_sha256\":\"%s\",\"context_version\":1,\"action_decoder\":\"sample\","
        "\"target_decoder\":\"%s\",\"eligibility\":\"%s\"}", reference, actor, sidecar, manifest_sha256,
        argmax ? "argmax" : "sample", slot_local ? "slot-local-v1" : "whole-turn-v1");
}

void target_behavior_write(FILE* out, const FactorizedActionChoice* choice,
    unsigned char learned_slots, const float policies[2][4], const double* full_turn_log_probability
) {
    int slot;
    fputs("{\"version\":2,\"scope\":\"full_turn_diagnostic\",\"ppo_ready\":false,\"full_turn_log_probability\":", out);
    if (full_turn_log_probability && isfinite(*full_turn_log_probability))
        fprintf(out, "%.17g", *full_turn_log_probability);
    else fputs("null", out);
    fprintf(out, ",\"likelihood_status\":\"%s\",\"slots\":[",
        full_turn_log_probability ? "supported" : "unsupported");
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
