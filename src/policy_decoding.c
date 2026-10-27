#include "policy_decoding.h"
#include "tactical_policy.h"

#include <math.h>

int policy_decode_target_argmax(const float policy[4], unsigned char legal) {
    int target, selected = -1;
    if (!policy || !legal || (legal & ~15)) return -1;
    for (target = 0; target < 4; ++target) if (legal & (1 << target)) {
        if (!isfinite(policy[target]) || policy[target] < 0) return -1;
        if (selected < 0 || policy[target] > policy[selected]) selected = target;
    }
    return selected;
}

typedef struct {
    const FactorizedPolicySnapshot* policy;
    const unsigned char* legal;
} GreedyContext;

int policy_target_distribution(const float policy[4], unsigned char legal, int argmax, float distribution[4]) {
    int first = policy_decode_target_argmax(policy, legal), target;
    float total = 0;
    if (first < 0 || !distribution) return 0;
    for (target = 0; target < 4; ++target) if (legal & (1u << target)) total += policy[target];
    if (!isfinite(total)) return 0;
    if (!argmax && total <= 1.0e-8f) {
        for (first = 0; !(legal & (1u << first)); ++first) {}
    }
    for (target = 0; target < 4; ++target) distribution[target] = argmax || total <= 1.0e-8f ?
        (target == first ? 1.0f : 0.0f) : (legal & (1u << target) ? policy[target] / total : 0.0f);
    return 1;
}

static double masked_probability(const float* policy, const unsigned char* mask, int count, int selected) {
    double total = 0.0;
    int first = -1, i;
    if (selected < 0 || selected >= count || !mask[selected]) return NAN;
    for (i = 0; i < count; ++i) {
        if (!mask[i]) continue;
        if (!isfinite(policy[i]) || policy[i] < 0.0f) return NAN;
        if (first < 0) first = i;
        total += policy[i];
    }
    /* Match the sampler's deterministic first-legal fallback for tiny mass. */
    return total <= 1.0e-8 ? (selected == first ? 1.0 : 0.0) : policy[selected] / total;
}

int policy_choice_log_probability(const FactorizedPolicySnapshot* p, const unsigned char* legal,
    const FactorizedActionChoice* choice, double* log_probability
) {
    int actions[2], slot;
    double result = 0;
    if (!p || !legal || !choice || !log_probability ||
            !factorized_action_choice_to_flat_actions(choice, &actions[0], &actions[1]) ||
            (actions[0] < 0 && actions[1] < 0)) return 0;
    if ((actions[0] >= 0 && !legal[actions[0]]) || (actions[1] >= 0 && !legal[actions[1]])) return 0;
    if (actions[0] >= 0 && actions[1] >= 0) {
        unsigned char mask[FACTORIZED_JOINT_DIM];
        int i;
        if (!p->has_joint_policy) return 0;
        for (i = 0; i < FACTORIZED_JOINT_DIM; ++i) {
            if (!isfinite(p->joint_policy[i]) || p->joint_policy[i] < 0) return 0;
            mask[i] = p->joint_policy[i] > 0;
        }
        int selected = actions[0] * FACTORIZED_LOCAL_ACTION_DIM + actions[1] - FACTORIZED_LOCAL_ACTION_DIM;
        /* A joint zero is outside the sampler's support, including forbidden
         * duplicate switches and double Tera. */
        if (!mask[selected]) return 0;
        result += log(masked_probability(p->joint_policy, mask, FACTORIZED_JOINT_DIM, selected));
    }
    for (slot = 0; slot < 2; ++slot) {
        int base = slot * FACTORIZED_LOCAL_ACTION_DIM;
        int local = actions[slot] - base;
        const float* kind_policy = slot ? p->slot1_kind_policy : p->slot0_kind_policy;
        const float* move_policy = slot ? p->slot1_move_policy : p->slot0_move_policy;
        const float* tera_policy = slot ? p->slot1_tera_policy : p->slot0_tera_policy;
        const float* switch_policy = slot ? p->slot1_switch_policy : p->slot0_switch_policy;
        const float* target_policy = slot ? p->slot1_target_policy : p->slot0_target_policy;
        unsigned char targets = slot ? choice->slot1_target_mask : choice->slot0_target_mask;
        int target = slot ? choice->slot1_target_index : choice->slot0_target_index;
        if (actions[slot] < 0) continue;
        if (actions[1 - slot] < 0) {
            unsigned char kind_mask[2] = {0}, move_mask[4] = {0}, switch_mask[6] = {0};
            int i;
            for (i = 0; i < 8; ++i) if (legal[base + i]) kind_mask[0] = move_mask[i % 4] = 1;
            for (i = 0; i < 6; ++i) if (legal[base + 8 + i]) kind_mask[1] = switch_mask[i] = 1;
            result += log(masked_probability(kind_policy, kind_mask, 2, local < 8 ? 0 : 1));
            if (local < 8) {
                unsigned char tera_mask[2] = {legal[base + local % 4], legal[base + 4 + local % 4]};
                result += log(masked_probability(move_policy, move_mask, 4, local % 4));
                result += log(masked_probability(tera_policy, tera_mask, 2, local / 4));
            } else result += log(masked_probability(switch_policy, switch_mask, 6, local - 8));
        }
        if (local < 8 && targets) {
            unsigned char mask[FACTORIZED_TARGET_DIM];
            factorized_target_mask_to_array(targets, mask);
            if (target < 0 || target >= FACTORIZED_TARGET_DIM || !mask[target]) return 0;
            result += log(masked_probability(target_policy, mask, FACTORIZED_TARGET_DIM, target));
        }
    }
    if (isnan(result) || result > 0) return 0;
    *log_probability = result;
    return 1;
}

static int evaluate_policy_choice(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    const GreedyContext* greedy = context;
    double log_probability;
    (void)observation;
    (void)request;
    if (!policy_choice_log_probability(greedy->policy, greedy->legal, choice, &log_probability) ||
            !isfinite(log_probability)) return 0;
    evaluation->score = (float)log_probability;
    return 1;
}

int policy_decode_greedy(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedPolicySnapshot* policy, FactorizedActionChoice* choice
) {
    TacticalDecision decision;
    GreedyContext context;
    if (!observation || !request || !policy || !choice) return 0;
    context.policy = policy;
    context.legal = observation->legal_mask;
    /* Reuse legal action/target enumeration, not the tactical power evaluator. */
    if (!tactical_select_choice(observation, request, evaluate_policy_choice, &context, &decision)) return 0;
    *choice = decision.choice;
    return 1;
}
