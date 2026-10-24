#include "tactical_policy.h"

#include "tactical_scoring.h"

#include <math.h>
#include <string.h>

enum { MAX_SLOT_CHOICES = 14 * FACTORIZED_TARGET_DIM + 1 };

typedef struct {
    int action;
    unsigned char target;
    unsigned char target_mask;
} SlotChoice;

static const ObsPokemon* active_pokemon(const ObsPokemon* team, int slot) {
    int i;
    for (i = 0; i < OBS_TEAM_SIZE; ++i) {
        if (team[i].active && !team[i].fainted && team[i].active_slot == slot + 1)
            return &team[i];
    }
    return NULL;
}

static unsigned char visible_target_mask(
    const Observation* observation, const ParsedRequest* request, int slot, int move
) {
    unsigned char mask = build_move_target_mask(request, slot, move);
    if (!active_pokemon(observation->opp_team, 0))
        mask &= (unsigned char)~FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_LEFT);
    if (!active_pokemon(observation->opp_team, 1))
        mask &= (unsigned char)~FACTORIZED_TARGET_BIT(FACTORIZED_TARGET_FOE_RIGHT);
    return mask;
}

static size_t slot_choices(
    const Observation* observation, const ParsedRequest* request,
    const ActionMask* mask, int slot, SlotChoice* choices
) {
    enum ObsAction actions[OBS_NUM_ACTIONS];
    size_t count = 0, action_count, i;
    if (!parsed_request_slot_needs_choice(request, slot)) {
        choices[0] = (SlotChoice){-1, 0, 0};
        return 1;
    }
    action_count = collect_slot_legal_actions(request, mask, slot, actions, OBS_NUM_ACTIONS);
    for (i = 0; i < action_count; ++i) {
        int local_action = (int)actions[i] - slot * 14;
        unsigned char targets = 0;
        if (local_action < 8) {
            unsigned char required = build_move_target_mask(request, slot, local_action % 4);
            targets = visible_target_mask(observation, request, slot, local_action % 4);
            if (required && !targets) continue;
        }
        if (!targets) {
            choices[count++] = (SlotChoice){actions[i], 0, 0};
        } else {
            int target;
            for (target = 0; target < FACTORIZED_TARGET_DIM; ++target) {
                if (targets & FACTORIZED_TARGET_BIT(target))
                    choices[count++] = (SlotChoice){actions[i], (unsigned char)target, targets};
            }
        }
    }
    /* Showdown permits a pass when two replacements share one remaining bench
     * Pokemon. Selection below still fills as many required slots as possible. */
    if (request->forced_switch_any && request->force_switch[slot])
        choices[count++] = (SlotChoice){-1, 0, 0};
    return count;
}

static int pair_valid(const ParsedRequest* request, const SlotChoice* a, const SlotChoice* b) {
    return request_choice_pair_is_valid(request,
        a->action >= 0, (enum ObsAction)a->action,
        b->action >= 0, (enum ObsAction)b->action);
}

int tactical_select_choice(
    const Observation* observation, const ParsedRequest* request,
    TacticalChoiceEvaluator evaluate, void* context, TacticalDecision* decision
) {
    ActionMask mask;
    SlotChoice choices[2][MAX_SLOT_CHOICES];
    size_t counts[2], i, j;
    int maximum_actions = 0, found = 0;
    if (!decision) return 0;
    memset(decision, 0, sizeof(*decision));
    if (!observation || !request || !evaluate || request->wait || request->team_preview ||
            !build_action_mask_from_request(&mask, request)) return 0;
    counts[0] = slot_choices(observation, request, &mask, 0, choices[0]);
    counts[1] = slot_choices(observation, request, &mask, 1, choices[1]);
    for (i = 0; i < counts[0]; ++i) {
        for (j = 0; j < counts[1]; ++j) {
            int actions = (choices[0][i].action >= 0) + (choices[1][j].action >= 0);
            if (pair_valid(request, &choices[0][i], &choices[1][j]) && actions > maximum_actions)
                maximum_actions = actions;
        }
    }
    if (!maximum_actions) return 0;
    for (i = 0; i < counts[0]; ++i) {
        for (j = 0; j < counts[1]; ++j) {
            FactorizedActionChoice choice;
            TacticalChoiceEvaluation evaluation = {0};
            if ((choices[0][i].action >= 0) + (choices[1][j].action >= 0) != maximum_actions ||
                    !pair_valid(request, &choices[0][i], &choices[1][j])) continue;
            factorized_action_choice_from_flat_actions(
                &choice, choices[0][i].action, choices[1][j].action);
            choice.slot0_target_index = choices[0][i].target;
            choice.slot0_target_mask = choices[0][i].target_mask;
            choice.slot1_target_index = choices[1][j].target;
            choice.slot1_target_mask = choices[1][j].target_mask;
            if (!evaluate(observation, request, &choice, &evaluation, context) ||
                    !isfinite(evaluation.score) || !isfinite(evaluation.opponent_power) ||
                    !isfinite(evaluation.friendly_power)) continue;
            ++decision->evaluated_choices;
            if (!found || evaluation.score > decision->evaluation.score) {
                decision->choice = choice;
                decision->evaluation = evaluation;
                found = 1;
            }
        }
    }
    return found;
}

static const ObsPokemon* partner_after_switch(
    const Observation* observation, const FactorizedActionChoice* choice, int slot
) {
    int partner = 1 - slot;
    int switching = partner == 0
        ? choice->slot0_has_action && choice->slot0_kind == FACTORIZED_ACTION_SWITCH
        : choice->slot1_has_action && choice->slot1_kind == FACTORIZED_ACTION_SWITCH;
    int index = partner == 0 ? choice->slot0_switch_index : choice->slot1_switch_index;
    return switching ? &observation->self_team[index]
        : active_pokemon(observation->self_team, partner);
}

static int add_attack_power(
    const MoveSemanticMetadata* move, int type, const ObsPokemon* attacker,
    const ObsPokemon* target, int friendly, float spread_multiplier,
    TacticalChoiceEvaluation* evaluation
) {
    TacticalAttackScore attack;
    if (!target) return 1;
    if (!attacker || !tactical_score_attack(move, type, attacker, target, &attack)) {
        return 0;
    }
    if (friendly) evaluation->friendly_power += attack.expected_power * spread_multiplier;
    else evaluation->opponent_power += attack.expected_power * spread_multiplier;
    return 1;
}

int tactical_evaluate_power(
    const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, TacticalChoiceEvaluation* evaluation, void* context
) {
    int slot;
    (void)context;
    if (!observation || !request || !choice || !evaluation) return 0;
    memset(evaluation, 0, sizeof(*evaluation));
    for (slot = 0; slot < 2; ++slot) {
        int has_action = slot == 0 ? choice->slot0_has_action : choice->slot1_has_action;
        int kind = slot == 0 ? choice->slot0_kind : choice->slot1_kind;
        int move_index = slot == 0 ? choice->slot0_move_index : choice->slot1_move_index;
        int tera = slot == 0 ? choice->slot0_use_tera : choice->slot1_use_tera;
        int target_index = slot == 0 ? choice->slot0_target_index : choice->slot1_target_index;
        int attacker_index = request->active_team_idx[slot];
        const ObsPokemon* attacker = active_pokemon(observation->self_team, slot);
        const ObsPokemon* ally = partner_after_switch(observation, choice, slot);
        MoveSemanticMetadata move;
        int move_id, type, fully_scored = 1;
        ParsedMoveTarget target_kind;
        if (!has_action || kind == FACTORIZED_ACTION_SWITCH) continue;
        move_id = request->active[slot].move_id[move_index];
        if (tera || (request->active_team_idx_known[slot] &&
                request->side_tera_used[attacker_index]) ||
                !move_semantics_from_id(move_id, &move) || !move.damaging || move.base_power <= 0) {
            ++evaluation->unscored_moves;
            continue;
        }
        type = move_type_from_id(move_id);
        target_kind = request->active[slot].move_target[move_index];
        if (target_kind == REQUEST_TARGET_ALL_ADJACENT_FOES ||
                target_kind == REQUEST_TARGET_ALL_ADJACENT) {
            const ObsPokemon* left = active_pokemon(observation->opp_team, 0);
            const ObsPokemon* right = active_pokemon(observation->opp_team, 1);
            int hits_ally = target_kind == REQUEST_TARGET_ALL_ADJACENT && ally != NULL;
            int targets = (left != NULL) + (right != NULL) + hits_ally;
            float multiplier = targets > 1 ? 0.75f : 1.0f;
            fully_scored &= add_attack_power(&move, type, attacker, left, 0, multiplier, evaluation);
            fully_scored &= add_attack_power(&move, type, attacker, right, 0, multiplier, evaluation);
            if (hits_ally) fully_scored &=
                add_attack_power(&move, type, attacker, ally, 1, multiplier, evaluation);
        } else if (build_move_target_mask(request, slot, move_index)) {
            const ObsPokemon* target;
            int friendly = target_index < FACTORIZED_TARGET_FOE_LEFT;
            if (!friendly) target = active_pokemon(observation->opp_team,
                target_index - FACTORIZED_TARGET_FOE_LEFT);
            else target = target_index == FACTORIZED_TARGET_SELF ? attacker : ally;
            fully_scored = add_attack_power(&move, type, attacker, target, friendly, 1.0f, evaluation);
        } else {
            fully_scored = 0;
        }
        if (!fully_scored) ++evaluation->unscored_moves;
    }
    evaluation->score = evaluation->opponent_power - evaluation->friendly_power;
    return 1;
}
