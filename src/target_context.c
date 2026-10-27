#include "target_context.h"
#include "id_tables.h"
#include <string.h>

_Static_assert(sizeof(TargetContext) == 32, "target context cache layout changed");

static const ObsPokemon* active_pokemon(const ObsPokemon* team, int slot) {
    int i;
    for (i = 0; i < OBS_TEAM_SIZE; ++i)
        if (team[i].active && !team[i].fainted && team[i].active_slot == slot + 1) return &team[i];
    return NULL;
}

void target_context_collect(const Observation* observation, const ParsedRequest* request,
    const FactorizedActionChoice* choice, int slot, TargetContext* context
) {
    int move = slot ? choice->slot1_move_index : choice->slot0_move_index;
    int partner = 1 - slot, switching = partner ? choice->slot1_kind : choice->slot0_kind;
    int switch_index = partner ? choice->slot1_switch_index : choice->slot0_switch_index;
    const ObsPokemon* pokemon[4] = {active_pokemon(observation->self_team, slot),
        active_pokemon(observation->self_team, partner), active_pokemon(observation->opp_team, 0),
        active_pokemon(observation->opp_team, 1)};
    int target, type;
    memset(context, 0, sizeof(*context));
    if (switching == FACTORIZED_ACTION_SWITCH) pokemon[1] = &observation->self_team[switch_index];
    context->move_id = (uint16_t)request->active[slot].move_id[move];
    type = move_type_from_id(context->move_id);
    if (type > 0 && type < OBS_NUM_TYPES) { context->move_type_id = (unsigned char)type; context->move_type_known = 1; }
    for (target = 0; target < 4; ++target) {
        TargetContextPokemon* out = &context->targets[target];
        const ObsPokemon* p = pokemon[target];
        out->position = (unsigned char)target;
        out->friendly = target < 2;
        if (!p || p->fainted) continue;
        out->present = 1;
        if (p->type1_known_mode && p->type1_id > 0 && p->type1_id < OBS_NUM_TYPES) {
            out->type1_id = (unsigned char)p->type1_id; out->type1_known_mode = p->type1_known_mode;
        }
        if (p->type2_known_mode && p->type2_id >= 0 && p->type2_id < OBS_NUM_TYPES) {
            out->type2_id = (unsigned char)p->type2_id; out->type2_known_mode = p->type2_known_mode;
        }
    }
}

int target_context_valid(const TargetContext* c) {
    int t;
    if (!c || c->move_id == 0 || c->move_type_id >= OBS_NUM_TYPES ||
            c->move_type_known > 1 || (!!c->move_type_known != !!c->move_type_id)) return 0;
    for (t = 0; t < 4; ++t) {
        const TargetContextPokemon* p = &c->targets[t];
        if (p->position != t || p->friendly != (t < 2) || p->present > 1 ||
                p->type1_id >= OBS_NUM_TYPES || p->type2_id >= OBS_NUM_TYPES ||
                p->type1_known_mode > 2 || p->type2_known_mode > 2 ||
                (!!p->type1_known_mode != !!p->type1_id) || (!p->type2_known_mode && p->type2_id) ||
                (!p->present && (p->type1_id || p->type2_id || p->type1_known_mode || p->type2_known_mode))) return 0;
    }
    return 1;
}

void target_context_write(FILE* out, const TargetContext* c) {
    int t;
    fprintf(out, "{\"version\":1,\"move_id\":%u,\"move_type_id\":%u,\"move_type_known\":%s,\"targets\":[",
        c->move_id, c->move_type_id, c->move_type_known ? "true" : "false");
    for (t = 0; t < 4; ++t) {
        const TargetContextPokemon* p = &c->targets[t];
        fprintf(out, "%s{\"position\":%u,\"present\":%s,\"friendly\":%s,\"type1_id\":%u,"
            "\"type1_known_mode\":%u,\"type2_id\":%u,\"type2_known_mode\":%u}", t ? "," : "",
            p->position, p->present ? "true" : "false", p->friendly ? "true" : "false",
            p->type1_id, p->type1_known_mode, p->type2_id, p->type2_known_mode);
    }
    fputs("]}", out);
}

size_t target_context_feature_indices(const TargetContext* c, int target, size_t indices[2]) {
    const TargetContextPokemon* p = &c->targets[target];
    size_t offset;
    if (!c->move_type_known || !c->move_type_id || !p->present || !p->type1_known_mode ||
            !p->type1_id || !p->type2_known_mode) return 0;
    offset = (p->friendly * OBS_NUM_TYPES + c->move_type_id) * OBS_NUM_TYPES;
    indices[0] = offset + p->type1_id;
    if (p->type2_id && p->type2_id != p->type1_id) { indices[1] = offset + p->type2_id; return 2; }
    return 1;
}
