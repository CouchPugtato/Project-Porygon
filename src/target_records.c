#include "target_records.h"
#include "target_behavior.h"
#include <inttypes.h>
#include <string.h>

void target_records_write(FILE* out, const EnvRuntime* runtime, const EnvSession* session, int supported) {
    _Static_assert(sizeof(float) == sizeof(uint32_t), "record schema requires float32");
    fprintf(out, "{\"type\":\"%s\",\"version\":1,\"battle_id\":\"%s\",\"request_id\":%d,"
        "\"decision_index\":%zu,\"training_eligible\":false,\"behavior_policy\":",
        supported ? "bundle_decision" : "bundle_decision_skipped", session->battle_id,
        session->last_request_id, session->decision_count);
    env_runtime_write_target_identity(runtime, out);
    if (!supported) { fputs(",\"reason\":\"unsupported_likelihood_or_acceptance\"}\n", out); return; }
    fprintf(out, ",\"action\":%d,\"action2\":%d,\"command\":\"%s\",\"hidden_dim\":%zu,\"hidden_f32_hex\":\"",
        session->pending_action, session->pending_action2, session->pending_command, gru_model_hidden_dim(runtime->model));
    size_t h;
    for (h = 0; h < gru_model_hidden_dim(runtime->model); ++h) {
        uint32_t bits;
        memcpy(&bits, session->hidden_state + h, sizeof(bits));
        fprintf(out, "%08" PRIx32, bits);
    }
    fputs("\",\"legal_mask\":[", out);
    int i, slot;
    for (i = 0; i < OBS_NUM_ACTIONS; ++i) fprintf(out, "%s%u", i ? "," : "", session->observation.legal_mask[i]);
    fputs("],\"choice\":[", out);
    const FactorizedActionChoice* c = &session->pending_factorized_choice;
    fprintf(out, "[%u,%u,%u,%u,%u,%u,%u],[%u,%u,%u,%u,%u,%u,%u]", c->slot0_has_action,
        c->slot0_kind, c->slot0_move_index, c->slot0_switch_index, c->slot0_use_tera, c->slot0_target_index, c->slot0_target_mask,
        c->slot1_has_action, c->slot1_kind, c->slot1_move_index, c->slot1_switch_index,
        c->slot1_use_tera, c->slot1_target_index, c->slot1_target_mask);
    fputs("],\"public_context\":[", out);
    for (slot = 0; slot < 2; ++slot) {
        int has = slot ? c->slot1_has_action : c->slot0_has_action;
        int kind = slot ? c->slot1_kind : c->slot0_kind;
        if (slot) fputc(',', out);
        if (has && kind == FACTORIZED_ACTION_MOVE) {
            TargetContext context;
            target_context_collect(&session->observation, &session->parsed_request, c, slot, &context);
            target_context_write(out, &context);
        } else fputs("null", out);
    }
    fputs("],\"behavior\":", out);
    target_behavior_write(out, c, session->pending_learned_target_slots,
        session->pending_target_distribution, &session->pending_behavior_log_prob);
    fputs("}\n", out);
}
