#include "action_record_learner.h"
#include "action_optimizer_state.h"
#include "checkpoint.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Internal Node-to-native pack, not a replacement for bundle/JSON replay
 * verification. uint32/float32 fields are little-endian on this native ABI. */
int action_record_learner_cli(int argc, char** argv) {
    if (argc != 8) { fputs("action-record diagnostic expects batch/reference/behavior/actor/optimizer/output-prefix\n", stderr); return 1; }
    FILE* file = fopen(argv[2], "rb");
    char magic[9], pin[65] = {0}, path[4096];
    uint32_t header[10];
    GruModel *reference = NULL, *behavior = NULL, *actor = NULL;
    GruActionHeadOptimizer* optimizer = NULL;
    ActionLearningRecord* rows = NULL;
    float* hidden = NULL;
    int ok = file && fread(magic, 1, 9, file) == 9 && !memcmp(magic, "PGACTREC1", 9) &&
        fread(header, sizeof(uint32_t), 10, file) == 10 && header[0] == 0x01020304 &&
        header[1] > 0 && header[1] <= 4096 && header[2] > 0 && header[2] <= 128 && fread(pin, 1, 64, file) == 64;
    if (!ok) goto done;
    rows = calloc(header[2], sizeof(*rows)); hidden = malloc(header[1] * header[2] * sizeof(float));
    if (!rows || !hidden) { ok = 0; goto done; }
    for (size_t i = 0; ok && i < header[2]; ++i) {
        unsigned char choices[14];
        rows[i].hidden = hidden + i * header[1];
        ok = fread(hidden + i * header[1], sizeof(float), header[1], file) == header[1] &&
            fread(rows[i].legal, 1, 28, file) == 28 && fread(choices, 1, 14, file) == 14 &&
            fread(&rows[i].terminal_return, sizeof(float), 1, file) == 1 &&
            fread(&rows[i].behavior_log_probability, sizeof(double), 1, file) == 1 &&
            fread(&rows[i].conditional_target_log_probability, sizeof(double), 1, file) == 1;
        FactorizedActionChoice* c = &rows[i].choice;
        if (ok) {
            c->slot0_has_action = choices[0]; c->slot0_kind = choices[1]; c->slot0_move_index = choices[2]; c->slot0_switch_index = choices[3];
            c->slot0_use_tera = choices[4]; c->slot0_target_index = choices[5]; c->slot0_target_mask = choices[6];
            c->slot1_has_action = choices[7]; c->slot1_kind = choices[8]; c->slot1_move_index = choices[9]; c->slot1_switch_index = choices[10];
            c->slot1_use_tera = choices[11]; c->slot1_target_index = choices[12]; c->slot1_target_mask = choices[13];
        }
    }
    if (!ok || fgetc(file) != EOF || ferror(file)) { ok = 0; goto done; }
    reference = checkpoint_load(argv[3], NULL); behavior = checkpoint_load(argv[4], NULL);
    actor = checkpoint_load(!strcmp(argv[5], "-") ? argv[4] : argv[5], NULL);
    if (!reference || !behavior || !actor || gru_model_hidden_dim(actor) != header[1]) { ok = 0; goto done; }
    optimizer = !strcmp(argv[6], "-") ? gru_action_head_optimizer_create(actor) :
        action_optimizer_state_load(argv[6], actor, reference, pin, NULL);
    ActionLearningOptions options = {header[7], header[5], header[6], 0, 0};
    memcpy(&options.learning_rate, header + 8, sizeof(float));
    memcpy(&options.max_abs_log_ratio, header + 9, sizeof(float));
    ActionLearningProgress progress = {header[3], header[4], 0, 0};
    ok = optimizer && action_record_learn(reference, behavior, actor, &optimizer, rows, header[2], &options, &progress);
    if (!ok) goto done;
    TrainerCheckpointState state = {0}; state.step = progress.updates;
    if (snprintf(path, sizeof(path), "%s.chk", argv[7]) >= (int)sizeof(path)) { ok = 0; goto done; }
    ok = checkpoint_save(path, actor, &state);
    if (snprintf(path, sizeof(path), "%s.opt", argv[7]) >= (int)sizeof(path)) { ok = 0; goto done; }
    ok = ok && action_optimizer_state_save(path, optimizer, reference, pin, NULL);
    if (ok) printf("{\"diagnostic\":\"action_record_learner\",\"training_eligible\":false,\"ppo_ready\":false,"
        "\"cursor\":%zu,\"updates\":%zu,\"guard_stopped\":%s,\"max_abs_log_ratio\":%.17g,\"rejected_abs_log_ratio\":%.17g}\n",
        progress.cursor, progress.updates, progress.guard_stopped ? "true" : "false", progress.max_abs_log_ratio, progress.rejected_abs_log_ratio);
done:
    if (file) fclose(file);
    if (!ok) fputs("action-record diagnostic rejected its input or update; no generation committed\n", stderr);
    free(rows); free(hidden); gru_action_head_optimizer_free(optimizer);
    gru_model_destroy(reference); gru_model_destroy(behavior); gru_model_destroy(actor);
    return ok ? 0 : 1;
}
