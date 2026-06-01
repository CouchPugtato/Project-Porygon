#include "checkpoint.h"
#include "env_session.h"
#include "gru_model.h"
#include "gru_trainer.h"
#include "id_tables.h"
#include "observation.h"
#include "runtime_protocol.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef HAVE_NATIVE_SHOWDOWN_CLIENT
#include "showdown_client.h"
#endif

static GruModel* create_default_model(void) {
    return gru_model_create(observation_flat_size(), 128, OBS_NUM_ACTIONS);
}

static int has_path_separator(const char* path) {
    return path && (strchr(path, '/') != NULL || strchr(path, '\\') != NULL);
}

static char* resolve_checkpoint_path(const char* checkpoint_arg) {
    const char* default_name = "model.chk";
    const char* name = checkpoint_arg && *checkpoint_arg ? checkpoint_arg : default_name;
    char* resolved;
    int needed;

    if (has_path_separator(name)) {
        resolved = (char*)malloc(strlen(name) + 1);
        if (!resolved) {
            return NULL;
        }
        strcpy(resolved, name);
        return resolved;
    }

    needed = snprintf(NULL, 0, "models/%s", name);
    if (needed <= 0) {
        return NULL;
    }
    resolved = (char*)malloc((size_t)needed + 1);
    if (!resolved) {
        return NULL;
    }
    snprintf(resolved, (size_t)needed + 1, "models/%s", name);
    return resolved;
}

static char* make_periodic_checkpoint_path(const char* base_path, size_t episodes_done) {
    const char* dot;
    size_t stem_len;
    int needed;
    char* out;

    if (!base_path || !*base_path) {
        return NULL;
    }
    dot = strrchr(base_path, '.');
    stem_len = dot ? (size_t)(dot - base_path) : strlen(base_path);
    needed = dot
        ? snprintf(NULL, 0, "%.*s_ep%zu%s", (int)stem_len, base_path, episodes_done, dot)
        : snprintf(NULL, 0, "%s_ep%zu", base_path, episodes_done);
    if (needed <= 0) {
        return NULL;
    }
    out = (char*)malloc((size_t)needed + 1);
    if (!out) {
        return NULL;
    }
    if (dot) {
        snprintf(out, (size_t)needed + 1, "%.*s_ep%zu%s", (int)stem_len, base_path, episodes_done, dot);
    } else {
        snprintf(out, (size_t)needed + 1, "%s_ep%zu", base_path, episodes_done);
    }
    return out;
}

static double elapsed_seconds_since(clock_t start_clock) {
    return (double)(clock() - start_clock) / (double)CLOCKS_PER_SEC;
}

static int run_demo_gru(void) {
        Observation obs;
        size_t obs_dim = observation_flat_size();
        float* flat = (float*)malloc(obs_dim * sizeof(float));
        float* hidden = NULL;
        float* policy = NULL;
        float value = 0.0f;
        GruModel* model = NULL;
        Episode episode;
        int action = -1;

        if (!flat) {
            fprintf(stderr, "Failed to allocate observation buffer\n");
            return 1;
        }

        model = gru_model_create(obs_dim, 128, OBS_NUM_ACTIONS);
        hidden = (float*)malloc(gru_model_hidden_dim(model) * sizeof(float));
        policy = (float*)malloc(OBS_NUM_ACTIONS * sizeof(float));
        if (!model || !hidden || !policy || !episode_init(&episode, 4, obs_dim)) {
            fprintf(stderr, "Failed to initialize GRU demo\n");
            gru_model_destroy(model);
            free(hidden);
            free(policy);
            free(flat);
            return 1;
        }

        gru_model_zero_state(model, hidden);
        observation_fill_demo(&obs);
        for (int t = 0; t < 3; ++t) {
            obs.turn_norm = 0.1f + (0.1f * (float)t);
            obs.self_team[0].hp_frac = 0.82f - (0.08f * (float)t);
            obs.opp_team[0].hp_frac = 0.66f - (0.05f * (float)t);
            if (observation_flatten(flat, obs_dim, &obs) != obs_dim) {
                fprintf(stderr, "Failed to flatten observation\n");
                episode_free(&episode);
                gru_model_destroy(model);
                free(hidden);
                free(policy);
                free(flat);
                return 1;
            }
            episode_append(&episode, flat, -1, 0.0f, (uint8_t)(t == 2));
        }

        gru_model_forward_sequence(model, episode.observations, episode.count, hidden, policy, &value);
        action = gru_model_select_action(policy, obs.legal_mask, OBS_NUM_ACTIONS);
        episode.actions[episode.count - 1] = action;

        printf("obs_dim=%zu hidden_dim=%zu episode_steps=%zu action=%d value=%.4f\n",
            obs_dim,
            gru_model_hidden_dim(model),
            episode.count,
            action,
            value);

        episode_free(&episode);
        gru_model_destroy(model);
        free(hidden);
        free(policy);
        free(flat);
        return 0;
}

static int run_runtime_mode(const char* checkpoint_path) {
    GruModel* model = NULL;
    TrainerCheckpointState state;
    EnvRuntime runtime;
    char line[16384];
    char json[512];
    char* resolved_checkpoint_path = NULL;
    FILE* replay_file = NULL;
    const char* replay_path = getenv("PORYGON_REPLAY_PATH");

    memset(&state, 0, sizeof(state));
    if (checkpoint_path) {
        resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path);
        model = checkpoint_load(resolved_checkpoint_path, &state);
    }
    if (!model) {
        model = create_default_model();
    }
    if (!model) {
        fprintf(stderr, "Failed to initialize runtime model\n");
        return 1;
    }
    if (replay_path && *replay_path) {
        replay_file = fopen(replay_path, "a");
    }
    if (!env_runtime_init(&runtime, model, replay_file, 0)) {
        fprintf(stderr, "Failed to initialize runtime\n");
        fclose(replay_file);
        gru_model_destroy(model);
        return 1;
    }
    runtime_emit_ready_json(json, sizeof(json));
    puts(json);
    fflush(stdout);

    while (fgets(line, sizeof(line), stdin)) {
        RuntimeMessage msg;
        runtime_message_init(&msg);
        if (!runtime_message_parse(&msg, line)) {
            runtime_emit_error_json(json, sizeof(json), "", "invalid runtime message");
            puts(json);
            fflush(stdout);
            continue;
        }
        if (!env_runtime_handle_message(&runtime, &msg, stdout)) {
            char errbuf[256];
            snprintf(errbuf, sizeof(errbuf), "failed to handle runtime message type=%d request_id=%d",
                (int)msg.type, msg.request_id);
            runtime_emit_error_json(json, sizeof(json), msg.battle_id, errbuf);
            puts(json);
            fflush(stdout);
            fprintf(stderr, "%s\n", errbuf);
        }
    }
    env_runtime_free(&runtime);
    fclose(replay_file);
    gru_model_destroy(model);
    free(resolved_checkpoint_path);
    return 0;
}

static int train_from_replay_file(const char* replay_path, const char* checkpoint_path, int rl_mode) {
    FILE* f;
    char line[16384];
    char* resolved_checkpoint_path = NULL;
    GruModel* model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    TrainerCheckpointState checkpoint_state;
    size_t i;
    size_t lines_read = 0;
    size_t parsed_messages = 0;
    size_t invalid_lines = 0;
    clock_t train_start_clock;
    clock_t train_loop_start_clock;

    if (!replay_path || !checkpoint_path) {
        return 1;
    }
    f = fopen(replay_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open replay file '%s': %s\n", replay_path, strerror(errno));
        return 1;
    }
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path);
    if (!resolved_checkpoint_path) {
        fclose(f);
        fprintf(stderr, "Failed to resolve checkpoint path\n");
        return 1;
    }
    memset(&checkpoint_state, 0, sizeof(checkpoint_state));
    model = checkpoint_load(resolved_checkpoint_path, &checkpoint_state);
    if (!model) {
        model = create_default_model();
        printf("[train] starting fresh model -> %s\n", resolved_checkpoint_path);
    } else {
        printf("[train] loaded checkpoint %s step=%zu\n", resolved_checkpoint_path, checkpoint_state.step);
    }
    if (!model) {
        fclose(f);
        free(resolved_checkpoint_path);
        return 1;
    }
    gru_trainer_init(&trainer,
        checkpoint_state.learning_rate > 0.0f ? checkpoint_state.learning_rate : 0.01f,
        checkpoint_state.bptt_window ? checkpoint_state.bptt_window : 16,
        checkpoint_state.gradient_clip,
        checkpoint_state.seed);
    trainer.step = checkpoint_state.step;

    if (!env_runtime_init(&runtime, model, NULL, 1)) {
        fclose(f);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    train_start_clock = clock();
    while (fgets(line, sizeof(line), f)) {
        RuntimeMessage msg;
        ++lines_read;
        runtime_message_init(&msg);
        if (!runtime_message_parse(&msg, line)) {
            ++invalid_lines;
            continue;
        }
        ++parsed_messages;
        env_runtime_handle_message(&runtime, &msg, NULL);
        if ((lines_read % 50000u) == 0u) {
            double elapsed = elapsed_seconds_since(train_start_clock);
            double lines_per_sec = elapsed > 0.0 ? (double)lines_read / elapsed : 0.0;
            printf("[train] ingest lines=%zu parsed=%zu invalid=%zu sessions=%zu\n",
                lines_read, parsed_messages, invalid_lines, runtime.count);
            printf("[train] ingest elapsed=%.1fs lines_per_sec=%.1f\n", elapsed, lines_per_sec);
        }
    }
    fclose(f);
    printf("[train] ingest complete lines=%zu parsed=%zu invalid=%zu sessions=%zu\n",
        lines_read, parsed_messages, invalid_lines, runtime.count);
    train_loop_start_clock = clock();

    for (i = 0; i < runtime.count; ++i) {
        if (rl_mode) {
            gru_trainer_policy_gradient_episode(&trainer, model, &runtime.sessions[i].episode);
        } else {
            gru_trainer_supervised_episode(&trainer, model, &runtime.sessions[i].episode);
        }
        if (((i + 1u) % 1u) == 0u || (i + 1u) == runtime.count) {
            double elapsed = elapsed_seconds_since(train_loop_start_clock);
            double episodes_per_sec = elapsed > 0.0 ? (double)(i + 1u) / elapsed : 0.0;
            double eta = episodes_per_sec > 0.0 ? (double)(runtime.count - (i + 1u)) / episodes_per_sec : 0.0;
            printf("[train] episodes=%zu/%zu step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f\n",
                i + 1u,
                runtime.count,
                trainer.step,
                trainer.last_action_loss,
                trainer.last_value_loss,
                trainer.last_accuracy);
            printf("[train] elapsed=%.1fs episodes_per_sec=%.2f eta=%.1fs\n",
                elapsed,
                episodes_per_sec,
                eta);
        }
        if (((i + 1u) % 500u) == 0u || (i + 1u) == runtime.count) {
            TrainerCheckpointState periodic_state = gru_trainer_checkpoint_state(&trainer);
            char* periodic_path = make_periodic_checkpoint_path(resolved_checkpoint_path, i + 1u);
            if (periodic_path) {
                if (checkpoint_save(periodic_path, model, &periodic_state)) {
                    printf("[train] saved periodic checkpoint %s\n", periodic_path);
                } else {
                    printf("[train] failed periodic checkpoint %s\n", periodic_path);
                }
                free(periodic_path);
            }
        }
    }
    checkpoint_state = gru_trainer_checkpoint_state(&trainer);
    if (!checkpoint_save(resolved_checkpoint_path, model, &checkpoint_state)) {
        fprintf(stderr, "Failed to save checkpoint '%s'\n", resolved_checkpoint_path);
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    printf("trained mode=%s step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f sessions=%zu\n",
        rl_mode ? "rl" : "supervised",
        trainer.step,
        trainer.last_action_loss,
        trainer.last_value_loss,
        trainer.last_accuracy,
        runtime.count);
    printf("[train] saved checkpoint %s\n", resolved_checkpoint_path);
    env_runtime_free(&runtime);
    gru_model_destroy(model);
    free(resolved_checkpoint_path);
    return 0;
}

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));
    if (!id_tables_init()) {
        fprintf(stderr, "Failed to initialize ID tables\n");
        return 1;
    }
    if (getenv("PORYGON_DEMO_GRU")) {
        return run_demo_gru();
    }
    if (argc >= 2 && strcmp(argv[1], "--runtime") == 0) {
        return run_runtime_mode(argc >= 3 ? argv[2] : NULL);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-supervised") == 0) {
        return train_from_replay_file(argv[2], argv[3], 0);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-rl") == 0) {
        return train_from_replay_file(argv[2], argv[3], 1);
    }

#ifdef HAVE_NATIVE_SHOWDOWN_CLIENT
    const char* host = "sim3.psim.us"; // Showdown sim host (rotates)
    const int   port = 443; // TLS (wss://)
    const char* path = "/showdown/websocket"; // Raw WS endpoint

    struct ShowdownClient* cli = showdown_client_create(host, port, path);
    if (!cli) {
        fprintf(stderr, "Failed to create ShowdownClient\n");
        return 1;
    }

    int rc = showdown_client_run(cli);
    showdown_client_destroy(cli);
    return rc;
#else
    fprintf(stderr,
        "Usage:\n"
        "  showdown_client --runtime [checkpoint]\n"
        "  showdown_client --train-supervised <replay.jsonl> <checkpoint.bin>\n"
        "  showdown_client --train-rl <replay.jsonl> <checkpoint.bin>\n"
        "  Set PORYGON_DEMO_GRU=1 for the demo mode.\n"
        "Legacy native websocket mode is disabled; use the Python communicator for live Showdown.\n");
    return 1;
#endif
}

