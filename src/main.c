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

#ifdef HAVE_NATIVE_SHOWDOWN_CLIENT
#include "showdown_client.h"
#endif

static GruModel* create_default_model(void) {
    return gru_model_create(observation_flat_size(), 128, OBS_NUM_ACTIONS);
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
    FILE* replay_file = NULL;
    const char* replay_path = getenv("PORYGON_REPLAY_PATH");

    memset(&state, 0, sizeof(state));
    if (checkpoint_path) {
        model = checkpoint_load(checkpoint_path, &state);
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
        env_runtime_handle_message(&runtime, &msg, stdout);
    }
    env_runtime_free(&runtime);
    fclose(replay_file);
    gru_model_destroy(model);
    return 0;
}

static int train_from_replay_file(const char* replay_path, const char* checkpoint_path, int rl_mode) {
    FILE* f;
    char line[16384];
    GruModel* model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    TrainerCheckpointState checkpoint_state;
    size_t i;

    if (!replay_path || !checkpoint_path) {
        return 1;
    }
    f = fopen(replay_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open replay file '%s': %s\n", replay_path, strerror(errno));
        return 1;
    }
    memset(&checkpoint_state, 0, sizeof(checkpoint_state));
    model = checkpoint_load(checkpoint_path, &checkpoint_state);
    if (!model) {
        model = create_default_model();
    }
    if (!model) {
        fclose(f);
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
        return 1;
    }
    while (fgets(line, sizeof(line), f)) {
        RuntimeMessage msg;
        runtime_message_init(&msg);
        if (!runtime_message_parse(&msg, line)) {
            continue;
        }
        env_runtime_handle_message(&runtime, &msg, NULL);
    }
    fclose(f);

    for (i = 0; i < runtime.count; ++i) {
        if (rl_mode) {
            gru_trainer_policy_gradient_episode(&trainer, model, &runtime.sessions[i].episode);
        } else {
            gru_trainer_supervised_episode(&trainer, model, &runtime.sessions[i].episode);
        }
    }
    checkpoint_state = gru_trainer_checkpoint_state(&trainer);
    if (!checkpoint_save(checkpoint_path, model, &checkpoint_state)) {
        fprintf(stderr, "Failed to save checkpoint '%s'\n", checkpoint_path);
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 1;
    }
    printf("trained mode=%s step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f sessions=%zu\n",
        rl_mode ? "rl" : "supervised",
        trainer.step,
        trainer.last_action_loss,
        trainer.last_value_loss,
        trainer.last_accuracy,
        runtime.count);
    env_runtime_free(&runtime);
    gru_model_destroy(model);
    return 0;
}

int main(int argc, char** argv) {
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

