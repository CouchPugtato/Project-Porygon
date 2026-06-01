#include "checkpoint.h"
#include "env_session.h"
#include "gru_model.h"
#include "gru_trainer.h"
#include "id_tables.h"
#include "observation.h"
#include "runtime_protocol.h"
#include <errno.h>
#include <math.h>
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

static char* make_epoch_checkpoint_path(const char* base_path, size_t epoch_number) {
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
        ? snprintf(NULL, 0, "%.*s_epoch%zu%s", (int)stem_len, base_path, epoch_number, dot)
        : snprintf(NULL, 0, "%s_epoch%zu", base_path, epoch_number);
    if (needed <= 0) {
        return NULL;
    }
    out = (char*)malloc((size_t)needed + 1);
    if (!out) {
        return NULL;
    }
    if (dot) {
        snprintf(out, (size_t)needed + 1, "%.*s_epoch%zu%s", (int)stem_len, base_path, epoch_number, dot);
    } else {
        snprintf(out, (size_t)needed + 1, "%s_epoch%zu", base_path, epoch_number);
    }
    return out;
}

static double elapsed_seconds_since(clock_t start_clock) {
    return (double)(clock() - start_clock) / (double)CLOCKS_PER_SEC;
}

static int parse_epochs_arg(int argc, char** argv, int default_epochs) {
    int i;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--epochs") == 0) {
            int parsed = atoi(argv[i + 1]);
            return parsed > 0 ? parsed : default_epochs;
        }
    }
    return default_epochs;
}

static int is_validation_session(size_t index, size_t total_sessions) {
    if (total_sessions < 10) {
        return 0;
    }
    return (index % 10u) == 0u;
}

static int evaluate_supervised_episode(
    const GruTrainer* trainer,
    const GruModel* model,
    const Episode* episode,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out,
    size_t* labels_out
) {
    size_t t;
    size_t trained = 0;
    float action_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float accuracy_sum = 0.0f;
    size_t hidden_dim;
    size_t action_dim;
    float* hidden;
    float* policy;

    if (!trainer || !model || !episode) {
        return 0;
    }
    hidden_dim = gru_model_hidden_dim(model);
    action_dim = gru_model_num_actions(model);
    hidden = (float*)malloc(hidden_dim * sizeof(float));
    policy = (float*)malloc(action_dim * sizeof(float));
    if (!hidden || !policy) {
        free(hidden);
        free(policy);
        return 0;
    }

    for (t = 0; t < episode->count; ++t) {
        size_t start = (t + 1 > trainer->bptt_window) ? (t + 1 - trainer->bptt_window) : 0;
        size_t steps = (t - start) + 1;
        float value = 0.0f;
        int predicted_action;
        int labels[2];
        int li;

        labels[0] = episode->actions[t];
        labels[1] = episode->actions2[t];

        if (labels[0] < 0 && labels[1] < 0) {
            continue;
        }

        gru_model_zero_state(model, hidden);
        gru_model_forward_sequence(model, episode->observations + (start * episode->obs_dim), steps, hidden, policy, &value);
        predicted_action = gru_model_select_action(policy, NULL, action_dim);

        for (li = 0; li < 2; ++li) {
            int label = labels[li];
            float prob;
            float err;
            if (label < 0) {
                continue;
            }
            prob = policy[label] > 1.0e-8f ? policy[label] : 1.0e-8f;
            action_loss_sum += -logf(prob);
            err = value - episode->rewards[t];
            value_loss_sum += 0.5f * err * err;
            accuracy_sum += (predicted_action == label) ? 1.0f : 0.0f;
            ++trained;
        }
    }

    free(hidden);
    free(policy);

    if (trained == 0) {
        if (action_loss_out) *action_loss_out = 0.0f;
        if (value_loss_out) *value_loss_out = 0.0f;
        if (accuracy_out) *accuracy_out = 0.0f;
        if (labels_out) *labels_out = 0;
        return 1;
    }
    if (action_loss_out) *action_loss_out = action_loss_sum / (float)trained;
    if (value_loss_out) *value_loss_out = value_loss_sum / (float)trained;
    if (accuracy_out) *accuracy_out = accuracy_sum / (float)trained;
    if (labels_out) *labels_out = trained;
    return 1;
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

static int train_from_replay_file(const char* replay_path, const char* checkpoint_path, int rl_mode, int epochs) {
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
    size_t train_sessions = 0;
    size_t val_sessions = 0;
    int epoch;

    if (!replay_path || !checkpoint_path || epochs <= 0) {
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
    for (i = 0; i < runtime.count; ++i) {
        if (is_validation_session(i, runtime.count)) {
            ++val_sessions;
        } else {
            ++train_sessions;
        }
    }
    printf("[train] split train_sessions=%zu val_sessions=%zu epochs=%d\n", train_sessions, val_sessions, epochs);

    for (epoch = 1; epoch <= epochs; ++epoch) {
        size_t trained_in_epoch = 0;
        float val_action_loss_sum = 0.0f;
        float val_value_loss_sum = 0.0f;
        float val_accuracy_sum = 0.0f;
        size_t val_labels = 0;

        printf("[train] epoch %d/%d start\n", epoch, epochs);
        train_loop_start_clock = clock();

        for (i = 0; i < runtime.count; ++i) {
            if (is_validation_session(i, runtime.count)) {
                continue;
            }
            if (rl_mode) {
                gru_trainer_policy_gradient_episode(&trainer, model, &runtime.sessions[i].episode);
            } else {
                gru_trainer_supervised_episode(&trainer, model, &runtime.sessions[i].episode);
            }
            ++trained_in_epoch;
            {
                double elapsed = elapsed_seconds_since(train_loop_start_clock);
                double episodes_per_sec = elapsed > 0.0 ? (double)trained_in_epoch / elapsed : 0.0;
                double eta = episodes_per_sec > 0.0 ? (double)(train_sessions - trained_in_epoch) / episodes_per_sec : 0.0;
                printf("[train] epoch=%d episodes=%zu/%zu step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f\n",
                    epoch,
                    trained_in_epoch,
                    train_sessions,
                    trainer.step,
                    trainer.last_action_loss,
                    trainer.last_value_loss,
                    trainer.last_accuracy);
                printf("[train] epoch=%d elapsed=%.1fs episodes_per_sec=%.2f eta=%.1fs\n",
                    epoch,
                    elapsed,
                    episodes_per_sec,
                    eta);
            }
            if ((trained_in_epoch % 500u) == 0u || trained_in_epoch == train_sessions) {
                TrainerCheckpointState periodic_state = gru_trainer_checkpoint_state(&trainer);
                char* periodic_path = make_periodic_checkpoint_path(resolved_checkpoint_path, ((size_t)(epoch - 1) * train_sessions) + trained_in_epoch);
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

        if (val_sessions > 0 && !rl_mode) {
            for (i = 0; i < runtime.count; ++i) {
                float action_loss = 0.0f;
                float value_loss = 0.0f;
                float accuracy = 0.0f;
                size_t labels = 0;
                if (!is_validation_session(i, runtime.count)) {
                    continue;
                }
                if (!evaluate_supervised_episode(&trainer, model, &runtime.sessions[i].episode,
                        &action_loss, &value_loss, &accuracy, &labels)) {
                    fprintf(stderr, "Failed validation evaluation on session %zu\n", i);
                    env_runtime_free(&runtime);
                    gru_model_destroy(model);
                    free(resolved_checkpoint_path);
                    return 1;
                }
                val_action_loss_sum += action_loss * (float)labels;
                val_value_loss_sum += value_loss * (float)labels;
                val_accuracy_sum += accuracy * (float)labels;
                val_labels += labels;
            }
            if (val_labels > 0) {
                printf("[train] epoch=%d validation action_loss=%.4f value_loss=%.4f accuracy=%.4f labels=%zu\n",
                    epoch,
                    val_action_loss_sum / (float)val_labels,
                    val_value_loss_sum / (float)val_labels,
                    val_accuracy_sum / (float)val_labels,
                    val_labels);
            } else {
                printf("[train] epoch=%d validation skipped no labels\n", epoch);
            }
        }

        {
            TrainerCheckpointState epoch_state = gru_trainer_checkpoint_state(&trainer);
            char* epoch_path = make_epoch_checkpoint_path(resolved_checkpoint_path, (size_t)epoch);
            if (epoch_path) {
                if (checkpoint_save(epoch_path, model, &epoch_state)) {
                    printf("[train] saved epoch checkpoint %s\n", epoch_path);
                } else {
                    printf("[train] failed epoch checkpoint %s\n", epoch_path);
                }
                free(epoch_path);
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

static int clean_replay_file(const char* input_path, const char* output_path) {
    FILE* in;
    FILE* out;
    char line[16384];
    size_t kept = 0;
    size_t skipped = 0;

    if (!input_path || !output_path) {
        return 1;
    }
    in = fopen(input_path, "r");
    if (!in) {
        fprintf(stderr, "Failed to open replay file '%s': %s\n", input_path, strerror(errno));
        return 1;
    }
    out = fopen(output_path, "w");
    if (!out) {
        fclose(in);
        fprintf(stderr, "Failed to open clean replay output '%s': %s\n", output_path, strerror(errno));
        return 1;
    }

    while (fgets(line, sizeof(line), in)) {
        RuntimeMessage msg;
        runtime_message_init(&msg);
        if (!runtime_message_parse(&msg, line)) {
            ++skipped;
            continue;
        }
        if (msg.type == RUNTIME_MSG_DECISION && msg.accepted == 0) {
            ++skipped;
            continue;
        }
        if (strstr(line, "\"type\":\"decision_proposed\"")) {
            ++skipped;
            continue;
        }
        fputs(line, out);
        ++kept;
    }

    fclose(in);
    fclose(out);
    printf("[clean] input=%s output=%s kept=%zu skipped=%zu\n", input_path, output_path, kept, skipped);
    return 0;
}

int main(int argc, char** argv) {
    int epochs = parse_epochs_arg(argc, argv, 1);
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
        return train_from_replay_file(argv[2], argv[3], 0, epochs);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-rl") == 0) {
        return train_from_replay_file(argv[2], argv[3], 1, epochs);
    }
    if (argc >= 4 && strcmp(argv[1], "--clean-replay") == 0) {
        return clean_replay_file(argv[2], argv[3]);
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
        "  showdown_client --train-supervised <replay.jsonl> <checkpoint.bin> [--epochs N]\n"
        "  showdown_client --train-rl <replay.jsonl> <checkpoint.bin> [--epochs N]\n"
        "  showdown_client --clean-replay <input.jsonl> <output.jsonl>\n"
        "  Set PORYGON_DEMO_GRU=1 for the demo mode.\n"
        "Legacy native websocket mode is disabled; use the Python communicator for live Showdown.\n");
    return 1;
#endif
}

