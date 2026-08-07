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
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAVE_NATIVE_SHOWDOWN_CLIENT
#include "showdown_client.h"
#endif

#define SHOWDOWN_CLIENT_DEFAULT_ARGS_PATH "config/showdown_client.toml"
#define SHOWDOWN_CLIENT_REWARD_CONFIG_PATH "config/reward_weights.toml"

typedef struct {
    float terminal_win;
    float terminal_loss;
    float terminal_draw;
    float terminal_disconnect_or_forfeit;
    EnvDenseRewardConfig dense_additive;
} RewardConfig;

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

static char* make_best_checkpoint_path(const char* base_path) {
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
        ? snprintf(NULL, 0, "%.*s_best%s", (int)stem_len, base_path, dot)
        : snprintf(NULL, 0, "%s_best", base_path);
    if (needed <= 0) {
        return NULL;
    }
    out = (char*)malloc((size_t)needed + 1);
    if (!out) {
        return NULL;
    }
    if (dot) {
        snprintf(out, (size_t)needed + 1, "%.*s_best%s", (int)stem_len, base_path, dot);
    } else {
        snprintf(out, (size_t)needed + 1, "%s_best", base_path);
    }
    return out;
}


static double elapsed_seconds_since(clock_t start_clock) {
    return (double)(clock() - start_clock) / (double)CLOCKS_PER_SEC;
}

static double ema_update(double previous, double sample, double alpha) {
    if (previous <= 0.0) {
        return sample;
    }
    return (alpha * sample) + ((1.0 - alpha) * previous);
}

static char* trim_whitespace(char* text) {
    char* end;
    if (!text) {
        return NULL;
    }
    while (*text && isspace((unsigned char)*text)) {
        ++text;
    }
    if (!*text) {
        return text;
    }
    end = text + strlen(text) - 1;
    while (end > text && isspace((unsigned char)*end)) {
        *end-- = '\0';
    }
    return text;
}

static int append_default_arg(char*** args, int* count, int* capacity, const char* token) {
    char* copy;
    if (*count >= *capacity) {
        int new_capacity = *capacity > 0 ? (*capacity * 2) : 8;
        char** resized = (char**)realloc(*args, (size_t)new_capacity * sizeof(char*));
        if (!resized) {
            return 0;
        }
        *args = resized;
        *capacity = new_capacity;
    }
    copy = (char*)malloc(strlen(token) + 1);
    if (!copy) {
        return 0;
    }
    strcpy(copy, token);
    (*args)[(*count)++] = copy;
    return 1;
}

static char** load_default_args_file(const char* path, int* out_argc) {
    FILE* fp;
    char line[1024];
    char** args = NULL;
    int count = 0;
    int capacity = 0;
    if (out_argc) {
        *out_argc = 0;
    }
    if (!path) {
        return NULL;
    }
    fp = fopen(path, "r");
    if (!fp) {
        return NULL;
    }
    while (fgets(line, sizeof(line), fp)) {
        char* comment = strchr(line, '#');
        char* key;
        char* value;
        char* end;
        char flag[256];
        size_t i;
        if (comment) {
            *comment = '\0';
        }
        key = trim_whitespace(line);
        if (!key || !*key) {
            continue;
        }
        value = strchr(key, '=');
        if (!value) {
            goto fail;
        }
        *value++ = '\0';
        key = trim_whitespace(key);
        value = trim_whitespace(value);
        if (!key || !*key || !value || !*value) {
            goto fail;
        }
        snprintf(flag, sizeof(flag), "--%s", key);
        for (i = 2; flag[i]; ++i) {
            if (flag[i] == '_') {
                flag[i] = '-';
            }
        }
        if (strcmp(key, "battle_agent") == 0) {
            if (_stricmp(value, "true") == 0) {
                if (!append_default_arg(&args, &count, &capacity, flag)) {
                    goto fail;
                }
            } else if (_stricmp(value, "false") != 0) {
                goto fail;
            }
            continue;
        }
        if (!append_default_arg(&args, &count, &capacity, flag)) {
            goto fail;
        }
        if (*value == '"') {
            char token[1024];
            size_t token_len = 0;
            int closed = 0;
            ++value;
            while (*value) {
                char ch = *value++;
                if (ch == '"') {
                    closed = 1;
                    break;
                }
                if (ch == '\\') {
                    char escaped = *value++;
                    if (!escaped) {
                        goto fail;
                    }
                    if (escaped == 'n') ch = '\n';
                    else if (escaped == 't') ch = '\t';
                    else if (escaped == '"' || escaped == '\\') ch = escaped;
                    else ch = escaped;
                }
                if (token_len + 1 >= sizeof(token)) {
                    goto fail;
                }
                token[token_len++] = ch;
            }
            if (!closed) {
                goto fail;
            }
            token[token_len] = '\0';
            end = trim_whitespace(value);
            if (end && *end) {
                goto fail;
            }
            if (!append_default_arg(&args, &count, &capacity, token)) {
                goto fail;
            }
        } else if (_stricmp(value, "true") == 0 || _stricmp(value, "false") == 0) {
            if (!append_default_arg(&args, &count, &capacity, _stricmp(value, "true") == 0 ? "1" : "0")) {
                goto fail;
            }
        } else {
            if (!append_default_arg(&args, &count, &capacity, value)) {
                goto fail;
            }
        }
    }
    fclose(fp);
    if (out_argc) {
        *out_argc = count;
    }
    return args;
fail:
    fclose(fp);
    if (args) {
        int i;
        for (i = 0; i < count; ++i) {
            free(args[i]);
        }
        free(args);
    }
    return NULL;
}

static void free_default_args(char** args, int argc) {
    int i;
    if (!args) {
        return;
    }
    for (i = 0; i < argc; ++i) {
        free(args[i]);
    }
    free(args);
}

static void reward_config_defaults(RewardConfig* config) {
    if (!config) {
        return;
    }
    config->terminal_win = 1.0f;
    config->terminal_loss = -1.0f;
    config->terminal_draw = 0.0f;
    config->terminal_disconnect_or_forfeit = 0.0f;
    config->dense_additive.hp_swing_weight = 0.10f;
    config->dense_additive.faint_swing_weight = 0.25f;
    config->dense_additive.reward_clip = 0.40f;
}

static int assign_reward_config_value(RewardConfig* config, const char* key, const char* value_text) {
    float value;
    char* endptr = NULL;
    if (!config || !key || !value_text) {
        return 0;
    }
    value = strtof(value_text, &endptr);
    if (endptr == value_text || (endptr && *trim_whitespace(endptr) != '\0')) {
        return 0;
    }
    if (strcmp(key, "terminal_win") == 0) {
        config->terminal_win = value;
        return 1;
    }
    if (strcmp(key, "terminal_loss") == 0) {
        config->terminal_loss = value;
        return 1;
    }
    if (strcmp(key, "terminal_draw") == 0) {
        config->terminal_draw = value;
        return 1;
    }
    if (strcmp(key, "terminal_disconnect_or_forfeit") == 0) {
        config->terminal_disconnect_or_forfeit = value;
        return 1;
    }
    if (strcmp(key, "dense_additive_hp_swing_weight") == 0) {
        config->dense_additive.hp_swing_weight = value;
        return 1;
    }
    if (strcmp(key, "dense_additive_faint_swing_weight") == 0) {
        config->dense_additive.faint_swing_weight = value;
        return 1;
    }
    if (strcmp(key, "dense_additive_reward_clip") == 0) {
        config->dense_additive.reward_clip = value;
        return 1;
    }
    return 1;
}

static int load_reward_config_file(const char* path, RewardConfig* out_config) {
    FILE* fp;
    char line[1024];
    int line_number = 0;
    if (!out_config) {
        return 0;
    }
    reward_config_defaults(out_config);
    if (!path) {
        return 1;
    }
    fp = fopen(path, "r");
    if (!fp) {
        return 1;
    }
    while (fgets(line, sizeof(line), fp)) {
        char* comment = strchr(line, '#');
        char* key;
        char* value;
        line_number += 1;
        if (comment) {
            *comment = '\0';
        }
        key = trim_whitespace(line);
        if (!key || !*key) {
            continue;
        }
        value = strchr(key, '=');
        if (!value) {
            fclose(fp);
            fprintf(stderr, "invalid reward config %s:%d: expected key = value\n", path, line_number);
            return 0;
        }
        *value++ = '\0';
        key = trim_whitespace(key);
        value = trim_whitespace(value);
        if (!key || !*key || !value || !*value) {
            fclose(fp);
            fprintf(stderr, "invalid reward config %s:%d: empty key or value\n", path, line_number);
            return 0;
        }
        if (!assign_reward_config_value(out_config, key, value)) {
            fclose(fp);
            fprintf(stderr, "invalid reward config %s:%d: bad value for %s\n", path, line_number, key);
            return 0;
        }
    }
    fclose(fp);
    return 1;
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

static float parse_float_flag(int argc, char** argv, const char* name, float default_value) {
    int i;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            float parsed = (float)atof(argv[i + 1]);
            return parsed;
        }
    }
    return default_value;
}

static int parse_int_flag(int argc, char** argv, const char* name, int default_value) {
    int i;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return default_value;
}

static int parse_bool01_flag(int argc, char** argv, const char* name, int default_value) {
    int parsed = parse_int_flag(argc, argv, name, default_value);
    return parsed ? 1 : 0;
}

static int parse_reward_mode(const char* reward_mode_name, EnvRewardMode* reward_mode_out) {
    if (!reward_mode_name || !reward_mode_out) {
        return 0;
    }
    if (strcmp(reward_mode_name, "terminal") == 0) {
        *reward_mode_out = ENV_REWARD_TERMINAL;
        return 1;
    }
    if (strcmp(reward_mode_name, "dense_additive") == 0) {
        *reward_mode_out = ENV_REWARD_DENSE_ADDITIVE;
        return 1;
    }
    return 0;
}

static const char* parse_string_flag(int argc, char** argv, const char* name, const char* default_value) {
    int i;
    for (i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return default_value;
}

static int is_validation_session(size_t index, size_t total_sessions) {
    if (total_sessions < 10) {
        return 0;
    }
    return (index % 10u) == 0u;
}

static void shuffle_indices(size_t* indices, size_t count) {
    size_t i;
    if (!indices || count < 2) {
        return;
    }
    for (i = count - 1; i > 0; --i) {
        size_t j = (size_t)(rand() % (int)(i + 1));
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static int build_split_indices(
    size_t total_sessions,
    size_t** train_indices_out,
    size_t* train_count_out,
    size_t** val_indices_out,
    size_t* val_count_out
) {
    size_t* train_indices = NULL;
    size_t* val_indices = NULL;
    size_t train_count = 0;
    size_t val_count = 0;
    size_t i;

    if (!train_indices_out || !train_count_out || !val_indices_out || !val_count_out) {
        return 0;
    }

    train_indices = (size_t*)malloc((total_sessions > 0 ? total_sessions : 1) * sizeof(size_t));
    val_indices = (size_t*)malloc((total_sessions > 0 ? total_sessions : 1) * sizeof(size_t));
    if (!train_indices || !val_indices) {
        free(train_indices);
        free(val_indices);
        return 0;
    }

    for (i = 0; i < total_sessions; ++i) {
        if (is_validation_session(i, total_sessions)) {
            val_indices[val_count++] = i;
        } else {
            train_indices[train_count++] = i;
        }
    }

    *train_indices_out = train_indices;
    *train_count_out = train_count;
    *val_indices_out = val_indices;
    *val_count_out = val_count;
    return 1;
}

static int episode_has_labels(const Episode* episode) {
    size_t t;
    if (!episode) {
        return 0;
    }
    for (t = 0; t < episode->count; ++t) {
        if (episode->actions[t] >= 0 || episode->actions2[t] >= 0) {
            return 1;
        }
    }
    return 0;
}

static void filter_labeled_train_indices(
    const EnvRuntime* runtime,
    size_t* train_indices,
    size_t* train_count_io
) {
    size_t read_i;
    size_t write_i = 0;

    if (!runtime || !train_indices || !train_count_io) {
        return;
    }

    for (read_i = 0; read_i < *train_count_io; ++read_i) {
        size_t session_index = train_indices[read_i];
        if (session_index < runtime->count && episode_has_labels(&runtime->sessions[session_index].episode)) {
            train_indices[write_i++] = session_index;
        }
    }

    *train_count_io = write_i;
}

static int evaluate_supervised_episode(
    const GruTrainer* trainer,
    const GruModel* model,
    const Episode* episode,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out,
    size_t* labels_out,
    float* top3_accuracy_out,
    float* action1_accuracy_out,
    float* action2_accuracy_out,
    size_t* action1_labels_out,
    size_t* action2_labels_out,
    size_t* skipped_steps_out
) {
    size_t t;
    size_t trained = 0;
    float action_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float accuracy_sum = 0.0f;
    float top3_accuracy_sum = 0.0f;
    float action1_accuracy_sum = 0.0f;
    float action2_accuracy_sum = 0.0f;
    size_t action1_labels = 0;
    size_t action2_labels = 0;
    size_t skipped_steps = 0;
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
            ++skipped_steps;
            continue;
        }

        gru_model_zero_state(model, hidden);
        gru_model_forward_sequence(model, episode->observations + (start * episode->obs_dim), steps, hidden, policy, &value);
        predicted_action = gru_model_select_action(policy, NULL, action_dim);

        for (li = 0; li < 2; ++li) {
            int label = labels[li];
            float prob;
            float err;
            int top_hits = 0;
            size_t k;
            if (label < 0) {
                continue;
            }
            prob = policy[label] > 1.0e-8f ? policy[label] : 1.0e-8f;
            action_loss_sum += -logf(prob);
            err = value - episode->rewards[t];
            value_loss_sum += 0.5f * err * err;
            accuracy_sum += (predicted_action == label) ? 1.0f : 0.0f;
            for (k = 0; k < action_dim; ++k) {
                if (policy[k] > prob) {
                    ++top_hits;
                    if (top_hits >= 3) {
                        break;
                    }
                }
            }
            top3_accuracy_sum += (top_hits < 3) ? 1.0f : 0.0f;
            if (li == 0) {
                action1_accuracy_sum += (predicted_action == label) ? 1.0f : 0.0f;
                ++action1_labels;
            } else {
                action2_accuracy_sum += (predicted_action == label) ? 1.0f : 0.0f;
                ++action2_labels;
            }
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
        if (top3_accuracy_out) *top3_accuracy_out = 0.0f;
        if (action1_accuracy_out) *action1_accuracy_out = 0.0f;
        if (action2_accuracy_out) *action2_accuracy_out = 0.0f;
        if (action1_labels_out) *action1_labels_out = 0;
        if (action2_labels_out) *action2_labels_out = 0;
        if (skipped_steps_out) *skipped_steps_out = skipped_steps;
        return 1;
    }
    if (action_loss_out) *action_loss_out = action_loss_sum / (float)trained;
    if (value_loss_out) *value_loss_out = value_loss_sum / (float)trained;
    if (accuracy_out) *accuracy_out = accuracy_sum / (float)trained;
    if (labels_out) *labels_out = trained;
    if (top3_accuracy_out) *top3_accuracy_out = top3_accuracy_sum / (float)trained;
    if (action1_accuracy_out) *action1_accuracy_out = action1_labels > 0 ? action1_accuracy_sum / (float)action1_labels : 0.0f;
    if (action2_accuracy_out) *action2_accuracy_out = action2_labels > 0 ? action2_accuracy_sum / (float)action2_labels : 0.0f;
    if (action1_labels_out) *action1_labels_out = action1_labels;
    if (action2_labels_out) *action2_labels_out = action2_labels;
    if (skipped_steps_out) *skipped_steps_out = skipped_steps;
    return 1;
}

static int evaluate_supervised_split(
    const GruTrainer* trainer,
    const GruModel* model,
    const EnvRuntime* runtime,
    const size_t* indices,
    size_t index_count,
    float* action_loss_out,
    float* value_loss_out,
    float* accuracy_out,
    size_t* labels_out,
    size_t* sessions_out,
    float* top3_accuracy_out,
    float* action1_accuracy_out,
    float* action2_accuracy_out,
    size_t* action1_labels_out,
    size_t* action2_labels_out,
    size_t* skipped_steps_out
) {
    size_t i;
    float action_loss_sum = 0.0f;
    float value_loss_sum = 0.0f;
    float accuracy_sum = 0.0f;
    float top3_accuracy_sum = 0.0f;
    float action1_accuracy_sum = 0.0f;
    float action2_accuracy_sum = 0.0f;
    size_t total_labels = 0;
    size_t total_sessions = 0;
    size_t total_action1_labels = 0;
    size_t total_action2_labels = 0;
    size_t total_skipped_steps = 0;

    if (!trainer || !model || !runtime) {
        return 0;
    }
    for (i = 0; i < index_count; ++i) {
        float action_loss = 0.0f;
        float value_loss = 0.0f;
        float accuracy = 0.0f;
        float top3_accuracy = 0.0f;
        float action1_accuracy = 0.0f;
        float action2_accuracy = 0.0f;
        size_t labels = 0;
        size_t action1_labels = 0;
        size_t action2_labels = 0;
        size_t skipped_steps = 0;
        size_t session_index = indices[i];
        ++total_sessions;
        if (!evaluate_supervised_episode(trainer, model, &runtime->sessions[session_index].episode,
                &action_loss, &value_loss, &accuracy, &labels,
                &top3_accuracy, &action1_accuracy, &action2_accuracy,
                &action1_labels, &action2_labels, &skipped_steps)) {
            return 0;
        }
        action_loss_sum += action_loss * (float)labels;
        value_loss_sum += value_loss * (float)labels;
        accuracy_sum += accuracy * (float)labels;
        top3_accuracy_sum += top3_accuracy * (float)labels;
        action1_accuracy_sum += action1_accuracy * (float)action1_labels;
        action2_accuracy_sum += action2_accuracy * (float)action2_labels;
        total_labels += labels;
        total_action1_labels += action1_labels;
        total_action2_labels += action2_labels;
        total_skipped_steps += skipped_steps;
    }

    if (action_loss_out) *action_loss_out = total_labels > 0 ? action_loss_sum / (float)total_labels : 0.0f;
    if (value_loss_out) *value_loss_out = total_labels > 0 ? value_loss_sum / (float)total_labels : 0.0f;
    if (accuracy_out) *accuracy_out = total_labels > 0 ? accuracy_sum / (float)total_labels : 0.0f;
    if (labels_out) *labels_out = total_labels;
    if (sessions_out) *sessions_out = total_sessions;
    if (top3_accuracy_out) *top3_accuracy_out = total_labels > 0 ? top3_accuracy_sum / (float)total_labels : 0.0f;
    if (action1_accuracy_out) *action1_accuracy_out = total_action1_labels > 0 ? action1_accuracy_sum / (float)total_action1_labels : 0.0f;
    if (action2_accuracy_out) *action2_accuracy_out = total_action2_labels > 0 ? action2_accuracy_sum / (float)total_action2_labels : 0.0f;
    if (action1_labels_out) *action1_labels_out = total_action1_labels;
    if (action2_labels_out) *action2_labels_out = total_action2_labels;
    if (skipped_steps_out) *skipped_steps_out = total_skipped_steps;
    return 1;
}

static int load_runtime_from_replay_file(
    const char* replay_path,
    GruModel* model,
    EnvRuntime* runtime,
    EnvRewardMode reward_mode,
    const EnvDenseRewardConfig* dense_reward_config
) {
    FILE* f;
    char line[16384];
    size_t lines_read = 0;
    size_t parsed_messages = 0;
    size_t invalid_lines = 0;
    clock_t ingest_start_clock;

    if (!replay_path || !model || !runtime) {
        return 0;
    }
    f = fopen(replay_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open replay file '%s': %s\n", replay_path, strerror(errno));
        return 0;
    }
    if (!env_runtime_init(runtime, model, NULL, 1, reward_mode, dense_reward_config)) {
        fclose(f);
        return 0;
    }

    ingest_start_clock = clock();
    while (fgets(line, sizeof(line), f)) {
        RuntimeMessage msg;
        ++lines_read;
        runtime_message_init(&msg);
        if (!runtime_message_parse(&msg, line)) {
            ++invalid_lines;
            continue;
        }
        ++parsed_messages;
        env_runtime_handle_message(runtime, &msg, NULL);
        if ((lines_read % 50000u) == 0u) {
            double elapsed = elapsed_seconds_since(ingest_start_clock);
            double lines_per_sec = elapsed > 0.0 ? (double)lines_read / elapsed : 0.0;
            printf("[train] ingest lines=%zu parsed=%zu invalid=%zu sessions=%zu\n",
                lines_read, parsed_messages, invalid_lines, runtime->count);
            printf("[train] ingest elapsed=%.1fs lines_per_sec=%.1f\n", elapsed, lines_per_sec);
        }
    }
    fclose(f);
    printf("[train] ingest complete lines=%zu parsed=%zu invalid=%zu sessions=%zu\n",
        lines_read, parsed_messages, invalid_lines, runtime->count);
    printf("[train] accepted_labels direct=%zu reconstructed=%zu failed=%zu\n",
        runtime->accepted_label_direct_count,
        runtime->accepted_label_reconstructed_count,
        runtime->accepted_label_failed_count);
    return 1;
}

static int read_line_dynamic(FILE* f, char** buffer, size_t* capacity) {
    size_t length = 0;
    int ch;
    char* resized;
    if (!f || !buffer || !capacity) {
        return -1;
    }
    if (!*buffer || *capacity == 0) {
        *capacity = 4096;
        *buffer = (char*)malloc(*capacity);
        if (!*buffer) {
            *capacity = 0;
            return -1;
        }
    }
    while ((ch = fgetc(f)) != EOF) {
        if (length + 2 >= *capacity) {
            *capacity *= 2u;
            resized = (char*)realloc(*buffer, *capacity);
            if (!resized) {
                return -1;
            }
            *buffer = resized;
        }
        (*buffer)[length++] = (char)ch;
        if (ch == '\n') {
            break;
        }
    }
    if (length == 0 && ch == EOF) {
        return 0;
    }
    (*buffer)[length] = '\0';
    return 1;
}

static int load_runtime_from_episode_batch_file(
    const char* input_path,
    GruModel* model,
    EnvRuntime* runtime,
    EnvRewardMode reward_mode,
    const EnvDenseRewardConfig* dense_reward_config
) {
    FILE* f;
    char* line = NULL;
    size_t line_capacity = 0;
    size_t lines_read = 0;
    size_t parsed_episodes = 0;
    size_t invalid_lines = 0;
    clock_t ingest_start_clock;

    if (!input_path || !model || !runtime) {
        return 0;
    }
    f = fopen(input_path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open episode batch file '%s': %s\n", input_path, strerror(errno));
        return 0;
    }
    if (!env_runtime_init(runtime, model, NULL, 1, reward_mode, dense_reward_config)) {
        fclose(f);
        return 0;
    }

    ingest_start_clock = clock();
    while (1) {
        Episode episode;
        EnvSession* new_sessions;
        char battle_id[RUNTIME_BATTLE_ID_LEN];
        char policy_tag[256];
        int read_rc = read_line_dynamic(f, &line, &line_capacity);
        if (read_rc < 0) {
            free(line);
            fclose(f);
            env_runtime_free(runtime);
            return 0;
        }
        if (read_rc == 0) {
            break;
        }
        ++lines_read;
        memset(&episode, 0, sizeof(episode));
        battle_id[0] = '\0';
        policy_tag[0] = '\0';
        if (!episode_parse_json_record(line, &episode, battle_id, sizeof(battle_id), policy_tag, sizeof(policy_tag))) {
            ++invalid_lines;
            continue;
        }
        if (runtime->count == runtime->capacity) {
            size_t new_capacity = runtime->capacity ? (runtime->capacity * 2u) : 16u;
            new_sessions = (EnvSession*)realloc(runtime->sessions, new_capacity * sizeof(EnvSession));
            if (!new_sessions) {
                episode_free(&episode);
                fclose(f);
                env_runtime_free(runtime);
                return 0;
            }
            runtime->sessions = new_sessions;
            runtime->capacity = new_capacity;
        }
        memset(&runtime->sessions[runtime->count], 0, sizeof(EnvSession));
        runtime->sessions[runtime->count].episode = episode;
        strncpy(runtime->sessions[runtime->count].battle_id, battle_id, sizeof(runtime->sessions[runtime->count].battle_id) - 1);
        runtime->sessions[runtime->count].battle_id[sizeof(runtime->sessions[runtime->count].battle_id) - 1] = '\0';
        runtime->sessions[runtime->count].format_known = 1;
        runtime->sessions[runtime->count].terminal = 1;
        runtime->count += 1;
        ++parsed_episodes;
        if ((lines_read % 1000u) == 0u) {
            double elapsed = elapsed_seconds_since(ingest_start_clock);
            double lines_per_sec = elapsed > 0.0 ? (double)lines_read / elapsed : 0.0;
            printf("[train-live-rl] ingest lines=%zu parsed=%zu invalid=%zu sessions=%zu\n",
                lines_read, parsed_episodes, invalid_lines, runtime->count);
            printf("[train-live-rl] ingest elapsed=%.1fs lines_per_sec=%.1f\n", elapsed, lines_per_sec);
        }
    }
    free(line);
    fclose(f);
    printf("[train-live-rl] ingest complete lines=%zu parsed=%zu invalid=%zu sessions=%zu\n",
        lines_read, parsed_episodes, invalid_lines, runtime->count);
    return 1;
}

static EnvSession* find_runtime_session_by_id(EnvRuntime* runtime, const char* battle_id) {
    size_t i;
    if (!runtime || !battle_id) {
        return NULL;
    }
    for (i = 0; i < runtime->count; ++i) {
        if (strcmp(runtime->sessions[i].battle_id, battle_id) == 0) {
            return &runtime->sessions[i];
        }
    }
    return NULL;
}

static const char* runtime_message_type_name(RuntimeMessageType type) {
    switch (type) {
        case RUNTIME_MSG_BATTLE_START: return "battle_start";
        case RUNTIME_MSG_REQUEST: return "request";
        case RUNTIME_MSG_EVENT: return "event";
        case RUNTIME_MSG_TERMINAL: return "terminal";
        case RUNTIME_MSG_BATTLE_END: return "battle_end";
        case RUNTIME_MSG_ERROR: return "error";
        case RUNTIME_MSG_HEARTBEAT: return "heartbeat";
        case RUNTIME_MSG_DECISION: return "decision";
        default: return "unknown";
    }
}

static const char* knowledge_level_name(KnowledgeLevel knowledge) {
    switch (knowledge) {
        case KNOW_INFERRED: return "inferred";
        case KNOW_CONFIRMED: return "confirmed";
        case KNOW_UNKNOWN:
        default:
            return "unknown";
    }
}

static const char* obs_action_name(enum ObsAction action) {
    switch (action) {
        case OBS_A1_MOVE1: return "a1_move1";
        case OBS_A1_MOVE2: return "a1_move2";
        case OBS_A1_MOVE3: return "a1_move3";
        case OBS_A1_MOVE4: return "a1_move4";
        case OBS_A1_MOVE1_TERA: return "a1_move1_tera";
        case OBS_A1_MOVE2_TERA: return "a1_move2_tera";
        case OBS_A1_MOVE3_TERA: return "a1_move3_tera";
        case OBS_A1_MOVE4_TERA: return "a1_move4_tera";
        case OBS_A1_SWITCH1: return "a1_switch1";
        case OBS_A1_SWITCH2: return "a1_switch2";
        case OBS_A1_SWITCH3: return "a1_switch3";
        case OBS_A1_SWITCH4: return "a1_switch4";
        case OBS_A1_SWITCH5: return "a1_switch5";
        case OBS_A1_SWITCH6: return "a1_switch6";
        case OBS_A2_MOVE1: return "a2_move1";
        case OBS_A2_MOVE2: return "a2_move2";
        case OBS_A2_MOVE3: return "a2_move3";
        case OBS_A2_MOVE4: return "a2_move4";
        case OBS_A2_MOVE1_TERA: return "a2_move1_tera";
        case OBS_A2_MOVE2_TERA: return "a2_move2_tera";
        case OBS_A2_MOVE3_TERA: return "a2_move3_tera";
        case OBS_A2_MOVE4_TERA: return "a2_move4_tera";
        case OBS_A2_SWITCH1: return "a2_switch1";
        case OBS_A2_SWITCH2: return "a2_switch2";
        case OBS_A2_SWITCH3: return "a2_switch3";
        case OBS_A2_SWITCH4: return "a2_switch4";
        case OBS_A2_SWITCH5: return "a2_switch5";
        case OBS_A2_SWITCH6: return "a2_switch6";
        default: return "unknown";
    }
}

static void json_write_escaped(FILE* out, const char* text) {
    const unsigned char* p = (const unsigned char*)(text ? text : "");
    fputc('"', out);
    while (*p) {
        switch (*p) {
            case '\\': fputs("\\\\", out); break;
            case '"': fputs("\\\"", out); break;
            case '\n': fputs("\\n", out); break;
            case '\r': fputs("\\r", out); break;
            case '\t': fputs("\\t", out); break;
            default:
                if (*p < 0x20u) {
                    fprintf(out, "\\u%04x", (unsigned int)(*p));
                } else {
                    fputc((int)(*p), out);
                }
                break;
        }
        ++p;
    }
    fputc('"', out);
}

static void json_write_tracked_int(FILE* out, const TrackedInt* tracked, const char* (*name_fn)(int)) {
    const char* resolved_name = NULL;
    if (!tracked) {
        fputs("null", out);
        return;
    }
    if (name_fn && tracked->value > 0) {
        resolved_name = name_fn(tracked->value);
    }
    fputs("{\"value\":", out);
    fprintf(out, "%d", tracked->value);
    fputs(",\"knowledge\":", out);
    json_write_escaped(out, knowledge_level_name(tracked->knowledge));
    fputs(",\"name\":", out);
    json_write_escaped(out, resolved_name ? resolved_name : "");
    fputs("}", out);
}

static void json_write_int_array(FILE* out, const int* values, size_t count) {
    size_t i;
    fputc('[', out);
    for (i = 0; i < count; ++i) {
        if (i) fputc(',', out);
        fprintf(out, "%d", values[i]);
    }
    fputc(']', out);
}

static void json_write_float_array(FILE* out, const float* values, size_t count) {
    size_t i;
    fputc('[', out);
    for (i = 0; i < count; ++i) {
        if (i) fputc(',', out);
        fprintf(out, "%.6f", values[i]);
    }
    fputc(']', out);
}

static void json_write_raw_pokemon(FILE* out, const RawPokemon* mon) {
    int i;
    if (!mon) {
        fputs("null", out);
        return;
    }
    fputs("{\"known\":", out); fprintf(out, "%d", mon->known);
    fputs(",\"active\":", out); fprintf(out, "%d", mon->active);
    fputs(",\"active_slot\":", out); fprintf(out, "%d", mon->active_slot);
    fputs(",\"revealed\":", out); fprintf(out, "%d", mon->revealed);
    fputs(",\"fainted\":", out); fprintf(out, "%d", mon->fainted);
    fputs(",\"species\":", out); json_write_tracked_int(out, &mon->species_id, species_name_from_id);
    fputs(",\"item\":", out); json_write_tracked_int(out, &mon->item_id, item_name_from_id);
    fputs(",\"ability\":", out); json_write_tracked_int(out, &mon->ability_id, ability_name_from_id);
    fputs(",\"tera_type\":", out); json_write_tracked_int(out, &mon->tera_type_id, NULL);
    fputs(",\"tera_used\":", out); fprintf(out, "%d", mon->tera_used);
    fputs(",\"can_tera\":", out); fprintf(out, "%d", mon->can_tera);
    fputs(",\"transformed\":", out); fprintf(out, "%d", mon->transformed);
    fputs(",\"substitute_active\":", out); fprintf(out, "%d", mon->substitute_active);
    fprintf(out,
        ",\"base_stats\":{\"hp\":%d,\"atk\":%d,\"def\":%d,\"spa\":%d,\"spd\":%d,\"spe\":%d}",
        mon->base_hp_stat, mon->base_atk_stat, mon->base_def_stat,
        mon->base_spa_stat, mon->base_spd_stat, mon->base_spe_stat);
    fputs(",\"current_hp\":", out); fprintf(out, "%d", mon->current_hp);
    fputs(",\"max_hp\":", out); fprintf(out, "%d", mon->max_hp);
    fputs(",\"status\":", out); json_write_tracked_int(out, &mon->status_id, condition_name_from_id);
    fputs(",\"sleep_turns_elapsed\":", out); fprintf(out, "%d", mon->sleep_turns_elapsed);
    fputs(",\"toxic_counter\":", out); fprintf(out, "%d", mon->toxic_counter);
    fputs(",\"type1\":", out); json_write_tracked_int(out, &mon->type1_id, type_name_from_id);
    fputs(",\"type2\":", out); json_write_tracked_int(out, &mon->type2_id, type_name_from_id);
    fputs(",\"boosts\":", out); json_write_int_array(out, mon->boosts, 7);
    fputs(",\"moves\":[", out);
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (i) fputc(',', out);
        fputs("{\"move\":", out); json_write_tracked_int(out, &mon->move_ids[i], move_name_from_id);
        fputs(",\"move_type\":", out); json_write_tracked_int(out, &mon->move_type_ids[i], type_name_from_id);
        fputs(",\"effective_move_type\":", out); json_write_tracked_int(out, &mon->effective_move_type_ids[i], type_name_from_id);
        fputs(",\"known\":", out); fprintf(out, "%d", mon->move_known[i]);
        fputs(",\"pp\":", out); fprintf(out, "%d", mon->move_pp[i]);
        fputs(",\"max_pp\":", out); fprintf(out, "%d", mon->move_max_pp[i]);
        fputs(",\"disabled\":", out); fprintf(out, "%d", mon->move_disabled[i]);
        fputs(",\"maybe_disabled\":", out); fprintf(out, "%d", mon->move_maybe_disabled[i]);
        fputs("}", out);
    }
    fputs("]", out);
    fputs(",\"encore_active\":", out); fprintf(out, "%d", mon->encore_active);
    fputs(",\"encore_turns\":", out); fprintf(out, "%d", mon->encore_turns);
    fputs(",\"disable_active\":", out); fprintf(out, "%d", mon->disable_active);
    fputs(",\"disable_turns\":", out); fprintf(out, "%d", mon->disable_turns);
    fputs(",\"taunt_active\":", out); fprintf(out, "%d", mon->taunt_active);
    fputs(",\"taunt_turns\":", out); fprintf(out, "%d", mon->taunt_turns);
    fputs(",\"torment_active\":", out); fprintf(out, "%d", mon->torment_active);
    fputs(",\"torment_turns\":", out); fprintf(out, "%d", mon->torment_turns);
    fputs(",\"heal_block_active\":", out); fprintf(out, "%d", mon->heal_block_active);
    fputs(",\"heal_block_turns\":", out); fprintf(out, "%d", mon->heal_block_turns);
    fputs(",\"embargo_active\":", out); fprintf(out, "%d", mon->embargo_active);
    fputs(",\"embargo_turns\":", out); fprintf(out, "%d", mon->embargo_turns);
    fputs(",\"yawn_active\":", out); fprintf(out, "%d", mon->yawn_active);
    fputs(",\"yawn_turns\":", out); fprintf(out, "%d", mon->yawn_turns);
    fputs(",\"encore_move_slot\":", out); fprintf(out, "%d", mon->encore_move_slot);
    fputs(",\"disable_move_slot\":", out); fprintf(out, "%d", mon->disable_move_slot);
    fputs(",\"protect_active\":", out); fprintf(out, "%d", mon->protect_active);
    fputs(",\"protect_chain_count\":", out); fprintf(out, "%d", mon->protect_chain_count);
    fputs(",\"helping_hand_active\":", out); fprintf(out, "%d", mon->helping_hand_active);
    fputs(",\"flinch_active\":", out); fprintf(out, "%d", mon->flinch_active);
    fputs(",\"trapped\":", out); fprintf(out, "%d", mon->trapped);
    fputs(",\"maybe_trapped\":", out); fprintf(out, "%d", mon->maybe_trapped);
    fputs(",\"commanding_active\":", out); fprintf(out, "%d", mon->commanding_active);
    fputs(",\"reviving\":", out); fprintf(out, "%d", mon->reviving);
    fputs(",\"confusion_active\":", out); fprintf(out, "%d", mon->confusion_active);
    fputs(",\"confusion_turns\":", out); fprintf(out, "%d", mon->confusion_turns);
    fputs(",\"seed_active\":", out); fprintf(out, "%d", mon->seed_active);
    fputs(",\"perish_song_counter\":", out); fprintf(out, "%d", mon->perish_song_counter);
    fputs(",\"charge_active\":", out); fprintf(out, "%d", mon->charge_active);
    fputs(",\"charge_turns\":", out); fprintf(out, "%d", mon->charge_turns);
    fputs(",\"last_move_id\":", out); fprintf(out, "%d", mon->last_move_id);
    fputs(",\"last_move_turn\":", out); fprintf(out, "%d", mon->last_move_turn);
    fputs(",\"switched_in_turn\":", out); fprintf(out, "%d", mon->switched_in_turn);
    fputs(",\"first_turn_on_field\":", out); fprintf(out, "%d", mon->first_turn_on_field);
    fputs(",\"ability_triggered_on_switch_in\":", out); fprintf(out, "%d", mon->ability_triggered_on_switch_in);
    fputs(",\"ident\":", out); json_write_escaped(out, mon->ident);
    fputs("}", out);
}

static void json_write_raw_side(FILE* out, const RawSideState* side) {
    if (!side) {
        fputs("null", out);
        return;
    }
    fprintf(out,
        "{\"stealth_rock\":%d,\"spikes\":%d,\"toxic_spikes\":%d,\"sticky_web\":%d,"
        "\"reflect\":%d,\"reflect_turns\":%d,\"light_screen\":%d,\"light_screen_turns\":%d,"
        "\"aurora_veil\":%d,\"aurora_veil_turns\":%d,\"tailwind\":%d,\"tailwind_turns\":%d,"
        "\"safeguard\":%d,\"safeguard_turns\":%d,\"mist\":%d,\"mist_turns\":%d,"
        "\"lucky_chant\":%d,\"lucky_chant_turns\":%d,\"quick_guard\":%d,\"wide_guard\":%d,"
        "\"crafty_shield\":%d,\"mat_block\":%d,\"remaining_pokemon\":%d}",
        side->stealth_rock, side->spikes, side->toxic_spikes, side->sticky_web,
        side->reflect, side->reflect_turns, side->light_screen, side->light_screen_turns,
        side->aurora_veil, side->aurora_veil_turns, side->tailwind, side->tailwind_turns,
        side->safeguard, side->safeguard_turns, side->mist, side->mist_turns,
        side->lucky_chant, side->lucky_chant_turns, side->quick_guard, side->wide_guard,
        side->crafty_shield, side->mat_block, side->remaining_pokemon);
}

static void json_write_raw_state(FILE* out, const RawBattleState* state) {
    int i;
    if (!state) {
        fputs("null", out);
        return;
    }
    fprintf(out,
        "{\"turn_number\":%d,\"can_tera\":%d,\"is_doubles\":%d,\"self_active_count\":%d,\"opp_active_count\":%d,"
        "\"weather_id\":%d,\"terrain_id\":%d,"
        "\"trick_room\":%d,\"trick_room_turns_remaining\":%d,\"magic_room\":%d,\"magic_room_turns_remaining\":%d,"
        "\"wonder_room\":%d,\"wonder_room_turns_remaining\":%d,\"gravity\":%d,\"gravity_turns_remaining\":%d,"
        "\"mud_sport\":%d,\"water_sport\":%d,\"ion_deluge\":%d,",
        state->turn_number, state->can_tera, state->is_doubles, state->self_active_count, state->opp_active_count,
        state->weather_id, state->terrain_id,
        state->trick_room, state->trick_room_turns_remaining, state->magic_room, state->magic_room_turns_remaining,
        state->wonder_room, state->wonder_room_turns_remaining, state->gravity, state->gravity_turns_remaining,
        state->mud_sport, state->water_sport, state->ion_deluge);
    fputs("\"weather_turns_remaining\":", out);
    json_write_tracked_int(out, &state->weather_turns_remaining, NULL);
    fputs(",\"terrain_turns_remaining\":", out);
    json_write_tracked_int(out, &state->terrain_turns_remaining, NULL);
    fputs(",\"self_side\":", out);
    json_write_raw_side(out, &state->self_side);
    fputs(",\"opp_side\":", out);
    json_write_raw_side(out, &state->opp_side);
    fputs(",\"self_team\":[", out);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        json_write_raw_pokemon(out, &state->self_team[i]);
    }
    fputs("],\"opp_team\":[", out);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        json_write_raw_pokemon(out, &state->opp_team[i]);
    }
    fputs("]}", out);
}

static void json_write_canonical_pokemon(FILE* out, const RawPokemon* mon) {
    int i;
    if (!mon) {
        fputs("null", out);
        return;
    }
    fputs("{\"canonical_ident\":", out); json_write_escaped(out, mon->canonical_ident[0] ? mon->canonical_ident : mon->ident);
    fputs(",\"known\":", out); fprintf(out, "%d", mon->known);
    fputs(",\"active\":", out); fprintf(out, "%d", mon->active);
    fputs(",\"active_slot\":", out); fprintf(out, "%d", mon->active_slot);
    fputs(",\"revealed\":", out); fprintf(out, "%d", mon->revealed);
    fputs(",\"fainted\":", out); fprintf(out, "%d", mon->fainted);
    fputs(",\"current_hp\":", out); fprintf(out, "%d", mon->current_hp);
    fputs(",\"max_hp\":", out); fprintf(out, "%d", mon->max_hp);
    fputs(",\"species\":", out); json_write_tracked_int(out, &mon->species_id, species_name_from_id);
    fputs(",\"effective_species\":", out); json_write_tracked_int(out, &mon->effective_species_id, species_name_from_id);
    fputs(",\"item\":", out); json_write_tracked_int(out, &mon->item_id, item_name_from_id);
    fputs(",\"ability\":", out); json_write_tracked_int(out, &mon->ability_id, ability_name_from_id);
    fputs(",\"status\":", out); json_write_tracked_int(out, &mon->status_id, condition_name_from_id);
    fputs(",\"tera_type\":", out); json_write_tracked_int(out, &mon->tera_type_id, type_name_from_id);
    fputs(",\"base_type1\":", out); json_write_tracked_int(out, &mon->type1_id, type_name_from_id);
    fputs(",\"base_type2\":", out); json_write_tracked_int(out, &mon->type2_id, type_name_from_id);
    fputs(",\"effective_type1\":", out); json_write_tracked_int(out, &mon->effective_type1_id, type_name_from_id);
    fputs(",\"effective_type2\":", out); json_write_tracked_int(out, &mon->effective_type2_id, type_name_from_id);
    fputs(",\"tera_used\":", out); fprintf(out, "%d", mon->tera_used);
    fputs(",\"can_tera\":", out); fprintf(out, "%d", mon->can_tera);
    fputs(",\"transformed\":", out); fprintf(out, "%d", mon->transformed);
    fprintf(out,
        ",\"base_stats\":{\"hp\":%d,\"atk\":%d,\"def\":%d,\"spa\":%d,\"spd\":%d,\"spe\":%d}",
        mon->base_hp_stat, mon->base_atk_stat, mon->base_def_stat,
        mon->base_spa_stat, mon->base_spd_stat, mon->base_spe_stat);
    fputs(",\"sleep_turns_elapsed\":", out); fprintf(out, "%d", mon->sleep_turns_elapsed);
    fputs(",\"toxic_counter\":", out); fprintf(out, "%d", mon->toxic_counter);
    fputs(",\"boosts\":", out); json_write_int_array(out, mon->boosts, 7);
    fputs(",\"moves\":[", out);
    for (i = 0; i < RAW_MOVE_SLOTS; ++i) {
        if (i) fputc(',', out);
        fputs("{\"base_move\":", out); json_write_tracked_int(out, &mon->move_ids[i], move_name_from_id);
        fputs(",\"effective_move\":", out); json_write_tracked_int(out, &mon->effective_move_ids[i], move_name_from_id);
        fputs(",\"base_move_type\":", out); json_write_tracked_int(out, &mon->move_type_ids[i], type_name_from_id);
        fputs(",\"effective_move_type\":", out); json_write_tracked_int(out, &mon->effective_move_type_ids[i], type_name_from_id);
        fputs(",\"effective_known\":", out); fprintf(out, "%d", mon->effective_move_known[i]);
        fputs(",\"effective_pp\":", out); fprintf(out, "%d", mon->effective_move_pp[i]);
        fputs(",\"effective_max_pp\":", out); fprintf(out, "%d", mon->effective_move_max_pp[i]);
        fputs(",\"effective_disabled\":", out); fprintf(out, "%d", mon->effective_move_disabled[i]);
        fputs(",\"effective_maybe_disabled\":", out); fprintf(out, "%d", mon->effective_move_maybe_disabled[i]);
        fputs("}", out);
    }
    fputs("]", out);
    fputs(",\"encore_active\":", out); fprintf(out, "%d", mon->encore_active);
    fputs(",\"encore_turns\":", out); fprintf(out, "%d", mon->encore_turns);
    fputs(",\"disable_active\":", out); fprintf(out, "%d", mon->disable_active);
    fputs(",\"disable_turns\":", out); fprintf(out, "%d", mon->disable_turns);
    fputs(",\"encore_move_slot\":", out); fprintf(out, "%d", mon->encore_move_slot);
    fputs(",\"disable_move_slot\":", out); fprintf(out, "%d", mon->disable_move_slot);
    fputs(",\"taunt_active\":", out); fprintf(out, "%d", mon->taunt_active);
    fputs(",\"taunt_turns\":", out); fprintf(out, "%d", mon->taunt_turns);
    fputs(",\"protect_active\":", out); fprintf(out, "%d", mon->protect_active);
    fputs(",\"protect_chain_count\":", out); fprintf(out, "%d", mon->protect_chain_count);
    fputs(",\"flinch_active\":", out); fprintf(out, "%d", mon->flinch_active);
    fputs(",\"trapped\":", out); fprintf(out, "%d", mon->trapped);
    fputs(",\"maybe_trapped\":", out); fprintf(out, "%d", mon->maybe_trapped);
    fputs(",\"commanding_active\":", out); fprintf(out, "%d", mon->commanding_active);
    fputs(",\"reviving\":", out); fprintf(out, "%d", mon->reviving);
    fputs(",\"confusion_active\":", out); fprintf(out, "%d", mon->confusion_active);
    fputs(",\"confusion_turns\":", out); fprintf(out, "%d", mon->confusion_turns);
    fputs(",\"substitute_active\":", out); fprintf(out, "%d", mon->substitute_active);
    fputs(",\"perish_song_counter\":", out); fprintf(out, "%d", mon->perish_song_counter);
    fputs("}", out);
}

static void json_write_debug_pokemon(FILE* out, const RawPokemon* mon) {
    if (!mon) {
        fputs("null", out);
        return;
    }
    fputs("{\"ident\":", out);
    json_write_escaped(out, mon->ident);
    fprintf(out,
        ",\"self_request_roster_index\":%d,\"first_turn_on_field\":%d,"
        "\"switched_in_turn\":%d,\"last_move_id\":%d,\"last_move_turn\":%d,"
        "\"ability_triggered_on_switch_in\":%d}",
        mon->self_request_roster_index,
        mon->first_turn_on_field,
        mon->switched_in_turn,
        mon->last_move_id,
        mon->last_move_turn,
        mon->ability_triggered_on_switch_in);
}

static void json_write_canonical_state(FILE* out, const RawBattleState* state) {
    int i;
    if (!state) {
        fputs("null", out);
        return;
    }
    fprintf(out,
        "{\"turn_number\":%d,\"can_tera\":%d,\"is_doubles\":%d,\"self_active_count\":%d,\"opp_active_count\":%d,"
        "\"weather_id\":%d,\"terrain_id\":%d,"
        "\"trick_room\":%d,\"trick_room_turns_remaining\":%d,\"magic_room\":%d,\"magic_room_turns_remaining\":%d,"
        "\"wonder_room\":%d,\"wonder_room_turns_remaining\":%d,\"gravity\":%d,\"gravity_turns_remaining\":%d,"
        "\"mud_sport\":%d,\"water_sport\":%d,\"ion_deluge\":%d,",
        state->turn_number, state->can_tera, state->is_doubles, state->self_active_count, state->opp_active_count,
        state->weather_id, state->terrain_id,
        state->trick_room, state->trick_room_turns_remaining, state->magic_room, state->magic_room_turns_remaining,
        state->wonder_room, state->wonder_room_turns_remaining, state->gravity, state->gravity_turns_remaining,
        state->mud_sport, state->water_sport, state->ion_deluge);
    fputs("\"weather_turns_remaining\":", out);
    json_write_tracked_int(out, &state->weather_turns_remaining, NULL);
    fputs(",\"terrain_turns_remaining\":", out);
    json_write_tracked_int(out, &state->terrain_turns_remaining, NULL);
    fputs(",\"self_active_slot_to_team_index\":", out);
    json_write_int_array(out, state->self_active_slot_to_team_index, 2);
    fputs(",\"opp_active_slot_to_team_index\":", out);
    json_write_int_array(out, state->opp_active_slot_to_team_index, 2);
    fputs(",\"self_side\":", out);
    json_write_raw_side(out, &state->self_side);
    fputs(",\"opp_side\":", out);
    json_write_raw_side(out, &state->opp_side);
    fputs(",\"self_team\":[", out);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        json_write_canonical_pokemon(out, &state->self_team[i]);
    }
    fputs("],\"opp_team\":[", out);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        json_write_canonical_pokemon(out, &state->opp_team[i]);
    }
    fputs("]}", out);
}

static void json_write_debug_state(FILE* out, const RawBattleState* state) {
    int i;
    if (!state) {
        fputs("null", out);
        return;
    }
    fputs("{\"self_team\":[", out);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        json_write_debug_pokemon(out, &state->self_team[i]);
    }
    fputs("],\"opp_team\":[", out);
    for (i = 0; i < RAW_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        json_write_debug_pokemon(out, &state->opp_team[i]);
    }
    fputs("]}", out);
}

static void json_write_request(FILE* out, const ParsedRequest* req, const ActionMask* mask) {
    int i;
    if (!req) {
        fputs("null", out);
        return;
    }
    fprintf(out,
        "{\"request_id\":%d,\"is_doubles\":%d,\"side_player\":%d,\"side_id\":",
        req->request_id, req->is_doubles, req->side_player);
    json_write_escaped(out, req->side_id);
    fprintf(out,
        ",\"wait\":%d,\"team_preview\":%d,\"max_chosen_team_size\":%d,"
        "\"active_count\":%d,\"living_active_count\":%d,\"can_tera\":%d,\"forced_switch_any\":%d,",
        req->wait, req->team_preview, req->max_chosen_team_size,
        req->active_count, req->living_active_count, req->can_tera, req->forced_switch_any);
    fputs("\"switch_available\":", out);
    json_write_int_array(out, req->switch_available, PARSED_REQUEST_TEAM_SIZE);
    fputs(",\"switch_fainted\":", out);
    json_write_int_array(out, req->switch_fainted, PARSED_REQUEST_TEAM_SIZE);
    fputs(",\"switch_active\":", out);
    json_write_int_array(out, req->switch_active, PARSED_REQUEST_TEAM_SIZE);
    fputs(",\"force_switch\":", out);
    json_write_int_array(out, req->force_switch, PARSED_REQUEST_ACTIVE_SLOTS);
    fputs(",\"active_team_idx\":", out);
    json_write_int_array(out, req->active_team_idx, PARSED_REQUEST_ACTIVE_SLOTS);
    fputs(",\"active_team_idx_known\":", out);
    fputc('[', out);
    for (i = 0; i < PARSED_REQUEST_ACTIVE_SLOTS; ++i) {
        if (i) fputc(',', out);
        fprintf(out, "%u", (unsigned int)req->active_team_idx_known[i]);
    }
    fputc(']', out);
    fputs(",\"bootstrap_slot_binding_ambiguous\":", out);
    fprintf(out, "%u", (unsigned int)req->bootstrap_slot_binding_ambiguous);
    fputs(",\"side\":[", out);
    for (i = 0; i < PARSED_REQUEST_TEAM_SIZE; ++i) {
        if (i) fputc(',', out);
        fputs("{\"ident\":", out);
        json_write_escaped(out, req->side_ident[i]);
        fputs(",\"species_id\":", out);
        fprintf(out, "%d", req->side_species_id[i]);
        fputs("}", out);
    }
    fputs("],\"active\":[", out);
    for (i = 0; i < PARSED_REQUEST_ACTIVE_SLOTS; ++i) {
        int m;
        char part[128];
        if (i) fputc(',', out);
        fprintf(out,
            "{\"slot_present\":%d,\"slot_needs_choice\":%d,\"slot_can_move\":%d,\"slot_can_switch\":%d,"
            "\"choice_kind\":%d,\"can_tera\":%d,\"tera_type_id\":%d,\"trapped\":%d,\"maybe_trapped\":%d,\"fainted\":%d,"
            "\"has_force_switch\":%d,\"moves\":[",
            req->slot_present[i], req->slot_needs_choice[i], req->slot_can_move[i], req->slot_can_switch[i],
            req->choice_kind[i], req->active[i].can_tera, req->active[i].tera_type_id, req->active[i].trapped,
            req->active[i].maybe_trapped, req->active[i].fainted, req->active[i].has_force_switch);
        for (m = 0; m < PARSED_REQUEST_MOVE_SLOTS; ++m) {
            if (m) fputc(',', out);
            fputs("{\"move_id\":", out); fprintf(out, "%d", req->active[i].move_id[m]);
            fputs(",\"move_name\":", out); json_write_escaped(out, move_name_from_id(req->active[i].move_id[m]));
            fputs(",\"disabled\":", out); fprintf(out, "%d", req->active[i].move_disabled[m]);
            fputs(",\"maybe_disabled\":", out); fprintf(out, "%d", req->active[i].move_maybe_disabled[m]);
            fputs(",\"pp\":", out); fprintf(out, "%d", req->active[i].move_pp[m]);
            fputs(",\"max_pp\":", out); fprintf(out, "%d", req->active[i].move_max_pp[m]);
            fputs(",\"target\":", out); fprintf(out, "%d", req->active[i].move_target[m]);
            fputs("}", out);
        }
        fputs("],\"legal_actions\":[", out);
        {
            int first = 1;
            enum ObsAction actions[OBS_NUM_ACTIONS];
            size_t count = mask ? collect_slot_legal_actions(req, mask, i, actions, OBS_NUM_ACTIONS) : 0;
            size_t a;
            for (a = 0; a < count; ++a) {
                if (!first) fputc(',', out);
                first = 0;
                part[0] = '\0';
                action_to_showdown_part(part, sizeof(part), actions[a], req);
                fprintf(out, "{\"index\":%d,\"name\":", (int)actions[a]);
                json_write_escaped(out, obs_action_name(actions[a]));
                fputs(",\"command_part\":", out);
                json_write_escaped(out, part);
                fputs("}", out);
            }
        }
        fputs("]}", out);
    }
    fputs("],\"raw_json\":", out);
    json_write_escaped(out, req->raw_json);
    fputs("}", out);
}

static int export_battle_snapshots(const char* replay_path, const char* battle_id, const char* output_path) {
    FILE* in = NULL;
    FILE* out = NULL;
    GruModel* model = NULL;
    EnvRuntime runtime;
    char line[16384];
    size_t snapshots = 0;
    int found_any = 0;
    int first_snapshot = 1;

    if (!replay_path || !battle_id || !output_path) {
        return 1;
    }
    model = create_default_model();
    if (!model) {
        fprintf(stderr, "Failed to create model for battle export\n");
        return 1;
    }
    if (!env_runtime_init(&runtime, model, NULL, 1, ENV_REWARD_TERMINAL, NULL)) {
        fprintf(stderr, "Failed to initialize runtime for battle export\n");
        gru_model_destroy(model);
        return 1;
    }
    in = fopen(replay_path, "r");
    if (!in) {
        fprintf(stderr, "Failed to open replay file '%s': %s\n", replay_path, strerror(errno));
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 1;
    }
    out = fopen(output_path, "w");
    if (!out) {
        fprintf(stderr, "Failed to open output file '%s': %s\n", output_path, strerror(errno));
        fclose(in);
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 1;
    }

    fputs("{\"battle_id\":", out);
    json_write_escaped(out, battle_id);
    fputs(",\"snapshots\":[", out);

    while (fgets(line, sizeof(line), in)) {
        RuntimeMessage msg;
        EnvSession* session;
        runtime_message_init(&msg);
        if (!runtime_message_parse(&msg, line)) {
            continue;
        }
        if (strcmp(msg.battle_id, battle_id) != 0) {
            continue;
        }
        found_any = 1;
        if (!env_runtime_handle_message(&runtime, &msg, NULL)) {
            continue;
        }
        session = find_runtime_session_by_id(&runtime, battle_id);
        if (!first_snapshot) {
            fputc(',', out);
        }
        first_snapshot = 0;
        fputs("{\"index\":", out);
        fprintf(out, "%zu", snapshots);
        fputs(",\"message_type\":", out);
        json_write_escaped(out, runtime_message_type_name(msg.type));
        fputs(",\"message\":{", out);
        fputs("\"request_id\":", out); fprintf(out, "%d", msg.request_id);
        fputs(",\"seq\":", out); fprintf(out, "%d", msg.seq);
        fputs(",\"result\":", out); json_write_escaped(out, msg.result);
        fputs(",\"reward\":", out); fprintf(out, "%.3f", msg.reward);
        fputs(",\"action\":", out); fprintf(out, "%d", msg.action);
        fputs(",\"action2\":", out); fprintf(out, "%d", msg.action2);
        fputs(",\"accepted\":", out); fprintf(out, "%d", msg.accepted);
        fputs(",\"command\":", out); json_write_escaped(out, msg.command);
        fputs(",\"line\":", out); json_write_escaped(out, msg.line);
        fputs(",\"payload\":", out); json_write_escaped(out, msg.payload);
        fputs(",\"raw_record\":", out); json_write_escaped(out, line);
        fputs("}", out);
        fputs(",\"session\":", out);
        if (!session) {
            fputs("null", out);
        } else {
            fputs("{\"last_request_id\":", out); fprintf(out, "%d", session->last_request_id);
            fputs(",\"terminal\":", out); fprintf(out, "%d", session->terminal);
            fputs(",\"ready_for_decision\":", out); fprintf(out, "%d", session->ready_for_decision);
            fputs(",\"pending_action\":", out); fprintf(out, "%d", session->pending_action);
            fputs(",\"pending_action2\":", out); fprintf(out, "%d", session->pending_action2);
            fputs(",\"pending_command\":", out); json_write_escaped(out, session->pending_command);
            fputs(",\"episode_count\":", out); fprintf(out, "%zu", session->episode.count);
            fputs(",\"canonical_state\":", out); json_write_canonical_state(out, &session->raw_state);
            fputs(",\"debug_only\":", out); json_write_debug_state(out, &session->raw_state);
            fputs(",\"request\":", out); json_write_request(out, &session->parsed_request, &session->action_mask);
            fputs(",\"flat_observation\":", out); json_write_float_array(out, session->flat_observation, runtime.obs_dim);
            fputs("}", out);
        }
        fputs("}", out);
        ++snapshots;
        if (msg.type == RUNTIME_MSG_BATTLE_END) {
            break;
        }
    }

    fputs("]}\n", out);
    fclose(out);
    fclose(in);
    env_runtime_free(&runtime);
    gru_model_destroy(model);

    if (!found_any) {
        fprintf(stderr, "Battle '%s' was not found in '%s'\n", battle_id, replay_path);
        return 1;
    }
    printf("[export-battle] replay=%s battle_id=%s output=%s snapshots=%zu\n",
        replay_path, battle_id, output_path, snapshots);
    return 0;
}

static int evaluate_checkpoint_on_replay_file(
    const char* replay_path,
    const char* checkpoint_path,
    const RewardConfig* reward_config
) {
    char* resolved_checkpoint_path = NULL;
    GruModel* model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    TrainerCheckpointState checkpoint_state;
    size_t* train_indices = NULL;
    size_t* val_indices = NULL;
    size_t train_sessions = 0;
    size_t val_sessions = 0;
    float val_action_loss = 0.0f;
    float val_value_loss = 0.0f;
    float val_accuracy = 0.0f;
    float val_top3_accuracy = 0.0f;
    float val_action1_accuracy = 0.0f;
    float val_action2_accuracy = 0.0f;
    size_t val_labels = 0;
    size_t evaluated_sessions = 0;
    size_t val_action1_labels = 0;
    size_t val_action2_labels = 0;
    size_t val_skipped_steps = 0;
    clock_t eval_start_clock;

    if (!replay_path || !checkpoint_path) {
        return 1;
    }
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path);
    if (!resolved_checkpoint_path) {
        fprintf(stderr, "Failed to resolve checkpoint path\n");
        return 1;
    }
    memset(&checkpoint_state, 0, sizeof(checkpoint_state));
    model = checkpoint_load(resolved_checkpoint_path, &checkpoint_state);
    if (!model) {
        fprintf(stderr, "Failed to load checkpoint '%s'\n", resolved_checkpoint_path);
        free(resolved_checkpoint_path);
        return 1;
    }
    gru_trainer_init(&trainer,
        checkpoint_state.learning_rate > 0.0f ? checkpoint_state.learning_rate : 0.01f,
        checkpoint_state.bptt_window ? checkpoint_state.bptt_window : 16,
        checkpoint_state.gradient_clip,
        checkpoint_state.seed);
    trainer.step = checkpoint_state.step;

    if (!load_runtime_from_replay_file(
            replay_path,
            model,
            &runtime,
            ENV_REWARD_TERMINAL,
            reward_config ? &reward_config->dense_additive : NULL)) {
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    if (!build_split_indices(runtime.count, &train_indices, &train_sessions, &val_indices, &val_sessions)) {
        fprintf(stderr, "Failed to build held-out split indices\n");
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }

    printf("[eval] checkpoint=%s train_sessions=%zu val_sessions=%zu\n",
        resolved_checkpoint_path, train_sessions, val_sessions);
    printf("[eval] accepted_labels direct=%zu reconstructed=%zu failed=%zu\n",
        runtime.accepted_label_direct_count,
        runtime.accepted_label_reconstructed_count,
        runtime.accepted_label_failed_count);
    if (val_sessions == 0) {
        printf("[eval] no held-out validation sessions available\n");
    } else {
        double elapsed;
        double sessions_per_sec;
        double labels_per_sec;
        eval_start_clock = clock();
        if (!evaluate_supervised_split(&trainer, model, &runtime, val_indices, val_sessions,
                &val_action_loss, &val_value_loss, &val_accuracy, &val_labels, &evaluated_sessions,
                &val_top3_accuracy, &val_action1_accuracy, &val_action2_accuracy,
                &val_action1_labels, &val_action2_labels, &val_skipped_steps)) {
        fprintf(stderr, "Failed to evaluate held-out split\n");
        free(train_indices);
        free(val_indices);
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
        }
        elapsed = elapsed_seconds_since(eval_start_clock);
        sessions_per_sec = elapsed > 0.0 ? (double)evaluated_sessions / elapsed : 0.0;
        labels_per_sec = elapsed > 0.0 ? (double)val_labels / elapsed : 0.0;
        printf("[eval] validation action_loss=%.4f value_loss=%.4f accuracy=%.4f labels=%zu sessions=%zu\n",
            val_action_loss, val_value_loss, val_accuracy, val_labels, evaluated_sessions);
        printf("[eval] validation top3_accuracy=%.4f action1_accuracy=%.4f action2_accuracy=%.4f action1_labels=%zu action2_labels=%zu skipped_steps=%zu\n",
            val_top3_accuracy, val_action1_accuracy, val_action2_accuracy,
            val_action1_labels, val_action2_labels, val_skipped_steps);
        printf("[eval] elapsed=%.1fs sessions_per_sec=%.2f labels_per_sec=%.2f\n",
            elapsed, sessions_per_sec, labels_per_sec);
    }

    free(train_indices);
    free(val_indices);
    env_runtime_free(&runtime);
    gru_model_destroy(model);
    free(resolved_checkpoint_path);
    return 0;
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
            episode_append(&episode, flat, obs.legal_mask, -1, 0.0f, (uint8_t)(t == 2));
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
        if (checkpoint_path && *checkpoint_path) {
            fprintf(stderr, "[runtime] failed to load checkpoint %s, starting fresh model\n",
                resolved_checkpoint_path ? resolved_checkpoint_path : checkpoint_path);
        } else {
            fprintf(stderr, "[runtime] no checkpoint provided, starting fresh model\n");
        }
    } else {
        fprintf(stderr, "[runtime] loaded checkpoint %s step=%zu\n",
            resolved_checkpoint_path ? resolved_checkpoint_path : checkpoint_path,
            state.step);
    }
    if (!model) {
        fprintf(stderr, "Failed to initialize runtime model\n");
        return 1;
    }
    if (replay_path && *replay_path) {
        replay_file = fopen(replay_path, "a");
    }
    if (!env_runtime_init(&runtime, model, replay_file, 0, ENV_REWARD_TERMINAL, NULL)) {
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
    if (replay_file) {
        fclose(replay_file);
    }
    gru_model_destroy(model);
    free(resolved_checkpoint_path);
    return 0;
}

static int train_from_input_file(
    const char* input_path,
    const char* checkpoint_path,
    int input_is_episode_batch,
    int rl_mode,
    int epochs,
    float rl_gamma,
    float rl_entropy_coef,
    int rl_advantage_norm,
    int supervised_profile,
    const char* rl_reward_mode,
    const RewardConfig* reward_config
) {
    char* resolved_checkpoint_path = NULL;
    GruModel* model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    EnvRewardMode reward_mode;
    TrainerCheckpointState checkpoint_state;
    size_t* train_indices = NULL;
    size_t* val_indices = NULL;
    clock_t train_loop_start_clock;
    size_t train_sessions = 0;
    size_t val_sessions = 0;
    float best_val_action_loss = 0.0f;
    int has_best_val = 0;
    int epoch;
    double train_eta_rate_ema = 0.0;
    const double train_eta_alpha = 0.2;
    const double train_eta_min_elapsed = 10.0;
    const size_t train_eta_min_episodes = 5;

    if (!input_path || !checkpoint_path || epochs <= 0) {
        return 1;
    }
    if (rl_mode) {
        if (!parse_reward_mode(rl_reward_mode, &reward_mode)) {
            fprintf(stderr, "Unsupported --reward-mode '%s'. Supported modes: terminal, dense_additive.\n",
                rl_reward_mode ? rl_reward_mode : "");
            return 1;
        }
        if (!input_is_episode_batch && strstr(input_path, "legacy") != NULL) {
            printf("[train-rl] warning: replay path contains 'legacy'; RL is intended for fresh post-fix runs only\n");
        }
    } else {
        reward_mode = ENV_REWARD_TERMINAL;
    }
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path);
    if (!resolved_checkpoint_path) {
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
        free(resolved_checkpoint_path);
        return 1;
    }
    gru_trainer_init(&trainer,
        checkpoint_state.learning_rate > 0.0f ? checkpoint_state.learning_rate : 0.01f,
        checkpoint_state.bptt_window ? checkpoint_state.bptt_window : 16,
        checkpoint_state.gradient_clip,
        checkpoint_state.seed);
    trainer.step = checkpoint_state.step;
    if (rl_mode) {
        trainer.gamma = rl_gamma;
        trainer.entropy_coef = rl_entropy_coef;
        trainer.advantage_norm = rl_advantage_norm ? 1 : 0;
    } else {
        trainer.supervised_profile_enabled = supervised_profile;
    }

    if (!(input_is_episode_batch
            ? load_runtime_from_episode_batch_file(
                input_path,
                model,
                &runtime,
                reward_mode,
                reward_config ? &reward_config->dense_additive : NULL)
            : load_runtime_from_replay_file(
                input_path,
                model,
                &runtime,
                reward_mode,
                reward_config ? &reward_config->dense_additive : NULL))) {
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    if (!build_split_indices(runtime.count, &train_indices, &train_sessions, &val_indices, &val_sessions)) {
        fprintf(stderr, "Failed to build train/validation split indices\n");
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    if (rl_mode) {
        size_t train_sessions_before_filter = train_sessions;
        filter_labeled_train_indices(&runtime, train_indices, &train_sessions);
        printf("[train-rl] filtered train_sessions labeled_only=%zu/%zu\n",
            train_sessions, train_sessions_before_filter);
    }
    printf("[train] split train_sessions=%zu val_sessions=%zu epochs=%d\n", train_sessions, val_sessions, epochs);
    printf("[train] accepted_labels direct=%zu reconstructed=%zu failed=%zu\n",
        runtime.accepted_label_direct_count,
        runtime.accepted_label_reconstructed_count,
        runtime.accepted_label_failed_count);
    if (rl_mode) {
        printf("[train-rl] config gamma=%.4f entropy_coef=%.4f advantage_norm=%d reward_mode=%s\n",
            trainer.gamma,
            trainer.entropy_coef,
            trainer.advantage_norm,
            rl_reward_mode);
    }

    for (epoch = 1; epoch <= epochs; ++epoch) {
        size_t trained_in_epoch = 0;
        float val_action_loss = 0.0f;
        float val_value_loss = 0.0f;
        float val_accuracy = 0.0f;
        float val_top3_accuracy = 0.0f;
        float val_action1_accuracy = 0.0f;
        float val_action2_accuracy = 0.0f;
        size_t val_labels = 0;
        size_t evaluated_sessions = 0;
        size_t val_action1_labels = 0;
        size_t val_action2_labels = 0;
        size_t val_skipped_steps = 0;
        clock_t val_eval_start_clock = 0;

        printf("[train] epoch %d/%d start\n", epoch, epochs);
        train_loop_start_clock = clock();
        shuffle_indices(train_indices, train_sessions);

        for (size_t order_i = 0; order_i < train_sessions; ++order_i) {
            size_t session_index = train_indices[order_i];
            if (rl_mode) {
                if (!gru_trainer_policy_gradient_episode(&trainer, model, &runtime.sessions[session_index].episode)) {
                    fprintf(stderr, "Failed RL training episode\n");
                    free(train_indices);
                    free(val_indices);
                    env_runtime_free(&runtime);
                    gru_model_destroy(model);
                    free(resolved_checkpoint_path);
                    return 1;
                }
            } else {
                if (!gru_trainer_supervised_episode(&trainer, model, &runtime.sessions[session_index].episode)) {
                    fprintf(stderr, "Failed supervised training episode\n");
                    free(train_indices);
                    free(val_indices);
                    env_runtime_free(&runtime);
                    gru_model_destroy(model);
                    free(resolved_checkpoint_path);
                    return 1;
                }
            }
            ++trained_in_epoch;
            {
                double elapsed = elapsed_seconds_since(train_loop_start_clock);
                double episodes_per_sec = elapsed > 0.0 ? (double)trained_in_epoch / elapsed : 0.0;
                double eta = 0.0;
                int eta_ready = (elapsed >= train_eta_min_elapsed || trained_in_epoch >= train_eta_min_episodes);
                if (episodes_per_sec > 0.0) {
                    train_eta_rate_ema = ema_update(train_eta_rate_ema, episodes_per_sec, train_eta_alpha);
                }
                if (eta_ready && train_eta_rate_ema > 0.0) {
                    eta = (double)(train_sessions - trained_in_epoch) / train_eta_rate_ema;
                }
                if (rl_mode) {
                    printf("[train-rl] epoch=%d episodes=%zu/%zu step=%zu mean_return=%.4f policy_loss=%.4f value_loss=%.4f mean_advantage=%.4f entropy=%.4f labels=%zu\n",
                        epoch,
                        trained_in_epoch,
                        train_sessions,
                        trainer.step,
                        trainer.last_mean_return,
                        trainer.last_policy_loss,
                        trainer.last_value_loss,
                        trainer.last_mean_advantage,
                        trainer.last_entropy,
                        trainer.last_rl_labels);
                } else {
                    printf("[train] epoch=%d episodes=%zu/%zu step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f\n",
                        epoch,
                        trained_in_epoch,
                        train_sessions,
                        trainer.step,
                        trainer.last_action_loss,
                        trainer.last_value_loss,
                        trainer.last_accuracy);
                    if (trainer.supervised_profile_enabled) {
                        printf("[train] epoch=%d supervised_profile cache=%.3fs update=%.3fs labels=%zu windows=%zu flushes=%zu\n",
                            epoch,
                            trainer.last_supervised_cache_seconds,
                            trainer.last_supervised_update_seconds,
                            trainer.last_supervised_label_count,
                            trainer.last_supervised_window_count,
                            trainer.last_supervised_batch_flushes);
                    }
                }
                if (eta_ready && train_eta_rate_ema > 0.0) {
                    printf("[train] epoch=%d elapsed=%.1fs episodes_per_sec=%.2f eta=%.1fs\n",
                        epoch,
                        elapsed,
                        train_eta_rate_ema,
                        eta);
                } else {
                    printf("[train] epoch=%d elapsed=%.1fs episodes_per_sec=%.2f eta=estimating\n",
                        epoch,
                        elapsed,
                        episodes_per_sec);
                }
                fflush(stdout);
            }
            if (!rl_mode && ((trained_in_epoch % 500u) == 0u || trained_in_epoch == train_sessions)) {
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
            TrainerCheckpointState best_state;
            char* best_path = NULL;
            double val_elapsed;
            double val_sessions_per_sec;
            double val_labels_per_sec;
            val_eval_start_clock = clock();
            if (!evaluate_supervised_split(&trainer, model, &runtime, val_indices, val_sessions,
                    &val_action_loss, &val_value_loss, &val_accuracy, &val_labels, &evaluated_sessions,
                    &val_top3_accuracy, &val_action1_accuracy, &val_action2_accuracy,
                    &val_action1_labels, &val_action2_labels, &val_skipped_steps)) {
                fprintf(stderr, "Failed held-out validation evaluation\n");
                free(train_indices);
                free(val_indices);
                env_runtime_free(&runtime);
                gru_model_destroy(model);
                free(resolved_checkpoint_path);
                return 1;
            }
            val_elapsed = elapsed_seconds_since(val_eval_start_clock);
            val_sessions_per_sec = val_elapsed > 0.0 ? (double)evaluated_sessions / val_elapsed : 0.0;
            val_labels_per_sec = val_elapsed > 0.0 ? (double)val_labels / val_elapsed : 0.0;
            if (val_labels > 0) {
                printf("[train] epoch=%d validation action_loss=%.4f value_loss=%.4f accuracy=%.4f labels=%zu\n",
                    epoch,
                    val_action_loss,
                    val_value_loss,
                    val_accuracy,
                    val_labels);
                printf("[train] epoch=%d validation top3_accuracy=%.4f action1_accuracy=%.4f action2_accuracy=%.4f action1_labels=%zu action2_labels=%zu skipped_steps=%zu\n",
                    epoch,
                    val_top3_accuracy,
                    val_action1_accuracy,
                    val_action2_accuracy,
                    val_action1_labels,
                    val_action2_labels,
                    val_skipped_steps);
                printf("[train] epoch=%d validation elapsed=%.1fs sessions_per_sec=%.2f labels_per_sec=%.2f\n",
                    epoch,
                    val_elapsed,
                    val_sessions_per_sec,
                    val_labels_per_sec);
                if (!has_best_val || val_action_loss < best_val_action_loss) {
                    has_best_val = 1;
                    best_val_action_loss = val_action_loss;
                    best_state = gru_trainer_checkpoint_state(&trainer);
                    best_path = make_best_checkpoint_path(resolved_checkpoint_path);
                    if (best_path) {
                        if (checkpoint_save(best_path, model, &best_state)) {
                            printf("[train] saved best validation checkpoint %s action_loss=%.4f\n",
                                best_path, best_val_action_loss);
                        } else {
                            printf("[train] failed best validation checkpoint %s\n", best_path);
                        }
                        free(best_path);
                    }
                }
            } else {
                printf("[train] epoch=%d validation skipped no labels\n", epoch);
            }
        }

        if (!rl_mode) {
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
        free(train_indices);
        free(val_indices);
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    if (rl_mode) {
        printf("trained mode=rl step=%zu policy_loss=%.4f value_loss=%.4f mean_return=%.4f mean_advantage=%.4f entropy=%.4f labels=%zu sessions=%zu\n",
            trainer.step,
            trainer.last_policy_loss,
            trainer.last_value_loss,
            trainer.last_mean_return,
            trainer.last_mean_advantage,
            trainer.last_entropy,
            trainer.last_rl_labels,
            runtime.count);
    } else {
        printf("trained mode=supervised step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f sessions=%zu\n",
            trainer.step,
            trainer.last_action_loss,
            trainer.last_value_loss,
            trainer.last_accuracy,
            runtime.count);
    }
    printf("[train] saved checkpoint %s\n", resolved_checkpoint_path);
    free(train_indices);
    free(val_indices);
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

static int showdown_client_main(int argc, char** argv) {
    int epochs = parse_epochs_arg(argc, argv, 1);
    float rl_gamma = parse_float_flag(argc, argv, "--gamma", 1.0f);
    float rl_entropy_coef = parse_float_flag(argc, argv, "--entropy-coef", 0.001f);
    int rl_advantage_norm = parse_int_flag(argc, argv, "--advantage-norm", 1);
    int supervised_profile = parse_bool01_flag(argc, argv, "--supervised-profile", 1);
    const char* rl_reward_mode = parse_string_flag(argc, argv, "--reward-mode", "terminal");
    RewardConfig reward_config;
    int training_or_eval_mode = 0;
    srand((unsigned int)time(NULL));
    if (!load_reward_config_file(SHOWDOWN_CLIENT_REWARD_CONFIG_PATH, &reward_config)) {
        return 1;
    }
    if (!id_tables_init()) {
        fprintf(stderr, "Failed to initialize ID tables\n");
        return 1;
    }
    if (getenv("PORYGON_DEMO_GRU")) {
        return run_demo_gru();
    }
    if (argc >= 2 &&
            (strcmp(argv[1], "--train-supervised") == 0 ||
             strcmp(argv[1], "--train-rl") == 0 ||
             strcmp(argv[1], "--train-live-rl") == 0 ||
             strcmp(argv[1], "--eval-supervised") == 0)) {
        training_or_eval_mode = 1;
    }
#ifdef _OPENMP
    {
        const char* omp_threads_text = getenv("PORYGON_OMP_THREADS");
        if (omp_threads_text && *omp_threads_text) {
            char* endptr = NULL;
            long threads = strtol(omp_threads_text, &endptr, 10);
            if (endptr && *endptr == '\0' && threads > 0) {
                omp_set_num_threads((int)threads);
            }
        }
        if (training_or_eval_mode) {
            printf("[train] OpenMP threads=%d\n", omp_get_max_threads());
        }
    }
#else
    if (training_or_eval_mode) {
        printf("[train] OpenMP unavailable; running single-threaded\n");
    }
#endif
    if (argc >= 2 && (strcmp(argv[1], "--battle-agent") == 0 || strcmp(argv[1], "--runtime") == 0)) {
        return run_runtime_mode(argc >= 3 ? argv[2] : NULL);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-supervised") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            0,
            0,
            epochs,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            supervised_profile,
            rl_reward_mode,
            &reward_config);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-rl") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            0,
            1,
            epochs,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            supervised_profile,
            rl_reward_mode,
            &reward_config);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-live-rl") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            1,
            1,
            epochs,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            supervised_profile,
            rl_reward_mode,
            &reward_config);
    }
    if (argc >= 4 && strcmp(argv[1], "--eval-supervised") == 0) {
        return evaluate_checkpoint_on_replay_file(argv[2], argv[3], &reward_config);
    }
    if (argc >= 4 && strcmp(argv[1], "--clean-replay") == 0) {
        return clean_replay_file(argv[2], argv[3]);
    }
    if (argc >= 5 && strcmp(argv[1], "--export-battle") == 0) {
        return export_battle_snapshots(argv[2], argv[3], argv[4]);
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
        "  showdown_client --battle-agent [checkpoint]\n"
        "  showdown_client --train-supervised <replay.jsonl> <checkpoint.bin> [--epochs N] [--supervised-profile 0|1]\n"
        "  showdown_client --train-rl <replay.jsonl> <checkpoint.bin> [--epochs N] [--gamma F] [--entropy-coef F] [--advantage-norm 0|1] [--reward-mode terminal|dense_additive]\n"
        "  showdown_client --train-live-rl <episode_batch.jsonl> <checkpoint.bin> [--epochs N] [--gamma F] [--entropy-coef F] [--advantage-norm 0|1] [--reward-mode terminal|dense_additive]\n"
        "  showdown_client --eval-supervised <replay.jsonl> <checkpoint.bin>\n"
        "  showdown_client --clean-replay <input.jsonl> <output.jsonl>\n"
        "  showdown_client --export-battle <replay.jsonl> <battle_id> <output.json>\n"
        "  Set PORYGON_DEMO_GRU=1 for the demo mode.\n"
        "Legacy native websocket mode is disabled; use the Python communicator for live Showdown.\n");
    return 1;
#endif
}

int main(int argc, char** argv) {
    char** config_args = NULL;
    char** merged_argv = NULL;
    int config_argc = 0;
    int effective_argc = argc;
    char** effective_argv = argv;
    int rc;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    if (argc == 1) {
        config_args = load_default_args_file(SHOWDOWN_CLIENT_DEFAULT_ARGS_PATH, &config_argc);
        if (config_args && config_argc > 0) {
            int i;
            merged_argv = (char**)malloc((size_t)(config_argc + 2) * sizeof(char*));
            if (!merged_argv) {
                free_default_args(config_args, config_argc);
                fprintf(stderr, "Failed to allocate default argv\n");
                return 1;
            }
            merged_argv[0] = argv[0];
            for (i = 0; i < config_argc; ++i) {
                merged_argv[i + 1] = config_args[i];
            }
            merged_argv[config_argc + 1] = NULL;
            effective_argc = config_argc + 1;
            effective_argv = merged_argv;
        }
    }

    rc = showdown_client_main(effective_argc, effective_argv);
    free(merged_argv);
    free_default_args(config_args, config_argc);
    return rc;
}

