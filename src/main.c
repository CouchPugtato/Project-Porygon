#include "checkpoint.h"
#include "env_session.h"
#include "gru_model.h"
#include "gru_trainer.h"
#include "learning_diagnostics.h"
#include "policy_evaluation.h"
#include "id_tables.h"
#include "observation.h"
#include "runtime_protocol.h"
#include "validation_split.h"
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
#define SHOWDOWN_CLIENT_RL_DEFAULTS_PATH "config/rl_defaults.toml"

typedef struct {
    float terminal_win;
    float terminal_loss;
    float terminal_draw;
    float terminal_disconnect_or_forfeit;
    EnvDenseRewardConfig dense_additive;
} RewardConfig;

typedef struct {
    float policy_gradient_gamma;
    float policy_gradient_entropy_coef;
    float ppo_gamma;
    float ppo_entropy_coef;
    int advantage_norm;
    float gae_lambda;
    float ppo_clip_epsilon;
    float ppo_value_clip_epsilon;
    float ppo_target_kl;
    int ppo_target_kl_min_episodes;
    int ppo_target_kl_min_labels;
    float ppo_target_kl_hard_multiplier;
    int ppo_target_kl_hard_consecutive_updates;
    int ppo_shuffle_seed;
    int ppo_minibatch_episodes;
    float adam_beta1;
    float adam_beta2;
    float adam_epsilon;
} RlDefaultsConfig;

typedef struct {
    size_t move_slot_counts[4];
    size_t switch_slot_counts[6];
    size_t move_count;
    size_t switch_count;
    size_t tera_count;
    size_t episode_count;
    size_t total_steps;
    size_t outcome_wins;
    size_t outcome_losses;
    size_t outcome_draws;
    double return_sum;
    float return_min;
    float return_max;
    double policy_loss_sum;
    double value_loss_sum;
    double entropy_sum;
    double anchor_loss_sum;
    double anchor_kl_mean_sum;
    double approx_kl_sum;
    double clip_fraction_sum;
    float anchor_kl_max;
    float target_kl_trigger;
    int target_kl_exceeded;
    int target_kl_hard_stop;
    size_t available_episode_count;
    size_t selected_episode_count;
    int episode_limit;
    int shuffle_seed;
    int minibatch_episodes;
    int target_kl_min_episodes;
    int target_kl_min_labels;
    float target_kl_hard_multiplier;
    int target_kl_hard_consecutive_updates;
    int target_kl_hard_breach_count;
    double value_mean_sum;
    double advantage_mean_sum;
    double abs_advantage_mean_sum;
    size_t label_weight_sum;
} RlTrainingSummary;

static GruModel* create_default_model(void) {
    return gru_model_create(observation_flat_size(), 128, OBS_NUM_ACTIONS);
}

static GruModel* load_current_checkpoint(
    const char* path,
    TrainerCheckpointState* state,
    CheckpointLoadResult* result
) {
    return checkpoint_load_compatible(
        path,
        state,
        observation_flat_size(),
        OBS_NUM_ACTIONS,
        result);
}

static void report_checkpoint_load_failure(
    const char* context,
    const char* path,
    const CheckpointLoadResult* result
) {
    CheckpointLoadStatus status = result ? result->status : CHECKPOINT_LOAD_INVALID_ARGUMENT;
    fprintf(stderr,
        "%s checkpoint='%s' reason='%s' stored_version=%u stored_input=%zu stored_hidden=%zu stored_actions=%zu stored_parameters=%zu expected_input=%zu expected_actions=%zu stored_checksum=%08x computed_checksum=%08x\n",
        context ? context : "checkpoint load failed",
        path ? path : "",
        checkpoint_load_status_string(status),
        result ? result->stored_version : 0u,
        result ? result->stored_input_dim : 0u,
        result ? result->stored_hidden_dim : 0u,
        result ? result->stored_num_actions : 0u,
        result ? result->stored_parameter_count : 0u,
        result ? result->expected_input_dim : observation_flat_size(),
        result ? result->expected_num_actions : OBS_NUM_ACTIONS,
        result ? (unsigned int)result->stored_checksum : 0u,
        result ? (unsigned int)result->computed_checksum : 0u);
}

static void report_checkpoint_load_success(
    const char* context,
    const char* path,
    const TrainerCheckpointState* state,
    const CheckpointLoadResult* result
) {
    fprintf(stderr, "%s checkpoint='%s' step=%zu version=%u checksum=%s layout=%s%s%s\n",
        context ? context : "loaded checkpoint",
        path ? path : "",
        state ? state->step : 0u,
        result ? result->stored_version : 0u,
        result && result->checksum_verified ? "verified" : "unverified",
        result && result->parameter_layout == CHECKPOINT_LAYOUT_LEGACY_FLAT ? "legacy_flat" : "factorized",
        result && result->migrated_legacy_heads ? " migrated_factorized_heads=1" : "",
        result && result->migrated_active_slot_inputs ? " migrated_active_slot_inputs=1" : "");
}

static void rl_training_summary_init(RlTrainingSummary* summary) {
    if (!summary) {
        return;
    }
    memset(summary, 0, sizeof(*summary));
    summary->return_min = 0.0f;
    summary->return_max = 0.0f;
}

static void rl_training_summary_record_action(RlTrainingSummary* summary, int action) {
    int move_slot = -1;
    int switch_slot = -1;
    int tera = 0;
    if (!summary) {
        return;
    }
    if (action >= OBS_A1_MOVE1 && action <= OBS_A1_MOVE4) {
        move_slot = action - OBS_A1_MOVE1;
    } else if (action >= OBS_A2_MOVE1 && action <= OBS_A2_MOVE4) {
        move_slot = action - OBS_A2_MOVE1;
    } else if (action >= OBS_A1_MOVE1_TERA && action <= OBS_A1_MOVE4_TERA) {
        move_slot = action - OBS_A1_MOVE1_TERA;
        tera = 1;
    } else if (action >= OBS_A2_MOVE1_TERA && action <= OBS_A2_MOVE4_TERA) {
        move_slot = action - OBS_A2_MOVE1_TERA;
        tera = 1;
    } else if (action >= OBS_A1_SWITCH1 && action <= OBS_A1_SWITCH6) {
        switch_slot = action - OBS_A1_SWITCH1;
    } else if (action >= OBS_A2_SWITCH1 && action <= OBS_A2_SWITCH6) {
        switch_slot = action - OBS_A2_SWITCH1;
    }
    if (move_slot >= 0 && move_slot < 4) {
        summary->move_slot_counts[move_slot] += 1u;
        summary->move_count += 1u;
        if (tera) {
            summary->tera_count += 1u;
        }
    }
    if (switch_slot >= 0 && switch_slot < 6) {
        summary->switch_slot_counts[switch_slot] += 1u;
        summary->switch_count += 1u;
    }
}

static void rl_training_summary_record_episode(RlTrainingSummary* summary, const Episode* episode) {
    size_t i;
    float episode_return = 0.0f;
    if (!summary || !episode) {
        return;
    }
    summary->episode_count += 1u;
    summary->total_steps += episode->count;
    for (i = 0; i < episode->count; ++i) {
        episode_return += episode->rewards[i];
        if (episode->actions[i] >= 0) {
            rl_training_summary_record_action(summary, episode->actions[i]);
        }
        if (episode->actions2[i] >= 0) {
            rl_training_summary_record_action(summary, episode->actions2[i]);
        }
    }
    if (summary->episode_count == 1u) {
        summary->return_min = episode_return;
        summary->return_max = episode_return;
    } else {
        if (episode_return < summary->return_min) {
            summary->return_min = episode_return;
        }
        if (episode_return > summary->return_max) {
            summary->return_max = episode_return;
        }
    }
    summary->return_sum += episode_return;
    if (episode->count > 0) {
        float terminal_reward = episode->rewards[episode->count - 1];
        if (terminal_reward > 0.0f) {
            summary->outcome_wins += 1u;
        } else if (terminal_reward < 0.0f) {
            summary->outcome_losses += 1u;
        } else {
            summary->outcome_draws += 1u;
        }
    }
}

static void rl_training_summary_record_trainer(RlTrainingSummary* summary, const GruTrainer* trainer) {
    if (!summary || !trainer || trainer->last_rl_labels == 0) {
        return;
    }
    summary->policy_loss_sum += (double)trainer->last_policy_loss * (double)trainer->last_rl_labels;
    summary->value_loss_sum += (double)trainer->last_value_loss * (double)trainer->last_rl_labels;
    summary->entropy_sum += (double)trainer->last_entropy * (double)trainer->last_rl_labels;
    summary->anchor_loss_sum += (double)trainer->last_anchor_loss * (double)trainer->last_rl_labels;
    summary->anchor_kl_mean_sum += (double)trainer->last_anchor_kl_mean * (double)trainer->last_rl_labels;
    summary->approx_kl_sum += (double)trainer->last_approx_kl * (double)trainer->last_rl_labels;
    summary->clip_fraction_sum += (double)trainer->last_clip_fraction * (double)trainer->last_rl_labels;
    summary->value_mean_sum += (double)trainer->last_mean_value * (double)trainer->last_rl_labels;
    summary->advantage_mean_sum += (double)trainer->last_mean_advantage * (double)trainer->last_rl_labels;
    summary->abs_advantage_mean_sum += (double)trainer->last_mean_abs_advantage * (double)trainer->last_rl_labels;
    summary->label_weight_sum += trainer->last_rl_labels;
    if (trainer->last_anchor_kl_max > summary->anchor_kl_max) {
        summary->anchor_kl_max = trainer->last_anchor_kl_max;
    }
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

static void rl_defaults_config_init(RlDefaultsConfig* config) {
    if (!config) {
        return;
    }
    config->policy_gradient_gamma = 1.0f;
    config->policy_gradient_entropy_coef = 0.001f;
    config->ppo_gamma = 0.99f;
    config->ppo_entropy_coef = 0.001f;
    config->advantage_norm = 1;
    config->gae_lambda = 0.95f;
    config->ppo_clip_epsilon = 0.20f;
    config->ppo_value_clip_epsilon = 0.20f;
    config->ppo_target_kl = 0.02f;
    config->ppo_target_kl_min_episodes = 20;
    config->ppo_target_kl_min_labels = 500;
    config->ppo_target_kl_hard_multiplier = 4.0f;
    config->ppo_target_kl_hard_consecutive_updates = 2;
    config->ppo_shuffle_seed = 1337;
    config->ppo_minibatch_episodes = 8;
    config->adam_beta1 = 0.90f;
    config->adam_beta2 = 0.999f;
    config->adam_epsilon = 1.0e-8f;
}

static int assign_rl_default_value(RlDefaultsConfig* config, const char* key, const char* value_text) {
    float value;
    char* endptr = NULL;
    if (!config || !key || !value_text) {
        return 0;
    }
    if (strcmp(key, "advantage_norm") == 0) {
        if (strcmp(value_text, "true") == 0 || strcmp(value_text, "1") == 0) {
            config->advantage_norm = 1;
            return 1;
        }
        if (strcmp(value_text, "false") == 0 || strcmp(value_text, "0") == 0) {
            config->advantage_norm = 0;
            return 1;
        }
        return 0;
    }
    value = strtof(value_text, &endptr);
    if (endptr == value_text || (endptr && *trim_whitespace(endptr) != '\0')) {
        return 0;
    }
    if (strcmp(key, "policy_gradient_gamma") == 0) config->policy_gradient_gamma = value;
    else if (strcmp(key, "policy_gradient_entropy_coef") == 0) config->policy_gradient_entropy_coef = value;
    else if (strcmp(key, "ppo_gamma") == 0) config->ppo_gamma = value;
    else if (strcmp(key, "ppo_entropy_coef") == 0) config->ppo_entropy_coef = value;
    else if (strcmp(key, "gae_lambda") == 0) config->gae_lambda = value;
    else if (strcmp(key, "ppo_clip_epsilon") == 0) config->ppo_clip_epsilon = value;
    else if (strcmp(key, "ppo_value_clip_epsilon") == 0) config->ppo_value_clip_epsilon = value;
    else if (strcmp(key, "ppo_target_kl") == 0) config->ppo_target_kl = value;
    else if (strcmp(key, "ppo_target_kl_min_episodes") == 0) config->ppo_target_kl_min_episodes = (int)value;
    else if (strcmp(key, "ppo_target_kl_min_labels") == 0) config->ppo_target_kl_min_labels = (int)value;
    else if (strcmp(key, "ppo_target_kl_hard_multiplier") == 0) config->ppo_target_kl_hard_multiplier = value;
    else if (strcmp(key, "ppo_target_kl_hard_consecutive_updates") == 0) config->ppo_target_kl_hard_consecutive_updates = (int)value;
    else if (strcmp(key, "ppo_shuffle_seed") == 0) config->ppo_shuffle_seed = (int)value;
    else if (strcmp(key, "ppo_minibatch_episodes") == 0) config->ppo_minibatch_episodes = (int)value;
    else if (strcmp(key, "adam_beta1") == 0) config->adam_beta1 = value;
    else if (strcmp(key, "adam_beta2") == 0) config->adam_beta2 = value;
    else if (strcmp(key, "adam_epsilon") == 0) config->adam_epsilon = value;
    return 1;
}

static int load_rl_defaults_file(const char* path, RlDefaultsConfig* out_config) {
    FILE* fp;
    char line[1024];
    int line_number = 0;
    if (!out_config) {
        return 0;
    }
    rl_defaults_config_init(out_config);
    fp = path ? fopen(path, "r") : NULL;
    if (!fp) {
        return 1;
    }
    while (fgets(line, sizeof(line), fp)) {
        char* comment = strchr(line, '#');
        char* key;
        char* value;
        line_number += 1;
        if (comment) *comment = '\0';
        key = trim_whitespace(line);
        if (!key || !*key) continue;
        value = strchr(key, '=');
        if (!value) {
            fclose(fp);
            fprintf(stderr, "invalid RL defaults %s:%d: expected key = value\n", path, line_number);
            return 0;
        }
        *value++ = '\0';
        key = trim_whitespace(key);
        value = trim_whitespace(value);
        if (!key || !*key || !value || !*value || !assign_rl_default_value(out_config, key, value)) {
            fclose(fp);
            fprintf(stderr, "invalid RL defaults %s:%d: bad key/value\n", path, line_number);
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

static int parse_supervised_optimizer(
    const char* name,
    GruSupervisedOptimizer* optimizer_out
) {
    if (!name || !optimizer_out) return 0;
    if (strcmp(name, "sgd") == 0) {
        *optimizer_out = GRU_SUPERVISED_OPTIMIZER_SGD;
        return 1;
    }
    if (strcmp(name, "adam") == 0) {
        *optimizer_out = GRU_SUPERVISED_OPTIMIZER_ADAM;
        return 1;
    }
    return 0;
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
    const EnvRuntime* runtime,
    unsigned int validation_seed,
    size_t** train_indices_out,
    size_t* train_count_out,
    size_t** val_indices_out,
    size_t* val_count_out
) {
    size_t* train_indices = NULL;
    size_t* val_indices = NULL;
    size_t train_count = 0;
    size_t val_count = 0;
    size_t total_sessions;
    size_t i;

    if (!runtime || !train_indices_out || !train_count_out || !val_indices_out || !val_count_out) {
        return 0;
    }
    total_sessions = runtime->count;

    train_indices = (size_t*)malloc((total_sessions > 0 ? total_sessions : 1) * sizeof(size_t));
    val_indices = (size_t*)malloc((total_sessions > 0 ? total_sessions : 1) * sizeof(size_t));
    if (!train_indices || !val_indices) {
        free(train_indices);
        free(val_indices);
        return 0;
    }

    for (i = 0; i < total_sessions; ++i) {
        if (validation_split_contains(runtime->sessions[i].battle_id, validation_seed)) {
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

static int build_all_train_indices(
    size_t total_sessions,
    size_t** train_indices_out,
    size_t* train_count_out,
    size_t** val_indices_out,
    size_t* val_count_out
) {
    size_t* train_indices = NULL;
    size_t* val_indices = NULL;
    size_t i;

    if (!train_indices_out || !train_count_out || !val_indices_out || !val_count_out) {
        return 0;
    }

    train_indices = (size_t*)malloc((total_sessions > 0 ? total_sessions : 1) * sizeof(size_t));
    val_indices = (size_t*)malloc(sizeof(size_t));
    if (!train_indices || !val_indices) {
        free(train_indices);
        free(val_indices);
        return 0;
    }

    for (i = 0; i < total_sessions; ++i) {
        train_indices[i] = i;
    }

    *train_indices_out = train_indices;
    *train_count_out = total_sessions;
    *val_indices_out = val_indices;
    *val_count_out = 0;
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

static int evaluate_supervised_split(
    const GruTrainer* trainer,
    const GruModel* model,
    const EnvRuntime* runtime,
    const size_t* indices,
    size_t index_count,
    PolicyEvaluationMetrics* metrics_out
) {
    size_t i;
    if (!trainer || !model || !runtime || !metrics_out) {
        return 0;
    }
    policy_evaluation_init(metrics_out);
    for (i = 0; i < index_count; ++i) {
        size_t session_index = indices[i];
        if (session_index >= runtime->count ||
                !policy_evaluation_add_episode(model, trainer->bptt_window,
                    &runtime->sessions[session_index].episode, metrics_out)) {
            return 0;
        }
    }
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
    if (!env_runtime_init(runtime, model, NULL, 1, reward_mode, dense_reward_config, NULL)) {
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
    const EnvDenseRewardConfig* dense_reward_config,
    const char* expected_policy_tag
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
    if (!env_runtime_init(runtime, model, NULL, 1, reward_mode, dense_reward_config, NULL)) {
        fclose(f);
        return 0;
    }

    ingest_start_clock = clock();
    {
        char first_policy_tag[256];
        int mixed_policy_tags = 0;
        first_policy_tag[0] = '\0';
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
        if (expected_policy_tag && *expected_policy_tag) {
            if (strcmp(policy_tag, expected_policy_tag) != 0) {
                fprintf(stderr,
                    "[train-live-rl] policy_tag mismatch battle=%s expected=%s actual=%s\n",
                    battle_id,
                    expected_policy_tag,
                    policy_tag);
                episode_free(&episode);
                free(line);
                fclose(f);
                env_runtime_free(runtime);
                return 0;
            }
        } else if (policy_tag[0] != '\0') {
            if (first_policy_tag[0] == '\0') {
                snprintf(first_policy_tag, sizeof(first_policy_tag), "%s", policy_tag);
            } else if (strcmp(first_policy_tag, policy_tag) != 0) {
                mixed_policy_tags = 1;
            }
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
        if (!expected_policy_tag && mixed_policy_tags) {
            printf("[train-live-rl] warning: mixed policy_tag values detected in episode batch\n");
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

static int write_rl_training_summary_json(
    const char* path,
    const RlTrainingSummary* summary,
    const GruTrainer* trainer,
    const char* input_path,
    const char* parent_checkpoint,
    const char* output_checkpoint,
    const char* anchor_checkpoint,
    float anchor_kl_coef,
    const char* reward_mode,
    const RewardConfig* reward_config
) {
    FILE* out;
    size_t i;
    if (!path || !*path || !summary || !trainer || !reward_mode || !reward_config) {
        return 0;
    }
    out = fopen(path, "w");
    if (!out) {
        return 0;
    }
    fputs("{\n", out);
    fputs("  \"input_episode_batch\": ", out); json_write_escaped(out, input_path ? input_path : ""); fputs(",\n", out);
    fputs("  \"parent_checkpoint\": ", out); json_write_escaped(out, parent_checkpoint ? parent_checkpoint : ""); fputs(",\n", out);
    fputs("  \"output_checkpoint\": ", out); json_write_escaped(out, output_checkpoint ? output_checkpoint : ""); fputs(",\n", out);
    fputs("  \"anchor_checkpoint\": ", out); json_write_escaped(out, anchor_checkpoint ? anchor_checkpoint : ""); fputs(",\n", out);
    fprintf(out, "  \"anchor_kl_coef\": %.6f,\n", anchor_kl_coef);
    fprintf(out, "  \"learning_rate\": %.9g,\n", trainer->learning_rate);
    fprintf(out, "  \"gamma\": %.9g,\n", trainer->gamma);
    fprintf(out, "  \"entropy_coef\": %.9g,\n", trainer->entropy_coef);
    fprintf(out, "  \"advantage_norm\": %s,\n", trainer->advantage_norm ? "true" : "false");
    fprintf(out, "  \"gae_lambda\": %.9g,\n", trainer->gae_lambda);
    fprintf(out, "  \"ppo_clip_epsilon\": %.9g,\n", trainer->ppo_clip_epsilon);
    fprintf(out, "  \"ppo_value_clip_epsilon\": %.9g,\n", trainer->ppo_value_clip_epsilon);
    fputs("  \"reward_mode\": ", out); json_write_escaped(out, reward_mode); fputs(",\n", out);
    fprintf(out, "  \"dense_additive_hp_swing_weight\": %.6f,\n", reward_config->dense_additive.hp_swing_weight);
    fprintf(out, "  \"dense_additive_faint_swing_weight\": %.6f,\n", reward_config->dense_additive.faint_swing_weight);
    fprintf(out, "  \"dense_additive_reward_clip\": %.6f,\n", reward_config->dense_additive.reward_clip);
    fprintf(out, "  \"episode_count\": %zu,\n", summary->episode_count);
    fprintf(out, "  \"available_episode_count\": %zu,\n", summary->available_episode_count);
    fprintf(out, "  \"selected_episode_count\": %zu,\n", summary->selected_episode_count);
    fprintf(out, "  \"episode_limit\": %d,\n", summary->episode_limit);
    fprintf(out, "  \"processed_episode_fraction\": %.6f,\n",
        summary->selected_episode_count > 0 ? (double)summary->episode_count / (double)summary->selected_episode_count : 0.0);
    fprintf(out, "  \"shuffle_seed\": %d,\n", summary->shuffle_seed);
    fprintf(out, "  \"minibatch_episodes\": %d,\n", summary->minibatch_episodes);
    fprintf(out, "  \"avg_episode_length\": %.6f,\n", summary->episode_count > 0 ? (double)summary->total_steps / (double)summary->episode_count : 0.0);
    fprintf(out, "  \"reward_mean\": %.6f,\n", summary->episode_count > 0 ? summary->return_sum / (double)summary->episode_count : 0.0);
    fprintf(out, "  \"reward_min\": %.6f,\n", summary->return_min);
    fprintf(out, "  \"reward_max\": %.6f,\n", summary->return_max);
    fprintf(out, "  \"outcome_wins\": %zu,\n", summary->outcome_wins);
    fprintf(out, "  \"outcome_losses\": %zu,\n", summary->outcome_losses);
    fprintf(out, "  \"outcome_draws\": %zu,\n", summary->outcome_draws);
    fprintf(out, "  \"tera_action_rate\": %.6f,\n", summary->move_count > 0 ? (double)summary->tera_count / (double)summary->move_count : 0.0);
    fprintf(out, "  \"tera_rate\": %.6f,\n", summary->move_count > 0 ? (double)summary->tera_count / (double)summary->move_count : 0.0);
    fputs("  \"metric_definitions\": {\"tera_action_rate\": \"tera move actions / move actions\", \"tera_rate\": \"backward-compatible alias of tera_action_rate\"},\n", out);
    fputs("  \"move_slot_rates\": {", out);
    for (i = 0; i < 4; ++i) {
        if (i) fputs(", ", out);
        fprintf(out, "\"slot_%zu\": %.6f", i + 1u, summary->move_count > 0 ? (double)summary->move_slot_counts[i] / (double)summary->move_count : 0.0);
    }
    fputs("},\n", out);
    fputs("  \"switch_slot_rates\": {", out);
    for (i = 0; i < 6; ++i) {
        if (i) fputs(", ", out);
        fprintf(out, "\"slot_%zu\": %.6f", i + 1u, summary->switch_count > 0 ? (double)summary->switch_slot_counts[i] / (double)summary->switch_count : 0.0);
    }
    fputs("},\n", out);
    fprintf(out, "  \"mean_value_prediction\": %.6f,\n", summary->label_weight_sum > 0 ? summary->value_mean_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"mean_advantage\": %.6f,\n", summary->label_weight_sum > 0 ? summary->advantage_mean_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"mean_absolute_advantage\": %.6f,\n", summary->label_weight_sum > 0 ? summary->abs_advantage_mean_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"policy_loss\": %.6f,\n", summary->label_weight_sum > 0 ? summary->policy_loss_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"value_loss\": %.6f,\n", summary->label_weight_sum > 0 ? summary->value_loss_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"entropy\": %.6f,\n", summary->label_weight_sum > 0 ? summary->entropy_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"anchor_loss\": %.6f,\n", summary->label_weight_sum > 0 ? summary->anchor_loss_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"anchor_kl_mean\": %.6f,\n", summary->label_weight_sum > 0 ? summary->anchor_kl_mean_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"approx_kl\": %.6f,\n", summary->label_weight_sum > 0 ? summary->approx_kl_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"clip_fraction\": %.6f,\n", summary->label_weight_sum > 0 ? summary->clip_fraction_sum / (double)summary->label_weight_sum : 0.0);
    fprintf(out, "  \"anchor_kl_max\": %.6f,\n", summary->anchor_kl_max);
    fprintf(out, "  \"target_kl\": %.6f,\n", trainer->target_kl);
    fprintf(out, "  \"target_kl_min_episodes\": %d,\n", summary->target_kl_min_episodes);
    fprintf(out, "  \"target_kl_min_labels\": %d,\n", summary->target_kl_min_labels);
    fprintf(out, "  \"target_kl_hard_multiplier\": %.6f,\n", summary->target_kl_hard_multiplier);
    fprintf(out, "  \"target_kl_hard_consecutive_updates\": %d,\n", summary->target_kl_hard_consecutive_updates);
    fprintf(out, "  \"target_kl_hard_breach_count\": %d,\n", summary->target_kl_hard_breach_count);
    fprintf(out, "  \"target_kl_exceeded\": %s,\n", summary->target_kl_exceeded ? "true" : "false");
    fprintf(out, "  \"target_kl_hard_stop\": %s,\n", summary->target_kl_hard_stop ? "true" : "false");
    fprintf(out, "  \"target_kl_trigger\": %.6f,\n", summary->target_kl_trigger);
    fprintf(out, "  \"labels\": %zu\n", summary->label_weight_sum);
    fputs("}\n", out);
    fclose(out);
    return 1;
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
    if (!env_runtime_init(&runtime, model, NULL, 1, ENV_REWARD_TERMINAL, NULL, NULL)) {
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
    unsigned int validation_seed,
    const RewardConfig* reward_config
) {
    char* resolved_checkpoint_path = NULL;
    GruModel* model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    TrainerCheckpointState checkpoint_state;
    CheckpointLoadResult checkpoint_result;
    size_t* train_indices = NULL;
    size_t* val_indices = NULL;
    size_t train_sessions = 0;
    size_t val_sessions = 0;
    PolicyEvaluationMetrics metrics;
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
    model = load_current_checkpoint(resolved_checkpoint_path, &checkpoint_state, &checkpoint_result);
    if (!model) {
        report_checkpoint_load_failure("[eval] failed to load", resolved_checkpoint_path, &checkpoint_result);
        free(resolved_checkpoint_path);
        return 1;
    }
    report_checkpoint_load_success("[eval] loaded", resolved_checkpoint_path, &checkpoint_state, &checkpoint_result);
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
    if (!build_split_indices(
            &runtime, validation_seed, &train_indices, &train_sessions, &val_indices, &val_sessions)) {
        fprintf(stderr, "Failed to build held-out split indices\n");
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }

    printf("[eval] checkpoint=%s train_sessions=%zu val_sessions=%zu validation_seed=%u\n",
        resolved_checkpoint_path, train_sessions, val_sessions, validation_seed);
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
        if (!evaluate_supervised_split(&trainer, model, &runtime, val_indices, val_sessions, &metrics)) {
        fprintf(stderr, "Failed to evaluate held-out split\n");
        free(train_indices);
        free(val_indices);
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
        }
        elapsed = elapsed_seconds_since(eval_start_clock);
        sessions_per_sec = elapsed > 0.0 ? (double)metrics.sessions / elapsed : 0.0;
        labels_per_sec = elapsed > 0.0 ? (double)metrics.action_labels / elapsed : 0.0;
        printf("[eval] validation action_loss=%.4f value_loss=%.4f accuracy=%.4f labels=%zu sessions=%zu\n",
            policy_evaluation_full_turn_nll(&metrics),
            policy_evaluation_value_loss(&metrics),
            policy_evaluation_full_turn_accuracy(&metrics),
            metrics.action_labels,
            metrics.sessions);
        printf("[eval] validation top3_accuracy=%.4f action1_accuracy=%.4f action2_accuracy=%.4f action1_labels=%zu action2_labels=%zu skipped_steps=%zu\n",
            policy_evaluation_top3_accuracy(&metrics),
            policy_evaluation_slot0_accuracy(&metrics),
            policy_evaluation_slot1_accuracy(&metrics),
            metrics.slot0_labels,
            metrics.slot1_labels,
            metrics.skipped_turns);
        printf("[eval] validation metrics_version=%d action_nll=%.9g target_nll=%.9g full_turn_nll=%.9g value_loss=%.9g full_turn_accuracy=%.9g top3_accuracy=%.9g slot0_accuracy=%.9g slot1_accuracy=%.9g joint_pair_accuracy=%.9g kind_accuracy=%.9g move_accuracy=%.9g switch_accuracy=%.9g tera_accuracy=%.9g target_accuracy=%.9g turns=%zu action_labels=%zu slot0_labels=%zu slot1_labels=%zu joint_pairs=%zu kind_labels=%zu move_labels=%zu switch_labels=%zu tera_labels=%zu target_labels=%zu skipped_turns=%zu illegal_predictions=%zu nonfinite_values=%zu\n",
            POLICY_EVALUATION_METRICS_VERSION,
            policy_evaluation_action_nll(&metrics),
            policy_evaluation_target_nll(&metrics),
            policy_evaluation_full_turn_nll(&metrics),
            policy_evaluation_value_loss(&metrics),
            policy_evaluation_full_turn_accuracy(&metrics),
            policy_evaluation_top3_accuracy(&metrics),
            policy_evaluation_slot0_accuracy(&metrics),
            policy_evaluation_slot1_accuracy(&metrics),
            policy_evaluation_joint_pair_accuracy(&metrics),
            policy_evaluation_kind_accuracy(&metrics),
            policy_evaluation_move_accuracy(&metrics),
            policy_evaluation_switch_accuracy(&metrics),
            policy_evaluation_tera_accuracy(&metrics),
            policy_evaluation_target_accuracy(&metrics),
            metrics.decision_turns,
            metrics.action_labels,
            metrics.slot0_labels,
            metrics.slot1_labels,
            metrics.joint_pair_labels,
            metrics.kind_labels,
            metrics.move_labels,
            metrics.switch_labels,
            metrics.tera_labels,
            metrics.target_labels,
            metrics.skipped_turns,
            metrics.illegal_predictions,
            metrics.nonfinite_values);
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
    CheckpointLoadResult checkpoint_result;
    EnvRuntime runtime;
    char line[16384];
    char json[512];
    char* resolved_checkpoint_path = NULL;
    FILE* replay_file = NULL;
    const char* replay_path = getenv("PORYGON_REPLAY_PATH");

    memset(&state, 0, sizeof(state));
    memset(&checkpoint_result, 0, sizeof(checkpoint_result));
    if (checkpoint_path && *checkpoint_path) {
        resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path);
        if (!resolved_checkpoint_path) {
            fprintf(stderr, "[runtime] failed to resolve checkpoint path '%s'\n", checkpoint_path);
            return 1;
        }
        model = load_current_checkpoint(resolved_checkpoint_path, &state, &checkpoint_result);
        if (!model) {
            report_checkpoint_load_failure("[runtime] failed to load", resolved_checkpoint_path, &checkpoint_result);
            free(resolved_checkpoint_path);
            return 1;
        }
        report_checkpoint_load_success("[runtime] loaded", resolved_checkpoint_path, &state, &checkpoint_result);
    }
    if (!model) {
        model = create_default_model();
        fprintf(stderr, "[runtime] no checkpoint provided, starting fresh model\n");
    }
    if (!model) {
        fprintf(stderr, "Failed to initialize runtime model\n");
        return 1;
    }
    if (replay_path && *replay_path) {
        replay_file = fopen(replay_path, "a");
    }
    if (!env_runtime_init(&runtime, model, replay_file, 0, ENV_REWARD_TERMINAL, NULL,
            resolved_checkpoint_path ? resolved_checkpoint_path : checkpoint_path)) {
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
    int ppo_mode,
    int epochs,
    float learning_rate_override,
    float rl_gamma,
    float rl_entropy_coef,
    int rl_advantage_norm,
    float gae_lambda,
    float ppo_clip_epsilon,
    float ppo_value_clip_epsilon,
    float ppo_target_kl,
    int ppo_target_kl_min_episodes,
    int ppo_target_kl_min_labels,
    float ppo_target_kl_hard_multiplier,
    int ppo_target_kl_hard_consecutive_updates,
    int ppo_shuffle_seed,
    int ppo_episode_limit,
    int ppo_minibatch_episodes,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon,
    int supervised_profile,
    GruSupervisedOptimizer supervised_optimizer,
    unsigned int validation_seed,
    int aux_checkpoints,
    const char* rl_reward_mode,
    const RewardConfig* reward_config,
    const char* expected_policy_tag,
    const char* training_summary_path,
    const char* parent_checkpoint_metadata,
    const char* anchor_checkpoint_path,
    float anchor_kl_coef
) {
    char* resolved_checkpoint_path = NULL;
    char* resolved_anchor_checkpoint_path = NULL;
    GruModel* model = NULL;
    GruModel* anchor_model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    EnvRewardMode reward_mode;
    TrainerCheckpointState checkpoint_state;
    TrainerCheckpointState anchor_checkpoint_state;
    CheckpointLoadResult checkpoint_result;
    CheckpointLoadResult anchor_checkpoint_result;
    RlTrainingSummary rl_summary;
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
    size_t starting_step;

    if (!input_path || !checkpoint_path || epochs <= 0) {
        return 1;
    }
    if (ppo_episode_limit < 0) {
        fprintf(stderr, "--episode-limit must be >= 0\n");
        return 1;
    }
    rl_training_summary_init(&rl_summary);
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
    memset(&anchor_checkpoint_state, 0, sizeof(anchor_checkpoint_state));
    model = load_current_checkpoint(resolved_checkpoint_path, &checkpoint_state, &checkpoint_result);
    if (!model) {
        if (checkpoint_result.status == CHECKPOINT_LOAD_NOT_FOUND) {
            model = create_default_model();
            printf("[train] checkpoint not found; starting fresh model -> %s\n", resolved_checkpoint_path);
        } else {
            report_checkpoint_load_failure("[train] refusing incompatible", resolved_checkpoint_path, &checkpoint_result);
            free(resolved_checkpoint_path);
            return 1;
        }
    } else {
        report_checkpoint_load_success("[train] loaded", resolved_checkpoint_path, &checkpoint_state, &checkpoint_result);
    }
    if (!model) {
        free(resolved_checkpoint_path);
        return 1;
    }
    if (rl_mode && anchor_checkpoint_path && *anchor_checkpoint_path && anchor_kl_coef > 0.0f) {
        resolved_anchor_checkpoint_path = resolve_checkpoint_path(anchor_checkpoint_path);
        if (!resolved_anchor_checkpoint_path) {
            fprintf(stderr, "Failed to resolve anchor checkpoint path\n");
            gru_model_destroy(model);
            free(resolved_checkpoint_path);
            return 1;
        }
        anchor_model = load_current_checkpoint(resolved_anchor_checkpoint_path, &anchor_checkpoint_state, &anchor_checkpoint_result);
        if (!anchor_model) {
            report_checkpoint_load_failure("[train] failed to load anchor", resolved_anchor_checkpoint_path, &anchor_checkpoint_result);
            gru_model_destroy(model);
            free(resolved_anchor_checkpoint_path);
            free(resolved_checkpoint_path);
            return 1;
        }
        report_checkpoint_load_success("[train] loaded anchor", resolved_anchor_checkpoint_path, &anchor_checkpoint_state, &anchor_checkpoint_result);
        if (gru_model_input_dim(anchor_model) != gru_model_input_dim(model) ||
                gru_model_hidden_dim(anchor_model) != gru_model_hidden_dim(model) ||
                gru_model_num_actions(anchor_model) != gru_model_num_actions(model)) {
            fprintf(stderr, "anchor checkpoint incompatible current=%s anchor=%s\n", resolved_checkpoint_path, resolved_anchor_checkpoint_path);
            gru_model_destroy(anchor_model);
            gru_model_destroy(model);
            free(resolved_anchor_checkpoint_path);
            free(resolved_checkpoint_path);
            return 1;
        }
    }
    gru_trainer_init(&trainer,
        learning_rate_override > 0.0f
            ? learning_rate_override
            : (checkpoint_state.learning_rate > 0.0f ? checkpoint_state.learning_rate : 0.01f),
        checkpoint_state.bptt_window ? checkpoint_state.bptt_window : 16,
        checkpoint_state.gradient_clip,
        checkpoint_state.seed);
    trainer.step = checkpoint_state.step;
    starting_step = trainer.step;
    if (rl_mode) {
        trainer.gamma = rl_gamma;
        trainer.entropy_coef = rl_entropy_coef;
        trainer.advantage_norm = rl_advantage_norm ? 1 : 0;
        trainer.gae_lambda = gae_lambda;
        trainer.ppo_clip_epsilon = ppo_clip_epsilon;
        trainer.ppo_value_clip_epsilon = ppo_value_clip_epsilon;
        trainer.target_kl = ppo_target_kl;
        trainer.adam_beta1 = adam_beta1;
        trainer.adam_beta2 = adam_beta2;
        trainer.adam_epsilon = adam_epsilon;
        trainer.anchor_model = anchor_model;
        trainer.anchor_kl_coef = anchor_kl_coef;
    } else {
        trainer.supervised_profile_enabled = supervised_profile;
        trainer.supervised_optimizer = supervised_optimizer;
    }

    if (!(input_is_episode_batch
            ? load_runtime_from_episode_batch_file(
                input_path,
                model,
                &runtime,
                reward_mode,
                reward_config ? &reward_config->dense_additive : NULL,
                expected_policy_tag)
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
    if (!(rl_mode
            ? build_all_train_indices(runtime.count, &train_indices, &train_sessions, &val_indices, &val_sessions)
            : build_split_indices(
                &runtime, validation_seed, &train_indices, &train_sessions, &val_indices, &val_sessions))) {
        fprintf(stderr, "Failed to build train/validation split indices\n");
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        free(resolved_checkpoint_path);
        return 1;
    }
    if (rl_mode) {
        size_t train_sessions_before_filter = train_sessions;
        filter_labeled_train_indices(&runtime, train_indices, &train_sessions);
        rl_summary.available_episode_count = train_sessions;
        rl_summary.episode_limit = ppo_episode_limit;
        rl_summary.shuffle_seed = ppo_shuffle_seed;
        rl_summary.minibatch_episodes = ppo_minibatch_episodes > 0 ? ppo_minibatch_episodes : 1;
        rl_summary.target_kl_min_episodes = ppo_target_kl_min_episodes;
        rl_summary.target_kl_min_labels = ppo_target_kl_min_labels;
        rl_summary.target_kl_hard_multiplier = ppo_target_kl_hard_multiplier;
        rl_summary.target_kl_hard_consecutive_updates =
            ppo_target_kl_hard_consecutive_updates > 0 ? ppo_target_kl_hard_consecutive_updates : 1;
        printf("[train-rl] filtered train_sessions labeled_only=%zu/%zu\n",
            train_sessions, train_sessions_before_filter);
        if (ppo_episode_limit > 0 && train_sessions > (size_t)ppo_episode_limit) {
            if (ppo_shuffle_seed >= 0) {
                srand((unsigned int)ppo_shuffle_seed);
            }
            shuffle_indices(train_indices, train_sessions);
            train_sessions = (size_t)ppo_episode_limit;
            printf("[train-rl] deterministic episode subset selected=%zu/%zu seed=%d\n",
                train_sessions, rl_summary.available_episode_count, ppo_shuffle_seed);
        }
        rl_summary.selected_episode_count = train_sessions;
        printf("[train-rl] validation_split disabled train_sessions=%zu val_sessions=%zu\n",
            train_sessions,
            val_sessions);
    }
    printf("[train] split train_sessions=%zu val_sessions=%zu epochs=%d validation_seed=%u aux_checkpoints=%d\n",
        train_sessions, val_sessions, epochs, validation_seed, aux_checkpoints);
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
    } else {
        printf("[train] supervised optimizer=%s\n",
            gru_supervised_optimizer_name(trainer.supervised_optimizer));
    }

    for (epoch = 1; epoch <= epochs; ++epoch) {
        size_t trained_in_epoch = 0;
        int ppo_early_stop = 0;
        int consecutive_hard_kl_updates = 0;
        PolicyEvaluationMetrics val_metrics;
        clock_t val_eval_start_clock = 0;

        printf("[train] epoch %d/%d start\n", epoch, epochs);
        train_loop_start_clock = clock();
        if (rl_mode && ppo_shuffle_seed >= 0) {
            srand((unsigned int)ppo_shuffle_seed + (unsigned int)(epoch - 1) * 0x9e3779b9u);
        }
        shuffle_indices(train_indices, train_sessions);

        for (size_t order_i = 0; order_i < train_sessions;) {
            size_t session_index = train_indices[order_i];
            size_t episodes_this_update = 1;
            if (rl_mode) {
                if (ppo_mode) {
                    size_t configured_batch = (size_t)(ppo_minibatch_episodes > 0 ? ppo_minibatch_episodes : 1);
                    size_t remaining = train_sessions - order_i;
                    const Episode** minibatch;
                    episodes_this_update = configured_batch < remaining ? configured_batch : remaining;
                    minibatch = (const Episode**)malloc(episodes_this_update * sizeof(*minibatch));
                    if (!minibatch) {
                        fprintf(stderr, "Failed to allocate PPO minibatch\n");
                        free(train_indices);
                        free(val_indices);
                        env_runtime_free(&runtime);
                        gru_model_destroy(model);
                        free(resolved_checkpoint_path);
                        return 1;
                    }
                    for (size_t batch_i = 0; batch_i < episodes_this_update; ++batch_i) {
                        size_t batch_session_index = train_indices[order_i + batch_i];
                        minibatch[batch_i] = &runtime.sessions[batch_session_index].episode;
                    }
                    if (!gru_trainer_ppo_minibatch(&trainer, model, minibatch, episodes_this_update)) {
                        free(minibatch);
                        fprintf(stderr, "Failed PPO minibatch update\n");
                        free(train_indices);
                        free(val_indices);
                        env_runtime_free(&runtime);
                        gru_model_destroy(model);
                        free(resolved_checkpoint_path);
                        return 1;
                    }
                    free(minibatch);
                } else if (!gru_trainer_policy_gradient_episode(
                        &trainer, model, &runtime.sessions[session_index].episode)) {
                    fprintf(stderr, "Failed RL training episode\n");
                    free(train_indices);
                    free(val_indices);
                    env_runtime_free(&runtime);
                    gru_model_destroy(model);
                    free(resolved_checkpoint_path);
                    return 1;
                }
                for (size_t batch_i = 0; batch_i < episodes_this_update; ++batch_i) {
                    size_t batch_session_index = train_indices[order_i + batch_i];
                    rl_training_summary_record_episode(&rl_summary, &runtime.sessions[batch_session_index].episode);
                }
                rl_training_summary_record_trainer(&rl_summary, &trainer);
                if (ppo_mode && trainer.target_kl > 0.0f) {
                    float running_approx_kl = rl_summary.label_weight_sum > 0
                        ? (float)(rl_summary.approx_kl_sum / (double)rl_summary.label_weight_sum)
                        : 0.0f;
                    int minimum_reached =
                        rl_summary.episode_count >= (size_t)(ppo_target_kl_min_episodes > 0 ? ppo_target_kl_min_episodes : 0) &&
                        rl_summary.label_weight_sum >= (size_t)(ppo_target_kl_min_labels > 0 ? ppo_target_kl_min_labels : 0);
                    int required_hard_breaches = ppo_target_kl_hard_consecutive_updates > 0
                        ? ppo_target_kl_hard_consecutive_updates
                        : 1;
                    int hard_stop = gru_trainer_ppo_hard_kl_stop_update(
                        trainer.last_approx_kl,
                        trainer.target_kl,
                        ppo_target_kl_hard_multiplier,
                        required_hard_breaches,
                        &consecutive_hard_kl_updates);
                    if (consecutive_hard_kl_updates > rl_summary.target_kl_hard_breach_count) {
                        rl_summary.target_kl_hard_breach_count = consecutive_hard_kl_updates;
                    }
                    if ((minimum_reached && running_approx_kl > trainer.target_kl) || hard_stop) {
                        ppo_early_stop = 1;
                        rl_summary.target_kl_exceeded = 1;
                        rl_summary.target_kl_hard_stop = hard_stop ? 1 : 0;
                        rl_summary.target_kl_trigger = hard_stop ? trainer.last_approx_kl : running_approx_kl;
                    }
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
            trained_in_epoch += episodes_this_update;
            order_i += episodes_this_update;
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
                    printf("[train-%s] epoch=%d episodes=%zu/%zu step=%zu mean_return=%.4f policy_loss=%.4f value_loss=%.4f mean_advantage=%.4f entropy=%.4f approx_kl=%.4f anchor_kl_mean=%.4f anchor_kl_max=%.4f clip_fraction=%.4f hard_kl_breaches=%d/%d labels=%zu\n",
                        ppo_mode ? "ppo" : "rl",
                        epoch,
                        trained_in_epoch,
                        train_sessions,
                        trainer.step,
                        trainer.last_mean_return,
                        trainer.last_policy_loss,
                        trainer.last_value_loss,
                        trainer.last_mean_advantage,
                        trainer.last_entropy,
                        trainer.last_approx_kl,
                        trainer.last_anchor_kl_mean,
                        trainer.last_anchor_kl_max,
                        trainer.last_clip_fraction,
                        consecutive_hard_kl_updates,
                        ppo_target_kl_hard_consecutive_updates > 0 ? ppo_target_kl_hard_consecutive_updates : 1,
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
            if (!rl_mode && aux_checkpoints &&
                    ((trained_in_epoch % 500u) == 0u || trained_in_epoch == train_sessions)) {
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
            if (ppo_early_stop) {
                break;
            }
        }

        if (ppo_early_stop) {
            printf("[train-ppo] early stop after epoch=%d episodes=%zu/%zu reason=%s kl_trigger=%.4f target_kl=%.4f\n",
                epoch,
                trained_in_epoch,
                train_sessions,
                rl_summary.target_kl_hard_stop ? "target_kl_hard_limit" : "target_kl_running_mean",
                rl_summary.target_kl_trigger,
                trainer.target_kl);
        }

        if (val_sessions > 0 && !rl_mode) {
            TrainerCheckpointState best_state;
            char* best_path = NULL;
            double val_elapsed;
            double val_sessions_per_sec;
            double val_labels_per_sec;
            val_eval_start_clock = clock();
            if (!evaluate_supervised_split(
                    &trainer, model, &runtime, val_indices, val_sessions, &val_metrics)) {
                fprintf(stderr, "Failed held-out validation evaluation\n");
                free(train_indices);
                free(val_indices);
                env_runtime_free(&runtime);
                gru_model_destroy(model);
                free(resolved_checkpoint_path);
                return 1;
            }
            val_elapsed = elapsed_seconds_since(val_eval_start_clock);
            val_sessions_per_sec = val_elapsed > 0.0 ? (double)val_metrics.sessions / val_elapsed : 0.0;
            val_labels_per_sec = val_elapsed > 0.0 ? (double)val_metrics.action_labels / val_elapsed : 0.0;
            if (val_metrics.action_labels > 0) {
                printf("[train] epoch=%d validation action_loss=%.4f value_loss=%.4f accuracy=%.4f labels=%zu\n",
                    epoch,
                    policy_evaluation_full_turn_nll(&val_metrics),
                    policy_evaluation_value_loss(&val_metrics),
                    policy_evaluation_full_turn_accuracy(&val_metrics),
                    val_metrics.action_labels);
                printf("[train] epoch=%d validation top3_accuracy=%.4f action1_accuracy=%.4f action2_accuracy=%.4f action1_labels=%zu action2_labels=%zu skipped_steps=%zu\n",
                    epoch,
                    policy_evaluation_top3_accuracy(&val_metrics),
                    policy_evaluation_slot0_accuracy(&val_metrics),
                    policy_evaluation_slot1_accuracy(&val_metrics),
                    val_metrics.slot0_labels,
                    val_metrics.slot1_labels,
                    val_metrics.skipped_turns);
                printf("[train] epoch=%d validation metrics_version=%d action_nll=%.9g target_nll=%.9g full_turn_nll=%.9g value_loss=%.9g full_turn_accuracy=%.9g top3_accuracy=%.9g slot0_accuracy=%.9g slot1_accuracy=%.9g joint_pair_accuracy=%.9g kind_accuracy=%.9g move_accuracy=%.9g switch_accuracy=%.9g tera_accuracy=%.9g target_accuracy=%.9g turns=%zu action_labels=%zu slot0_labels=%zu slot1_labels=%zu joint_pairs=%zu kind_labels=%zu move_labels=%zu switch_labels=%zu tera_labels=%zu target_labels=%zu skipped_turns=%zu illegal_predictions=%zu nonfinite_values=%zu\n",
                    epoch,
                    POLICY_EVALUATION_METRICS_VERSION,
                    policy_evaluation_action_nll(&val_metrics),
                    policy_evaluation_target_nll(&val_metrics),
                    policy_evaluation_full_turn_nll(&val_metrics),
                    policy_evaluation_value_loss(&val_metrics),
                    policy_evaluation_full_turn_accuracy(&val_metrics),
                    policy_evaluation_top3_accuracy(&val_metrics),
                    policy_evaluation_slot0_accuracy(&val_metrics),
                    policy_evaluation_slot1_accuracy(&val_metrics),
                    policy_evaluation_joint_pair_accuracy(&val_metrics),
                    policy_evaluation_kind_accuracy(&val_metrics),
                    policy_evaluation_move_accuracy(&val_metrics),
                    policy_evaluation_switch_accuracy(&val_metrics),
                    policy_evaluation_tera_accuracy(&val_metrics),
                    policy_evaluation_target_accuracy(&val_metrics),
                    val_metrics.decision_turns,
                    val_metrics.action_labels,
                    val_metrics.slot0_labels,
                    val_metrics.slot1_labels,
                    val_metrics.joint_pair_labels,
                    val_metrics.kind_labels,
                    val_metrics.move_labels,
                    val_metrics.switch_labels,
                    val_metrics.tera_labels,
                    val_metrics.target_labels,
                    val_metrics.skipped_turns,
                    val_metrics.illegal_predictions,
                    val_metrics.nonfinite_values);
                printf("[train] epoch=%d validation elapsed=%.1fs sessions_per_sec=%.2f labels_per_sec=%.2f\n",
                    epoch,
                    val_elapsed,
                    val_sessions_per_sec,
                    val_labels_per_sec);
                if (aux_checkpoints &&
                        (!has_best_val || policy_evaluation_full_turn_nll(&val_metrics) < best_val_action_loss)) {
                    has_best_val = 1;
                    best_val_action_loss = (float)policy_evaluation_full_turn_nll(&val_metrics);
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

        if (!rl_mode && aux_checkpoints) {
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

        if (ppo_early_stop) {
            break;
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
        printf("trained mode=%s step=%zu policy_loss=%.4f value_loss=%.4f mean_return=%.4f mean_advantage=%.4f entropy=%.4f approx_kl=%.4f clip_fraction=%.4f labels=%zu sessions=%zu\n",
            ppo_mode ? "ppo" : "rl",
            trainer.step,
            trainer.last_policy_loss,
            trainer.last_value_loss,
            trainer.last_mean_return,
            trainer.last_mean_advantage,
            trainer.last_entropy,
            trainer.last_approx_kl,
            trainer.last_clip_fraction,
            trainer.last_rl_labels,
            runtime.count);
        if (training_summary_path && *training_summary_path) {
            if (!write_rl_training_summary_json(
                    training_summary_path,
                    &rl_summary,
                    &trainer,
                    input_path,
                    parent_checkpoint_metadata && *parent_checkpoint_metadata
                        ? parent_checkpoint_metadata
                        : resolved_checkpoint_path,
                    resolved_checkpoint_path,
                    resolved_anchor_checkpoint_path ? resolved_anchor_checkpoint_path : "",
                    anchor_kl_coef,
                    rl_reward_mode,
                    reward_config)) {
                fprintf(stderr, "Failed to write RL training summary '%s'\n", training_summary_path);
            }
        }
    } else {
        printf("trained mode=supervised step=%zu action_loss=%.4f value_loss=%.4f accuracy=%.4f labels=%zu sessions=%zu\n",
            trainer.step,
            trainer.last_action_loss,
            trainer.last_value_loss,
            trainer.last_accuracy,
            trainer.step - starting_step,
            runtime.count);
    }
    printf("[train] saved checkpoint %s\n", resolved_checkpoint_path);
    free(train_indices);
    free(val_indices);
    env_runtime_free(&runtime);
    gru_model_destroy(anchor_model);
    gru_model_destroy(model);
    free(resolved_anchor_checkpoint_path);
    free(resolved_checkpoint_path);
    return 0;
}

static int episode_has_target_choice(const Episode* episode) {
    size_t i;
    if (!episode) return 0;
    for (i = 0; i < episode->count; ++i) {
        const FactorizedActionChoice* choice = &episode->factorized_actions[i];
        if ((choice->slot0_has_action && choice->slot0_kind == FACTORIZED_ACTION_MOVE &&
                choice->slot0_target_mask != 0u) ||
                (choice->slot1_has_action && choice->slot1_kind == FACTORIZED_ACTION_MOVE &&
                choice->slot1_target_mask != 0u)) {
            return 1;
        }
    }
    return 0;
}

static int select_overfit_sessions(const EnvRuntime* runtime, size_t selected[2]) {
    size_t found = 0;
    size_t i;
    if (!runtime || !selected) return 0;
    for (i = 0; i < runtime->count; ++i) {
        const EnvSession* session = &runtime->sessions[i];
        size_t position;
        if (!episode_has_labels(&session->episode) ||
                !episode_has_target_choice(&session->episode)) {
            continue;
        }
        for (position = 0; position < found; ++position) {
            if (strcmp(session->battle_id, runtime->sessions[selected[position]].battle_id) < 0) {
                break;
            }
        }
        if (position >= 2) continue;
        if (found < 2) ++found;
        if (found == 2 && position == 0) selected[1] = selected[0];
        selected[position] = i;
    }
    return found == 2;
}

static int run_supervised_overfit_check(
    const char* replay_path,
    const char* report_path,
    size_t epochs,
    float learning_rate,
    unsigned int seed,
    GruSupervisedOptimizer optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon,
    const RewardConfig* reward_config
) {
    GruModel* model = NULL;
    GruTrainer trainer;
    EnvRuntime runtime;
    size_t selected[2];
    const Episode* episodes[2];
    SupervisedOverfitResult result;
    int rc = 1;

    if (!replay_path || !report_path || epochs == 0) return 1;
    srand(seed);
    model = create_default_model();
    if (!model) {
        fprintf(stderr, "[overfit] failed to create fresh current-architecture model\n");
        return 1;
    }
    if (!load_runtime_from_replay_file(
            replay_path,
            model,
            &runtime,
            ENV_REWARD_TERMINAL,
            reward_config ? &reward_config->dense_additive : NULL)) {
        gru_model_destroy(model);
        return 1;
    }
    if (!select_overfit_sessions(&runtime, selected)) {
        fprintf(stderr,
            "[overfit] replay needs at least two labelled sessions with explicit target choices\n");
        env_runtime_free(&runtime);
        gru_model_destroy(model);
        return 1;
    }

    gru_trainer_init(&trainer, learning_rate, 16u, 1.0f, seed);
    trainer.supervised_optimizer = optimizer;
    trainer.supervised_profile_enabled = 0;
    trainer.adam_beta1 = adam_beta1;
    trainer.adam_beta2 = adam_beta2;
    trainer.adam_epsilon = adam_epsilon;
    episodes[0] = &runtime.sessions[selected[0]].episode;
    episodes[1] = &runtime.sessions[selected[1]].episode;
    printf("[overfit] sessions=%s,%s epochs=%zu optimizer=%s learning_rate=%.9g\n",
        runtime.sessions[selected[0]].battle_id,
        runtime.sessions[selected[1]].battle_id,
        epochs,
        gru_supervised_optimizer_name(optimizer),
        learning_rate);
    if (!learning_diagnostic_run_supervised_overfit(
            &trainer, model, episodes, 2u, epochs, &result)) {
        fprintf(stderr, "[overfit] diagnostic execution failed\n");
    } else if (!learning_diagnostic_write_supervised_report(
            report_path,
            replay_path,
            runtime.sessions[selected[0]].battle_id,
            runtime.sessions[selected[1]].battle_id,
            seed,
            epochs,
            &trainer,
            &result)) {
        fprintf(stderr, "[overfit] failed to write report '%s': %s\n", report_path, strerror(errno));
    } else {
        printf("[overfit] passed=%d action_loss_reduction=%.4f full_turn_accuracy=%.4f report=%s\n",
            result.passed,
            result.action_loss_reduction,
            policy_evaluation_full_turn_accuracy(&result.after),
            report_path);
        rc = result.passed ? 0 : 1;
    }
    env_runtime_free(&runtime);
    gru_model_destroy(model);
    return rc;
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
    int overfit_command = argc >= 2 && strcmp(argv[1], "--check-supervised-overfit") == 0;
    int overfit_epochs = parse_int_flag(argc, argv, "--epochs", 200);
    int overfit_seed = parse_int_flag(argc, argv, "--seed", 20260902);
    float learning_rate_override;
    const char* expected_policy_tag = parse_string_flag(argc, argv, "--policy-tag-expected", "");
    const char* training_summary_path = parse_string_flag(argc, argv, "--training-summary-path", "");
    const char* parent_checkpoint_metadata = parse_string_flag(argc, argv, "--parent-checkpoint", "");
    const char* anchor_checkpoint_path = parse_string_flag(argc, argv, "--anchor-checkpoint", "");
    float anchor_kl_coef;
    float rl_gamma;
    float rl_entropy_coef;
    int rl_advantage_norm;
    float gae_lambda;
    float ppo_clip_epsilon;
    float ppo_value_clip_epsilon;
    float ppo_target_kl;
    int ppo_target_kl_min_episodes;
    int ppo_target_kl_min_labels;
    float ppo_target_kl_hard_multiplier;
    int ppo_target_kl_hard_consecutive_updates;
    int ppo_shuffle_seed;
    int ppo_episode_limit;
    int ppo_minibatch_episodes;
    float adam_beta1;
    float adam_beta2;
    float adam_epsilon;
    int supervised_profile = parse_bool01_flag(argc, argv, "--supervised-profile", 1);
    int validation_seed = parse_int_flag(argc, argv, "--validation-seed", 1337);
    int aux_checkpoints = parse_bool01_flag(argc, argv, "--aux-checkpoints", 1);
    GruSupervisedOptimizer supervised_optimizer;
    const char* supervised_optimizer_name = parse_string_flag(
        argc, argv, "--supervised-optimizer", overfit_command ? "adam" : "sgd");
    const char* rl_reward_mode = parse_string_flag(argc, argv, "--reward-mode", "terminal");
    RewardConfig reward_config;
    RlDefaultsConfig rl_defaults;
    int ppo_command = argc >= 2 && strcmp(argv[1], "--train-live-ppo") == 0;
    int training_or_eval_mode = 0;
    srand((unsigned int)time(NULL));
    if (!load_reward_config_file(SHOWDOWN_CLIENT_REWARD_CONFIG_PATH, &reward_config)) {
        return 1;
    }
    if (!load_rl_defaults_file(SHOWDOWN_CLIENT_RL_DEFAULTS_PATH, &rl_defaults)) {
        return 1;
    }
    learning_rate_override = parse_float_flag(argc, argv, "--learning-rate", -1.0f);
    anchor_kl_coef = parse_float_flag(argc, argv, "--anchor-kl-coef", 0.0f);
    rl_gamma = parse_float_flag(argc, argv, "--gamma",
        ppo_command ? rl_defaults.ppo_gamma : rl_defaults.policy_gradient_gamma);
    rl_entropy_coef = parse_float_flag(argc, argv, "--entropy-coef",
        ppo_command ? rl_defaults.ppo_entropy_coef : rl_defaults.policy_gradient_entropy_coef);
    rl_advantage_norm = parse_int_flag(argc, argv, "--advantage-norm", rl_defaults.advantage_norm);
    gae_lambda = parse_float_flag(argc, argv, "--gae-lambda", rl_defaults.gae_lambda);
    ppo_clip_epsilon = parse_float_flag(argc, argv, "--ppo-clip-epsilon", rl_defaults.ppo_clip_epsilon);
    ppo_value_clip_epsilon = parse_float_flag(argc, argv, "--ppo-value-clip-epsilon", rl_defaults.ppo_value_clip_epsilon);
    ppo_target_kl = parse_float_flag(argc, argv, "--target-kl", rl_defaults.ppo_target_kl);
    ppo_target_kl_min_episodes = parse_int_flag(argc, argv, "--target-kl-min-episodes", rl_defaults.ppo_target_kl_min_episodes);
    ppo_target_kl_min_labels = parse_int_flag(argc, argv, "--target-kl-min-labels", rl_defaults.ppo_target_kl_min_labels);
    ppo_target_kl_hard_multiplier = parse_float_flag(argc, argv, "--target-kl-hard-multiplier", rl_defaults.ppo_target_kl_hard_multiplier);
    ppo_target_kl_hard_consecutive_updates = parse_int_flag(
        argc, argv, "--target-kl-hard-consecutive-updates", rl_defaults.ppo_target_kl_hard_consecutive_updates);
    ppo_shuffle_seed = parse_int_flag(argc, argv, "--shuffle-seed", rl_defaults.ppo_shuffle_seed);
    ppo_episode_limit = parse_int_flag(argc, argv, "--episode-limit", 0);
    ppo_minibatch_episodes = parse_int_flag(argc, argv, "--ppo-minibatch-episodes", rl_defaults.ppo_minibatch_episodes);
    adam_beta1 = parse_float_flag(argc, argv, "--adam-beta1", rl_defaults.adam_beta1);
    adam_beta2 = parse_float_flag(argc, argv, "--adam-beta2", rl_defaults.adam_beta2);
    adam_epsilon = parse_float_flag(argc, argv, "--adam-epsilon", rl_defaults.adam_epsilon);
    if (!parse_supervised_optimizer(supervised_optimizer_name, &supervised_optimizer)) {
        fprintf(stderr,
            "Unsupported --supervised-optimizer '%s'. Supported optimizers: sgd, adam.\n",
            supervised_optimizer_name);
        return 1;
    }
    if (overfit_command && (overfit_epochs <= 0 || overfit_seed < 0)) {
        fprintf(stderr, "--check-supervised-overfit requires --epochs > 0 and --seed >= 0\n");
        return 1;
    }
    if (validation_seed < 0) {
        fprintf(stderr, "--validation-seed must be >= 0\n");
        return 1;
    }
    {
        float override_value;
        override_value = parse_float_flag(argc, argv, "--dense-additive-hp-swing-weight", NAN);
        if (!isnan(override_value)) {
            reward_config.dense_additive.hp_swing_weight = override_value;
        }
        override_value = parse_float_flag(argc, argv, "--dense-additive-faint-swing-weight", NAN);
        if (!isnan(override_value)) {
            reward_config.dense_additive.faint_swing_weight = override_value;
        }
        override_value = parse_float_flag(argc, argv, "--dense-additive-reward-clip", NAN);
        if (!isnan(override_value)) {
            reward_config.dense_additive.reward_clip = override_value;
        }
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
            strcmp(argv[1], "--train-live-ppo") == 0 ||
            strcmp(argv[1], "--check-supervised-overfit") == 0 ||
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
    if (argc >= 4 && overfit_command) {
        return run_supervised_overfit_check(
            argv[2],
            argv[3],
            (size_t)overfit_epochs,
            learning_rate_override > 0.0f ? learning_rate_override : 0.001f,
            (unsigned int)overfit_seed,
            supervised_optimizer,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            &reward_config);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-supervised") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            0,
            0,
            0,
            epochs,
            learning_rate_override,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            gae_lambda,
            ppo_clip_epsilon,
            ppo_value_clip_epsilon,
            ppo_target_kl,
            ppo_target_kl_min_episodes,
            ppo_target_kl_min_labels,
            ppo_target_kl_hard_multiplier,
            ppo_target_kl_hard_consecutive_updates,
            ppo_shuffle_seed,
            ppo_episode_limit,
            ppo_minibatch_episodes,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            supervised_profile,
            supervised_optimizer,
            (unsigned int)validation_seed,
            aux_checkpoints,
            rl_reward_mode,
            &reward_config,
            expected_policy_tag,
            training_summary_path,
            parent_checkpoint_metadata,
            anchor_checkpoint_path,
            anchor_kl_coef);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-rl") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            0,
            1,
            0,
            epochs,
            learning_rate_override,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            gae_lambda,
            ppo_clip_epsilon,
            ppo_value_clip_epsilon,
            ppo_target_kl,
            ppo_target_kl_min_episodes,
            ppo_target_kl_min_labels,
            ppo_target_kl_hard_multiplier,
            ppo_target_kl_hard_consecutive_updates,
            ppo_shuffle_seed,
            ppo_episode_limit,
            ppo_minibatch_episodes,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            supervised_profile,
            supervised_optimizer,
            (unsigned int)validation_seed,
            aux_checkpoints,
            rl_reward_mode,
            &reward_config,
            expected_policy_tag,
            training_summary_path,
            parent_checkpoint_metadata,
            anchor_checkpoint_path,
            anchor_kl_coef);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-live-rl") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            1,
            1,
            0,
            epochs,
            learning_rate_override,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            gae_lambda,
            ppo_clip_epsilon,
            ppo_value_clip_epsilon,
            ppo_target_kl,
            ppo_target_kl_min_episodes,
            ppo_target_kl_min_labels,
            ppo_target_kl_hard_multiplier,
            ppo_target_kl_hard_consecutive_updates,
            ppo_shuffle_seed,
            ppo_episode_limit,
            ppo_minibatch_episodes,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            supervised_profile,
            supervised_optimizer,
            (unsigned int)validation_seed,
            aux_checkpoints,
            rl_reward_mode,
            &reward_config,
            expected_policy_tag,
            training_summary_path,
            parent_checkpoint_metadata,
            anchor_checkpoint_path,
            anchor_kl_coef);
    }
    if (argc >= 4 && strcmp(argv[1], "--train-live-ppo") == 0) {
        return train_from_input_file(
            argv[2],
            argv[3],
            1,
            1,
            1,
            epochs,
            learning_rate_override,
            rl_gamma,
            rl_entropy_coef,
            rl_advantage_norm,
            gae_lambda,
            ppo_clip_epsilon,
            ppo_value_clip_epsilon,
            ppo_target_kl,
            ppo_target_kl_min_episodes,
            ppo_target_kl_min_labels,
            ppo_target_kl_hard_multiplier,
            ppo_target_kl_hard_consecutive_updates,
            ppo_shuffle_seed,
            ppo_episode_limit,
            ppo_minibatch_episodes,
            adam_beta1,
            adam_beta2,
            adam_epsilon,
            supervised_profile,
            supervised_optimizer,
            (unsigned int)validation_seed,
            aux_checkpoints,
            rl_reward_mode,
            &reward_config,
            expected_policy_tag,
            training_summary_path,
            parent_checkpoint_metadata,
            anchor_checkpoint_path,
            anchor_kl_coef);
    }
    if (argc >= 4 && strcmp(argv[1], "--eval-supervised") == 0) {
        return evaluate_checkpoint_on_replay_file(
            argv[2], argv[3], (unsigned int)validation_seed, &reward_config);
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
        "  showdown_client --train-supervised <replay.jsonl> <checkpoint.bin> [--epochs N] [--learning-rate F] [--supervised-optimizer sgd|adam] [--validation-seed N] [--aux-checkpoints 0|1] [--supervised-profile 0|1]\n"
        "  showdown_client --check-supervised-overfit <replay.jsonl> <report.json> [--epochs N] [--learning-rate F] [--seed N] [--supervised-optimizer sgd|adam]\n"
        "  showdown_client --train-rl <replay.jsonl> <checkpoint.bin> [--epochs N] [--learning-rate F] [--gamma F] [--entropy-coef F] [--advantage-norm 0|1] [--reward-mode terminal|dense_additive]\n"
        "  showdown_client --train-live-rl <episode_batch.jsonl> <checkpoint.bin> [--epochs N] [--learning-rate F] [--gamma F] [--entropy-coef F] [--advantage-norm 0|1] [--reward-mode terminal|dense_additive] [--policy-tag-expected TAG]\n"
        "  showdown_client --train-live-ppo <episode_batch.jsonl> <checkpoint.bin> [--epochs N] [--learning-rate F] [--gamma F] [--entropy-coef F] [--advantage-norm 0|1] [--ppo-minibatch-episodes N] [--target-kl F] [--shuffle-seed N] [--episode-limit N] [--reward-mode terminal|dense_additive] [--policy-tag-expected TAG]\n"
        "  showdown_client --eval-supervised <replay.jsonl> <checkpoint.bin> [--validation-seed N]\n"
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

