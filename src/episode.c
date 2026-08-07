#include "episode.h"
#include "observation.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* skip_ws(const char* p) {
    while (p && *p && isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static const char* find_after_key(const char* json, const char* key) {
    char pattern[64];
    size_t n;
    if (!json || !key) {
        return NULL;
    }
    n = strlen(key);
    if (n + 4 >= sizeof(pattern)) {
        return NULL;
    }
    pattern[0] = '"';
    memcpy(pattern + 1, key, n);
    pattern[n + 1] = '"';
    pattern[n + 2] = ':';
    pattern[n + 3] = '\0';
    return strstr(json, pattern);
}

static int extract_json_string_value(const char* json, const char* key, char* out, size_t out_len) {
    const char* p = find_after_key(json, key);
    size_t i = 0;
    if (!p || !out || out_len == 0) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    p = skip_ws(p + 1);
    if (*p != '"') {
        return 0;
    }
    ++p;
    while (*p && *p != '"' && i + 1 < out_len) {
        if (*p == '\\' && p[1]) {
            ++p;
        }
        out[i++] = *p++;
    }
    out[i] = '\0';
    return *p == '"';
}

static int extract_json_size_t_value(const char* json, const char* key, size_t* out) {
    const char* p = find_after_key(json, key);
    char* endptr = NULL;
    unsigned long long value;
    if (!p || !out) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    p = skip_ws(p + 1);
    value = strtoull(p, &endptr, 10);
    if (endptr == p) {
        return 0;
    }
    *out = (size_t)value;
    return 1;
}

static int extract_json_array_bounds(const char* json, const char* key, const char** start_out, const char** end_out) {
    const char* p = find_after_key(json, key);
    const char* start;
    const char* end;
    int depth = 0;
    if (!p || !start_out || !end_out) {
        return 0;
    }
    p = strchr(p, ':');
    if (!p) {
        return 0;
    }
    start = skip_ws(p + 1);
    if (*start != '[') {
        return 0;
    }
    end = start;
    while (*end) {
        if (*end == '[') {
            ++depth;
        } else if (*end == ']') {
            --depth;
            if (depth == 0) {
                *start_out = start + 1;
                *end_out = end;
                return 1;
            }
        }
        ++end;
    }
    return 0;
}

static int parse_float_array(const char* start, const char* end, float* out, size_t expected_count) {
    size_t count = 0;
    const char* p = start;
    while (p && p < end) {
        char* endptr = NULL;
        p = skip_ws(p);
        if (p >= end) {
            break;
        }
        out[count++] = strtof(p, &endptr);
        if (endptr == p || count > expected_count) {
            return 0;
        }
        p = endptr;
        while (p < end && (*p == ',' || isspace((unsigned char)*p))) {
            ++p;
        }
    }
    return count == expected_count;
}

static int parse_int_array(const char* start, const char* end, int* out, size_t expected_count) {
    size_t count = 0;
    const char* p = start;
    while (p && p < end) {
        char* endptr = NULL;
        long value;
        p = skip_ws(p);
        if (p >= end) {
            break;
        }
        value = strtol(p, &endptr, 10);
        if (endptr == p || count > expected_count) {
            return 0;
        }
        out[count++] = (int)value;
        p = endptr;
        while (p < end && (*p == ',' || isspace((unsigned char)*p))) {
            ++p;
        }
    }
    return count == expected_count;
}

static int parse_u8_array(const char* start, const char* end, uint8_t* out, size_t expected_count) {
    size_t count = 0;
    const char* p = start;
    while (p && p < end) {
        char* endptr = NULL;
        long value;
        p = skip_ws(p);
        if (p >= end) {
            break;
        }
        value = strtol(p, &endptr, 10);
        if (endptr == p || count > expected_count || value < 0 || value > 255) {
            return 0;
        }
        out[count++] = (uint8_t)value;
        p = endptr;
        while (p < end && (*p == ',' || isspace((unsigned char)*p))) {
            ++p;
        }
    }
    return count == expected_count;
}

static void write_json_escaped(FILE* out, const char* text) {
    const unsigned char* p = (const unsigned char*)(text ? text : "");
    while (*p) {
        if (*p == '\\' || *p == '"') {
            fputc('\\', out);
            fputc((int)*p, out);
        } else if (*p == '\n') {
            fputs("\\n", out);
        } else if (*p == '\r') {
            fputs("\\r", out);
        } else if (*p == '\t') {
            fputs("\\t", out);
        } else {
            fputc((int)*p, out);
        }
        ++p;
    }
}

static int episode_grow(Episode* episode, size_t min_capacity) {
    size_t next_capacity;
    float* new_observations;
    uint8_t* new_legal_masks;
    int* new_actions;
    int* new_actions2;
    float* new_rewards;
    uint8_t* new_dones;

    if (!episode || episode->obs_dim == 0) {
        return 0;
    }

    next_capacity = episode->capacity ? episode->capacity * 2 : 4;
    if (next_capacity < min_capacity) {
        next_capacity = min_capacity;
    }

    new_observations = (float*)malloc(next_capacity * episode->obs_dim * sizeof(float));
    new_legal_masks = (uint8_t*)malloc(next_capacity * OBS_NUM_ACTIONS * sizeof(uint8_t));
    new_actions = (int*)malloc(next_capacity * sizeof(int));
    new_actions2 = (int*)malloc(next_capacity * sizeof(int));
    new_rewards = (float*)malloc(next_capacity * sizeof(float));
    new_dones = (uint8_t*)malloc(next_capacity * sizeof(uint8_t));

    if (!new_observations || !new_legal_masks || !new_actions || !new_actions2 || !new_rewards || !new_dones) {
        free(new_observations);
        free(new_legal_masks);
        free(new_actions);
        free(new_actions2);
        free(new_rewards);
        free(new_dones);
        return 0;
    }

    if (episode->count > 0) {
        memcpy(new_observations, episode->observations, episode->count * episode->obs_dim * sizeof(float));
        memcpy(new_legal_masks, episode->legal_masks, episode->count * OBS_NUM_ACTIONS * sizeof(uint8_t));
        memcpy(new_actions, episode->actions, episode->count * sizeof(int));
        memcpy(new_actions2, episode->actions2, episode->count * sizeof(int));
        memcpy(new_rewards, episode->rewards, episode->count * sizeof(float));
        memcpy(new_dones, episode->dones, episode->count * sizeof(uint8_t));
    }

    free(episode->observations);
    free(episode->legal_masks);
    free(episode->actions);
    free(episode->actions2);
    free(episode->rewards);
    free(episode->dones);

    episode->observations = new_observations;
    episode->legal_masks = new_legal_masks;
    episode->actions = new_actions;
    episode->actions2 = new_actions2;
    episode->rewards = new_rewards;
    episode->dones = new_dones;
    episode->capacity = next_capacity;
    return 1;
}

int episode_init(Episode* episode, size_t capacity, size_t obs_dim) {
    if (!episode || obs_dim == 0) {
        return 0;
    }
    memset(episode, 0, sizeof(*episode));
    episode->obs_dim = obs_dim;
    return episode_grow(episode, capacity ? capacity : 4);
}

void episode_free(Episode* episode) {
    if (!episode) {
        return;
    }
    free(episode->observations);
    free(episode->legal_masks);
    free(episode->actions);
    free(episode->actions2);
    free(episode->rewards);
    free(episode->dones);
    memset(episode, 0, sizeof(*episode));
}

int episode_append(
    Episode* episode,
    const float* observation,
    const uint8_t* legal_mask,
    int action,
    float reward,
    uint8_t done
) {
    float* dst;
    uint8_t* legal_dst;

    if (!episode || !observation) {
        return 0;
    }
    if (episode->count == episode->capacity && !episode_grow(episode, episode->count + 1)) {
        return 0;
    }

    dst = episode->observations + (episode->count * episode->obs_dim);
    memcpy(dst, observation, episode->obs_dim * sizeof(float));
    legal_dst = episode->legal_masks + (episode->count * OBS_NUM_ACTIONS);
    if (legal_mask) {
        memcpy(legal_dst, legal_mask, OBS_NUM_ACTIONS * sizeof(uint8_t));
    } else {
        memset(legal_dst, 0, OBS_NUM_ACTIONS * sizeof(uint8_t));
    }
    episode->actions[episode->count] = action;
    episode->actions2[episode->count] = -1;
    episode->rewards[episode->count] = reward;
    episode->dones[episode->count] = done;
    episode->count += 1;
    return 1;
}

const float* episode_observation_at(const Episode* episode, size_t index) {
    if (!episode || index >= episode->count) {
        return NULL;
    }
    return episode->observations + (index * episode->obs_dim);
}

int episode_write_json_record(FILE* out, const Episode* episode, const char* battle_id, const char* policy_tag) {
    size_t i;
    size_t obs_count;
    size_t legal_count;
    if (!out || !episode) {
        return 0;
    }
    obs_count = episode->count * episode->obs_dim;
    legal_count = episode->count * OBS_NUM_ACTIONS;
    fputs("{\"type\":\"episode_complete\",\"battle_id\":\"", out);
    write_json_escaped(out, battle_id ? battle_id : "");
    fputs("\",\"policy_tag\":\"", out);
    write_json_escaped(out, policy_tag ? policy_tag : "");
    fprintf(out, "\",\"obs_dim\":%zu,\"count\":%zu,\"observations\":[", episode->obs_dim, episode->count);
    for (i = 0; i < obs_count; ++i) {
        if (i > 0) fputc(',', out);
        fprintf(out, "%.9g", episode->observations[i]);
    }
    fputs("],\"legal_masks\":[", out);
    for (i = 0; i < legal_count; ++i) {
        if (i > 0) fputc(',', out);
        fprintf(out, "%u", (unsigned int)episode->legal_masks[i]);
    }
    fputs("],\"actions\":[", out);
    for (i = 0; i < episode->count; ++i) {
        if (i > 0) fputc(',', out);
        fprintf(out, "%d", episode->actions[i]);
    }
    fputs("],\"actions2\":[", out);
    for (i = 0; i < episode->count; ++i) {
        if (i > 0) fputc(',', out);
        fprintf(out, "%d", episode->actions2[i]);
    }
    fputs("],\"rewards\":[", out);
    for (i = 0; i < episode->count; ++i) {
        if (i > 0) fputc(',', out);
        fprintf(out, "%.9g", episode->rewards[i]);
    }
    fputs("],\"dones\":[", out);
    for (i = 0; i < episode->count; ++i) {
        if (i > 0) fputc(',', out);
        fprintf(out, "%u", (unsigned int)episode->dones[i]);
    }
    fputs("]}\n", out);
    return 1;
}

int episode_parse_json_record(
    const char* json,
    Episode* episode,
    char* battle_id,
    size_t battle_id_len,
    char* policy_tag,
    size_t policy_tag_len
) {
    char type[32];
    const char* start;
    const char* end;
    size_t count = 0;
    size_t obs_dim = 0;
    if (!json || !episode) {
        return 0;
    }
    if (!extract_json_string_value(json, "type", type, sizeof(type)) || strcmp(type, "episode_complete") != 0) {
        return 0;
    }
    if (battle_id && battle_id_len > 0) {
        extract_json_string_value(json, "battle_id", battle_id, battle_id_len);
    }
    if (policy_tag && policy_tag_len > 0) {
        extract_json_string_value(json, "policy_tag", policy_tag, policy_tag_len);
    }
    if (!extract_json_size_t_value(json, "count", &count) || !extract_json_size_t_value(json, "obs_dim", &obs_dim)) {
        return 0;
    }
    if (!episode_init(episode, count, obs_dim)) {
        return 0;
    }
    episode->count = count;
    if (!extract_json_array_bounds(json, "observations", &start, &end) ||
            !parse_float_array(start, end, episode->observations, count * obs_dim) ||
            !extract_json_array_bounds(json, "legal_masks", &start, &end) ||
            !parse_u8_array(start, end, episode->legal_masks, count * OBS_NUM_ACTIONS) ||
            !extract_json_array_bounds(json, "actions", &start, &end) ||
            !parse_int_array(start, end, episode->actions, count) ||
            !extract_json_array_bounds(json, "actions2", &start, &end) ||
            !parse_int_array(start, end, episode->actions2, count) ||
            !extract_json_array_bounds(json, "rewards", &start, &end) ||
            !parse_float_array(start, end, episode->rewards, count) ||
            !extract_json_array_bounds(json, "dones", &start, &end) ||
            !parse_u8_array(start, end, episode->dones, count)) {
        episode_free(episode);
        return 0;
    }
    return 1;
}
