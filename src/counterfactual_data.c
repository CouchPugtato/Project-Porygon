#include "counterfactual_data.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* skip_space(const char* text) {
    while (text && *text && isspace((unsigned char)*text)) ++text;
    return text;
}

static size_t normalize_path_tag(const char* source, char* output, size_t capacity) {
    size_t length = 0;
    if (!source || !output || capacity == 0) return 0;
    while (source[0] == '.' && (source[1] == '/' || source[1] == '\\')) source += 2;
    while (*source && length + 1u < capacity) {
        unsigned char ch = (unsigned char)*source++;
        if (ch == '\\') ch = '/';
#ifdef _WIN32
        ch = (unsigned char)tolower(ch);
#endif
        output[length++] = (char)ch;
    }
    output[length] = '\0';
    return length;
}

static int path_tags_match(const char* left, const char* right) {
    char left_normalized[COUNTERFACTUAL_POLICY_TAG_LEN];
    char right_normalized[COUNTERFACTUAL_POLICY_TAG_LEN];
    size_t left_length = normalize_path_tag(
        left, left_normalized, sizeof(left_normalized));
    size_t right_length = normalize_path_tag(
        right, right_normalized, sizeof(right_normalized));
    const char* longer;
    const char* shorter;
    size_t longer_length;
    size_t shorter_length;
    if (left_length == 0 || right_length == 0) return 0;
    if (strcmp(left_normalized, right_normalized) == 0) return 1;
    if (left_length > right_length) {
        longer = left_normalized;
        longer_length = left_length;
        shorter = right_normalized;
        shorter_length = right_length;
    } else {
        longer = right_normalized;
        longer_length = right_length;
        shorter = left_normalized;
        shorter_length = left_length;
    }
    return longer_length > shorter_length &&
        longer[longer_length - shorter_length - 1u] == '/' &&
        strcmp(longer + longer_length - shorter_length, shorter) == 0;
}

static const char* find_value(const char* json, const char* key) {
    char pattern[96];
    const char* found;
    if (!json || !key || strlen(key) + 4u >= sizeof(pattern)) return NULL;
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    found = strstr(json, pattern);
    if (!found) return NULL;
    found = strchr(found + strlen(pattern), ':');
    return found ? skip_space(found + 1) : NULL;
}

static int read_string(
    const char* json,
    const char* key,
    char* output,
    size_t output_size
) {
    const char* value = find_value(json, key);
    size_t length = 0;
    if (!value || *value != '"' || !output || output_size == 0) return 0;
    ++value;
    while (*value && *value != '"') {
        if (*value == '\\' && value[1]) ++value;
        if (length + 1u >= output_size) return 0;
        output[length++] = *value++;
    }
    output[length] = '\0';
    return *value == '"';
}

static int read_int(const char* json, const char* key, int* output) {
    const char* value = find_value(json, key);
    char* end = NULL;
    long parsed;
    if (!value || !output) return 0;
    parsed = strtol(value, &end, 10);
    if (end == value) return 0;
    *output = (int)parsed;
    return 1;
}

static int read_float(const char* json, const char* key, float* output) {
    const char* value = find_value(json, key);
    char* end = NULL;
    float parsed;
    if (!value || !output) return 0;
    parsed = strtof(value, &end);
    if (end == value || !isfinite(parsed)) return 0;
    *output = parsed;
    return 1;
}

static int read_float_array(
    const char* json,
    const char* key,
    float* output,
    size_t expected_count
) {
    const char* value = find_value(json, key);
    size_t count = 0;
    if (!value || *value != '[' || !output) return 0;
    value = skip_space(value + 1);
    while (*value && *value != ']') {
        char* end = NULL;
        float parsed;
        if (count >= expected_count) return 0;
        parsed = strtof(value, &end);
        if (end == value || !isfinite(parsed)) return 0;
        output[count++] = parsed;
        value = skip_space(end);
        if (*value == ',') value = skip_space(value + 1);
        else if (*value != ']') return 0;
    }
    return *value == ']' && count == expected_count;
}

static int read_mask(
    const char* json,
    const char* key,
    unsigned char* output,
    size_t expected_count
) {
    const char* value = find_value(json, key);
    size_t count = 0;
    if (!value || *value != '[' || !output) return 0;
    value = skip_space(value + 1);
    while (*value && *value != ']') {
        char* end = NULL;
        long parsed;
        if (count >= expected_count) return 0;
        parsed = strtol(value, &end, 10);
        if (end == value || (parsed != 0 && parsed != 1)) return 0;
        output[count++] = (unsigned char)parsed;
        value = skip_space(end);
        if (*value == ',') value = skip_space(value + 1);
        else if (*value != ']') return 0;
    }
    return *value == ']' && count == expected_count;
}

static int read_choice(const char* json, CounterfactualSample* sample) {
    int value;
#define READ_CHOICE_INT(key, field) \
    do { \
        if (!read_int(json, key, &value) || value < 0 || value > 255) return 0; \
        sample->choice.field = (unsigned char)value; \
    } while (0)
    READ_CHOICE_INT("slot0_has_action", slot0_has_action);
    READ_CHOICE_INT("slot0_kind", slot0_kind);
    READ_CHOICE_INT("slot0_move_index", slot0_move_index);
    READ_CHOICE_INT("slot0_switch_index", slot0_switch_index);
    READ_CHOICE_INT("slot0_use_tera", slot0_use_tera);
    READ_CHOICE_INT("slot0_target_index", slot0_target_index);
    READ_CHOICE_INT("slot0_target_mask", slot0_target_mask);
    READ_CHOICE_INT("slot1_has_action", slot1_has_action);
    READ_CHOICE_INT("slot1_kind", slot1_kind);
    READ_CHOICE_INT("slot1_move_index", slot1_move_index);
    READ_CHOICE_INT("slot1_switch_index", slot1_switch_index);
    READ_CHOICE_INT("slot1_use_tera", slot1_use_tera);
    READ_CHOICE_INT("slot1_target_index", slot1_target_index);
    READ_CHOICE_INT("slot1_target_mask", slot1_target_mask);
#undef READ_CHOICE_INT
    return 1;
}

static int parse_sample(
    const char* line,
    size_t hidden_dim,
    CounterfactualSample* sample
) {
    char type[48];
    int encoded_hidden_dim;
    memset(sample, 0, sizeof(*sample));
    factorized_action_choice_init(&sample->choice);
    if (!read_string(line, "type", type, sizeof(type)) ||
            strcmp(type, "counterfactual_sample") != 0 ||
            !read_string(line, "pair_id", sample->pair_id, sizeof(sample->pair_id)) ||
            !read_string(line, "policy_tag", sample->policy_tag, sizeof(sample->policy_tag)) ||
            !read_int(line, "action_rank", &sample->action_rank) ||
            !read_int(line, "decision_index", &sample->decision_index) ||
            !read_int(line, "action", &sample->action) ||
            !read_int(line, "action2", &sample->action2) ||
            !read_int(line, "hidden_dim", &encoded_hidden_dim) ||
            encoded_hidden_dim < 1 || (size_t)encoded_hidden_dim != hidden_dim ||
            !read_float(line, "baseline_value", &sample->baseline_value) ||
            !read_float(line, "target_value", &sample->target_value) ||
            !read_choice(line, sample) ||
            !read_mask(line, "legal_mask", sample->legal_mask, OBS_NUM_ACTIONS)) return 0;
    sample->hidden_state = (float*)malloc(hidden_dim * sizeof(*sample->hidden_state));
    if (!sample->hidden_state ||
            !read_float_array(line, "hidden_state", sample->hidden_state, hidden_dim)) {
        free(sample->hidden_state);
        sample->hidden_state = NULL;
        return 0;
    }
    return sample->pair_id[0] != '\0' && sample->policy_tag[0] != '\0' &&
        sample->decision_index >= 0 && sample->action_rank >= 0 &&
        sample->action_rank <= 1 && sample->target_value >= -1.0f &&
        sample->target_value <= 1.0f &&
        sample->action >= -1 && sample->action < OBS_NUM_ACTIONS &&
        sample->action2 >= -1 && sample->action2 < OBS_NUM_ACTIONS &&
        (sample->action >= 0 || sample->action2 >= 0) &&
        (sample->action < 0 || sample->legal_mask[sample->action]) &&
        (sample->action2 < 0 || sample->legal_mask[sample->action2]);
}

static int grow_dataset(CounterfactualDataset* dataset) {
    size_t capacity = dataset->capacity ? dataset->capacity * 2u : 128u;
    CounterfactualSample* samples = (CounterfactualSample*)realloc(
        dataset->samples, capacity * sizeof(*samples));
    if (!samples) return 0;
    dataset->samples = samples;
    dataset->capacity = capacity;
    return 1;
}

static int read_line(FILE* file, char** buffer, size_t* capacity) {
    size_t length = 0;
    int ch;
    if (!*buffer) {
        *capacity = 4096u;
        *buffer = (char*)malloc(*capacity);
        if (!*buffer) return -1;
    }
    while ((ch = fgetc(file)) != EOF) {
        if (length + 2u >= *capacity) {
            char* resized;
            *capacity *= 2u;
            resized = (char*)realloc(*buffer, *capacity);
            if (!resized) return -1;
            *buffer = resized;
        }
        (*buffer)[length++] = (char)ch;
        if (ch == '\n') break;
    }
    if (length == 0 && ch == EOF) return 0;
    (*buffer)[length] = '\0';
    return 1;
}

static int samples_form_pair(
    const CounterfactualSample* first,
    const CounterfactualSample* second,
    size_t hidden_dim
) {
    if (strcmp(first->pair_id, second->pair_id) != 0 ||
            !path_tags_match(first->policy_tag, second->policy_tag) ||
            first->action_rank != 0 || second->action_rank != 1 ||
            first->decision_index != second->decision_index ||
            memcmp(first->hidden_state, second->hidden_state,
                hidden_dim * sizeof(*first->hidden_state)) != 0 ||
            memcmp(first->legal_mask, second->legal_mask, OBS_NUM_ACTIONS) != 0) return 0;
    return first->action != second->action || first->action2 != second->action2 ||
        first->choice.slot0_target_index != second->choice.slot0_target_index ||
        first->choice.slot1_target_index != second->choice.slot1_target_index;
}

int counterfactual_dataset_load(
    CounterfactualDataset* dataset,
    const char* path,
    size_t expected_hidden_dim,
    const char* expected_policy_tag
) {
    FILE* file;
    char* line = NULL;
    size_t line_capacity = 0;
    int status;
    if (!dataset || !path || !*path || expected_hidden_dim == 0) return 0;
    memset(dataset, 0, sizeof(*dataset));
    dataset->hidden_dim = expected_hidden_dim;
    file = fopen(path, "r");
    if (!file) return 0;
    while ((status = read_line(file, &line, &line_capacity)) > 0) {
        CounterfactualSample sample;
        if (!parse_sample(line, expected_hidden_dim, &sample) ||
                (expected_policy_tag && *expected_policy_tag &&
                 !path_tags_match(sample.policy_tag, expected_policy_tag))) {
            free(sample.hidden_state);
            status = -1;
            break;
        }
        if (dataset->count == dataset->capacity && !grow_dataset(dataset)) {
            free(sample.hidden_state);
            status = -1;
            break;
        }
        dataset->samples[dataset->count++] = sample;
    }
    free(line);
    fclose(file);
    if (status < 0 || dataset->count == 0 || dataset->count % 2u != 0) {
        counterfactual_dataset_free(dataset);
        return 0;
    }
    for (dataset->pair_count = 0; dataset->pair_count * 2u < dataset->count;
            ++dataset->pair_count) {
        size_t index = dataset->pair_count * 2u;
        if (!samples_form_pair(
                &dataset->samples[index], &dataset->samples[index + 1u],
                expected_hidden_dim)) {
            counterfactual_dataset_free(dataset);
            return 0;
        }
    }
    return 1;
}

void counterfactual_dataset_free(CounterfactualDataset* dataset) {
    size_t i;
    if (!dataset) return;
    for (i = 0; i < dataset->count; ++i) free(dataset->samples[i].hidden_state);
    free(dataset->samples);
    memset(dataset, 0, sizeof(*dataset));
}
