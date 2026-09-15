#include "counterfactual_data.h"
#include "validation_split.h"

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

static char* duplicate_text(const char* text) {
    size_t length;
    char* copy;
    if (!text) return NULL;
    length = strlen(text) + 1u;
    copy = (char*)malloc(length);
    if (copy) memcpy(copy, text, length);
    return copy;
}

static int path_tags_equal(const char* left, const char* right) {
    if (!left || !right) return 0;
    while (left[0] == '.' && (left[1] == '/' || left[1] == '\\')) left += 2;
    while (right[0] == '.' && (right[1] == '/' || right[1] == '\\')) right += 2;
    while (*left && *right) {
        unsigned char left_ch = (unsigned char)*left++;
        unsigned char right_ch = (unsigned char)*right++;
        if (left_ch == '\\') left_ch = '/';
        if (right_ch == '\\') right_ch = '/';
#ifdef _WIN32
        left_ch = (unsigned char)tolower(left_ch);
        right_ch = (unsigned char)tolower(right_ch);
#endif
        if (left_ch != right_ch) return 0;
    }
    return *left == '\0' && *right == '\0';
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
    int rollout_count;
    int has_rollout_count;
    int has_rollout_variance;
    int has_rollout_returns;
    memset(sample, 0, sizeof(*sample));
    sample->rollout_count = 1u;
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
    has_rollout_count = find_value(line, "rollout_count") != NULL;
    has_rollout_variance = find_value(line, "rollout_variance") != NULL;
    has_rollout_returns = find_value(line, "rollout_returns") != NULL;
    if (has_rollout_count != has_rollout_variance) return 0;
    if (has_rollout_returns && !has_rollout_count) return 0;
    if (has_rollout_count) {
        if (!read_int(line, "rollout_count", &rollout_count) || rollout_count < 2 ||
                !read_float(line, "rollout_variance", &sample->rollout_variance) ||
                !isfinite(sample->rollout_variance) ||
                sample->rollout_variance < 0.0f) return 0;
        sample->rollout_count = (size_t)rollout_count;
        sample->has_rollout_statistics = 1;
    }
    if (has_rollout_returns) {
        sample->rollout_returns = (float*)malloc(
            sample->rollout_count * sizeof(*sample->rollout_returns));
        if (!sample->rollout_returns || !read_float_array(
                line, "rollout_returns", sample->rollout_returns,
                sample->rollout_count)) {
            free(sample->rollout_returns);
            sample->rollout_returns = NULL;
            return 0;
        }
    }
    sample->hidden_state = (float*)malloc(hidden_dim * sizeof(*sample->hidden_state));
    if (!sample->hidden_state ||
            !read_float_array(line, "hidden_state", sample->hidden_state, hidden_dim)) {
        free(sample->hidden_state);
        free(sample->rollout_returns);
        sample->hidden_state = NULL;
        sample->rollout_returns = NULL;
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
            first->has_rollout_statistics != second->has_rollout_statistics ||
            (first->has_rollout_statistics &&
                first->rollout_count != second->rollout_count) ||
            (first->rollout_returns != NULL) !=
                (second->rollout_returns != NULL) ||
            memcmp(first->hidden_state, second->hidden_state,
                hidden_dim * sizeof(*first->hidden_state)) != 0 ||
            memcmp(first->legal_mask, second->legal_mask, OBS_NUM_ACTIONS) != 0) return 0;
    return first->action != second->action || first->action2 != second->action2 ||
        first->choice.slot0_target_index != second->choice.slot0_target_index ||
        first->choice.slot1_target_index != second->choice.slot1_target_index;
}

static int set_paired_preference_confidence(
    CounterfactualSample* first,
    CounterfactualSample* second
) {
    double mean = 0.0;
    double variance = 0.0;
    double confidence;
    size_t i;
    if (!first || !second) return 0;
    if (!first->rollout_returns || !second->rollout_returns) return 1;
    if (first->rollout_count < 2u ||
            first->rollout_count != second->rollout_count) return 0;
    for (i = 0; i < first->rollout_count; ++i) {
        mean += (double)first->rollout_returns[i] - second->rollout_returns[i];
    }
    mean /= (double)first->rollout_count;
    for (i = 0; i < first->rollout_count; ++i) {
        double difference =
            (double)first->rollout_returns[i] - second->rollout_returns[i];
        double residual = difference - mean;
        variance += residual * residual;
    }
    variance /= (double)(first->rollout_count - 1u);
    if (!isfinite(mean) || !isfinite(variance) || variance < 0.0) return 0;
    if (fabs(mean) <= 1.0e-12) confidence = 0.0;
    else if (variance <= 1.0e-12) confidence = 1.0;
    else {
        double z_score = fabs(mean) /
            sqrt(variance / (double)first->rollout_count);
        confidence = erf(z_score * 0.7071067811865475);
    }
    if (!isfinite(confidence) || confidence < 0.0 || confidence > 1.0) return 0;
    first->paired_preference_confidence = (float)confidence;
    second->paired_preference_confidence = (float)confidence;
    first->has_paired_preference_confidence = 1;
    second->has_paired_preference_confidence = 1;
    return 1;
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
            free(sample.rollout_returns);
            free(sample.hidden_state);
            status = -1;
            break;
        }
        if (dataset->count == dataset->capacity && !grow_dataset(dataset)) {
            free(sample.rollout_returns);
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
                expected_hidden_dim) ||
                !set_paired_preference_confidence(
                    &dataset->samples[index], &dataset->samples[index + 1u])) {
            counterfactual_dataset_free(dataset);
            return 0;
        }
    }
    dataset->source_paths = (char**)malloc(sizeof(*dataset->source_paths));
    if (!dataset->source_paths) {
        counterfactual_dataset_free(dataset);
        return 0;
    }
    dataset->source_paths[0] = duplicate_text(path);
    if (!dataset->source_paths[0]) {
        counterfactual_dataset_free(dataset);
        return 0;
    }
    dataset->source_count = 1u;
    return 1;
}

static char* trim_manifest_entry(char* line) {
    char* start = line;
    char* end;
    while (*start && isspace((unsigned char)*start)) ++start;
    end = start + strlen(start);
    while (end > start && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return start;
}

int counterfactual_dataset_contains_source(
    const CounterfactualDataset* dataset,
    const char* path
) {
    size_t index;
    for (index = 0; index < dataset->source_count; ++index) {
        if (path_tags_equal(dataset->source_paths[index], path)) return 1;
    }
    return 0;
}

static int append_dataset(
    CounterfactualDataset* destination,
    CounterfactualDataset* source
) {
    CounterfactualSample* samples;
    char** source_paths;
    size_t index;
    size_t source_index;
    if (!destination || !source || source->source_count != 1u ||
            source->hidden_dim != destination->hidden_dim ||
            counterfactual_dataset_contains_source(destination, source->source_paths[0])) return 0;
    samples = (CounterfactualSample*)realloc(
        destination->samples,
        (destination->count + source->count) * sizeof(*destination->samples));
    if (!samples) return 0;
    destination->samples = samples;
    source_paths = (char**)realloc(
        destination->source_paths,
        (destination->source_count + 1u) * sizeof(*destination->source_paths));
    if (!source_paths) return 0;
    destination->source_paths = source_paths;

    source_index = destination->source_count;
    for (index = 0; index < source->count; ++index) {
        source->samples[index].source_index = source_index;
        destination->samples[destination->count + index] = source->samples[index];
    }
    destination->source_paths[source_index] = source->source_paths[0];
    source->source_paths[0] = NULL;
    destination->source_count++;
    destination->count += source->count;
    destination->capacity = destination->count;
    destination->pair_count += source->pair_count;

    free(source->source_paths);
    free(source->samples);
    memset(source, 0, sizeof(*source));
    return 1;
}

int counterfactual_dataset_load_manifest(
    CounterfactualDataset* dataset,
    const char* manifest_path,
    size_t expected_hidden_dim,
    const char* expected_policy_tag
) {
    FILE* manifest;
    char* line = NULL;
    size_t line_capacity = 0;
    int status;
    if (!dataset || !manifest_path || !*manifest_path || expected_hidden_dim == 0) {
        return 0;
    }
    memset(dataset, 0, sizeof(*dataset));
    dataset->hidden_dim = expected_hidden_dim;
    manifest = fopen(manifest_path, "r");
    if (!manifest) return 0;
    while ((status = read_line(manifest, &line, &line_capacity)) > 0) {
        CounterfactualDataset batch;
        char* path = trim_manifest_entry(line);
        if (!*path || *path == '#') continue;
        memset(&batch, 0, sizeof(batch));
        if (!counterfactual_dataset_load(
                &batch, path, expected_hidden_dim, expected_policy_tag) ||
                !append_dataset(dataset, &batch)) {
            counterfactual_dataset_free(&batch);
            status = -1;
            break;
        }
    }
    free(line);
    fclose(manifest);
    if (status < 0 || dataset->source_count == 0 || dataset->pair_count == 0) {
        counterfactual_dataset_free(dataset);
        return 0;
    }
    return 1;
}

void counterfactual_dataset_free(CounterfactualDataset* dataset) {
    size_t i;
    if (!dataset) return;
    for (i = 0; i < dataset->count; ++i) {
        free(dataset->samples[i].rollout_returns);
        free(dataset->samples[i].hidden_state);
    }
    for (i = 0; i < dataset->source_count; ++i) free(dataset->source_paths[i]);
    free(dataset->source_paths);
    free(dataset->samples);
    memset(dataset, 0, sizeof(*dataset));
}

static void append_pair(
    CounterfactualSample** destination,
    size_t* count,
    CounterfactualSample* first
) {
    destination[(*count)++] = first;
    destination[(*count)++] = first + 1;
}

static int pair_split_hash(
    const CounterfactualDataset* dataset,
    const CounterfactualSample* sample,
    unsigned int validation_seed,
    uint64_t* hash
) {
    const char* source_path;
    char* key;
    size_t source_length;
    size_t pair_length;
    size_t index;
    if (!dataset || !sample || !hash) return 0;
    if (dataset->source_count <= 1u) {
        *hash = validation_split_hash(sample->pair_id, validation_seed);
        return 1;
    }
    if (sample->source_index >= dataset->source_count) return 0;
    source_path = dataset->source_paths[sample->source_index];
    while (source_path[0] == '.' &&
            (source_path[1] == '/' || source_path[1] == '\\')) source_path += 2;
    source_length = strlen(source_path);
    pair_length = strlen(sample->pair_id);
    key = (char*)malloc(source_length + pair_length + 2u);
    if (!key) return 0;
    for (index = 0; index < source_length; ++index) {
        unsigned char ch = (unsigned char)source_path[index];
        if (ch == '\\') ch = '/';
#ifdef _WIN32
        ch = (unsigned char)tolower(ch);
#endif
        key[index] = (char)ch;
    }
    key[source_length] = '\n';
    memcpy(key + source_length + 1u, sample->pair_id, pair_length + 1u);
    *hash = validation_split_hash(key, validation_seed);
    free(key);
    return 1;
}

int counterfactual_dataset_split(
    CounterfactualDataset* training,
    CounterfactualDataset* external_holdout,
    unsigned int validation_seed,
    CounterfactualDatasetSplit* split
) {
    size_t pair_index;
    int use_external;
    if (!training || !split || training->pair_count == 0) return 0;
    memset(split, 0, sizeof(*split));
    use_external = external_holdout && external_holdout->pair_count > 0;
    split->train = (CounterfactualSample**)malloc(
        training->count * sizeof(*split->train));
    split->selection = (CounterfactualSample**)malloc(
        training->count * sizeof(*split->selection));
    split->holdout = (CounterfactualSample**)malloc(
        (use_external ? external_holdout->count : training->count) *
        sizeof(*split->holdout));
    if (!split->train || !split->selection || !split->holdout) {
        counterfactual_dataset_split_free(split);
        return 0;
    }

    for (pair_index = 0; pair_index < training->pair_count; ++pair_index) {
        CounterfactualSample* first = &training->samples[pair_index * 2u];
        uint64_t hash;
        uint64_t bucket;
        if (!pair_split_hash(training, first, validation_seed, &hash)) {
            counterfactual_dataset_split_free(split);
            return 0;
        }
        bucket = hash % UINT64_C(10);
        if (use_external) {
            if (bucket == UINT64_C(0)) {
                append_pair(split->selection, &split->selection_count, first);
            } else {
                append_pair(split->train, &split->train_count, first);
            }
        } else if (bucket == UINT64_C(0)) {
            append_pair(split->holdout, &split->holdout_count, first);
        } else if (bucket == UINT64_C(1)) {
            append_pair(split->selection, &split->selection_count, first);
        } else {
            append_pair(split->train, &split->train_count, first);
        }
    }
    if (use_external) {
        for (pair_index = 0; pair_index < external_holdout->pair_count; ++pair_index) {
            append_pair(
                split->holdout, &split->holdout_count,
                &external_holdout->samples[pair_index * 2u]);
        }
        split->external_holdout = 1;
    }
    if (split->train_count == 0 || split->selection_count == 0 ||
            split->holdout_count == 0) {
        counterfactual_dataset_split_free(split);
        return 0;
    }
    return 1;
}

int counterfactual_dataset_overfit_subset(
    CounterfactualDataset* dataset,
    size_t requested_pair_count,
    CounterfactualDatasetSplit* split
) {
    size_t sample_count;
    size_t sample_index;
    if (!dataset || !split || requested_pair_count == 0 ||
            dataset->pair_count == 0) return 0;
    memset(split, 0, sizeof(*split));
    if (requested_pair_count > dataset->pair_count) {
        requested_pair_count = dataset->pair_count;
    }
    sample_count = requested_pair_count * 2u;
    split->train = (CounterfactualSample**)malloc(
        sample_count * sizeof(*split->train));
    split->selection = (CounterfactualSample**)malloc(
        sample_count * sizeof(*split->selection));
    split->holdout = (CounterfactualSample**)malloc(
        sample_count * sizeof(*split->holdout));
    if (!split->train || !split->selection || !split->holdout) {
        counterfactual_dataset_split_free(split);
        return 0;
    }
    for (sample_index = 0; sample_index < sample_count; ++sample_index) {
        CounterfactualSample* sample = &dataset->samples[sample_index];
        split->train[sample_index] = sample;
        split->selection[sample_index] = sample;
        split->holdout[sample_index] = sample;
    }
    split->train_count = sample_count;
    split->selection_count = sample_count;
    split->holdout_count = sample_count;
    return 1;
}

void counterfactual_dataset_split_free(CounterfactualDatasetSplit* split) {
    if (!split) return;
    free(split->holdout);
    free(split->selection);
    free(split->train);
    memset(split, 0, sizeof(*split));
}

static int filter_pair_array(
    CounterfactualSample** samples,
    size_t* sample_count,
    float minimum_confidence
) {
    size_t read_index;
    size_t write_count = 0;
    if (!samples || !sample_count || *sample_count % 2u != 0) return 0;
    for (read_index = 0; read_index < *sample_count; read_index += 2u) {
        CounterfactualSample* first = samples[read_index];
        CounterfactualSample* second = samples[read_index + 1u];
        if (!first || !second || strcmp(first->pair_id, second->pair_id) != 0) {
            return 0;
        }
        if (counterfactual_pair_preference_weight(first, second) + 1.0e-7f <
                minimum_confidence) continue;
        samples[write_count++] = first;
        samples[write_count++] = second;
    }
    *sample_count = write_count;
    return 1;
}

int counterfactual_dataset_split_filter_confidence(
    CounterfactualDatasetSplit* split,
    float minimum_confidence
) {
    if (!split || !isfinite(minimum_confidence) || minimum_confidence < 0.0f ||
            minimum_confidence > 1.0f) return 0;
    if (minimum_confidence <= 0.0f) return 1;
    return filter_pair_array(split->train, &split->train_count, minimum_confidence) &&
        filter_pair_array(
            split->selection, &split->selection_count, minimum_confidence) &&
        filter_pair_array(split->holdout, &split->holdout_count, minimum_confidence) &&
        split->train_count > 0u && split->selection_count > 0u &&
        split->holdout_count > 0u;
}

float counterfactual_pair_preference_weight(
    const CounterfactualSample* first,
    const CounterfactualSample* second
) {
    float target_gap;
    float variance;
    float z_score;
    if (!first || !second) return 0.0f;
    target_gap = fabsf(first->target_value - second->target_value);
    if (!isfinite(target_gap) || target_gap <= 1.0e-6f) return 0.0f;
    if (first->has_paired_preference_confidence &&
            second->has_paired_preference_confidence) {
        return first->paired_preference_confidence;
    }
    if (!first->has_rollout_statistics || !second->has_rollout_statistics ||
            first->rollout_count < 2u || second->rollout_count < 2u) return 1.0f;
    variance = first->rollout_variance / (float)first->rollout_count +
        second->rollout_variance / (float)second->rollout_count;
    if (!isfinite(variance)) return 0.0f;
    if (variance <= 1.0e-12f) return 1.0f;
    z_score = target_gap / sqrtf(variance);
    return erff(z_score * 0.7071067811865475f);
}
