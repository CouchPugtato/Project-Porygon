#include "id_tables.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char** tokens;
    int count;
    int capacity;
} IdTable;

static IdTable g_species_table = {0};
static IdTable g_move_table = {0};
static IdTable g_item_table = {0};
static IdTable g_ability_table = {0};
static IdTable g_condition_table = {0};

static void normalize_token(const char* in, char* out, size_t out_len) {
    size_t i = 0;
    if (!in || out_len == 0) {
        return;
    }
    while (*in && i + 1 < out_len) {
        unsigned char ch = (unsigned char)*in++;
        if (isalnum(ch)) {
            out[i++] = (char)tolower(ch);
        }
    }
    out[i] = '\0';
}

static char* dup_string(const char* text) {
    size_t len;
    char* copy;
    if (!text) {
        return NULL;
    }
    len = strlen(text);
    copy = (char*)malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, text, len + 1);
    return copy;
}

static int table_append(IdTable* table, const char* token) {
    char* copy;
    char** resized;
    int new_capacity;
    if (!table || !token || !*token) {
        return 0;
    }
    if (table->count == table->capacity) {
        new_capacity = table->capacity ? table->capacity * 2 : 32;
        resized = (char**)realloc(table->tokens, (size_t)new_capacity * sizeof(char*));
        if (!resized) {
            return 0;
        }
        table->tokens = resized;
        table->capacity = new_capacity;
    }
    copy = dup_string(token);
    if (!copy) {
        return 0;
    }
    table->tokens[table->count++] = copy;
    return 1;
}

static int load_table_file(IdTable* table, const char* path) {
    FILE* f;
    char line[256];
    if (!table || !path) {
        return 0;
    }
    f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        char normalized[128];
        char* hash = strchr(line, '#');
        char* start = line;
        char* csv = NULL;
        size_t len;
        if (hash) {
            *hash = '\0';
        }
        while (*start && isspace((unsigned char)*start)) {
            ++start;
        }
        csv = strchr(start, ',');
        if (csv) {
            char* second = csv + 1;
            char* second_end = strchr(second, ',');
            start = second;
            if (second_end) {
                *second_end = '\0';
            }
        }
        len = strlen(start);
        while (len > 0 && isspace((unsigned char)start[len - 1])) {
            start[--len] = '\0';
        }
        normalize_token(start, normalized, sizeof(normalized));
        if (!normalized[0] || strcmp(normalized, "identifier") == 0) {
            continue;
        }
        if (!table_append(table, normalized)) {
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return 1;
}

static int lookup_id(const IdTable* table, const char* name) {
    char normalized[128];
    int i;
    normalize_token(name, normalized, sizeof(normalized));
    if (!normalized[0] || !table) {
        return 0;
    }
    for (i = 0; i < table->count; ++i) {
        if (strcmp(table->tokens[i], normalized) == 0) {
            return i + 1;
        }
    }
    return 0;
}

static const char* lookup_name(const IdTable* table, int id) {
    if (!table || id <= 0 || id > table->count) {
        return "";
    }
    return table->tokens[id - 1];
}

int id_tables_init(void) {
    if (g_species_table.count > 0) {
        return 1;
    }
    if (!(
        load_table_file(&g_species_table, "data/species_ids.txt") &&
        load_table_file(&g_move_table, "data/move_ids.txt") &&
        load_table_file(&g_item_table, "data/item_ids.txt") &&
        load_table_file(&g_ability_table, "data/ability_ids.txt"))) {
        return 0;
    }
    /* Optional until all environments regenerate vocab files. */
    load_table_file(&g_condition_table, "data/conditions_ids.txt");
    return 1;
}

int species_id_from_name(const char* name) {
    return lookup_id(&g_species_table, name);
}

int move_id_from_name(const char* name) {
    return lookup_id(&g_move_table, name);
}

int item_id_from_name(const char* name) {
    return lookup_id(&g_item_table, name);
}

int ability_id_from_name(const char* name) {
    return lookup_id(&g_ability_table, name);
}

int condition_id_from_name(const char* name) {
    return lookup_id(&g_condition_table, name);
}

const char* species_name_from_id(int id) {
    return lookup_name(&g_species_table, id);
}

const char* move_name_from_id(int id) {
    return lookup_name(&g_move_table, id);
}

const char* item_name_from_id(int id) {
    return lookup_name(&g_item_table, id);
}

const char* ability_name_from_id(int id) {
    return lookup_name(&g_ability_table, id);
}

const char* condition_name_from_id(int id) {
    return lookup_name(&g_condition_table, id);
}
