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
static IdTable g_type_table = {0};
static int* g_species_type1 = NULL;
static int* g_species_type2 = NULL;
static int* g_move_type = NULL;
static MoveSemanticMetadata* g_move_semantics = NULL;
static int g_species_type_capacity = 0;
static int g_species_type_entries = 0;
static int g_move_type_capacity = 0;
static int g_move_type_entries = 0;
static int g_move_semantic_capacity = 0;
static int g_move_semantic_entries = 0;

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

static int lookup_id(const IdTable* table, const char* name);

static int ensure_species_type_capacity(int species_id) {
    int new_capacity;
    int* resized_type1;
    int* resized_type2;
    int i;
    if (species_id <= g_species_type_capacity) {
        return 1;
    }
    new_capacity = g_species_type_capacity ? g_species_type_capacity : 32;
    while (new_capacity < species_id) {
        new_capacity *= 2;
    }
    resized_type1 = (int*)realloc(g_species_type1, (size_t)new_capacity * sizeof(int));
    resized_type2 = (int*)realloc(g_species_type2, (size_t)new_capacity * sizeof(int));
    if (!resized_type1 || !resized_type2) {
        free(resized_type1);
        free(resized_type2);
        return 0;
    }
    g_species_type1 = resized_type1;
    g_species_type2 = resized_type2;
    for (i = g_species_type_capacity; i < new_capacity; ++i) {
        g_species_type1[i] = 0;
        g_species_type2[i] = 0;
    }
    g_species_type_capacity = new_capacity;
    return 1;
}

static int load_species_types_file(const char* path) {
    FILE* f;
    char line[256];
    if (!path) {
        return 0;
    }
    f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        char species[128];
        char type1[64];
        char type2[64];
        char* p = line;
        char* comma1;
        char* comma2;
        int species_id;
        int type1_id;
        int type2_id;
        while (*p && isspace((unsigned char)*p)) {
            ++p;
        }
        if (!*p || *p == '#') {
            continue;
        }
        strncpy(species, p, sizeof(species) - 1);
        species[sizeof(species) - 1] = '\0';
        comma1 = strchr(species, ',');
        if (!comma1) {
            continue;
        }
        *comma1++ = '\0';
        comma2 = strchr(comma1, ',');
        if (!comma2) {
            continue;
        }
        *comma2++ = '\0';
        strncpy(type1, comma1, sizeof(type1) - 1);
        type1[sizeof(type1) - 1] = '\0';
        strncpy(type2, comma2, sizeof(type2) - 1);
        type2[sizeof(type2) - 1] = '\0';
        {
            size_t len;
            len = strlen(type2);
            while (len > 0 && isspace((unsigned char)type2[len - 1])) {
                type2[--len] = '\0';
            }
        }
        species_id = species_id_from_name(species);
        type1_id = lookup_id(&g_type_table, type1);
        type2_id = lookup_id(&g_type_table, type2);
        if (species_id <= 0 || type1_id <= 0) {
            continue;
        }
        if (!ensure_species_type_capacity(species_id)) {
            fclose(f);
            return 0;
        }
        g_species_type1[species_id - 1] = type1_id;
        g_species_type2[species_id - 1] = type2_id;
        g_species_type_entries += 1;
    }
    fclose(f);
    return 1;
}

static int ensure_move_type_capacity(int move_id) {
    int new_capacity;
    int* resized;
    int i;
    if (move_id <= g_move_type_capacity) {
        return 1;
    }
    new_capacity = g_move_type_capacity ? g_move_type_capacity : 32;
    while (new_capacity < move_id) {
        new_capacity *= 2;
    }
    resized = (int*)realloc(g_move_type, (size_t)new_capacity * sizeof(int));
    if (!resized) {
        return 0;
    }
    g_move_type = resized;
    for (i = g_move_type_capacity; i < new_capacity; ++i) {
        g_move_type[i] = 0;
    }
    g_move_type_capacity = new_capacity;
    return 1;
}

static int ensure_move_semantic_capacity(int move_id) {
    int new_capacity;
    MoveSemanticMetadata* resized;
    if (move_id <= g_move_semantic_capacity) return 1;
    new_capacity = g_move_semantic_capacity ? g_move_semantic_capacity : 32;
    while (new_capacity < move_id) new_capacity *= 2;
    resized = (MoveSemanticMetadata*)realloc(
        g_move_semantics, (size_t)new_capacity * sizeof(*resized));
    if (!resized) return 0;
    memset(resized + g_move_semantic_capacity, 0,
        (size_t)(new_capacity - g_move_semantic_capacity) * sizeof(*resized));
    g_move_semantics = resized;
    g_move_semantic_capacity = new_capacity;
    return 1;
}

static int load_move_semantics_file(const char* path) {
    FILE* f = fopen(path, "r");
    char line[512];
    if (!f) return 0;
    while (fgets(line, sizeof(line), f)) {
        char* fields[20];
        char* token;
        int count = 0;
        int move_id;
        MoveSemanticMetadata metadata;
        if (line[0] == '#') continue;
        token = strtok(line, ",\r\n");
        while (token && count < 20) {
            fields[count++] = token;
            token = strtok(NULL, ",\r\n");
        }
        if (count != 20) continue;
        move_id = move_id_from_name(fields[0]);
        if (move_id <= 0 || !ensure_move_semantic_capacity(move_id)) {
            fclose(f);
            return 0;
        }
        memset(&metadata, 0, sizeof(metadata));
        metadata.base_power = atoi(fields[1]);
        metadata.accuracy = atoi(fields[2]);
        metadata.category = atoi(fields[3]);
        metadata.priority = atoi(fields[4]);
#define READ_ROLE(field, index) metadata.field = (unsigned char)(atoi(fields[index]) != 0)
        READ_ROLE(spread, 5);
        READ_ROLE(healing, 6);
        READ_ROLE(protection, 7);
        READ_ROLE(setup, 8);
        READ_ROLE(redirection, 9);
        READ_ROLE(damaging, 10);
        READ_ROLE(recoil, 11);
        READ_ROLE(drain, 12);
        READ_ROLE(self_switch, 13);
        READ_ROLE(force_switch, 14);
        READ_ROLE(status_effect, 15);
        READ_ROLE(weather, 16);
        READ_ROLE(terrain, 17);
        READ_ROLE(field_effect, 18);
        READ_ROLE(ally_support, 19);
#undef READ_ROLE
        g_move_semantics[move_id - 1] = metadata;
        ++g_move_semantic_entries;
    }
    fclose(f);
    return g_move_semantic_entries > 0;
}

static int load_move_types_file(const char* path) {
    FILE* f;
    char line[256];
    if (!path) {
        return 0;
    }
    f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        char move[128];
        char type[64];
        char* p = line;
        char* comma;
        int move_id;
        int type_id;
        while (*p && isspace((unsigned char)*p)) {
            ++p;
        }
        if (!*p || *p == '#') {
            continue;
        }
        strncpy(move, p, sizeof(move) - 1);
        move[sizeof(move) - 1] = '\0';
        comma = strchr(move, ',');
        if (!comma) {
            continue;
        }
        *comma++ = '\0';
        strncpy(type, comma, sizeof(type) - 1);
        type[sizeof(type) - 1] = '\0';
        {
            size_t len = strlen(type);
            while (len > 0 && isspace((unsigned char)type[len - 1])) {
                type[--len] = '\0';
            }
        }
        move_id = move_id_from_name(move);
        type_id = lookup_id(&g_type_table, type);
        if (move_id <= 0 || type_id <= 0) {
            continue;
        }
        if (!ensure_move_type_capacity(move_id)) {
            fclose(f);
            return 0;
        }
        g_move_type[move_id - 1] = type_id;
        g_move_type_entries += 1;
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
    static const char* kTypeNames[] = {
        "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
        "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"
    };
    size_t i;
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
    for (i = 0; i < sizeof(kTypeNames) / sizeof(kTypeNames[0]); ++i) {
        if (!table_append(&g_type_table, kTypeNames[i])) {
            return 0;
        }
    }
    /* Optional while the repository is being upgraded to exact type reconstruction. */
    load_species_types_file("data/species_types.txt");
    load_move_types_file("data/move_types.txt");
    load_move_semantics_file("data/move_semantics.txt");
    if (g_species_type_entries > 0 && g_species_type_entries < 100) {
        fprintf(stderr,
            "[id_tables] warning: data/species_types.txt only loaded %d entries; rerun py/tools/showdown_vocab_export.py\n",
            g_species_type_entries);
    }
    if (g_move_type_entries > 0 && g_move_type_entries < 100) {
        fprintf(stderr,
            "[id_tables] warning: data/move_types.txt only loaded %d entries; rerun py/tools/showdown_vocab_export.py\n",
            g_move_type_entries);
    }
    if (g_move_semantic_entries < 100) {
        fprintf(stderr,
            "[id_tables] warning: data/move_semantics.txt only loaded %d entries; rerun node tools/export_move_semantics.js\n",
            g_move_semantic_entries);
    }
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

int type_id_from_name(const char* name) {
    return lookup_id(&g_type_table, name);
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

const char* type_name_from_id(int id) {
    return lookup_name(&g_type_table, id);
}

int species_type1_from_id(int id) {
    if (id <= 0 || id > g_species_type_capacity || !g_species_type1) {
        return 0;
    }
    return g_species_type1[id - 1];
}

int species_type2_from_id(int id) {
    if (id <= 0 || id > g_species_type_capacity || !g_species_type2) {
        return 0;
    }
    return g_species_type2[id - 1];
}

int move_type_from_id(int id) {
    if (id <= 0 || id > g_move_type_capacity || !g_move_type) {
        return 0;
    }
    return g_move_type[id - 1];
}

int move_semantics_from_id(int id, MoveSemanticMetadata* output) {
    if (!output) return 0;
    memset(output, 0, sizeof(*output));
    if (id <= 0 || id > g_move_semantic_capacity || !g_move_semantics) return 0;
    *output = g_move_semantics[id - 1];
    return 1;
}
