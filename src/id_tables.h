#ifndef ID_TABLES_H
#define ID_TABLES_H

typedef struct {
    int base_power;
    int accuracy;
    int category;
    int priority;
    unsigned char spread;
    unsigned char healing;
    unsigned char protection;
    unsigned char setup;
    unsigned char redirection;
    unsigned char damaging;
    unsigned char recoil;
    unsigned char drain;
    unsigned char self_switch;
    unsigned char force_switch;
    unsigned char status_effect;
    unsigned char weather;
    unsigned char terrain;
    unsigned char field_effect;
    unsigned char ally_support;
} MoveSemanticMetadata;

int id_tables_init(void);

int species_id_from_name(const char* name);
int move_id_from_name(const char* name);
int item_id_from_name(const char* name);
int ability_id_from_name(const char* name);
int condition_id_from_name(const char* name);
int type_id_from_name(const char* name);

const char* species_name_from_id(int id);
const char* move_name_from_id(int id);
const char* item_name_from_id(int id);
const char* ability_name_from_id(int id);
const char* condition_name_from_id(int id);
const char* type_name_from_id(int id);
int species_type1_from_id(int id);
int species_type2_from_id(int id);
int move_type_from_id(int id);
int move_semantics_from_id(int id, MoveSemanticMetadata* output);

#endif
