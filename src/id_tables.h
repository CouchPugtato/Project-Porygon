#ifndef ID_TABLES_H
#define ID_TABLES_H

int id_tables_init(void);

int species_id_from_name(const char* name);
int move_id_from_name(const char* name);
int item_id_from_name(const char* name);
int ability_id_from_name(const char* name);

const char* species_name_from_id(int id);
const char* move_name_from_id(int id);
const char* item_name_from_id(int id);
const char* ability_name_from_id(int id);

#endif
