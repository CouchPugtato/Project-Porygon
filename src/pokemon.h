#ifndef POKEMON_H
#define POKEMON_H

#include <stdbool.h>

typedef struct {
    char species[50];
    int hp;
    int maxHp;
    bool active;
    bool fainted;
    char moves[4][50];
} Pokemon;

typedef struct {
    Pokemon team[6];
    int active1;
    int active2;
} Team;

typedef struct {
    Team self;
    Team opponent;
} BattleState;

#endif