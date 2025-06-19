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
    Pokemon pokemon[6]; // check format, and when this is applicable, if it is decided that singles will also be supported
    int active1;
    int active2; // check if it is a doubles battle and if this is real
} Team;

typedef struct BattleState {
    Team self;
    Team opponent;
};
