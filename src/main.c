#include <stdio.h>
#include <stdlib.h>
#include "pokemon.h"

int main(){
    
    
    BattleState battle = {
        .self = {
            .team = {0},
            .active1 = -1,
            .active2 = -1
        },
        .opponent = {
            .team = {0},
            .active1 = -1,
            .active2 = -1
        }
    };
 
    return 0;
}