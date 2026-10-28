'use strict';

const {historyFixture} = require('./move_history');

function opposingSwitchSpecs() {
    return [
        {name: 'opponent_out_and_back', turns: [
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
        ]},
        {name: 'opponent_damage_and_pp', turns: [
            {p1: 'move 2, move 1 1', p2: 'move 1 1, move 2'},
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
        ]},
        {name: 'new_entrant_reveals_one_move', turns: [
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
            {p1: 'move 2, move 2', p2: 'move 2, move 2'},
        ]},
        {name: 'both_sides_switch_with_different_bench_order', changeTeams: teams => {
            teams[0].push({...structuredClone(teams[0][2]), species: 'Arcanine', ability: 'Flash Fire',
                moves: ['flamethrower', 'protect'], teraType: 'Fire'});
            teams[1].push(structuredClone(teams[1][2]));
            Object.assign(teams[1][2], {species: 'Slowbro', ability: 'Own Tempo', gender: 'M',
                moves: ['watergun', 'protect'], teraType: 'Water'});
        }, turns: [
            {p1: 'switch 3, switch 4', p2: 'switch 3, switch 4'},
            {p1: 'switch 3, switch 4', p2: 'switch 3, switch 4'},
        ]},
        {name: 'opposite_perspective', side: 'p2', turns: [
            {p1: 'switch 3, move 2', p2: 'move 2, move 2'},
            {p1: 'switch 3, move 2', p2: 'move 2, move 2'},
        ]},
    ];
}

function opposingSwitchFixture(spec) { return historyFixture(spec); }

module.exports = {opposingSwitchSpecs, opposingSwitchFixture};
