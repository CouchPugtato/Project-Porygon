'use strict';

const {historyFixture} = require('./move_history');

function ownSwitchSpecs() {
    return [
        {name: 'switch_out_and_back', turns: [
            {p1: 'switch 3, move 2', p2: 'move 2, move 2'},
            {p1: 'switch 3, move 2', p2: 'move 2, move 2'},
        ]},
        {name: 'preserve_damage_and_pp', turns: [
            {p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'},
            {p1: 'move 2, switch 3', p2: 'move 2, move 2'},
            {p1: 'move 2, switch 3', p2: 'move 2, move 2'},
        ]},
        {name: 'preserve_status', status: true, turns: [
            {p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'},
            {p1: 'move 2, switch 3', p2: 'move 2, move 2'},
            {p1: 'move 2, switch 3', p2: 'move 2, move 2'},
        ]},
        {name: 'two_switch_destinations', changeTeams: teams => {
            teams[0].push({...structuredClone(teams[0][2]), species: 'Arcanine', ability: 'Flash Fire',
                moves: ['flamethrower', 'protect'], teraType: 'Fire'});
        }, turns: [
            {p1: 'switch 3, switch 4', p2: 'move 2, move 2'},
            {p1: 'switch 3, switch 4', p2: 'move 2, move 2'},
        ]},
        {name: 'opposite_perspective', side: 'p2', turns: [
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
            {p1: 'move 2, move 2', p2: 'switch 3, move 2'},
        ]},
    ];
}

function ownSwitchFixture(spec) { return historyFixture(spec); }

module.exports = {ownSwitchSpecs, ownSwitchFixture};
