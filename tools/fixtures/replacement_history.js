'use strict';

const assert = require('node:assert/strict');
const {createFixture} = require('./position_simulator');
const {playerObservation, pendingSides} = require('../position_simulator');

function replacementFixture({kind = 'single', side = 'p1', finish = false, seed, changeTeams} = {}) {
    const battle = createFixture({seed, changeTeams: teams => {
        Object.assign(teams[0][0], {level: 1, moves: ['thunderbolt', 'protect', 'thunderwave']});
        teams[0][1].moves = ['watergun', 'protect'];
        Object.assign(teams[1][0], {species: 'Raichu', ability: 'Static', gender: 'M', moves: ['quickattack', 'protect']});
        teams[1][1].moves = ['tackle', 'protect'];
        if (kind.startsWith('double')) {
            Object.assign(teams[0][1], {species: 'Jigglypuff', ability: 'Cute Charm', gender: 'M', level: 1,
                moves: ['thunderwave', 'protect']});
            if (kind === 'double_two_reserves') teams[0].push({...structuredClone(teams[0][2]),
                species: 'Arcanine', ability: 'Flash Fire', moves: ['flamethrower', 'protect'], teraType: 'Fire'});
        }
        if (kind === 'both') Object.assign(teams[1][1], {species: 'Hoppip', ability: 'Leaf Guard',
            gender: 'M', level: 1, moves: ['splash']});
        if (kind === 'terminal') {
            teams[0][0].level = 100; teams[1].length = 2;
            Object.assign(teams[1][0], {species: 'Sunkern', ability: 'Early Bird', gender: 'M', level: 1, moves: ['absorb']});
            Object.assign(teams[1][1], {species: 'Hoppip', ability: 'Leaf Guard', gender: 'M', level: 1, moves: ['splash']});
        }
        teams.flat().forEach(mon => { mon.evs = {hp: 1}; });
        changeTeams?.(teams);
    }});
    try {
        const history = {observations: [playerObservation(battle, side)], own_choices: []};
        const advance = choices => {
            const required = pendingSides(battle);
            assert.deepEqual(Object.keys(choices).sort(), [...required].sort());
            for (const player of required) assert(battle.choose(player, choices[player]));
            history.own_choices.push(choices[side] ?? null);
            history.observations.push(playerObservation(battle, side));
        };
        const choices = kind.startsWith('double') ? {p1: 'move 1 1, move 1 1', p2: 'move 1 1, move 1 2'} :
            kind === 'both' ? {p1: 'move 1 1, move 1 2', p2: 'move 1 1, move 1'} :
            kind === 'terminal' ? {p1: 'move 1 1, move 1 2', p2: 'move 1 1, move 1'} :
            {p1: 'move 1 1, move 1 2', p2: 'move 1 1, move 2'};
        advance(choices);
        if (finish && !battle.ended) {
            const replacements = {};
            if (pendingSides(battle).includes('p1')) replacements.p1 = kind === 'double_two_reserves' ?
                'switch 3, switch 4' : 'switch 3, pass';
            if (pendingSides(battle).includes('p2')) replacements.p2 = 'pass, switch 3';
            advance(replacements);
        }
        return {battle, history};
    } catch (error) { battle.destroy(); throw error; }
}

module.exports = {replacementFixture};
