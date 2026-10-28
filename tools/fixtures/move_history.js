'use strict';

const assert = require('node:assert/strict');
const {createFixture} = require('./position_simulator');
const {playerObservation} = require('../position_simulator');

function historyFixture({status = false, changeTeams, seed, side = 'p1', turns} = {}) {
    const battle = createFixture({seed, changeTeams: teams => {
        teams[0][0].moves = ['thunderbolt', 'protect', 'thunderwave'];
        teams[0][1].moves = ['watergun', 'protect'];
        teams[1][0].moves = ['tackle', 'protect'];
        teams[1][1].moves = ['tackle', 'protect'];
        if (status) {
            Object.assign(teams[1][1], {species: 'Clefable', ability: 'Magic Guard',
                gender: 'M', moves: ['thunderwave', 'protect']});
            Object.assign(teams[1][2], {species: 'Bronzong', ability: 'Levitate', gender: '', moves: ['protect']});
        }
        teams.flat().forEach(mon => { mon.evs = {hp: 1}; });
        changeTeams?.(teams);
    }});
    try {
        const history = {observations: [playerObservation(battle, side)], own_choices: []};
        for (const choices of turns || [{p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'}]) {
            assert(battle.choose('p1', choices.p1)); assert(battle.choose('p2', choices.p2));
            history.own_choices.push(choices[side]);
            history.observations.push(playerObservation(battle, side));
        }
        return {battle, history};
    } catch (error) { battle.destroy(); throw error; }
}

module.exports = {historyFixture};
