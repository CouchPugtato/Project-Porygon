'use strict';

const {createFixture} = require('./position_simulator');

function tacticalOpening({absorb, protect = false} = {}) {
    return createFixture({changeTeams: teams => {
        Object.assign(teams[0][0], {species: 'Raichu', ability: 'Static', moves: ['thunderbolt', 'protect']});
        teams[0][1].moves = ['watergun', 'protect'];
        Object.assign(teams[1][0], absorb ?
            {species: 'Lanturn', ability: absorb, moves: ['amnesia']} :
            {species: 'Gyarados', ability: 'Intimidate', moves: protect ? ['splash', 'protect'] : ['splash']});
        teams[1][1].moves = ['irondefense'];
    }});
}

// Separate matchup fixtures for proposal comparisons, not published builds.
function coverageOpening(index) {
    const matchups = [
        [['Gardevoir', 'Synchronize', 'psychic', 'Fairy'], ['Arcanine', 'Flash Fire', 'flamethrower', 'Fire'],
            ['Toxicroak', 'Dry Skin', 'poisonjab', 'Poison'], ['Corviknight', 'Pressure', 'ironhead', 'Steel']],
        [['Gastrodon', 'Sticky Hold', 'earthpower', 'Ground'], ['Kilowattrel', 'Volt Absorb', 'thunderbolt', 'Electric'],
            ['Coalossal', 'Flame Body', 'flamethrower', 'Fire'], ['Pelipper', 'Keen Eye', 'waterpulse', 'Water']],
        [['Sylveon', 'Pixilate', 'hypervoice', 'Fairy'], ['Lucario', 'Inner Focus', 'aurasphere', 'Fighting'],
            ['Hydreigon', 'Levitate', 'darkpulse', 'Dark'], ['Kingambit', 'Defiant', 'ironhead', 'Steel']],
    ];
    if (!matchups[index]) throw new Error('unknown coverage fixture');
    return createFixture({changeTeams: teams => {
        matchups[index].forEach(([species, ability, move, teraType], i) => {
            Object.assign(teams[Math.floor(i / 2)][i % 2], {species, ability, moves: [move, 'protect'], teraType});
        });
    }});
}

module.exports = {tacticalOpening, coverageOpening};
