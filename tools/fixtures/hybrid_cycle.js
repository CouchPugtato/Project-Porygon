'use strict';

const {createFixture} = require('./position_simulator');

function createHybridFixture(options = {}) {
    return createFixture({...options, changeTeams: teams => {
        for (const team of teams) for (const mon of team) mon.evs = {hp: 1};
        options.changeTeams?.(teams);
    }});
}

module.exports = {createHybridFixture};
