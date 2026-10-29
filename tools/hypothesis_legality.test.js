'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {TeamValidator} = require('../external/pokemon-showdown/dist/sim/team-validator');
const {validateHypotheticalSet, validateHypotheticalTeam} = require('./hypothesis_legality');
const {sampleRevealedSets, observeBeliefs} = require('./belief_state');
const {playerObservation} = require('./position_simulator');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {constructOpeningWorld} = require('./opening_world');

function pecharunt() {
    return {species: 'Pecharunt', level: 78, ability: 'Poison Puppeteer', moves: ['shadowball', 'protect'],
        gender: 'N', evs: {hp: 1}};
}

test('native level adjustment separates acquisition requirements from observed random battle levels', () => {
    const set = pecharunt();
    assert(new TeamValidator('gen9randomdoublesbattle').validateSet(structuredClone(set), {}));
    assert.equal(validateHypotheticalSet(set), null); assert.equal(set.level, 78);
    const original = structuredClone(set);
    const partner = {species: 'Snorlax', level: 80, ability: 'Immunity', moves: ['protect'], evs: {hp: 1}};
    assert.equal(validateHypotheticalTeam([set, partner]), null); assert.deepEqual(set, original);
});

test('scaled hypotheses still reject illegal moves abilities investment and duplicate species', () => {
    for (const update of [{moves: ['spacialrend']}, {ability: 'Wonder Guard'}, {evs: {hp: 252, spa: 252, spe: 252}}])
        assert(validateHypotheticalSet({...pecharunt(), ...update}));
    const set = pecharunt(); assert.equal(validateHypotheticalSet(set), null);
    assert(validateHypotheticalTeam([set, structuredClone(set)]));
});

test('public scaled event species can receive rule only completions and exact startup worlds', () => {
    const battle = createHybridFixture({changeTeams: teams => { teams[1][0] = pecharunt(); }});
    try {
        const view = playerObservation(battle, 'p1'), original = structuredClone(view);
        const result = sampleRevealedSets(observeBeliefs(view), {seed: [358, 1, 2, 3], samples: 1, attempts_per_set: 32});
        assert.equal(result.particles.length, 1, JSON.stringify(result.unsupported));
        const sampled = result.particles[0].sets.find(entry => entry.set.species === 'Pecharunt').set;
        assert.equal(sampled.level, 78); assert.equal(validateHypotheticalSet(sampled), null);
        const world = constructOpeningWorld(view, {seed: [358, 4, 5, 6], chance_seed: [358, 7, 8, 9], attempts: 32});
        assert(world.supported, JSON.stringify(world.unsupported)); assert.deepEqual(view, original);
    } finally { battle.destroy(); }
});
