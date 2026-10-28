'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {proposeOpeningTurns} = require('./opening_proposals');
const {openingActions, checkWorldChoice} = require('./opening_actions');
const {capturePosition, playerObservation} = require('./position_simulator');
const {createFixture} = require('./fixtures/position_simulator');

const rng = () => new PRNG('327,1,2,3');

test('coverage proposals include attacks, defense and both switching slots without changing legality', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1'), original = structuredClone(view);
        const proposals = proposeOpeningTurns(view, 12, rng());
        assert.deepEqual(proposals.coverage.missing, []);
        assert.equal(proposals.commands.length, 12); assert.equal(new Set(proposals.commands).size, 12);
        assert.equal(proposals.coverage.exploratory_choices, 1);
        assert(proposals.coverage.features.includes('ally_support'));
        const position = capturePosition(battle, 'fixture');
        for (const command of proposals.commands)
            assert(checkWorldChoice(position, view, command, {mode: 'sample', seed: [327, 4, 5, 6]}).accepted);
        assert.deepEqual(view, original);
    } finally { battle.destroy(); }
});

test('small budgets report uncovered categories rather than implying exhaustive coverage', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1');
        const one = proposeOpeningTurns(view, 1, rng());
        assert.equal(one.commands.length, 1); assert(one.coverage.missing.length > 0);
        const knownTrap = structuredClone(view); knownTrap.request.active[0].trapped = true;
        const trapped = proposeOpeningTurns(knownTrap, 12, rng());
        assert(!trapped.coverage.available.includes('switch_0'));
        assert(trapped.commands.every(command => !command.startsWith('switch')));
        assert.throws(() => proposeOpeningTurns(view, 100, rng()), /1..32/);
    } finally { battle.destroy(); }
});

test('uniform and coverage modes consume identical proposal RNG and remain deterministic', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1'), uniformRng = rng(), coverageRng = rng();
        const uniform = proposeOpeningTurns(view, 8, uniformRng, 'uniform');
        const coverage = proposeOpeningTurns(view, 8, coverageRng);
        assert.equal(uniformRng.random(65536), coverageRng.random(65536));
        assert.deepEqual(coverage, proposeOpeningTurns(view, 8, rng()));
        assert.notDeepEqual(coverage.commands, uniform.commands);
    } finally { battle.destroy(); }
});

test('legal friendly fire stays in the pool and can be selected by exploration', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1');
        let exploredAllyDamage = false;
        for (let seed = 0; seed < 64 && !exploredAllyDamage; ++seed) {
            const proposal = proposeOpeningTurns(view, 8, new PRNG(`327,${seed},2,3`));
            exploredAllyDamage = proposal.commands.some(command => /^move [12] -2/.test(command));
        }
        assert(exploredAllyDamage);
        assert(openingActions(view).turns.some(turn => turn.command === 'move 1 -2, move 2'));
    } finally { battle.destroy(); }
});

test('hidden builds and uncertain traps cannot change proposal coverage', () => {
    const battles = ['Sturdy', 'Magnet Pull'].map(ability => createFixture({changeTeams: teams => {
        Object.assign(teams[0][1], {species: 'Metagross', ability: 'Clear Body', moves: ['protect']});
        Object.assign(teams[1][0], {species: 'Magnezone', ability});
    }}));
    try {
        const views = battles.map(battle => playerObservation(battle, 'p1'));
        assert.deepEqual(views[0], views[1]);
        assert.deepEqual(proposeOpeningTurns(views[0], 12, rng()), proposeOpeningTurns(views[1], 12, rng()));
        const opposite = proposeOpeningTurns(playerObservation(battles[0], 'p2'), 8, rng());
        assert(opposite.commands.length > 0);
    } finally { battles.forEach(battle => battle.destroy()); }
});
