'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {openingActions, checkWorldChoice} = require('./opening_actions');
const {capturePosition, playerObservation} = require('./position_simulator');
const {createFixture, comparable} = require('./fixtures/position_simulator');

const SAMPLE = {mode: 'sample', seed: [325, 1, 2, 3]};

test('all opening combinations validate without mutating the root, including friendly fire', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1'), original = structuredClone(view);
        const position = capturePosition(battle, 'fixture'), root = comparable(position);
        const generated = openingActions(view);
        assert.deepEqual(generated.slot_counts, [17, 17]);
        assert.equal(generated.turns.length, 224);
        assert.equal(new Set(generated.turns.map(turn => turn.command)).size, 224);
        assert(generated.turns.some(turn => turn.command === 'move 1 -2, move 3'));
        for (const turn of generated.turns) {
            const checked = checkWorldChoice(position, view, turn.command, SAMPLE);
            assert(checked.accepted, `${turn.command}: ${checked.error}`);
            assert.equal(checked.request_updated, false);
        }
        assert.deepEqual(view, original); assert.deepEqual(comparable(position), root);
        assert.deepEqual(playerObservation(battle, 'p1'), original);
    } finally { battle.destroy(); }
});

test('Helping Hand targets only the partner; spread and self moves omit targets', () => {
    const battle = createFixture();
    try {
        const turns = openingActions(playerObservation(battle, 'p1')).turns;
        const first = turns.flatMap(turn => [turn.actions[0]]);
        assert(first.filter(a => a.move === 3).every(a => a.target === -2));
        assert(first.filter(a => a.move === 4).every(a => a.target === 0));
        assert(turns.map(t => t.actions[1]).filter(a => a.move === 3).every(a => a.target === 0));
        const position = capturePosition(battle, 'fixture'), view = playerObservation(battle, 'p1');
        for (const command of ['move 3, move 2', 'move 3 1, move 2', 'move 4 1, move 2',
            'switch 3, switch 3', 'move 4 terastallize, move 2 terastallize'])
            assert.equal(checkWorldChoice(position, view, command, SAMPLE).accepted, false, command);
    } finally { battle.destroy(); }
});

test('known trap, disabled moves, unavailable Tera and fainted reserves respect the request', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1');
        view.request.active[0].trapped = true;
        view.request.active[0].moves[0].disabled = true;
        delete view.request.active[1].canTerastallize;
        view.request.side.pokemon[2].condition = '0 fnt';
        const turns = openingActions(view).turns;
        assert(turns.length > 0);
        assert(turns.every(t => t.actions.every(a => a.kind === 'move')));
        assert(turns.every(t => t.actions[0].move !== 1 && !t.actions[1].tera));
    } finally { battle.destroy(); }
});

test('hidden trapping changes world acceptance, never the visible candidate pool', () => {
    const free = createFixture({changeTeams: teams => {
        Object.assign(teams[0][1], {species: 'Metagross', ability: 'Clear Body', moves: ['protect']});
        Object.assign(teams[1][0], {species: 'Magnezone', ability: 'Sturdy'});
    }});
    const trapped = createFixture({changeTeams: teams => {
        Object.assign(teams[0][1], {species: 'Metagross', ability: 'Clear Body', moves: ['protect']});
        Object.assign(teams[1][0], {species: 'Magnezone', ability: 'Magnet Pull'});
    }});
    try {
        const view = playerObservation(free, 'p1'), other = playerObservation(trapped, 'p1');
        assert.deepEqual(view, other);
        assert.deepEqual(openingActions(view), openingActions(other));
        const command = 'move 4, switch 3';
        assert(openingActions(view).turns.some(t => t.command === command));
        assert(checkWorldChoice(capturePosition(free, 'fixture'), view, command, SAMPLE).accepted);
        const rejected = checkWorldChoice(capturePosition(trapped, 'fixture'), other, command, SAMPLE);
        assert.equal(rejected.accepted, false); assert.match(rejected.error, /trapped/);
        assert.equal(rejected.request_updated, true);
        assert.deepEqual(playerObservation(trapped, 'p1'), other);
    } finally { free.destroy(); trapped.destroy(); }
});

test('both perspectives and request-specific targets work without static move overrides', () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p2'), position = capturePosition(battle, 'fixture');
        for (const turn of openingActions(view).turns)
            assert(checkWorldChoice(position, view, turn.command, SAMPLE).accepted, turn.command);
        const changed = structuredClone(view);
        changed.request.active[0].moves[0].target = 'adjacentAllyOrSelf';
        const targets = new Set(openingActions(changed).turns.map(t => t.actions[0]).filter(a => a.move === 1).map(a => a.target));
        assert.deepEqual(targets, new Set([-1, -2]));
        assert.throws(() => checkWorldChoice(position, changed, 'move 1 -1, move 2', SAMPLE), /observation/);
        assert.throws(() => openingActions({...view, turn: 2}), /initial/);
        const injured = structuredClone(view); injured.request.side.pokemon[0].condition = '1/100';
        assert.throws(() => openingActions(injured), /initial/);
    } finally { battle.destroy(); }
});
