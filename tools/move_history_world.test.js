'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {constructMoveHistoryWorld, replaySteps} = require('./move_history_world');
const {constructOpeningWorld} = require('./opening_world');
const {playerObservation, restorePosition} = require('./position_simulator');
const {comparable} = require('./fixtures/position_simulator');
const {historyFixture} = require('./fixtures/move_history');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');

const SETTINGS = {seed: [329, 1, 2, 3], chance_seed: [329, 4, 5, 6]};
const SAMPLE = {mode: 'sample', seed: [329, 7, 8, 9]};

for (const spec of [
    {name: 'protection'},
    {name: 'opposite_perspective', side: 'p2'},
    {name: 'outgoing_damage', turns: [{p1: 'move 2, move 1 1', p2: 'move 1 1, move 2'}]},
    {name: 'incoming_damage', turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
    {name: 'own_status', status: true, turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
    {name: 'two_completed_turns', turns: [
        {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
        {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
    ]},
    {name: 'three_completed_turns', turns: [
        {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
        {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
        {p1: 'move 2, move 2', p2: 'move 2, move 2'},
    ]},
]) test(`move history reconstruction ${spec.name}`, () => {
    const {battle, history} = historyFixture(spec);
    try {
        assertLegalTeams(battle);
        const before = structuredClone(history), result = constructMoveHistoryWorld(history, SETTINGS);
        assert(result.supported, JSON.stringify({...result, position: null}));
        assert.equal(result.position.origin, 'belief-sample');
        assert.equal(result.matched_observations, history.observations.length);
        assert.equal(result.search_ready, false);
        const restored = restorePosition(result.position, SAMPLE);
        try {
            const side = history.observations[0].side;
            assert.deepEqual(playerObservation(restored, side), history.observations.at(-1));
            assertLegalTeams(restored);
            const enemy = restored.getSide(side === 'p1' ? 'p2' : 'p1');
            const seen = enemy.active.flatMap(mon => mon.moveSlots.filter(move => move.pp < move.maxpp));
            assert(seen.length >= 2);
        } finally { restored.destroy(); }
        assert.deepEqual(history, before);
        if (spec.name === 'outgoing_damage') assert(history.observations.at(-1).log.some(line => line.startsWith('|-damage|p2a:')));
        if (spec.name === 'incoming_damage') assert(history.observations.at(-1).log.some(line => line.startsWith('|-damage|p1b:')));
        if (spec.name === 'own_status') assert.match(history.observations.at(-1).request.side.pokemon[1].condition, /par$/);
    } finally { battle.destroy(); }
});

test('history replay depends on observations and independent seeds rather than actual hidden truth', () => {
    const first = historyFixture(), second = historyFixture({seed: [900, 1, 2, 3], changeTeams: teams => {
        teams[1][0].ability = 'Thick Fat'; teams[1][0].item = 'Choice Band';
        teams[1][0].moves = ['tackle', 'bellydrum', 'rest'];
        Object.assign(teams[1][2], {species: 'Slowbro', ability: 'Own Tempo'});
    }});
    try {
        assert.deepEqual(first.history, second.history);
        const a = constructMoveHistoryWorld(first.history, SETTINGS), b = constructMoveHistoryWorld(second.history, SETTINGS);
        assert(a.supported); assert(b.supported);
        assert.deepEqual(comparable(a.position), comparable(b.position));
        assert.equal(a.nodes, b.nodes); assert.equal(a.history_sha256, b.history_sha256);
        const ordinaryOpening = constructOpeningWorld(first.history.observations[0],
            {seed: SETTINGS.seed, chance_seed: SETTINGS.chance_seed, attempts: 32});
        assert.equal(ordinaryOpening.scope, 'initial_move_request');
    } finally { first.battle.destroy(); second.battle.destroy(); }
});

test('unsupported histories abstain without patching a state', () => {
    const {battle, history} = historyFixture();
    try {
        const changed = structuredClone(history); changed.observations[1].log.push('|-terastallize|p2a: Snorlax|Normal');
        const missingMove = structuredClone(history);
        missingMove.observations[1].log = missingMove.observations[1].log.filter(line => !line.startsWith('|move|p2a:'));
        const wrongPrefix = structuredClone(history); wrongPrefix.observations[1].log[0] = '|t:|different';
        const switched = structuredClone(history); switched.own_choices[0] = 'switch 3, move 2';
        for (const input of [changed, missingMove, wrongPrefix, switched]) {
            const result = constructMoveHistoryWorld(input, SETTINGS);
            assert.equal(result.supported, false); assert.equal(result.position, null);
            assert(result.unsupported.length); assert.equal(result.nodes, 0);
        }
        assert.throws(() => replaySteps({observations: history.observations, own_choices: [], state_json: '{}'}), /observations/);
        assert.throws(() => constructMoveHistoryWorld(history, {...SETTINGS, max_nodes: 999999}), /bounded/);
    } finally { battle.destroy(); }
});

test('nonmatching health or pp evidence exhausts a small budget without publishing a patched world', () => {
    const {battle, history} = historyFixture();
    try {
        const impossible = structuredClone(history);
        impossible.observations[1].request.active[0].moves[0].pp = 999;
        const result = constructMoveHistoryWorld(impossible,
            {...SETTINGS, world_attempts: 2, chance_attempts: 2, max_nodes: 2});
        assert.equal(result.supported, false); assert.equal(result.position, null);
        assert(result.nodes <= 2); assert(result.unsupported.length > 0);
    } finally { battle.destroy(); }
});

test('every intermediate observation must match even when the final observation is unchanged', () => {
    const {battle, history} = historyFixture({turns: [
        {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
        {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
    ]});
    try {
        history.observations[1].request.active[0].moves[0].pp = 999;
        const result = constructMoveHistoryWorld(history,
            {...SETTINGS, world_attempts: 2, chance_attempts: 2, max_nodes: 4});
        assert.equal(result.supported, false); assert.equal(result.position, null);
        assert(result.rejected.observation_mismatch > 0);
    } finally { battle.destroy(); }
});
