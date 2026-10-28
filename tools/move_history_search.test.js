'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {searchMoveHistory} = require('./move_history_search');
const {ordinaryActions, checkWorldChoice} = require('./opening_actions');
const {compareOrdinaryTurns} = require('./opening_search');
const {capturePosition, playerObservation, restorePosition} = require('./position_simulator');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {historyFixture} = require('./fixtures/move_history');

const OPTIONS = {seed: [330, 1, 2, 3], worlds: 2, candidates: 6, responses: 2,
    chance_samples: 2, max_nodes: 256, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [330, 4, 5, 6]};

for (const spec of [
    {name: 'turn_two'},
    {name: 'turn_three', turns: [
        {p1: 'move 2, move 2', p2: 'move 1 1, move 1 2'},
        {p1: 'move 1 1, move 2', p2: 'move 2, move 2'},
    ]},
    {name: 'damaged_own_active', turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
    {name: 'paralyzed_active', status: true, turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]},
    {name: 'opposite_perspective', side: 'p2'},
]) test(`history search ${spec.name}`, async () => {
    const {battle, history} = historyFixture(spec);
    try {
        const before = structuredClone(history), result = await searchMoveHistory(history, OPTIONS);
        assert(result.supported, JSON.stringify(result));
        assert.equal(result.root_turn, history.observations.at(-1).turn);
        assert.equal(result.fallback_required, false); assert.equal(result.fixture_worlds, false);
        assert.equal(result.nodes, result.reconstruction_nodes + result.search_nodes);
        assert(result.nodes <= OPTIONS.max_nodes);
        const pool = ordinaryActions(history.observations.at(-1)).turns;
        assert(pool.some(turn => turn.command === result.selected_command));
        assert(!/switch|terastallize/.test(result.selected_command));
        assert(result.candidates.filter(row => row.status === 'completed').every(row => row.worlds.length === 2));
        assert.deepEqual(history, before);
    } finally { battle.destroy(); }
});

test('later turn candidates remove unavailable moves without changing the observation', () => {
    const {battle, history} = historyFixture();
    try {
        const view = structuredClone(history.observations.at(-1));
        view.request.active[0].moves[0].disabled = true;
        view.request.active[1].moves[0].pp = 0;
        const pool = ordinaryActions(view).turns;
        assert(pool.length > 0);
        assert(pool.every(turn => turn.actions[0].move !== 1 && turn.actions[1].move !== 1));
        assert(pool.every(turn => turn.actions.every(action => !action.tera && action.kind === 'move')));
        view.request.active[0].moves.forEach(move => { move.pp = 0; });
        assert.throws(() => ordinaryActions(view), /move action space/);
    } finally { battle.destroy(); }
});

test('known fainted friendly slot produces a native legal pass rather than a move', () => {
    const {battle, history} = historyFixture({changeTeams: teams => {
        teams[0].splice(2); teams[0][0].level = 1;
    }, turns: [{p1: 'move 1 1, move 2', p2: 'move 1 1, move 2'}]});
    try {
        const view = history.observations.at(-1), position = capturePosition(battle, 'fixture');
        assert.equal(view.phase, 'move'); assert.equal(view.request.side.pokemon[0].condition, '0 fnt');
        const pool = ordinaryActions(view).turns;
        assert(pool.every(turn => turn.actions[0].kind === 'pass'));
        for (const turn of pool) assert(checkWorldChoice(position, view, turn.command, SAMPLE).accepted);
    } finally { battle.destroy(); }
});

test('later turn item pivot is not scored as a completed turn', async () => {
    const {battle, history} = historyFixture({changeTeams: teams => { teams[1][0].item = 'Eject Button'; }});
    try {
        const result = await compareOrdinaryTurns(history.observations.at(-1),
            [{position: capturePosition(battle, 'fixture'), responses: ['move 1 1, move 2']}],
            ['move 1 1, move 2', 'move 2, move 2'], {...OPTIONS, worlds: 1});
        assert.equal(result.root_turn, 2);
        assert.equal(result.candidates[0].status, 'unsupported_continuation');
        assert.equal(result.candidates[0].score, null);
        assert.equal(result.selected_command, 'move 2, move 2');
    } finally { battle.destroy(); }
});

test('later turn values exclude damage already present in the observed history', async () => {
    const {battle, history} = historyFixture({turns: [{p1: 'move 2, move 1 1', p2: 'move 2, move 1 2'}]});
    try {
        assert(battle.p1.active[1].hp < battle.p1.active[1].maxhp);
        const result = await compareOrdinaryTurns(history.observations.at(-1),
            [{position: capturePosition(battle, 'fixture'), responses: ['move 2, move 2']}],
            ['move 2, move 2'], {...OPTIONS, worlds: 1});
        assert.equal(result.selected_score, 0);
    } finally { battle.destroy(); }
});

test('reconstruction and search share a budget and never invent a fallback command', async () => {
    const {battle, history} = historyFixture();
    try {
        const tiny = await searchMoveHistory(history, {...OPTIONS, max_nodes: 1});
        assert.equal(tiny.supported, false); assert.equal(tiny.selected_command, null);
        assert.equal(tiny.fallback_required, true); assert(tiny.nodes <= 1);
        const unsupported = structuredClone(history); unsupported.own_choices[0] = 'switch 3, move 2';
        const abstention = await searchMoveHistory(unsupported, OPTIONS);
        assert.equal(abstention.selected_command, null); assert.equal(abstention.nodes, 0);
        const deadline = await searchMoveHistory(history, {...OPTIONS, max_elapsed_ms: 10},
            {waitForLaunch: () => new Promise(resolve => setTimeout(resolve, 20))});
        assert.equal(deadline.supported, false); assert.equal(deadline.nodes, 0);
    } finally { battle.destroy(); }
});

test('identical observed histories yield identical decisions despite different hidden truth', async () => {
    const first = historyFixture(), second = historyFixture({seed: [900, 1, 2, 3], changeTeams: teams => {
        Object.assign(teams[1][0], {ability: 'Thick Fat', item: 'Choice Band', moves: ['tackle', 'rest']});
        Object.assign(teams[1][2], {species: 'Slowbro', ability: 'Own Tempo'});
    }});
    try {
        assert.deepEqual(first.history, second.history);
        const a = await searchMoveHistory(first.history, OPTIONS), b = await searchMoveHistory(second.history, OPTIONS);
        assert(a.supported); assert(b.supported);
        assert.equal(a.selected_command, b.selected_command); assert.equal(a.nodes, b.nodes);
        assert.deepEqual(a.candidates, b.candidates);
        const world = constructMoveHistoryWorld(first.history,
            {seed: OPTIONS.seed, chance_seed: SAMPLE.seed});
        assert(world.supported);
        const restored = restorePosition(world.position, SAMPLE);
        try { assert.deepEqual(playerObservation(restored, 'p1'), first.history.observations.at(-1)); }
        finally { restored.destroy(); }
    } finally { first.battle.destroy(); second.battle.destroy(); }
});
