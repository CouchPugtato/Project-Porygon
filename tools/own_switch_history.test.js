'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {searchMoveHistory} = require('./move_history_search');
const {ownSwitchActions, ordinaryActions, checkWorldChoice} = require('./opening_actions');
const {compareOwnSwitchTurns} = require('./opening_search');
const {capturePosition, playerObservation, restorePosition} = require('./position_simulator');
const {ownSwitchSpecs, ownSwitchFixture} = require('./fixtures/own_switch_history');
const {historyFixture} = require('./fixtures/move_history');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');

const WORLD = {seed: [331, 1, 2, 3], chance_seed: [331, 4, 5, 6]};
const SEARCH = {seed: [331, 1, 2, 3], worlds: 1, candidates: 8, responses: 2,
    chance_samples: 2, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [331, 7, 8, 9]};

for (const spec of ownSwitchSpecs()) test(`own switch replay ${spec.name}`, async () => {
    const {battle, history} = ownSwitchFixture(spec);
    try {
        assertLegalTeams(battle);
        const before = structuredClone(history), world = constructMoveHistoryWorld(history, WORLD);
        assert(world.supported, JSON.stringify({...world, position: null}));
        const restored = restorePosition(world.position, SAMPLE);
        try {
            assert.deepEqual(playerObservation(restored, spec.side || 'p1'), history.observations.at(-1));
            assertLegalTeams(restored);
        } finally { restored.destroy(); }
        if (spec.name.startsWith('preserve_')) {
            const injured = history.observations[1].request.side.pokemon.find(mon => mon.ident === 'p1: Blastoise');
            const returned = history.observations.at(-1).request.side.pokemon.find(mon => mon.ident === injured.ident);
            assert.equal(returned.active, true); assert.equal(returned.condition, injured.condition);
            assert.deepEqual(history.observations.at(-1).request.active[1].moves,
                history.observations[1].request.active[1].moves);
            if (spec.name === 'preserve_damage_and_pp') assert(battle.p1.active[1].hp < battle.p1.active[1].maxhp);
            else assert.match(returned.condition, /par$/);
        }
        const result = await searchMoveHistory(history, SEARCH);
        assert(result.supported, JSON.stringify(result));
        assert(result.nodes <= SEARCH.max_nodes);
        assert(ownSwitchActions(history.observations.at(-1)).turns.some(turn => turn.command === result.selected_command));
        assert.deepEqual(history, before);
    } finally { battle.destroy(); }
});

test('switch proposals respect current roster positions and reject duplicate destinations', () => {
    const spec = ownSwitchSpecs().find(entry => entry.name === 'two_switch_destinations');
    const {battle, history} = ownSwitchFixture({...spec, turns: spec.turns.slice(0, 1)});
    try {
        const view = history.observations.at(-1), pool = ownSwitchActions(view).turns;
        assert.equal(view.request.side.pokemon[2].ident, 'p1: Pikachu');
        assert(pool.some(turn => turn.command === 'switch 3, switch 4'));
        assert(!pool.some(turn => turn.command === 'switch 3, switch 3'));
        assert(ordinaryActions(view).turns.every(turn => turn.actions.every(action => action.kind !== 'switch')));
        const position = capturePosition(battle, 'fixture');
        for (const turn of pool) assert(checkWorldChoice(position, view, turn.command, SAMPLE).accepted, turn.command);
        const trapped = structuredClone(view); trapped.request.active[0].trapped = true;
        assert(ownSwitchActions(trapped).turns.every(turn => turn.actions[0].kind !== 'switch'));
        const fainted = structuredClone(view); fainted.request.side.pokemon[2].condition = '0 fnt';
        assert(ownSwitchActions(fainted).turns.every(turn => turn.actions.every(action => action.switch !== 3)));
    } finally { battle.destroy(); }
});

test('forced opposing switches and mismatched own switch evidence abstain before replay', () => {
    const {battle, history} = ownSwitchFixture(ownSwitchSpecs()[0]);
    try {
        const duplicate = structuredClone(history); duplicate.own_choices[0] = 'switch 3, switch 3';
        const active = structuredClone(history); active.own_choices[0] = 'switch 1, move 2';
        const wrongIdentity = structuredClone(history);
        wrongIdentity.observations[1].log = wrongIdentity.observations[1].log.map(line =>
            line.startsWith('|switch|p1a: Venusaur|') ? line.replace('p1a: Venusaur', 'p1a: NotVenusaur') : line);
        for (const input of [duplicate, active, wrongIdentity]) {
            const result = constructMoveHistoryWorld(input, WORLD);
            assert.equal(result.supported, false); assert.equal(result.position, null); assert.equal(result.nodes, 0);
        }
    } finally { battle.destroy(); }
    const opposing = historyFixture({turns: [{p1: 'move 2, move 2', p2: 'switch 3, move 2'}]});
    try {
        opposing.history.observations[1].log = opposing.history.observations[1].log.map(line =>
            line.startsWith('|switch|p2a: Clefable|') ? line.replace('|switch|', '|drag|') : line);
        const result = constructMoveHistoryWorld(opposing.history, WORLD);
        assert.equal(result.supported, false); assert.equal(result.nodes, 0);
        assert(result.unsupported.includes('unsupported_event_drag'));
    } finally { opposing.battle.destroy(); }
});

test('unrevealed trapping cannot change proposals and is rejected only inside the trapping world', async () => {
    const fixtures = ['Sturdy', 'Magnet Pull'].map(ability => historyFixture({changeTeams: teams => {
        Object.assign(teams[0][1], {species: 'Metagross', ability: 'Clear Body', gender: '', moves: ['tackle', 'protect']});
        Object.assign(teams[1][0], {species: 'Magnezone', ability, gender: ''});
    }}));
    try {
        const views = fixtures.map(({history}) => history.observations.at(-1));
        assert.deepEqual(views[0], views[1]);
        assert.deepEqual(ownSwitchActions(views[0]), ownSwitchActions(views[1]));
        const probe = 'move 2, switch 3', safe = 'move 2, move 2';
        assert(ownSwitchActions(views[0]).turns.some(turn => turn.command === probe));
        const worlds = fixtures.map(({battle}) => ({position: capturePosition(battle, 'fixture'), responses: ['move 2, move 2']}));
        const result = await compareOwnSwitchTurns(views[0], worlds, [probe, safe], {...SEARCH, worlds: 2});
        assert.equal(result.candidates[0].status, 'world_conditional_rejection');
        assert.equal(result.candidates[0].rejected_world, 1);
        assert.equal(result.candidates[0].score, null); assert.equal(result.selected_command, safe);
        await assert.rejects(compareOwnSwitchTurns(views[0],
            [{position: worlds[0].position, responses: ['switch 3, move 2']}], [safe], SEARCH), /responses/);
        assert.deepEqual(playerObservation(fixtures[1].battle, 'p1'), views[1]);
    } finally { fixtures.forEach(({battle}) => battle.destroy()); }
});
