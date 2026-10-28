'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {replacementActions} = require('./replacement_actions');
const {isPostKnockoutReplacement} = require('./replacement_actions');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {observeBeliefs} = require('./belief_state');
const {constructMoveHistoryWorld} = require('./move_history_world');
const {searchMoveHistory} = require('./move_history_search');
const {capturePosition, restorePosition, playerObservation, advancePosition, pendingSides} = require('./position_simulator');
const {checkWorldChoice} = require('./opening_actions');
const {replacementFixture} = require('./fixtures/replacement_history');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {createFixture, comparable} = require('./fixtures/position_simulator');

const WORLD = {seed: [333, 1, 2, 3], chance_seed: [333, 4, 5, 6]};
const SEARCH = {seed: [333, 1, 2, 3], worlds: 1, candidates: 6, responses: 2,
    chance_samples: 2, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [333, 7, 8, 9]};

for (const kind of ['single', 'double_one_reserve', 'double_two_reserves', 'both'])
    test(`replacement request and replay ${kind}`, async () => {
        const {battle, history} = replacementFixture({kind});
        try {
            assertLegalTeams(battle);
            const before = structuredClone(history), view = history.observations.at(-1);
            assert.equal(view.phase, 'switch'); assert.equal(battle.queue.list.length, 0);
            const choices = replacementActions(view).turns;
            assert.equal(choices.length, kind.startsWith('double') ? 2 : 1);
            const position = capturePosition(battle, 'fixture');
            for (const choice of choices) assert(checkWorldChoice(position, view, choice.command, SAMPLE).accepted);
            const world = constructMoveHistoryWorld(history, WORLD);
            assert(world.supported, JSON.stringify({...world, position: null}));
            const copy = restorePosition(world.position, SAMPLE);
            try {
                assert.equal(copy.queue.list.length, 0);
                assert.deepEqual(playerObservation(copy, 'p1'), view); assertLegalTeams(copy);
            } finally { copy.destroy(); }
            const result = await searchMoveHistory(history, SEARCH);
            assert(result.supported, JSON.stringify(result));
            assert.equal(result.decision_phase, 'replacement'); assert.equal(result.ranked, false);
            assert(choices.some(choice => choice.command === result.selected_command));
            assert(result.nodes <= SEARCH.max_nodes);
            const inputs = {p1: result.selected_command};
            if (pendingSides(battle).includes('p2')) inputs.p2 = replacementActions(playerObservation(battle, 'p2')).turns[0].command;
            const next = advancePosition(position, inputs, SAMPLE);
            assert.equal(next.phase, 'move'); assert.equal(next.turn, 2);
            assert.deepEqual(history, before);
        } finally { battle.destroy(); }
    });

for (const side of ['p1', 'p2']) for (const kind of ['single', 'both'])
    test(`history through replacement ${kind} ${side}`, async () => {
        const {battle, history} = replacementFixture({kind, side, finish: true});
        try {
            const world = constructMoveHistoryWorld(history, WORLD);
            assert(world.supported, JSON.stringify({...world, position: null}));
            const copy = restorePosition(world.position, SAMPLE);
            try { assert.deepEqual(playerObservation(copy, side), history.observations.at(-1)); }
            finally { copy.destroy(); }
            assert.equal(history.observations.at(-1).phase, 'move');
            if (side === 'p2' && kind === 'single') assert.equal(history.own_choices[1], null);
            if (side === 'p1' && kind === 'single') {
                const resumed = await searchMoveHistory(history, SEARCH);
                assert(resumed.supported, JSON.stringify(resumed));
                assert.equal(resumed.root_turn, 2);
            }
        } finally { battle.destroy(); }
    });

test('waiting player submits no choice while the opponent replaces', async () => {
    const {battle, history} = replacementFixture({side: 'p2'});
    try {
        assert.equal(history.observations.at(-1).request.wait, true);
        const result = await searchMoveHistory(history, SEARCH);
        assert.equal(result.waiting, true); assert.equal(result.selected_command, null);
        assert.equal(result.fallback_required, false); assert.equal(result.nodes, 0);
    } finally { battle.destroy(); }
});

for (const side of ['p1', 'p2']) test(`terminal reconstruction ${side} has no replacement command`, async () => {
    const {battle, history} = replacementFixture({kind: 'terminal', side});
    try {
        assertLegalTeams(battle); assert.equal(battle.ended, true);
        const world = constructMoveHistoryWorld(history, WORLD);
        assert(world.supported, JSON.stringify({...world, position: null}));
        const copy = restorePosition(world.position, SAMPLE);
        try {
            if (side === 'p1') {
                assert(copy.p2.pokemon.every(mon => mon.moveSlots.every(move => move.pp === move.maxpp)));
                assert(observeBeliefs(history.observations.at(-1)).opponent.pokemon.every(mon => mon.used_moves.length === 0));
            }
        } finally { copy.destroy(); }
        const result = await searchMoveHistory(history, SEARCH);
        assert.equal(result.terminal, true); assert.equal(result.selected_command, null);
        assert.equal(result.fallback_required, false);
        assert.throws(() => replacementActions(history.observations.at(-1)), /pending replacement/);
    } finally { battle.destroy(); }
});

test('remaining ordinary actions execute before the end of turn replacement request', () => {
    const {battle, history} = replacementFixture();
    try {
        const log = history.observations.at(-1).log;
        const faint = log.findIndex(line => line.startsWith('|faint|p1a:'));
        const laterMove = log.findIndex(line => line.startsWith('|move|p1b:') && line.includes('|Water Gun|'));
        assert(faint >= 0 && laterMove > faint);
        const water = battle.p1.active[1].moveSlots.find(move => move.id === 'watergun');
        assert.equal(water.maxpp - water.pp, 1); assert.equal(battle.queue.list.length, 0);
    } finally { battle.destroy(); }
});

test('replacement masks require all available reserves without allowing duplicates or living outgoing slots', () => {
    const {battle, history} = replacementFixture({kind: 'double_two_reserves'});
    try {
        const view = history.observations.at(-1), commands = replacementActions(view).turns.map(turn => turn.command);
        assert.deepEqual(new Set(commands), new Set(['switch 3, switch 4', 'switch 4, switch 3']));
        assert(!commands.includes('switch 3, switch 3')); assert(!commands.some(command => command.includes('pass')));
        const invalid = structuredClone(view); invalid.request.side.pokemon[0].condition = '1/11';
        assert.throws(() => replacementActions(invalid), /fainted slot/);
    } finally { battle.destroy(); }
});

test('replacement selection is isolated from actual hidden builds and future rng', async () => {
    const make = variant => replacementFixture({changeTeams: teams => {
        teams[0][0].ability = 'Lightning Rod';
        teams[1][0].ability = variant ? 'Lightning Rod' : 'Static';
        teams[1][0].item = variant ? 'Choice Band' : 'Leftovers';
        teams[1][0].evs = variant ? {atk: 252, spe: 4} : {hp: 1};
        Object.assign(teams[1][2], {species: variant ? 'Slowbro' : 'Espeon',
            ability: variant ? 'Own Tempo' : 'Synchronize',
            moves: variant ? ['watergun', 'protect'] : ['psychic', 'protect'], teraType: 'Psychic'});
    }});
    const first = make(0), second = make(1);
    try {
        assertLegalTeams(first.battle); assertLegalTeams(second.battle);
        second.battle.prng = new PRNG('999,1,2,3');
        assert.deepEqual(first.history, second.history);
        const a = constructMoveHistoryWorld(first.history, WORLD), b = constructMoveHistoryWorld(second.history, WORLD);
        assert(a.supported); assert(b.supported);
        assert.deepEqual(comparable(a.position), comparable(b.position));
        const selectedA = await searchMoveHistory(first.history, SEARCH), selectedB = await searchMoveHistory(second.history, SEARCH);
        assert.equal(selectedA.selected_command, selectedB.selected_command);
        assert.equal(selectedA.nodes, selectedB.nodes); assert.equal(selectedA.ranked, false);
    } finally { first.battle.destroy(); second.battle.destroy(); }
});

test('midturn living pivots are not treated as knockout replacement requests', async () => {
    const battle = createFixture();
    try {
        const initial = playerObservation(battle, 'p1');
        assert(battle.choose('p1', 'move 4, move 4 1')); assert(battle.choose('p2', 'move 1 2, move 3'));
        const view = playerObservation(battle, 'p1');
        assert(battle.queue.list.length > 0); assert.equal(isPostKnockoutReplacement(battle), false);
        assert.throws(() => replacementActions(view), /fainted slot/);
        const result = await searchMoveHistory({observations: [initial, view], own_choices: ['move 4, move 4 1']}, SEARCH);
        assert.equal(result.supported, false); assert.equal(result.selected_command, null);
    } finally { battle.destroy(); }
});
