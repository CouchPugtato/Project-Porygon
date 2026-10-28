'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {constructMoveHistoryWorld, replaySteps} = require('./move_history_world');
const {constructReplayOpening} = require('./opening_world');
const {observeBeliefs} = require('./belief_state');
const {searchMoveHistory} = require('./move_history_search');
const {compareSwitchTurns} = require('./opening_search');
const {ownSwitchActions} = require('./opening_actions');
const {capturePosition, playerObservation, restorePosition} = require('./position_simulator');
const {opposingSwitchSpecs, opposingSwitchFixture} = require('./fixtures/opponent_switch_history');
const {historyFixture} = require('./fixtures/move_history');
const {comparable} = require('./fixtures/position_simulator');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');

const WORLD = {seed: [332, 1, 2, 3], chance_seed: [332, 4, 5, 6]};
const SEARCH = {seed: [332, 1, 2, 3], worlds: 1, candidates: 8, responses: 4,
    chance_samples: 2, max_nodes: 512, max_elapsed_ms: 5000};
const SAMPLE = {mode: 'sample', seed: [332, 7, 8, 9]};

for (const spec of opposingSwitchSpecs()) test(`opponent switch replay ${spec.name}`, async () => {
    const {battle, history} = opposingSwitchFixture(spec);
    try {
        assertLegalTeams(battle);
        const before = structuredClone(history), world = constructMoveHistoryWorld(history, WORLD);
        assert(world.supported, JSON.stringify({...world, position: null}));
        const restored = restorePosition(world.position, SAMPLE);
        try {
            const side = spec.side || 'p1', enemy = restored.getSide(side).foe;
            assertLegalTeams(restored); assert.deepEqual(playerObservation(restored, side), history.observations.at(-1));
            if (spec.name === 'opponent_damage_and_pp') {
                const snorlax = enemy.pokemon.find(mon => mon.species.id === 'snorlax');
                assert(snorlax.hp < snorlax.maxhp);
                const tackle = snorlax.moveSlots.find(move => move.id === 'tackle');
                assert.equal(tackle.maxpp - tackle.pp, 1);
            }
            if (spec.name === 'both_sides_switch_with_different_bench_order') {
                assert.notDeepEqual(enemy.pokemon.slice(2).map(mon => mon.fullname),
                    battle.p2.pokemon.slice(2).map(mon => mon.fullname));
            }
            if (spec.name === 'new_entrant_reveals_one_move') {
                const known = observeBeliefs(history.observations.at(-1)).opponent.pokemon.find(mon => mon.species === 'clefable');
                assert.deepEqual(known.moves, ['protect']);
                const hypothesized = enemy.pokemon.find(mon => mon.species.id === 'clefable');
                assert(hypothesized.moveSlots.some(move => move.id === 'protect'));
                assert.equal(hypothesized.moveSlots.length, 4);
            }
        } finally { restored.destroy(); }
        const result = await searchMoveHistory(history, SEARCH);
        assert(result.supported, JSON.stringify(result)); assert(result.nodes <= SEARCH.max_nodes);
        assert(result.candidates.filter(row => row.status === 'completed').some(row =>
            row.worlds.some(sample => sample.responses.some(response => response.command.includes('switch')))));
        assert.deepEqual(history, before);
    } finally { battle.destroy(); }
});

test('opponent response pool accepts voluntary switches and rejects duplicate bench destinations', async () => {
    const {battle, history} = historyFixture();
    try {
        const view = history.observations.at(-1), position = capturePosition(battle, 'fixture');
        const result = await compareSwitchTurns(view, [{position, responses: ['switch 3, move 2']}],
            ['move 2, move 2'], SEARCH);
        assert.equal(result.supported, true);
        assert.equal(result.candidates[0].worlds[0].responses[0].command, 'switch 3, move 2');
        await assert.rejects(compareSwitchTurns(view, [{position, responses: ['switch 3, switch 3']}],
            ['move 2, move 2'], SEARCH), /responses/);
        assert(ownSwitchActions(playerObservation(battle, 'p2')).turns.some(turn => turn.command === 'switch 3, move 2'));
    } finally { battle.destroy(); }
});

test('unrevealed reserve builds and real bench order are not read by reconstruction', () => {
    const spec = {...opposingSwitchSpecs()[0], turns: opposingSwitchSpecs()[0].turns.slice(0, 1)};
    const a = opposingSwitchFixture(spec), b = opposingSwitchFixture({...spec, seed: [900, 1, 2, 3], changeTeams: teams => {
        Object.assign(teams[1][2], {ability: 'Unaware', item: 'Choice Specs',
            moves: ['cosmicpower', 'wish', 'moonlight', 'minimize'], nature: 'Timid',
            evs: {hp: 252, spe: 252, def: 4}});
    }});
    try {
        assert.deepEqual(a.history, b.history);
        const first = constructMoveHistoryWorld(a.history, WORLD), second = constructMoveHistoryWorld(b.history, WORLD);
        assert(first.supported); assert(second.supported);
        assert.deepEqual(comparable(first.position), comparable(second.position));
        assert.equal(first.nodes, second.nodes);
        assert.deepEqual(observeBeliefs(a.history.observations.at(-1)).opponent.pokemon
            .find(mon => mon.species === 'clefable').moves, []);
    } finally { a.battle.destroy(); b.battle.destroy(); }
});

test('historical reserve conditioning does not expose later entrants to the opening actor', () => {
    const {battle, history} = opposingSwitchFixture({...opposingSwitchSpecs()[0], turns: opposingSwitchSpecs()[0].turns.slice(0, 1)});
    try {
        const initial = history.observations[0], latest = history.observations.at(-1);
        const replay = constructReplayOpening(initial, latest, {...WORLD, attempts: 32});
        assert(replay.supported);
        assert(replay.opponent_hypotheses.some(mon => mon.key === 'p2: Clefable'));
        assert.equal(observeBeliefs(initial).opponent.pokemon.length, 2);
        assert.equal(observeBeliefs(latest).opponent.pokemon.length, 3);
        const steps = replaySteps(history);
        assert.deepEqual(steps.steps[0].opponent_actions[0], {switch_ident: 'p2: Clefable'});
    } finally { battle.destroy(); }
});

test('impossible opposing switches and forced replacements abstain before simulation', () => {
    const {battle, history} = opposingSwitchFixture({...opposingSwitchSpecs()[0], turns: opposingSwitchSpecs()[0].turns.slice(0, 1)});
    try {
        const active = structuredClone(history);
        active.observations[1].log = active.observations[1].log.map(line =>
            line.startsWith('|switch|p2a: Clefable|') ? line.replace('p2a: Clefable|Clefable, M', 'p2a: Snorlax|Snorlax, M') : line);
        const forced = structuredClone(history);
        forced.observations[1].log = forced.observations[1].log.map(line => line.startsWith('|switch|p2a:') ?
            line.replace('|switch|', '|drag|') : line);
        for (const input of [active, forced]) {
            const result = constructMoveHistoryWorld(input, WORLD);
            assert.equal(result.supported, false); assert.equal(result.position, null); assert.equal(result.nodes, 0);
        }
    } finally { battle.destroy(); }
});

test('a sampled opposing trap probe requires retry handling rather than an invented response value', async () => {
    const {battle, history} = historyFixture({changeTeams: teams => {
        Object.assign(teams[0][0], {species: 'Magnezone', ability: 'Magnet Pull', gender: '', moves: ['tackle', 'protect']});
        Object.assign(teams[1][1], {species: 'Metagross', ability: 'Clear Body', gender: '', moves: ['tackle', 'protect']});
    }});
    try {
        const opponentView = playerObservation(battle, 'p2');
        assert.equal(opponentView.request.active[1].maybeTrapped, true);
        assert.notEqual(opponentView.request.active[1].trapped, true);
        assert(ownSwitchActions(opponentView).turns.some(turn => turn.command === 'move 2, switch 3'));
        const result = await compareSwitchTurns(history.observations.at(-1),
            [{position: capturePosition(battle, 'fixture'), responses: ['move 2, switch 3']}],
            ['move 2, move 2'], SEARCH);
        assert.equal(result.supported, false); assert.equal(result.selected_command, null);
        assert.equal(result.reason, 'opponent_response_requires_retry'); assert.equal(result.nodes, 0);
        assert.deepEqual(playerObservation(battle, 'p2'), opponentView);
    } finally { battle.destroy(); }
});

test('newly revealed nicknames cannot collide with synthetic identities for remaining reserves', () => {
    const {battle, history} = opposingSwitchFixture({turns: opposingSwitchSpecs()[0].turns.slice(0, 1), changeTeams: teams => {
        teams[1][2].name = 'Unrevealed 1';
        teams[1].push({...structuredClone(teams[1][2]), species: 'Slowbro', ability: 'Own Tempo', name: 'Slowbro',
            moves: ['watergun', 'protect'], teraType: 'Water'});
    }});
    try {
        const opening = constructReplayOpening(history.observations[0], history.observations.at(-1), {...WORLD, attempts: 32});
        assert(opening.supported);
        const keys = opening.opponent_hypotheses.map(mon => mon.key);
        assert.equal(new Set(keys).size, 4); assert(keys.includes('p2: Unrevealed 1'));
        assert(keys.includes('p2: Unrevealed 2'));
        const world = constructMoveHistoryWorld(history, WORLD);
        assert(world.supported, JSON.stringify({...world, position: null}));
    } finally { battle.destroy(); }
});
