'use strict';

const assert = require('node:assert/strict');
const {Battle} = require('../../external/pokemon-showdown/dist/sim/battle');
const {State} = require('../../external/pokemon-showdown/dist/sim/state');
const {Dex} = require('../../external/pokemon-showdown/dist/sim/dex');
const {FORMAT, capturePosition, restorePosition, playerObservation, advancePosition} = require('../position_simulator');

const REPLAY = Object.freeze({mode: 'fixture-replay'});
const SELF_BUFFS = 'move 3, move 3';

// Hand-authored mechanics fixtures, not competitive builds or sampled set lists.
function teams() {
    const pokemon = (species, ability, moves, teraType) => ({species, ability, moves,
        teraType, level: 100, gender: Dex.species.get(species).gender || 'M', nature: 'Serious', item: ''});
    return [
        [pokemon('Pikachu', 'Static', ['thunderbolt', 'thunder', 'helpinghand', 'protect'], 'Electric'),
            pokemon('Blastoise', 'Torrent', ['watergun', 'protect', 'surf', 'flipturn'], 'Water'),
            pokemon('Venusaur', 'Overgrow', ['tackle', 'protect', 'sleeppowder', 'gigadrain'], 'Grass')],
        [pokemon('Snorlax', 'Immunity', ['tackle', 'protect', 'amnesia', 'rest'], 'Normal'),
            pokemon('Bronzong', 'Levitate', ['tackle', 'protect', 'irondefense', 'trickroom'], 'Steel'),
            pokemon('Clefable', 'Magic Guard', ['moonblast', 'protect', 'helpinghand', 'thunderwave'], 'Fairy')],
    ];
}

function sixPokemon(selected) {
    const [a, b] = teams(); selected[0].push(...b); selected[1].push(...a);
}

function createFixture({seed = [322, 1, 0, 1], changeTeams, prepare} = {}) {
    const selected = teams();
    if (changeTeams) changeTeams(selected);
    const battle = new Battle({formatid: FORMAT, seed: seed.join(','), strictChoices: true,
        p1: {name: 'FixtureA', team: selected[0]}, p2: {name: 'FixtureB', team: selected[1]}});
    try {
        assert.equal(battle.requestState, 'move');
        assert.equal(battle.teamGenerator, null);
        if (prepare) prepare(battle);
        return battle;
    } catch (error) { battle.destroy(); throw error; }
}

function comparable(position) {
    return State.normalize(JSON.parse(position.state_json));
}

function summary(position) {
    const battle = restorePosition(position, REPLAY);
    try {
        return {turn: battle.turn, phase: battle.requestState, queue_length: battle.queue.list.length,
            active: battle.sides.map(side => side.active.map(p => p && ({species: p.species.id,
                hp: p.hp, maxhp: p.maxhp, tera: p.terastallized,
                pp: p.moveSlots.map(move => move.pp)})))};
    } finally { battle.destroy(); }
}

function openingFixtureChoices(position, seed) {
    const battle = restorePosition(position, {mode: 'sample', seed});
    try {
        assert.equal(battle.turn, 1);
        assert.equal(battle.requestState, 'move');
        const choices = {};
        for (const side of battle.sides) {
            side.autoChoose();
            choices[side.id] = side.getChoice().split(',').map((command, index) => {
                const action = side.choice.actions[index];
                const request = side.activeRequest.active[index].moves.find(m => m.id === action.moveid);
                if (action.choice !== 'move' || action.targetLoc ||
                        !['normal', 'any', 'adjacentFoe', 'adjacentAlly', 'adjacentAllyOrSelf'].includes(request?.target))
                    return command.trim();
                const target = [1, 2, -1, -2].find(loc => battle.validTargetLoc(loc, action.pokemon, request.target));
                assert(target !== undefined, 'fixture auto-choice has no legal target');
                return `${command.trim()} ${target}`;
            }).join(', ');
        }
        return choices;
    } finally { battle.destroy(); }
}

function checkReplay({name, setup = {}, choices, verify}) {
    const battle = createFixture(setup);
    let restored;
    try {
        const parent = capturePosition(battle, 'fixture');
        const original = parent.state_json;
        const observations = ['p1', 'p2'].map(side => playerObservation(battle, side));
        restored = restorePosition(parent, REPLAY);
        assert.deepEqual(comparable(capturePosition(restored, 'fixture')), comparable(parent));
        assert.deepEqual(['p1', 'p2'].map(side => playerObservation(restored, side)), observations);
        for (const side of Object.keys(choices)) assert.equal(battle.choose(side, choices[side]), true);
        const expected = capturePosition(battle, 'fixture');
        const first = advancePosition(parent, choices, REPLAY);
        const second = advancePosition(parent, choices, REPLAY);
        assert.deepEqual(comparable(first.position), comparable(expected));
        assert.deepEqual(comparable(second.position), comparable(first.position));
        assert.deepEqual(first.observations,
            Object.fromEntries(['p1', 'p2'].map(side => [side, playerObservation(battle, side)])));
        assert.equal(parent.state_json, original);
        assert.deepEqual(comparable(capturePosition(restored, 'fixture')), comparable(parent));
        if (verify) verify(first, parent);
        return {name, passed: true, turn: first.turn, phase: first.phase,
            snapshot_bytes: Buffer.byteLength(parent.state_json)};
    } finally { restored?.destroy(); battle.destroy(); }
}

const scenarios = [
    {name: 'targeted_damage_and_move_pp',
        choices: {p1: 'move 1 2, move 2', p2: SELF_BUFFS},
        verify(result, parent) {
            const before = summary(parent), after = summary(result.position);
            assert.equal(after.active[1][0].hp, before.active[1][0].hp);
            assert(after.active[1][1].hp < before.active[1][1].hp);
            assert.equal(after.active[0][0].pp[0], before.active[0][0].pp[0] - 1);
        }},
    {name: 'protect_blocks_both_attacks',
        choices: {p1: 'move 4, move 2', p2: 'move 1 1, move 1 2'},
        verify(result, parent) {
            assert.deepEqual(summary(result.position).active[0].map(p => p.hp),
                summary(parent).active[0].map(p => p.hp));
        }},
    {name: 'helping_hand_explicit_ally_target',
        choices: {p1: 'move 3 -2, move 1 1', p2: SELF_BUFFS},
        verify(result) {
            assert(result.observations.p1.log.some(line => line.includes('|Helping Hand|p1b: Blastoise')));
        }},
    {name: 'spread_move_damages_partner_and_opponents',
        choices: {p1: 'move 3 -2, move 3', p2: SELF_BUFFS},
        verify(result, parent) {
            const before = summary(parent).active, after = summary(result.position).active;
            assert(after[0][0].hp < before[0][0].hp);
            assert(after[1][0].hp < before[1][0].hp);
            assert(after[1][1].hp < before[1][1].hp);
        }},
    {name: 'tera_and_priority_before_speed',
        choices: {p1: 'move 1 1 terastallize, move 2', p2: SELF_BUFFS},
        verify(result) {
            assert.equal(summary(result.position).active[0][0].tera, 'Electric');
            const moves = result.observations.p1.log.filter(line => line.startsWith('|move|'));
            assert(moves[0].includes('|Protect|'));
            assert(moves[1].includes('|Thunderbolt|'));
        }},
    {name: 'voluntary_switch_changes_roster_order',
        choices: {p1: 'switch 3, move 2', p2: SELF_BUFFS},
        verify(result) {
            assert.equal(summary(result.position).active[0][0].species, 'venusaur');
        }},
    {name: 'faint_then_forced_replacement',
        setup: {prepare: battle => { battle.p1.active[0].hp = 1; }},
        choices: {p1: 'move 3 -2, move 2', p2: 'move 1 1, move 3'},
        verify(result) {
            assert.equal(result.phase, 'switch');
            assert.deepEqual(result.pending_sides, ['p1']);
            const next = advancePosition(result.position, {p1: 'switch 3, pass'}, REPLAY);
            assert.equal(next.phase, 'move');
            assert.equal(summary(next.position).active[0][0].species, 'venusaur');
        }},
    {name: 'pivot_restores_midturn_queue',
        choices: {p1: 'move 4, move 4 1', p2: 'move 1 2, move 3'},
        verify(result) {
            assert.equal(result.phase, 'switch');
            assert(summary(result.position).queue_length > 0);
            const next = advancePosition(result.position, {p1: 'pass, switch 3'}, REPLAY);
            assert.equal(next.phase, 'move');
            assert.equal(summary(next.position).active[0][1].species, 'venusaur');
            const live = restorePosition(result.position, REPLAY);
            try {
                assert(live.choose('p1', 'pass, switch 3'));
                assert.deepEqual(comparable(next.position), comparable(capturePosition(live, 'fixture')));
            } finally { live.destroy(); }
        }},
    {name: 'pending_first_player_choice_survives_restore',
        setup: {prepare: battle => { assert(battle.choose('p1', 'move 1 1, move 2')); }},
        choices: {p2: SELF_BUFFS}},
    {name: 'single_surviving_active_slot',
        setup: {changeTeams: selected => { selected[0].length = 2; },
            prepare: battle => {
                battle.p1.active[0].hp = 1;
                assert(battle.choose('p1', 'move 3 -2, move 2'));
                assert(battle.choose('p2', 'move 1 1, move 3'));
            }},
        choices: {p1: 'pass, move 1 1', p2: SELF_BUFFS}},
    {name: 'terminal_result',
        setup: {changeTeams: selected => {
            selected[1].length = 2;
            selected[0][0].moves[0] = 'discharge';
        }, prepare: battle => { for (const p of battle.p2.active) p.hp = 1; }},
        choices: {p1: 'move 1, move 2', p2: SELF_BUFFS},
        verify(result) { assert.equal(result.ended, true); assert.equal(result.winner, 'FixtureA'); }},
];

module.exports = {REPLAY, SELF_BUFFS, teams, createFixture, comparable, summary, checkReplay, scenarios,
    openingFixtureChoices, sixPokemon};
