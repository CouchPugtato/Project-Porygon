'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const {aggregateGames, closeGame} = require('./tactical_teacher_smoke');

function game(overrides = {}) {
    return {status: 'completed', coverage: {
        policy_tag: 'tactical-eval-v1', training_eligible: false,
        tactical_attempts: 10, tactical_applied: 6, tactical_forced_switches: 2,
        tactical_fallbacks: 4, ...overrides,
    }};
}

test('coverage counts decisions rather than averaging game percentages', () => {
    const result = aggregateGames([game(), game({tactical_attempts: 20, tactical_applied: 2,
        tactical_forced_switches: 0, tactical_fallbacks: 18}), {status: 'failed'}]);
    assert.equal(result.coverage_rate, 8 / 30);
    assert.equal(result.power_decisions, 6);
    assert.equal(result.completed, 2);
    assert.equal(result.failed, 1);
});

test('ordinary policy episodes and inconsistent counters cannot count as teacher evidence', () => {
    assert.throws(() => aggregateGames([game({policy_tag: 'parent.chk'})]), /evaluation-only/);
    assert.throws(() => aggregateGames([game({training_eligible: true})]), /evaluation-only/);
    assert.throws(() => aggregateGames([game({tactical_fallbacks: 5})]), /inconsistent/);
    assert.throws(() => aggregateGames([game({tactical_attempts: NaN})]), /invalid coverage/);
});

test('cleanup closes an already-ended Showdown stream without destroying it twice', async () => {
    const sim = require('../external/pokemon-showdown/dist/sim');
    const battle = new sim.BattleStream();
    const streams = sim.getPlayerStreams(battle);
    battle.pushEnd();
    await closeGame(streams, []);
    assert.equal(battle.atEOF, true);
});

test('cleanup failures still close every agent and remain reportable', async () => {
    let closed = 0;
    const agents = [0, 1].map(() => ({terminate() {}, async close() { ++closed; }}));
    await assert.rejects(closeGame({omniscient: {writeEnd() { throw new Error('cleanup fixture'); }}}, agents),
        error => error instanceof AggregateError && error.errors[0].message === 'cleanup fixture');
    assert.equal(closed, 2);
});

function requestChoiceBattle(team) {
    const {Battle} = require('../external/pokemon-showdown/dist/sim');
    const battle = new Battle({formatid: 'gen9doublescustomgame', seed: [1, 2, 3, 4],
        p1: {name: 'RequestFixture', team}, p2: {name: 'Opponent', team: [
            {species: 'Magikarp', moves: ['splash']}, {species: 'Magikarp', moves: ['splash']}]} });
    if (battle.requestState === 'teampreview')
        battle.makeChoices(`team ${team.map((_, index) => index + 1).join('')}`, 'team 12');
    battle.p1.active[0].fainted = true;
    battle.p1.active[0].hp = 0;
    battle.p1.pokemonLeft = team.length - 1;
    return battle;
}

test('Showdown accepts the lone-survivor fallback at a fainted ally location', () => {
    const battle = requestChoiceBattle([{species: 'Deoxys-Speed', moves: ['superpower']},
        {species: 'Cresselia', moves: ['lunarblessing', 'helpinghand', 'psychic', 'icywind']}]);
    try {
        battle.p1.active[1].moveSlots.forEach((move, index) => { if (index !== 1) move.pp = 0; });
        battle.makeRequest('move');
        assert.equal(battle.p1.choose('move 2 -1'), true);
        assert.equal(battle.p1.choice.error, '');
        assert.deepEqual(battle.p1.choice.actions.map(action => action.choice), ['pass', 'move']);
        assert.equal(battle.p1.choice.actions[1].moveid, 'helpinghand');
        assert.equal(battle.p1.choice.actions[1].targetLoc, -1);
    } finally { battle.destroy(); }
});

test('Showdown revival selection can name a fainted active partner', () => {
    const battle = requestChoiceBattle([{species: 'Klawf', moves: ['rockslide']},
        {species: 'Rabsca', moves: ['revivalblessing']}, {species: 'Mew', moves: ['psychic']}]);
    try {
        battle.p1.slotConditions[1].revivalblessing = {id: 'revivalblessing', target: battle.p1.active[1]};
        battle.p1.active[0].switchFlag = false;
        battle.p1.active[1].switchFlag = true;
        battle.makeRequest('switch');
        assert.deepEqual(battle.p1.activeRequest.forceSwitch, [false, true]);
        assert.equal(battle.p1.choose('pass, switch 1'), true);
        assert.equal(battle.p1.choice.error, '');
        assert.deepEqual(battle.p1.choice.actions.map(action => action.choice), ['pass', 'revivalblessing']);
        assert.equal(battle.p1.choice.actions[1].target, battle.p1.active[0]);
    } finally { battle.destroy(); }
});
