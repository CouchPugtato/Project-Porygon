'use strict';

const assert = require('node:assert/strict');
const {Dex} = require('../../external/pokemon-showdown/dist/sim/dex');
const {TeamValidator} = require('../../external/pokemon-showdown/dist/sim/team-validator');
const {FORMAT} = require('../position_simulator');
const {createFixture} = require('./position_simulator');

function probeBattle(changes) {
    return createFixture({changeTeams: teams => {
        const put = (side, slot, species, ability, moves) => Object.assign(teams[side][slot],
            {species, ability, moves, gender: Dex.species.get(species).gender || 'M'});
        put(0, 1, 'Blastoise', 'Torrent', ['protect']);
        put(1, 0, 'Snorlax', 'Immunity', ['amnesia']);
        put(1, 1, 'Bronzong', 'Levitate', ['irondefense']);
        changes(put);
        // deliberate stat neutral investment for the native intent check
        teams.flat().forEach(mon => { mon.evs = {hp: 1}; });
    }});
}

function assertLegalTeams(battle) {
    const format = new TeamValidator(FORMAT).format;
    const validator = new TeamValidator({...format, team: undefined});
    for (const side of battle.sides) {
        const problems = validator.validateTeam(side.pokemon.map(mon => structuredClone(mon.set)));
        assert.equal(problems, null, JSON.stringify(problems));
    }
}

// mechanics probes authored after the proposal comparison
// declared expectations frozen before running this suite
function tacticalProbes() {
    return [
        {name: 'ground_immunity', expectation: 'strict_preference',
            verify_choice: 'alternative',
            preferred: 'move 1 2, move 1', alternative: 'move 1 1, move 1', response: 'move 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Pikachu', 'Static', ['thunderbolt']);
                put(1, 0, 'Gastrodon', 'Sticky Hold', ['recover']);
            }), verify: battle => assert(battle.log.some(line => line.startsWith('|-immune|p2a:')))},
        {name: 'protected_target', expectation: 'strict_preference',
            verify_choice: 'alternative',
            preferred: 'move 1 2, move 1', alternative: 'move 1 1, move 1', response: 'move 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Blastoise', 'Torrent', ['watergun']);
                put(0, 1, 'Pikachu', 'Static', ['protect']);
                put(1, 0, 'Talonflame', 'Gale Wings', ['protect']);
            }), verify: battle => assert(battle.log.some(line => line.startsWith('|-activate|p2a:') && line.includes('Protect')))},
        {name: 'follow_me_redirects_both_targets', expectation: 'equivalent',
            preferred: 'move 1 1, move 1', alternative: 'move 1 2, move 1', response: 'move 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Blastoise', 'Torrent', ['watergun']);
                put(0, 1, 'Pikachu', 'Static', ['protect']);
                put(1, 0, 'Maushold', 'Friend Guard', ['followme']);
                put(1, 1, 'Coalossal', 'Flame Body', ['irondefense']);
            }), verify: battle => {
                assert(battle.p2.active[0].hp < battle.p2.active[0].maxhp);
                assert.equal(battle.p2.active[1].hp, battle.p2.active[1].maxhp);
            }},
        {name: 'priority_flinch_prevents_faster_knockout', expectation: 'strict_preference',
            preferred: 'move 1 1, move 1', alternative: 'move 2 1, move 1', response: 'move 1 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Weavile', 'Pressure', ['fakeout', 'iciclecrash']);
                put(1, 0, 'Chien-Pao', 'Sword of Ruin', ['sacredsword']);
            }), verify: battle => {
                assert(battle.log.some(line => line.startsWith('|cant|p2a:') && line.includes('flinch')));
                assert(battle.p1.active[0].hp > 0);
            }},
        {name: 'switch_into_fire_immunity', expectation: 'strict_preference',
            preferred: 'switch 3, move 1', alternative: 'move 1 1, move 1', response: 'move 1 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Venusaur', 'Overgrow', ['gigadrain']);
                put(0, 2, 'Arcanine', 'Flash Fire', ['protect']);
                put(1, 0, 'Charizard', 'Blaze', ['flamethrower']);
            }), verify: battle => {
                assert.equal(battle.p1.active[0].species.id, 'arcanine');
                assert.equal(battle.p1.active[0].hp, battle.p1.active[0].maxhp);
                assert(battle.log.some(line => line.includes('Flash Fire')));
            }},
        {name: 'protect_partner_from_spread_damage', expectation: 'strict_preference',
            preferred: 'move 1, move 1', alternative: 'move 1, move 2', response: 'move 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Garchomp', 'Rough Skin', ['earthquake']);
                // fragile partner for an unambiguous friendly knockout cost
                put(0, 1, 'Azumarill', 'Huge Power', ['protect', 'splash']).level = 20;
                put(1, 0, 'Coalossal', 'Flame Body', ['irondefense']);
                put(1, 1, 'Kingambit', 'Defiant', ['swordsdance']);
            }), verify: battle => assert.equal(battle.p1.active[1].hp, battle.p1.active[1].maxhp)},
        {name: 'tailwind_has_unscored_future_value', expectation: 'unscored_benefit',
            preferred: 'move 1, move 1', alternative: 'move 2, move 1', response: 'move 1, move 1',
            create: () => probeBattle(put => put(0, 0, 'Tornadus', 'Prankster', ['tailwind', 'protect'])),
            verify: battle => assert(battle.p1.sideConditions.tailwind)},
        {name: 'pivot_requires_another_choice', expectation: 'continuation_gap',
            preferred: 'move 1 1, move 1', alternative: 'move 2, move 1', response: 'move 1, move 1',
            create: () => probeBattle(put => {
                put(0, 0, 'Blastoise', 'Torrent', ['flipturn', 'protect']);
                put(0, 1, 'Pikachu', 'Static', ['protect']);
            }), verify: battle => {
                assert.equal(battle.turn, 1); assert.equal(battle.requestState, 'switch');
                assert(battle.queue.list.length > 0);
            }},
    ];
}

module.exports = {tacticalProbes, assertLegalTeams};
