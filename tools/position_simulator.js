'use strict';

const {Battle, extractChannelMessages} = require('../external/pokemon-showdown/dist/sim/battle');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');

const FORMAT = 'gen9randomdoublesbattle';
const SIDES = ['p1', 'p2'];
const ORIGINS = ['fixture', 'belief-sample'];

function checkBattle(battle) {
    if (battle.format.id !== FORMAT || battle.gameType !== 'doubles' ||
            battle.sides.length !== 2 || battle.sides.some(side => !side))
        throw new Error('position adapter supports started Gen 9 Random Doubles only');
    if (battle.teamGenerator) throw new Error('position fixtures require explicit teams, not a set generator');
}

function capturePosition(battle, origin) {
    checkBattle(battle);
    if (!ORIGINS.includes(origin)) throw new Error('position origin must be fixture or belief-sample');
    // Showdown serializes some live arrays/objects by reference. A string also
    // prevents fromJSON from sharing its restored log or Pokemon sets with us.
    const state = battle.toJSON();
    // fromJSON normally regenerates requests, dropping same-phase update flags
    // and potentially changing knowledge/trap caches. Preserve the actual views.
    state.sides.forEach((side, i) => { side.activeRequest = battle.sides[i].activeRequest; });
    return Object.freeze({version: 1, scope: 'hypothetical-complete-state', origin,
        format: FORMAT, state_json: JSON.stringify(state)});
}

function restorePosition(position, randomness) {
    if (position?.version !== 1 || position.scope !== 'hypothetical-complete-state' ||
            position.format !== FORMAT || !ORIGINS.includes(position.origin) ||
            typeof position.state_json !== 'string')
        throw new Error('unsupported hypothetical position');
    if (randomness?.mode === 'fixture-replay') {
        if (position.origin !== 'fixture' || Object.hasOwn(randomness, 'seed'))
            throw new Error('saved RNG replay is restricted to fixtures');
    } else if (randomness?.mode === 'sample') {
        if (!Array.isArray(randomness.seed) || randomness.seed.length !== 4 ||
                randomness.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535))
            throw new Error('sample mode requires an independent four-word seed');
    } else {
        throw new Error('choose fixture-replay or independently seeded sample mode');
    }
    const state = JSON.parse(position.state_json);
    if (state.formatid !== position.format) throw new Error('position format mismatch');
    const battle = Battle.fromJSON(state);
    try {
        battle.restart(() => {});
        state.sides.forEach((side, i) => { battle.sides[i].activeRequest = side.activeRequest; });
        checkBattle(battle);
        if (randomness.mode === 'sample') battle.prng = new PRNG(randomness.seed.join(','));
        return battle;
    } catch (error) {
        battle.destroy();
        throw error;
    }
}

function playerObservation(battle, side) {
    if (!SIDES.includes(side)) throw new Error('observation side must be p1 or p2');
    return {side, turn: battle.turn, phase: battle.ended ? 'ended' : battle.requestState,
        ended: battle.ended, winner: battle.ended ? battle.winner : null,
        request: structuredClone(battle.getSide(side).activeRequest),
        log: extractChannelMessages(battle.log.join('\n'), [Number(side[1])])[Number(side[1])]
            .map(line => line.startsWith('|t:|') ? '|t:|' : line)};
}

function pendingSides(battle) {
    return battle.sides.filter(side => side.requestState && !side.isChoiceDone()).map(side => side.id);
}

function advancePosition(position, choices, randomness) {
    const battle = restorePosition(position, randomness);
    try {
        if (battle.ended) throw new Error('cannot advance a terminal position');
        const required = pendingSides(battle);
        if (!required.length || !choices || Array.isArray(choices) ||
                Object.keys(choices).length !== required.length ||
                required.some(side => !Object.hasOwn(choices, side) ||
                    typeof choices[side] !== 'string' || !choices[side].trim()))
            throw new Error(`supply complete choices for exactly: ${required.join(', ')}`);
        // Both inputs are fixed before either is submitted. Invalid or incomplete
        // choices only affect this disposable branch, never the caller's state.
        for (const side of required) {
            if (!battle.choose(side, choices[side]))
                throw new Error(`${side}: ${battle.getSide(side).choice.error || 'incomplete choice'}`);
        }
        return {position: capturePosition(battle, position.origin),
            observations: Object.fromEntries(SIDES.map(side => [side, playerObservation(battle, side)])),
            pending_sides: pendingSides(battle), turn: battle.turn,
            phase: battle.ended ? 'ended' : battle.requestState,
            ended: battle.ended, winner: battle.ended ? battle.winner : null};
    } finally {
        battle.destroy();
    }
}

module.exports = {FORMAT, capturePosition, restorePosition, playerObservation, pendingSides, advancePosition};
