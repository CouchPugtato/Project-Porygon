'use strict';

const {Dex} = require('../external/pokemon-showdown/dist/sim/dex');
const {FORMAT} = require('./position_simulator');
const {openingActions} = require('./opening_actions');

const CORE = ['focus_1', 'focus_2', 'defense', 'switch_0', 'switch_1', 'split_targets'];

function describeTurn(view, turn) {
    const dex = Dex.forFormat(FORMAT), features = new Set();
    let friendlyFire = 0, tera = 0, attacks = 0;
    const targets = [];
    for (const [slot, action] of turn.actions.entries()) {
        if (action.kind === 'switch') { features.add(`switch_${slot}`); targets.push([]); continue; }
        const request = view.request.active[slot].moves[action.move - 1];
        const move = dex.moves.get(request.id);
        features.add(`move_${slot}_${action.move}`);
        if (action.tera) { features.add(`tera_${slot}`); ++tera; }
        if (move.stallingMove || ['wideguard', 'quickguard', 'craftyshield', 'matblock'].includes(move.id))
            features.add('defense');
        if (move.category === 'Status') {
            features.add(`status_${slot}`);
            if (action.target < 0 && action.target !== -(slot + 1)) features.add('ally_support');
            targets.push([]);
        } else {
            ++attacks;
            if (action.target < 0 || request.target === 'allAdjacent') ++friendlyFire;
            const spread = ['allAdjacent', 'allAdjacentFoes'].includes(request.target);
            if (spread) features.add('spread');
            targets.push(spread ? [1, 2] : action.target > 0 ? [action.target] : []);
        }
    }
    if (attacks === 2 && !friendlyFire) {
        for (const foe of [1, 2]) if (targets.every(list => list.includes(foe))) features.add(`focus_${foe}`);
        if (targets[0].some(a => targets[1].some(b => a !== b))) features.add('split_targets');
    }
    return {turn, features, preference: -100 * friendlyFire - 10 * tera + attacks};
}

function proposeOpeningTurns(view, count, rng, mode = 'coverage') {
    if (!Number.isInteger(count) || count < 1 || count > 32 || !['coverage', 'uniform'].includes(mode))
        throw new Error('opening proposals require 1..32 choices and coverage or uniform mode');
    const pool = openingActions(view).turns.map(turn => describeTurn(view, turn));
    // Both modes consume the same randomness, keeping world construction matched.
    for (let i = pool.length - 1; i > 0; --i) {
        const j = rng.random(i + 1); [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const selected = [], covered = new Set();
    const add = entry => { selected.push(entry); entry.features.forEach(feature => covered.add(feature)); };
    const limit = Math.min(count, pool.length);
    const exploration = mode === 'coverage' && limit >= 8 && limit < pool.length ? 1 : 0;
    if (mode === 'uniform') pool.slice(0, limit).forEach(add);
    else {
        for (const feature of CORE) {
            if (selected.length >= limit - exploration) break;
            if (covered.has(feature)) continue;
            const choices = pool.filter(entry => entry.features.has(feature) && !selected.includes(entry));
            choices.sort((a, b) => b.preference - a.preference);
            if (choices.length) add(choices[0]);
        }
        while (selected.length < limit - exploration) {
            const remaining = pool.filter(entry => !selected.includes(entry));
            const novelty = entry => [...entry.features].filter(feature => !covered.has(feature)).length;
            remaining.sort((a, b) => novelty(b) - novelty(a) || b.preference - a.preference);
            add(remaining[0]);
        }
        // This slot can select any remaining legal turn, including ally damage.
        if (exploration) add(pool.find(entry => !selected.includes(entry)));
    }
    const available = new Set(pool.flatMap(entry => [...entry.features]));
    return {commands: selected.map(entry => entry.turn.command).sort(), mode, pool_size: pool.length,
        coverage: {available: CORE.filter(feature => available.has(feature)),
            covered: CORE.filter(feature => covered.has(feature)),
            missing: CORE.filter(feature => available.has(feature) && !covered.has(feature)),
            features: [...covered].sort(), exploratory_choices: exploration}};
}

module.exports = {proposeOpeningTurns};
