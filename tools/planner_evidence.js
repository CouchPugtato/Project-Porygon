'use strict';

const {isDeepStrictEqual} = require('node:util');

function mismatchEvidence(actual, expected) {
    if (isDeepStrictEqual(actual, expected)) return {kind: 'matched', distance: 0};
    const tail = Math.min(actual.log.length, expected.log.length);
    let index = 0;
    while (index < tail && actual.log[index] === expected.log[index]) ++index;
    const observed = expected.log[index] || '', simulated = actual.log[index] || '';
    const events = [observed, simulated].map(line => line.split('|')[1]);
    let kind = 'dynamic_state_mismatch';
    if (events.includes('switch')) kind = 'reserve_or_switch_mismatch';
    else if (events.includes('move')) kind = 'move_or_speed_order_mismatch';
    else if (events.some(event => ['-damage', '-heal', '-crit', '-miss'].includes(event))) kind = 'damage_or_chance_mismatch';
    else if (isDeepStrictEqual(actual.log, expected.log)) kind = 'private_request_or_hidden_stats_mismatch';
    // distance guides proposals only and never relaxes exact world acceptance
    let distance = expected.log.length - index + Math.abs(actual.log.length - expected.log.length);
    for (const [i, mon] of (expected.request?.side?.pokemon || []).entries()) {
        const other = actual.request?.side?.pokemon?.[i];
        if (!other) { distance += 1000; continue; }
        const hp = text => Number(/^\d+/.exec(text || '')?.[0] || 0);
        distance += Math.abs(hp(mon.condition) - hp(other.condition)) / 100;
    }
    return {kind, distance: Math.max(0.001, distance), first_log_index: index,
        observed_event: observed, simulated_event: simulated};
}

function recordRejection(diagnostics, kind) {
    if (diagnostics) diagnostics[kind] = (diagnostics[kind] || 0) + 1;
}

function revealedConstraints(battle, view) {
    const {observeBeliefs} = require('./belief_state');
    const facts = observeBeliefs(view), side = battle.getSide(facts.opponent.side);
    for (const known of facts.opponent.pokemon) {
        const mon = side.pokemon.find(entry => entry.fullname === known.key);
        if (!mon) return 'newly_revealed_reserve_missing';
        if (known.moves.some(move => !mon.baseMoveSlots.some(slot => slot.id === move))) return 'revealed_move_missing';
        if (known.tera_type && mon.set.teraType.toLowerCase() !== known.tera_type) return 'revealed_tera_mismatch';
    }
    return null;
}

module.exports = {mismatchEvidence, recordRejection, revealedConstraints};
