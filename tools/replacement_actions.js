'use strict';

const {observeBeliefs} = require('./belief_state');

function replacementActions(view) {
    const beliefs = observeBeliefs(view), request = view.request;
    if (view.ended || view.phase !== 'switch' || beliefs.issues.length || request?.wait)
        throw new Error('replacement actions require a pending replacement request');
    const forced = request?.forceSwitch, roster = request?.side?.pokemon;
    const active = roster?.filter(mon => mon.active);
    if (!Array.isArray(forced) || forced.length !== 2 || !forced.some(Boolean) || active?.length !== 2 ||
            forced.some((flag, slot) => typeof flag !== 'boolean' || flag && active[slot].condition !== '0 fnt') ||
            roster.some(mon => mon.reviving || mon.commanding))
        throw new Error('only fainted slot replacement is supported without pivots or revival');
    const reserves = roster.flatMap((mon, i) => !mon.active && !/(^0\b|\bfnt\b)/.test(mon.condition) ? [i + 1] : []);
    const needed = Math.min(reserves.length, forced.filter(Boolean).length), turns = [];
    const visit = (slot, actions, used) => {
        if (slot === 2) {
            if (used.length === needed) turns.push({command: actions.map(action => action.command).join(', '), actions});
            return;
        }
        // unused fainted slots need explicit passes when reserves run out
        if (!forced[slot] || reserves.length < forced.filter(Boolean).length)
            visit(slot + 1, [...actions, {command: 'pass', kind: 'pass', tera: false}], used);
        if (forced[slot]) for (const destination of reserves) {
            if (!used.includes(destination)) visit(slot + 1,
                [...actions, {command: `switch ${destination}`, kind: 'switch', switch: destination, tera: false}],
                [...used, destination]);
        }
    };
    visit(0, [], []);
    if (!turns.length || needed === 0) throw new Error('replacement request has no living reserve');
    return {version: 1, scope: 'post_knockout_replacement', turns, training_eligible: false};
}

function isPostKnockoutReplacement(battle) {
    return !battle.ended && battle.requestState === 'switch' && battle.queue.list.length === 0 &&
        battle.sides.some(side => side.activeRequest?.forceSwitch?.some(Boolean)) &&
        battle.sides.every(side => !side.activeRequest?.forceSwitch?.some((flag, slot) => flag && !side.active[slot].fainted));
}

module.exports = {replacementActions, isPostKnockoutReplacement};
