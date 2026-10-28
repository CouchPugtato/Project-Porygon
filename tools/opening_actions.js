'use strict';

const {isDeepStrictEqual} = require('node:util');
const {Battle} = require('../external/pokemon-showdown/dist/sim/battle');
const {BattleActions} = require('../external/pokemon-showdown/dist/sim/battle-actions');
const {observeBeliefs} = require('./belief_state');
const {restorePosition, playerObservation} = require('./position_simulator');

function openingActions(view) {
    const beliefs = observeBeliefs(view);
    const request = view.request, roster = request?.side?.pokemon;
    const active = roster?.filter(mon => mon.active);
    if (view.turn !== 1 || view.phase !== 'move' || view.ended || beliefs.issues.length ||
            !Array.isArray(request?.active) || request.active.length !== 2 || active?.length !== 2 ||
            active.some(mon => !/^(\d+)\/\1$/.test(mon.condition) || mon.commanding || mon.reviving) ||
            !beliefs.opponent.active.a || !beliefs.opponent.active.b ||
            view.log.some(line => /^\|(move|cant|faint|drag|swap|replace|detailschange)\|/.test(line)))
        throw new Error('action generation supports initial two-active move requests only');

    const slots = request.active.map((entry, index) => {
        const options = [];
        // Only slot geometry is needed by Showdown's targeting predicate, not
        // an opponent build or a privileged Pokemon object.
        const source = {getLocOf: () => -(index + 1)};
        const geometry = {activePerHalf: 2, gameType: 'doubles'};
        for (const [moveIndex, move] of entry.moves.entries()) {
            if (move.disabled || move.pp === 0) continue;
            const targets = BattleActions.prototype.targetTypeChoices(move.target) ?
                [1, 2, -1, -2].filter(loc => Battle.prototype.validTargetLoc.call(geometry, loc, source, move.target)) : [0];
            for (const target of targets) {
                for (const tera of entry.canTerastallize ? [false, true] : [false]) {
                    options.push({command: `move ${moveIndex + 1}${target ? ` ${target}` : ''}${tera ? ' terastallize' : ''}`,
                        kind: 'move', move: moveIndex + 1, target, tera});
                }
            }
        }
        if (!entry.trapped) roster.forEach((mon, i) => {
            if (!mon.active && !/(^0\b|\bfnt\b)/.test(mon.condition))
                options.push({command: `switch ${i + 1}`, kind: 'switch', switch: i + 1, tera: false});
        });
        return options;
    });
    const turns = [];
    for (const first of slots[0]) for (const second of slots[1]) {
        if (first.tera && second.tera || first.kind === 'switch' && second.kind === 'switch' &&
                first.switch === second.switch) continue;
        turns.push({command: `${first.command}, ${second.command}`, actions: [first, second]});
    }
    if (!turns.length || turns.length > 4096) throw new Error('opening action space outside supported bound');
    return {version: 1, scope: 'initial_move_request', turns,
        slot_counts: slots.map(options => options.length), training_eligible: false};
}

function checkWorldChoice(position, view, command, randomness) {
    const battle = restorePosition(position, randomness);
    try {
        if (!isDeepStrictEqual(playerObservation(battle, view.side), view))
            throw new Error('world does not reproduce the supplied player observation');
        const side = battle.getSide(view.side);
        if (side.isChoiceDone()) throw new Error('world already has a choice for this player');
        // Side.choose validates without committing the turn or sampling outcomes.
        // A fresh branch per check also isolates trap/disabling knowledge updates.
        battle.strictChoices = false;
        const accepted = side.choose(command) && side.isChoiceDone();
        return {accepted, error: accepted ? null : side.choice.error || 'incomplete choice',
            request_updated: !isDeepStrictEqual(side.activeRequest, view.request),
            scope: 'this_hypothetical_world_only'};
    } finally { battle.destroy(); }
}

module.exports = {openingActions, checkWorldChoice};
