'use strict';

const {isDeepStrictEqual} = require('node:util');
const {Battle} = require('../external/pokemon-showdown/dist/sim/battle');
const {BattleActions} = require('../external/pokemon-showdown/dist/sim/battle-actions');
const {observeBeliefs} = require('./belief_state');
const {restorePosition, playerObservation} = require('./position_simulator');
const {isOrdinaryMove} = require('./ordinary_moves');

function moveRequestActions(view, openingOnly, allowSwitches = openingOnly) {
    const beliefs = observeBeliefs(view);
    const request = view.request, roster = request?.side?.pokemon;
    const active = roster?.filter(mon => mon.active);
    if (!Number.isInteger(view.turn) || view.turn < 1 || openingOnly && view.turn !== 1 ||
            view.phase !== 'move' || view.ended || beliefs.issues.length ||
            !Array.isArray(request?.active) || request.active.length !== 2 || active?.length !== 2 ||
            active.some(mon => (openingOnly ? !/^(\d+)\/\1$/.test(mon.condition) :
                !/^(?:[1-9]\d*\/[1-9]\d*(?: (?:brn|par|psn|tox|slp|frz))?|0 fnt)$/.test(mon.condition)) ||
                mon.commanding || mon.reviving) ||
            !beliefs.opponent.active.a || !beliefs.opponent.active.b ||
            openingOnly && view.log.some(line => /^\|(move|cant|faint|drag|swap|replace|detailschange)\|/.test(line)))
        throw new Error(openingOnly ? 'action generation supports initial two-active move requests only' :
            'ordinary action generation requires a supported two-slot move request');

    const slots = request.active.map((entry, index) => {
        if (!openingOnly && active[index].condition === '0 fnt')
            return [{command: 'pass', kind: 'pass', tera: false}];
        const options = [];
        // slot geometry without a privileged opponent object
        const source = {getLocOf: () => -(index + 1)};
        const geometry = {activePerHalf: 2, gameType: 'doubles'};
        for (const [moveIndex, move] of entry.moves.entries()) {
            if (move.disabled || move.pp === 0 || !openingOnly && !isOrdinaryMove(move.id)) continue;
            const targets = BattleActions.prototype.targetTypeChoices(move.target) ?
                [1, 2, -1, -2].filter(loc => Battle.prototype.validTargetLoc.call(geometry, loc, source, move.target) &&
                    (openingOnly || loc > 0 || active[-loc - 1].condition !== '0 fnt')) : [0];
            for (const target of targets) {
                for (const tera of openingOnly && entry.canTerastallize ? [false, true] : [false]) {
                    options.push({command: `move ${moveIndex + 1}${target ? ` ${target}` : ''}${tera ? ' terastallize' : ''}`,
                        kind: 'move', move: moveIndex + 1, target, tera});
                }
            }
        }
        if (allowSwitches && !entry.trapped) roster.forEach((mon, i) => {
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
    if (!turns.length || turns.length > 4096) throw new Error('move action space outside supported bound');
    return {version: 1, scope: openingOnly ? 'initial_move_request' : 'ordinary_move_request', turns,
        slot_counts: slots.map(options => options.length), training_eligible: false};
}

function checkWorldChoice(position, view, command, randomness) {
    const battle = restorePosition(position, randomness);
    try {
        if (!isDeepStrictEqual(playerObservation(battle, view.side), view))
            throw new Error('world does not reproduce the supplied player observation');
        const side = battle.getSide(view.side);
        if (side.isChoiceDone()) throw new Error('world already has a choice for this player');
        // native choice validation without committing or sampling a turn
        // fresh branches to isolate trap and disabling knowledge updates
        battle.strictChoices = false;
        const accepted = side.choose(command) && side.isChoiceDone();
        return {accepted, error: accepted ? null : side.choice.error || 'incomplete choice',
            request_updated: !isDeepStrictEqual(side.activeRequest, view.request),
            scope: 'this_hypothetical_world_only'};
    } finally { battle.destroy(); }
}

function openingActions(view) { return moveRequestActions(view, true); }
function ordinaryActions(view) { return moveRequestActions(view, false); }
function ownSwitchActions(view) { return moveRequestActions(view, false, true); }

module.exports = {openingActions, ordinaryActions, ownSwitchActions, checkWorldChoice};
