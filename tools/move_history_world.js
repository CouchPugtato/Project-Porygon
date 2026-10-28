'use strict';

const {isDeepStrictEqual} = require('node:util');
const {performance} = require('node:perf_hooks');
const {createHash} = require('node:crypto');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {Dex, toID} = require('../external/pokemon-showdown/dist/sim/dex');
const {BattleActions} = require('../external/pokemon-showdown/dist/sim/battle-actions');
const {FORMAT, advancePosition, restorePosition, pendingSides} = require('./position_simulator');
const {observeBeliefs} = require('./belief_state');
const {constructReplayOpening} = require('./opening_world');
const {isOrdinaryMove} = require('./ordinary_moves');
const {replacementActions, isPostKnockoutReplacement} = require('./replacement_actions');

const EVENTS = new Set(['', 't:', 'turn', 'move', '-damage', '-heal', '-status', '-curestatus',
    '-activate', '-singleturn', '-fail', '-immune', '-supereffective', '-resisted', '-crit', '-miss',
    'cant', 'upkeep', 'message', 'switch', 'faint', 'win', 'tie']);

function validateOwnReplacement(previous, own, switches, issue) {
    if (previous.request?.wait) {
        // no synthetic pass command for a player without a pending request
        if (own !== null) issue('waiting_player_must_not_submit_a_choice');
        if (switches.size) issue('unexpected_own_replacement');
        return;
    }
    let pool = [];
    try { pool = replacementActions(previous).turns; } catch { issue('unsupported_replacement_request'); }
    const command = typeof own === 'string' ? own.split(',').map(part => part.trim()).join(', ') : '';
    if (!pool.some(turn => turn.command === command)) issue('invalid_own_replacement');
    for (const [slot, part] of command.split(', ').entries()) {
        if (!part.startsWith('switch ')) continue;
        const mon = previous.request.side.pokemon[Number(part.slice(7)) - 1];
        if (switches.get(slot) !== mon?.ident) issue('own_switch_evidence_mismatch');
        switches.delete(slot);
    }
    if (switches.size) issue('uncommanded_own_switch');
}

function validateOwnMoveTurn(previous, own, switches, faintedOwn, ownMoved, issue) {
    if (typeof own !== 'string' || own.split(',').length !== 2) { issue('complete_own_moves_required'); return; }
    const destinations = new Set(), active = previous.request?.side?.pokemon?.filter(mon => mon.active);
    for (const [slot, command] of own.split(',').entries()) {
        const change = /^switch ([1-6])$/.exec(command.trim());
        if (change) {
            // destinations refer to the current roster rather than opening positions
            const destination = Number(change[1]), mon = previous.request?.side?.pokemon?.[destination - 1];
            if (!mon || mon.active || /(^0\b|\bfnt\b)/.test(mon.condition) ||
                    previous.request?.active?.[slot]?.trapped || destinations.has(destination))
                issue('invalid_own_switch_destination');
            if (!mon || switches.get(slot) !== mon.ident) issue('own_switch_evidence_mismatch');
            destinations.add(destination); switches.delete(slot);
            continue;
        }
        if (command.trim() === 'pass' && active?.[slot]?.condition === '0 fnt') continue;
        const match = /^move ([a-z0-9]+)(?: (-?[12]))?$/.exec(command.trim());
        const moves = previous.request?.active?.[slot]?.moves || [];
        const id = match && (/^\d+$/.test(match[1]) ? moves[Number(match[1]) - 1]?.id : toID(match[1]));
        const cancelled = faintedOwn.has(active?.[slot]?.ident) && !ownMoved.has(slot);
        if (!id || !isOrdinaryMove(id) && !cancelled) issue('unsupported_own_command');
    }
    if (switches.size) issue('uncommanded_own_switch');
}

function replaySteps(history) {
    if (!history || Object.keys(history).length !== 2 || !Array.isArray(history.observations) ||
            !Array.isArray(history.own_choices) || history.observations.length < 2 ||
            history.observations.length > 7 || history.own_choices.length !== history.observations.length - 1)
        throw new Error('supply two to seven observations and one own command or wait marker per boundary');
    const observations = history.observations;
    const beliefs = observations.map(view => observeBeliefs(view));
    const unsupported = [], steps = [], dex = Dex.forFormat(FORMAT);
    const issue = code => { if (!unsupported.includes(code)) unsupported.push(code); };
    const initial = observations[0];
    if (initial.turn !== 1) issue('opening_observation_required');
    const opponent = initial.side === 'p1' ? 'p2' : 'p1';
    for (const [i, view] of observations.entries()) {
        if (view.side !== initial.side || !['move', 'switch', 'ended'].includes(view.phase) ||
                view.turn < 1 || view.turn > 4 || view.ended && i !== observations.length - 1 ||
                view.phase === 'move' && (!Array.isArray(view.request?.active) || view.request.active.length !== 2))
            issue('unsupported_decision_boundary');
        if (view.phase === 'switch' && !view.request?.wait) {
            try { replacementActions(view); } catch { issue('unsupported_pivot_or_replacement_request'); }
        }
        if (beliefs[i].issues.length) issue('unsupported_observation_evidence');
        if (!i) continue;
        const previous = observations[i - 1];
        if (previous.phase === 'ended' || view.turn < previous.turn || view.turn > previous.turn + 1 ||
                previous.phase === 'switch' && view.phase === 'switch') issue('unsupported_boundary_transition');
        if (previous.log.some((line, j) => view.log[j] !== line)) issue('history_prefix_mismatch');
        const segment = view.log.slice(previous.log.length), commands = [null, null], switches = new Map();
        const opposingDestinations = new Set(), faintedOpponents = new Set(), faintedOwn = new Set(), ownMoved = new Set();
        for (const line of segment) {
            const parts = line.split('|'), event = parts[1];
            if (!EVENTS.has(event)) issue(`unsupported_event_${event}`);
            if (event === 'faint' && parts[2]?.startsWith(opponent))
                faintedOpponents.add(parts[2].replace(/^(p[12])[ab]: /, '$1: '));
            if (event === 'faint' && parts[2]?.startsWith(initial.side))
                faintedOwn.add(parts[2].replace(/^(p[12])[ab]: /, '$1: '));
            if (event === 'switch') {
                const actor = /^(p[12])([ab]): (.+)$/.exec(parts[2]);
                if (!actor) issue('ambiguous_switch_identity');
                else if (actor[1] === opponent) {
                    const slot = actor[2] === 'a' ? 0 : 1, ident = `${actor[1]}: ${actor[3]}`;
                    const earlier = beliefs[i - 1].opponent;
                    if (Object.values(earlier.active).includes(ident) || opposingDestinations.has(ident) ||
                            earlier.pokemon.some(mon => mon.key === ident && mon.fainted))
                        issue('invalid_observed_opponent_switch');
                    if (commands[slot]) issue('multiple_actions_for_opponent_slot');
                    commands[slot] = {switch_ident: ident}; opposingDestinations.add(ident);
                }
                else {
                    const slot = actor[2] === 'a' ? 0 : 1;
                    if (switches.has(slot)) issue('multiple_switches_for_own_slot');
                    switches.set(slot, `${actor[1]}: ${actor[3]}`);
                }
            }
            if (event === 'move') {
                const move = toID(parts[3]);
                if (!isOrdinaryMove(move) || parts.some(part => part.startsWith('[from]')))
                    issue('unsupported_move_mechanics');
                const actor = /^(p[12])([ab]): /.exec(parts[2]);
                if (actor?.[1] === initial.side) ownMoved.add(actor[2] === 'a' ? 0 : 1);
                if (actor?.[1] !== opponent) continue;
                const slot = actor[2] === 'a' ? 0 : 1;
                if (commands[slot]) issue('multiple_actions_for_opponent_slot');
                const targetType = dex.moves.get(move).target;
                let target = '';
                if (BattleActions.prototype.targetTypeChoices(targetType)) {
                    const location = /^(p[12])([ab]): /.exec(parts[4]);
                    if (!location) issue('opponent_target_not_observed');
                    else target = ` ${location[1] === opponent ? '-' : ''}${location[2] === 'a' ? 1 : 2}`;
                }
                commands[slot] = `move ${move}${target}`;
            }
        }
        if (previous.phase === 'move') for (let slot = 0; slot < 2; ++slot) {
            if (commands[slot]) continue;
            const ident = beliefs[i - 1].opponent.active[slot === 0 ? 'a' : 'b'];
            const known = beliefs[i - 1].opponent.pokemon.find(mon => mon.key === ident);
            if (known?.fainted) commands[slot] = 'pass';
            else if (faintedOpponents.has(ident)) commands[slot] = {unobserved_faint_action: true};
            else issue('opponent_action_not_observed');
        }
        const own = history.own_choices[i - 1];
        if (previous.phase === 'switch') validateOwnReplacement(previous, own, switches, issue);
        else validateOwnMoveTurn(previous, own, switches, faintedOwn, ownMoved, issue);
        steps.push({own_side: initial.side, own_command: own, opponent, opponent_actions: commands,
            phase: previous.phase, expected: view});
    }
    return {steps, unsupported};
}

function resolveReplayChoices(position, step, randomness) {
    if (step.phase === 'move' && step.opponent_actions.every(action => typeof action === 'string'))
        return {[step.own_side]: step.own_command, [step.opponent]: step.opponent_actions.join(', ')};
    const battle = restorePosition(position, randomness);
    try {
        const side = battle.getSide(step.opponent);
        const pending = pendingSides(battle);
        if (step.phase === 'switch' && !isPostKnockoutReplacement(battle))
            throw new Error('unsupported midturn replacement in replay hypothesis');
        const commands = step.opponent_actions.map((action, slot) => {
            if (!action && step.phase === 'switch') return 'pass';
            if (typeof action === 'string') return action;
            if (action.unobserved_faint_action) {
                // hidden cancelled choices remain hypotheses without asserted move evidence
                const mon = side.active[slot], moves = mon.getMoveRequestData().moves;
                const index = moves.findIndex(move => !move.disabled && move.pp !== 0 &&
                    battle.dex.moves.get(move.id).priority <= 0);
                if (index < 0) throw new Error('no supported cancelled choice in replay hypothesis');
                const move = moves[index];
                const target = BattleActions.prototype.targetTypeChoices(move.target) ?
                    [1, 2, -1, -2].find(loc => battle.validTargetLoc(loc, mon, move.target)) : 0;
                if (target === undefined) throw new Error('no supported cancelled choice in replay hypothesis');
                return `move ${index + 1}${target ? ` ${target}` : ''}`;
            }
            // observed identity resolved against this hypothetical roster only
            // no inference of the real hidden party order
            const index = side.pokemon.findIndex(mon => mon.fullname === action.switch_ident);
            if (index < 0) throw new Error('replay hypothesis missing observed switch identity');
            return `switch ${index + 1}`;
        });
        const choices = {};
        if (pending.includes(step.own_side)) choices[step.own_side] = step.own_command;
        if (pending.includes(step.opponent)) choices[step.opponent] = commands.join(', ');
        return choices;
    } finally { battle.destroy(); }
}

function constructMoveHistoryWorld(history, options = {}) {
    const settings = {world_attempts: 16, chance_attempts: 16, max_nodes: 512, max_elapsed_ms: 5000, ...options};
    const limits = {world_attempts: 32, chance_attempts: 32, max_nodes: 512, max_elapsed_ms: 10000};
    const validSeed = seed => Array.isArray(seed) && seed.length === 4 && seed.every(n =>
        Number.isInteger(n) && n >= 0 && n <= 65535);
    if (!validSeed(settings.seed) || !validSeed(settings.chance_seed) ||
            Object.keys(settings).some(key => !['seed', 'chance_seed'].includes(key) && !Object.hasOwn(limits, key)) ||
            Object.entries(limits).some(([key, max]) => !Number.isInteger(settings[key]) || settings[key] < 1 || settings[key] > max))
        throw new Error('move history reconstruction requires independent seeds and bounded integer settings');
    const started = performance.now(), replay = replaySteps(history);
    const initial = history.observations[0], latest = history.observations.at(-1);
    const result = {version: 1, kind: 'observation_replayed_world', scope: 'bounded_ordinary_move_history',
        supported: false, simulation_ready: false, search_ready: false, training_eligible: false,
        position: null, settings, nodes: 0, world_attempts: 0, rejected: {}, unsupported: replay.unsupported,
        history_sha256: createHash('sha256').update(JSON.stringify(history)).digest('hex'),
        information_sha256: observeBeliefs(latest).information_sha256,
        assumptions: ['Accepted worlds reproduce every supplied player observation without state patching.',
            'Rule based proposals and matching chance samples are not a calibrated posterior.',
            'Ordinary moves, voluntary switches and post knockout replacements only; midturn pivots remain unsupported.',
            'Unexecuted opposing choices cancelled by fainting are hypotheses rather than revealed moves.']};
    if (result.unsupported.length) return result;
    const proposalRng = new PRNG(settings.seed.join(',')), chanceRng = new PRNG(settings.chance_seed.join(','));
    const seedOf = rng => Array.from({length: 4}, () => rng.random(65536));
    const reject = code => { result.rejected[code] = (result.rejected[code] || 0) + 1; };
    const exhausted = () => result.nodes >= settings.max_nodes || performance.now() - started >= settings.max_elapsed_ms;
    for (let worldIndex = 0; worldIndex < settings.world_attempts && !exhausted(); ++worldIndex) {
        ++result.world_attempts;
        const opening = constructReplayOpening(initial, latest,
            {seed: seedOf(proposalRng), chance_seed: seedOf(chanceRng), attempts: 1});
        if (!opening.supported) { reject('opening_proposal_rejected'); continue; }
        for (let attempt = 0; attempt < settings.chance_attempts && !exhausted(); ++attempt) {
            let position = opening.position, matched = true;
            for (const step of replay.steps) {
                if (exhausted()) { matched = false; break; }
                ++result.nodes;
                let next;
                try {
                    const randomness = {mode: 'sample', seed: seedOf(chanceRng)};
                    const choices = resolveReplayChoices(position, step, randomness);
                    next = advancePosition(position, choices, randomness);
                }
                catch (error) {
                    if (!/\[(?:Invalid|Unavailable) choice\]|incomplete choice|unsupported midturn replacement|no supported cancelled choice/.test(error.message)) throw error;
                    reject('choice_rejected_in_hypothesis'); matched = false; break;
                }
                if (!isDeepStrictEqual(next.observations[initial.side], step.expected)) {
                    reject('observation_mismatch'); matched = false; break;
                }
                if (next.phase === 'switch') {
                    const boundary = restorePosition(next.position, {mode: 'sample', seed: settings.chance_seed});
                    try {
                        if (!isPostKnockoutReplacement(boundary)) {
                            reject('unsupported_midturn_replacement'); matched = false; break;
                        }
                    } finally { boundary.destroy(); }
                }
                position = next.position;
            }
            if (matched) {
                result.supported = true; result.simulation_ready = true; result.position = position;
                result.matched_observations = history.observations.length;
                result.elapsed_seconds = (performance.now() - started) / 1000;
                return result;
            }
        }
    }
    result.unsupported.push(exhausted() ? 'replay_budget_exhausted' : 'no_observation_consistent_replay_found');
    result.elapsed_seconds = (performance.now() - started) / 1000;
    return result;
}

module.exports = {constructMoveHistoryWorld, replaySteps};
