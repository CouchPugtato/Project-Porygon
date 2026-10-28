'use strict';

const {isDeepStrictEqual} = require('node:util');
const {performance} = require('node:perf_hooks');
const {createHash} = require('node:crypto');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {Dex, toID} = require('../external/pokemon-showdown/dist/sim/dex');
const {BattleActions} = require('../external/pokemon-showdown/dist/sim/battle-actions');
const {FORMAT, advancePosition} = require('./position_simulator');
const {observeBeliefs} = require('./belief_state');
const {constructReplayOpening} = require('./opening_world');
const {isOrdinaryMove} = require('./ordinary_moves');

const EVENTS = new Set(['', 't:', 'turn', 'move', '-damage', '-heal', '-status', '-curestatus',
    '-activate', '-singleturn', '-fail', '-immune', '-supereffective', '-resisted', '-crit', '-miss',
    'cant', 'upkeep', 'message']);

function replaySteps(history) {
    if (!history || Object.keys(history).length !== 2 || !Array.isArray(history.observations) ||
            !Array.isArray(history.own_choices) || history.observations.length < 2 ||
            history.observations.length > 4 || history.own_choices.length !== history.observations.length - 1)
        throw new Error('supply two to four observations and one accepted own command per completed turn');
    const observations = history.observations;
    const beliefs = observations.map(view => observeBeliefs(view));
    const unsupported = [], steps = [], dex = Dex.forFormat(FORMAT);
    const issue = code => { if (!unsupported.includes(code)) unsupported.push(code); };
    const initial = observations[0];
    if (initial.turn !== 1) issue('opening_observation_required');
    const opponent = initial.side === 'p1' ? 'p2' : 'p1';
    for (const [i, view] of observations.entries()) {
        if (view.side !== initial.side || view.phase !== 'move' || view.ended ||
                view.turn !== i + 1 || !Array.isArray(view.request?.active) || view.request.active.length !== 2)
            issue('consecutive_two_active_move_requests_required');
        if (beliefs[i].issues.length) issue('unsupported_observation_evidence');
        if (!i) continue;
        const previous = observations[i - 1];
        if (previous.log.some((line, j) => view.log[j] !== line)) issue('history_prefix_mismatch');
        const segment = view.log.slice(previous.log.length), commands = [null, null];
        for (const line of segment) {
            const parts = line.split('|'), event = parts[1];
            if (!EVENTS.has(event)) issue(`unsupported_event_${event}`);
            if (event === 'move') {
                const move = toID(parts[3]);
                if (!isOrdinaryMove(move) || parts.some(part => part.startsWith('[from]')))
                    issue('unsupported_move_mechanics');
                const actor = /^(p[12])([ab]): /.exec(parts[2]);
                if (actor?.[1] !== opponent) continue;
                const slot = actor[2] === 'a' ? 0 : 1;
                if (commands[slot]) issue('multiple_moves_for_opponent_slot');
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
        if (commands.some(command => !command)) issue('opponent_action_not_observed');
        const own = history.own_choices[i - 1];
        if (typeof own !== 'string' || own.split(',').length !== 2) issue('complete_own_moves_required');
        else for (const [slot, command] of own.split(',').entries()) {
            const match = /^move ([a-z0-9]+)(?: (-?[12]))?$/.exec(command.trim());
            const moves = previous.request?.active?.[slot]?.moves || [];
            const id = match && (/^\d+$/.test(match[1]) ? moves[Number(match[1]) - 1]?.id : toID(match[1]));
            if (!id || !isOrdinaryMove(id)) issue('unsupported_own_command');
        }
        steps.push({choices: {[initial.side]: own, [opponent]: commands.join(', ')}, expected: view});
    }
    return {steps, unsupported};
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
            'Only the explicit ordinary move subset is supported; unknown opposing actions require abstention.']};
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
                try { next = advancePosition(position, step.choices, {mode: 'sample', seed: seedOf(chanceRng)}); }
                catch (error) {
                    if (!/\[(?:Invalid|Unavailable) choice\]|incomplete choice/.test(error.message)) throw error;
                    reject('choice_rejected_in_hypothesis'); matched = false; break;
                }
                if (!isDeepStrictEqual(next.observations[initial.side], step.expected)) {
                    reject('observation_mismatch'); matched = false; break;
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
