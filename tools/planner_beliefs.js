'use strict';

const {isDeepStrictEqual} = require('node:util');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {toID} = require('../external/pokemon-showdown/dist/sim/dex');
const {observeBeliefs} = require('./belief_state');
const {constructOpeningWorld, constructReplayOpening} = require('./opening_world');
const {restorePosition, playerObservation, pendingSides, advancePosition} = require('./position_simulator');
const {decisionActions, UnsupportedPlannerRequest} = require('./planner_actions');

function nextSeed(rng) { return Array.from({length: 4}, () => rng.random(65536)); }

function choiceFailure(error) {
    return /\[(?:Invalid|Unavailable) choice\]|incomplete choice/.test(error.message);
}

function observedResponses(battle, before, after) {
    const opponent = before.side === 'p1' ? 'p2' : 'p1';
    if (!pendingSides(battle).includes(opponent)) return [null];
    const view = playerObservation(battle, opponent), side = battle.getSide(opponent);
    const segment = after.log.slice(before.log.length), evidence = [null, null], tera = new Set();
    for (const line of segment) {
        const parts = line.split('|'), actor = /^(p[12])([ab]): (.+)$/.exec(parts[2] || '');
        if (!actor || actor[1] !== opponent) continue;
        const slot = actor[2] === 'a' ? 0 : 1;
        if (parts[1] === '-terastallize') tera.add(slot);
        // only actions by the currently acting pokemon belong to this boundary
        if (parts[1] === 'move' && side.active[slot]?.name === actor[3] && !evidence[slot]) {
            const target = /^(p[12])([ab]): /.exec(parts[4] || '');
            evidence[slot] = {kind: 'move', id: toID(parts[3]),
                target: target ? (target[1] === opponent ? -1 : 1) * (target[2] === 'a' ? 1 : 2) : 0};
        }
        if (parts[1] === 'switch' && !evidence[slot])
            evidence[slot] = {kind: 'switch', ident: `${opponent}: ${actor[3]}`};
    }
    return decisionActions(view).turns.filter(turn => turn.actions.every((action, slot) => {
        const observed = evidence[slot];
        if (action.tera !== tera.has(slot)) return false;
        if (!observed) return true;
        if (action.kind !== observed.kind) return false;
        if (action.kind === 'switch') return side.pokemon[action.switch - 1]?.fullname === observed.ident;
        return view.request.active[slot].moves[action.move - 1]?.id === observed.id &&
            (!action.target || !observed.target || action.target === observed.target);
    })).map(turn => turn.command);
}

function propagateWorld(position, before, own, after, rng, budget, attempts = 8) {
    const branch = restorePosition(position, {mode: 'sample', seed: nextSeed(rng)});
    let responses, pending;
    try {
        if (!isDeepStrictEqual(playerObservation(branch, before.side), before))
            throw new Error('belief parent does not match the previous observation');
        pending = pendingSides(branch);
        if (pending.includes(before.side) !== (typeof own === 'string'))
            throw new Error('accepted own command required only for a pending player');
        try { responses = observedResponses(branch, before, after); }
        catch (error) { if (error instanceof UnsupportedPlannerRequest) return null; throw error; }
    } finally { branch.destroy(); }
    for (let attempt = 0; responses.length && attempt < attempts && budget.available(); ++attempt) {
        const response = responses[attempt % responses.length];
        const opponent = before.side === 'p1' ? 'p2' : 'p1', choices = {};
        if (pending.includes(before.side)) choices[before.side] = own;
        if (pending.includes(opponent)) choices[opponent] = response;
        budget.consume();
        let result;
        try { result = advancePosition(position, choices, {mode: 'sample', seed: nextSeed(rng)}); }
        catch (error) { if (choiceFailure(error)) continue; throw error; }
        // matching the entire projection includes pp queues field effects and private updates
        if (isDeepStrictEqual(result.observations[before.side], after)) return result.position;
    }
    return null;
}

class PlannerBeliefs {
    constructor(options) {
        this.settings = {particles: 2, max_nodes: 256, max_elapsed_ms: 2000, max_boundaries: 128, ...options};
        const limits = {particles: 4, max_nodes: 1024, max_elapsed_ms: 10000, max_boundaries: 512};
        if (!Array.isArray(this.settings.seed) || this.settings.seed.length !== 4 ||
                this.settings.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535) ||
                Object.keys(this.settings).some(key => key !== 'seed' && !Object.hasOwn(limits, key)) ||
                Object.entries(limits).some(([key, max]) => !Number.isInteger(this.settings[key]) ||
                    this.settings[key] < 1 || this.settings[key] > max)) throw new Error('invalid bounded belief settings');
        this.rng = new PRNG(this.settings.seed.join(','));
        this.history = {observations: [], own_choices: []};
        this.worlds = [];
        this.last = null;
    }

    update(view, acceptedOwn = null) {
        const facts = observeBeliefs(view), previous = this.history.observations.at(-1);
        if (previous && (view.side !== previous.side || view.turn < previous.turn ||
                previous.log.some((line, i) => line !== view.log[i])))
            throw new Error('belief updates require chronological observations from the same player');
        if (!previous && acceptedOwn !== null) throw new Error('opening update cannot include a previous choice');
        if (previous && isDeepStrictEqual(previous, view)) return this.last;
        const started = performance.now();
        let nodes = 0;
        const budget = {available: () => nodes < this.settings.max_nodes &&
            performance.now() - started < this.settings.max_elapsed_ms, consume: () => ++nodes};
        const report = {supported: false, particles: 0, retained: 0, rebuilt: 0, nodes: 0,
            calibrated: false, reason: null, training_eligible: false};
        if (this.history.observations.length >= this.settings.max_boundaries ||
                Buffer.byteLength(JSON.stringify(this.history)) + Buffer.byteLength(JSON.stringify(view)) > 16 * 1024 ** 2) {
            this.worlds = []; return this.last = {...report, reason: 'history_capacity_exhausted'};
        }
        if (previous) this.history.own_choices.push(acceptedOwn);
        this.history.observations.push(structuredClone(view));
        if (facts.issues.length) {
            this.worlds = []; return this.last = {...report, reason: 'unsupported_visible_evidence', issues: facts.issues};
        }
        const survivors = [];
        if (previous) for (const position of this.worlds) {
            if (!budget.available()) break;
            const next = propagateWorld(position, previous, acceptedOwn, view, this.rng, budget);
            if (next) { survivors.push(next); ++report.retained; }
        }
        // replenish from revealed history without patching inconsistent surviving states
        for (let attempt = 0; survivors.length < this.settings.particles && attempt < 16 && budget.available(); ++attempt) {
            budget.consume();
            const settings = {seed: nextSeed(this.rng), chance_seed: nextSeed(this.rng), attempts: 1};
            const opening = previous ? constructReplayOpening(this.history.observations[0], view, settings) :
                constructOpeningWorld(view, settings);
            if (!opening.supported) continue;
            let position = opening.position;
            for (let i = 1; position && i < this.history.observations.length; ++i)
                position = propagateWorld(position, this.history.observations[i - 1],
                    this.history.own_choices[i - 1], this.history.observations[i], this.rng, budget);
            if (position) { survivors.push(position); ++report.rebuilt; }
        }
        this.worlds = survivors;
        return this.last = {...report, supported: survivors.length > 0, particles: survivors.length, nodes,
            reason: survivors.length ? null : 'no_observation_consistent_world',
            elapsed_seconds: (performance.now() - started) / 1000};
    }
}

module.exports = {PlannerBeliefs, propagateWorld, observedResponses, nextSeed, choiceFailure};
