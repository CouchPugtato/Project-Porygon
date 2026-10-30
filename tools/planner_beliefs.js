'use strict';

const {isDeepStrictEqual} = require('node:util');
const {performance} = require('node:perf_hooks');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {toID} = require('../external/pokemon-showdown/dist/sim/dex');
const {observeBeliefs} = require('./belief_state');
const {constructOpeningWorld, constructReplayOpening} = require('./opening_world');
const {restorePosition, playerObservation, pendingSides, advancePosition, capturePosition} = require('./position_simulator');
const {decisionActions, UnsupportedPlannerRequest} = require('./planner_actions');
const {mismatchEvidence, recordRejection, revealedConstraints} = require('./planner_evidence');
const {mutateProposal} = require('./planner_recovery');

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
        if (own && typeof own === 'object' && typeof own.rejected === 'string') {
            if (!budget.available()) return null;
            budget.consume();
            branch.strictChoices = false;
            const accepted = branch.choose(before.side, own.rejected);
            if (!accepted && isDeepStrictEqual(playerObservation(branch, before.side), after))
                return capturePosition(branch, position.origin);
            recordRejection(budget.diagnostics, 'rejected_choice_disclosure_mismatch'); return null;
        }
        if (pending.includes(before.side) !== (typeof own === 'string'))
            throw new Error('accepted own command required only for a pending player');
        if (budget.filter_evidence) {
            const conflict = revealedConstraints(branch, after);
            if (conflict) {
                recordRejection(budget.diagnostics, conflict);
                budget.trace?.({stage: 'evidence_filter', reason: conflict, position}); return null;
            }
        }
        try { responses = observedResponses(branch, before, after); }
        catch (error) {
            if (error instanceof UnsupportedPlannerRequest) { recordRejection(budget.diagnostics, 'unsupported_request'); return null; }
            throw error;
        }
    } finally { branch.destroy(); }
    for (let attempt = 0; responses.length && attempt < attempts && budget.available(); ++attempt) {
        const response = responses[attempt % responses.length];
        const opponent = before.side === 'p1' ? 'p2' : 'p1', choices = {};
        if (pending.includes(before.side)) choices[before.side] = own;
        if (pending.includes(opponent)) choices[opponent] = response;
        budget.consume();
        let result;
        const seed = nextSeed(rng);
        try { result = advancePosition(position, choices, {mode: 'sample', seed}); }
        catch (error) { if (choiceFailure(error)) { recordRejection(budget.diagnostics, 'choice_rejected'); continue; } throw error; }
        // matching the entire projection includes pp queues field effects and private updates
        if (isDeepStrictEqual(result.observations[before.side], after)) {
            budget.accept?.({seed, choices}); return result.position;
        }
        const mismatch = mismatchEvidence(result.observations[before.side], after);
        recordRejection(budget.diagnostics, mismatch.kind);
        budget.trace?.({stage: 'simulation', position, choices, seed, mismatch,
            expected: after, actual: result.observations[before.side]});
        budget.consider?.(mismatch);
    }
    if (!responses.length) {
        recordRejection(budget.diagnostics, 'observed_response_not_in_hypothesis');
        budget.trace?.({stage: 'action_inference', reason: 'observed_response_not_in_hypothesis', position});
    }
    if (!budget.available()) recordRejection(budget.diagnostics, 'replay_budget_exhausted');
    return null;
}

class PlannerBeliefs {
    constructor(options) {
        this.settings = {particles: 2, max_nodes: 256, max_elapsed_ms: 2000, max_boundaries: 128,
            filter_evidence: true, recovery: false, ...options};
        const limits = {particles: 4, max_nodes: 1024, max_elapsed_ms: 10000, max_boundaries: 512};
        if (!Array.isArray(this.settings.seed) || this.settings.seed.length !== 4 ||
                this.settings.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535) ||
                ['filter_evidence', 'recovery'].some(key => typeof this.settings[key] !== 'boolean') ||
                Object.keys(this.settings).some(key => !['seed', 'filter_evidence', 'recovery'].includes(key) && !Object.hasOwn(limits, key)) ||
                Object.entries(limits).some(([key, max]) => !Number.isInteger(this.settings[key]) ||
                    this.settings[key] < 1 || this.settings[key] > max)) throw new Error('invalid bounded belief settings');
        this.rng = new PRNG(this.settings.seed.join(','));
        this.history = {observations: [], own_choices: []};
        this.worlds = [];
        this.last = null;
        this.bestProposal = null;
        this.firstFailure = null;
        this.lastFailure = null;
        this.paths = new Map();
    }

    update(view, acceptedOwn = null) {
        const facts = observeBeliefs(view), previous = this.history.observations.at(-1);
        if (previous && (view.side !== previous.side || view.turn < previous.turn ||
                previous.log.some((line, i) => line !== view.log[i])))
            throw new Error('belief updates require chronological observations from the same player');
        if (!previous && acceptedOwn !== null) throw new Error('opening update cannot include a previous choice');
        if (previous && isDeepStrictEqual(previous, view)) return this.last;
        const started = performance.now();
        this.lastFailure = null;
        let activePath = null;
        const nextPaths = new Map();
        let nodes = 0;
        const rejections = {};
        let firstTrace = null;
        let simulationTrace = null;
        const budget = {available: () => nodes < this.settings.max_nodes &&
            performance.now() - started < this.settings.max_elapsed_ms, consume: () => ++nodes,
            diagnostics: rejections, filter_evidence: this.settings.filter_evidence,
            trace: trace => {
                if (!this.firstFailure && !firstTrace) firstTrace = structuredClone(trace);
                if (!simulationTrace && trace.stage === 'simulation')
                    simulationTrace = structuredClone(trace);
            }, accept: step => {
                if (activePath) activePath = {...activePath, steps: [...activePath.steps, step]};
            }};
        const report = {supported: false, particles: 0, retained: 0, rebuilt: 0, nodes: 0,
            calibrated: false, reason: null, training_eligible: false, rejections};
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
            activePath = this.paths.get(position.state_json) || null;
            const next = propagateWorld(position, previous, acceptedOwn, view, this.rng, budget);
            if (next) {
                survivors.push(next); ++report.retained;
                if (activePath) nextPaths.set(next.state_json, activePath);
            }
        }
        // replenish from revealed history without patching inconsistent surviving states
        let bestDistance = Infinity;
        for (let attempt = 0; survivors.length < this.settings.particles && attempt < 64 && budget.available(); ++attempt) {
            budget.consume();
            const settings = {seed: nextSeed(this.rng), chance_seed: nextSeed(this.rng), attempts: 1};
            if (this.settings.recovery && this.bestProposal && attempt % 2) {
                const proposal = mutateProposal(this.bestProposal, this.rng);
                for (const known of facts.opponent.pokemon) {
                    const entry = proposal.find(p => p.key === known.key);
                    if (!entry) continue;
                    entry.set.moves = [...new Set([...known.moves, ...entry.set.moves])].slice(0, 4);
                    if (known.tera_type) entry.set.teraType = known.tera_type;
                    if (known.ability) entry.set.ability = known.ability;
                    if (known.item) entry.set.item = known.item.last_revealed;
                }
                settings.proposal_sets = proposal;
            }
            const opening = previous ? constructReplayOpening(this.history.observations[0], view, settings) :
                constructOpeningWorld(view, settings);
            if (!opening.supported) {
                for (const [kind, count] of Object.entries(opening.rejected))
                    rejections[kind] = (rejections[kind] || 0) + count;
                continue;
            }
            let position = opening.position;
            activePath = {opening: opening.position, steps: []};
            for (let i = 1; position && i < this.history.observations.length; ++i) {
                budget.consider = mismatch => {
                    const distance = (this.history.observations.length - i) * 10000 + mismatch.distance;
                    if (distance < bestDistance) {
                        bestDistance = distance; this.bestProposal = structuredClone(opening.opponent_hypotheses);
                    }
                };
                position = propagateWorld(position, this.history.observations[i - 1],
                    this.history.own_choices[i - 1], this.history.observations[i], this.rng, budget);
            }
            if (position) {
                survivors.push(position); ++report.rebuilt;
                nextPaths.set(position.state_json, activePath);
                this.bestProposal = structuredClone(opening.opponent_hypotheses);
            }
        }
        this.worlds = survivors;
        this.paths = nextPaths;
        if (previous && !survivors.length) {
            const failure = {version: 1, training_eligible: false, source: 'player_observations_and_sampled_world_only',
                history: structuredClone(this.history), settings: this.settings, rejections,
                trace: firstTrace, simulation_trace: simulationTrace};
            this.lastFailure = Buffer.byteLength(JSON.stringify(failure)) <= 512 * 1024 ? failure :
                {version: 1, reason: 'failure_capture_size_limit', rejections};
            if (!this.firstFailure) this.firstFailure = this.lastFailure;
        }
        return this.last = {...report, supported: survivors.length > 0, particles: survivors.length, nodes,
            reason: survivors.length ? null : 'no_observation_consistent_world',
            budget_exhausted: !budget.available(),
            elapsed_seconds: (performance.now() - started) / 1000};
    }
}

module.exports = {PlannerBeliefs, propagateWorld, observedResponses, nextSeed, choiceFailure};
