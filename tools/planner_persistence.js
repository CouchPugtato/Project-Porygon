'use strict';

const {performance} = require('node:perf_hooks');
const {isDeepStrictEqual} = require('node:util');
const {createHash} = require('node:crypto');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {advancePosition} = require('./position_simulator');
const {propagateWorld} = require('./planner_beliefs');
const {fitJointTransition, jointSettings} = require('./planner_joint_constraints');
const {completeMoveHistory} = require('./planner_move_completion');

function historyHash(history, count = history.observations.length) {
    return createHash('sha256').update(JSON.stringify({observations: history.observations.slice(0, count),
        own_choices: (history.own_choices || []).slice(0, Math.max(0, count - 1))})).digest('hex');
}

class TrajectoryBank {
    constructor(maxEntries = 4, maxBytes = 4 * 1024 ** 2) {
        if (!Number.isInteger(maxEntries) || maxEntries < 1 || maxEntries > 4 ||
                !Number.isInteger(maxBytes) || maxBytes < 1 || maxBytes > 4 * 1024 ** 2)
            throw new Error('bounded trajectory bank required');
        this.maxEntries = maxEntries; this.maxBytes = maxBytes; this.entries = [];
    }

    remember(paths, worlds, history) {
        const boundary = history.observations.length - 1, side = history.observations[0]?.side;
        if (!side) return;
        const opponent = side === 'p1' ? 1 : 0;
        const context = historyHash(history);
        for (const position of worlds) {
            const path = paths.get(position.state_json);
            if (!path || path.steps.length !== boundary) continue;
            // whole history witnesses retained as proposals rather than calibrated particles
            // independent full replay remains mandatory when a hidden build changes
            const state = JSON.parse(path.opening.state_json);
            const identity = createHash('sha256').update(JSON.stringify(state.sides[opponent].pokemon.map(mon => mon.set))).digest('hex');
            const bytes = Buffer.byteLength(path.opening.state_json) + Buffer.byteLength(position.state_json) +
                Buffer.byteLength(JSON.stringify(path.steps));
            if (bytes > this.maxBytes) continue;
            const entry = {identity, context, path: structuredClone(path), position, boundary, bytes};
            const old = this.entries.findIndex(saved => saved.identity === identity);
            if (old >= 0 && this.entries[old].boundary > boundary) continue;
            if (old >= 0) this.entries.splice(old, 1);
            this.entries.push(entry);
        }
        this.entries.sort((a, b) => b.boundary - a.boundary || a.identity.localeCompare(b.identity));
        while (this.entries.length > this.maxEntries || this.bytes > this.maxBytes) this.entries.pop();
    }

    get bytes() { return this.entries.reduce((sum, entry) => sum + entry.bytes, 0); }
}

class PersistentRecovery {
    constructor(options) {
        this.settings = jointSettings(options);
        this.bank = new TrajectoryBank(); this.tried = new Set();
        this.seconds = 0; this.nodes = 0;
        this.rng = new PRNG(this.settings.seed.join(','));
    }

    remember(beliefs) { this.bank.remember(beliefs.paths, beliefs.worlds, beliefs.history); }

    async recover(beliefs, view, acceptedOwn, gate) {
        const boundary = beliefs.history.observations.length - 1;
        if (boundary < 1 || boundary >= this.settings.max_boundaries) return null;
        if (!isDeepStrictEqual(beliefs.history.observations.at(-1), view) ||
                !isDeepStrictEqual(beliefs.history.own_choices.at(-1), acceptedOwn))
            throw new Error('persistent recovery must match the recorded boundary and accepted choice');
        const captured = beliefs.lastFailure?.simulation_trace;
        if (!this.bank.entries.length && boundary === 1 && captured?.position) {
            // a failed first transition can still supply an exactly matched opening proposal
            const path = {opening: captured.position, steps: []};
            this.bank.remember(new Map([[captured.position.state_json, path]]), [captured.position],
                {observations: [beliefs.history.observations[0]]});
        }
        const remaining = this.settings.max_total_ms - this.seconds * 1000;
        if (this.tried.has(boundary) || this.tried.size >= this.settings.max_attempts || remaining < 1 ||
                boundary < 1 || boundary >= this.settings.max_boundaries || !this.bank.entries.length)
            return null;
        this.tried.add(boundary);
        const started = performance.now(), parent = beliefs.history.observations.at(-2);
        const report = {boundary: boundary + 1, transition_index: boundary, turn: view.turn, stage: 'chance', verified: false, nodes: 0,
            chance_nodes: 0, fit_nodes: 0, matched_boundaries: 0, reasons: {}, witnesses: [], bank_entries: this.bank.entries.length,
            roll_diagnostics: [], investment_constraints: [], move_nodes: 0, move_completions: []};
        const available = () => performance.now() - started < Math.min(remaining, this.settings.max_elapsed_ms);
        if (this.settings.move_completion) {
            const moveStarted = performance.now();
            const budget = {available: () => available() && performance.now() - moveStarted < 250 && report.move_nodes < 64,
                consume: () => ++report.move_nodes, wait: () => gate?.waitForCapacity?.() ?? gate?.waitForLaunch?.()};
            for (const entry of this.bank.entries) {
                if (!budget.available()) break;
                const completed = await completeMoveHistory(entry.path, beliefs.history, this.settings.seed, budget);
                report.move_completions.push({verified: completed.verified, reason: completed.reason, proposals: completed.proposals});
                if (completed.verified) { report.witnesses.push(...completed.witnesses); report.stage = 'move_completion'; break; }
            }
        }
        const chanceStarted = performance.now();
        const chanceAvailable = () => available() && performance.now() - chanceStarted < 250 && report.chance_nodes < 128;
        const context = historyHash(beliefs.history, beliefs.history.observations.length - 1);
        for (const entry of this.bank.entries.filter(entry => entry.boundary === boundary - 1 && entry.context === context)) {
            if (report.witnesses.length) break;
            await (gate?.waitForCapacity?.() ?? gate?.waitForLaunch?.());
            if (!chanceAvailable()) break;
            let accepted;
            const budget = {available: chanceAvailable, consume: () => ++report.chance_nodes,
                diagnostics: report.reasons, filter_evidence: true, accept: step => { accepted = step; }};
            const position = propagateWorld(entry.position, parent, acceptedOwn, view, this.rng, budget, 128);
            if (!position || !accepted) continue;
            const verified = advancePosition(entry.position, accepted.choices, {mode: 'sample', seed: accepted.seed});
            if (!isDeepStrictEqual(verified.observations[view.side], view))
                throw new Error('persistent chance witness failed independent replay');
            report.witnesses.push({opening: entry.path.opening, steps: [...entry.path.steps, accepted], position});
            if (report.witnesses.length >= this.settings.particles) break;
        }
        if (!report.witnesses.length && available() && (beliefs.lastFailure || beliefs.firstFailure)) {
            report.stage = 'history_fit';
            // allocate shorter attempts to alternate verified roots instead of unrelated random teams
            // preserve the total game budget and compare complete histories before accepting any result
            const roots = this.bank.entries.slice(0, 3);
            for (const [index, entry] of roots.entries()) {
                if (!available()) break;
                const time = Math.floor(Math.min(remaining, this.settings.max_elapsed_ms) - (performance.now() - started));
                if (time < 1) break;
                const failure = {history: beliefs.history, opening_position: entry.path.opening, replay_steps: entry.path.steps,
                    simulation_trace: {position: entry.path.opening, seed: this.settings.seed, expected: view,
                        choices: typeof acceptedOwn === 'string' ? {[view.side]: acceptedOwn} : {}}};
                const fitted = await fitJointTransition(failure, gate, {...this.settings,
                    max_elapsed_ms: Math.max(1, Math.floor(index === 0 && roots.length > 1 ? time * 0.75 : time / (roots.length - index)))});
                report.fit_nodes += fitted.proposals || 0;
                report.roll_diagnostics.push(...(fitted.roll_diagnostics || []));
                if (fitted.investment_constraints) report.investment_constraints.push(fitted.investment_constraints);
                report.matched_boundaries = Math.max(report.matched_boundaries, fitted.matched_boundaries || 0);
                const reason = fitted.best_mismatch?.kind || fitted.reason || fitted.assessment;
                report.reasons[reason] = (report.reasons[reason] || 0) + 1;
                if (fitted.validation_gate_passed) { report.witnesses = fitted.witnesses; break; }
            }
        }
        report.verified = report.witnesses.length > 0;
        if (report.verified) report.matched_boundaries = boundary;
        report.nodes = report.chance_nodes + report.fit_nodes + report.move_nodes;
        report.seconds = (performance.now() - started) / 1000;
        this.seconds += report.seconds; this.nodes += report.nodes;
        return report;
    }
}

module.exports = {TrajectoryBank, PersistentRecovery};
