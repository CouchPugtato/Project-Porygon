'use strict';

const {PlannerBeliefs} = require('./planner_beliefs');
const {searchBeliefWorlds, plannerSettings} = require('./planner_search');
const {UnsupportedPlannerRequest} = require('./planner_actions');
const {isDeepStrictEqual} = require('node:util');
const {fitJointTransition, jointSettings} = require('./planner_joint_constraints');
const {analyzeDecision, interventionSettings} = require('./planner_analysis');
const {PersistentRecovery} = require('./planner_persistence');

class HybridPlanner {
    constructor({seed, beliefs = {}, search = {}, reconstruction = null, intervention = {}}, fallback, gate) {
        if (typeof fallback !== 'function') throw new Error('an observation only fallback player is required');
        this.beliefs = new PlannerBeliefs({...beliefs, seed});
        this.search = plannerSettings({...search, seed});
        this.fallback = fallback;
        this.gate = gate;
        this.intervention = interventionSettings(intervention);
        this.reconstruction = reconstruction ? jointSettings({...reconstruction, seed}) : null;
        this.persistence = this.reconstruction?.persistent ? new PersistentRecovery(this.reconstruction) : null;
        this.reconstructionAttempts = new Set();
        this.reconstructionPath = null;
        this.stats = {decisions: 0, searched: 0, fallbacks: 0, waiting: 0, terminal: 0, reasons: {}, nodes: 0,
            belief_rejections: {}, search_failures: {}, reconstruction_attempts: 0, reconstructed_particles: 0,
            reconstruction_nodes: 0, reconstruction_seconds: 0, searches_from_joint_reconstruction: 0,
            search_completed: 0, interventions: 0, guard_rejections: {}, latest_verified_turn: 0,
            reconstruction_results: [], unsupported_evidence: {}, belief_losses: [], belief_loss_count: 0};
        this.memo = null;
    }

    adopt(witnesses, belief, nodes, seconds) {
        this.beliefs.worlds = witnesses.map(witness => witness.position);
        this.beliefs.paths = new Map(witnesses.map(witness => [witness.position.state_json,
            {opening: witness.opening, steps: witness.steps}]));
        this.reconstructionPath = this.beliefs.paths.values().next().value;
        this.stats.reconstructed_particles += this.beliefs.worlds.length;
        return this.beliefs.last = {...belief, supported: true, reason: null,
            particles: this.beliefs.worlds.length, rebuilt: this.beliefs.worlds.length,
            nodes: belief.nodes + nodes, elapsed_seconds: belief.elapsed_seconds + seconds, joint_reconstructed: true};
    }

    async decide(view, acceptedOwn = null) {
        if (acceptedOwn === null && this.memo?.key === JSON.stringify(view))
            return structuredClone(this.memo.decision);
        await (this.gate?.waitForCapacity?.() ?? this.gate?.waitForLaunch?.());
        this.persistence?.remember(this.beliefs);
        const previousPath = this.beliefs.paths.values().next().value;
        const hadVerifiedParent = this.beliefs.worlds.length > 0;
        if (previousPath) this.reconstructionPath = previousPath;
        let belief = this.beliefs.update(view, acceptedOwn);
        for (const issue of belief.issues || [])
            this.stats.unsupported_evidence[issue.code] = (this.stats.unsupported_evidence[issue.code] || 0) + 1;
        if (!belief.supported) ++this.stats.belief_loss_count;
        if (!belief.supported && this.stats.belief_losses.length < 64) this.stats.belief_losses.push({
            boundary: this.beliefs.history.observations.length, turn: view.turn, reason: belief.reason,
            had_verified_parent: hadVerifiedParent,
            issues: (belief.issues || []).map(issue => issue.code), rejections: belief.rejections});
        if (this.persistence && !belief.supported && !belief.issues?.length) {
            const recovered = await this.persistence.recover(this.beliefs, view, acceptedOwn, this.gate);
            if (recovered) {
                ++this.stats.reconstruction_attempts;
                this.stats.reconstruction_nodes += recovered.nodes;
                this.stats.reconstruction_seconds += recovered.seconds;
                const {witnesses, ...record} = recovered;
                this.stats.reconstruction_results.push(record);
                if (recovered.verified) belief = this.adopt(witnesses, belief, recovered.nodes, recovered.seconds);
            }
        }
        const failure = this.beliefs.lastFailure || this.beliefs.firstFailure;
        const boundary = this.beliefs.history.observations.length;
        const remaining = this.reconstruction && this.reconstruction.max_total_ms - this.stats.reconstruction_seconds * 1000;
        if (!this.persistence && this.reconstruction && !this.reconstructionAttempts.has(boundary) && !belief.supported && remaining >= 1 &&
                this.reconstructionAttempts.size < this.reconstruction.max_attempts &&
                boundary <= this.reconstruction.max_boundaries &&
                (boundary === 2 || this.reconstructionPath) &&
                isDeepStrictEqual(failure?.history?.observations.at(-1), view) &&
                (this.reconstructionPath || isDeepStrictEqual(failure?.simulation_trace?.expected, view))) {
            this.reconstructionAttempts.add(boundary); ++this.stats.reconstruction_attempts;
            const input = {...failure, opening_position: this.reconstructionPath?.opening,
                replay_steps: this.reconstructionPath?.steps};
            if (this.reconstructionPath && (boundary > 2 || !isDeepStrictEqual(failure.simulation_trace?.expected, view)))
                input.simulation_trace = {position: this.reconstructionPath.opening,
                seed: this.reconstruction.seed, expected: view,
                choices: typeof acceptedOwn === 'string' ? {[view.side]: acceptedOwn} : {}};
            const fitted = await fitJointTransition(input, this.gate, {...this.reconstruction,
                max_elapsed_ms: Math.min(this.reconstruction.max_elapsed_ms, Math.floor(remaining))});
            this.stats.reconstruction_nodes += fitted.proposals || 0;
            this.stats.reconstruction_seconds += fitted.elapsed_seconds || 0;
            this.stats.reconstruction_results.push({turn: view.turn, boundary, verified: !!fitted.validation_gate_passed,
                matched_boundaries: fitted.matched_boundaries || 0, best_loss: fitted.best_loss ?? null,
                reason: fitted.reason || fitted.assessment, seconds: fitted.elapsed_seconds || 0,
                mismatch: fitted.best_mismatch || null,
                differences: fitted.best_projection_differences?.slice(0, 4) || []});
            if (fitted.validation_gate_passed) {
                // only independently replayed complete projections enter the exact belief pool
                // later boundaries still pass the normal full history validation path
                belief = this.adopt(fitted.witnesses, belief, fitted.proposals, fitted.elapsed_seconds);
            }
        }
        this.persistence?.remember(this.beliefs);
        for (const [kind, count] of Object.entries(belief.rejections || {}))
            this.stats.belief_rejections[kind] = (this.stats.belief_rejections[kind] || 0) + count;
        if (view.ended || view.request?.wait) {
            ++this.stats[view.ended ? 'terminal' : 'waiting'];
            return {command: null, source: view.ended ? 'terminal' : 'waiting', belief};
        }
        ++this.stats.decisions;
        // native fallback proposal scored under the same worlds responses and chance samples
        const reference = await this.fallback(structuredClone(view));
        if (typeof reference !== 'string' || !reference.trim()) throw new Error('fallback did not return a command');
        if (belief.supported) this.stats.latest_verified_turn = Math.max(this.stats.latest_verified_turn, view.turn);
        let search = null;
        if (belief.supported) {
            try { search = await searchBeliefWorlds(view, this.beliefs.worlds, this.search, this.gate, reference); }
            catch (error) {
                if (!(error instanceof UnsupportedPlannerRequest)) throw error;
                search = {supported: false, nodes: 0, reason: 'unsupported_native_request', detail: error.message};
            }
        }
        this.stats.nodes += belief.nodes + (search?.nodes || 0);
        const analysis = analyzeDecision(view, search, reference, this.intervention);
        if (search?.supported) ++this.stats.search_completed;
        if (analysis.intervention) {
            ++this.stats.searched;
            ++this.stats.interventions;
            if (belief.joint_reconstructed) ++this.stats.searches_from_joint_reconstruction;
            const decision = {command: search.selected_command, source: 'search', belief, search, analysis};
            this.memo = {key: JSON.stringify(view), decision: structuredClone(decision)};
            return decision;
        }
        const reason = search?.supported ? analysis.reason : search?.reason || belief.reason;
        if (search?.supported) this.stats.guard_rejections[reason] = (this.stats.guard_rejections[reason] || 0) + 1;
        if (search && !search.supported) {
            const stage = search.budget_stage || search.reason;
            this.stats.search_failures[stage] = (this.stats.search_failures[stage] || 0) + 1;
        }
        ++this.stats.fallbacks;
        this.stats.reasons[reason] = (this.stats.reasons[reason] || 0) + 1;
        const decision = {command: reference, source: 'fallback', reason, belief, search, analysis};
        this.memo = {key: JSON.stringify(view), decision: structuredClone(decision)};
        return decision;
    }
}

module.exports = {HybridPlanner};
