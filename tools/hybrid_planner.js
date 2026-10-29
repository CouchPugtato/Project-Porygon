'use strict';

const {PlannerBeliefs} = require('./planner_beliefs');
const {searchBeliefWorlds, plannerSettings} = require('./planner_search');
const {UnsupportedPlannerRequest} = require('./planner_actions');
const {isDeepStrictEqual} = require('node:util');
const {fitJointTransition, jointSettings} = require('./planner_joint_constraints');

class HybridPlanner {
    constructor({seed, beliefs = {}, search = {}, reconstruction = null}, fallback, gate) {
        if (typeof fallback !== 'function') throw new Error('an observation only fallback player is required');
        this.beliefs = new PlannerBeliefs({...beliefs, seed});
        this.search = plannerSettings({...search, seed});
        this.fallback = fallback;
        this.gate = gate;
        this.reconstruction = reconstruction ? jointSettings({...reconstruction, seed}) : null;
        this.reconstructionAttempted = false;
        this.stats = {decisions: 0, searched: 0, fallbacks: 0, waiting: 0, terminal: 0, reasons: {}, nodes: 0,
            belief_rejections: {}, search_failures: {}, reconstruction_attempts: 0, reconstructed_particles: 0,
            reconstruction_nodes: 0, reconstruction_seconds: 0, searches_from_joint_reconstruction: 0};
        this.memo = null;
    }

    async decide(view, acceptedOwn = null) {
        if (acceptedOwn === null && this.memo?.key === JSON.stringify(view))
            return structuredClone(this.memo.decision);
        await (this.gate?.waitForCapacity?.() ?? this.gate?.waitForLaunch?.());
        let belief = this.beliefs.update(view, acceptedOwn);
        const failure = this.beliefs.firstFailure;
        if (this.reconstruction && !this.reconstructionAttempted && !belief.supported &&
                failure?.history?.observations.length === 2 && this.beliefs.history.observations.length === 2 &&
                isDeepStrictEqual(failure.simulation_trace?.expected, view)) {
            this.reconstructionAttempted = true; ++this.stats.reconstruction_attempts;
            const fitted = await fitJointTransition(failure, this.gate, this.reconstruction);
            this.stats.reconstruction_nodes += fitted.proposals || 0;
            this.stats.reconstruction_seconds += fitted.elapsed_seconds || 0;
            if (fitted.validation_gate_passed) {
                // only independently replayed complete projections enter the exact belief pool
                // later boundaries still pass the normal full history validation path
                this.beliefs.worlds = fitted.witnesses.map(witness => witness.position);
                this.stats.reconstructed_particles += this.beliefs.worlds.length;
                belief = this.beliefs.last = {...belief, supported: true, reason: null,
                    particles: this.beliefs.worlds.length, rebuilt: this.beliefs.worlds.length,
                    nodes: belief.nodes + fitted.proposals,
                    elapsed_seconds: belief.elapsed_seconds + fitted.elapsed_seconds, joint_reconstructed: true};
            }
        }
        for (const [kind, count] of Object.entries(belief.rejections || {}))
            this.stats.belief_rejections[kind] = (this.stats.belief_rejections[kind] || 0) + count;
        if (view.ended || view.request?.wait) {
            ++this.stats[view.ended ? 'terminal' : 'waiting'];
            return {command: null, source: view.ended ? 'terminal' : 'waiting', belief};
        }
        ++this.stats.decisions;
        let search = null;
        if (belief.supported) {
            try { search = await searchBeliefWorlds(view, this.beliefs.worlds, this.search, this.gate); }
            catch (error) {
                if (!(error instanceof UnsupportedPlannerRequest)) throw error;
                search = {supported: false, nodes: 0, reason: 'unsupported_native_request', detail: error.message};
            }
        }
        this.stats.nodes += belief.nodes + (search?.nodes || 0);
        if (search?.supported) {
            ++this.stats.searched;
            if (belief.joint_reconstructed) ++this.stats.searches_from_joint_reconstruction;
            const decision = {command: search.selected_command, source: 'search', belief, search};
            this.memo = {key: JSON.stringify(view), decision: structuredClone(decision)};
            return decision;
        }
        const reason = search?.reason || belief.reason;
        if (search) {
            const stage = search.budget_stage || search.reason;
            this.stats.search_failures[stage] = (this.stats.search_failures[stage] || 0) + 1;
        }
        ++this.stats.fallbacks;
        this.stats.reasons[reason] = (this.stats.reasons[reason] || 0) + 1;
        // fallback receives only the same player projection rather than a simulator handle
        const command = await this.fallback(structuredClone(view));
        if (typeof command !== 'string' || !command.trim()) throw new Error('fallback did not return a command');
        return {command, source: 'fallback', reason, belief, search};
    }
}

module.exports = {HybridPlanner};
