'use strict';

const {PlannerBeliefs} = require('./planner_beliefs');
const {searchBeliefWorlds, plannerSettings} = require('./planner_search');
const {UnsupportedPlannerRequest} = require('./planner_actions');

class HybridPlanner {
    constructor({seed, beliefs = {}, search = {}}, fallback, gate) {
        if (typeof fallback !== 'function') throw new Error('an observation only fallback player is required');
        this.beliefs = new PlannerBeliefs({...beliefs, seed});
        this.search = plannerSettings({...search, seed});
        this.fallback = fallback;
        this.gate = gate;
        this.stats = {decisions: 0, searched: 0, fallbacks: 0, waiting: 0, terminal: 0, reasons: {}, nodes: 0};
    }

    async decide(view, acceptedOwn = null) {
        await this.gate?.waitForLaunch();
        const belief = this.beliefs.update(view, acceptedOwn);
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
            return {command: search.selected_command, source: 'search', belief, search};
        }
        const reason = search?.reason || belief.reason;
        ++this.stats.fallbacks;
        this.stats.reasons[reason] = (this.stats.reasons[reason] || 0) + 1;
        // fallback receives only the same player projection rather than a simulator handle
        const command = await this.fallback(structuredClone(view));
        if (typeof command !== 'string' || !command.trim()) throw new Error('fallback did not return a command');
        return {command, source: 'fallback', reason, belief, search};
    }
}

module.exports = {HybridPlanner};
