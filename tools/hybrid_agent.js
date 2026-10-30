'use strict';

const {HybridPlanner} = require('./hybrid_planner');

class HybridAgent {
    constructor(baseline, options, gate) {
        if (baseline.targetOptions?.labels) throw new Error('hybrid intervention cannot collect training eligible labels');
        this.baseline = baseline;
        this.log = [];
        this.view = null;
        this.acceptedOwn = null;
        this.rejected = null;
        this.analysis = {records: [], omitted: 0, max_records: 64, max_bytes: 256 * 1024};
        this.analysisBytes = 0;
        this.planner = new HybridPlanner(options, () => this.action.command.replace(/^\/choose\s+/, ''), gate);
    }

    start() { return this.baseline.start(); }
    close() { return this.baseline.close(); }
    terminate() { return this.baseline.terminate(); }

    send(message) {
        if (message.type === 'event') this.log.push(message.line.startsWith('|t:|') ? '|t:|' : message.line);
        if (message.type === 'decision_accepted') {
            this.acceptedOwn = message.command.replace(/^\/choose\s+/, ''); this.rejected = null;
        }
        if (message.type === 'decision_rejected') {
            this.rejected = {rejected: message.command.replace(/^\/choose\s+/, '')}; this.acceptedOwn = null;
        }
        this.baseline.send(message);
    }

    async observeRequest(request) {
        const turnLine = this.log.findLast(line => line.startsWith('|turn|'));
        const side = request.side?.id || this.view?.side;
        if (!side || !turnLine) throw new Error('hybrid bridge requires a started player observation');
        this.view = {side, turn: Number(turnLine.slice(6)), phase: request.forceSwitch || request.wait ? 'switch' : 'move',
            ended: false, winner: null, request: structuredClone(request), log: [...this.log]};
        if (request.wait) {
            await this.planner.decide(this.view, this.rejected || this.acceptedOwn);
            this.acceptedOwn = null; this.rejected = null;
        }
    }

    async nextMessage(type) {
        const message = await this.baseline.nextMessage(type);
        if (type === 'episode_complete') return {...message, policy_tag: 'hybrid-search-eval-v1',
            training_eligible: false, hybrid_baseline_episode: message,
            hybrid_coverage: structuredClone(this.planner.stats)};
        if (type !== 'action') return message;
        this.action = message;
        const decision = await this.planner.decide(this.view, this.rejected || this.acceptedOwn);
        const bytes = Buffer.byteLength(JSON.stringify(decision.analysis));
        if (this.analysis.records.length < this.analysis.max_records && this.analysisBytes + bytes <= this.analysis.max_bytes) {
            this.analysis.records.push(decision.analysis); this.analysisBytes += bytes;
        } else { ++this.analysis.omitted; }
        this.acceptedOwn = null; this.rejected = null;
        if (decision.command === message.command.replace(/^\/choose\s+/, '')) return {...message, training_eligible: false};
        // native accepted command reconstruction rather than stale baseline action indices
        return {...message, action: -1, action2: -1, command: decision.command,
            training_eligible: false, hybrid_source: decision.source};
    }
}

module.exports = {HybridAgent};
