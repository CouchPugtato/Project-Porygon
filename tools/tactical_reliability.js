'use strict';

function assessTacticalProbe(probe, result, proposed) {
    const preferred = result.candidates.find(row => row.command === probe.preferred);
    const alternative = result.candidates.find(row => row.command === probe.alternative);
    if (!preferred || !alternative) throw new Error('tactical probe requires both comparison rows');
    const evidence = {preferred_status: preferred.status, alternative_status: alternative.status,
        preferred_score: preferred.score, alternative_score: alternative.score,
        preferred_probe_proposed: proposed.includes(probe.preferred)};
    if (result.budget_exhausted) return {...evidence, classification: 'budget_gap'};
    if (preferred.status === 'world_conditional_rejection' || alternative.status === 'world_conditional_rejection')
        return {...evidence, classification: 'world_conditional_rejection'};
    if (preferred.status === 'unsupported_continuation' || alternative.status === 'unsupported_continuation')
        return {...evidence, classification: 'continuation_gap'};
    if (preferred.status !== 'completed' || alternative.status !== 'completed')
        return {...evidence, classification: 'incomplete_comparison'};
    const margin = preferred.score - alternative.score;
    // equal material scores despite a verified field benefit
    if (probe.expectation === 'unscored_benefit')
        return {...evidence, margin, classification: margin <= 1e-10 ? 'scoring_gap' : 'benefit_distinguished'};
    if (probe.expectation === 'equivalent')
        return {...evidence, margin, classification: Math.abs(margin) <= 1e-10 ? 'passed' : 'scoring_gap'};
    if (margin <= 1e-10) return {...evidence, margin, classification: 'scoring_gap'};
    // missing reference probe rather than proof of every good action missing
    return {...evidence, margin, classification: evidence.preferred_probe_proposed ? 'passed' : 'candidate_probe_missing'};
}

module.exports = {assessTacticalProbe};
