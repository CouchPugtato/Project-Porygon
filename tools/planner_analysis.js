'use strict';

function interventionSettings(options = {}) {
    const settings = {enabled: true, min_worlds: 2, min_chance_samples: 2, min_mean_gain: 0.1,
        min_worst_gain: 0, require_complete: true, ...options};
    if (Object.keys(settings).some(key => !['enabled', 'min_worlds', 'min_chance_samples', 'min_mean_gain',
        'min_worst_gain', 'require_complete'].includes(key)) || typeof settings.enabled !== 'boolean' ||
            typeof settings.require_complete !== 'boolean' || !Number.isInteger(settings.min_worlds) ||
            settings.min_worlds < 1 || settings.min_worlds > 4 || !Number.isInteger(settings.min_chance_samples) ||
            settings.min_chance_samples < 1 || settings.min_chance_samples > 8 ||
            !Number.isFinite(settings.min_mean_gain) || settings.min_mean_gain < 0 ||
            !Number.isFinite(settings.min_worst_gain) || settings.min_worst_gain < 0)
        throw new Error('invalid bounded intervention settings');
    return settings;
}

function pairedGains(row, reference) {
    if (!row || !reference || row.status !== 'completed' || reference.status !== 'completed' ||
            !Array.isArray(row.samples) || !row.samples.length || !Array.isArray(reference.samples) ||
            row.samples.length !== reference.samples.length) return null;
    const gains = [], worldMeans = [];
    for (let w = 0; w < row.samples.length; ++w) {
        const own = row.samples[w], baseline = reference.samples[w], world = [];
        if (!Array.isArray(own) || !own.length || !Array.isArray(baseline) || own.length !== baseline.length) return null;
        for (let r = 0; r < own.length; ++r) {
            if (!Array.isArray(own[r]) || !own[r].length || !Array.isArray(baseline[r]) || own[r].length !== baseline[r].length) return null;
            for (let s = 0; s < own[r].length; ++s) {
                if (!Number.isFinite(own[r][s]) || !Number.isFinite(baseline[r][s])) return null;
                const gain = own[r][s] - baseline[r][s];
                if (!Number.isFinite(gain)) return null;
                gains.push(gain); world.push(gain);
            }
        }
        worldMeans.push(world.reduce((a, b) => a + b, 0) / world.length);
    }
    return {mean: worldMeans.reduce((a, b) => a + b, 0) / worldMeans.length,
        worst: Math.min(...gains), best: Math.max(...gains), world_means: worldMeans, samples: gains.length};
}

function analyzeDecision(view, search, referenceCommand, options = {}) {
    const settings = interventionSettings(options), rows = search?.candidates || [];
    const reference = rows.find(row => row.command === referenceCommand);
    const selected = rows.find(row => row.command === search?.selected_command);
    const gains = pairedGains(selected, reference);
    let reason = !search?.supported ? search?.reason || 'no_verified_search' :
        selected?.command === referenceCommand ? 'reference_preferred' :
        !gains ? 'reference_comparison_incomplete' :
        settings.require_complete && !search.comparison_complete ? 'search_comparison_incomplete' :
        (search.distinct_active_worlds || 0) < settings.min_worlds ? 'insufficient_distinct_worlds' :
        (search.settings?.chance_samples || 0) < settings.min_chance_samples ? 'insufficient_chance_samples' :
        gains.mean < settings.min_mean_gain ? 'mean_gain_too_small' :
        gains.worst < settings.min_worst_gain ? 'sampled_outcome_disagreement' : 'sampled_robust_gain';
    if (!settings.enabled && search?.supported && selected?.command !== referenceCommand) reason = 'unguarded_experiment';
    const approved = ['sampled_robust_gain', 'unguarded_experiment'].includes(reason);
    return {version: 1, turn: view.turn, phase: view.phase, reference_command: referenceCommand,
        proposed_command: search?.selected_command || null,
        selected_command: approved ? search.selected_command : referenceCommand,
        intervention: approved, reason, guard: settings, paired_gain: gains,
        distinct_active_worlds: search?.distinct_active_worlds || 0,
        candidates: rows.map(row => ({command: row.command, status: row.status, heuristic_score: row.score,
            world_scores: row.worlds, gain_over_reference: pairedGains(row, reference)})),
        units: 'heuristic_position_value_not_win_probability', calibrated: false,
        assumptions: ['Only exactly observation consistent hypothetical worlds are searched.',
            'Worlds are not a calibrated posterior and may omit important possibilities.',
            'Opponent choices and chance outcomes are bounded samples rather than exhaustive coverage.',
            'Paired gains describe sampled continuations and are not a confidence interval or a strength claim.'],
        budget: search ? {nodes: search.nodes, elapsed_seconds: search.elapsed_seconds,
            exhausted: search.budget_exhausted, stage: search.budget_stage, complete: search.comparison_complete} : null,
        training_eligible: false, strength_claim: false};
}

module.exports = {interventionSettings, pairedGains, analyzeDecision};
