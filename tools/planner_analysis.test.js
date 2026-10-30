'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {analyzeDecision, pairedGains, interventionSettings} = require('./planner_analysis');
const {renderAnalysis} = require('./battle_analysis');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, playerObservation} = require('./position_simulator');
const {searchBeliefWorlds} = require('./planner_search');

function comparison() {
    const row = (command, samples) => ({command, status: 'completed', samples, score: 0, worlds: [0, 0]});
    return {supported: true, selected_command: 'move 1 1, move 2', comparison_complete: true,
        settings: {chance_samples: 2}, distinct_active_worlds: 2, nodes: 8, elapsed_seconds: 0.1,
        candidates: [row('move 4, move 2', [[[0, 0]], [[0, 0]]]),
            row('move 1 1, move 2', [[[1, 1]], [[1, 1]]])]};
}

const VIEW = {turn: 2, phase: 'move'}, REFERENCE = 'move 4, move 2';

test('guard accepts a complete sampled robust gain and labels it as heuristic rather than confidence', () => {
    const record = analyzeDecision(VIEW, comparison(), REFERENCE);
    assert(record.intervention); assert.equal(record.reason, 'sampled_robust_gain');
    assert.equal(record.paired_gain.worst, 1); assert.equal(record.paired_gain.samples, 4);
    assert.equal(record.calibrated, false); assert.equal(record.strength_claim, false);
    assert(renderAnalysis(record).includes('not_win_probability'));
});

test('guard refuses adverse worlds poor coverage incomplete work and tiny gains', () => {
    for (const [change, reason] of [
        [search => { search.candidates[1].samples[1][0][0] = -0.1; }, 'sampled_outcome_disagreement'],
        [search => { search.distinct_active_worlds = 1; }, 'insufficient_distinct_worlds'],
        [search => { search.settings.chance_samples = 1; }, 'insufficient_chance_samples'],
        [search => { search.comparison_complete = false; }, 'search_comparison_incomplete'],
        [search => { search.candidates[0].status = 'incomplete_or_rejected'; }, 'reference_comparison_incomplete'],
        [search => { search.candidates[1].samples = [[[0, 0]], [[0, 0]]]; }, 'mean_gain_too_small'],
    ]) {
        const search = comparison(); change(search);
        const result = analyzeDecision(VIEW, search, REFERENCE);
        assert.equal(result.reason, reason); assert.equal(result.intervention, false);
        assert.equal(result.selected_command, REFERENCE);
    }
    assert.equal(pairedGains({status: 'completed', samples: [[[NaN]]]}, {status: 'completed', samples: [[[0]]]}), null);
    assert.throws(() => interventionSettings({min_worst_gain: -1}), /bounded/);
});

test('no worlds still produces useful fallback analysis without invented scores', () => {
    const record = analyzeDecision(VIEW, null, REFERENCE);
    assert.equal(record.selected_command, REFERENCE); assert.equal(record.paired_gain, null);
    assert.equal(record.candidates.length, 0); assert(renderAnalysis(record).includes('no_verified_search'));
});

test('native search reserves a root row for the reference and preserves paired sample shapes', async () => {
    const battle = createHybridFixture();
    try {
        const result = await searchBeliefWorlds(playerObservation(battle, 'p1'), [capturePosition(battle, 'fixture')],
            {seed: [367, 1, 2, 3], candidates: 2, responses: 1, chance_samples: 2, max_nodes: 32, max_elapsed_ms: 2000}, null, REFERENCE);
        assert.equal(result.candidates[0].command, REFERENCE);
        assert.equal(result.candidates[0].samples[0][0].length, 2);
        assert.equal(result.distinct_active_worlds, 1);
        const rejected = analyzeDecision(VIEW, result, REFERENCE);
        assert.equal(rejected.intervention, false);
    } finally { battle.destroy(); }
});

test('native paired search can justify replacing an obviously harmful reference across distinct worlds', async () => {
    const first = createHybridFixture(), second = createHybridFixture({changeTeams: teams => {
        teams[1][0].evs = {hp: 124, spd: 124};
    }});
    try {
        const harmful = 'move 1 -2, move 2', view = playerObservation(first, 'p1');
        assert.deepEqual(view, playerObservation(second, 'p1'));
        const result = await searchBeliefWorlds(view, [capturePosition(first, 'fixture'), capturePosition(second, 'fixture')],
            {seed: [371, 1, 2, 3], candidates: 4, responses: 1, chance_samples: 2,
                continuation_choices: 1, max_nodes: 64, max_elapsed_ms: 2000}, null, harmful);
        const analysis = analyzeDecision(view, result, harmful);
        assert(analysis.intervention, analysis.reason); assert(analysis.paired_gain.worst > 0);
        assert.notEqual(analysis.selected_command, harmful); assert.equal(analysis.strength_claim, false);
    } finally { first.destroy(); second.destroy(); }
});
