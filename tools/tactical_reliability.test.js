'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {capturePosition, playerObservation, advancePosition, restorePosition} = require('./position_simulator');
const {compareOpeningTurns} = require('./opening_search');
const {proposeOpeningTurns} = require('./opening_proposals');
const {tacticalProbes, assertLegalTeams} = require('./fixtures/tactical_reliability');
const {assessTacticalProbe} = require('./tactical_reliability');

const OPTIONS = {seed: [328, 1, 2, 3], worlds: 1, candidates: 2, responses: 1, chance_samples: 4};
const SAMPLE = {mode: 'sample', seed: [328, 4, 5, 6]};

for (const probe of tacticalProbes()) test(`tactical reliability ${probe.name}`, async () => {
    const battle = probe.create();
    try {
        assertLegalTeams(battle);
        const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const original = position.state_json;
        const next = advancePosition(position, {p1: probe[probe.verify_choice || 'preferred'], p2: probe.response}, SAMPLE);
        const branch = restorePosition(next.position, SAMPLE);
        try { probe.verify(branch); } finally { branch.destroy(); }
        const result = await compareOpeningTurns(view, [{position, responses: [probe.response]}],
            [probe.preferred, probe.alternative], OPTIONS);
        const proposed = proposeOpeningTurns(view, 8, new PRNG(OPTIONS.seed.join(',')));
        const assessment = assessTacticalProbe(probe, result, proposed.commands);
        const allowed = probe.expectation === 'unscored_benefit' ? ['scoring_gap'] :
            probe.expectation === 'continuation_gap' ? ['continuation_gap'] : ['passed', 'candidate_probe_missing'];
        assert(allowed.includes(assessment.classification), JSON.stringify(assessment));
        assert.equal(position.state_json, original);
        assert.deepEqual(playerObservation(battle, 'p1'), view);
    } finally { battle.destroy(); }
});

test('tactical classification never ranks partial rows or treats an absent probe as proof of all missing good actions', () => {
    const probe = {preferred: 'a', alternative: 'b', expectation: 'strict_preference'};
    const row = (command, score, status = 'completed') => ({command, score, status});
    const result = {candidates: [row('a', 2), row('b', 1)], budget_exhausted: false};
    assert.equal(assessTacticalProbe(probe, result, []).classification, 'candidate_probe_missing');
    assert.equal(assessTacticalProbe(probe, result, ['a']).classification, 'passed');
    result.budget_exhausted = true;
    assert.equal(assessTacticalProbe(probe, result, ['a']).classification, 'budget_gap');
    result.budget_exhausted = false; result.candidates[0] = row('a', null, 'unsupported_continuation');
    assert.equal(assessTacticalProbe(probe, result, ['a']).classification, 'continuation_gap');
    result.candidates = [row('a', 0), row('b', 1)];
    assert.equal(assessTacticalProbe(probe, result, ['a']).classification, 'scoring_gap');
});
