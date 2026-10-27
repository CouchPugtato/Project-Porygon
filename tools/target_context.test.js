'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {validateContext, writeContext, verifyReproduction, contextFitEligible} = require('./target_context');
const {createTargetCollector, packDataset} = require('./target_learning');
const {validateConfig} = require('./tactical_teacher_benchmark');
const {createBattleAgents, battleSeeds} = require('./tactical_teacher_smoke');

function context() {
    return {version: 1, move_id: 935, move_type_id: 2, move_type_known: true,
        targets: Array.from({length: 4}, (_, position) => ({position, present: true, friendly: position < 2,
            type1_id: position + 1, type1_known_mode: 2, type2_id: 0, type2_known_mode: 2}))};
}

test('evaluation export requires exact replay, not just a better loss or epoch', () => {
    const reference = {weights_published: false, gameplay_authorized: false, parent_unchanged: true,
        context_lower_validation_nll: true, type_pair_residual: {best_epoch: 10, validation: {nll: .55}},
        provenance: {dataset_sha256: 'recorded'}};
    const {provenance, ...replay} = reference;
    verifyReproduction(reference, replay);
    verifyReproduction(reference, {...replay, weights_published: true}, true);
    assert.throws(() => verifyReproduction(reference, {...replay, weights_published: true}), /exactly/);
    assert.throws(() => verifyReproduction(reference, {...replay, gameplay_authorized: true}), /exactly/);
    assert.throws(() => verifyReproduction(reference, {...replay,
        type_pair_residual: {best_epoch: 11, validation: {nll: .54}}}), /exactly/);
});

test('context evaluation is explicitly routed and cannot reuse legacy fit eligibility', () => {
    const old = require('../config/experimental/target_exploratory_learned.json');
    const config = {...old, context_target_evaluation: true, exploratory_target_evaluation: false};
    validateConfig(config);
    assert.throws(() => validateConfig({...config, exploratory_target_evaluation: true}), /distinct/);
    assert.throws(() => validateConfig({...config, target_sidecar: undefined}), /distinct/);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(config, battleSeeds(config.seed, side, true), side, 'parent', null);
        assert.equal(agents[side].targetOptions.contextual, true);
        assert.equal(agents[1 - side].targetOptions.contextual, false);
    }
    const metadata = {format: 'PGTARGTC1', evaluation_only: true, reproduced_exactly: true};
    const report = {evaluation_only: true, reproduced_exactly: true, parent_unchanged: true,
        weights_published: true, gameplay_authorized: false, context_lower_validation_nll: true};
    assert.equal(contextFitEligible(metadata, report), true);
    for (const key of ['evaluation_only', 'reproduced_exactly', 'parent_unchanged', 'weights_published', 'context_lower_validation_nll'])
        assert.equal(contextFitEligible(metadata, {...report, [key]: false}), false);
    assert.equal(contextFitEligible({...metadata, format: 'PGTARGET1'}, report), false);
    const smoke = require('../config/experimental/target_context_smoke.json');
    validateConfig(smoke);
    assert.equal(smoke.games_per_side * smoke.opponents.length * 2, 4);
    assert.equal(smoke.agent_omp_threads, 1);
});

test('visible context uses a stable 32-byte layout and rejects unredacted or mislocated targets', () => {
    const c = context(), bytes = Buffer.alloc(32);
    validateContext(c);
    assert.equal(writeContext(bytes, 0, c), 32);
    assert.equal(bytes.readUInt16LE(0), c.move_id);
    assert.equal(bytes[2], c.move_type_id);
    assert.equal(bytes[4 + 2 * 7], 2);
    c.targets[2].type1_known_mode = 0;
    assert.throws(() => validateContext(c), /unredacted/);
    c.targets[2].type1_id = 0;
    validateContext(c);
    c.targets[3].position = 0;
    assert.throws(() => validateContext(c), /visible-target/);
});

test('context collection cannot accept a legacy label action as context-complete', () => {
    const action = {request_id: 1, command: '/choose move 1 1', training_eligible: false,
        target_labels: {parent_fingerprint: '1234567890abcdef', hidden: [0, 0],
            examples: [{slot: 0, move: 0, legal: 12, best: 4}]}};
    assert.throws(() => createTargetCollector(1).propose(action), /required target context/);
    action.target_labels.examples[0].context = context();
    const collector = createTargetCollector(1);
    collector.propose(action);
    collector.observe(JSON.stringify({type: 'decision_accepted', request_id: 1, command: action.command}));
    assert.equal(collector.finish({policy_tag: 'target-labels-v1', training_eligible: false, label_decisions: 1}).length, 1);
});

test('context and baseline caches retain identical examples, hidden states and battle splits', () => {
    const battles = Array.from({length: 400}, (_, i) => ({battle_id: `context-${i}`, decisions: [{request_id: 1,
        parent_fingerprint: '1234567890abcdef', hidden: [.2, -.3], examples: [{slot: 0, move: i % 4, legal: 12, best: 4, context: context()}]}]}));
    const baseline = packDataset(battles, 20261026), enriched = packDataset(battles, 20261026, true);
    assert.deepEqual(enriched.counts, baseline.counts);
    assert.equal(enriched.bytes.subarray(0, 9).toString(), 'PGTCONTX1');
    for (let i = 0; i < 400; ++i)
        assert.deepEqual(enriched.bytes.subarray(25 + i * 45, 25 + i * 45 + 13), baseline.bytes.subarray(25 + i * 13, 25 + (i + 1) * 13));
    delete battles[0].decisions[0].examples[0].context;
    assert.throws(() => packDataset(battles, 20261026, true), /selected-move context/);
    assert.equal(packDataset(battles, 20261026).counts.examples, 400);
});

test('fresh context collection stays frozen, serial and explicitly versioned', () => {
    const config = require('../config/experimental/target_context_collection.json');
    const old = require('../config/experimental/target_learning_collection.json');
    validateConfig(config);
    assert.deepEqual({...config, seed: old.seed, target_context_version: undefined}, {...old, target_context_version: undefined});
    assert.notEqual(config.seed, old.seed);
    assert.equal(config.games_per_side * 4, 500);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(config, battleSeeds(config.seed, side, false), side, 'parent', null);
        assert.equal(agents[side].targetOptions.contextVersion, 1);
        assert.equal(agents[1 - side].targetOptions.contextVersion, 0);
    }
    assert.throws(() => validateConfig({...config, collect_target_labels: false}), /visible target context/);
});
