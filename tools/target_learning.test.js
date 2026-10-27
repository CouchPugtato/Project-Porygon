'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const {createTargetCollector, packDataset, publishLabels, readLabels, splitBattle, validateLabels} = require('./target_learning');
const {validateConfig} = require('./tactical_teacher_benchmark');
const {createBattleAgents, battleSeeds} = require('./tactical_teacher_smoke');

function labels() {
    return {parent_fingerprint: '1234567890abcdef', hidden: [.2, -.1],
        examples: [{slot: 0, move: 0, legal: 3, best: 1}]};
}

test('target collector retains accepted choices only, with terminal accounting', () => {
    const collector = createTargetCollector();
    const action = {request_id: 3, command: '/choose move 1 1', training_eligible: false, target_labels: labels()};
    collector.propose(action);
    collector.observe(JSON.stringify({type: 'decision_rejected', request_id: 3, command: action.command}));
    collector.propose(action);
    collector.observe(JSON.stringify({type: 'decision_accepted', request_id: 3, command: action.command}));
    assert.equal(collector.finish({policy_tag: 'target-labels-v1', training_eligible: false, label_decisions: 1}).length, 1);
    assert.throws(() => validateLabels({...labels(), examples: [{slot: 0, move: 0, legal: 3, best: 3}]}), /uninformative/);
    assert.throws(() => validateLabels({...labels(), hidden: [NaN]}), /hidden/);
});

test('target cache packs a fixed whole-battle split and deterministic example order', () => {
    const battles = Array.from({length: 400}, (_, i) => ({battle_id: `battle-${i}`,
        decisions: [{request_id: 1, ...labels()}, {request_id: 2, ...labels()}]}));
    const packed = packDataset(battles, 20261026);
    assert.equal(packed.counts.examples, 800);
    assert.equal(packed.bytes.subarray(0, 9).toString(), 'PGTLABEL1');
    assert.equal(packed.bytes.readUInt32LE(9), 2);
    assert.equal(packed.bytes.readUInt32LE(13), 800);
    assert.equal(packed.counts.holdout, battles.filter(b => splitBattle(b.battle_id, 20261026)).length * 2);
    assert.deepEqual(packDataset([...battles].reverse(), 20261026).bytes, packed.bytes);
    assert.throws(() => packDataset([...battles, battles[0]], 20261026), /duplicate battle/);
    battles[0].decisions[0].parent_fingerprint = '0000000000000001';
    assert.throws(() => packDataset(battles, 20261026), /different parents/);
});

test('cached labels resume only with intact accepted-choice files', async t => {
    const fs = require('node:fs/promises'), path = require('node:path'), os = require('node:os');
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-target-labels-'));
    t.after(() => fs.rm(directory, {recursive: true, force: true}));
    const game = {result: {result: 'loss'}, target_examples: [{request_id: 1, ...labels()}]};
    const reference = await publishLabels(directory, game, 'parent', 0);
    assert.equal(game.target_examples, undefined);
    assert.equal(reference.examples, 1);
    assert.equal((await readLabels(directory, reference)).result.result, 'loss');
    await fs.appendFile(path.join(directory, reference.path), ' ');
    await assert.rejects(readLabels(directory, reference), /checksum/);
    await fs.unlink(path.join(directory, reference.path));
    await assert.rejects(readLabels(directory, reference), {code: 'ENOENT'});
    await assert.rejects(readLabels(directory, {...reference, path: '../wrong.json'}), /missing target/);
});

test('collection freezes student play and routes passive labels to the candidate only', () => {
    const config = require('../config/experimental/target_learning_collection.json');
    validateConfig(config);
    assert.equal(config.games_per_side * 4, 500);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(config, battleSeeds(config.seed, side, false), side, 'parent', null);
        assert.equal(agents[side].targetOptions.labels, true);
        assert.equal(agents[1 - side].targetOptions.labels, false);
        assert.equal(agents[side].tacticalTeacher, false);
        assert.equal(agents[side].tacticalTargets, false);
    }
    for (const change of [{tactical_targets: true}, {collect_demonstrations: true}, {candidate_inference_policy: 'greedy'}])
        assert.throws(() => validateConfig({...config, ...change}), /target collection/);
});

test('learned targets route only to the candidate and reject other overrides', () => {
    const config = {...require('../config/experimental/target_learning_collection.json'),
        collect_target_labels: false, target_sidecar: 'models/example/target_sidecar.bin'};
    validateConfig(config);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(config, battleSeeds(config.seed, side, true), side, 'parent', null);
        assert.ok(agents[side].targetOptions.sidecar.endsWith('target_sidecar.bin'));
        assert.equal(agents[1 - side].targetOptions.sidecar, null);
        assert.equal(agents[side].tacticalTargets, false);
        assert.equal(agents[side].tacticalTeacher, false);
        assert.equal(agents[side].targetOptions.labels, false);
    }
    for (const change of [{collect_target_labels: true}, {tactical_targets: true}, {candidate_inference_policy: 'greedy'}])
        assert.throws(() => validateConfig({...config, ...change}), /learned target/);
});
