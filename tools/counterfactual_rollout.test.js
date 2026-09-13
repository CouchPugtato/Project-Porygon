'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = fs.promises;
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
    acquireRunLock,
    buildCounterfactualSample,
    choicesDiffer,
    equalPrefix,
    recoverManifestFromPairFiles,
    seedTuple,
    writeJsonAtomic,
} = require('./counterfactual_rollout');

function sampleAction() {
    return {
        counterfactual: {decision_index: 3, action_rank: 1},
        action: 2,
        action2: 17,
        slot0_has_action: 1,
        slot0_kind: 0,
        slot0_move_index: 2,
        slot0_switch_index: 0,
        slot0_use_tera: 0,
        slot0_target_index: 1,
        slot0_target_mask: 3,
        slot1_has_action: 1,
        slot1_kind: 0,
        slot1_move_index: 3,
        slot1_switch_index: 0,
        slot1_use_tera: 0,
        slot1_target_index: 0,
        slot1_target_mask: 1,
        baseline_value: 0.2,
        behavior_log_prob: -2.3,
        hidden_dim: 3,
        hidden_state: [0.1, 0.2, 0.3],
        legal_mask: Array(28).fill(1),
    };
}

test('counterfactual records store the compact recurrent state and intervention', () => {
    const record = buildCounterfactualSample(sampleAction(), 'pair-7', 'parent.chk', -1);
    assert.equal(record.type, 'counterfactual_sample');
    assert.equal(record.pair_id, 'pair-7');
    assert.equal(record.action_rank, 1);
    assert.equal(record.decision_index, 3);
    assert.equal(record.action2, 17);
    assert.equal(record.target_value, -1);
    assert.deepEqual(record.hidden_state, [0.1, 0.2, 0.3]);
    assert.equal(record.legal_mask.length, 28);
});

test('paired-state and action comparisons are strict', () => {
    assert.equal(equalPrefix([1, 2, 9], [1, 2, 8], 2), true);
    assert.equal(equalPrefix([1, 2], [1, 3], 2), false);
    assert.equal(choicesDiffer(
        {action: 1, action2: 15, slot0_target: 0, slot1_target: 1},
        {action: 1, action2: 15, slot0_target: 0, slot1_target: 2}
    ), true);
});

test('manifest recovery keeps complete pairs and skips crash-damaged files', async t => {
    const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'porygon-counterfactual-'));
    t.after(() => fsp.rm(root, {recursive: true, force: true}));
    const pairs = path.join(root, 'pairs');
    await fsp.mkdir(pairs);
    const rank0Action = sampleAction();
    rank0Action.counterfactual.action_rank = 0;
    const rank0 = buildCounterfactualSample(
        rank0Action, 'counterfactual-pair-0', 'parent.chk', 1);
    const rank1Action = sampleAction();
    rank1Action.action = 4;
    const rank1 = buildCounterfactualSample(
        rank1Action, 'counterfactual-pair-0', 'parent.chk', -1);
    await fsp.writeFile(
        path.join(pairs, 'pair_000000.jsonl'),
        `${JSON.stringify(rank0)}\n${JSON.stringify(rank1)}\n`
    );
    await fsp.writeFile(path.join(pairs, 'pair_000002.jsonl'), Buffer.alloc(128));

    const options = {
        runName: 'recovery-test', format: 'gen9randomdoublesbattle',
        checkpoint: 'parent.chk', opponentCheckpoint: 'opponent.chk', pairs: 5,
        seed: 7, decisionMin: 0, decisionMax: 7, battleTimeoutSeconds: 180,
    };
    const manifest = await recoverManifestFromPairFiles(pairs, options);
    assert.equal(manifest.completed_pairs, 1);
    assert.equal(manifest.attempts, 3);
    assert.equal(manifest.invalid_attempts, 2);
    assert.equal(manifest.rank0_better, 1);
    assert.equal(manifest.recovery.unreadable_pair_files, 1);
});

test('atomic JSON writes retain the previous valid document as a backup', async t => {
    const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'porygon-manifest-'));
    t.after(() => fsp.rm(root, {recursive: true, force: true}));
    const target = path.join(root, 'manifest.json');
    await writeJsonAtomic(target, {generation: 1});
    await writeJsonAtomic(target, {generation: 2});
    assert.deepEqual(JSON.parse(await fsp.readFile(target, 'utf8')), {generation: 2});
    assert.deepEqual(JSON.parse(await fsp.readFile(`${target}.bak`, 'utf8')), {generation: 1});
    await writeJsonAtomic(target, {generation: 3});
    assert.deepEqual(JSON.parse(await fsp.readFile(target, 'utf8')), {generation: 3});
    assert.deepEqual(JSON.parse(await fsp.readFile(`${target}.bak`, 'utf8')), {generation: 2});
});

test('run lock excludes a second collector and releases cleanly', async t => {
    const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'porygon-run-lock-'));
    t.after(() => fsp.rm(root, {recursive: true, force: true}));
    const lock = await acquireRunLock(root);
    await assert.rejects(
        acquireRunLock(root),
        /another counterfactual collector is already running/
    );
    await lock.release();
    const nextLock = await acquireRunLock(root);
    await nextLock.release();
});

test('Showdown seed tuples are deterministic and attempt-specific', () => {
    assert.equal(seedTuple(17, 4, 2), seedTuple(17, 4, 2));
    assert.notEqual(seedTuple(17, 4, 2), seedTuple(17, 5, 2));
    assert.match(seedTuple(17, 4, 2), /^\d+,\d+,\d+,\d+$/);
});
