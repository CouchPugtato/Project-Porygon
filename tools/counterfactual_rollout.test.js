'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = fs.promises;
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
    acquireRunLock,
    appendJsonLineDurably,
    battleFailureReason,
    buildCounterfactualSample,
    choiceRejectionReason,
    choicesDiffer,
    drivePlayer,
    equalPrefix,
    recoverManifestFromPairFiles,
    seedTuple,
    summarizeFailureLog,
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

test('choice errors are rejected and retried before accepting a decision', async () => {
    const actions = [
        {type: 'action', action: 8, action2: 14, command: '/choose switch 3, move 1 -1'},
        {type: 'action', action: 0, action2: 14, command: '/choose move 1 1, move 1 -1'},
    ];
    const sent = [];
    const writes = [];
    const agent = {
        send(message) {
            sent.push(message);
        },
        async nextMessage(type) {
            if (type === 'action') return actions.shift();
            if (type === 'episode_complete') return {steps: 1};
            throw new Error(`unexpected message request: ${type}`);
        },
    };
    const chunks = [
        '|request|{"active":[{},{}]}\n',
        '|error|[Invalid choice] Can\'t switch: The active Pokémon is trapped\n' +
            '|request|{"active":[{"trapped":true},{}]}\n',
        '|move|p1a: A|Tackle|p2a: B\n|win|Candidate\n',
    ];
    const stream = {
        async *[Symbol.asyncIterator]() {
            for (const chunk of chunks) yield chunk;
        },
        async write(command) {
            writes.push(command);
        },
    };

    const result = await drivePlayer(
        stream, agent, 'retry-test', 'Candidate', 'gen9randomdoublesbattle');
    const decisions = sent.filter(message => message.type.startsWith('decision_'));
    assert.deepEqual(decisions.map(message => message.type), [
        'decision_rejected',
        'decision_accepted',
    ]);
    assert.deepEqual(writes, [
        'switch 3, move 1 -1',
        'move 1 1, move 1 -1',
    ]);
    assert.equal(result.choiceRejections.length, 1);
    assert.equal(result.choiceRejections[0].request_id, 0);
});

test('failure categories and durable JSONL retain exact messages', async t => {
    assert.equal(choiceRejectionReason("Can't switch: The active Pokémon is trapped"),
        'trapped_switch');
    assert.equal(choiceRejectionReason("Can't move: Helping Hand needs a target"),
        'missing_move_target');
    assert.equal(battleFailureReason(new Error('battle branch exceeded timeout')),
        'battle_timeout');
    assert.equal(battleFailureReason(new Error('showdown_client exited with 1')),
        'agent_process_error');

    const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'porygon-failure-log-'));
    t.after(() => fsp.rm(root, {recursive: true, force: true}));
    const target = path.join(root, 'failures.jsonl');
    await appendJsonLineDurably(target, {
        type: 'attempt_failure', attempt: 7, reason: 'battle_error', message: 'exact error',
    });
    const record = JSON.parse((await fsp.readFile(target, 'utf8')).trim());
    assert.equal(record.attempt, 7);
    assert.equal(record.message, 'exact error');
    await appendJsonLineDurably(target, {
        type: 'choice_rejection_recovered', attempt: 8, action_rank: 0,
        role: 'candidate', player: 'A', request_id: 4, command: '/choose switch 3',
        reason: 'trapped_switch', message: "Can't switch: trapped",
    });
    const summary = await summarizeFailureLog(target);
    assert.equal(summary.logged_attempt_failures, 1);
    assert.deepEqual(summary.logged_failure_reasons, {battle_error: 1});
    assert.equal(summary.choice_rejections_recovered, 1);
    assert.deepEqual(summary.choice_rejection_reasons, {trapped_switch: 1});
});

test('Showdown seed tuples are deterministic and attempt-specific', () => {
    assert.equal(seedTuple(17, 4, 2), seedTuple(17, 4, 2));
    assert.notEqual(seedTuple(17, 4, 2), seedTuple(17, 5, 2));
    assert.match(seedTuple(17, 4, 2), /^\d+,\d+,\d+,\d+$/);
});
