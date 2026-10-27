'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const {validateConfig, prepareRecords, packRecords, runDiagnostic, readGeneration} = require('./action_record_learner');
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const bytesHash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function fixture(t) {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-action-learner-test-'));
    t.after(async () => {
        // Every path belongs to this test-created directory, including nested
        // staging/generation folders. Never remove a repository run directory.
        await fs.rm(directory, {recursive: true});
    });
    const config = {run_name: 'toy', manifest: 'unused', manifest_sha256: 'a'.repeat(64), sources: [],
        record_limit: 4, minibatch: 1, max_updates: 4, learning_rate: .00001, max_abs_log_ratio: .02};
    const bundle = {sha256: config.manifest_sha256, manifest: {kind: 'frozen-target-action-actor-v1',
        sidecar_parameters: '1'.repeat(16), target_decoder: 'argmax', eligibility: 'slot-local-v1'}};
    const identity = {version: 2, kind: bundle.manifest.kind, parameter_fingerprint_algorithm: 'fnv1a64-native',
        reference_parameters: '2'.repeat(16), actor_parameters: '2'.repeat(16), sidecar_parameters: '1'.repeat(16),
        manifest_sha256: bundle.sha256, context_version: 1, action_decoder: 'sample', target_decoder: 'argmax', eligibility: 'slot-local-v1'};
    const inputs = [{type: 'battle_start', battle_id: 'toy-records'}], records = [];
    for (let i = 0; i < 4; ++i) {
        inputs.push({type: 'request', battle_id: 'toy-records', request_id: i, payload: {}});
        inputs.push({type: 'decision_accepted', battle_id: 'toy-records', request_id: i, action: 8, action2: 23, command: '/choose switch 1, switch 2'});
        records.push({type: 'bundle_decision', version: 1, battle_id: 'toy-records', request_id: i, decision_index: i + 1,
            training_eligible: false, behavior_policy: identity, action: 8, action2: 23, command: '/choose switch 1, switch 2',
            hidden_dim: 2, hidden_f32_hex: '800000003f000000', legal_mask: Array(28).fill(1),
            choice: [[1, 2, 0, 0, 0, 0, 0], [1, 2, 0, 1, 0, 0, 0]], public_context: [null, null],
            behavior: {version: 2, scope: 'full_turn_diagnostic', ppo_ready: false,
                full_turn_log_probability: -2, likelihood_status: 'supported', slots: [null, null]}});
    }
    inputs.push({type: 'terminal', battle_id: 'toy-records', result: 'win', reward: 1}, {type: 'battle_end', battle_id: 'toy-records'});
    const body = {format: 'porygon-frozen-bundle-records-v1', training_eligible: false, manifest_sha256: bundle.sha256,
        input_sha256: hash(inputs), battle_id: 'toy-records', identity, inputs, records, skipped: [], terminal: {}};
    const artifact = {...body, content_sha256: hash(body)};
    const source = path.join(directory, 'source.json'), bytes = JSON.stringify(artifact);
    await fs.writeFile(source, bytes);
    config.sources = [{path: source, sha256: bytesHash(bytes)}];
    const output = path.join(directory, 'output'); await fs.mkdir(output);
    return {directory, output, config, bundle, source, artifact};
}
const replay = async () => ({likelihood_ratio: 1});

function identity(config, data) { return {config, dataset_sha256: hash(data), executable_sha256: 'toy'}; }
function executor({fail = false, guard = false, seen = []} = {}) {
    return async ({previous, stage}) => {
        const input = await fs.readFile(path.join(stage, 'input.bin'));
        const cursor = input.readUInt32LE(21), updates = input.readUInt32LE(25), limit = input.readUInt32LE(29), invocation = input.readUInt32LE(33);
        seen.push(cursor);
        if (previous) assert.equal(await fs.readFile(previous.optimizer, 'utf8'), `moments:${updates}`);
        const next = guard ? updates : Math.min(limit, updates + invocation);
        await fs.writeFile(path.join(stage, 'actor.chk'), `actor:${next}`);
        if (fail) throw new Error('simulated crash after actor write');
        await fs.writeFile(path.join(stage, 'actor.opt'), `moments:${next}`);
        return {diagnostic: 'action_record_learner', training_eligible: false, ppo_ready: false,
            cursor: next, updates: next, guard_stopped: guard, max_abs_log_ratio: .001 * next};
    };
}

test('diagnostic bounds reject large runs, unknown settings and missing bundle/source pins', async t => {
    const f = await fixture(t); validateConfig(f.config);
    for (const change of [{record_limit: 129}, {max_updates: 9}, {minibatch: 9}, {learning_rate: .001},
        {max_abs_log_ratio: .1}, {shuffle: true}, {sources: []}, {manifest_sha256: ''}])
        assert.throws(() => validateConfig({...f.config, ...change}), /bounded/);
});

test('accepted records require pinned sources and successful full-bundle replay before packing', async t => {
    const f = await fixture(t); let replays = 0;
    const data = await prepareRecords(f.config, f.bundle, async () => { replays++; return replay(); });
    assert.equal(replays, 1); assert.equal(data.rows.length, 4); assert.equal(data.rows[0].terminal_return, 1);
    await assert.rejects(prepareRecords(f.config, f.bundle, async () => { throw Error('likelihood mismatch'); }), /likelihood mismatch/);
    await assert.rejects(prepareRecords({...f.config, sources: [...f.config.sources, ...f.config.sources]}, f.bundle, replay), /duplicate battle/);
    await fs.appendFile(f.source, ' ');
    await assert.rejects(prepareRecords(f.config, f.bundle, replay), /SHA256/);
});

test('packing preserves signed-zero hidden bits, fixed choice fields, target log and resume position', async t => {
    const f = await fixture(t), data = await prepareRecords(f.config, f.bundle, replay);
    const packed = packRecords(data, f.config, {cursor: 2, updates: 2}, 1);
    assert.equal(packed.subarray(0, 9).toString(), 'PGACTREC1');
    assert.equal(packed.readUInt32LE(21), 2); assert.equal(packed.readUInt32LE(25), 2);
    assert.equal(packed.readUInt32LE(33), 1); assert.equal(packed.subarray(49, 113).toString(), f.bundle.sha256);
    assert.equal(packed.readUInt32LE(113), 0x80000000); assert.equal(packed.readFloatLE(117), .5);
    assert.equal(packed.readDoubleLE(113 + 8 + 28 + 14 + 4), -2);
    assert.equal(packed.readDoubleLE(113 + 8 + 28 + 14 + 12), 0);
});

test('committed actor/optimizer/cursor resume matches uninterrupted execution and does not revisit records', async t => {
    const f = await fixture(t), data = await prepareRecords(f.config, f.bundle, replay), id = identity(f.config, data), seen = [];
    const first = await runDiagnostic(f.config, f.bundle, data, f.output, id, 2, executor({seen}));
    assert.equal(first.progress.cursor, 2);
    const resumed = await runDiagnostic(f.config, f.bundle, data, f.output, id, 2, executor({seen}));
    assert.deepEqual(seen, [0, 2]); assert.equal(resumed.progress.updates, 4);
    const done = await runDiagnostic(f.config, f.bundle, data, f.output, id, 2, async () => { throw Error('completed run must not update'); });
    assert.deepEqual(done, resumed);
    const other = path.join(f.directory, 'uninterrupted'); await fs.mkdir(other);
    const whole = await runDiagnostic(f.config, f.bundle, data, other, id, 4, executor());
    assert.deepEqual(whole.progress, resumed.progress);
    assert.equal(whole.files['actor.chk'], resumed.files['actor.chk']); assert.equal(whole.files['actor.opt'], resumed.files['actor.opt']);
});

test('crashes before commit preserve the previous cursor; orphan staging is not credited', async t => {
    const f = await fixture(t), data = await prepareRecords(f.config, f.bundle, replay), id = identity(f.config, data), seen = [];
    await runDiagnostic(f.config, f.bundle, data, f.output, id, 1, executor());
    const committed = await fs.readFile(path.join(f.output, 'current.json'), 'utf8');
    await assert.rejects(runDiagnostic(f.config, f.bundle, data, f.output, id, 1, executor({fail: true})), /simulated crash/);
    assert.equal(await fs.readFile(path.join(f.output, 'current.json'), 'utf8'), committed);
    await runDiagnostic(f.config, f.bundle, data, f.output, id, 1, executor({seen}));
    assert.deepEqual(seen, [1]);
    assert.ok((await fs.readdir(f.output)).some(n => n.startsWith('.pending-')));
});

test('changed dataset/settings, corrupted generations and guard stops cannot silently resume', async t => {
    const f = await fixture(t), data = await prepareRecords(f.config, f.bundle, replay), id = identity(f.config, data);
    await runDiagnostic(f.config, f.bundle, data, f.output, id, 1, executor());
    const changed = {...f.config, learning_rate: .00002};
    await assert.rejects(runDiagnostic(changed, f.bundle, data, f.output, identity(changed, data), 1, executor()), /inputs\/settings changed/);
    const pointer = JSON.parse(await fs.readFile(path.join(f.output, 'current.json')));
    const generation = await readGeneration(f.output, pointer, id);
    await fs.appendFile(generation.optimizer, 'corrupt');
    await assert.rejects(runDiagnostic(f.config, f.bundle, data, f.output, id, 1, executor()), /file mismatch/);
    const guarded = path.join(f.directory, 'guarded'); await fs.mkdir(guarded);
    const stopped = await runDiagnostic(f.config, f.bundle, data, guarded, id, 1, executor({guard: true}));
    assert.equal(stopped.progress.cursor, 0); assert.equal(stopped.progress.guard_stopped, true);
    assert.deepEqual(await runDiagnostic(f.config, f.bundle, data, guarded, id, 1, async () => { throw Error('must not retry stopped update'); }), stopped);
});
