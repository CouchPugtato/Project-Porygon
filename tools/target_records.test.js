'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {validateTrace, buildArtifact, validateArtifact, replayArtifact, collectTrace} = require('./target_records');
const clone = value => structuredClone(value);
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');

function fixture() {
    const bundle = {sha256: 'a'.repeat(64), manifest: {kind: 'frozen-target-action-actor-v1',
        sidecar_parameters: '123456789abcdef0', target_decoder: 'argmax', eligibility: 'slot-local-v1'}};
    const identity = {version: 2, kind: bundle.manifest.kind, parameter_fingerprint_algorithm: 'fnv1a64-native',
        reference_parameters: '1'.repeat(16), actor_parameters: '2'.repeat(16),
        sidecar_parameters: bundle.manifest.sidecar_parameters, manifest_sha256: bundle.sha256,
        context_version: 1, action_decoder: 'sample', target_decoder: 'argmax', eligibility: 'slot-local-v1'};
    const inputs = [{type: 'battle_start', battle_id: 'records-test'},
        {type: 'request', battle_id: 'records-test', request_id: 1, payload: {}},
        {type: 'decision_rejected', battle_id: 'records-test', request_id: 1, action: 8, action2: 23, command: '/choose switch 1, switch 2'},
        {type: 'request', battle_id: 'records-test', request_id: 2, payload: {}},
        {type: 'decision_accepted', battle_id: 'records-test', request_id: 2, action: 8, action2: 23, command: '/choose switch 1, switch 2'},
        {type: 'terminal', battle_id: 'records-test', reward: 1}, {type: 'battle_end', battle_id: 'records-test'}];
    const record = {type: 'bundle_decision', version: 1, battle_id: 'records-test', request_id: 2, decision_index: 1,
        training_eligible: false, behavior_policy: identity, action: 8, action2: 23, command: inputs[4].command,
        hidden_dim: 2, hidden_f32_hex: '800000003f000000', legal_mask: Array(28).fill(1),
        choice: [[1, 2, 0, 0, 0, 0, 0], [1, 2, 0, 1, 0, 0, 0]], public_context: [null, null],
        behavior: {version: 2, scope: 'full_turn_diagnostic', ppo_ready: false,
            full_turn_log_probability: -2, likelihood_status: 'supported', slots: [null, null]}};
    const terminal = {type: 'episode_complete', battle_id: 'records-test', training_eligible: false,
        behavior_policy: identity, target_decisions: 1, likelihood_replay: {accepted_supported: 1, accepted_unsupported: 0}};
    const outputs = [{type: 'ready', accepted_records: true, capabilities: {training: false}, behavior_policy: identity}, record, terminal];
    return {bundle, inputs, record, outputs};
}

async function directory(t) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-records-test-'));
    t.after(async () => {
        for (const name of await fs.readdir(dir)) await fs.unlink(path.join(dir, name));
        await fs.rmdir(dir);
    });
    return dir;
}

test('only accepted decisions become snapshots; signed-zero hidden bits survive JSON and replay', async () => {
    const f = fixture(), artifact = buildArtifact(f.bundle, f.inputs, f.outputs);
    assert.equal(artifact.records.length, 1);
    assert.equal(artifact.records[0].request_id, 2);
    assert.equal(Object.is(Buffer.from(artifact.records[0].hidden_f32_hex, 'hex').readFloatBE(), -0), true);
    const saved = JSON.parse(JSON.stringify(artifact));
    assert.deepEqual(await replayArtifact(saved, f.bundle, async () => f.outputs), {accepted: 1, skipped: 0, likelihood_ratio: 1});
});

test('duplicate outcomes, rejected snapshots and incomplete or mixed battles fail closed', () => {
    const f = fixture();
    const duplicate = clone(f.inputs); duplicate.splice(5, 0, duplicate[4]);
    assert.throws(() => validateTrace(duplicate), /outcome/);
    assert.throws(() => validateTrace(f.inputs.slice(0, -1)), /complete/);
    const mixed = clone(f.inputs); mixed[1].battle_id = 'other';
    assert.throws(() => validateTrace(mixed), /mixed/);
    const rejected = clone(f.outputs); rejected[1].request_id = 1;
    assert.throws(() => buildArtifact(f.bundle, f.inputs, rejected), /rejected/);
    assert.throws(() => buildArtifact(f.bundle, f.inputs, [...f.outputs.slice(0, 2), f.record, f.outputs[2]]), /duplicate/);
});

test('unsupported accepted decisions remain audit-only and terminal coverage is checked', () => {
    const f = fixture(), outputs = clone(f.outputs);
    outputs[1] = {type: 'bundle_decision_skipped', version: 1, battle_id: 'records-test', request_id: 2,
        decision_index: 1, training_eligible: false, behavior_policy: outputs[0].behavior_policy,
        reason: 'unsupported_likelihood_or_acceptance'};
    outputs[2].likelihood_replay = {accepted_supported: 0, accepted_unsupported: 1};
    const artifact = buildArtifact(f.bundle, f.inputs, outputs);
    assert.equal(artifact.records.length, 0); assert.equal(artifact.skipped.length, 1);
    outputs[1].hidden_f32_hex = '00000000';
    assert.throws(() => buildArtifact(f.bundle, f.inputs, outputs), /usable/);
    assert.throws(() => buildArtifact(f.bundle, f.inputs, f.outputs.slice(0, 1).concat(f.outputs[2])), /coverage/);
});

test('illegal masks, nonfinite hidden values, altered likelihood and changed identities are rejected', () => {
    const f = fixture();
    for (const mutate of [r => { r.legal_mask[8] = 0; }, r => { r.hidden_f32_hex = '7f80000000000000'; },
        r => { r.behavior.full_turn_log_probability = NaN; }, r => { r.behavior_policy.actor_parameters = 'bad'; },
        r => { r.public_context[0] = {}; }]) {
        const outputs = clone(f.outputs); mutate(outputs[1]);
        assert.throws(() => buildArtifact(f.bundle, f.inputs, outputs));
    }
});

test('selected-move context and target probabilities retain only legal target support', () => {
    const f = fixture();
    f.inputs[4].action = 0; f.inputs[4].command = '/choose move 1 1, switch 2';
    f.record.action = 0; f.record.command = f.inputs[4].command;
    f.record.choice[0] = [1, 1, 0, 0, 0, 2, 12];
    f.record.public_context[0] = {version: 1, move_id: 1, move_type_id: 1, move_type_known: true,
        targets: [0, 1, 2, 3].map(position => ({position, present: true, friendly: position < 2,
            type1_id: 1, type2_id: 0, type1_known_mode: 1, type2_known_mode: 0}))};
    f.record.behavior.slots[0] = {source: 'sidecar', legal_mask: 12, selected: 2,
        probabilities: [0, 0, 1, 0], selected_probability: 1};
    assert.equal(buildArtifact(f.bundle, f.inputs, f.outputs).records.length, 1);
    f.record.behavior.slots[0].probabilities = [.1, 0, .9, 0];
    assert.throws(() => buildArtifact(f.bundle, f.inputs, f.outputs), /target probabilities/);
    f.record.behavior.slots[0].probabilities = [0, 0, 1, 0];
    f.record.public_context[0].targets[2].position = 0;
    assert.throws(() => buildArtifact(f.bundle, f.inputs, f.outputs), /context/);
});

test('atomic battle resume does not duplicate or rewrite; interrupted temporary output is not credited', async t => {
    const f = fixture(), dir = await directory(t);
    let executions = 0;
    const execute = async () => { executions++; return f.outputs; };
    const first = await collectTrace(f.bundle, f.inputs, dir, execute);
    const before = await fs.readFile(first.filename, 'utf8');
    const again = await collectTrace(f.bundle, f.inputs, dir, execute);
    assert.equal(again.reused, true); assert.equal(executions, 2);
    assert.equal(await fs.readFile(first.filename, 'utf8'), before);
    assert.equal((await fs.readdir(dir)).filter(x => /^[a-f0-9]{64}\.json$/.test(x)).length, 1);
    // Simulate loss before the atomic rename, leaving only the uncommitted temp.
    await fs.rename(first.filename, first.filename + '.tmp');
    const recovered = await collectTrace(f.bundle, f.inputs, dir, execute);
    assert.equal(recovered.reused, false); assert.equal(executions, 3);
    assert.equal(await fs.readFile(first.filename, 'utf8'), before);
});

test('resume refuses another bundle or changed trace; saved corruption and replay differences fail', async t => {
    const f = fixture(), dir = await directory(t);
    const first = await collectTrace(f.bundle, f.inputs, dir, async () => f.outputs);
    await assert.rejects(collectTrace({...f.bundle, sha256: 'b'.repeat(64)}, f.inputs, dir, async () => f.outputs), /mix bundle/);
    const changed = clone(f.inputs); changed[5].reward = -1;
    await assert.rejects(collectTrace(f.bundle, changed, dir, async () => f.outputs), /different trace/);
    const artifact = JSON.parse(await fs.readFile(first.filename)); artifact.records[0].hidden_f32_hex = '000000003f000000';
    assert.throws(() => validateArtifact(artifact, f.bundle), /hash/);
    const {content_sha256, ...body} = artifact; artifact.content_sha256 = hash(body);
    await assert.rejects(replayArtifact(artifact, f.bundle, async () => f.outputs), /differ on replay/);
    const corruptOutputs = clone(f.outputs); corruptOutputs[1].behavior.full_turn_log_probability = -3;
    await assert.rejects(collectTrace(f.bundle, f.inputs, dir, async () => corruptOutputs), /differ on replay/);
    assert.equal(JSON.parse(await fs.readFile(first.filename)).content_sha256, content_sha256);
});
