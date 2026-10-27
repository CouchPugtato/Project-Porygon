'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const {validateManifest, loadManifest, stageMembers, nativeArguments, parseArgs} = require('./target_bundle');

const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function fixture(t) {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-bundle-test-'));
    const manifest = {version: 1, kind: 'frozen-target-action-actor-v1', sidecar_parameters: '123456789abcdef0',
        context_version: 1, action_decoder: 'sample', target_decoder: 'argmax', eligibility: 'slot-local-v1',
        inference_seed: 31601, omp_threads: 1, training_eligible: false};
    const filenames = [];
    t.after(async () => { for (const filename of filenames) await fs.rm(filename, {force: true}); await fs.rmdir(directory); });
    for (const member of ['reference', 'actor', 'sidecar', 'executable']) {
        const bytes = `fixture ${member}`;
        const filename = path.join(directory, `${member}.bin`); filenames.push(filename);
        await fs.writeFile(filename, bytes);
        manifest[member] = {path: `${member}.bin`, sha256: sha(bytes)};
    }
    const filename = path.join(directory, 'bundle.json'); filenames.push(filename);
    const bytes = JSON.stringify(manifest);
    await fs.writeFile(filename, bytes);
    return {directory, manifest, filename, pin: sha(bytes)};
}

test('bundle manifest pins all members and resolves paths relative to itself', async t => {
    const f = await fixture(t), bundle = await loadManifest(f.filename, f.pin);
    assert.equal(bundle.files.actor, path.join(f.directory, 'actor.bin'));
    assert.equal(bundle.sha256, f.pin);
    assert.throws(() => validateManifest({...f.manifest, training_eligible: true}), /unsupported/);
    assert.throws(() => validateManifest({...f.manifest, extra: true}), /fields/);
    for (const changes of [{omp_threads: 8}, {action_decoder: 'greedy'}, {target_decoder: 'other'},
        {eligibility: 'other'}, {context_version: 2}, {sidecar_parameters: '0000000000000000'}])
        assert.throws(() => validateManifest({...f.manifest, ...changes}), /unsupported/);
});

test('manifest pin rejects altered decoder or eligibility settings', async t => {
    const f = await fixture(t);
    await assert.rejects(loadManifest(f.filename), /pinned/);
    for (const changes of [{target_decoder: 'sample'}, {eligibility: 'whole-turn-v1'}]) {
        await fs.writeFile(f.filename, JSON.stringify({...f.manifest, ...changes}));
        await assert.rejects(loadManifest(f.filename, f.pin), /manifest SHA256 mismatch/);
    }
});

test('each changed artifact is rejected before process launch', async t => {
    const f = await fixture(t);
    for (const member of ['reference', 'actor', 'sidecar', 'executable']) {
        const filename = path.join(f.directory, `${member}.bin`), original = await fs.readFile(filename);
        await fs.writeFile(filename, 'changed');
        await assert.rejects(loadManifest(f.filename, f.pin), new RegExp(`${member} SHA256 mismatch`));
        await fs.writeFile(filename, original);
    }
});

test('staging pins copied bytes and detects source changes after initial validation', async t => {
    const f = await fixture(t), bundle = await loadManifest(f.filename, f.pin);
    const staged = await stageMembers(bundle);
    try {
        await fs.writeFile(bundle.files.actor, 'changed');
        assert.equal(sha(await fs.readFile(staged.files.actor)), f.manifest.actor.sha256);
        await assert.rejects(stageMembers(bundle), /changed while staging/);
    } finally { await staged.cleanup(); }
});

test('native routing is entirely manifest-derived and CLI overrides are rejected', async t => {
    const f = await fixture(t), bundle = await loadManifest(f.filename, f.pin);
    const args = nativeArguments(bundle, {reference: 'staged-reference', actor: 'staged-actor', sidecar: 'staged-sidecar'});
    assert.equal(args[args.indexOf('--battle-agent') + 1], 'staged-actor');
    assert.equal(args[args.indexOf('--action-bundle-reference') + 1], 'staged-reference');
    assert.equal(args[args.indexOf('--action-bundle-manifest-sha256') + 1], f.pin);
    assert.equal(args[args.indexOf('--target-slot-local') + 1], '1');
    assert.deepEqual(parseArgs(['--manifest', f.filename, '--manifest-sha256', f.pin]), {filename: f.filename, sha256: f.pin});
    assert.throws(() => parseArgs(['--manifest', f.filename, '--manifest-sha256', f.pin, '--target-decoder', 'sample']), /overrides/);
});
