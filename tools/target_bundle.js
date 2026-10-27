'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const {spawn} = require('node:child_process');
const {hashFile} = require('./tactical_teacher_smoke');
const {createResourceGate} = require('./counterfactual_rollout');

const ROOT = path.resolve(__dirname, '..');
const SHA256 = /^[a-f0-9]{64}$/;
const MEMBERS = ['reference', 'actor', 'sidecar', 'executable'];

function exactKeys(object, keys) {
    if (!object || Array.isArray(object) || typeof object !== 'object' ||
            Object.keys(object).length !== keys.length || keys.some(key => !Object.hasOwn(object, key)))
        throw new Error('unexpected or missing bundle fields');
}

function validateManifest(manifest) {
    exactKeys(manifest, ['version', 'kind', ...MEMBERS, 'sidecar_parameters', 'context_version',
        'action_decoder', 'target_decoder', 'eligibility', 'inference_seed', 'omp_threads', 'training_eligible']);
    if (manifest.version !== 1 || manifest.kind !== 'frozen-target-action-actor-v1' ||
            manifest.context_version !== 1 || manifest.action_decoder !== 'sample' ||
            !['sample', 'argmax'].includes(manifest.target_decoder) ||
            !['whole-turn-v1', 'slot-local-v1'].includes(manifest.eligibility) ||
            !Number.isSafeInteger(manifest.inference_seed) || manifest.inference_seed < 0 || manifest.inference_seed > 2147483647 ||
            manifest.omp_threads !== 1 || manifest.training_eligible !== false ||
            typeof manifest.sidecar_parameters !== 'string' || !/^[a-f0-9]{16}$/.test(manifest.sidecar_parameters) || /^0+$/.test(manifest.sidecar_parameters))
        throw new Error('unsupported bundle policy/settings');
    for (const member of MEMBERS) {
        exactKeys(manifest[member], ['path', 'sha256']);
        if (typeof manifest[member].path !== 'string' || !manifest[member].path ||
                manifest[member].path.includes('\0') || typeof manifest[member].sha256 !== 'string' || !SHA256.test(manifest[member].sha256))
            throw new Error(`invalid ${member} path/hash`);
    }
}

async function loadManifest(filename, expectedSha256) {
    if (typeof expectedSha256 !== 'string' || !SHA256.test(expectedSha256)) throw new Error('a pinned manifest SHA256 is required');
    const resolved = path.resolve(filename);
    if ((await fs.stat(resolved)).size > 65536) throw new Error('bundle manifest exceeds 64 KiB');
    const bytes = await fs.readFile(resolved);
    if (crypto.createHash('sha256').update(bytes).digest('hex') !== expectedSha256)
        throw new Error('manifest SHA256 mismatch');
    const manifest = JSON.parse(bytes);
    validateManifest(manifest);
    const files = {};
    for (const member of MEMBERS) {
        files[member] = path.resolve(path.dirname(resolved), manifest[member].path);
        if (await hashFile(files[member]) !== manifest[member].sha256) throw new Error(`${member} SHA256 mismatch`);
    }
    return {manifest, files, sha256: expectedSha256};
}

async function stageMembers(bundle) {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-target-bundle-'));
    const files = {};
    async function cleanup() {
        for (const filename of Object.values(files)) await fs.rm(filename, {force: true});
        await fs.rmdir(directory);
    }
    try {
        for (const member of ['reference', 'actor', 'sidecar']) {
            files[member] = path.join(directory, `${member}.bin`);
            await fs.copyFile(bundle.files[member], files[member]);
            if (await hashFile(files[member]) !== bundle.manifest[member].sha256)
                throw new Error(`${member} changed while staging`);
        }
        return {files, cleanup};
    } catch (error) { await cleanup(); throw error; }
}

function nativeArguments(bundle, staged, acceptedRecords = false) {
    const args = ['--battle-agent', staged.actor, '--target-sidecar', staged.sidecar,
        '--action-bundle-reference', staged.reference, '--action-bundle-sidecar-id', bundle.manifest.sidecar_parameters,
        '--action-bundle-manifest-sha256', bundle.sha256, '--inference-policy', 'sample',
        '--target-decoder', bundle.manifest.target_decoder, '--target-slot-local',
        bundle.manifest.eligibility === 'slot-local-v1' ? '1' : '0',
        '--inference-seed', String(bundle.manifest.inference_seed), '--reward-mode', 'terminal'];
    if (acceptedRecords) args.push('--action-bundle-records', '1');
    return args;
}

function parseArgs(argv) {
    if (argv.length !== 4 || argv[0] !== '--manifest' || argv[2] !== '--manifest-sha256')
        throw new Error('usage: node tools/target_bundle.js --manifest <file.json> --manifest-sha256 <SHA256>; policy overrides are not accepted');
    return {filename: argv[1], sha256: argv[3]};
}

async function startBundle(bundle, acceptedRecords = false) {
    const staged = await stageMembers(bundle);
    try {
        await createResourceGate({maxSystemMemoryPercent: 65, maxSystemCpuPercent: 65,
            resourceCheckSeconds: 2, launchStaggerMilliseconds: 750}).waitForLaunch();
        // Model inputs are staged and verified; pin the executable again just
        // before launch. This is local provenance checking, not a sandbox.
        if (await hashFile(bundle.files.executable) !== bundle.manifest.executable.sha256)
            throw new Error('executable changed before launch');
        const env = {...process.env, PORYGON_OMP_THREADS: '1', OMP_NUM_THREADS: '1'};
        delete env.PORYGON_REPLAY_PATH;
        delete env.PORYGON_DEMO_GRU;
        const child = spawn(bundle.files.executable, nativeArguments(bundle, staged.files, acceptedRecords),
            {cwd: ROOT, env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe']});
        return {child, cleanup: staged.cleanup};
    } catch (error) { await staged.cleanup(); throw error; }
}

async function runBundle(filename, sha256) {
    const bundle = await loadManifest(filename, sha256);
    const {child, cleanup} = await startBundle(bundle);
    try {
        child.stdout.pipe(process.stdout); child.stderr.pipe(process.stderr);
        const terminate = () => child.kill();
        process.once('SIGINT', terminate); process.once('SIGTERM', terminate);
        child.stdin.on('error', error => { if (error.code !== 'EPIPE') console.error(error.message); });
        process.stdin.pipe(child.stdin);
        try {
            return await new Promise((resolve, reject) => {
                child.once('error', reject);
                child.once('close', (code, signal) => resolve(code ?? (signal ? 1 : 0)));
            });
        } finally {
            process.stdin.unpipe(child.stdin);
            process.stdin.pause();
            process.removeListener('SIGINT', terminate); process.removeListener('SIGTERM', terminate);
        }
    } finally { await cleanup(); }
}

if (require.main === module) {
    Promise.resolve().then(() => { const args = parseArgs(process.argv.slice(2)); return runBundle(args.filename, args.sha256); })
        .then(code => { process.exitCode = code; })
        .catch(error => { console.error(`[target-bundle] ${error.message}`); process.exitCode = 1; });
}

module.exports = {validateManifest, loadManifest, stageMembers, nativeArguments, parseArgs, startBundle};
