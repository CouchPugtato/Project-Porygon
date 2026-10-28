'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFile} = require('node:child_process');
const {promisify, isDeepStrictEqual} = require('node:util');
const {loadManifest, stageMembers} = require('./target_bundle');
const {validateArtifact, validateSnapshot, replayArtifact} = require('./target_records');
const {verifyGame} = require('./target_record_collection');
const {hashFile} = require('./tactical_teacher_smoke');
const {acquireRunLock, createResourceGate, writeJsonAtomic} = require('./counterfactual_rollout');
const ROOT = path.resolve(__dirname, '..');
const EXECUTABLE = path.join(ROOT, 'build-local/model-recovery/showdown_client.exe');
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const FORMAT = 'porygon-action-record-learner-v1';

function validateConfig(c) {
    const keys = ['run_name', 'manifest', 'manifest_sha256', 'sources', 'record_limit', 'minibatch', 'max_updates', 'learning_rate', 'max_abs_log_ratio'];
    if (c?.objective === 'ppo_frozen_value_v1') keys.push('objective', 'ppo_clip_epsilon');
    if (!c || Object.keys(c).length !== keys.length || keys.some(k => !Object.hasOwn(c, k)) ||
        typeof c.run_name !== 'string' || !/^[A-Za-z0-9_-]+$/.test(c.run_name) || typeof c.manifest !== 'string' ||
        !/^[a-f0-9]{64}$/.test(c.manifest_sha256) || !Array.isArray(c.sources) || !c.sources.length || c.sources.length > 6 ||
        c.sources.some(s => !s || Object.keys(s).length !== 2 || typeof s.path !== 'string' || !/^[a-f0-9]{64}$/.test(s.sha256)) ||
        !Number.isInteger(c.record_limit) || c.record_limit < 1 || c.record_limit > 128 ||
        !Number.isInteger(c.minibatch) || c.minibatch < 1 || c.minibatch > 8 ||
        !Number.isInteger(c.max_updates) || c.max_updates < 1 || c.max_updates > 8 ||
        !Number.isFinite(c.learning_rate) || c.learning_rate <= 0 || c.learning_rate > .0001 ||
        !Number.isFinite(c.max_abs_log_ratio) || c.max_abs_log_ratio <= 0 || c.max_abs_log_ratio > .05)
        throw new Error('invalid bounded action-record diagnostic config');
    if (c.objective === 'ppo_frozen_value_v1' && (!Number.isFinite(c.ppo_clip_epsilon) || c.ppo_clip_epsilon <= 0 || c.ppo_clip_epsilon > .2))
        throw new Error('invalid bounded PPO clip epsilon');
}

async function prepareRecords(config, bundle, replay = replayArtifact) {
    validateConfig(config);
    const rows = [], keys = new Set(), battles = new Set();
    let hiddenDim;
    for (const source of config.sources) {
        const filename = path.resolve(ROOT, source.path);
        if ((await fs.stat(filename)).size > 32 * 1024 * 1024) throw new Error('source size or SHA256 mismatch');
        const bytes = await fs.readFile(filename);
        if (crypto.createHash('sha256').update(bytes).digest('hex') !== source.sha256)
            throw new Error('source size or SHA256 mismatch');
        const saved = JSON.parse(bytes);
        const artifact = saved.artifact || saved;
        validateArtifact(artifact, bundle);
        if (battles.has(artifact.battle_id)) throw new Error('duplicate battle in diagnostic dataset');
        battles.add(artifact.battle_id);
        if (saved.artifact) {
            const match = /^tactical-random-(\d+)-(\d+)$/.exec(artifact.battle_id);
            if (!match) throw new Error('unsupported live battle identity');
            await verifyGame(saved, bundle, {seed: Number(match[1])}, Number(match[2]), replay);
        } else await replay(artifact, bundle);
        const terminal = artifact.inputs.find(x => x.type === 'terminal');
        if (![-1, 0, 1].includes(terminal.reward)) throw new Error('terminal-return diagnostic requires -1/0/+1 rewards');
        const accepted = new Map(artifact.inputs.filter(x => x.type === 'decision_accepted').map(x => [x.request_id, x]));
        for (const record of artifact.records) {
            validateSnapshot(record, accepted.get(record.request_id) || {});
            const key = `${artifact.battle_id}:${record.request_id}`;
            if (keys.has(key)) throw new Error('duplicate accepted decision');
            keys.add(key);
            hiddenDim ??= record.hidden_dim;
            if (record.hidden_dim !== hiddenDim) throw new Error('mixed hidden dimensions');
            rows.push({key, record, terminal_return: terminal.reward});
        }
    }
    if (!rows.length) throw new Error('no supported accepted records');
    return {rows: rows.slice(0, config.record_limit), hiddenDim, available: rows.length};
}

function packRecords(data, config, progress, invocationUpdates) {
    const {rows, hiddenDim} = data;
    const ppo = config.objective === 'ppo_frozen_value_v1', headerBytes = ppo ? 121 : 113;
    const stride = hiddenDim * 4 + 28 + 14 + 4 + 16;
    const bytes = Buffer.alloc(headerBytes + stride * rows.length);
    bytes.write(ppo ? 'PGACTREC2' : 'PGACTREC1');
    [0x01020304, hiddenDim, rows.length, progress.cursor, progress.updates, config.max_updates,
        invocationUpdates, config.minibatch].forEach((n, i) => bytes.writeUInt32LE(n, 9 + 4 * i));
    bytes.writeFloatLE(config.learning_rate, 41); bytes.writeFloatLE(config.max_abs_log_ratio, 45);
    if (ppo) { bytes.writeUInt32LE(1, 49); bytes.writeFloatLE(config.ppo_clip_epsilon, 53); }
    bytes.write(config.manifest_sha256, ppo ? 57 : 49);
    let offset = headerBytes;
    for (const {record: r, terminal_return} of rows) {
        const bits = Buffer.from(r.hidden_f32_hex, 'hex');
        for (let h = 0; h < hiddenDim; ++h) { bytes.writeUInt32LE(bits.readUInt32BE(4 * h), offset); offset += 4; }
        for (const value of r.legal_mask) bytes[offset++] = value;
        for (const slot of r.choice) for (const value of slot) bytes[offset++] = value;
        bytes.writeFloatLE(terminal_return, offset); offset += 4;
        bytes.writeDoubleLE(r.behavior.full_turn_log_probability, offset); offset += 8;
        const targetLog = r.behavior.slots.reduce((sum, slot) => sum + (slot ? Math.log(slot.selected_probability) : 0), 0);
        bytes.writeDoubleLE(targetLog, offset); offset += 8;
    }
    return bytes;
}

async function runNative({bundle, previous, stage, executable, executableSha256}) {
    await createResourceGate({maxSystemMemoryPercent: 65, maxSystemCpuPercent: 65,
        resourceCheckSeconds: 2, launchStaggerMilliseconds: 750}).waitForLaunch();
    const env = {...process.env, PORYGON_OMP_THREADS: '1', OMP_NUM_THREADS: '1'};
    delete env.PORYGON_DEMO_GRU; delete env.PORYGON_REPLAY_PATH;
    const staged = await stageMembers(bundle);
    try {
        if (previous) for (const [member, source, name] of [['resume_actor', previous.actor, 'actor.chk'], ['resume_optimizer', previous.optimizer, 'actor.opt']]) {
            staged.files[member] = path.join(path.dirname(staged.files.actor), member);
            await fs.copyFile(source, staged.files[member]);
            if (await hashFile(staged.files[member]) !== previous.state.files[name]) throw new Error('resume member changed while staging');
        }
        if (await hashFile(executable) !== executableSha256) throw new Error('learner executable changed before launch');
        const {stdout} = await promisify(execFile)(executable, ['--diagnose-action-records', path.join(stage, 'input.bin'),
            staged.files.reference, staged.files.actor, staged.files.resume_actor || '-', staged.files.resume_optimizer || '-', path.join(stage, 'actor')],
        {cwd: ROOT, env, windowsHide: true, timeout: 30000, maxBuffer: 1024 * 1024});
        return JSON.parse(stdout.trim());
    } finally { await staged.cleanup(); }
}

async function readGeneration(directory, pointer, identity) {
    if (pointer?.format !== FORMAT || !/^generation_\d+_[a-f0-9]{64}$/.test(pointer.generation)) throw new Error('invalid learner commit pointer');
    const folder = path.join(directory, pointer.generation), filename = path.join(folder, 'state.json');
    if (await hashFile(filename) !== pointer.sha256) throw new Error('learner state manifest hash mismatch');
    const state = JSON.parse(await fs.readFile(filename, 'utf8'));
    if (state.format !== FORMAT || !isDeepStrictEqual(state.identity, identity) || state.training_eligible !== false)
        throw new Error('learner generation identity mismatch');
    for (const [name, expected] of Object.entries(state.files)) {
        if (!['actor.chk', 'actor.opt', 'input.bin'].includes(name) || await hashFile(path.join(folder, name)) !== expected)
            throw new Error('learner generation file mismatch');
    }
    if (Object.keys(state.files).length !== 3) throw new Error('incomplete learner generation');
    return {state, actor: path.join(folder, 'actor.chk'), optimizer: path.join(folder, 'actor.opt')};
}

async function runDiagnostic(config, bundle, data, directory, identity, invocationUpdates = 8, execute = runNative) {
    validateConfig(config);
    if (!isDeepStrictEqual(identity.config, config) || identity.dataset_sha256 !== hash(data) || bundle.sha256 !== config.manifest_sha256)
        throw new Error('diagnostic dataset/config identity mismatch');
    if (!Number.isInteger(invocationUpdates) || invocationUpdates < 1 || invocationUpdates > 8) throw new Error('invalid invocation update limit');
    const lock = await acquireRunLock(directory);
    try {
        let priorIdentity, pointer;
        try { priorIdentity = JSON.parse(await fs.readFile(path.join(directory, 'identity.json'), 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (priorIdentity) {
            if (!isDeepStrictEqual(priorIdentity, identity)) throw new Error('learner inputs/settings changed; use a new run name');
        } else {
            const files = (await fs.readdir(directory)).filter(n => !['identity.json.tmp', '.counterfactual_rollout.lock'].includes(n));
            if (files.length) throw new Error('unidentified nonempty learner directory');
            await writeJsonAtomic(path.join(directory, 'identity.json'), identity);
        }
        try { pointer = JSON.parse(await fs.readFile(path.join(directory, 'current.json'), 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        const previous = pointer ? await readGeneration(directory, pointer, identity) : null;
        const progress = previous?.state.progress || {cursor: 0, updates: 0, guard_stopped: false};
        if (progress.guard_stopped || progress.cursor === data.rows.length || progress.updates === config.max_updates) return previous.state;
        const stage = await fs.mkdtemp(path.join(directory, '.pending-'));
        await fs.writeFile(path.join(stage, 'input.bin'), packRecords(data, config, progress, invocationUpdates));
        const result = await execute({bundle, previous, stage, executable: EXECUTABLE, executableSha256: identity.executable_sha256});
        if (config.objective === 'ppo_frozen_value_v1' && (result.objective !== config.objective ||
            ![result.ratio_min, result.ratio_max, result.mean_advantage, result.mean_value, result.surrogate_mean, result.clipped_fraction].every(Number.isFinite) ||
            result.ratio_min <= 0 || result.ratio_max < result.ratio_min || result.clipped_fraction < 0 || result.clipped_fraction > 1))
            throw new Error('native learner returned invalid PPO metrics/objective');
        if (result.diagnostic !== 'action_record_learner' || result.training_eligible !== false || result.ppo_ready !== false ||
            !Number.isInteger(result.updates) || result.updates < progress.updates || result.updates > Math.min(config.max_updates, progress.updates + invocationUpdates) ||
            result.cursor !== Math.min(result.updates * config.minibatch, data.rows.length) ||
            typeof result.guard_stopped !== 'boolean' || !Number.isFinite(result.max_abs_log_ratio) || result.max_abs_log_ratio < 0 ||
            result.max_abs_log_ratio > config.max_abs_log_ratio + 1e-8 || (!result.guard_stopped && result.updates === progress.updates))
            throw new Error('native learner returned invalid progress/guard metrics');
        for (const name of ['actor.chk', 'actor.opt', 'input.bin']) {
            const handle = await fs.open(path.join(stage, name), 'r+');
            try { await handle.sync(); } finally { await handle.close(); }
        }
        const files = Object.fromEntries(await Promise.all(['actor.chk', 'actor.opt', 'input.bin'].map(async n => [n, await hashFile(path.join(stage, n))])));
        const state = {format: FORMAT, training_eligible: false, identity, progress: result, files};
        await writeJsonAtomic(path.join(stage, 'state.json'), state);
        const generation = `generation_${result.updates}_${hash(state)}`;
        const target = path.join(directory, generation);
        try { await fs.rename(stage, target); }
        catch (error) { if (!['EEXIST', 'ENOTEMPTY', 'EPERM'].includes(error.code)) throw error;
            await readGeneration(directory, {format: FORMAT, generation, sha256: await hashFile(path.join(stage, 'state.json'))}, identity); }
        await writeJsonAtomic(path.join(directory, 'current.json'), {format: FORMAT, generation, sha256: await hashFile(path.join(target, 'state.json'))});
        return state;
    } finally { await lock.release(); }
}

async function main(argv) {
    if (![2, 4].includes(argv.length) || argv[0] !== '--config' || (argv.length === 4 && argv[2] !== '--steps'))
        throw new Error('usage: node tools/action_record_learner.js --config <approved-diagnostic.json> [--steps 1..8]');
    const config = JSON.parse(await fs.readFile(path.resolve(argv[1]), 'utf8')); validateConfig(config);
    const bundle = await loadManifest(path.resolve(ROOT, config.manifest), config.manifest_sha256);
    const data = await prepareRecords(config, bundle);
    const sources = ['tools/action_record_learner.js', 'tools/target_records.js', 'tools/target_bundle.js',
        'tools/target_record_collection.js', 'tools/counterfactual_rollout.js'];
    const identity = {config, dataset_sha256: hash(data), executable_sha256: await hashFile(EXECUTABLE),
        source_sha256: Object.fromEntries(await Promise.all(sources.map(async name => [name, await hashFile(path.join(ROOT, name))]))),
        objective: config.objective || 'terminal_return_score_function_v1', order: 'source_then_record', rng: 'unused'};
    const directory = path.join(ROOT, 'models/diagnostics', config.run_name);
    await fs.mkdir(directory, {recursive: true});
    const state = await runDiagnostic(config, bundle, data, directory, identity, argv[3] === undefined ? 8 : Number(argv[3]));
    console.log(JSON.stringify({diagnostic: 'action_record_learner', training_eligible: false, strength_claim: false, ...state.progress}));
}
if (require.main === module) main(process.argv.slice(2)).catch(error => { console.error(`[action-record-learner] ${error.stack || error.message}`); process.exitCode = 1; });
module.exports = {validateConfig, prepareRecords, packRecords, readGeneration, runDiagnostic};
