'use strict';

const fs = require('node:fs/promises');
const crypto = require('node:crypto');
const path = require('node:path');
const {acquireRunLock, writeJsonAtomic, writeTextAtomic} = require('./counterfactual_rollout');
const {hashFile} = require('./tactical_teacher_smoke');
const {inspectReplay, verifyReplays} = require('./tactical_demonstrations');
const {isValid} = require('./tactical_teacher_benchmark');
const ROOT = path.resolve(__dirname, '..');

function validationHash(battleId, seed) {
    let hash = 14695981039346656037n;
    const bytes = Buffer.alloc(4);
    bytes.writeUInt32LE(seed);
    for (const byte of Buffer.concat([bytes, Buffer.from(battleId, 'utf8')]))
        hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 1099511628211n);
    return hash;
}

function validateConfig(config) {
    if (!Array.isArray(config.collections) || !config.collections.length ||
            config.collections.some(source => typeof source !== 'string' || !source) ||
            !Number.isSafeInteger(config.expected_battles) || config.expected_battles < 1 ||
            !Number.isSafeInteger(config.expected_decision_turns) || config.expected_decision_turns < 1 ||
            !Number.isSafeInteger(config.battles_per_shard) || config.battles_per_shard < 1 || config.battles_per_shard > 50 ||
            !Number.isSafeInteger(config.max_shard_bytes) || config.max_shard_bytes < 1 || config.max_shard_bytes > 8388608 ||
            !Number.isInteger(config.validation_seed) || config.validation_seed < 0 || config.validation_seed > 0xffffffff)
        throw new Error('dataset requires collection summaries, expected totals, a uint32 split seed, and shards at most 50 battles/8 MiB');
}

async function prepareDataset(config, name, root = ROOT) {
    validateConfig(config);
    if (!/^[A-Za-z0-9_-]+$/.test(name)) throw new Error('a simple --run-name is required');
    const collections = [];
    let teacherIdentity;
    for (const relative of config.collections) {
        const filename = path.resolve(root, relative);
        const summary = JSON.parse(await fs.readFile(filename, 'utf8'));
        if (summary.status !== 'completed' || !summary.identity?.config?.collect_demonstrations ||
                summary.identity.config.tactical_teacher === false || !Array.isArray(summary.games))
            throw new Error(`source is not a completed teacher collection: ${relative}`);
        const fingerprint = JSON.stringify([summary.identity.checkpoint_sha256, summary.identity.executable_sha256]);
        if (teacherIdentity && fingerprint !== teacherIdentity) throw new Error('collections use different teacher checkpoints or executables');
        teacherIdentity = fingerprint;
        await verifyReplays(path.dirname(filename), summary.games, isValid);
        const games = summary.games.filter(isValid);
        if (games.length !== summary.demonstration_count || games.length !== summary.valid_games)
            throw new Error(`source demonstration counts disagree: ${relative}`);
        collections.push({filename, sha256: await hashFile(filename), games, identity: summary.identity});
    }
    const identity = {config, sources: collections.map(source => ({summary: source.filename, sha256: source.sha256})),
        teacher_checkpoint_sha256: collections[0].identity.checkpoint_sha256,
        executable_sha256: collections[0].identity.executable_sha256};
    const directory = path.join(root, 'matches/runs', name);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const manifestPath = path.join(directory, `${name}_manifest.json`);
        let prior;
        try { prior = JSON.parse(await fs.readFile(manifestPath, 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (prior && JSON.stringify(prior.identity) !== JSON.stringify(identity))
            throw new Error('dataset inputs changed; choose a new run name');
        if (prior?.status === 'completed') {
            for (const shard of [...prior.shards, ...prior.holdout_shards]) {
                if (path.basename(shard.file) !== shard.file || await hashFile(path.join(directory, shard.file)) !== shard.sha256)
                    throw new Error(`prepared shard changed or is missing: ${shard.file}`);
            }
            return prior;
        }
        if (!prior && (await fs.readdir(directory)).some(file => file !== '.counterfactual_rollout.lock'))
            throw new Error('output directory already contains unrelated files');
        const manifest = {dataset_kind: 'supervised_tactical_demonstrations', run_name: name, status: 'in_progress',
            identity, validation_seed: config.validation_seed, ppo_eligible: false,
            counts: {battles: 0, decision_turns: 0, train_battles: 0, holdout_battles: 0, holdout_decision_turns: 0},
            battles: [], shards: [], holdout_shards: []};
        await writeJsonAtomic(manifestPath, manifest);
        const buffers = {shard: [], holdout: []};
        const sizes = {shard: 0, holdout: 0};
        async function flush(kind) {
            const buffer = buffers[kind];
            if (!buffer.length) return;
            const records = kind === 'shard' ? manifest.shards : manifest.holdout_shards;
            const file = `${kind}_${String(records.length).padStart(3, '0')}_raw.jsonl`;
            await writeTextAtomic(path.join(directory, file), buffer.map(record => record.content).join(''));
            records.push({file, sha256: await hashFile(path.join(directory, file)), bytes: sizes[kind],
                battles: buffer.length, battle_ids: buffer.map(record => record.battle_id),
                decision_turns: buffer.reduce((sum, record) => sum + record.decision_turns, 0)});
            buffers[kind] = [];
            sizes[kind] = 0;
        }
        async function append(kind, record) {
            const bytes = Buffer.byteLength(record.content);
            if (bytes > config.max_shard_bytes) throw new Error(`battle exceeds shard byte limit: ${record.battle_id}`);
            if (buffers[kind].length >= config.battles_per_shard || sizes[kind] + bytes > config.max_shard_bytes) await flush(kind);
            buffers[kind].push(record);
            sizes[kind] += bytes;
        }
        const seen = new Set();
        for (const source of collections) {
            for (const game of source.games) {
                const replay = game.demonstration;
                const filename = path.join(path.dirname(source.filename), replay.file);
                if ((await fs.stat(filename)).size > config.max_shard_bytes)
                    throw new Error(`battle exceeds shard byte limit: ${replay.battle_id}`);
                const content = await fs.readFile(filename, 'utf8');
                if (crypto.createHash('sha256').update(content).digest('hex') !== replay.sha256)
                    throw new Error(`source changed during preparation: ${replay.file}`);
                const metadata = inspectReplay(content.trimEnd().split('\n'));
                if (metadata.battle_id !== replay.battle_id || metadata.decision_turns !== replay.decision_turns ||
                        metadata.result !== replay.result || seen.has(metadata.battle_id))
                    throw new Error(`duplicate or inconsistent battle: ${metadata.battle_id}`);
                seen.add(metadata.battle_id);
                const holdout = validationHash(metadata.battle_id, config.validation_seed) % 10n === 0n;
                const record = {...metadata, content: content.endsWith('\n') ? content : `${content}\n`};
                await append('shard', record);
                if (holdout) await append('holdout', record);
                ++manifest.counts.battles;
                manifest.counts.decision_turns += metadata.decision_turns;
                ++manifest.counts[holdout ? 'holdout_battles' : 'train_battles'];
                if (holdout) manifest.counts.holdout_decision_turns += metadata.decision_turns;
                manifest.battles.push({...metadata, source: filename, sha256: replay.sha256, split: holdout ? 'holdout' : 'train'});
            }
        }
        await flush('shard');
        await flush('holdout');
        if (manifest.counts.battles !== config.expected_battles || manifest.counts.decision_turns !== config.expected_decision_turns ||
                !manifest.counts.train_battles || !manifest.counts.holdout_battles)
            throw new Error('prepared totals or split do not match the expected dataset');
        manifest.status = 'completed';
        await writeJsonAtomic(manifestPath, manifest);
        return manifest;
    } finally { await lock.release(); }
}

async function main(argv = process.argv.slice(2)) {
    const args = {};
    for (let i = 0; i < argv.length; i += 2) {
        if (!['--run-name', '--config'].includes(argv[i]) || !argv[i + 1])
            throw new Error('usage: node tools/prepare_tactical_dataset.js --run-name NAME [--config PATH]');
        args[argv[i]] = argv[i + 1];
    }
    const config = JSON.parse(await fs.readFile(path.resolve(ROOT,
        args['--config'] || 'config/experimental/tactical_dataset_1000.json'), 'utf8'));
    const manifest = await prepareDataset(config, args['--run-name'] || '');
    console.log(`[tactical-dataset] ${JSON.stringify({status: manifest.status, ...manifest.counts,
        shards: manifest.shards.length, holdout_shards: manifest.holdout_shards.length})}`);
}

if (require.main === module) main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; });
module.exports = {prepareDataset, validateConfig, validationHash};
