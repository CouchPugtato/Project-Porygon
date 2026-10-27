'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {isDeepStrictEqual} = require('node:util');
const {loadManifest, startBundle} = require('./target_bundle');
const {validateContext} = require('./target_context');
const {writeJsonAtomic, acquireRunLock} = require('./counterfactual_rollout');

const FORMAT = 'porygon-frozen-bundle-records-v1';
const INPUT_LIMIT = 8 * 1024 * 1024;
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
function requireRecord(condition, message) { if (!condition) throw new Error(message); }

// A complete protocol trace, not accepted observations alone, reproduces GRU
// state after rejected requests. Resume replays an interrupted battle from zero.
function validateTrace(inputs) {
    requireRecord(Array.isArray(inputs) && inputs.length >= 3 &&
        Buffer.byteLength(JSON.stringify(inputs)) <= INPUT_LIMIT, 'invalid or oversized protocol trace');
    const id = inputs[0].battle_id;
    requireRecord(typeof id === 'string' && /^[A-Za-z0-9_-]{1,127}$/.test(id) &&
        inputs[0].type === 'battle_start' && inputs.at(-1).type === 'battle_end', 'complete single-battle trace required');
    const requests = new Set(), accepted = new Map();
    let pending = null, terminal = false;
    for (const [index, message] of inputs.entries()) {
        requireRecord(message.battle_id === id, 'mixed battle IDs');
        switch (message.type) {
        case 'battle_start': requireRecord(index === 0, 'duplicate battle start'); break;
        case 'event': requireRecord(!terminal && typeof message.line === 'string', 'invalid public event'); break;
        case 'request':
            requireRecord(!terminal && pending === null && Number.isSafeInteger(message.request_id) &&
                message.request_id >= 0 && !requests.has(message.request_id) &&
                message.payload && typeof message.payload === 'object', 'duplicate or unresolved request');
            requests.add(message.request_id);
            if (!message.payload.wait) pending = message.request_id;
            break;
        case 'decision_accepted': case 'decision_rejected':
            requireRecord(!terminal && pending !== null && message.request_id === pending &&
                Number.isInteger(message.action) && Number.isInteger(message.action2) &&
                typeof message.command === 'string', 'duplicate or unmatched decision outcome');
            if (message.type === 'decision_accepted') accepted.set(pending, message);
            pending = null;
            break;
        case 'terminal':
            requireRecord(!terminal && pending === null && Number.isFinite(message.reward) &&
                Math.abs(message.reward) <= 1, 'invalid terminal outcome');
            terminal = true;
            break;
        case 'battle_end': requireRecord(index === inputs.length - 1 && terminal, 'missing terminal outcome'); break;
        default: throw new Error(`unsupported trace message: ${message.type}`);
        }
    }
    requireRecord(terminal && accepted.size > 0 && requests.size <= 1024, 'empty, incomplete or oversized battle');
    return {id, accepted};
}

function validateIdentity(identity, bundle) {
    const manifest = bundle.manifest;
    requireRecord(identity?.version === 2 && identity.kind === manifest.kind &&
        identity.parameter_fingerprint_algorithm === 'fnv1a64-native' &&
        /^[0-9a-f]{16}$/.test(identity.reference_parameters) && /^[0-9a-f]{16}$/.test(identity.actor_parameters) &&
        identity.manifest_sha256 === bundle.sha256 && identity.sidecar_parameters === manifest.sidecar_parameters &&
        identity.context_version === 1 && identity.action_decoder === 'sample' &&
        identity.target_decoder === manifest.target_decoder && identity.eligibility === manifest.eligibility,
    'bundle identity mismatch');
}

function validateSnapshot(record, accepted) {
    requireRecord(record.action === accepted.action && record.action2 === accepted.action2 &&
        record.command === accepted.command, 'accepted action mismatch');
    requireRecord(Number.isInteger(record.hidden_dim) && record.hidden_dim > 0 && record.hidden_dim <= 4096 &&
        typeof record.hidden_f32_hex === 'string' && /^[0-9a-f]+$/.test(record.hidden_f32_hex) &&
        record.hidden_f32_hex.length === 8 * record.hidden_dim, 'invalid hidden float32 bits');
    const hidden = Buffer.from(record.hidden_f32_hex, 'hex');
    for (let i = 0; i < record.hidden_dim; ++i)
        requireRecord(Number.isFinite(hidden.readFloatBE(i * 4)), 'non-finite hidden state');
    requireRecord(Array.isArray(record.legal_mask) && record.legal_mask.length === 28 &&
        record.legal_mask.every(x => x === 0 || x === 1), 'invalid legal action mask');
    requireRecord(Array.isArray(record.choice) && record.choice.length === 2 &&
        Array.isArray(record.public_context) && record.public_context.length === 2, 'missing selected choices/context');
    const behavior = record.behavior;
    requireRecord(behavior?.version === 2 && behavior.scope === 'full_turn_diagnostic' && behavior.ppo_ready === false &&
        behavior.likelihood_status === 'supported' && Number.isFinite(behavior.full_turn_log_probability) &&
        behavior.full_turn_log_probability <= 0 && Array.isArray(behavior.slots) && behavior.slots.length === 2,
    'invalid full-turn likelihood');
    for (const [slot, choice] of record.choice.entries()) {
        requireRecord(Array.isArray(choice) && choice.length === 7 && choice.every(Number.isInteger), 'invalid factorized choice');
        const [has, kind, move, bench, tera, target, mask] = choice;
        requireRecord((has === 0 || has === 1) && kind >= 0 && kind <= 2 && move >= 0 && move < 4 &&
            bench >= 0 && bench < 6 && (tera === 0 || tera === 1) && target >= 0 && target < 4 && mask >= 0 && mask < 16,
        'out-of-range factorized choice');
        const action = slot ? record.action2 : record.action;
        const expected = !has ? -1 : slot * 14 + (kind === 1 ? move + (tera ? 4 : 0) : 8 + bench);
        requireRecord((!has || kind !== 0) && action === expected &&
            (!has || record.legal_mask[action] === 1), 'illegal selected action');
        if (has && kind === 1) validateContext(record.public_context[slot], true);
        else requireRecord(record.public_context[slot] === null, 'unexpected selected-move context');
        const policy = behavior.slots[slot];
        if (!mask) { requireRecord(policy === null, 'unexpected target distribution'); continue; }
        requireRecord(has && kind === 1 && (mask & (1 << target)) && policy?.legal_mask === mask && policy.selected === target &&
            ['sidecar', 'checkpoint'].includes(policy.source) && Array.isArray(policy.probabilities) && policy.probabilities.length === 4,
        'illegal target choice');
        requireRecord(policy.probabilities.every((p, i) => Number.isFinite(p) && p >= 0 && p <= 1 &&
            ((mask & (1 << i)) || p === 0)) && Math.abs(policy.probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-6 &&
            policy.selected_probability > 0 && policy.selected_probability === policy.probabilities[target], 'invalid target probabilities');
    }
}

function buildArtifact(bundle, inputs, outputs) {
    const {id, accepted} = validateTrace(inputs);
    requireRecord(Array.isArray(outputs) && outputs[0]?.type === 'ready' && outputs[0].accepted_records === true &&
        outputs[0].capabilities?.training === false && !outputs.some(x => x.type === 'error'), 'native record mode failed');
    const identity = outputs[0].behavior_policy;
    validateIdentity(identity, bundle);
    const records = [], skipped = [], seen = new Set();
    const terminalFrames = outputs.filter(x => x.type === 'episode_complete');
    requireRecord(terminalFrames.length === 1, 'missing or duplicate native terminal');
    for (const frame of outputs.filter(x => ['bundle_decision', 'bundle_decision_skipped'].includes(x.type))) {
        requireRecord(frame.version === 1 && frame.battle_id === id && frame.training_eligible === false &&
            isDeepStrictEqual(frame.behavior_policy, identity) && accepted.has(frame.request_id) &&
            !seen.has(frame.request_id) && frame.decision_index === seen.size + 1, 'duplicate, rejected or mismatched accepted record');
        seen.add(frame.request_id);
        if (frame.type === 'bundle_decision') { validateSnapshot(frame, accepted.get(frame.request_id)); records.push(frame); }
        else {
            requireRecord(frame.reason === 'unsupported_likelihood_or_acceptance' && !('hidden_f32_hex' in frame) &&
                !('behavior' in frame), 'unsupported decision must not contain a usable snapshot');
            skipped.push(frame);
        }
    }
    const terminal = terminalFrames[0];
    requireRecord(seen.size === accepted.size && terminal.battle_id === id && terminal.training_eligible === false &&
        isDeepStrictEqual(terminal.behavior_policy, identity) && terminal.target_decisions === accepted.size &&
        terminal.likelihood_replay?.accepted_supported === records.length &&
        terminal.likelihood_replay?.accepted_unsupported === skipped.length, 'accepted/terminal coverage mismatch');
    const body = {format: FORMAT, training_eligible: false, manifest_sha256: bundle.sha256,
        input_sha256: hash(inputs), battle_id: id, identity, inputs, records, skipped, terminal};
    return {...body, content_sha256: hash(body)};
}

async function executeTrace(bundle, inputs) {
    const {child, cleanup} = await startBundle(bundle, true);
    let stdout = '', stderr = '', size = 0, failure;
    const stop = reason => { failure ??= new Error(reason); child.kill(); };
    const interrupt = () => stop('record replay interrupted');
    const timer = setTimeout(() => stop('record replay exceeded 30 seconds'), 30000);
    process.once('SIGINT', interrupt); process.once('SIGTERM', interrupt);
    child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
    child.stdout.on('data', chunk => { size += Buffer.byteLength(chunk); if (size > 16 * 1024 * 1024) stop('native output exceeds limit'); else stdout += chunk; });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-8192); });
    child.stdin.on('error', error => { if (error.code !== 'EPIPE') stop(error.message); });
    try {
        const completed = new Promise((resolve, reject) => {
            child.once('error', reject);
            child.once('close', code => resolve(code));
        });
        child.stdin.end(inputs.map(x => JSON.stringify(x)).join('\n') + '\n');
        const code = await completed;
        if (failure) throw failure;
        requireRecord(code === 0, `native record replay failed (${code}): ${stderr}`);
        return stdout.trim().split(/\r?\n/).map(line => JSON.parse(line));
    } finally {
        clearTimeout(timer);
        process.removeListener('SIGINT', interrupt); process.removeListener('SIGTERM', interrupt);
        await cleanup();
    }
}

function validateArtifact(artifact, bundle) {
    const {content_sha256, ...body} = artifact;
    requireRecord(artifact.format === FORMAT && artifact.training_eligible === false &&
        artifact.manifest_sha256 === bundle.sha256 && artifact.input_sha256 === hash(artifact.inputs) &&
        content_sha256 === hash(body), 'saved record hash or bundle mismatch');
    validateTrace(artifact.inputs);
    validateIdentity(artifact.identity, bundle);
}

async function replayArtifact(artifact, bundle, execute = executeTrace) {
    validateArtifact(artifact, bundle);
    const fresh = buildArtifact(bundle, artifact.inputs, await execute(bundle, artifact.inputs));
    requireRecord(isDeepStrictEqual(fresh, artifact), 'saved state, actions or likelihoods differ on replay');
    return {accepted: artifact.records.length, skipped: artifact.skipped.length, likelihood_ratio: 1};
}

async function collectTrace(bundle, inputs, directory, execute = executeTrace) {
    const {id} = validateTrace(inputs);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const metadataPath = path.join(directory, 'identity.json');
        const metadata = {format: FORMAT, manifest_sha256: bundle.sha256};
        let prior;
        try { prior = JSON.parse(await fs.readFile(metadataPath, 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (prior) requireRecord(isDeepStrictEqual(prior, metadata), 'refusing to mix bundle identities');
        else {
            const existing = (await fs.readdir(directory)).filter(name => !['.counterfactual_rollout.lock', 'identity.json.tmp'].includes(name));
            requireRecord(existing.length === 0, 'unidentified nonempty record directory');
            await writeJsonAtomic(metadataPath, metadata);
        }
        const filename = path.join(directory, `${hash(id)}.json`);
        let saved;
        try { saved = JSON.parse(await fs.readFile(filename, 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (saved) {
            requireRecord(saved.input_sha256 === hash(inputs), 'battle ID already committed with a different trace');
            return {filename, reused: true, ...await replayArtifact(saved, bundle, execute)};
        }
        const artifact = buildArtifact(bundle, inputs, await execute(bundle, inputs));
        await writeJsonAtomic(filename, artifact);
        return {filename, reused: false, accepted: artifact.records.length, skipped: artifact.skipped.length};
    } finally { await lock.release(); }
}

async function main(argv) {
    requireRecord((argv.length === 8 && argv[4] === '--trace' && argv[6] === '--output' ||
        argv.length === 6 && argv[4] === '--verify') && argv[0] === '--manifest' && argv[2] === '--manifest-sha256',
    'usage: node tools/target_records.js --manifest <json> --manifest-sha256 <SHA256> --trace <complete.jsonl> --output <directory> | --verify <artifact.json>');
    const bundle = await loadManifest(argv[1], argv[3]);
    requireRecord((await fs.stat(argv[5])).size <= (argv[4] === '--trace' ? INPUT_LIMIT : 32 * 1024 * 1024), 'input file exceeds limit');
    const bytes = await fs.readFile(argv[5], 'utf8');
    const result = argv[4] === '--verify' ? await replayArtifact(JSON.parse(bytes), bundle) :
        await collectTrace(bundle, bytes.trim().split(/\r?\n/).map(line => JSON.parse(line)), path.resolve(argv[7]));
    console.log(JSON.stringify({diagnostic: 'accepted_bundle_records', training_eligible: false, ...result}));
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[target-records] ${error.message}`); process.exitCode = 1;
});
module.exports = {validateTrace, validateSnapshot, buildArtifact, validateArtifact, replayArtifact, collectTrace};
