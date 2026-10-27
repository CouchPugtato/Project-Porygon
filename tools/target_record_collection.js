'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {isDeepStrictEqual} = require('node:util');
const {AgentProcess, acquireRunLock, createResourceGate, writeJsonAtomic} = require('./counterfactual_rollout');
const {hashFile, playGame} = require('./tactical_teacher_smoke');
const {loadManifest, startBundle} = require('./target_bundle');
const {buildArtifact, replayArtifact} = require('./target_records');
const ROOT = path.resolve(__dirname, '..');
const FORMAT = 'porygon-live-bundle-battle-v1';
const RESOURCE_LIMITS = {maxSystemMemoryPercent: 65, maxSystemCpuPercent: 65,
    resourceCheckSeconds: 2, launchStaggerMilliseconds: 750};
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');

class BundleRecordingAgent extends AgentProcess {
    constructor(bundle) {
        const inputs = [];
        let bytes = 0;
        super(bundle.files.actor, bundle.manifest.inference_seed, null, 1, false, line => {
            bytes += Buffer.byteLength(line);
            if (bytes > 8 * 1024 * 1024) throw new Error('live protocol trace exceeds 8 MiB');
            inputs.push(JSON.parse(line));
        }, 'sample', false, false, {sidecar: bundle.files.sidecar, contextual: true,
            decoder: bundle.manifest.target_decoder, slotLocal: bundle.manifest.eligibility === 'slot-local-v1'});
        this.bundle = bundle;
        this.launchBundle = startBundle;
        this.inputs = inputs;
        this.outputs = [];
        this.outputBytes = 0;
    }

    async launch() {
        const {child, cleanup} = await this.launchBundle(this.bundle, true);
        const closed = new Promise(resolve => child.once('close', resolve));
        // The shared bridge reports native errors through nextMessage. Spawn
        // failures need the same path instead of an unhandled child error.
        child.once('error', error => super.pushMessage({type: 'error', message: error.message}));
        child.stdin.on('error', error => {
            if (error.code !== 'EPIPE') super.pushMessage({type: 'error', message: error.message});
        });
        if (this.closing) {
            child.kill();
            await closed;
            await cleanup();
            throw new Error('battle closed while waiting to launch its bundle');
        }
        this.cleanupBundle = cleanup;
        this.closed = closed;
        this.interrupt = () => {
            super.pushMessage({type: 'error', message: 'live collection interrupted'});
            this.terminate();
        };
        process.once('SIGINT', this.interrupt); process.once('SIGTERM', this.interrupt);
        return child;
    }

    pushMessage(message) {
        this.outputBytes += Buffer.byteLength(JSON.stringify(message));
        if (this.outputBytes > 16 * 1024 * 1024) {
            super.pushMessage({type: 'error', message: 'live native output exceeds 16 MiB'});
            this.terminate();
            return;
        }
        this.outputs.push(message);
        if (!['bundle_decision', 'bundle_decision_skipped'].includes(message.type)) super.pushMessage(message);
    }

    async close() {
        this.closing = true;
        if (this.process?.stdin.writable) this.process.stdin.end();
        try { if (this.closed) await this.closed; }
        finally {
            this.process = null;
            if (this.interrupt) {
                process.removeListener('SIGINT', this.interrupt); process.removeListener('SIGTERM', this.interrupt);
            }
            if (this.cleanupBundle) { await this.cleanupBundle(); this.cleanupBundle = null; }
        }
    }
}

function validateConfig(config) {
    const keys = ['run_name', 'manifest', 'manifest_sha256', 'games', 'seed', 'format', 'battle_timeout_seconds'];
    if (!config || Object.keys(config).length !== keys.length || keys.some(key => !Object.hasOwn(config, key)) ||
            typeof config.run_name !== 'string' || !/^[A-Za-z0-9_-]+$/.test(config.run_name) ||
            typeof config.manifest !== 'string' || !/^[a-f0-9]{64}$/.test(config.manifest_sha256) ||
            !Number.isInteger(config.games) || config.games < 1 || config.games > 6 ||
            !Number.isInteger(config.seed) || config.seed < 1 || config.seed > 2147483000 ||
            config.format !== 'gen9randomdoublesbattle' || !Number.isInteger(config.battle_timeout_seconds) ||
            config.battle_timeout_seconds < 30 || config.battle_timeout_seconds > 180)
        throw new Error('live record smoke requires 1-6 serial random-doubles games and a pinned bundle');
}

async function playRecordedGame(bundle, config, index) {
    const sim = require('../external/pokemon-showdown/dist/sim');
    let agent;
    const game = await playGame(sim, {seed: config.seed, format: config.format,
        battle_timeout_seconds: config.battle_timeout_seconds, tactical_teacher: false}, index,
    {opponent: 'random', agentFactory: (_config, _seeds, side) => {
        agent = new BundleRecordingAgent(bundle);
        return side === 0 ? [agent, null] : [null, agent];
    }});
    if (game.status !== 'completed') {
        const error = new Error([game.error, game.cleanup_error].filter(Boolean).join('\n'));
        error.capture = {game, inputs: agent?.inputs || [], outputs: agent?.outputs || []};
        throw error;
    }
    // playGame's ordinary inference seed is not used by a pinned bundle.
    // Record the actual native seed, rather than mislabelling that schedule.
    game.bundle_inference_seed = bundle.manifest.inference_seed;
    game.seeds.inference[index % 2] = bundle.manifest.inference_seed;
    const artifact = buildArtifact(bundle, agent.inputs, agent.outputs);
    const body = {format: FORMAT, training_eligible: false, manifest_sha256: bundle.sha256, game, artifact};
    return {...body, content_sha256: hash(body)};
}

async function verifyGame(saved, bundle, config, index, replay = replayArtifact) {
    const {content_sha256, ...body} = saved;
    const game = saved.game;
    const terminal = saved.artifact?.inputs.find(message => message.type === 'terminal');
    if (content_sha256 !== hash(body) || saved.format !== FORMAT || saved.training_eligible !== false || saved.manifest_sha256 !== bundle.sha256 ||
            game?.status !== 'completed' || game.index !== index || game.teacher_side !== (index % 2 ? 'b' : 'a') ||
            game.bundle_inference_seed !== bundle.manifest.inference_seed || !Array.isArray(game.choice_rejections) ||
            saved.artifact?.battle_id !== `tactical-random-${config.seed}-${index}` ||
            !['win', 'loss', 'draw'].includes(game.result?.result) || terminal?.result !== game.result.result ||
            terminal?.reward !== game.result.reward)
        throw new Error('saved live battle identity/outcome mismatch');
    return replay(saved.artifact, bundle);
}

// The battle file is the only commit point. Summary loss after a crash is
// recoverable by scanning these files; unfinished simulator state is not.
async function collectGames(config, bundle, identity, directory, {play = playRecordedGame, replay = replayArtifact,
    gate = createResourceGate(RESOURCE_LIMITS)} = {}) {
    validateConfig(config);
    if (!isDeepStrictEqual(identity.config, config) || identity.manifest_sha256 !== bundle.sha256)
        throw new Error('live collection identity does not match config/bundle');
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const identityPath = path.join(directory, 'identity.json');
        let prior;
        try { prior = JSON.parse(await fs.readFile(identityPath, 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (prior) {
            if (!isDeepStrictEqual(prior, identity)) throw new Error('live collection inputs changed; choose a new run name');
        } else {
            const files = (await fs.readdir(directory)).filter(name => !['identity.json.tmp', '.counterfactual_rollout.lock'].includes(name));
            if (files.length) throw new Error('unidentified nonempty live record directory');
            await writeJsonAtomic(identityPath, identity);
        }
        const battles = [];
        for (let index = 0; index < config.games; ++index) {
            const filename = path.join(directory, `battle_${String(index).padStart(4, '0')}.json`);
            let saved;
            try { saved = JSON.parse(await fs.readFile(filename, 'utf8')); }
            catch (error) { if (error.code !== 'ENOENT') throw error; }
            const reused = !!saved;
            try {
                if (!saved) { await gate.waitForLaunch(); saved = await play(bundle, config, index); }
                await verifyGame(saved, bundle, config, index, replay);
            } catch (error) {
                await writeJsonAtomic(path.join(directory, `failure_${String(index).padStart(4, '0')}.json`),
                    {index, manifest_sha256: bundle.sha256, training_eligible: false, error: error.message,
                        capture: error.capture || null});
                throw error;
            }
            if (!reused) await writeJsonAtomic(filename, saved);
            battles.push({index, file: path.basename(filename), sha256: await hashFile(filename),
                side: saved.game.teacher_side, result: saved.game.result.result,
                accepted: saved.artifact.records.length, skipped: saved.artifact.skipped.length,
                choice_rejections: saved.game.choice_rejections, replay_likelihood_ratio: 1});
            const summary = {diagnostic: 'live_accepted_bundle_records', run_name: config.run_name,
                status: battles.length === config.games ? 'completed' : 'running', training_eligible: false,
                strength_claim: false, identity, completed: battles.length, games: battles};
            await writeJsonAtomic(path.join(directory, 'summary.json'), summary);
            console.log(`[bundle-records] games=${battles.length}/${config.games} accepted=${battles.reduce((sum, b) => sum + b.accepted, 0)} ` +
                `skipped=${battles.reduce((sum, b) => sum + b.skipped, 0)} replay=exact ${reused ? 'resumed' : 'published'}`);
        }
        return battles;
    } finally { await lock.release(); }
}

async function main(argv) {
    if (argv.length !== 0 && (argv.length !== 2 || argv[0] !== '--config'))
        throw new Error('usage: node tools/target_record_collection.js [--config <json>]');
    const config = JSON.parse(await fs.readFile(path.resolve(argv[1] || path.join(ROOT, 'config/experimental/target_record_live_smoke.json')), 'utf8'));
    validateConfig(config);
    const bundle = await loadManifest(path.resolve(ROOT, config.manifest), config.manifest_sha256);
    const sources = ['tools/target_record_collection.js', 'tools/target_records.js', 'tools/target_bundle.js',
        'tools/counterfactual_rollout.js', 'tools/tactical_teacher_smoke.js',
        'external/pokemon-showdown/dist/sim/battle-stream.js', 'external/pokemon-showdown/dist/sim/tools/random-player-ai.js'];
    const identity = {config, manifest_sha256: bundle.sha256, source_sha256: Object.fromEntries(
        await Promise.all(sources.map(async file => [file, await hashFile(path.join(ROOT, file))])))};
    await collectGames(config, bundle, identity, path.join(ROOT, 'matches/runs', config.run_name));
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[bundle-records] ${error.stack || error.message}`); process.exitCode = 1;
});
module.exports = {BundleRecordingAgent, validateConfig, playRecordedGame, verifyGame, collectGames};
