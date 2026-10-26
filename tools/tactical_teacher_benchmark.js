'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {aggregateGames, hashFile, playGame} = require('./tactical_teacher_smoke');
const {acquireRunLock, createResourceGate, writeJsonAtomic} = require('./counterfactual_rollout');
const {publishReplay, verifyReplays} = require('./tactical_demonstrations');
const {summarizeDecisions} = require('./tactical_decision_audit');
const ROOT = path.resolve(__dirname, '..');
const OPPONENTS = ['parent', 'random'];
const SUPPORTED_OPPONENTS = [...OPPONENTS, 'pilot', 'sampled'];

function wilson(score, count) {
    if (!count) return null;
    const z2 = 1.959963984540054 ** 2;
    const p = score / count;
    const center = p + z2 / (2 * count);
    const radius = Math.sqrt(z2 * (p * (1 - p) / count + z2 / (4 * count * count)));
    return {low: (center - radius) / (1 + z2 / count), high: (center + radius) / (1 + z2 / count)};
}

function isValid(game) {
    return game.status === 'completed' && Array.isArray(game.choice_rejections) &&
        game.choice_rejections.length === 0 && ['win', 'loss', 'draw'].includes(game.result?.result);
}

function summarize(games, opponents = OPPONENTS) {
    const results = {};
    for (const opponent of new Set([...opponents, ...games.map(game => game.opponent)])) {
        const records = games.filter(game => game.opponent === opponent);
        const tally = subset => {
            const valid = subset.filter(isValid);
            const wins = valid.filter(game => game.result.result === 'win').length;
            const losses = valid.filter(game => game.result.result === 'loss').length;
            const draws = valid.length - wins - losses;
            return {attempts: subset.length, valid_games: valid.length, invalid_games: subset.length - valid.length,
                wins, losses, draws, score_rate: valid.length ? (wins + draws / 2) / valid.length : null,
                confidence_95: wilson(wins + draws / 2, valid.length)};
        };
        results[opponent] = {...tally(records), sides: Object.fromEntries(['a', 'b'].map(side =>
            [side, tally(records.filter(game => game.teacher_side === side))])),
        coverage: records.some(game => game.policy_mode === 'neural') ? null : aggregateGames(records.filter(isValid)),
        choice_rejections: records.reduce((sum, game) => sum + (game.choice_rejections?.length || 0), 0)};
    }
    return results;
}

function nextGame(config, games) {
    const results = summarize(games, config.opponents);
    for (const opponent of config.opponents || OPPONENTS) {
        const sides = results[opponent].sides;
        const pending = ['a', 'b'].filter(side => sides[side].valid_games < config.games_per_side);
        if (!pending.length) continue;
        if (pending.some(side => sides[side].attempts >= config.max_attempts_per_side))
            throw new Error(`${opponent}: exhausted attempts before reaching the valid-game target`);
        pending.sort((a, b) => sides[a].attempts - sides[b].attempts);
        const side = pending[0];
        return {opponent, index: sides[side].attempts * 2 + (side === 'b' ? 1 : 0), teacher_side: side};
    }
    return null;
}

function validateResume(saved, identity, config) {
    if (JSON.stringify(saved.identity) !== JSON.stringify(identity))
        throw new Error('benchmark inputs changed; choose a new run name');
    if (!Array.isArray(saved.games)) throw new Error('saved benchmark has no game records');
    const checked = [];
    for (const game of saved.games) {
        const expected = nextGame(config, checked);
        if (!expected || game.opponent !== expected.opponent || game.index !== expected.index ||
                game.teacher_side !== expected.teacher_side || !['completed', 'failed'].includes(game.status) ||
                !Number.isFinite(game.duration_seconds) || game.duration_seconds < 0)
            throw new Error('saved benchmark records do not match the deterministic schedule');
        if (game.status === 'completed') {
            if (config.decision_audit) summarizeDecisions([game]);
            if ((game.inference_policy || 'sample') !== (config.candidate_inference_policy || 'sample'))
                throw new Error('saved candidate inference policy differs from the configured decoder');
            if (!Array.isArray(game.choice_rejections) || !['win', 'loss', 'draw'].includes(game.result?.result))
                throw new Error('saved completed game has no terminal result or rejection record');
            if ((game.policy_mode === 'neural') !== (config.tactical_teacher === false))
                throw new Error('saved candidate policy mode differs from the configured mode');
            if (config.tactical_teacher !== false) aggregateGames([game]);
            else if (game.coverage !== null) throw new Error('neural evaluation must not contain tactical coverage');
        }
        checked.push(game);
    }
    return checked;
}

async function loadSaved(filename) {
    try { return JSON.parse(await fs.readFile(filename, 'utf8')); }
    catch (error) {
        if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error;
        try {
            const backup = JSON.parse(await fs.readFile(`${filename}.bak`, 'utf8'));
            console.warn('[tactical-benchmark] recovering the last atomic backup');
            return backup;
        } catch (backupError) {
            if (error.code === 'ENOENT' && backupError.code === 'ENOENT') return null;
            throw error;
        }
    }
}

function validateConfig(config) {
    if ((config.collect_demonstrations !== undefined && typeof config.collect_demonstrations !== 'boolean') ||
            (config.decision_audit !== undefined && typeof config.decision_audit !== 'boolean') ||
            (config.decision_audit && (config.tactical_teacher !== false || config.collect_demonstrations)) ||
            (config.candidate_inference_policy !== undefined && !['sample', 'greedy'].includes(config.candidate_inference_policy)) ||
            (config.candidate_inference_policy === 'greedy' && (config.tactical_teacher !== false || config.collect_demonstrations)) ||
            (config.opponents !== undefined && (!Array.isArray(config.opponents) || !config.opponents.length ||
                config.opponents.some(opponent => !SUPPORTED_OPPONENTS.includes(opponent)) ||
                new Set(config.opponents).size !== config.opponents.length)) ||
            (config.opponents?.includes('pilot') &&
                (typeof config.pilot_checkpoint !== 'string' || !config.pilot_checkpoint.trim())) ||
            (config.tactical_teacher !== undefined && typeof config.tactical_teacher !== 'boolean') ||
            (config.candidate_checkpoint !== undefined &&
                (typeof config.candidate_checkpoint !== 'string' || !config.candidate_checkpoint.trim())) ||
            (config.collect_demonstrations && config.tactical_teacher === false) ||
            config.format !== 'gen9randomdoublesbattle' || config.agent_omp_threads !== 1 ||
            !Number.isSafeInteger(config.games_per_side) || config.games_per_side < 1 || config.games_per_side > 1000 ||
            !Number.isSafeInteger(config.max_attempts_per_side) ||
            config.max_attempts_per_side < config.games_per_side || config.max_attempts_per_side > 5 * config.games_per_side ||
            !Number.isSafeInteger(config.seed) || config.seed < 0 ||
            !(config.battle_timeout_seconds > 0 && config.battle_timeout_seconds <= 300) ||
            !(config.max_system_memory_percent > 0 && config.max_system_memory_percent <= 65) ||
            !(config.max_system_cpu_percent > 0 && config.max_system_cpu_percent <= 65) ||
            !(config.resource_check_seconds > 0) || !(config.launch_stagger_ms >= 0))
        throw new Error('benchmark requires bounded valid-game targets, one thread per agent, and resource gates at most 65%');
}

async function main(argv = process.argv.slice(2)) {
    const args = {};
    for (let i = 0; i < argv.length; i += 2) {
        if (!['--run-name', '--config'].includes(argv[i]) || !argv[i + 1])
            throw new Error('usage: node tools/tactical_teacher_benchmark.js --run-name NAME [--config PATH]');
        args[argv[i]] = argv[i + 1];
    }
    const name = args['--run-name'];
    if (!name || !/^[A-Za-z0-9_-]+$/.test(name)) throw new Error('a simple --run-name is required');
    const config = JSON.parse(await fs.readFile(path.resolve(ROOT,
        args['--config'] || 'config/experimental/tactical_teacher_benchmark.json'), 'utf8'));
    validateConfig(config);
    const targetGames = 2 * config.games_per_side * (config.opponents || OPPONENTS).length;
    const sources = ['tools/tactical_teacher_benchmark.js', 'tools/tactical_teacher_smoke.js',
        'tools/counterfactual_rollout.js', 'tools/tactical_demonstrations.js', 'tools/tactical_decision_audit.js',
        'external/pokemon-showdown/dist/sim/battle.js',
        'external/pokemon-showdown/dist/sim/battle-stream.js',
        'external/pokemon-showdown/dist/sim/tools/random-player-ai.js'];
    const identity = {config, checkpoint_sha256: await hashFile(path.resolve(ROOT, config.checkpoint)),
        candidate_checkpoint_sha256: await hashFile(path.resolve(ROOT, config.candidate_checkpoint || config.checkpoint)),
        executable_sha256: await hashFile(path.join(ROOT, 'build-fresh/showdown_client.exe')),
        source_sha256: Object.fromEntries(await Promise.all(sources.map(async source =>
            [source, await hashFile(path.join(ROOT, source))])))};
    if (config.opponents?.includes('pilot'))
        identity.pilot_checkpoint_sha256 = await hashFile(path.resolve(ROOT, config.pilot_checkpoint));
    const directory = path.join(ROOT, 'matches/runs', name);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const filename = path.join(directory, `${name}_summary.json`);
        const prior = await loadSaved(filename);
        const games = prior ? validateResume(prior, identity, config) : [];
        if (config.collect_demonstrations) await verifyReplays(directory, games, isValid);
        const gate = createResourceGate({maxSystemMemoryPercent: config.max_system_memory_percent,
            maxSystemCpuPercent: config.max_system_cpu_percent, resourceCheckSeconds: config.resource_check_seconds,
            launchStaggerMilliseconds: config.launch_stagger_ms});
        const sim = require('../external/pokemon-showdown/dist/sim');
        async function publish(status, error = null) {
            const results = summarize(games, config.opponents);
            const valid = Object.values(results).reduce((sum, result) => sum + result.valid_games, 0);
            const elapsed = games.reduce((sum, game) => sum + game.duration_seconds, 0);
            await writeJsonAtomic(filename, {diagnostic: config.collect_demonstrations ?
                'tactical_teacher_demonstration_collection' : config.tactical_teacher === false ?
                    'neural_candidate_strength_benchmark' : 'tactical_teacher_strength_benchmark', run_name: name,
                identity, status, error, training_eligible: false, target_valid_games: targetGames,
                candidate_policy_mode: config.tactical_teacher === false ? 'neural' : 'tactical',
                candidate_inference_policy: config.candidate_inference_policy || 'sample',
                decision_audit: config.decision_audit ? Object.fromEntries(Object.keys(results).map(opponent =>
                    [opponent, summarizeDecisions(games.filter(game => game.opponent === opponent && isValid(game)))])) : null,
                demonstration_purpose: config.collect_demonstrations ? 'supervised_imitation' : null,
                ppo_eligible: false,
                valid_games: valid, elapsed_seconds: elapsed,
                estimated_remaining_seconds: valid ? elapsed / valid * (targetGames - valid) : null,
                demonstration_count: games.filter(game => game.demonstration).length,
                demonstration_turns: games.reduce((sum, game) => sum + (game.demonstration?.decision_turns || 0), 0),
                interval_method: config.collect_demonstrations ? '95% Wilson score; draws count half' :
                    '95% Wilson score; draws count half; paired games are not independent',
                results, games});
        }
        try {
            await publish('in_progress');
            for (let next; (next = nextGame(config, games));) {
                const start = Date.now();
                console.log(`[tactical-benchmark] ${next.opponent} side=${next.teacher_side} ` +
                    `attempt=${Math.floor(next.index / 2) + 1}/${config.max_attempts_per_side}`);
                await gate.waitForLaunch();
                const replay = [];
                const battleConfig = config.collect_demonstrations && next.opponent === 'random' ?
                    {...config, seed: config.seed + 10000000} : config;
                const game = await playGame(sim, battleConfig, next.index, {opponent: next.opponent,
                    matchedTeams: !config.collect_demonstrations,
                    recordTeacher: config.collect_demonstrations ? line => replay.push(line) : null});
                if (config.collect_demonstrations && isValid(game))
                    game.demonstration = await publishReplay(directory, next.opponent, next.index, replay, game);
                games.push({...game, ...next, duration_seconds: (Date.now() - start) / 1000});
                await publish('in_progress');
                const results = summarize(games, config.opponents);
                const valid = Object.values(results).reduce((sum, result) => sum + result.valid_games, 0);
                const elapsed = games.reduce((sum, record) => sum + record.duration_seconds, 0);
                const eta = valid ? `${Math.ceil(elapsed / valid * (targetGames - valid) / 60)} min` : 'unknown';
                console.log(`[tactical-benchmark] valid=${valid}/${targetGames} ` +
                    `invalid=${games.length - valid} elapsed=${Math.ceil(elapsed / 60)} min total_eta=${eta}`);
                if (!isValid(game)) console.error(`[tactical-benchmark] invalid: ` +
                    JSON.stringify({error: game.error, cleanup_error: game.cleanup_error, rejections: game.choice_rejections}));
            }
            await publish('completed');
            console.log(`[tactical-benchmark] completed summary=${filename}`);
        } catch (error) {
            await publish('failed', error.stack || error.message);
            throw error;
        }
    } finally { await lock.release(); }
}

if (require.main === module) main().catch(error => {
    console.error(`[tactical-benchmark] ${error.stack || error.message}`);
    process.exitCode = 1;
});

module.exports = {isValid, loadSaved, nextGame, summarize, validateConfig, validateResume, wilson};
