'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {aggregateGames, hashFile, playGame} = require('./tactical_teacher_smoke');
const {acquireRunLock, createResourceGate, writeJsonAtomic} = require('./counterfactual_rollout');
const ROOT = path.resolve(__dirname, '..');
const OPPONENTS = ['parent', 'random'];

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

function summarize(games) {
    const results = {};
    for (const opponent of OPPONENTS) {
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
        coverage: aggregateGames(records.filter(isValid)),
        choice_rejections: records.reduce((sum, game) => sum + (game.choice_rejections?.length || 0), 0)};
    }
    return results;
}

function nextGame(config, games) {
    const results = summarize(games);
    for (const opponent of OPPONENTS) {
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
            if (!Array.isArray(game.choice_rejections) || !['win', 'loss', 'draw'].includes(game.result?.result))
                throw new Error('saved completed game has no terminal result or rejection record');
            aggregateGames([game]);
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
    if (config.format !== 'gen9randomdoublesbattle' || config.agent_omp_threads !== 1 ||
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
    const sources = ['tools/tactical_teacher_benchmark.js', 'tools/tactical_teacher_smoke.js',
        'tools/counterfactual_rollout.js', 'external/pokemon-showdown/dist/sim/battle.js',
        'external/pokemon-showdown/dist/sim/battle-stream.js',
        'external/pokemon-showdown/dist/sim/tools/random-player-ai.js'];
    const identity = {config, checkpoint_sha256: await hashFile(path.resolve(ROOT, config.checkpoint)),
        executable_sha256: await hashFile(path.join(ROOT, 'build-fresh/showdown_client.exe')),
        source_sha256: Object.fromEntries(await Promise.all(sources.map(async source =>
            [source, await hashFile(path.join(ROOT, source))])))};
    const directory = path.join(ROOT, 'matches/runs', name);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const filename = path.join(directory, `${name}_summary.json`);
        const prior = await loadSaved(filename);
        const games = prior ? validateResume(prior, identity, config) : [];
        const gate = createResourceGate({maxSystemMemoryPercent: config.max_system_memory_percent,
            maxSystemCpuPercent: config.max_system_cpu_percent, resourceCheckSeconds: config.resource_check_seconds,
            launchStaggerMilliseconds: config.launch_stagger_ms});
        const sim = require('../external/pokemon-showdown/dist/sim');
        async function publish(status, error = null) {
            const results = summarize(games);
            const valid = Object.values(results).reduce((sum, result) => sum + result.valid_games, 0);
            const elapsed = games.reduce((sum, game) => sum + game.duration_seconds, 0);
            await writeJsonAtomic(filename, {diagnostic: 'tactical_teacher_strength_benchmark', run_name: name,
                identity, status, error, training_eligible: false, target_valid_games: 4 * config.games_per_side,
                valid_games: valid, elapsed_seconds: elapsed,
                estimated_remaining_seconds: valid ? elapsed / valid * (4 * config.games_per_side - valid) : null,
                interval_method: '95% Wilson score; draws count half; paired games are not independent',
                results, games});
        }
        try {
            await publish('in_progress');
            for (let next; (next = nextGame(config, games));) {
                const start = Date.now();
                console.log(`[tactical-benchmark] ${next.opponent} side=${next.teacher_side} ` +
                    `attempt=${Math.floor(next.index / 2) + 1}/${config.max_attempts_per_side}`);
                await gate.waitForLaunch();
                const game = await playGame(sim, config, next.index, {opponent: next.opponent, matchedTeams: true});
                games.push({...game, ...next, duration_seconds: (Date.now() - start) / 1000});
                await publish('in_progress');
                const results = summarize(games);
                const valid = Object.values(results).reduce((sum, result) => sum + result.valid_games, 0);
                const elapsed = games.reduce((sum, record) => sum + record.duration_seconds, 0);
                const eta = valid ? `${Math.ceil(elapsed / valid * (4 * config.games_per_side - valid) / 60)} min` : 'unknown';
                console.log(`[tactical-benchmark] valid=${valid}/${4 * config.games_per_side} ` +
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
