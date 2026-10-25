'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {
    AgentProcess, acquireRunLock, createResourceGate, drivePlayer,
    limitUtf8, seedTuple, writeJsonAtomic,
} = require('./counterfactual_rollout');

const REPO_ROOT = path.resolve(__dirname, '..');

async function hashFile(filename) {
    const hash = crypto.createHash('sha256');
    for await (const chunk of require('node:fs').createReadStream(filename)) hash.update(chunk);
    return hash.digest('hex');
}

function aggregateGames(games) {
    const totals = {completed: 0, failed: 0, attempts: 0, applied: 0, forced_switches: 0, fallbacks: 0};
    for (const game of games) {
        if (game.status !== 'completed') { ++totals.failed; continue; }
        ++totals.completed;
        const report = game.coverage;
        for (const key of ['attempts', 'applied', 'forced_switches', 'fallbacks']) {
            const value = report[`tactical_${key}`];
            if (!Number.isSafeInteger(value) || value < 0) throw new Error(`invalid coverage: ${key}`);
            totals[key] += value;
        }
        if (report.training_eligible !== false || report.policy_tag !== 'tactical-eval-v1' ||
                report.tactical_attempts !== report.tactical_applied + report.tactical_fallbacks ||
                report.tactical_forced_switches > report.tactical_applied) {
            throw new Error('missing or inconsistent evaluation-only teacher coverage');
        }
    }
    totals.coverage_rate = totals.attempts ? totals.applied / totals.attempts : 0;
    totals.power_decisions = totals.applied - totals.forced_switches;
    return totals;
}

async function closeGame(streams, agents) {
    agents.forEach(agent => agent.terminate());
    // Showdown's writeEnd checks atEOF; destroy() resolves the read stream a
    // second time after a completed battle and throws before results are saved.
    const cleanup = await Promise.allSettled([
        Promise.resolve().then(() => streams.omniscient.writeEnd()),
        ...agents.map(agent => Promise.resolve().then(() => agent.close())),
    ]);
    const failures = cleanup.filter(result => result.status === 'rejected');
    if (failures.length) throw new AggregateError(failures.map(result => result.reason), 'battle cleanup failed');
}

async function driveRandom(stream, seed) {
    const {RandomPlayerAI} = require('../external/pokemon-showdown/dist/sim/tools/random-player-ai');
    const choiceRejections = [];
    class RecordedRandomPlayer extends RandomPlayerAI {
        receiveError(error) {
            choiceRejections.push({player: 'RandomPolicy', message: error.message});
            return super.receiveError(error);
        }
    }
    const player = new RecordedRandomPlayer(stream, {seed});
    await player.start();
    const terminal = player.log.find(line => line.startsWith('|win|') || line === '|tie|');
    if (!terminal) throw new Error('random player ended without a result');
    return {result: {result: terminal === '|tie|' ? 'draw' :
        terminal === '|win|RandomPolicy' ? 'win' : 'loss'}, choiceRejections};
}

function battleSeeds(seed, index, matchedTeams) {
    const teacherSide = index % 2;
    const position = matchedTeams ? Math.floor(index / 2) : index;
    return {battle: seedTuple(seed, position, 0), random: seedTuple(seed, position, 3),
        players: [0, 1].map(side => seedTuple(seed, position,
            matchedTeams ? (side === teacherSide ? 1 : 2) : side + 1)),
        inference: [0, 1].map(side => seed + position * 2 +
            (matchedTeams ? (side === teacherSide ? 0 : 1) : side))};
}

function createBattleAgents(config, seeds, candidateSide, opponent, recordTeacher) {
    return [0, 1].map(side => opponent === 'random' && side !== candidateSide ? null : new AgentProcess(
        path.resolve(REPO_ROOT, side === candidateSide ? (config.candidate_checkpoint || config.checkpoint) : config.checkpoint),
        seeds.inference[side], null, config.agent_omp_threads,
        side === candidateSide && config.tactical_teacher !== false, side === candidateSide ? recordTeacher : null));
}

async function playGame(sim, config, index, {opponent = 'parent', matchedTeams = false, recordTeacher = null} = {}) {
    const battleId = `tactical-${opponent}-${config.seed}-${index}`;
    const teacherSide = index % 2 === 0 ? 0 : 1;
    const opponentName = opponent === 'random' ? 'RandomPolicy' : 'ParentPolicy';
    const candidateName = config.tactical_teacher === false ? 'NeuralCandidate' : 'TacticalTeacher';
    const names = teacherSide === 0 ? [candidateName, opponentName] : [opponentName, candidateName];
    const battle = new sim.BattleStream();
    const streams = sim.getPlayerStreams(battle);
    const seeds = battleSeeds(config.seed, index, matchedTeams);
    const agents = createBattleAgents(config, seeds, teacherSide, opponent, recordTeacher);
    let timeout;
    let game;
    try {
        const work = (async () => {
            await Promise.all(agents.filter(Boolean).map(agent => agent.start()));
            const players = [streams.p1, streams.p2].map((stream, side) =>
                agents[side] ? drivePlayer(stream, agents[side], battleId, names[side], config.format) :
                    driveRandom(stream, seeds.random));
            const drain = (async () => { for await (const chunk of streams.omniscient) { void chunk; } })();
            const results = Promise.all([...players, drain]);
            // Attach rejection handling before the first simulator write can
            // trigger a player error. The awaited result still reports it below.
            results.catch(() => {});
            await streams.omniscient.write(
                `>start ${JSON.stringify({formatid: config.format, seed: seeds.battle})}\n` +
                [0, 1].map(side => `>player p${side + 1} ${JSON.stringify({name: names[side],
                    seed: seeds.players[side]})}`)
                    .join('\n'));
            const [first, second] = await results;
            const teacher = teacherSide === 0 ? first : second;
            const other = teacherSide === 0 ? second : first;
            if (!['win', 'loss', 'draw'].includes(teacher.result.result) ||
                    (teacher.result.result === 'draw' ? other.result.result !== 'draw' :
                        other.result.result !== (teacher.result.result === 'win' ? 'loss' : 'win'))) {
                throw new Error('players reported inconsistent terminal results');
            }
            const game = {
                index, status: 'completed', teacher_side: teacherSide === 0 ? 'a' : 'b', seeds,
                result: teacher.result, policy_mode: config.tactical_teacher === false ? 'neural' : 'tactical',
                policy_tag: teacher.episode.policy_tag,
                coverage: config.tactical_teacher === false ? null : teacher.episode,
                choice_rejections: [...first.choiceRejections, ...second.choiceRejections],
            };
            if (config.tactical_teacher !== false) aggregateGames([game]);
            return game;
        })();
        game = await Promise.race([work, new Promise((_, reject) => {
            timeout = setTimeout(() => reject(new Error('battle timeout')), config.battle_timeout_seconds * 1000);
        })]);
    } catch (error) {
        game = {index, status: 'failed', teacher_side: teacherSide === 0 ? 'a' : 'b',
            error: limitUtf8(error.stack || error.message, 16384).text};
    } finally {
        clearTimeout(timeout);
        try { await closeGame(streams, agents.filter(Boolean)); }
        catch (error) {
            game = {...game, index, status: 'failed',
                cleanup_error: limitUtf8(error.stack || error.message, 16384).text};
        }
    }
    return game;
}

async function main(argv = process.argv.slice(2)) {
    const args = {};
    for (let i = 0; i < argv.length; i += 2) {
        if (!['--run-name', '--config'].includes(argv[i]) || !argv[i + 1])
            throw new Error('usage: node tools/tactical_teacher_smoke.js --run-name NAME [--config PATH]');
        args[argv[i]] = argv[i + 1];
    }
    const name = args['--run-name'];
    if (!name || !/^[A-Za-z0-9_-]+$/.test(name)) throw new Error('a simple --run-name is required');
    const config = JSON.parse(await fs.readFile(path.resolve(REPO_ROOT,
        args['--config'] || 'config/experimental/tactical_teacher_smoke.json'), 'utf8'));
    if (config.format !== 'gen9randomdoublesbattle' ||
            !Number.isSafeInteger(config.games) || config.games < 2 || config.games > 20 || config.games % 2 ||
            config.agent_omp_threads !== 1 || !Number.isSafeInteger(config.seed) || config.seed < 0 ||
            !(config.battle_timeout_seconds > 0 && config.battle_timeout_seconds <= 300) ||
            !(config.max_system_memory_percent > 0 && config.max_system_memory_percent <= 65) ||
            !(config.max_system_cpu_percent > 0 && config.max_system_cpu_percent <= 65) ||
            !(config.resource_check_seconds > 0) || !(config.launch_stagger_ms >= 0)) {
        throw new Error('smoke config requires 2-20 even games, one thread per agent, and resource limits at most 65%');
    }
    const identity = {
        config, checkpoint_sha256: await hashFile(path.resolve(REPO_ROOT, config.checkpoint)),
        executable_sha256: await hashFile(path.join(REPO_ROOT, 'build-fresh/showdown_client.exe')),
    };
    const directory = path.join(REPO_ROOT, 'matches/runs', name);
    await fs.mkdir(directory, {recursive: true});
    const lock = await acquireRunLock(directory);
    try {
        const summaryPath = path.join(directory, `${name}_summary.json`);
        let prior;
        try { prior = JSON.parse(await fs.readFile(summaryPath, 'utf8')); }
        catch (error) { if (error.code !== 'ENOENT') throw error; }
        if (prior && JSON.stringify(prior.identity) !== JSON.stringify(identity))
            throw new Error('config, checkpoint, or executable changed; choose a new run name');
        const games = prior?.games || [];
        if (!Array.isArray(games) || games.length > config.games ||
                games.some((game, index) => game.index !== index ||
                    !['completed', 'failed'].includes(game.status))) {
            throw new Error('saved smoke game records are inconsistent');
        }
        const gate = createResourceGate({
            maxSystemMemoryPercent: config.max_system_memory_percent,
            maxSystemCpuPercent: config.max_system_cpu_percent,
            resourceCheckSeconds: config.resource_check_seconds,
            launchStaggerMilliseconds: config.launch_stagger_ms,
        });
        const sim = require(path.join(REPO_ROOT, 'external/pokemon-showdown/dist/sim'));
        async function publish() {
            const coverage = aggregateGames(games);
            const summary = {
                diagnostic: 'tactical_teacher_runtime_smoke', run_name: name, identity,
                status: games.length < config.games ? 'in_progress' : coverage.failed ? 'failed' : 'completed',
                direction_step: 'project_direction.txt: step 2, runtime legality and coverage',
                strength_claim: false, training_eligible: false, coverage, games,
            };
            await writeJsonAtomic(summaryPath, summary);
            return summary;
        }
        await publish();
        for (let index = games.length; index < config.games; ++index) {
            await gate.waitForLaunch();
            games.push(await playGame(sim, config, index));
            const summary = await publish();
            console.log(`[tactical-smoke] games=${games.length}/${config.games} ` +
                `failed=${summary.coverage.failed} coverage=${(100 * summary.coverage.coverage_rate).toFixed(1)}%`);
            const game = games[index];
            if (game.status === 'failed') console.error(`[tactical-smoke] game=${index + 1} ` +
                [game.error, game.cleanup_error].filter(Boolean).join('\n'));
        }
        const summary = await publish();
        console.log(`[tactical-smoke] status=${summary.status} summary=${summaryPath}`);
        if (summary.status !== 'completed') process.exitCode = 1;
    } finally { await lock.release(); }
}

if (require.main === module) main().catch(error => {
    console.error(`[tactical-smoke] ${error.stack || error.message}`);
    process.exitCode = 1;
});

module.exports = {aggregateGames, battleSeeds, closeGame, createBattleAgents, driveRandom, hashFile, playGame};
