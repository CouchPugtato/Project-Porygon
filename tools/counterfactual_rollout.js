#!/usr/bin/env node

'use strict';

const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const readline = require('readline');
const {spawn} = require('child_process');

const REPO_ROOT = path.resolve(__dirname, '..');
const CLIENT_EXE = path.join(REPO_ROOT, 'build-fresh', 'showdown_client.exe');
const SHOWDOWN_DIR = path.join(REPO_ROOT, 'external', 'pokemon-showdown');

function parseToml(text) {
    const values = {};
    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.replace(/\s+#.*$/, '').trim();
        if (!line || line.startsWith('[')) continue;
        const match = line.match(/^([A-Za-z0-9_]+)\s*=\s*(.+)$/);
        if (!match) continue;
        let value = match[2].trim();
        if (value.startsWith('"') && value.endsWith('"')) {
            value = value.slice(1, -1);
        } else if (value === 'true' || value === 'false') {
            value = value === 'true';
        } else if (/^-?\d+$/.test(value)) {
            value = Number.parseInt(value, 10);
        }
        values[match[1].replaceAll('_', '-')] = value;
    }
    return values;
}

function parseArgs(argv) {
    const cli = {};
    for (let i = 0; i < argv.length; ++i) {
        const token = argv[i];
        if (!token.startsWith('--')) throw new Error(`unexpected argument: ${token}`);
        const key = token.slice(2);
        if (i + 1 >= argv.length || argv[i + 1].startsWith('--')) {
            cli[key] = true;
        } else {
            cli[key] = argv[++i];
        }
    }
    const configPath = path.resolve(REPO_ROOT, cli.config || 'config/experimental/counterfactual_rollout.toml');
    let config = {};
    if (fs.existsSync(configPath)) config = parseToml(fs.readFileSync(configPath, 'utf8'));
    return {...config, ...cli, config: configPath};
}

function integerOption(options, name, fallback, minimum = Number.MIN_SAFE_INTEGER) {
    const value = options[name] === undefined ? fallback : Number.parseInt(String(options[name]), 10);
    if (!Number.isSafeInteger(value) || value < minimum) {
        throw new Error(`--${name} must be an integer >= ${minimum}`);
    }
    return value;
}

function booleanOption(options, name, fallback) {
    if (options[name] === undefined) return fallback;
    const value = String(options[name]).toLowerCase();
    if (value === 'true' || value === '1') return true;
    if (value === 'false' || value === '0') return false;
    throw new Error(`--${name} must be true or false`);
}

function resolveOptions(argv) {
    const options = parseArgs(argv);
    const runName = String(options['run-name'] || '').trim();
    const checkpoint = path.resolve(REPO_ROOT, String(options.checkpoint || ''));
    const opponentCheckpoint = path.resolve(
        REPO_ROOT, String(options['opponent-checkpoint'] || options.checkpoint || '')
    );
    const decisionMin = integerOption(options, 'decision-min', 0, 0);
    const decisionMax = integerOption(options, 'decision-max', 7, decisionMin);
    if (!runName) throw new Error('--run-name is required');
    if (!options.checkpoint) throw new Error('--checkpoint is required');
    if (!fs.existsSync(checkpoint)) throw new Error(`checkpoint not found: ${checkpoint}`);
    if (!fs.existsSync(opponentCheckpoint)) {
        throw new Error(`opponent checkpoint not found: ${opponentCheckpoint}`);
    }
    if (!fs.existsSync(CLIENT_EXE)) throw new Error(`trainer executable not found: ${CLIENT_EXE}`);
    if (!fs.existsSync(path.join(SHOWDOWN_DIR, 'dist', 'sim', 'index.js'))) {
        throw new Error(`built Showdown simulator not found: ${SHOWDOWN_DIR}`);
    }
    const format = String(options.format || 'gen9randomdoublesbattle');
    if (!format.toLowerCase().includes('double') && !format.toLowerCase().includes('vgc')) {
        throw new Error('counterfactual rollout currently supports doubles formats only');
    }
    return {
        runName,
        checkpoint,
        opponentCheckpoint,
        format,
        pairs: integerOption(options, 'pairs', 500, 1),
        concurrency: integerOption(options, 'concurrency', 4, 1),
        decisionMin,
        decisionMax,
        seed: integerOption(options, 'seed', 20260911, 0),
        battleTimeoutSeconds: integerOption(options, 'battle-timeout-seconds', 180, 1),
        maxAttemptMultiplier: integerOption(options, 'max-attempt-multiplier', 3, 1),
        manifestCheckpointResults: integerOption(
            options, 'manifest-checkpoint-results', 10, 1),
        resume: booleanOption(options, 'resume', true),
        configPath: options.config,
    };
}

function mix32(value) {
    let x = value >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;
    return x >>> 0;
}

function seedTuple(base, attempt, lane) {
    const values = [];
    for (let i = 0; i < 4; ++i) {
        values.push(mix32(base + Math.imul(attempt + 1, 0x9e3779b9) + lane * 17 + i) & 0xffff);
    }
    return values.join(',');
}

function inferenceSeed(base, attempt, lane) {
    return mix32(base + Math.imul(attempt + 1, 0x85ebca6b) + lane * 101) & 0x7fffffff;
}

class AgentProcess {
    constructor(checkpoint, inferenceSeedValue, intervention) {
        this.checkpoint = checkpoint;
        this.inferenceSeed = inferenceSeedValue;
        this.intervention = intervention;
        this.process = null;
        this.messages = [];
        this.waiters = [];
        this.stderr = [];
        this.exitError = null;
    }

    async start() {
        const args = [
            '--battle-agent', this.checkpoint,
            '--inference-seed', String(this.inferenceSeed),
            '--reward-mode', 'terminal',
        ];
        if (this.intervention) {
            args.push(
                '--counterfactual-decision', String(this.intervention.decisionIndex),
                '--counterfactual-rank', String(this.intervention.actionRank)
            );
        }
        this.process = spawn(CLIENT_EXE, args, {
            cwd: REPO_ROOT,
            windowsHide: true,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        const stdout = readline.createInterface({input: this.process.stdout});
        const stderr = readline.createInterface({input: this.process.stderr});
        stdout.on('line', line => {
            try {
                this.pushMessage(JSON.parse(line));
            } catch (_) {
                this.stderr.push(`stdout: ${line}`);
            }
        });
        stderr.on('line', line => {
            this.stderr.push(line);
            if (this.stderr.length > 20) this.stderr.shift();
        });
        this.process.once('exit', code => {
            if (code !== 0 || this.waiters.length > 0) {
                this.exitError = new Error(
                    `showdown_client exited with ${code}: ${this.stderr.join(' | ')}`
                );
                for (const waiter of this.waiters.splice(0)) waiter.reject(this.exitError);
            }
        });
        await this.nextMessage('ready');
    }

    pushMessage(message) {
        if (message.type === 'error') {
            const error = new Error(`showdown_client: ${message.message || 'runtime error'}`);
            for (const waiter of this.waiters.splice(0)) waiter.reject(error);
            this.exitError = error;
            return;
        }
        const waiterIndex = this.waiters.findIndex(waiter => waiter.type === message.type);
        if (waiterIndex >= 0) {
            const [waiter] = this.waiters.splice(waiterIndex, 1);
            waiter.resolve(message);
        } else {
            this.messages.push(message);
        }
    }

    nextMessage(type) {
        if (this.exitError) return Promise.reject(this.exitError);
        const index = this.messages.findIndex(message => message.type === type);
        if (index >= 0) return Promise.resolve(this.messages.splice(index, 1)[0]);
        return new Promise((resolve, reject) => this.waiters.push({type, resolve, reject}));
    }

    send(payload) {
        if (!this.process || !this.process.stdin.writable) {
            throw new Error('attempted to write to a closed showdown_client');
        }
        this.process.stdin.write(`${JSON.stringify(payload)}\n`);
    }

    async close() {
        if (!this.process) return;
        const process = this.process;
        this.process = null;
        if (process.stdin.writable) process.stdin.end();
        if (process.exitCode === null) {
            await new Promise(resolve => process.once('exit', resolve));
        }
    }

    terminate() {
        if (this.process && this.process.exitCode === null) this.process.kill();
    }
}

function runtimeEvent(battleId, sequence, line) {
    return {type: 'event', battle_id: battleId, seq: sequence, line};
}

function terminalResult(line, playerName) {
    if (line.startsWith('|tie|')) return {result: 'draw', reward: 0};
    if (!line.startsWith('|win|')) return null;
    const winner = line.slice('|win|'.length);
    return winner === playerName ? {result: 'win', reward: 1} : {result: 'loss', reward: -1};
}

async function drivePlayer(stream, agent, battleId, playerName, format) {
    let sequence = 0;
    let episode = null;
    let intervention = null;
    let pendingDecision = null;
    let result = null;
    const choiceRejections = [];
    agent.send({
        type: 'battle_start',
        battle_id: battleId,
        format,
        is_doubles: true,
    });
    for await (const chunk of stream) {
        for (const line of chunk.split('\n')) {
            if (!line) continue;
            if (line.startsWith('|error|')) {
                if (!pendingDecision) {
                    throw new Error(`Showdown reported an unexpected choice error: ${line}`);
                }
                agent.send({
                    ...pendingDecision,
                    type: 'decision_rejected',
                    reason: line,
                });
                choiceRejections.push({
                    player: playerName,
                    request_id: pendingDecision.request_id,
                    command: pendingDecision.command,
                    message: line,
                });
                pendingDecision = null;
                ++sequence;
                continue;
            }
            if (pendingDecision) {
                agent.send(pendingDecision);
                pendingDecision = null;
            }
            if (line.startsWith('|request|')) {
                const request = JSON.parse(line.slice('|request|'.length));
                if (!request || request.wait) continue;
                agent.send({
                    type: 'request',
                    battle_id: battleId,
                    request_id: sequence,
                    payload: request,
                });
                let action;
                try {
                    action = await agent.nextMessage('action');
                } catch (error) {
                    error.message = `${error.message}; battle_id=${battleId}; request=${JSON.stringify(request)}`;
                    throw error;
                }
                if (action.counterfactual) intervention = action;
                pendingDecision = {
                    type: 'decision_accepted',
                    battle_id: battleId,
                    request_id: sequence,
                    action: action.action,
                    action2: action.action2,
                    command: action.command,
                };
                const command = String(action.command).replace(/^\/choose\s+/, '');
                await stream.write(command);
            } else {
                const terminal = terminalResult(line, playerName);
                if (terminal && !result) {
                    result = terminal;
                    agent.send({type: 'terminal', battle_id: battleId, ...terminal});
                    agent.send({type: 'battle_end', battle_id: battleId});
                    episode = await agent.nextMessage('episode_complete');
                } else if (!terminal) {
                    agent.send(runtimeEvent(battleId, sequence, line));
                }
            }
            ++sequence;
        }
    }
    if (!episode || !result) throw new Error(`battle ${battleId} ended without an episode`);
    return {episode, intervention, result, choiceRejections};
}

async function runBranch(sim, options, attempt, decisionIndex, actionRank) {
    const battleId = `counterfactual-${attempt}-${actionRank}`;
    const candidateName = 'CounterfactualCandidate';
    const opponentName = 'CounterfactualOpponent';
    const battle = new sim.BattleStream();
    const streams = sim.getPlayerStreams(battle);
    const candidate = new AgentProcess(
        options.checkpoint,
        inferenceSeed(options.seed, attempt, 0),
        {decisionIndex, actionRank}
    );
    const opponent = new AgentProcess(
        options.opponentCheckpoint,
        inferenceSeed(options.seed, attempt, 1),
        null
    );
    let timeout;
    try {
        const work = (async () => {
            await Promise.all([candidate.start(), opponent.start()]);
            const candidateDrive = drivePlayer(
                streams.p1, candidate, battleId, candidateName, options.format);
            const opponentDrive = drivePlayer(
                streams.p2, opponent, battleId, opponentName, options.format);
            const drain = (async () => {
                for await (const _chunk of streams.omniscient) {
                    // Draining prevents the simulator's output buffer from becoming the bottleneck.
                }
            })();
            const spec = {formatid: options.format, seed: seedTuple(options.seed, attempt, 0)};
            const p1 = {name: candidateName, seed: seedTuple(options.seed, attempt, 1)};
            const p2 = {name: opponentName, seed: seedTuple(options.seed, attempt, 2)};
            await streams.omniscient.write(
                `>start ${JSON.stringify(spec)}\n` +
                `>player p1 ${JSON.stringify(p1)}\n` +
                `>player p2 ${JSON.stringify(p2)}`
            );
            const [candidateResult, opponentResult] = await Promise.all([
                candidateDrive, opponentDrive, drain,
            ]);
            candidateResult.choiceRejections = [
                ...candidateResult.choiceRejections.map(rejection => ({
                    ...rejection,
                    role: 'candidate',
                    action_rank: actionRank,
                })),
                ...opponentResult.choiceRejections.map(rejection => ({
                    ...rejection,
                    role: 'opponent',
                    action_rank: actionRank,
                })),
            ];
            return candidateResult;
        })();
        const timedOut = new Promise((_, reject) => {
            timeout = setTimeout(() => {
                candidate.terminate();
                opponent.terminate();
                reject(new Error(
                    `battle branch exceeded ${options.battleTimeoutSeconds}s timeout`));
            }, options.battleTimeoutSeconds * 1000);
        });
        return await Promise.race([work, timedOut]);
    } finally {
        clearTimeout(timeout);
        try {
            await streams.omniscient.writeEnd();
        } catch (_) {
            // A completed BattleStream has already closed its input.
        }
        await Promise.all([candidate.close(), opponent.close()]);
    }
}

function equalPrefix(left, right, count) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length < count || right.length < count) {
        return false;
    }
    for (let i = 0; i < count; ++i) {
        if (left[i] !== right[i]) return false;
    }
    return true;
}

function interventionChoice(action) {
    return {
        action: action.action,
        action2: action.action2,
        slot0_target: action.slot0_target_index,
        slot1_target: action.slot1_target_index,
    };
}

function choicesDiffer(left, right) {
    return left.action !== right.action || left.action2 !== right.action2 ||
        left.slot0_target !== right.slot0_target || left.slot1_target !== right.slot1_target;
}

function buildCounterfactualSample(action, pairId, policyTag, returnValue) {
    return {
        type: 'counterfactual_sample',
        pair_id: pairId,
        policy_tag: policyTag,
        action_rank: action.counterfactual.action_rank,
        decision_index: action.counterfactual.decision_index,
        action: action.action,
        action2: action.action2,
        slot0_has_action: action.slot0_has_action,
        slot0_kind: action.slot0_kind,
        slot0_move_index: action.slot0_move_index,
        slot0_switch_index: action.slot0_switch_index,
        slot0_use_tera: action.slot0_use_tera,
        slot0_target_index: action.slot0_target_index,
        slot0_target_mask: action.slot0_target_mask,
        slot1_has_action: action.slot1_has_action,
        slot1_kind: action.slot1_kind,
        slot1_move_index: action.slot1_move_index,
        slot1_switch_index: action.slot1_switch_index,
        slot1_use_tera: action.slot1_use_tera,
        slot1_target_index: action.slot1_target_index,
        slot1_target_mask: action.slot1_target_mask,
        baseline_value: action.baseline_value,
        behavior_log_prob: action.behavior_log_prob,
        target_value: returnValue,
        hidden_dim: action.hidden_dim,
        hidden_state: action.hidden_state,
        legal_mask: action.legal_mask,
    };
}

async function runPair(sim, options, attempt) {
    const decisionSpan = options.decisionMax - options.decisionMin + 1;
    const decisionIndex = options.decisionMin + (attempt % decisionSpan);
    const [rank0, rank1] = await Promise.all([
        runBranch(sim, options, attempt, decisionIndex, 0),
        runBranch(sim, options, attempt, decisionIndex, 1),
    ]);
    const choiceRejections = [
        ...rank0.choiceRejections,
        ...rank1.choiceRejections,
    ];
    if (!rank0.intervention || !rank1.intervention) {
        return {
            valid: false,
            reason: 'intervention_not_reached',
            message: 'one or both branches ended before reaching the requested intervention',
            choiceRejections,
        };
    }
    if (rank0.intervention.hidden_dim !== rank1.intervention.hidden_dim ||
            !equalPrefix(
                rank0.intervention.hidden_state, rank1.intervention.hidden_state,
                rank0.intervention.hidden_dim) ||
            !equalPrefix(rank0.intervention.legal_mask, rank1.intervention.legal_mask, 28)) {
        return {
            valid: false,
            reason: 'pre_intervention_state_mismatch',
            message: 'ranked branches did not expose the same pre-intervention state',
            choiceRejections,
        };
    }
    const choice0 = interventionChoice(rank0.intervention);
    const choice1 = interventionChoice(rank1.intervention);
    if (!choicesDiffer(choice0, choice1)) {
        return {
            valid: false,
            reason: 'ranked_actions_not_distinct',
            message: 'rank zero and rank one resolved to the same action and targets',
            choiceRejections,
        };
    }
    const pairId = `counterfactual-pair-${attempt}`;
    return {
        valid: true,
        pairId,
        attempt,
        decisionIndex,
        rank0Return: rank0.result.reward,
        rank1Return: rank1.result.reward,
        choice0,
        choice1,
        choiceRejections,
        records: [
            buildCounterfactualSample(
                rank0.intervention, pairId, options.checkpoint, rank0.result.reward),
            buildCounterfactualSample(
                rank1.intervention, pairId, options.checkpoint, rank1.result.reward),
        ],
    };
}

async function syncFile(filePath) {
    const handle = await fsp.open(filePath, 'r+');
    try {
        await handle.sync();
    } finally {
        await handle.close();
    }
}

async function writeTextAtomic(filePath, content, keepBackup = false) {
    const temporary = `${filePath}.tmp`;
    const handle = await fsp.open(temporary, 'w');
    try {
        await handle.writeFile(content, 'utf8');
        await handle.sync();
    } finally {
        await handle.close();
    }

    if (keepBackup && fs.existsSync(filePath)) {
        const backup = `${filePath}.bak`;
        const backupTemporary = `${backup}.tmp`;
        await fsp.copyFile(filePath, backupTemporary);
        await syncFile(backupTemporary);
        await fsp.rename(backupTemporary, backup);
    }
    await fsp.rename(temporary, filePath);
}

async function writeJsonAtomic(filePath, value) {
    await writeTextAtomic(filePath, `${JSON.stringify(value, null, 2)}\n`, true);
}

async function appendJsonLineDurably(filePath, value) {
    const handle = await fsp.open(filePath, 'a');
    try {
        await handle.writeFile(`${JSON.stringify(value)}\n`, 'utf8');
        await handle.sync();
    } finally {
        await handle.close();
    }
}

function choiceRejectionReason(message) {
    const normalized = String(message || '').toLowerCase();
    if (normalized.includes("can't switch") && normalized.includes('trapped')) {
        return 'trapped_switch';
    }
    if (normalized.includes('needs a target')) return 'missing_move_target';
    if (normalized.includes('invalid target')) return 'invalid_move_target';
    return 'other_choice_rejection';
}

function battleFailureReason(error) {
    const message = String(error?.message || error || '');
    if (message.toLowerCase().includes('timeout')) return 'battle_timeout';
    if (message.includes('showdown_client exited')) return 'agent_process_error';
    return 'battle_error';
}

async function summarizeFailureLog(filePath) {
    const summary = {
        choice_rejections_recovered: 0,
        choice_rejection_reasons: {},
        logged_attempt_failures: 0,
        logged_failure_reasons: {},
        highest_attempt: -1,
    };
    if (!fs.existsSync(filePath)) return summary;
    const lines = (await fsp.readFile(filePath, 'utf8')).split(/\r?\n/);
    const rejectionKeys = new Set();
    const failedAttempts = new Map();
    for (const line of lines) {
        if (!line.trim()) continue;
        let record;
        try {
            record = JSON.parse(line);
        } catch (_) {
            continue;
        }
        if (Number.isSafeInteger(record.attempt)) {
            summary.highest_attempt = Math.max(summary.highest_attempt, record.attempt);
        }
        if (record.type === 'attempt_failure' && Number.isSafeInteger(record.attempt)) {
            failedAttempts.set(record.attempt, record.reason || 'unknown');
        } else if (record.type === 'choice_rejection_recovered') {
            rejectionKeys.add(JSON.stringify([
                record.attempt,
                record.action_rank,
                record.role,
                record.player,
                record.request_id,
                record.command,
                record.message,
                record.reason,
            ]));
        }
    }
    for (const reason of failedAttempts.values()) {
        ++summary.logged_attempt_failures;
        summary.logged_failure_reasons[reason] =
            (summary.logged_failure_reasons[reason] || 0) + 1;
    }
    for (const key of rejectionKeys) {
        const reason = JSON.parse(key)[7] || 'other_choice_rejection';
        ++summary.choice_rejections_recovered;
        summary.choice_rejection_reasons[reason] =
            (summary.choice_rejection_reasons[reason] || 0) + 1;
    }
    return summary;
}

function emptyManifest(options) {
    return {
        schema_version: 2,
        status: 'running',
        run_name: options.runName,
        format: options.format,
        checkpoint: options.checkpoint,
        opponent_checkpoint: options.opponentCheckpoint,
        requested_pairs: options.pairs,
        seed: options.seed,
        decision_min: options.decisionMin,
        decision_max: options.decisionMax,
        battle_timeout_seconds: options.battleTimeoutSeconds,
        attempts: 0,
        completed_pairs: 0,
        invalid_attempts: 0,
        invalid_reasons: {},
        choice_rejections_recovered: 0,
        choice_rejection_reasons: {},
        logged_attempt_failures: 0,
        logged_failure_reasons: {},
        rank0_better: 0,
        rank1_better: 0,
        equal_returns: 0,
        elapsed_seconds: 0,
        pairs: [],
    };
}

function assertResumeCompatible(manifest, options) {
    for (const [key, expected] of [
        ['schema_version', 2],
        ['run_name', options.runName], ['format', options.format],
        ['checkpoint', options.checkpoint], ['opponent_checkpoint', options.opponentCheckpoint],
        ['seed', options.seed], ['decision_min', options.decisionMin],
        ['decision_max', options.decisionMax],
        ['battle_timeout_seconds', options.battleTimeoutSeconds],
    ]) {
        if (manifest[key] !== expected) {
            throw new Error(`resume mismatch for ${key}: ${manifest[key]} != ${expected}`);
        }
    }
}

function choiceFromRecord(record) {
    return {
        action: record.action,
        action2: record.action2,
        slot0_target: record.slot0_target_index,
        slot1_target: record.slot1_target_index,
    };
}

function pairEntryFromRecords(filename, attempt, records) {
    if (records.length !== 2) return null;
    const byRank = new Map(records.map(record => [record.action_rank, record]));
    const rank0 = byRank.get(0);
    const rank1 = byRank.get(1);
    if (!rank0 || !rank1 || rank0.type !== 'counterfactual_sample' ||
            rank1.type !== 'counterfactual_sample' ||
            rank0.pair_id !== rank1.pair_id ||
            rank0.decision_index !== rank1.decision_index ||
            !Number.isFinite(rank0.target_value) || !Number.isFinite(rank1.target_value)) {
        return null;
    }
    return {
        pair_id: rank0.pair_id,
        attempt,
        file: `pairs/${filename}`,
        decision_index: rank0.decision_index,
        rank0_return: rank0.target_value,
        rank1_return: rank1.target_value,
        rank0_choice: choiceFromRecord(rank0),
        rank1_choice: choiceFromRecord(rank1),
    };
}

async function readJsonIfValid(filePath) {
    try {
        const text = await fsp.readFile(filePath, 'utf8');
        if (!text.trim()) return null;
        const value = JSON.parse(text);
        return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
    } catch (_) {
        return null;
    }
}

function processIsRunning(pid) {
    if (!Number.isSafeInteger(pid) || pid <= 0) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch (error) {
        return error.code === 'EPERM';
    }
}

async function acquireRunLock(runDir) {
    const lockPath = path.join(runDir, '.counterfactual_rollout.lock');
    const token = `${process.pid}-${Date.now()}`;
    for (let attempt = 0; attempt < 2; ++attempt) {
        try {
            const handle = await fsp.open(lockPath, 'wx');
            await handle.writeFile(`${JSON.stringify({pid: process.pid, token})}\n`, 'utf8');
            await handle.sync();
            let released = false;
            return {
                async release() {
                    if (released) return;
                    released = true;
                    await handle.close();
                    const current = await readJsonIfValid(lockPath);
                    if (current?.token === token) await fsp.unlink(lockPath);
                },
            };
        } catch (error) {
            if (error.code !== 'EEXIST') throw error;
            const owner = await readJsonIfValid(lockPath);
            if (owner && processIsRunning(owner.pid)) {
                throw new Error(
                    `another counterfactual collector is already running for this run (pid ${owner.pid})`
                );
            }
            await fsp.unlink(lockPath).catch(unlinkError => {
                if (unlinkError.code !== 'ENOENT') throw unlinkError;
            });
        }
    }
    throw new Error(`could not acquire counterfactual run lock: ${lockPath}`);
}

async function recoverManifestFromPairFiles(pairDir, options, priorManifest = null) {
    const manifest = emptyManifest(options);
    const filenames = (await fsp.readdir(pairDir))
        .filter(name => /^pair_\d+\.jsonl$/.test(name))
        .sort();
    let highestAttempt = -1;
    let unreadablePairFiles = 0;

    for (const filename of filenames) {
        const attempt = Number.parseInt(filename.slice(5, -6), 10);
        highestAttempt = Math.max(highestAttempt, attempt);
        try {
            const text = await fsp.readFile(path.join(pairDir, filename), 'utf8');
            const lines = text.split(/\r?\n/).filter(line => line.trim());
            const entry = pairEntryFromRecords(
                filename, attempt, lines.map(line => JSON.parse(line)));
            if (entry) manifest.pairs.push(entry);
            else ++unreadablePairFiles;
        } catch (_) {
            ++unreadablePairFiles;
        }
    }

    manifest.pairs.sort((left, right) => left.attempt - right.attempt);
    manifest.completed_pairs = manifest.pairs.length;
    manifest.attempts = Math.max(
        highestAttempt + 1,
        Number.isSafeInteger(priorManifest?.attempts) ? priorManifest.attempts : 0
    );
    for (const pair of manifest.pairs) {
        if (pair.rank0_return > pair.rank1_return) ++manifest.rank0_better;
        else if (pair.rank1_return > pair.rank0_return) ++manifest.rank1_better;
        else ++manifest.equal_returns;
    }

    manifest.invalid_attempts = Math.max(0, manifest.attempts - manifest.completed_pairs);
    const knownReasons = priorManifest?.invalid_reasons;
    if (knownReasons && typeof knownReasons === 'object' && !Array.isArray(knownReasons)) {
        manifest.invalid_reasons = {...knownReasons};
    }
    const classified = Object.values(manifest.invalid_reasons)
        .reduce((total, count) => total + (Number.isSafeInteger(count) ? count : 0), 0);
    if (classified < manifest.invalid_attempts) {
        manifest.invalid_reasons.recovered_unclassified = manifest.invalid_attempts - classified;
    }
    manifest.elapsed_seconds = Number(priorManifest?.elapsed_seconds) || 0;
    manifest.choice_rejections_recovered =
        Number(priorManifest?.choice_rejections_recovered) || 0;
    if (priorManifest?.choice_rejection_reasons &&
            typeof priorManifest.choice_rejection_reasons === 'object') {
        manifest.choice_rejection_reasons = {...priorManifest.choice_rejection_reasons};
    }
    manifest.recovery = {
        recovered_at: new Date().toISOString(),
        pair_files_found: filenames.length,
        valid_pair_files: manifest.completed_pairs,
        unreadable_pair_files: unreadablePairFiles,
    };
    return manifest;
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return 'unknown';
    if (seconds < 60) return `${Math.ceil(seconds)}s`;
    if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
}

async function combinePairFiles(runDir, manifest) {
    const temporary = path.join(runDir, 'counterfactual_action_batch.jsonl.tmp');
    const output = path.join(runDir, 'counterfactual_action_batch.jsonl');
    const pairs = [...manifest.pairs].sort((left, right) => left.attempt - right.attempt);
    await fsp.writeFile(temporary, '', 'utf8');
    for (const pair of pairs) {
        const content = await fsp.readFile(path.join(runDir, pair.file), 'utf8');
        await fsp.appendFile(temporary, content, 'utf8');
    }
    await fsp.rename(temporary, output);
    return output;
}

async function collect(
    sim,
    options,
    runDir,
    pairDir,
    manifestPath,
    summaryPath,
    failureLogPath
) {
    let manifest = emptyManifest(options);
    if (options.resume && fs.existsSync(manifestPath)) {
        let priorManifest = await readJsonIfValid(manifestPath);
        let recoverySource = 'manifest';
        if (!priorManifest) {
            priorManifest = await readJsonIfValid(`${manifestPath}.bak`);
            recoverySource = priorManifest ? 'backup' : 'pair files';
        }
        if (priorManifest) assertResumeCompatible(priorManifest, options);
        manifest = await recoverManifestFromPairFiles(pairDir, options, priorManifest);
        manifest.recovery.source = recoverySource;
        manifest.status = 'running';
        console.log(
            `[counterfactual] resume source=${recoverySource} ` +
            `pairs=${manifest.completed_pairs} attempts=${manifest.attempts} ` +
            `unreadable_pair_files=${manifest.recovery.unreadable_pair_files}`
        );
    } else if (!options.resume && fs.existsSync(manifestPath)) {
        throw new Error(`run already exists; use --resume true or a new run name: ${runDir}`);
    }

    const existingFailureSummary = await summarizeFailureLog(failureLogPath);
    manifest.choice_rejections_recovered =
        existingFailureSummary.choice_rejections_recovered;
    manifest.choice_rejection_reasons =
        existingFailureSummary.choice_rejection_reasons;
    manifest.logged_attempt_failures = existingFailureSummary.logged_attempt_failures;
    manifest.logged_failure_reasons = existingFailureSummary.logged_failure_reasons;
    if (existingFailureSummary.highest_attempt >= manifest.attempts) {
        manifest.attempts = existingFailureSummary.highest_attempt + 1;
        manifest.invalid_attempts = Math.max(
            manifest.invalid_attempts,
            manifest.attempts - manifest.completed_pairs
        );
    }

    const initialElapsed = Number(manifest.elapsed_seconds) || 0;
    const startedAt = Date.now();
    const maximumAttempts = options.pairs * options.maxAttemptMultiplier;
    let nextAttempt = manifest.attempts;
    let writeLock = Promise.resolve();
    let resultsSinceCheckpoint = 0;

    async function recordResult(attempt, result) {
        const previous = writeLock;
        let release;
        writeLock = new Promise(resolve => { release = resolve; });
        await previous;
        try {
            manifest.attempts = Math.max(manifest.attempts, attempt + 1);
            for (const rejection of result.choiceRejections || []) {
                const reason = choiceRejectionReason(rejection.message);
                await appendJsonLineDurably(failureLogPath, {
                    type: 'choice_rejection_recovered',
                    attempt,
                    reason,
                    ...rejection,
                });
                ++manifest.choice_rejections_recovered;
                manifest.choice_rejection_reasons[reason] =
                    (manifest.choice_rejection_reasons[reason] || 0) + 1;
            }
            if (!result.valid) {
                await appendJsonLineDurably(failureLogPath, {
                    type: 'attempt_failure',
                    attempt,
                    reason: result.reason,
                    message: result.message || result.reason,
                });
                ++manifest.invalid_attempts;
                manifest.invalid_reasons[result.reason] =
                    (manifest.invalid_reasons[result.reason] || 0) + 1;
                ++manifest.logged_attempt_failures;
                manifest.logged_failure_reasons[result.reason] =
                    (manifest.logged_failure_reasons[result.reason] || 0) + 1;
            } else if (manifest.completed_pairs < options.pairs) {
                const filename = `pairs/pair_${String(attempt).padStart(6, '0')}.jsonl`;
                const pairPath = path.join(runDir, filename);
                const records = `${result.records.map(record => JSON.stringify(record)).join('\n')}\n`;
                await writeTextAtomic(pairPath, records);
                manifest.pairs.push({
                    pair_id: result.pairId,
                    attempt,
                    file: filename,
                    decision_index: result.decisionIndex,
                    rank0_return: result.rank0Return,
                    rank1_return: result.rank1Return,
                    rank0_choice: result.choice0,
                    rank1_choice: result.choice1,
                });
                ++manifest.completed_pairs;
                if (result.rank0Return > result.rank1Return) ++manifest.rank0_better;
                else if (result.rank1Return > result.rank0Return) ++manifest.rank1_better;
                else ++manifest.equal_returns;
            }
            const elapsed = initialElapsed + (Date.now() - startedAt) / 1000;
            manifest.elapsed_seconds = elapsed;
            if (++resultsSinceCheckpoint >= options.manifestCheckpointResults) {
                await writeJsonAtomic(manifestPath, manifest);
                resultsSinceCheckpoint = 0;
            }
            const newPairs = Math.max(1, manifest.completed_pairs);
            const remaining = Math.max(0, options.pairs - manifest.completed_pairs);
            const eta = elapsed * remaining / newPairs;
            console.log(
                `[counterfactual] pairs=${manifest.completed_pairs}/${options.pairs} ` +
                `attempts=${manifest.attempts}/${maximumAttempts} invalid=${manifest.invalid_attempts} ` +
                `elapsed=${formatDuration(elapsed)} eta=${formatDuration(eta)}`
            );
        } finally {
            release();
        }
    }

    async function worker() {
        while (manifest.completed_pairs < options.pairs) {
            const attempt = nextAttempt++;
            if (attempt >= maximumAttempts) return;
            try {
                await recordResult(attempt, await runPair(sim, options, attempt));
            } catch (error) {
                const reason = battleFailureReason(error);
                await recordResult(attempt, {
                    valid: false,
                    reason,
                    message: String(error.stack || error.message || error),
                });
                console.error(`[counterfactual] attempt=${attempt} failed: ${error.message}`);
            }
        }
    }

    await Promise.all(Array.from({length: options.concurrency}, () => worker()));
    await writeLock;
    const finalFailureSummary = await summarizeFailureLog(failureLogPath);
    manifest.choice_rejections_recovered = finalFailureSummary.choice_rejections_recovered;
    manifest.choice_rejection_reasons = finalFailureSummary.choice_rejection_reasons;
    manifest.logged_attempt_failures = finalFailureSummary.logged_attempt_failures;
    manifest.logged_failure_reasons = finalFailureSummary.logged_failure_reasons;
    manifest.status = manifest.completed_pairs >= options.pairs ? 'completed' : 'insufficient_valid_pairs';
    const batchPath = await combinePairFiles(runDir, manifest);
    const summary = {
        ...manifest,
        batch_path: batchPath,
        failure_log_path: failureLogPath,
        return_disagreement_pairs: manifest.rank0_better + manifest.rank1_better,
        return_disagreement_rate: manifest.completed_pairs > 0
            ? (manifest.rank0_better + manifest.rank1_better) / manifest.completed_pairs
            : 0,
        elapsed_seconds: initialElapsed + (Date.now() - startedAt) / 1000,
    };
    await writeJsonAtomic(manifestPath, manifest);
    await writeJsonAtomic(summaryPath, summary);
    console.log(`[counterfactual] status=${manifest.status} batch=${batchPath} summary=${summaryPath}`);
    if (manifest.status !== 'completed') process.exitCode = 1;
}

async function main(argv = process.argv.slice(2)) {
    const options = resolveOptions(argv);
    const sim = require(path.join(SHOWDOWN_DIR, 'dist', 'sim'));
    const runDir = path.join(REPO_ROOT, 'matches', 'runs', options.runName);
    const pairDir = path.join(runDir, 'pairs');
    const manifestPath = path.join(runDir, `${options.runName}_counterfactual_manifest.json`);
    const summaryPath = path.join(runDir, `${options.runName}_counterfactual_summary.json`);
    const failureLogPath = path.join(runDir, 'counterfactual_failures.jsonl');
    await fsp.mkdir(runDir, {recursive: true});
    const runLock = await acquireRunLock(runDir);
    try {
        await fsp.mkdir(pairDir, {recursive: true});
        await collect(
            sim, options, runDir, pairDir, manifestPath, summaryPath, failureLogPath);
    } finally {
        await runLock.release();
    }
}

if (require.main === module) {
    main().catch(error => {
        console.error(`[counterfactual] ${error.stack || error.message}`);
        process.exitCode = 1;
    });
}

module.exports = {
    appendJsonLineDurably,
    battleFailureReason,
    buildCounterfactualSample,
    choicesDiffer,
    choiceRejectionReason,
    drivePlayer,
    equalPrefix,
    acquireRunLock,
    inferenceSeed,
    mix32,
    pairEntryFromRecords,
    parseToml,
    readJsonIfValid,
    recoverManifestFromPairFiles,
    seedTuple,
    summarizeFailureLog,
    writeJsonAtomic,
};
