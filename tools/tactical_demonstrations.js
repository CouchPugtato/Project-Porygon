'use strict';

const path = require('node:path');
const {hashFile} = require('./tactical_teacher_smoke');
const {writeTextAtomic} = require('./counterfactual_rollout');

function inspectReplay(lines) {
    const messages = lines.map(line => JSON.parse(line));
    const first = messages[0];
    if (first?.type !== 'battle_start' || messages.at(-1)?.type !== 'battle_end' ||
            messages.some(message => message.battle_id !== first.battle_id))
        throw new Error('demonstration must contain one complete teacher battle');
    const requests = new Map();
    const accepted = new Set();
    let terminal = null;
    for (const message of messages) {
        if (message.type === 'request') {
            if (requests.has(message.request_id)) throw new Error('duplicate demonstration request');
            requests.set(message.request_id, message.payload);
        } else if (message.type === 'decision_accepted') {
            if (!requests.has(message.request_id) || accepted.has(message.request_id) ||
                    !Number.isInteger(message.action) || !Number.isInteger(message.action2) || !message.command)
                throw new Error('accepted demonstration decision has no unique request or actions');
            accepted.add(message.request_id);
        } else if (message.type === 'terminal') {
            if (terminal || !['win', 'loss', 'draw'].includes(message.result))
                throw new Error('invalid demonstration terminal result');
            terminal = message.result;
        } else if (!['battle_start', 'event', 'battle_end'].includes(message.type)) {
            throw new Error(`unexpected demonstration record: ${message.type}`);
        }
    }
    if (!terminal || !accepted.size || accepted.size !== requests.size)
        throw new Error('demonstration has missing decisions or terminal result');
    return {battle_id: first.battle_id, decision_turns: accepted.size, result: terminal};
}

async function publishReplay(directory, opponent, index, lines, game) {
    const metadata = inspectReplay(lines);
    if (metadata.result !== game.result.result || metadata.decision_turns !== game.coverage.tactical_attempts)
        throw new Error('demonstration disagrees with teacher coverage or outcome');
    const filename = `teacher_${opponent}_${String(index).padStart(4, '0')}_raw.jsonl`;
    const target = path.join(directory, filename);
    await writeTextAtomic(target, lines.join(''));
    return {...metadata, file: filename, sha256: await hashFile(target), purpose: 'supervised_imitation',
        ppo_eligible: false, policy_tag: 'tactical-eval-v1'};
}

async function verifyReplays(directory, games, validGame) {
    for (const game of games) {
        if (!validGame(game)) {
            if (game.demonstration) throw new Error('invalid game references a training demonstration');
            continue;
        }
        const replay = game.demonstration;
        const expected = `teacher_${game.opponent}_${String(game.index).padStart(4, '0')}_raw.jsonl`;
        if (!replay || replay.file !== expected || replay.purpose !== 'supervised_imitation' || replay.ppo_eligible !== false ||
                await hashFile(path.join(directory, expected)) !== replay.sha256)
            throw new Error('saved teacher demonstration is missing, changed, or has inconsistent provenance');
    }
}

module.exports = {inspectReplay, publishReplay, verifyReplays};
