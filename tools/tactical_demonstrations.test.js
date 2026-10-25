'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const {AgentProcess, drivePlayer} = require('./counterfactual_rollout');
const {inspectReplay, publishReplay, verifyReplays} = require('./tactical_demonstrations');
const {isValid, validateConfig} = require('./tactical_teacher_benchmark');

function replay(result = 'loss') {
    return [
        {type: 'battle_start', format: 'gen9randomdoublesbattle', is_doubles: true},
        {type: 'event', seq: 0, line: '|message|A "quoted" message'},
        {type: 'request', request_id: 1, payload: {active: [{moves: []}]}},
        {type: 'decision_accepted', request_id: 1, action: 0, action2: -1, command: '/choose move 1 1'},
        {type: 'terminal', result, reward: result === 'win' ? 1 : -1},
        {type: 'battle_end'},
    ].map(message => `${JSON.stringify({...message, battle_id: 'teacher-test'})}\n`);
}

test('teacher process capture preserves exactly the JSON sent to the native runtime', () => {
    const captured = [], sent = [];
    const agent = new AgentProcess('unused', 1, null, 1, true, line => captured.push(line));
    agent.process = {stdin: {writable: true, write(line) { sent.push(line); }}};
    agent.send({type: 'event', battle_id: 'test', line: '|message|A "quoted" message'});
    assert.deepEqual(captured, sent);
    assert.equal(JSON.parse(captured[0]).line, '|message|A "quoted" message');
});

test('demonstrations retain losses and reject partial, rejected, duplicated, or PPO records', () => {
    assert.equal(inspectReplay(replay()).result, 'loss');
    assert.equal(inspectReplay(replay('win')).result, 'win');
    assert.throws(() => inspectReplay(replay().slice(0, -1)), /complete/);
    const rejected = replay();
    rejected[3] = rejected[3].replace('decision_accepted', 'decision_rejected');
    assert.throws(() => inspectReplay(rejected), /unexpected/);
    const duplicated = replay();
    duplicated.splice(4, 0, duplicated[3]);
    assert.throws(() => inspectReplay(duplicated), /unique/);
    const wrongBattle = replay();
    wrongBattle[2] = wrongBattle[2].replace('teacher-test', 'other-battle');
    assert.throws(() => inspectReplay(wrongBattle), /one complete/);
    const ppo = replay();
    ppo.splice(4, 0, JSON.stringify({type: 'episode_complete', battle_id: 'teacher-test', steps: []}));
    assert.throws(() => inspectReplay(ppo), /unexpected/);
});

test('existing player bridge produces a complete accepted-decision replay', async () => {
    const lines = [];
    const agent = {
        send(message) { lines.push(`${JSON.stringify(message)}\n`); },
        async nextMessage(type) {
            if (type === 'action') return {action: 0, action2: -1, command: '/choose move 1 1'};
            return {policy_tag: 'tactical-eval-v1'};
        },
    };
    const stream = {async *[Symbol.asyncIterator]() {
        yield '|request|{"active":[{"moves":[]}]}\n|move|p1a: A|Tackle|p2a: B\n|win|Teacher';
    }, async write() {}};
    await drivePlayer(stream, agent, 'bridge-test', 'Teacher', 'gen9randomdoublesbattle');
    assert.deepEqual(inspectReplay(lines), {battle_id: 'bridge-test', decision_turns: 1, result: 'win'});
});

test('atomic replay publication and resume reject missing or changed data', async t => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tactical-demo-'));
    t.after(() => fs.rm(directory, {recursive: true, force: true}));
    const game = {opponent: 'parent', index: 0, status: 'completed', choice_rejections: [],
        result: {result: 'loss'}, coverage: {tactical_attempts: 1}};
    game.demonstration = await publishReplay(directory, 'parent', 0, replay(), game);
    assert.equal(game.demonstration.ppo_eligible, false);
    await verifyReplays(directory, [game], isValid);
    const target = path.join(directory, game.demonstration.file);
    assert.equal(await fs.readFile(target, 'utf8'), replay().join(''));
    await fs.appendFile(target, 'corrupt');
    await assert.rejects(verifyReplays(directory, [game], isValid), /changed/);
    await assert.rejects(publishReplay(directory, 'parent', 0, replay('win'), game), /disagrees/);
    await assert.rejects(verifyReplays(directory, [{...game, status: 'failed'}], isValid), /invalid game/);
});

test('collection pilot stays small, serial, fresh-seeded, and explicitly opt-in', () => {
    const config = require('../config/experimental/tactical_teacher_collection.json');
    validateConfig(config);
    assert.equal(config.collect_demonstrations, true);
    assert.equal(4 * config.games_per_side, 200);
    assert.equal(config.agent_omp_threads, 1);
    assert.notEqual(config.seed, require('../config/experimental/tactical_teacher_confirmation.json').seed);
    assert.throws(() => validateConfig({...config, collect_demonstrations: 'true'}), /bounded/);
});
