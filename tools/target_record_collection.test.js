'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const {EventEmitter} = require('node:events');
const {drivePlayer} = require('./counterfactual_rollout');
const {buildArtifact} = require('./target_records');
const {BundleRecordingAgent, validateConfig, verifyGame, collectGames} = require('./target_record_collection');
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const config = {...require('../config/experimental/target_record_live_smoke.json'), games: 2};
const bundle = {sha256: config.manifest_sha256, files: {actor: 'unused', sidecar: 'unused'},
    manifest: {kind: 'frozen-target-action-actor-v1', sidecar_parameters: '1'.repeat(16),
        inference_seed: 31701, target_decoder: 'argmax', eligibility: 'slot-local-v1'}};
const identity = {config, manifest_sha256: bundle.sha256};
const policy = {version: 2, kind: bundle.manifest.kind, parameter_fingerprint_algorithm: 'fnv1a64-native',
    reference_parameters: '2'.repeat(16), actor_parameters: '2'.repeat(16),
    sidecar_parameters: bundle.manifest.sidecar_parameters, manifest_sha256: bundle.sha256,
    context_version: 1, action_decoder: 'sample', target_decoder: 'argmax', eligibility: 'slot-local-v1'};
const action = {type: 'action', action: 8, action2: 23, command: '/choose switch 1, switch 2'};
const ready = {type: 'ready', accepted_records: true, capabilities: {training: false}, behavior_policy: policy};
function record(battle_id, request_id) {
    return {type: 'bundle_decision', version: 1, battle_id, request_id, decision_index: 1,
        training_eligible: false, behavior_policy: policy, action: 8, action2: 23, command: action.command,
        hidden_dim: 1, hidden_f32_hex: '80000000', legal_mask: Array(28).fill(1),
        choice: [[1, 2, 0, 0, 0, 0, 0], [1, 2, 0, 1, 0, 0, 0]], public_context: [null, null],
        behavior: {version: 2, scope: 'full_turn_diagnostic', ppo_ready: false,
            full_turn_log_probability: -2, likelihood_status: 'supported', slots: [null, null]}};
}
function terminal(battle_id) {
    return {type: 'episode_complete', battle_id, training_eligible: false, behavior_policy: policy,
        target_decisions: 1, likelihood_replay: {accepted_supported: 1, accepted_unsupported: 0}};
}
function battle(index) {
    const battle_id = `tactical-random-${config.seed}-${index}`;
    const inputs = [{type: 'battle_start', battle_id}, {type: 'request', battle_id, request_id: 1, payload: {}},
        {type: 'decision_accepted', battle_id, request_id: 1, action: 8, action2: 23, command: action.command},
        {type: 'terminal', battle_id, result: 'win', reward: 1}, {type: 'battle_end', battle_id}];
    const artifact = buildArtifact(bundle, inputs, [ready, record(battle_id, 1), terminal(battle_id)]);
    const body = {format: 'porygon-live-bundle-battle-v1', training_eligible: false, manifest_sha256: bundle.sha256,
        game: {index, status: 'completed', teacher_side: index % 2 ? 'b' : 'a', bundle_inference_seed: 31701,
            result: {result: 'win', reward: 1}, choice_rejections: []}, artifact};
    return {...body, content_sha256: hash(body)};
}
async function directory(t) {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-live-record-test-'));
    t.after(async () => { for (const file of await fs.readdir(dir)) await fs.unlink(path.join(dir, file)); await fs.rmdir(dir); });
    return dir;
}
const gate = {async waitForLaunch() {}};
const replay = async artifact => { assert.equal(artifact.records.length, 1); return {likelihood_ratio: 1}; };

test('live smoke remains capped at six serial games with an explicit bundle pin', () => {
    validateConfig(config);
    for (const change of [{games: 100}, {manifest_sha256: ''}, {battle_timeout_seconds: 3600}, {extra: true}])
        assert.throws(() => validateConfig({...config, ...change}), /smoke/);
});

test('a delayed launch after battle timeout is killed and staging is cleaned after child close', async () => {
    const agent = new BundleRecordingAgent(bundle);
    let release, cleaned = 0, closed = false;
    agent.launchBundle = () => new Promise(resolve => { release = resolve; });
    const pending = agent.launch();
    await agent.close();
    const child = new EventEmitter(); child.stdin = new EventEmitter();
    child.kill = () => { queueMicrotask(() => { closed = true; child.emit('close', 1); }); };
    release({child, cleanup: async () => { assert.equal(closed, true); cleaned++; }});
    await assert.rejects(pending, /closed while waiting/);
    assert.equal(cleaned, 1); assert.equal(agent.process, null);
});

test('live bridge captures exact inputs and excludes a rejected retry from accepted snapshots', async () => {
    const agent = new BundleRecordingAgent(bundle), sent = [];
    agent.pushMessage(ready);
    agent.process = {stdin: {writable: true, write(line) {
        sent.push(line);
        const message = JSON.parse(line);
        if (message.type === 'request') agent.pushMessage(action);
        if (message.type === 'decision_accepted') agent.pushMessage(record(message.battle_id, message.request_id));
        if (message.type === 'terminal') agent.pushMessage(terminal(message.battle_id));
    }}};
    const stream = {async *[Symbol.asyncIterator]() {
        yield '|request|{"active":[{},{}]}\n';
        yield '|error|[Invalid choice] Can\'t switch: The active Pokémon is trapped\n|request|{"active":[{"trapped":true},{}]}\n';
        yield '|move|p1a: A|Tackle|p2a: B\n|win|Candidate';
    }, async write() {}};
    const result = await drivePlayer(stream, agent, 'live-rejection', 'Candidate', config.format);
    assert.deepEqual(agent.inputs, sent.map(line => JSON.parse(line)));
    const artifact = buildArtifact(bundle, agent.inputs, agent.outputs);
    assert.equal(result.choiceRejections.length, 1);
    assert.equal(artifact.records.length, 1);
    assert.notEqual(artifact.records[0].request_id, artifact.inputs.find(x => x.type === 'decision_rejected').request_id);
    assert.equal(artifact.training_eligible, false);
    agent.process = null;
});

test('completed battles replay on resume without simulation or rewriting, even after summary loss', async t => {
    const dir = await directory(t); let runs = 0, replays = 0;
    const options = {gate, play: async (_bundle, _config, index) => { runs++; return battle(index); },
        replay: async artifact => { replays++; return replay(artifact); }};
    await collectGames(config, bundle, identity, dir, options);
    const original = await fs.readFile(path.join(dir, 'battle_0000.json'), 'utf8');
    await fs.unlink(path.join(dir, 'summary.json'));
    await collectGames(config, bundle, identity, dir, options);
    assert.equal(runs, 2); assert.equal(replays, 4);
    assert.equal(await fs.readFile(path.join(dir, 'battle_0000.json'), 'utf8'), original);
    assert.equal(JSON.parse(await fs.readFile(path.join(dir, 'summary.json'))).completed, 2);
});

test('interrupted battles have no commit and restart at the same index, retaining earlier battles', async t => {
    const dir = await directory(t), seen = [];
    const play = async (_bundle, _config, index) => {
        seen.push(index);
        if (seen.length === 2) { const error = new Error('interrupted fixture'); error.capture = {inputs: ['partial']}; throw error; }
        return battle(index);
    };
    await assert.rejects(collectGames(config, bundle, identity, dir, {gate, play, replay}), /interrupted/);
    const failure = JSON.parse(await fs.readFile(path.join(dir, 'failure_0001.json')));
    assert.deepEqual(failure.capture.inputs, ['partial']);
    await fs.writeFile(path.join(dir, 'battle_0001.json.tmp'), 'incomplete');
    await collectGames(config, bundle, identity, dir, {gate, play, replay});
    assert.deepEqual(seen, [0, 1, 1]);
    assert.equal(JSON.parse(await fs.readFile(path.join(dir, 'summary.json'))).completed, 2);
});

test('changed bundle, altered completed files and replay failures cannot be credited or overwritten', async t => {
    const dir = await directory(t);
    await collectGames(config, bundle, identity, dir, {gate, play: async (_b, _c, i) => battle(i), replay});
    const filename = path.join(dir, 'battle_0000.json'), original = await fs.readFile(filename, 'utf8');
    const otherConfig = {...config, manifest_sha256: 'f'.repeat(64)}, otherBundle = {...bundle, sha256: 'f'.repeat(64)};
    await assert.rejects(collectGames(otherConfig, otherBundle, {config: otherConfig, manifest_sha256: otherBundle.sha256}, dir), /inputs changed/);
    const altered = JSON.parse(original); altered.game.choice_rejections.push({message: 'altered'});
    await assert.rejects(verifyGame(altered, bundle, config, 0, replay), /mismatch/);
    await assert.rejects(collectGames(config, bundle, identity, dir, {gate,
        play: async () => { throw Error('must not simulate'); }, replay: async () => { throw Error('replay divergence'); }}), /divergence/);
    assert.equal(await fs.readFile(filename, 'utf8'), original);
});
