'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {hashFile} = require('./tactical_teacher_smoke');
const {prepareDataset, validationHash} = require('./prepare_tactical_dataset');

async function fixture(t) {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'teacher-dataset-'));
    t.after(() => fs.rm(root, {recursive: true, force: true}));
    const collections = [];
    for (const [source, ids] of [['pilot', ['battle-4', 'battle-alpha']], ['extra', ['battle-beta']]]) {
        const directory = path.join(root, source);
        await fs.mkdir(directory);
        const games = [];
        for (const [index, battle_id] of ids.entries()) {
            const result = index ? 'loss' : 'win';
            const file = `teacher_parent_${String(index).padStart(4, '0')}_raw.jsonl`;
            const content = [
                {type: 'battle_start', is_doubles: true}, {type: 'request', request_id: 1, payload: {}},
                {type: 'decision_accepted', request_id: 1, action: 0, action2: -1, command: '/choose move 1 1'},
                {type: 'terminal', result}, {type: 'battle_end'},
            ].map(message => JSON.stringify({...message, battle_id})).join('\n') + '\n';
            await fs.writeFile(path.join(directory, file), content);
            games.push({index, opponent: 'parent', status: 'completed', choice_rejections: [], result: {result},
                demonstration: {file, battle_id, result, decision_turns: 1, sha256: await hashFile(path.join(directory, file)),
                    policy_tag: 'tactical-eval-v1', purpose: 'supervised_imitation', ppo_eligible: false}});
        }
        const summary = path.join(directory, 'summary.json');
        await fs.writeFile(summary, JSON.stringify({status: 'completed', valid_games: games.length,
            demonstration_count: games.length, identity: {config: {collect_demonstrations: true},
                checkpoint_sha256: 'teacher-fixture', executable_sha256: 'runtime-fixture'}, games}));
        collections.push(summary);
    }
    const config = {collections, expected_battles: 3, expected_decision_turns: 3,
        battles_per_shard: 2, max_shard_bytes: 4096, validation_seed: 1337};
    return {root, config};
}

test('split hash matches the native/Python stable-hash fixture', () => {
    assert.equal(validationHash('battle-4', 1337), 5406646322143240280n);
    assert.equal(validationHash('battle-4', 1337) % 10n, 0n);
    assert.notEqual(validationHash('battle-alpha', 1337) % 10n, 0n);
});

test('packing preserves all battles and exposes exactly the same holdout independently of shards', async t => {
    const {root, config} = await fixture(t);
    const manifest = await prepareDataset(config, 'dataset', root);
    assert.equal(manifest.status, 'completed');
    assert.equal(manifest.counts.battles, 3);
    assert.equal(manifest.shards.length, 2);
    assert.ok(manifest.shards.every(shard => shard.battles <= 2 && shard.bytes <= 4096));
    const ids = manifest.shards.flatMap(shard => shard.battle_ids);
    assert.equal(new Set(ids).size, 3);
    const holdout = manifest.holdout_shards.flatMap(shard => shard.battle_ids).sort();
    assert.deepEqual(holdout, manifest.battles.filter(battle => battle.split === 'holdout').map(battle => battle.battle_id).sort());
    assert.ok(holdout.includes('battle-4'));
    assert.equal(manifest.counts.holdout_decision_turns, holdout.length);
    assert.ok(manifest.battles.some(battle => battle.result === 'loss'));
    const resumed = await prepareDataset(config, 'dataset', root);
    assert.deepEqual(resumed, manifest);
    const changed = path.join(root, 'matches/runs/dataset', manifest.shards[0].file);
    await fs.appendFile(changed, 'corrupt');
    await assert.rejects(prepareDataset(config, 'dataset', root), /changed/);
});

test('a byte cap makes smaller shards without splitting battles', async t => {
    const {root, config} = await fixture(t);
    const manifest = await prepareDataset({...config, max_shard_bytes: 600}, 'small', root);
    assert.equal(manifest.shards.length, 3);
    assert.ok(manifest.shards.every(shard => shard.battles === 1 && shard.bytes <= 600));
});

test('dataset refuses incompatible teachers, changed inputs, and unrelated output files', async t => {
    const {root, config} = await fixture(t);
    const summary = JSON.parse(await fs.readFile(config.collections[1], 'utf8'));
    summary.identity.executable_sha256 = 'different';
    await fs.writeFile(config.collections[1], JSON.stringify(summary));
    await assert.rejects(prepareDataset(config, 'incompatible', root), /different teacher/);
    summary.identity.executable_sha256 = 'runtime-fixture';
    await fs.writeFile(config.collections[1], JSON.stringify(summary));
    await prepareDataset(config, 'dataset', root);
    await assert.rejects(prepareDataset({...config, validation_seed: 1338}, 'dataset', root), /inputs changed/);
    await fs.mkdir(path.join(root, 'matches/runs/occupied'));
    await fs.writeFile(path.join(root, 'matches/runs/occupied/user-file'), 'keep');
    await assert.rejects(prepareDataset(config, 'occupied', root), /unrelated/);
    assert.equal(await fs.readFile(path.join(root, 'matches/runs/occupied/user-file'), 'utf8'), 'keep');
});

test('dataset rejects duplicate battle IDs, wrong totals, and oversized battles', async t => {
    const {root, config} = await fixture(t);
    await assert.rejects(prepareDataset({...config, collections: [config.collections[0], config.collections[0]]}, 'duplicates', root), /duplicate/);
    await assert.rejects(prepareDataset({...config, expected_battles: 4}, 'wrong-total', root), /totals/);
    await assert.rejects(prepareDataset({...config, max_shard_bytes: 10}, 'oversized', root), /byte limit/);
});
