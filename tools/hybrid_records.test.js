'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {detachDetails, loadDetails} = require('./hybrid_records');
const {recordHash, validateLiveResume} = require('./hybrid_benchmark');

test('bounded detail publication preserves diagnostics without retaining worlds in the summary', async t => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'porygon-hybrid-records-'));
    assert(directory.startsWith(path.resolve(os.tmpdir()) + path.sep));
    t.after(() => fs.rm(directory, {recursive: true, force: true}));
    const game = {profile: 'guarded', opponent: 'random', index: 0, status: 'completed',
        result: {result: 'win'}, choice_rejections: [], first_belief_failure: {history: 'sampled world'},
        hybrid_coverage: {decisions: 1, belief_loss_count: 1, belief_losses: [{boundary: 2, reason: 'missing_move'}]},
        battle_analysis: {records: [{turn: 2}], omitted: 0}};
    const original = structuredClone(game);
    const summary = await detachDetails(game, directory, 0);
    assert.equal(summary.first_belief_failure, undefined); assert.equal(summary.battle_analysis, undefined);
    assert.equal(summary.hybrid_coverage.belief_losses, undefined); assert.equal(summary.hybrid_coverage.belief_loss_count, 1);
    const detailed = await loadDetails(summary, directory);
    assert.deepEqual(detailed.first_belief_failure, game.first_belief_failure);
    assert.deepEqual(detailed.battle_analysis, game.battle_analysis); assert.deepEqual(game, original);
    assert.deepEqual(detailed.hybrid_coverage.belief_losses, game.hybrid_coverage.belief_losses);
    assert.deepEqual(await detachDetails(game, directory, 0), summary);
    const other = await detachDetails({...game, result: {result: 'loss'}, battle_analysis: {records: [], omitted: 1}}, directory, 0);
    assert.notEqual(other.details.path, summary.details.path);
    assert.deepEqual((await loadDetails(summary, directory)).battle_analysis, game.battle_analysis);
    summary.record_hash = recordHash(summary);
    validateLiveResume({identity: {}, games: [summary]}, {}, [{profile: 'guarded', opponent: 'random', index: 0}]);
    await assert.rejects(loadDetails({...summary, details: {...summary.details, path: '../outside.json'}}, directory), /pointer/);
    await fs.appendFile(path.join(directory, summary.details.path), '\n');
    await assert.rejects(loadDetails(summary, directory), /integrity/);
});

test('legacy inline details stay readable and oversized new records fail before publication', async () => {
    const inline = {first_belief_failure: {legacy: true}};
    assert.equal(await loadDetails(inline, '.'), inline);
    await assert.rejects(detachDetails({battle_analysis: {text: 'x'.repeat(2 * 1024 ** 2)}}, '.', 0), /budget/);
});
