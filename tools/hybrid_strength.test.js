'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const {benchmarkAgents, assessBenchmark, validateBenchmark, compareProfiles, reuseBaseline,
    schedule, recordHash, validateLiveResume} = require('./hybrid_benchmark');

const CONFIG = require('../config/experimental/hybrid_guarded_benchmark.json');

test('confirmed opponent uses the same checkpoint sidecar and inference restrictions on both sides', () => {
    for (const side of [0, 1]) {
        const agents = benchmarkAgents({...CONFIG, tactical_teacher: false}, {inference: [1, 2]}, side, 'confirmed', null);
        assert.equal(agents[0].checkpoint, agents[1].checkpoint);
        assert.equal(agents[0].checkpoint, path.resolve(CONFIG.candidate_checkpoint));
        for (const agent of agents) {
            assert.equal(agent.targetOptions.sidecar, path.resolve(CONFIG.target_sidecar));
            assert.equal(agent.targetOptions.contextual, true); assert.equal(agent.targetOptions.slotLocal, true);
            assert.equal(agent.targetOptions.decoder, 'argmax'); assert.equal(agent.ompThreads, 1);
        }
    }
});

test('tiny matched screens cannot authorize promotion or claim a strength gain', () => {
    validateBenchmark(structuredClone(CONFIG));
    const config = {...CONFIG, games_per_side: 1};
    const games = config.profiles.flatMap(profile => config.opponents.flatMap(opponent => [0, 1].map(index =>
        ({profile: profile.name, opponent, index, teacher_side: index ? 'b' : 'a', status: 'completed',
            policy_mode: profile.baseline ? 'neural' : 'hybrid', elapsed_seconds: 1, choice_rejections: [],
            result: {result: profile.baseline ? 'loss' : 'win'}, hybrid_coverage: {interventions: 1}}))));
    const result = assessBenchmark(games, config).profiles.guarded;
    assert.equal(result.status, 'insufficient_strength_evidence'); assert.equal(result.promotion_authorized, false);
    assert.equal(result.strength_claim, false);
    assert.equal(result.baseline_comparison.random.exploratory_cluster_interval, null);
    assert.equal(compareProfiles(games, config).guarded.results.random.confidence_95, null);
    for (const game of games) game.hybrid_coverage.interventions = 0;
    assert.equal(assessBenchmark(games, config).profiles.guarded.status, 'no_policy_interventions');
    games.at(-1).choice_rejections.push({message: 'illegal move'});
    assert.equal(assessBenchmark(games, config).profiles.guarded.status, 'invalid_games_require_review');
});

test('fixed policy restart reuses only verified native baseline games and retains version provenance', async () => {
    const config = {...CONFIG, games_per_side: 1}, jobs = schedule(config);
    const oldIdentity = {config, executable: 'same', sources: {'hybrid_benchmark.js': 'old',
        'planner_joint_constraints.js': 'old', 'counterfactual_rollout.js': 'unchanged'}};
    const identity = {...structuredClone(oldIdentity), sources: {...oldIdentity.sources,
        'hybrid_benchmark.js': 'fixed', 'planner_joint_constraints.js': 'fixed'}};
    const games = jobs.slice(0, 6).map((job, i) => {
        const game = {...job, status: i === 5 ? 'failed' : 'completed', choice_rejections: [], result: {result: 'win'}};
        game.record_hash = recordHash(game); return game;
    });
    const source = {identity: oldIdentity, games, status: 'failed'}, original = structuredClone(source);
    const resumed = await reuseBaseline(source, identity, jobs, 'run_0372_source', '.', '.');
    assert.equal(resumed.games.length, 4); assert.equal(resumed.resumed_from.guarded_games_not_reused, 1);
    assert.equal(resumed.resumed_from.retained_failed_source_games, 1);
    for (const [i, game] of resumed.games.entries()) {
        assert.equal(game.reused_from.record_hash, source.games[i].record_hash);
        assert.equal(game.result.result, source.games[i].result.result); assert.equal(game.record_hash, recordHash(game));
    }
    validateLiveResume(resumed, identity, jobs); assert.deepEqual(source, original);
    const changedNative = {...identity, sources: {...identity.sources, 'counterfactual_rollout.js': 'changed'}};
    await assert.rejects(reuseBaseline(source, changedNative, jobs, 'run_0372_source', '.', '.'), /outside/);
    await assert.rejects(reuseBaseline(source, {...identity, executable: 'changed'}, jobs, 'run_0372_source', '.', '.'), /identical/);
    const corrupt = structuredClone(source); corrupt.games.at(-1).error = 'changed after publication';
    await assert.rejects(reuseBaseline(corrupt, identity, jobs, 'run_0372_source', '.', '.'), /integrity/);
});
