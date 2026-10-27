'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const {battleSeeds} = require('./tactical_teacher_smoke');
const {compareMatchedGames, compareScreens, pairedDifference} = require('./target_comparison');
const {nextGame} = require('./tactical_teacher_benchmark');
const {targetPolicyTag} = require('./target_context');

function game(index, result = 'win') {
    return {index, opponent: 'parent', teacher_side: index % 2 ? 'b' : 'a',
        status: 'completed', result: {result}, choice_rejections: [], seeds: battleSeeds(30501, index, true)};
}

test('matched contrasts use seed pairs rather than counting swapped sides as independent', () => {
    const games = {control: [game(0, 'loss'), game(1, 'loss'), game(2), game(3)],
        whole_turn: [game(0, 'loss'), game(1, 'loss'), game(2, 'loss'), game(3, 'loss')],
        local: [game(0), game(1), game(2, 'loss'), game(3, 'loss')]};
    const s = compareMatchedGames(games, 30501, 'parent');
    assert.equal(s.complete_seed_pairs, 2);
    assert.equal(s.contrasts.local_minus_whole_turn.difference_pp, 50);
    assert.equal(s.contrasts.local_minus_whole_turn.matched_games_per_policy, 4);
    const r = s.contrasts.local_minus_control;
    assert.equal(r.difference_pp, 0); assert.equal(r.positive_pairs, 1); assert.equal(r.negative_pairs, 1);
    assert.ok(r.approximate_95_pp[0] < -100 && r.approximate_95_pp[1] > 100,
        'small-sample normal intervals are not silently presented as exact confidence bounds');
});

test('one rejected or missing side removes the entire seed pair from every contrast', () => {
    const games = Object.fromEntries(['control', 'whole_turn', 'local'].map(n => [n, [game(0), game(1), game(2), game(3)]]));
    games.control[0].choice_rejections.push({message: 'trapped'});
    let s = compareMatchedGames(games, 30501, 'parent');
    assert.deepEqual(s.included_pair_indices, [1]);
    assert.equal(s.excluded_seed_pairs[0].unavailable[0].reason, 'choice_rejection');
    assert.equal(s.contrasts.local_minus_whole_turn.complete_seed_pairs, 1);
    assert.equal(s.contrasts.local_minus_whole_turn.approximate_95_pp, null);
    games.local.pop();
    s = compareMatchedGames(games, 30501, 'parent');
    assert.equal(s.complete_seed_pairs, 0); assert.equal(s.contrasts.local_minus_control.difference_pp, null);
});

test('changed seeds, duplicated attempts and side mismatches cannot enter a matched comparison', () => {
    const games = Object.fromEntries(['control', 'whole_turn', 'local'].map(n => [n, [game(0), game(1)]]));
    games.local[0].seeds = battleSeeds(30502, 0, true);
    assert.throws(() => compareMatchedGames(games, 30501, 'parent'), /seeds differ/);
    games.local[0] = game(0); games.local.push(game(0));
    assert.throws(() => compareMatchedGames(games, 30501, 'parent'), /duplicate/);
    games.local.pop(); games.local[0].teacher_side = 'b';
    assert.throws(() => compareMatchedGames(games, 30501, 'parent'), /mislocated/);
});

test('paired score differences retain draws and do not invent uncertainty for one pair', () => {
    assert.equal(pairedDifference([.75], [.5]).difference_pp, 25);
    assert.equal(pairedDifference([], []).approximate_95_pp, null);
    assert.equal(pairedDifference([0], [0]).approximate_95_pp, null);
    assert.throws(() => pairedDifference([0], []));
});

function screens() {
    const runs = {};
    for (const name of ['control', 'whole_turn', 'local']) {
        const config = {...require(`../config/experimental/target_slot_local_${name}.json`), games_per_side: 2, max_attempts_per_side: 3};
        const identity = {config, checkpoint_sha256: 'parent', candidate_checkpoint_sha256: 'student',
            executable_sha256: 'runtime', source_sha256: {runner: 'frozen'}};
        if (name !== 'control') identity.target_sidecar = {sha256: 'frozen-sidecar'};
        const games = [];
        for (let next; (next = nextGame(config, games));) {
            const record = {...game(next.index), ...next, duration_seconds: 1, policy_mode: 'neural', coverage: null};
            if (name !== 'control') {
                record.policy_tag = targetPolicyTag(config);
                record.learned_targets = {training_eligible: false, target_decoder: 'argmax',
                    target_slot_local: name === 'local', target_decisions: 10, learned_target_decisions: 4};
            }
            games.push(record);
        }
        runs[name] = {status: 'completed', run_name: name, identity, games};
    }
    return runs;
}

test('screen summary checks frozen provenance and complete quotas without changing raw records', () => {
    const runs = screens(), before = JSON.stringify(runs), result = compareScreens(runs);
    assert.equal(result.total_valid_games, 24); assert.equal(result.total_excluded_attempts, 0);
    assert.equal(result.comparisons.parent.complete_seed_pairs, 2);
    assert.equal(result.results.local.opponents.parent.targeting.decisions, 40);
    assert.equal(JSON.stringify(runs), before);
    for (const field of ['candidate_checkpoint_sha256', 'executable_sha256', 'source_sha256']) {
        const changed = structuredClone(runs); changed.local.identity[field] = 'changed';
        assert.throws(() => compareScreens(changed), /identity differs/);
    }
    const missing = structuredClone(runs); missing.local.games.pop();
    assert.throws(() => compareScreens(missing), /quotas/);
    const drifted = structuredClone(runs); drifted.whole_turn.identity.config.seed++;
    assert.throws(() => compareScreens(drifted), /settings differ/);
});
