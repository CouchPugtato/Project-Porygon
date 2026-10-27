'use strict';
const assert = require('node:assert/strict');
const {isValid, nextGame, summarize, validateConfig, validateResume} = require('./tactical_teacher_benchmark');
const {battleSeeds} = require('./tactical_teacher_smoke');

const POLICIES = ['control', 'whole_turn', 'local'];
const CONTRASTS = [['local', 'whole_turn'], ['local', 'control'], ['whole_turn', 'control']];
const score = game => game.result.result === 'win' ? 1 : game.result.result === 'draw' ? .5 : 0;

function pairedDifference(a, b) {
    assert.equal(a.length, b.length);
    const n = a.length;
    if (!n) return {complete_seed_pairs: 0, matched_games_per_policy: 0, difference_pp: null, approximate_95_pp: null};
    const differences = a.map((v, i) => v - b[i]);
    const mean = differences.reduce((s, v) => s + v, 0) / n;
    const variance = n > 1 ? differences.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : null;
    const radius = variance === null ? null : 1.959963984540054 * Math.sqrt(variance / n);
    return {complete_seed_pairs: n, matched_games_per_policy: 2 * n,
        score_rate_a: a.reduce((s, v) => s + v, 0) / n, score_rate_b: b.reduce((s, v) => s + v, 0) / n,
        difference_pp: mean * 100, approximate_95_pp: radius === null ? null : [(mean - radius) * 100, (mean + radius) * 100],
        positive_pairs: differences.filter(v => v > 0).length, negative_pairs: differences.filter(v => v < 0).length,
        tied_pairs: differences.filter(v => v === 0).length};
}

function compareMatchedGames(games, seed, opponent) {
    const indexed = {}, pairs = new Set();
    for (const name of POLICIES) {
        indexed[name] = new Map();
        for (const game of games[name].filter(g => g.opponent === opponent)) {
            if (!Number.isSafeInteger(game.index) || game.index < 0 || indexed[name].has(game.index) ||
                    game.teacher_side !== (game.index % 2 ? 'b' : 'a')) throw new Error('duplicate or mislocated matched game');
            if (isValid(game)) assert.deepEqual(game.seeds, battleSeeds(seed, game.index, true), 'matched game seeds differ');
            indexed[name].set(game.index, game); pairs.add(Math.floor(game.index / 2));
        }
    }
    const means = Object.fromEntries(POLICIES.map(name => [name, []]));
    const included = [], excluded = [];
    for (const pair of [...pairs].sort((a, b) => a - b)) {
        const missing = [];
        for (const name of POLICIES) for (const offset of [0, 1]) {
            const game = indexed[name].get(pair * 2 + offset);
            if (!game || !isValid(game)) missing.push({policy: name, side: offset ? 'b' : 'a',
                reason: !game ? 'missing_attempt' : game.status !== 'completed' ? 'failed_attempt' : 'choice_rejection'});
        }
        if (missing.length) { excluded.push({pair_index: pair, unavailable: missing}); continue; }
        included.push(pair);
        for (const name of POLICIES) means[name].push((score(indexed[name].get(pair * 2)) + score(indexed[name].get(pair * 2 + 1))) / 2);
    }
    return {complete_seed_pairs: included.length, included_pair_indices: included, excluded_seed_pairs: excluded,
        contrasts: Object.fromEntries(CONTRASTS.map(([a, b]) => [`${a}_minus_${b}`, pairedDifference(means[a], means[b])]))};
}

function compareScreens(runs) {
    const common = runs.control.identity;
    const baseConfig = c => {
        const {target_sidecar, context_target_evaluation, target_decoder, target_slot_local, ...base} = c;
        return base;
    };
    assert(!common.config.target_sidecar && !common.config.target_slot_local, 'control must use unchanged student targets');
    const results = {}, games = {};
    for (const name of POLICIES) {
        const run = runs[name], config = run.identity.config;
        validateConfig(config);
        if (run.status !== 'completed' || config.target_audit || config.collect_demonstrations || config.collect_target_labels)
            throw new Error('comparison requires completed matched strength screens, not collection/audit runs');
        assert.deepEqual(baseConfig(config), baseConfig(common.config), 'non-target screen settings differ');
        for (const field of ['checkpoint_sha256', 'candidate_checkpoint_sha256', 'executable_sha256', 'source_sha256']) {
            assert(common[field], `missing screen identity: ${field}`);
            assert.deepEqual(run.identity[field], common[field], `frozen screen identity differs: ${field}`);
        }
        if (name !== 'control') {
            assert(config.context_target_evaluation && config.target_decoder === 'argmax', 'learned screens must use contextual argmax');
            assert.equal(!!config.target_slot_local, name === 'local');
            assert.equal(config.target_sidecar, runs.whole_turn.identity.config.target_sidecar);
            assert(run.identity.target_sidecar, 'missing context sidecar identity');
            assert.deepEqual(run.identity.target_sidecar, runs.whole_turn.identity.target_sidecar, 'context sidecars differ');
        }
        games[name] = validateResume(run, run.identity, config);
        assert.equal(nextGame(config, games[name]), null, 'screen has not reached its valid-game quotas');
        results[name] = {run_name: run.run_name, opponents: summarize(games[name], config.opponents),
            excluded: games[name].filter(g => !isValid(g)).map(g => ({opponent: g.opponent, index: g.index,
                side: g.teacher_side, status: g.status, error: g.error || g.cleanup_error || null,
                rejections: g.choice_rejections || []})),
            recorded_elapsed_seconds: games[name].reduce((s, g) => s + g.duration_seconds, 0)};
        for (const opponent of config.opponents) {
            const records = games[name].filter(g => g.opponent === opponent && isValid(g));
            const decisions = records.reduce((s, g) => s + (g.learned_targets?.target_decisions || 0), 0);
            const applied = records.reduce((s, g) => s + (g.learned_targets?.learned_target_decisions || 0), 0);
            results[name].opponents[opponent].targeting = config.target_sidecar ?
                {decisions, applied, rate: decisions ? applied / decisions : null,
                    definition: 'Accepted decisions using learned targets, not necessarily changes from default targets.'} : null;
        }
    }
    return {diagnostic: 'slot_local_target_strength_comparison', development_only: true, automatic_promotion: false,
        seed: common.config.seed, primary_contrast: 'local_minus_whole_turn',
        interval_method: 'Nominal approximate normal 95% intervals over side-swapped seed-pair mean differences. All contrasts use the same complete common-valid pairs across all three policies, separately per opponent. Draws count half; no multiplicity adjustment.',
        total_valid_games: POLICIES.reduce((s, n) => s + games[n].filter(isValid).length, 0),
        total_excluded_attempts: POLICIES.reduce((s, n) => s + results[n].excluded.length, 0),
        recorded_elapsed_seconds: POLICIES.reduce((s, n) => s + results[n].recorded_elapsed_seconds, 0),
        timing_note: 'Recorded durations include launch-gate waits but are not end-to-end wall time.',
        results, comparisons: Object.fromEntries(common.config.opponents.map(opponent =>
            [opponent, compareMatchedGames(games, common.config.seed, opponent)])),
        provenance: Object.fromEntries(POLICIES.map(name => [name, runs[name].identity]))};
}

if (require.main === module) {
    const fs = require('node:fs'), crypto = require('node:crypto');
    if (process.argv.length !== 5) throw new Error('usage: node tools/target_comparison.js CONTROL_SUMMARY WHOLE_TURN_SUMMARY LOCAL_SUMMARY');
    const artifacts = {}, runs = {};
    for (const [i, name] of POLICIES.entries()) {
        const source = process.argv[i + 2], bytes = fs.readFileSync(source);
        runs[name] = JSON.parse(bytes);
        artifacts[source] = crypto.createHash('sha256').update(bytes).digest('hex');
    }
    artifacts['tools/target_comparison.js'] = crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
    console.log(JSON.stringify({...compareScreens(runs), artifact_sha256: artifacts}, null, 2));
}

module.exports = {compareMatchedGames, compareScreens, pairedDifference};
