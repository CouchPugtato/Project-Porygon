'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {battleSeeds, createBattleAgents, driveRandom} = require('./tactical_teacher_smoke');
const {isValid, loadSaved, nextGame, summarize, validateConfig, validateResume, wilson} = require('./tactical_teacher_benchmark');
const config = {...require('../config/experimental/tactical_teacher_benchmark.json'), games_per_side: 1, max_attempts_per_side: 2};

function record(next, overrides = {}) {
    return {...next, status: 'completed', duration_seconds: 1, result: {result: 'win'}, choice_rejections: [],
        coverage: {policy_tag: 'tactical-eval-v1', training_eligible: false, tactical_attempts: 2,
            tactical_applied: 1, tactical_forced_switches: 0, tactical_fallbacks: 1}, ...overrides};
}

test('paired seeds swap teams and preserve policy-role inference seeds', () => {
    const a = battleSeeds(26601, 0, true);
    const b = battleSeeds(26601, 1, true);
    assert.deepEqual(a.battle, b.battle);
    assert.deepEqual(a.players[0], b.players[1]);
    assert.deepEqual(a.players[1], b.players[0]);
    assert.equal(a.inference[0], b.inference[1]);
    assert.notDeepEqual(a.players, battleSeeds(26601, 2, true).players);
});

test('scheduler replaces invalid games without crediting either side', () => {
    const games = [];
    games.push(record(nextGame(config, games), {status: 'failed', error: 'timeout'}));
    assert.equal(nextGame(config, games).teacher_side, 'b');
    games.push(record(nextGame(config, games)));
    assert.deepEqual(nextGame(config, games), {opponent: 'parent', teacher_side: 'a', index: 2});
    games.push(record(nextGame(config, games)));
    assert.equal(nextGame(config, games).opponent, 'random');
    games.push(record(nextGame(config, games)));
    games.push(record(nextGame(config, games)));
    assert.equal(nextGame(config, games), null);
    const result = summarize(games);
    assert.equal(result.parent.invalid_games, 1);
    assert.equal(result.parent.valid_games, 2);
    assert.equal(result.random.valid_games, 2);
});

test('rejected choices invalidate an otherwise completed game', () => {
    const game = record({opponent: 'parent', teacher_side: 'a', index: 0},
        {choice_rejections: [{message: 'trapped'}]});
    assert.equal(isValid(game), false);
    assert.equal(summarize([game]).parent.choice_rejections, 1);
    assert.equal(summarize([game]).parent.sides.a.valid_games, 0);
    const failed = record({opponent: 'parent', teacher_side: 'a', index: 2}, {status: 'failed'});
    assert.throws(() => nextGame(config, [game, failed]), /exhausted/);
});

test('score summaries retain draws, losses, side splits and Wilson bounds', () => {
    const result = summarize([record({opponent: 'parent', teacher_side: 'a'}, {result: {result: 'draw'}}),
        record({opponent: 'parent', teacher_side: 'b'}, {result: {result: 'loss'}})]).parent;
    assert.equal(result.score_rate, 0.25);
    assert.equal(result.draws, 1);
    assert.equal(result.losses, 1);
    assert.equal(result.sides.a.score_rate, 0.5);
    assert.equal(result.coverage.coverage_rate, 0.5);
    assert.equal(wilson(0, 0), null);
    assert.ok(Math.abs(wilson(50, 100).low - 0.40383153) < 1e-7);
    assert.ok(wilson(100, 100).low > 0.96);
});

test('resume continues the same schedule and rejects changed provenance or damaged records', () => {
    const identity = {config, hash: 'fixture'};
    const games = [record(nextGame(config, []))];
    assert.deepEqual(validateResume({identity, games}, identity, config), games);
    assert.equal(nextGame(config, games).teacher_side, 'b');
    assert.throws(() => validateResume({identity, games}, {...identity, hash: 'changed'}, config), /inputs changed/);
    assert.throws(() => validateResume({identity, games: [games[0], games[0]]}, identity, config), /schedule/);
    assert.throws(() => validateResume({identity, games: [{...games[0], coverage: {}}]}, identity, config), /coverage/);
});

test('crash recovery uses the atomic backup, not an empty new benchmark', async () => {
    const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'tactical-benchmark-'));
    try {
        const filename = path.join(directory, 'summary.json');
        await fs.writeFile(filename, '{partial');
        await fs.writeFile(`${filename}.bak`, JSON.stringify({games: [1]}));
        assert.deepEqual(await loadSaved(filename), {games: [1]});
        assert.equal(await loadSaved(path.join(directory, 'missing.json')), null);
    } finally { await fs.rm(directory, {recursive: true}); }
});

test('random opponent uses a seeded player stream and records choice errors', async () => {
    const stream = {async *[Symbol.asyncIterator]() {
        yield '|error|[Unavailable choice] fixture\n|win|RandomPolicy';
    }};
    const result = await driveRandom(stream, [1, 2, 3, 4]);
    assert.equal(result.result.result, 'win');
    assert.equal(result.choiceRejections.length, 1);
    await assert.rejects(driveRandom({async *[Symbol.asyncIterator]() { yield '|turn|1'; }}, [1, 2, 3, 4]), /without a result/);
});

test('benchmark resource settings remain conservative', () => {
    validateConfig(config);
    for (const changes of [{agent_omp_threads: 2}, {max_system_cpu_percent: 90},
        {max_system_memory_percent: 90}, {max_attempts_per_side: 0}])
        assert.throws(() => validateConfig({...config, ...changes}), /bounded/);
});

test('confirmation changes only the seed and does not reuse screening battle or team seeds', () => {
    const screening = require('../config/experimental/tactical_teacher_benchmark.json');
    const confirmation = require('../config/experimental/tactical_teacher_confirmation.json');
    validateConfig(confirmation);
    assert.notEqual(confirmation.seed, screening.seed);
    assert.deepEqual({...confirmation, seed: screening.seed}, screening);
    const tuples = settings => Array.from({length: settings.max_attempts_per_side}, (_, pair) => {
        const seeds = battleSeeds(settings.seed, pair * 2, true);
        return [seeds.battle, seeds.random, ...seeds.players];
    }).flat();
    const seen = new Set(tuples(screening));
    assert.equal(tuples(confirmation).some(seed => seen.has(seed)), false);
});

test('student evaluation loads the candidate on either side without enabling tactical overrides', () => {
    const student = require('../config/experimental/tactical_student_benchmark.json');
    validateConfig(student);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(student, battleSeeds(student.seed, side, true), side, 'parent', null);
        assert.ok(agents[side].checkpoint.endsWith('tactical_student_best.chk'));
        assert.ok(agents[1 - side].checkpoint.endsWith('current_arch_full.chk'));
        assert.equal(agents[side].tacticalTeacher, false);
        assert.equal(agents[1 - side].tacticalTeacher, false);
        const againstRandom = createBattleAgents(student, battleSeeds(student.seed, side, true), side, 'random', null);
        assert.equal(againstRandom[1 - side], null);
        assert.equal(againstRandom[side].tacticalTeacher, false);
    }
    const teacher = createBattleAgents(config, battleSeeds(config.seed, 0, true), 0, 'parent', null);
    assert.equal(teacher[0].tacticalTeacher, true);
    assert.equal(teacher[1].tacticalTeacher, false);
    assert.throws(() => validateConfig({...student, collect_demonstrations: true}), /bounded/);
});

test('neural results resume without fabricated tactical coverage and reject a policy-mode mismatch', () => {
    const neural = {...config, tactical_teacher: false};
    const game = record(nextGame(neural, []), {policy_mode: 'neural', coverage: null});
    const identity = {config: neural};
    assert.deepEqual(validateResume({identity, games: [game]}, identity, neural), [game]);
    assert.equal(summarize([game]).parent.coverage, null);
    assert.equal(summarize([game]).parent.wins, 1);
    assert.throws(() => validateResume({identity, games: [game]}, identity, config), /policy mode/);
    assert.throws(() => validateResume({identity, games: [{...game, coverage: {}}]}, identity, neural), /must not contain/);
});

test('larger student confirmation freezes the policy and uses fresh battle and team seeds', () => {
    const screening = require('../config/experimental/tactical_student_benchmark.json');
    const confirmation = require('../config/experimental/tactical_student_confirmation.json');
    validateConfig(confirmation);
    assert.equal(confirmation.games_per_side, 500);
    assert.equal(confirmation.max_attempts_per_side, 625);
    assert.deepEqual({...confirmation, seed: screening.seed, games_per_side: screening.games_per_side,
        max_attempts_per_side: screening.max_attempts_per_side}, screening);
    const screeningSeeds = new Set();
    for (let pair = 0; pair < screening.max_attempts_per_side; ++pair) {
        const seeds = battleSeeds(screening.seed, pair * 2, true);
        for (const tuple of [seeds.battle, seeds.random, ...seeds.players]) screeningSeeds.add(tuple);
    }
    for (let pair = 0; pair < confirmation.max_attempts_per_side; ++pair) {
        const seeds = battleSeeds(confirmation.seed, pair * 2, true);
        for (const tuple of [seeds.battle, seeds.random, ...seeds.players])
            assert.equal(screeningSeeds.has(tuple), false);
    }
});

test('parent control reuses the student random-matchup settings and skips the parent matchup', () => {
    const control = require('../config/experimental/tactical_parent_control.json');
    const student = require('../config/experimental/tactical_student_confirmation.json');
    validateConfig(control);
    const {opponents, ...settings} = control;
    assert.deepEqual(opponents, ['random']);
    assert.deepEqual({...settings, candidate_checkpoint: student.candidate_checkpoint}, student);
    assert.equal(control.candidate_checkpoint, control.checkpoint);
    const tiny = {...control, games_per_side: 1, max_attempts_per_side: 2};
    const games = [];
    assert.equal(nextGame(tiny, games).opponent, 'random');
    games.push(record(nextGame(tiny, games), {policy_mode: 'neural', coverage: null}));
    games.push(record(nextGame(tiny, games), {policy_mode: 'neural', coverage: null}));
    assert.equal(nextGame(tiny, games), null);
    assert.equal(summarize(games).parent.attempts, 0);
    assert.equal(summarize(games).random.valid_games, 2);
    for (const opponents of [[], ['unknown'], ['random', 'random'], 'random'])
        assert.throws(() => validateConfig({...control, opponents}), /bounded/);
});
