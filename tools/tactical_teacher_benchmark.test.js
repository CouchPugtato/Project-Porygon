'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {battleSeeds, createBattleAgents, driveRandom} = require('./tactical_teacher_smoke');
const {AgentProcess} = require('./counterfactual_rollout');
const {isValid, loadSaved, nextGame, summarize, validateConfig, validateResume, wilson} = require('./tactical_teacher_benchmark');
const config = {...require('../config/experimental/tactical_teacher_benchmark.json'), games_per_side: 1, max_attempts_per_side: 2};

test('passive audit routes only to the student and cannot collect training demonstrations', () => {
    const settings = require('../config/experimental/tactical_student_decision_audit.json');
    validateConfig(settings);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(settings, battleSeeds(settings.seed, side, true), side, 'parent', null);
        assert.equal(agents[side].tacticalAudit, true);
        assert.equal(agents[1 - side].tacticalAudit, false);
        assert.equal(agents[side].inferencePolicy, 'sample');
        assert.equal(agents[side].tacticalTeacher, false);
    }
    assert.throws(() => validateConfig({...settings, collect_demonstrations: true}), /bounded/);
    assert.throws(() => validateConfig({...settings, tactical_teacher: true}), /bounded/);
    const identity = {config: settings};
    const game = record(nextGame(settings, []), {policy_mode: 'neural', coverage: null});
    assert.throws(() => validateResume({identity, games: [game]}, identity, settings), /no accepted decisions/);
});

function seedTuples(settings) {
    return Array.from({length: settings.max_attempts_per_side}, (_, pair) => {
        const seeds = battleSeeds(settings.seed, pair * 2, true);
        return [seeds.battle, seeds.random, ...seeds.players];
    }).flat();
}

test('target-only experiment freezes seeds, models and sampling while routing only the candidate intervention', () => {
    const control = require('../config/experimental/tactical_target_control.json');
    const intervention = require('../config/experimental/tactical_target_intervention.json');
    validateConfig(control);
    validateConfig(intervention);
    assert.deepEqual({...intervention, tactical_targets: false}, control);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(intervention, battleSeeds(intervention.seed, side, true), side, 'parent', null);
        assert.equal(agents[side].tacticalTargets, true);
        assert.equal(agents[1 - side].tacticalTargets, false);
        assert.ok(agents.every(agent => agent.inferencePolicy === 'sample' && !agent.tacticalTeacher));
    }
    for (const invalid of [{tactical_teacher: true}, {collect_demonstrations: true},
        {decision_audit: true}, {candidate_inference_policy: 'greedy'}])
        assert.throws(() => validateConfig({...intervention, ...invalid}), /bounded/);
    assert.throws(() => new AgentProcess('model', 1, null, 1, false, null, 'greedy', false, true), /plain sampled/);
    const tiny = {...intervention, games_per_side: 1, max_attempts_per_side: 2};
    const identity = {config: tiny};
    const game = record(nextGame(tiny, []), {policy_mode: 'neural', coverage: null, policy_tag: 'tactical-targets-v1',
        target_intervention: {training_eligible: false, target_decisions: 10, retargeted_decisions: 3}});
    assert.deepEqual(validateResume({identity, games: [game]}, identity, tiny), [game]);
    assert.equal(summarize([game]).parent.target_intervention.rate, 0.3);
    assert.throws(() => validateResume({identity, games: [game]}, identity, {...tiny, tactical_targets: false}), /plain inference/);
    assert.throws(() => validateResume({identity, games: [{...game, target_intervention: null}]}, identity, tiny), /lacks target/);
});

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
    const seen = new Set(seedTuples(screening));
    assert.equal(seedTuples(confirmation).some(seed => seen.has(seed)), false);
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
    const seen = new Set(seedTuples(screening));
    assert.equal(seedTuples(confirmation).some(seed => seen.has(seed)), false);
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

test('larger-data screen routes the frozen epoch and pilot correctly without tactical overrides', () => {
    const screen = require('../config/experimental/tactical_student_1000_screen.json');
    validateConfig(screen);
    assert.deepEqual(screen.opponents, ['parent', 'random', 'pilot']);
    assert.equal(2 * screen.games_per_side * screen.opponents.length, 600);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(screen, battleSeeds(screen.seed, side, true), side, 'pilot', null);
        assert.ok(agents[side].checkpoint.endsWith('tactical_student_dataset_epoch003.chk'));
        assert.ok(agents[1 - side].checkpoint.endsWith('run_0269_tactical_student_pilot' + path.sep + 'tactical_student_best.chk'));
        assert.ok(agents.every(agent => agent.tacticalTeacher === false));
    }
    assert.throws(() => validateConfig({...screen, pilot_checkpoint: ''}), /bounded/);
    assert.throws(() => validateConfig({...screen, pilot_checkpoint: undefined}), /bounded/);
});

test('three-opponent screening keeps separate results and resumes into the pilot matchup', () => {
    const screen = {...config, tactical_teacher: false, opponents: ['parent', 'random', 'pilot'], pilot_checkpoint: 'pilot.chk'};
    const games = [];
    for (let i = 0; i < 4; ++i)
        games.push(record(nextGame(screen, games), {policy_mode: 'neural', coverage: null}));
    assert.deepEqual(nextGame(screen, games), {opponent: 'pilot', index: 0, teacher_side: 'a'});
    const identity = {config: screen, pilot_checkpoint_sha256: 'fixture'};
    assert.deepEqual(validateResume({identity, games}, identity, screen), games);
    for (let i = 0; i < 2; ++i)
        games.push(record(nextGame(screen, games), {policy_mode: 'neural', coverage: null}));
    assert.equal(nextGame(screen, games), null);
    const results = summarize(games, screen.opponents);
    assert.deepEqual(Object.keys(results), ['parent', 'random', 'pilot']);
    assert.equal(results.pilot.valid_games, 2);
    assert.equal(results.parent.valid_games, 2);
    assert.throws(() => validateResume({identity, games}, {...identity, pilot_checkpoint_sha256: 'changed'}, screen), /inputs changed/);
});

test('three-opponent confirmation freezes all models and mode while increasing the fresh-seed sample', () => {
    const screen = require('../config/experimental/tactical_student_1000_screen.json');
    const confirmation = require('../config/experimental/tactical_student_1000_confirmation.json');
    validateConfig(confirmation);
    assert.equal(confirmation.games_per_side, 500);
    assert.equal(confirmation.max_attempts_per_side, 625);
    assert.equal(2 * confirmation.games_per_side * confirmation.opponents.length, 3000);
    assert.deepEqual({...confirmation, seed: screen.seed, games_per_side: screen.games_per_side,
        max_attempts_per_side: screen.max_attempts_per_side}, screen);
    const seen = new Set(seedTuples(screen));
    assert.equal(seedTuples(confirmation).some(seed => seen.has(seed)), false);
});

test('matched decoder profiles change only candidate decoding and add a same-checkpoint sampled opponent', () => {
    const sampled = require('../config/experimental/tactical_student_decoding_sample.json');
    const greedy = require('../config/experimental/tactical_student_decoding_greedy.json');
    validateConfig(sampled);
    validateConfig(greedy);
    assert.deepEqual({...greedy, candidate_inference_policy: 'sample', opponents: sampled.opponents}, sampled);
    assert.deepEqual(greedy.opponents, ['parent', 'random', 'sampled']);
    assert.equal(2 * greedy.games_per_side * greedy.opponents.length, 600);
    for (const side of [0, 1]) {
        const agents = createBattleAgents(greedy, battleSeeds(greedy.seed, side, true), side, 'sampled', null);
        assert.equal(agents[0].checkpoint, agents[1].checkpoint);
        assert.equal(agents[side].inferencePolicy, 'greedy');
        assert.equal(agents[1 - side].inferencePolicy, 'sample');
        assert.ok(agents.every(agent => agent.tacticalTeacher === false));
        const parent = createBattleAgents(greedy, battleSeeds(greedy.seed, side, true), side, 'parent', null);
        assert.ok(parent[1 - side].checkpoint.endsWith('current_arch_full.chk'));
        assert.equal(parent[1 - side].inferencePolicy, 'sample');
    }
    assert.throws(() => validateConfig({...greedy, tactical_teacher: true}), /bounded/);
    assert.throws(() => validateConfig({...greedy, collect_demonstrations: true}), /bounded/);
    assert.throws(() => validateConfig({...greedy, candidate_inference_policy: 'unknown'}), /bounded/);
    assert.throws(() => new AgentProcess('model', 1, null, 1, true, null, 'greedy'), /evaluation-only/);
    assert.throws(() => new AgentProcess('model', 1, {decisionIndex: 0}, 1, false, null, 'greedy'), /evaluation-only/);
});

test('greedy results cannot resume as sampled results, including the direct sampled matchup', () => {
    const greedy = {...config, tactical_teacher: false, candidate_inference_policy: 'greedy', opponents: ['sampled']};
    const identity = {config: greedy};
    const game = record(nextGame(greedy, []), {policy_mode: 'neural', inference_policy: 'greedy', coverage: null});
    assert.equal(game.opponent, 'sampled');
    assert.deepEqual(validateResume({identity, games: [game]}, identity, greedy), [game]);
    assert.equal(summarize([game], greedy.opponents).sampled.valid_games, 1);
    assert.throws(() => validateResume({identity, games: [{...game, inference_policy: 'sample'}]}, identity, greedy), /inference policy/);
});
