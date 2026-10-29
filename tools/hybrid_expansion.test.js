'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {capturePosition, restorePosition, advancePosition, playerObservation} = require('./position_simulator');
const {propagateWorld, nextSeed, PlannerBeliefs} = require('./planner_beliefs');
const {mismatchEvidence, revealedConstraints} = require('./planner_evidence');
const {mutateProposal} = require('./planner_recovery');
const {solveMatrix, solveBeliefGame} = require('./planner_matrix');
const {evaluatePosition} = require('./planner_value');
const {searchBeliefWorlds} = require('./planner_search');
const {HybridPlanner} = require('./hybrid_planner');
const {HybridAgent} = require('./hybrid_agent');
const {schedule, validateBenchmark, compareProfiles, computeComparison, recordHash, validateLiveResume} = require('./hybrid_benchmark');

const SEED = [338, 1, 2, 3], SAMPLE = {mode: 'sample', seed: SEED};

test('mixed solver finds matching pennies mixture and reports finite game bounds', () => {
    const solution = solveMatrix([[1, -1], [-1, 1]]);
    assert(Math.abs(solution.policy[0] - 0.5) < 0.02);
    assert(solution.gap < 0.05); assert(Math.abs(solution.lower) < 0.05);
    const dominated = solveMatrix([[2, 2], [0, 0]]);
    assert(dominated.policy[0] > 0.98);
    assert.throws(() => solveMatrix([[NaN]]), /finite/);
    const belief = solveBeliefGame([[[2], [0]], [[0], [2]]]);
    assert(Math.abs(belief.lower - 1) < 0.01);
});

test('positional evaluator distinguishes status and boosts from unchanged material', () => {
    const battle = createHybridFixture({changeTeams: teams => { teams[0][0].moves[2] = 'thunderwave'; }});
    try {
        const material = evaluatePosition(battle, 'p1', 'material'), score = evaluatePosition(battle, 'p1', 'positional');
        assert(battle.choose('p1', 'move 3 1, move 2')); assert(battle.choose('p2', 'move 3, move 3'));
        assert.equal(battle.p2.active[0].status, 'par');
        assert(evaluatePosition(battle, 'p1', 'positional') > score);
        assert.equal(evaluatePosition(battle, 'p1', 'material'), material);
    } finally { battle.destroy(); }
});

test('failure diagnostics separate damage private requests and visible move order', () => {
    const view = {log: ['|move|p1a: A|Tackle|p2a: B', '|-damage|p2a: B|80/100'], request: null};
    const damage = structuredClone(view); damage.log[1] = '|-damage|p2a: B|70/100';
    assert.equal(mismatchEvidence(view, damage).kind, 'damage_or_chance_mismatch');
    const order = structuredClone(view); order.log[0] = '|move|p2a: B|Tackle|p1a: A';
    assert.equal(mismatchEvidence(view, order).kind, 'move_or_speed_order_mismatch');
    assert.equal(mismatchEvidence(view, {...view, request: {wait: true}}).kind, 'private_request_or_hidden_stats_mismatch');
});

test('recovery mutates only proposal builds without mutating source worlds', () => {
    const sets = [{key: 'p2: Test', set: {nature: 'Serious', ivs: {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31},
        evs: {hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0}, moves: ['protect']}}];
    const before = structuredClone(sets), next = mutateProposal(sets, new PRNG(SEED.join(',')));
    assert.deepEqual(sets, before); assert.notDeepEqual(next, sets);
    assert.deepEqual(next[0].set.moves, sets[0].set.moves);
});

const mechanics = [
    {name: 'hazards', change(teams) { teams[1][1].moves[2] = 'stealthrock'; }, choices: {p1: 'move 4, move 2', p2: 'move 3, move 3'}, event: '|-sidestart|p1:'},
    {name: 'weather', change(teams) { teams[1][1].moves[2] = 'raindance'; }, choices: {p1: 'move 4, move 2', p2: 'move 3, move 3'}, event: '|-weather|RainDance'},
    {name: 'terrain', change(teams) { teams[0][0].moves[2] = 'electricterrain'; }, choices: {p1: 'move 3, move 2', p2: 'move 3, move 3'}, event: '|-fieldstart|move: Electric Terrain'},
    {name: 'choice lock', change(teams) { teams[0][0].item = 'Choice Specs'; }, choices: {p1: 'move 1 1, move 2', p2: 'move 3, move 3'}, event: '|move|p1a: Pikachu|Thunderbolt'},
    {name: 'redirection', change(teams) { Object.assign(teams[1][0], {species: 'Clefairy', ability: 'Magic Guard', moves: ['followme', 'protect', 'calmmind', 'moonblast']}); }, choices: {p1: 'move 1 2, move 2', p2: 'move 1, move 3'}, event: '|-singleturn|p2a: Clefairy|move: Follow Me'},
    {name: 'forced drag', change(teams) { Object.assign(teams[1][0], {species: 'Hippowdon', ability: 'Sand Force', moves: ['protect', 'slackoff', 'whirlwind', 'earthquake']}); }, choices: {p1: 'move 3 -2, move 2', p2: 'move 3 1, move 3'}, event: '|drag|'},
    {name: 'ability immunity', choices: {p1: 'move 1 2, move 2', p2: 'move 3, move 3'}, change(teams) { teams[1][1].ability = 'Volt Absorb'; teams[1][1].species = 'Lanturn'; teams[1][1].moves = ['thunderbolt', 'protect', 'stockpile', 'raindance']; teams[1][1].gender = 'M'; }, event: '|-immune|'},
];
for (const mechanic of mechanics) test(`native reconstruction regression for ${mechanic.name}`, () => {
    const battle = createHybridFixture({changeTeams: mechanic.change});
    try {
        assertLegalTeams(battle);
        const parent = capturePosition(battle, 'fixture'), rng = new PRNG(SEED.join(',')); nextSeed(rng);
        const next = advancePosition(parent, mechanic.choices, {mode: 'sample', seed: nextSeed(rng)});
        assert(next.observations.p1.log.some(line => line.startsWith(mechanic.event)), mechanic.name);
        const diagnostics = {}, budget = {available: () => true, consume() {}, diagnostics};
        for (const side of ['p1', 'p2']) {
            const matched = propagateWorld(parent, playerObservation(battle, side), mechanic.choices[side],
                next.observations[side], new PRNG(SEED.join(',')), budget, 1);
            assert(matched, `${side} ${JSON.stringify(diagnostics)}`);
        }
    } finally { battle.destroy(); }
});

test('hidden trapping disclosures replay as rejected choices without advancing the turn', () => {
    const battle = createHybridFixture({changeTeams: teams => {
        Object.assign(teams[1][0], {species: 'Gothitelle', ability: 'Shadow Tag', moves: ['protect', 'calmmind', 'psychic', 'thunderwave']});
    }});
    try {
        assertLegalTeams(battle);
        const before = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        battle.strictChoices = false;
        const accepted = battle.choose('p1', 'switch 3, move 2'); assert.equal(accepted, false);
        const after = playerObservation(battle, 'p1');
        const replayed = propagateWorld(position, before, {rejected: 'switch 3, move 2'}, after,
            new PRNG(SEED.join(',')), {available: () => true, consume() {}}, 1);
        assert(replayed); assert.equal(after.turn, before.turn);
    } finally { battle.destroy(); }
});

test('selective deeper mixed search stays within node budget and returns only completed root rows', async () => {
    const battle = createHybridFixture();
    try {
        const result = await searchBeliefWorlds(playerObservation(battle, 'p1'), [capturePosition(battle, 'fixture')],
            {seed: SEED, candidates: 2, responses: 1, chance_samples: 1, continuation_choices: 1,
                depth_turns: 2, strategy: 'mixed', evaluator: 'positional', max_nodes: 32, max_elapsed_ms: 3000});
        assert(result.supported, JSON.stringify(result)); assert(result.nodes <= 32);
        assert(result.nodes > result.candidates.length);
        assert(result.mixture.commands.includes(result.selected_command));
        assert(result.candidates.filter(row => row.status === 'completed').every(row => Number.isFinite(row.score)));
    } finally { battle.destroy(); }
});

test('unchanged decisions safely reuse the exact previous search without counting extra work', async () => {
    const battle = createHybridFixture();
    try {
        const planner = new HybridPlanner({seed: SEED, beliefs: {particles: 1},
            search: {candidates: 1, responses: 1, chance_samples: 1, max_nodes: 16}}, () => 'move 4, move 2');
        const view = playerObservation(battle, 'p1');
        const a = await planner.decide(view), nodes = planner.stats.nodes;
        const b = await planner.decide(view);
        assert.equal(a.command, b.command); assert.equal(planner.stats.nodes, nodes);
    } finally { battle.destroy(); }
});

test('balanced benchmark schedules matched profiles on both sides and compares compute without promoting a winner', () => {
    const config = validateBenchmark(require('../config/experimental/hybrid_benchmark.json'));
    const jobs = schedule(config);
    assert.equal(jobs.length, config.profiles.length * config.opponents.length * config.games_per_side * 2);
    for (const profile of config.profiles) for (const opponent of config.opponents) {
        const subset = jobs.filter(job => job.profile === profile.name && job.opponent === opponent);
        assert.equal(subset.filter(job => job.index % 2 === 0).length, config.games_per_side);
        assert.equal(subset.filter(job => job.index % 2 === 1).length, config.games_per_side);
    }
    assert.equal(schedule(config, true).length, 2);
    assert(compareProfiles([], config).pure_high.results.parent.valid_games === 0);
});

test('bridge normalizes only public clocks and keeps evaluation records ineligible for training', async () => {
    const battle = createHybridFixture(), sent = [];
    const baseline = {send: message => sent.push(message), nextMessage: async type => type === 'episode_complete' ?
        {training_eligible: true} : {command: 'move 4, move 2', action: 0, action2: 0}};
    try {
        const wrapper = new HybridAgent(baseline, {seed: SEED, beliefs: {particles: 1}, search: {max_nodes: 1}});
        const view = playerObservation(battle, 'p1');
        for (const line of view.log) wrapper.send({type: 'event', line});
        await wrapper.observeRequest(view.request);
        assert.deepEqual(wrapper.view, view);
        await wrapper.nextMessage('action');
        const terminal = await wrapper.nextMessage('episode_complete');
        assert.equal(terminal.training_eligible, false); assert.equal(terminal.policy_tag, 'hybrid-search-eval-v1');
        assert(sent.length > 0);
    } finally { battle.destroy(); }
});

test('benchmark resume binds completed records and refuses changed outcomes or saved failures', () => {
    const job = {profile: 'pure_low', opponent: 'random', index: 0};
    const game = {...job, status: 'completed', result: {result: 'win'}, choice_rejections: []};
    game.record_hash = recordHash(game);
    const saved = {identity: {source: 'test'}, games: [game]};
    assert.equal(validateLiveResume(saved, saved.identity, [job]).games.length, 1);
    const corrupt = structuredClone(saved); corrupt.games[0].result.result = 'loss';
    assert.throws(() => validateLiveResume(corrupt, saved.identity, [job]), /integrity/);
    const failed = structuredClone(saved); failed.games[0].status = 'failed';
    failed.games[0].record_hash = recordHash(failed.games[0]);
    assert.throws(() => validateLiveResume(failed, saved.identity, [job]), /failed game/);
});

test('compute comparisons keep paired team blocks and refuse confidence claims on tiny samples', () => {
    const games = [];
    for (const profile of ['pure_low', 'pure_high']) for (const index of [0, 1])
        games.push({profile, index, opponent: 'parent', status: 'completed', choice_rejections: [],
            result: {result: profile === 'pure_high' ? 'win' : 'loss'}});
    const result = computeComparison(games).parent;
    assert.equal(result.matched_team_blocks, 1); assert.equal(result.mean_score_delta, 1);
    assert.equal(result.exploratory_cluster_interval, null);
    games.pop(); assert.equal(computeComparison(games).parent.matched_team_blocks, 0);
});

test('unproven recovery remains opt in and impossible revealed moves are filtered without rollouts', () => {
    assert.equal(new PlannerBeliefs({seed: SEED}).settings.recovery, false);
    const battle = createHybridFixture();
    try {
        const before = playerObservation(battle, 'p1'), after = structuredClone(before);
        after.log.push('|move|p2a: Snorlax|Thunderbolt|p1a: Pikachu');
        assert.equal(revealedConstraints(battle, after), 'revealed_move_missing');
        let nodes = 0;
        const diagnostics = {};
        const result = propagateWorld(capturePosition(battle, 'fixture'), before, 'move 4, move 2', after,
            new PRNG(SEED.join(',')), {available: () => true, consume: () => ++nodes, filter_evidence: true, diagnostics});
        assert.equal(result, null); assert.equal(nodes, 0); assert.equal(diagnostics.revealed_move_missing, 1);
    } finally { battle.destroy(); }
});
