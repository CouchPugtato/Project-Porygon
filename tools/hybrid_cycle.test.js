'use strict';

const assert = require('node:assert/strict');
const {test} = require('node:test');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {comparable} = require('./fixtures/position_simulator');
const {createHybridFixture: createFixture} = require('./fixtures/hybrid_cycle');
const {replacementFixture} = require('./fixtures/replacement_history');
const {assertLegalTeams} = require('./fixtures/tactical_reliability');
const {capturePosition, restorePosition, playerObservation, advancePosition} = require('./position_simulator');
const {decisionActions} = require('./planner_actions');
const {checkWorldChoice} = require('./opening_actions');
const {propagateWorld, PlannerBeliefs, nextSeed} = require('./planner_beliefs');
const {searchBeliefWorlds} = require('./planner_search');
const {HybridPlanner} = require('./hybrid_planner');
const {playFixtureGame, validateSaved, validateConfig} = require('./hybrid_cycle_experiment');

const SEED = [334, 1, 2, 3], SAMPLE = {mode: 'sample', seed: SEED};
const SEARCH = {seed: SEED, candidates: 4, responses: 1, chance_samples: 1,
    continuation_choices: 1, max_nodes: 128, max_elapsed_ms: 3000};
function budget() { return {available: () => true, consume() {}}; }

test('expanded requests include spread support buffs recovery fields pivots and tera', () => {
    const battle = createFixture();
    try {
        assertLegalTeams(battle);
        const position = capturePosition(battle, 'fixture');
        for (const side of ['p1', 'p2']) {
            const view = playerObservation(battle, side), actions = decisionActions(view);
            const pool = actions.turns.map(turn => turn.command);
            assert(pool.some(command => command.includes('terastallize')));
            assert(actions.turns.every(turn => turn.actions.filter(action => action.tera).length <= 1));
            for (const command of side === 'p1' ? ['move 3 -2, move 3', 'move 4, move 4 1'] :
                ['move 4, move 4', 'move 3, move 3']) {
                assert(pool.includes(command), command);
                assert(checkWorldChoice(position, view, command, SAMPLE).accepted);
            }
        }
    } finally { battle.destroy(); }
});

for (const side of ['p1', 'p2']) test(`pivot queue and later actions replay exactly from ${side}`, () => {
    const battle = createFixture();
    try {
        const start = capturePosition(battle, 'fixture'), before = playerObservation(battle, side);
        const commands = {p1: 'move 4, move 4 1', p2: 'move 1 2, move 3'};
        const next = advancePosition(start, commands, SAMPLE), pivot = next.observations[side];
        assert.equal(pivot.phase, 'switch');
        const matching = propagateWorld(start, before, commands[side], pivot, new PRNG(SEED.join(',')), budget(), 128);
        assert(matching);
        const pool = decisionActions(next.observations.p1).turns;
        assert(pool.some(turn => turn.command === 'pass, switch 3'));
        assert.equal(decisionActions(next.observations.p2).turns.length, 0);
        const completed = advancePosition(next.position, {p1: 'pass, switch 3'}, SAMPLE);
        const replayed = propagateWorld(next.position, pivot, side === 'p1' ? 'pass, switch 3' : null,
            completed.observations[side], new PRNG(SEED.join(',')), budget(), 128);
        assert(replayed);
        const copy = restorePosition(replayed, SAMPLE);
        try { assert.deepEqual(playerObservation(copy, side), completed.observations[side]); }
        finally { copy.destroy(); }
        assert.deepEqual(comparable(capturePosition(battle, 'fixture')), comparable(start));
    } finally { battle.destroy(); }
});

test('tera evidence conditions later worlds without exposing the hidden type at the opening', () => {
    const battle = createFixture();
    try {
        const tracker = new PlannerBeliefs({seed: SEED, particles: 1});
        const before = playerObservation(battle, 'p1');
        tracker.update(before);
        const next = advancePosition(capturePosition(battle, 'fixture'),
            {p1: 'move 4, move 2', p2: 'move 2 terastallize, move 2'}, SAMPLE);
        const report = tracker.update(next.observations.p1, 'move 4, move 2');
        assert(report.supported, JSON.stringify(report));
        const copy = restorePosition(tracker.worlds[0], SAMPLE);
        try {
            assert.equal(copy.p2.active[0].terastallized, 'Normal');
            assert.deepEqual(playerObservation(copy, 'p1'), next.observations.p1);
        } finally { copy.destroy(); }
        assert(!JSON.stringify(before).includes('-terastallize'));
    } finally { battle.destroy(); }
});

test('persistent worlds carry pp field state and stat stages beyond the old history horizon', () => {
    const battle = createFixture();
    try {
        const tracker = new PlannerBeliefs({seed: SEED, particles: 1, max_nodes: 512});
        let report = tracker.update(playerObservation(battle, 'p1'));
        assert(report.supported);
        for (let turn = 0; turn < 5; ++turn) {
            assert(battle.choose('p1', 'move 3 -2, move 2'));
            assert(battle.choose('p2', 'move 3, move 3'));
            report = tracker.update(playerObservation(battle, 'p1'), 'move 3 -2, move 2');
            assert(report.supported, JSON.stringify(report));
        }
        assert.equal(battle.turn, 6);
        assert(report.retained > 0);
        const copy = restorePosition(tracker.worlds[0], SAMPLE);
        try {
            assert.equal(copy.p2.active[0].boosts.spd, 6);
            assert.equal(copy.p2.active[1].boosts.def, 6);
            assert.deepEqual(playerObservation(copy, 'p1'), playerObservation(battle, 'p1'));
        } finally { copy.destroy(); }
    } finally { battle.destroy(); }
});

test('replacement decisions receive complete next turn scores rather than legal unranked incumbents', async () => {
    const {battle, history} = replacementFixture({kind: 'double_two_reserves'});
    try {
        const before = capturePosition(battle, 'fixture');
        const result = await searchBeliefWorlds(history.observations.at(-1), [before], SEARCH);
        assert(result.supported, JSON.stringify(result));
        assert(result.candidates.filter(row => row.status === 'completed').length >= 2);
        assert(new Set(result.candidates.map(row => row.score)).size > 1);
        assert(result.nodes > result.candidates.length);
        assert(Number.isFinite(result.selected_score));
        assert.deepEqual(comparable(capturePosition(battle, 'fixture')), comparable(before));
    } finally { battle.destroy(); }
});

test('pivot search finishes queued actions and scores a following ordinary turn', async () => {
    const battle = createFixture();
    try {
        const pivot = advancePosition(capturePosition(battle, 'fixture'),
            {p1: 'move 4, move 4 1', p2: 'move 1 2, move 3'}, SAMPLE);
        const result = await searchBeliefWorlds(pivot.observations.p1, [pivot.position], SEARCH);
        assert(result.supported, JSON.stringify(result));
        assert.equal(result.selected_command, 'pass, switch 3');
        assert(result.nodes >= 2);
    } finally { battle.destroy(); }
});

test('budget exhaustion cannot promote an incompletely evaluated candidate', async () => {
    const {battle, history} = replacementFixture({kind: 'double_two_reserves'});
    try {
        const result = await searchBeliefWorlds(history.observations.at(-1), [capturePosition(battle, 'fixture')],
            {...SEARCH, max_nodes: 1});
        assert(!result.supported); assert.equal(result.selected_command, null);
        assert(result.budget_exhausted); assert.equal(result.nodes, 1);
    } finally { battle.destroy(); }
});

test('full battle integration retains an explicit observation only fallback', async () => {
    const battle = createFixture();
    try {
        let received;
        const planner = new HybridPlanner({seed: SEED, beliefs: {max_boundaries: 1, particles: 1},
            search: {candidates: 1, responses: 1, chance_samples: 1, max_nodes: 1}}, view => {
            received = view; return decisionActions(view).turns[0].command;
        });
        await planner.decide(playerObservation(battle, 'p1'));
        assert(battle.choose('p1', 'move 3 -2, move 2')); assert(battle.choose('p2', 'move 3, move 3'));
        const decision = await planner.decide(playerObservation(battle, 'p1'), 'move 3 -2, move 2');
        assert.equal(decision.source, 'fallback');
        assert.equal(decision.reason, 'history_capacity_exhausted');
        assert.deepEqual(received, playerObservation(battle, 'p1'));
        assert(!Object.hasOwn(received, 'state_json'));
        assert(planner.stats.fallbacks > 0);
    } finally { battle.destroy(); }
});

for (const profile of [
    {name: 'spread and partner support', commands: {p1: 'move 3 -2, move 3', p2: 'move 3, move 3'}},
    {name: 'recovery and field effects', commands: {p1: 'move 4, move 2', p2: 'move 4, move 4'},
        prepare(battle) {
            assert(battle.choose('p1', 'move 1 1, move 2'));
            assert(battle.choose('p2', 'move 3, move 3'));
            assert(battle.p2.active[0].hp < battle.p2.active[0].maxhp);
        }},
    {name: 'opposing tera and self stat boosts', commands: {p1: 'move 4, move 2', p2: 'move 3 terastallize, move 3'}},
]) test(`native broad mechanics replay ${profile.name}`, () => {
    const battle = createFixture();
    try {
        profile.prepare?.(battle);
        const parent = capturePosition(battle, 'fixture');
        // aligned chance isolates command inference from rare multi damage matches
        const chance = new PRNG(SEED.join(',')); nextSeed(chance);
        const next = advancePosition(parent, profile.commands, {mode: 'sample', seed: nextSeed(chance)});
        if (profile.name === 'recovery and field effects') {
            assert(next.observations.p1.log.some(line => line.startsWith('|-heal|p2a: Snorlax|')));
            assert(next.observations.p1.log.some(line => line.startsWith('|-fieldstart|move: Trick Room')));
        }
        for (const side of ['p1', 'p2']) {
            const replayed = propagateWorld(parent, playerObservation(battle, side), profile.commands[side],
                next.observations[side], new PRNG(SEED.join(',')), budget(), 256);
            assert(replayed, side);
            const copy = restorePosition(replayed, SAMPLE);
            try { assert.deepEqual(playerObservation(copy, side), next.observations[side]); }
            finally { copy.destroy(); }
        }
    } finally { battle.destroy(); }
});

test('unmatched evidence discards particles rather than patching health or spent pp', () => {
    const battle = createFixture();
    try {
        const tracker = new PlannerBeliefs({seed: SEED, particles: 1, max_nodes: 32});
        assert(tracker.update(playerObservation(battle, 'p1')).supported);
        const next = advancePosition(capturePosition(battle, 'fixture'),
            {p1: 'move 4, move 2', p2: 'move 3, move 3'}, SAMPLE);
        next.observations.p1.request.active[0].moves[0].pp = 999;
        const report = tracker.update(next.observations.p1, 'move 4, move 2');
        assert(!report.supported); assert.equal(tracker.worlds.length, 0);
        assert(report.nodes <= 32);
    } finally { battle.destroy(); }
});

test('initial decisions are invariant to hidden opposing sets tera types and actual future rng', async () => {
    const first = createFixture(), second = createFixture({seed: [999, 8, 7, 6], changeTeams: teams => {
        teams[1][0].teraType = 'Ghost'; teams[1][0].moves = ['protect', 'rest'];
        teams[1][0].item = 'Leftovers'; teams[1][0].evs = {hp: 252, def: 1};
        teams[1][2].species = 'Espeon'; teams[1][2].ability = 'Synchronize';
        teams[1][2].moves = ['psychic', 'protect']; teams[1][2].teraType = 'Psychic';
    }});
    try {
        assertLegalTeams(first); assertLegalTeams(second);
        assert.deepEqual(playerObservation(first, 'p1'), playerObservation(second, 'p1'));
        const decide = battle => new HybridPlanner({seed: SEED, beliefs: {particles: 1}, search: SEARCH},
            view => decisionActions(view).turns[0].command).decide(playerObservation(battle, 'p1'));
        const a = await decide(first), b = await decide(second);
        assert.equal(a.command, b.command); assert.equal(a.source, b.source);
        assert.deepEqual(a.search?.candidates, b.search?.candidates);
    } finally { first.destroy(); second.destroy(); }
});

test('bounded battle resume verifies completed native traces and refuses corruption', async () => {
    const config = validateConfig({...require('../config/experimental/hybrid_cycle_smoke.json'),
        beliefs: {particles: 1, max_boundaries: 1}, search: {max_nodes: 1, candidates: 1, responses: 1, chance_samples: 1}});
    const game = await playFixtureGame(config, 0);
    assert.equal(game.status, 'completed', game.failure);
    assert.equal(game.policy_mode, 'hybrid');
    const identity = {test: 'native replay'}, saved = {identity, config, games: [game]};
    assert.equal(validateSaved(saved, identity, config).length, 1);
    assert.throws(() => validateSaved(saved, {test: 'changed'}, config), /incompatible/);
    const corrupt = structuredClone(saved); corrupt.games[0].choices.pop();
    assert.throws(() => validateSaved(corrupt, identity, config), /incomplete/);
    const changedResult = structuredClone(saved); changedResult.games[0].result.result =
        game.result.result === 'win' ? 'loss' : 'win';
    assert.throws(() => validateSaved(changedResult, identity, config), /terminal replay/);
    assert.throws(() => validateConfig({...config, search: {max_nodes: 999999}}), /planner settings/);
});
