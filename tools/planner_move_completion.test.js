'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, advancePosition, playerObservation} = require('./position_simulator');
const {moveCompletions, completeMoveHistory} = require('./planner_move_completion');
const {PersistentRecovery} = require('./planner_persistence');

const SEED = [379, 1, 2, 3];

function sampleHistory(actual, hypothesis) {
    const before = playerObservation(actual, 'p1');
    assert.deepEqual(before, playerObservation(hypothesis, 'p1'));
    const opening = capturePosition(hypothesis, 'fixture');
    const firstChoices = {p1: 'move 1 1, move 2', p2: 'move 3, move 3'};
    const first = advancePosition(capturePosition(actual, 'fixture'), firstChoices, {mode: 'sample', seed: SEED});
    const proposed = advancePosition(opening, firstChoices, {mode: 'sample', seed: SEED});
    assert.deepEqual(first.observations.p1, proposed.observations.p1);
    const lastChoices = {p1: 'move 4, move 2', p2: 'move 1 1, move 3'};
    const last = advancePosition(first.position, lastChoices, {mode: 'sample', seed: SEED});
    return {path: {opening, steps: [{choices: firstChoices, seed: SEED}]},
        history: {observations: [before, first.observations.p1, last.observations.p1],
            own_choices: [firstChoices.p1, lastChoices.p1]}, proposed, lastChoices};
}

test('new observed moves replace only unseen slots without mutating builds or prior evidence', () => {
    const sets = [{key: 'p2: Snorlax', set: {moves: ['amnesia', 'rest', 'protect', 'bodyslam'], item: 'Leftovers'}}];
    const original = structuredClone(sets), facts = {opponent: {pokemon: [{key: sets[0].key, moves: ['amnesia', 'tackle']}]}};
    const result = moveCompletions(sets, facts);
    assert(result.supported); assert(result.proposals.length > 1);
    for (const proposal of result.proposals) {
        assert(proposal[0].set.moves.includes('amnesia')); assert(proposal[0].set.moves.includes('tackle'));
        assert.equal(proposal[0].set.moves.length, 4); assert.equal(proposal[0].set.item, 'Leftovers');
    }
    assert.deepEqual(sets, original);
    assert.equal(moveCompletions(sets, {opponent: {pokemon: [{key: 'p2: Unknown', moves: []}]}}).supported, false);
});

test('move completion fills free move slots and refuses more than four observed moves', () => {
    const sets = [{key: 'p2: Snorlax', set: {moves: ['amnesia']}}];
    const facts = {opponent: {pokemon: [{key: sets[0].key, moves: ['amnesia', 'tackle']}]}};
    assert(moveCompletions(sets, facts).proposals[0][0].set.moves.includes('amnesia'));
    facts.opponent.pokemon[0].moves = ['amnesia', 'tackle', 'protect', 'rest', 'bodyslam'];
    assert.equal(moveCompletions(sets, facts).supported, false);
});

test('a newly revealed legal move rebuilds and independently replays the entire retained history', async () => {
    const actual = createHybridFixture(), hypothesis = createHybridFixture({changeTeams: teams => { teams[1][0].moves[0] = 'bodyslam'; }});
    try {
        const {path, history} = sampleHistory(actual, hypothesis), original = structuredClone({path, history});
        let nodes = 0;
        const result = await completeMoveHistory(path, history, SEED, {available: () => nodes < 64, consume: () => ++nodes});
        assert(result.verified, result.reason); assert.equal(result.witnesses[0].steps.length, 2);
        assert(nodes <= 64); assert.deepEqual({path, history}, original);
        let position = result.witnesses[0].opening;
        for (const [index, step] of result.witnesses[0].steps.entries()) {
            const next = advancePosition(position, step.choices, {mode: 'sample', seed: step.seed});
            assert.deepEqual(next.observations.p1, history.observations[index + 1]); position = next.position;
        }
        history.observations[1].request.side.pokemon[0].condition = '999/999'; nodes = 0;
        const rejected = await completeMoveHistory(path, history, SEED, {available: () => nodes < 64, consume: () => ++nodes});
        assert.equal(rejected.verified, false); assert.deepEqual(rejected.witnesses, []);
    } finally { actual.destroy(); hypothesis.destroy(); }
});

test('persistent recovery admits a move completion only after full history verification', async () => {
    const actual = createHybridFixture(), hypothesis = createHybridFixture({changeTeams: teams => { teams[1][0].moves[0] = 'bodyslam'; }});
    try {
        const {path, history, proposed, lastChoices} = sampleHistory(actual, hypothesis);
        const recovery = new PersistentRecovery({seed: SEED, persistent: true, move_completion: true, max_elapsed_ms: 1000});
        recovery.bank.remember(new Map([[proposed.position.state_json, path]]), [proposed.position],
            {observations: history.observations.slice(0, 2), own_choices: history.own_choices.slice(0, 1)});
        const result = await recovery.recover({history, lastFailure: {simulation_trace: {position: proposed.position}}},
            history.observations.at(-1), lastChoices.p1);
        assert(result.verified); assert.equal(result.stage, 'move_completion');
        assert(result.move_nodes > 0); assert.equal(result.fit_nodes, 0);
        assert.equal(result.witnesses[0].steps.length, 2);
        assert.throws(() => new PersistentRecovery({seed: SEED, move_completion: true}), /bounded/);
    } finally { actual.destroy(); hypothesis.destroy(); }
});

test('an illegal newly revealed move and exhausted budgets never publish a completed world', async () => {
    const actual = createHybridFixture(), hypothesis = createHybridFixture({changeTeams: teams => { teams[1][0].moves[0] = 'bodyslam'; }});
    try {
        const {path, history} = sampleHistory(actual, hypothesis);
        const exhausted = await completeMoveHistory(path, history, SEED,
            {available: () => false, consume() { throw new Error('exhausted budget consumed'); }});
        assert.equal(exhausted.verified, false); assert.deepEqual(exhausted.witnesses, []);
        history.observations.at(-1).log = history.observations.at(-1).log.map(line =>
            line.startsWith('|move|p2a: Snorlax|Tackle|') ? line.replace('|Tackle|', '|Spore|') : line);
        let nodes = 0;
        const illegal = await completeMoveHistory(path, history, SEED,
            {available: () => nodes < 64, consume: () => ++nodes});
        assert.equal(illegal.verified, false); assert.deepEqual(illegal.witnesses, []); assert(nodes <= 64);
    } finally { actual.destroy(); hypothesis.destroy(); }
});
