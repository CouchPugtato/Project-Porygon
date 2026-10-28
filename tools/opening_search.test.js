'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {compareOpeningTurns, searchOpening} = require('./opening_search');
const {capturePosition, playerObservation} = require('./position_simulator');
const {createFixture, comparable} = require('./fixtures/position_simulator');
const {tacticalOpening} = require('./fixtures/opening_search');

const OPTIONS = {seed: [326, 1, 2, 3], worlds: 2, candidates: 4, responses: 2, chance_samples: 4};
const ELECTRIC = 'move 1 1, move 2', OTHER = 'move 1 2, move 2', PASSIVE = 'move 2, move 2';
const WATER = 'move 2, move 1 1', IDLE = 'move 1, move 1', PROTECT = 'move 2, move 1';

test('one-turn search prefers a super-effective knockout to passive or resisted moves', async () => {
    const battle = tacticalOpening();
    try {
        const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const root = comparable(position);
        const result = await compareOpeningTurns(view, [{position, responses: [IDLE]}],
            [PASSIVE, OTHER, ELECTRIC], OPTIONS);
        assert.equal(result.selected_command, ELECTRIC);
        assert(result.selected_score >= 3, 'enemy HP fraction and one faint should be rewarded');
        assert.equal(result.nodes, 12); assert.equal(result.comparison_complete, true);
        assert.equal(result.fixture_worlds, true);
        assert.deepEqual(comparable(position), root); assert.deepEqual(playerObservation(battle, 'p1'), view);
        const opposite = await compareOpeningTurns(playerObservation(battle, 'p2'),
            [{position, responses: [ELECTRIC]}], [IDLE], OPTIONS);
        assert.equal(opposite.selected_score, -result.selected_score);
    } finally { battle.destroy(); }
});

test('opponent Protect changes the selected target; worst-response comparison is simultaneous', async () => {
    const battle = tacticalOpening({protect: true});
    try {
        const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const compare = responses => compareOpeningTurns(view, [{position, responses}], [ELECTRIC, OTHER], OPTIONS);
        assert.equal((await compare([IDLE])).selected_command, ELECTRIC);
        assert.equal((await compare([PROTECT])).selected_command, OTHER);
        const robust = await compare([IDLE, PROTECT]);
        assert.equal(robust.selected_command, OTHER);
        assert(robust.candidates.every(row => row.worlds[0].responses.length === 2));
        for (const row of robust.candidates)
            assert.equal(row.score, Math.min(...row.worlds[0].responses.map(response => response.mean)));
        const reordered = await compare([PROTECT, IDLE]);
        for (const row of robust.candidates) {
            const other = reordered.candidates.find(candidate => candidate.command === row.command);
            assert.deepEqual([...other.worlds[0].responses].sort((a, b) => a.command.localeCompare(b.command)),
                [...row.worlds[0].responses].sort((a, b) => a.command.localeCompare(b.command)));
        }
    } finally { battle.destroy(); }
});

test('different best actions in hidden worlds yield one shared action, not a clairvoyant average', async () => {
    const water = tacticalOpening({absorb: 'Water Absorb'}), volt = tacticalOpening({absorb: 'Volt Absorb'});
    try {
        const view = playerObservation(water, 'p1'); assert.deepEqual(playerObservation(volt, 'p1'), view);
        const worlds = [water, volt].map(battle => ({position: capturePosition(battle, 'fixture'), responses: [IDLE]}));
        const result = await compareOpeningTurns(view, worlds, [ELECTRIC, WATER], OPTIONS);
        const [electric, waterMove] = result.candidates;
        assert(electric.worlds[0].score > waterMove.worlds[0].score);
        assert(waterMove.worlds[1].score > electric.worlds[1].score);
        for (const row of result.candidates)
            assert.equal(row.score, (row.worlds[0].score + row.worlds[1].score) / 2);
        const oracle = [0, 1].reduce((sum, i) => sum + Math.max(...result.candidates.map(row => row.worlds[i].score)) / 2, 0);
        assert(oracle > result.selected_score);
        assert.equal(result.selected_score, Math.max(...result.candidates.map(row => row.score)));
        const reversed = await compareOpeningTurns(view, worlds, [WATER, ELECTRIC], OPTIONS);
        assert.equal(reversed.selected_command, result.selected_command);
        assert.equal(reversed.selected_score, result.selected_score, 'chance samples are shared across candidate ordering');
    } finally { water.destroy(); volt.destroy(); }
});

test('node exhaustion keeps only a fully compared incumbent and discards an incomplete row', async () => {
    const battle = tacticalOpening({protect: true});
    try {
        const result = await compareOpeningTurns(playerObservation(battle, 'p1'),
            [{position: capturePosition(battle, 'fixture'), responses: [IDLE, PROTECT]}],
            [OTHER, ELECTRIC], {...OPTIONS, chance_samples: 2, max_nodes: 5});
        assert.equal(result.nodes, 5); assert.equal(result.budget_exhausted, true);
        assert.equal(result.selected_command, OTHER);
        assert.deepEqual(result.candidates.map(row => row.status), ['completed', 'incomplete_budget']);
        assert.equal(result.candidates[1].score, null);
    } finally { battle.destroy(); }
});

test('resource wait consumes wall-clock budget without publishing an unfinished score', async () => {
    const battle = tacticalOpening();
    try {
        const result = await compareOpeningTurns(playerObservation(battle, 'p1'),
            [{position: capturePosition(battle, 'fixture'), responses: [IDLE]}], [ELECTRIC],
            {...OPTIONS, max_elapsed_ms: 10}, {waitForLaunch: () => new Promise(resolve => setTimeout(resolve, 20))});
        assert.equal(result.supported, false); assert.equal(result.selected_command, null);
        assert.equal(result.nodes, 0); assert.equal(result.budget_exhausted, true);
    } finally { battle.destroy(); }
});

test('a pivot interrupt is excluded rather than valued as the end of its turn', async () => {
    const battle = createFixture();
    try {
        const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const result = await compareOpeningTurns(view, [{position, responses: ['move 3, move 3']}],
            ['move 4, move 4 1', 'move 4, move 2'], OPTIONS);
        assert.equal(result.candidates[0].status, 'unsupported_continuation');
        assert.equal(result.candidates[0].score, null);
        assert.equal(result.selected_command, 'move 4, move 2');
        assert.equal(result.comparison_complete, false);
    } finally { battle.destroy(); }
});

test('player-facing search is invariant to changed actual hidden builds and future RNG', async () => {
    const original = createFixture(), hidden = createFixture({seed: [900, 1, 2, 3], changeTeams: teams => {
        Object.assign(teams[1][0], {ability: 'Thick Fat', item: 'Leftovers', moves: ['bellydrum', 'rest']});
        Object.assign(teams[1][2], {species: 'Slowbro', ability: 'Own Tempo'});
    }});
    try {
        const firstView = playerObservation(original, 'p1'), secondView = playerObservation(hidden, 'p1');
        assert.deepEqual(firstView, secondView);
        const settings = {...OPTIONS, candidates: 6, chance_samples: 2};
        const first = await searchOpening(firstView, settings), second = await searchOpening(secondView, settings);
        assert(first.supported, JSON.stringify(first));
        assert.equal(first.scope, 'observation_only_opening_search'); assert.equal(first.fixture_worlds, false);
        assert.equal(first.settings.proposal_mode, 'coverage');
        assert.equal(first.settings.response_proposal_mode, 'uniform');
        assert.deepEqual(second.candidates, first.candidates);
        assert.equal(second.selected_command, first.selected_command); assert.equal(second.nodes, first.nodes);
        await assert.rejects(searchOpening(capturePosition(original, 'fixture'), settings), /player observation/);
    } finally { original.destroy(); hidden.destroy(); }
});

test('malformed budgets and information-mismatched worlds fail before simulation', async () => {
    const battle = tacticalOpening();
    try {
        const view = playerObservation(battle, 'p1'), position = capturePosition(battle, 'fixture');
        const worlds = [{position, responses: [IDLE]}];
        await assert.rejects(compareOpeningTurns(view, worlds, [ELECTRIC], {...OPTIONS, max_nodes: 100000}), /bounded/);
        await assert.rejects(compareOpeningTurns(view, worlds, [ELECTRIC, ELECTRIC], OPTIONS), /distinct/);
        await assert.rejects(compareOpeningTurns(view, [{position, responses: ['move 9, move 1']}], [ELECTRIC], OPTIONS), /responses/);
        const changed = structuredClone(view); changed.request.side.pokemon[0].item = 'leftovers';
        await assert.rejects(compareOpeningTurns(changed, worlds, [ELECTRIC], OPTIONS), /same player observation/);
    } finally { battle.destroy(); }
});

test('uncertain switches are reported as world-conditional exclusions, not scored as successful', async () => {
    const battles = ['Sturdy', 'Magnet Pull'].map(ability => createFixture({changeTeams: teams => {
        Object.assign(teams[0][1], {species: 'Metagross', ability: 'Clear Body', moves: ['protect']});
        Object.assign(teams[1][0], {species: 'Magnezone', ability, moves: ['protect']});
        teams[1][1].moves = ['protect'];
    }}));
    try {
        const view = playerObservation(battles[0], 'p1');
        assert.deepEqual(playerObservation(battles[1], 'p1'), view);
        const worlds = battles.map(battle => ({position: capturePosition(battle, 'fixture'), responses: [IDLE]}));
        const safe = 'move 4, move 1', probe = 'move 4, switch 3';
        const result = await compareOpeningTurns(view, worlds, [probe, safe], OPTIONS);
        assert.equal(result.candidates[0].status, 'world_conditional_rejection');
        assert.equal(result.candidates[0].rejected_world, 1);
        assert.equal(result.candidates[0].score, null); assert.equal(result.selected_command, safe);

        // Do not silently remove a sampled opposing response that itself needs
        // a hidden-trap retry; the declared response experiment is unsupported.
        const opponentView = playerObservation(battles[1], 'p2');
        const rejected = await compareOpeningTurns(opponentView,
            [{position: worlds[1].position, responses: [probe]}], [IDLE], OPTIONS);
        assert.equal(rejected.supported, false); assert.equal(rejected.nodes, 0);
        assert.equal(rejected.reason, 'opponent_response_requires_retry');
    } finally { battles.forEach(battle => battle.destroy()); }
});
