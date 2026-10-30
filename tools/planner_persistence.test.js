'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, playerObservation, advancePosition} = require('./position_simulator');
const {TrajectoryBank, PersistentRecovery} = require('./planner_persistence');
const {compareCoverage} = require('./belief_coverage_audit');

const SEED = [375, 1, 2, 3];

function controls(battle) {
    const before = playerObservation(battle, 'p1'), opening = capturePosition(battle, 'fixture');
    const choices = {p1: 'move 4, move 2', p2: 'move 3, move 3'};
    const next = advancePosition(opening, choices, {mode: 'sample', seed: SEED});
    return {before, opening, choices, next};
}

test('trajectory bank preserves diverse detached roots within explicit memory and entry limits', () => {
    const a = createHybridFixture(), b = createHybridFixture({changeTeams: teams => { teams[1][0].evs.hp = 125; }});
    try {
        const bank = new TrajectoryBank(2), records = [a, b].map(battle => controls(battle));
        for (const record of records) {
            const path = {opening: record.opening, steps: [{seed: [...SEED], choices: record.choices}]};
            bank.remember(new Map([[record.next.position.state_json, path]]), [record.next.position],
                {observations: [record.before, record.next.observations.p1]});
            path.steps[0].seed[0] = 999;
            assert.notEqual(bank.entries.find(entry => entry.position === record.next.position).path.steps[0].seed[0], 999);
        }
        assert.equal(bank.entries.length, 2); assert(bank.bytes <= bank.maxBytes);
        const tiny = new TrajectoryBank(1, 1), record = records[0];
        tiny.remember(new Map([[record.opening.state_json, {opening: record.opening, steps: []}]]), [record.opening],
            {observations: [record.before]});
        assert.equal(tiny.entries.length, 0);
    } finally { a.destroy(); b.destroy(); }
});

test('chance recovery advances a verified parent without changing its build or replaying old turns', async () => {
    const battle = createHybridFixture();
    try {
        const {before, opening, choices, next} = controls(battle);
        const recovery = new PersistentRecovery({seed: SEED, particles: 1, max_elapsed_ms: 500, max_total_ms: 1000});
        const beliefs = {paths: new Map([[opening.state_json, {opening, steps: []}]]), worlds: [opening],
            history: {observations: [before], own_choices: []}};
        recovery.remember(beliefs);
        beliefs.history = {observations: [before, next.observations.p1], own_choices: [choices.p1]};
        const original = opening.state_json;
        const report = await recovery.recover(beliefs, next.observations.p1, choices.p1);
        assert(report.verified); assert.equal(report.stage, 'chance'); assert.equal(report.fit_nodes, 0);
        assert(report.chance_nodes > 0); assert.equal(report.witnesses[0].opening.state_json, original);
        assert.equal(opening.state_json, original); assert.equal(report.witnesses[0].steps.length, 1);
        assert.equal(await recovery.recover(beliefs, next.observations.p1, choices.p1), null);
    } finally { battle.destroy(); }
});

test('a forged latest request never becomes a recovered world and total recovery remains bounded', async () => {
    const battle = createHybridFixture();
    try {
        const {before, opening, choices, next} = controls(battle), expected = structuredClone(next.observations.p1);
        expected.request.side.pokemon[0].condition = '999/999';
        const recovery = new PersistentRecovery({seed: SEED, max_elapsed_ms: 25, max_total_ms: 25});
        const beliefs = {history: {observations: [before, expected], own_choices: [choices.p1]},
            lastFailure: {simulation_trace: {position: opening}}};
        const report = await recovery.recover(beliefs, expected, choices.p1);
        assert.equal(report.verified, false); assert.equal(report.witnesses.length, 0);
        assert(report.chance_nodes <= 128); assert.equal(recovery.bank.entries.length, 1);
    } finally { battle.destroy(); }
});

test('coverage advancement requires enough games broader coverage stable legality and comparable runtime', () => {
    const game = (profile, turn, searched, seconds) => ({profile, status: 'completed', elapsed_seconds: seconds,
        choice_rejections: [], hybrid_coverage: {decisions: 10, search_completed: searched,
            latest_verified_turn: turn, reconstruction_results: []}});
    const games = Array.from({length: 8}, () => game('guarded', 2, 1, 10))
        .concat(Array.from({length: 8}, () => game('persistent', 4, 3, 10)));
    assert(compareCoverage(games).advancement_gate_passed);
    games.at(-1).choice_rejections.push({message: 'illegal'});
    assert.equal(compareCoverage(games).advancement_gate_passed, false);
    assert.equal(compareCoverage(games.slice(0, 10)).enough_screen_games, false);
});

test('persistent recovery abstains at the opening and refuses detached later boundaries', async () => {
    const battle = createHybridFixture();
    try {
        const {before, choices, next} = controls(battle), recovery = new PersistentRecovery({seed: SEED});
        assert.equal(await recovery.recover({history: {observations: [before], own_choices: []}}, before, null), null);
        const history = {observations: [before, next.observations.p1], own_choices: [choices.p1]};
        await assert.rejects(recovery.recover({history}, next.observations.p1, 'move 1 1, move 2'), /recorded boundary/);
    } finally { battle.destroy(); }
});
