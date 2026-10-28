'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {capturePosition, restorePosition, playerObservation, advancePosition} = require('./position_simulator');
const {REPLAY, SELF_BUFFS, createFixture, comparable, checkReplay, scenarios} = require('./fixtures/position_simulator');

for (const scenario of scenarios) test(`position replay: ${scenario.name}`, () => checkReplay(scenario));

test('snapshot, requests, logs, sets and PRNG state are independent of live/sibling mutations', () => {
    const live = createFixture(), siblings = [];
    try {
        const parent = capturePosition(live, 'fixture');
        const original = comparable(parent);
        const a = restorePosition(parent, REPLAY), b = restorePosition(parent, REPLAY);
        siblings.push(a, b);
        const view = playerObservation(a, 'p1');
        view.request.side.pokemon[0].condition = 'mutated'; view.log.push('mutated');
        a.p1.active[0].set.moves[0] = 'splash'; a.log.push('changed'); a.prng.random();
        live.p1.active[0].hp = 1; live.log.push('live changed');
        assert.deepEqual(comparable(capturePosition(b, 'fixture')), original);
        assert.deepEqual(comparable(parent), original);
        assert.notEqual(playerObservation(b, 'p1').request.side.pokemon[0].condition, 'mutated');
        assert(!playerObservation(b, 'p1').log.includes('mutated'));
    } finally { siblings.forEach(b => b.destroy()); live.destroy(); }
});

test('chance mode is explicit, and belief samples cannot replay a saved fixture RNG', () => {
    const live = createFixture();
    try {
        const parent = capturePosition(live, 'fixture');
        assert.throws(() => capturePosition(live), /origin/);
        assert.throws(() => restorePosition(parent), /choose fixture-replay/);
        assert.throws(() => restorePosition(capturePosition(live, 'belief-sample'), REPLAY), /restricted/);
        for (const seed of [undefined, [1], [1, 2, 3, -1], [1, 2, 3, 65536]])
            assert.throws(() => restorePosition(parent, {mode: 'sample', seed}), /seed/);
        const copied = restorePosition(parent, {mode: 'sample', seed: [10, 20, 30, 40]});
        try { assert.notEqual(copied.prng.getSeed(), live.prng.getSeed()); }
        finally { copied.destroy(); }
    } finally { live.destroy(); }
});

test('restoration preserves exact same-phase request updates rather than regenerating them', () => {
    const live = createFixture();
    let restored;
    try {
        live.p1.activeRequest.update = true;
        live.p1.activeRequest.active[0].maybeTrapped = true;
        const before = playerObservation(live, 'p1');
        restored = restorePosition(capturePosition(live, 'fixture'), REPLAY);
        assert.deepEqual(playerObservation(restored, 'p1'), before);
    } finally { restored?.destroy(); live.destroy(); }
});

test('invalid complete turns, required targets and trapped switches cannot mutate a parent', () => {
    const live = createFixture(), trapped = createFixture({changeTeams: selected => {
        selected[1][0].species = 'Dugtrio'; selected[1][0].ability = 'Arena Trap';
    }});
    try {
        const parent = capturePosition(live, 'fixture'), original = comparable(parent);
        for (const choices of [{p1: 'move 4, move 2'},
            {p1: 'move 4, move 2', p2: SELF_BUFFS, p3: 'pass'},
            {p1: 'move 1 1', p2: SELF_BUFFS},
            {p1: 'move 3, move 2', p2: SELF_BUFFS}])
            assert.throws(() => advancePosition(parent, choices, REPLAY));
        assert.deepEqual(comparable(parent), original);
        assert.deepEqual(comparable(capturePosition(live, 'fixture')), original);
        const trapParent = capturePosition(trapped, 'fixture');
        assert.throws(() => advancePosition(trapParent,
            {p1: 'switch 3, move 2', p2: 'move 2, move 2'}, REPLAY), /trapped/i);
        assert.deepEqual(comparable(capturePosition(trapped, 'fixture')), comparable(trapParent));
    } finally { live.destroy(); trapped.destroy(); }
});

test('player projections omit opposing requests, unrevealed reserves and pending choices', () => {
    const live = createFixture();
    try {
        const before = playerObservation(live, 'p1');
        assert(before.request.side.pokemon.some(p => p.details.includes('Venusaur')));
        assert(!JSON.stringify(before).includes('Clefable'));
        assert(!JSON.stringify(before).includes('Magic Guard'));
        assert(!JSON.stringify(before).includes('Iron Defense'));
        assert(live.choose('p2', SELF_BUFFS));
        assert.deepEqual(playerObservation(live, 'p1'), before);
        assert.throws(() => playerObservation(live, 'p3'), /side/);
    } finally { live.destroy(); }
});

test('sampling changes chance outcomes without changing the parent or using set catalogs', () => {
    const live = createFixture();
    try {
        const parent = capturePosition(live, 'fixture'), original = comparable(parent);
        const outcomes = new Set();
        for (let i = 0; i < 24; ++i) {
            const result = advancePosition(parent, {p1: 'move 2 1, move 2', p2: SELF_BUFFS},
                {mode: 'sample', seed: [322, i + 1, 1, 97]});
            outcomes.add(result.observations.p1.log.some(line => line.startsWith('|-miss|')));
        }
        assert.equal(outcomes.size, 2);
        assert.deepEqual(comparable(parent), original);
        const catalogs = Object.keys(require.cache).filter(file => /[\\/]random-battles[\\/]/.test(file));
        assert.deepEqual(catalogs, []);
    } finally { live.destroy(); }
});
