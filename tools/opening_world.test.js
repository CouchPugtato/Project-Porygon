'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {TeamValidator} = require('../external/pokemon-showdown/dist/sim/team-validator');
const {playerObservation, restorePosition, advancePosition} = require('./position_simulator');
const {createFixture, comparable, SELF_BUFFS, openingFixtureChoices, sixPokemon} = require('./fixtures/position_simulator');
const {constructOpeningWorld, recoverOwnSet} = require('./opening_world');

const SETTINGS = {seed: [324, 1, 2, 3], chance_seed: [324, 4, 5, 6], attempts: 32};
const SAMPLE = {mode: 'sample', seed: [324, 7, 8, 9]};

function withFixture(run, options) {
    const battle = createFixture(options);
    try { return run(playerObservation(battle, 'p1'), battle); }
    finally { battle.destroy(); }
}

test('opening world reconstructs own view exactly and samples a complete legal opponent roster', () => withFixture(view => {
    const before = structuredClone(view), world = constructOpeningWorld(view, SETTINGS);
    assert.equal(world.supported, true);
    assert.equal(world.simulation_ready, true); assert.equal(world.search_ready, false);
    assert.equal(world.position.origin, 'belief-sample');
    const battle = restorePosition(world.position, SAMPLE);
    try {
        assert.deepEqual(playerObservation(battle, 'p1'), view);
        assert.equal(battle.p2.pokemon.length, 3);
        assert.equal(battle.p2.active[0].species.id, 'snorlax');
        assert.equal(battle.p2.active[1].species.id, 'bronzong');
        assert.equal(battle.p2.pokemon[2].name, 'Unrevealed 1');
        const validator = new TeamValidator({...new TeamValidator('gen9randomdoublesbattle').format, team: undefined});
        assert.equal(validator.validateTeam(structuredClone(battle.p1.team)), null);
        assert.equal(validator.validateTeam(structuredClone(battle.p2.team)), null);
        assert.equal(battle.teamGenerator, null);
    } finally { battle.destroy(); }
    assert.deepEqual(view, before);
}));

test('six-Pokemon worlds include all unknown reserves without importing published teams', () => withFixture(view => {
    const world = constructOpeningWorld(view, SETTINGS);
    assert.equal(world.supported, true, JSON.stringify(world.unsupported));
    const battle = restorePosition(world.position, SAMPLE);
    try {
        assert.equal(battle.p1.pokemon.length, 6); assert.equal(battle.p2.pokemon.length, 6);
        assert.equal(world.opponent_hypotheses.filter(p => p.key.includes('Unrevealed')).length, 4);
        assert.equal(new Set(battle.p2.pokemon.map(p => p.species.baseSpecies)).size, 6);
        assert.deepEqual(playerObservation(battle, 'p1'), view);
    } finally { battle.destroy(); }
}, {changeTeams: sixPokemon}));

test('hidden team, stats, items, moves and live RNG cannot influence a constructed world', () => {
    const a = createFixture(), b = createFixture({seed: [600, 500, 400, 300], changeTeams: selected => {
        selected[1][0].ability = 'Thick Fat'; selected[1][0].item = 'Choice Band';
        selected[1][0].moves = ['bellydrum', 'rest', 'bodyslam', 'crunch'];
        selected[1][0].evs = {hp: 252, atk: 252, spe: 4};
        selected[1][2].species = 'Alakazam'; selected[1][2].ability = 'Synchronize';
    }});
    try {
        const av = playerObservation(a, 'p1'), bv = playerObservation(b, 'p1');
        assert.deepEqual(av, bv);
        const first = constructOpeningWorld(av, SETTINGS), second = constructOpeningWorld(bv, SETTINGS);
        assert(first.supported); assert(second.supported);
        assert.deepEqual(comparable(first.position), comparable(second.position));
        assert.deepEqual(first.opponent_hypotheses, second.opponent_hypotheses);
    } finally { a.destroy(); b.destroy(); }
});

test('own stat inversion handles non-neutral nature, varied IVs and invested spreads without knowing them', () => withFixture((view, live) => {
    const validator = new TeamValidator('gen9randomdoublesbattle');
    const entry = view.request.side.pokemon[0], recovered = recoverOwnSet(entry, validator);
    assert(recovered); assert.equal(validator.validateSet(structuredClone(recovered), {}), null);
    const world = constructOpeningWorld(view, SETTINGS);
    assert.equal(world.supported, true);
    const rebuilt = restorePosition(world.position, SAMPLE);
    try {
        assert.deepEqual(rebuilt.p1.active[0].baseStoredStats, live.p1.active[0].baseStoredStats);
        assert.equal(rebuilt.p1.active[0].maxhp, live.p1.active[0].maxhp);
        assert.deepEqual(playerObservation(rebuilt, 'p1'), view);
    } finally { rebuilt.destroy(); }
}, {changeTeams: selected => {
    Object.assign(selected[0][0], {nature: 'Adamant', level: 80,
        ivs: {hp: 30, atk: 31, def: 17, spa: 0, spd: 9, spe: 31},
        evs: {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85}});
}}));

test('startup weather is produced by simulation rather than patched from hidden snapshots', () => withFixture(view => {
    const world = constructOpeningWorld(view, SETTINGS); assert(world.supported);
    const battle = restorePosition(world.position, SAMPLE);
    try {
        assert.equal(battle.field.weather, 'sunnyday');
        assert.deepEqual(playerObservation(battle, 'p1'), view);
    } finally { battle.destroy(); }
}, {changeTeams: selected => { Object.assign(selected[0][0], {species: 'Ninetales',
    ability: 'Drought', moves: ['flamethrower', 'protect', 'solarbeam', 'willowisp']}); }}));

test('constructed worlds branch through the existing adapter without mutating their root', () => withFixture(view => {
    const world = constructOpeningWorld(view, SETTINGS); assert(world.supported);
    const original = comparable(world.position);
    const choices = openingFixtureChoices(world.position, SAMPLE.seed);
    const next = advancePosition(world.position, choices, SAMPLE);
    assert(next.turn > 1 || next.phase === 'switch' || next.ended);
    assert.deepEqual(comparable(world.position), original);
    assert.throws(() => restorePosition(world.position, {mode: 'fixture-replay'}), /restricted/);
}));

test('both perspectives reconstruct only their own private request', () => withFixture((_view, live) => {
    const view = playerObservation(live, 'p2'), world = constructOpeningWorld(view, SETTINGS);
    assert.equal(world.supported, true, JSON.stringify(world.unsupported));
    const battle = restorePosition(world.position, SAMPLE);
    try { assert.deepEqual(playerObservation(battle, 'p2'), view); }
    finally { battle.destroy(); }
}));

test('later turns, incompatible requests and impossible startup evidence abstain explicitly', () => withFixture((view, live) => {
    assert(live.choose('p1', 'move 4, move 2')); assert(live.choose('p2', SELF_BUFFS));
    const later = constructOpeningWorld(playerObservation(live, 'p1'), SETTINGS);
    assert.equal(later.supported, false); assert.equal(later.attempts, 0);
    assert(later.unsupported.some(i => i.code === 'opening_move_request_only'));
    const damaged = structuredClone(view); damaged.request.side.pokemon[0].condition = '1/211';
    assert.equal(constructOpeningWorld(damaged, SETTINGS).unsupported[0].code, 'own_request_not_reconstructable');
    const impossible = {...view, log: [...view.log, '|-weather|RainDance']};
    const rejected = constructOpeningWorld(impossible, {...SETTINGS, attempts: 2});
    assert.equal(rejected.supported, false); assert.equal(rejected.position, null);
    assert.equal(rejected.attempts, 2);
    assert(rejected.unsupported.some(i => i.code === 'opening_world_budget_exhausted'));
}));

test('world construction rejects privileged snapshot fields and unbounded work', () => withFixture(view => {
    assert.throws(() => constructOpeningWorld({...view, state_json: '{}'}, SETTINGS), /player observation/);
    for (const change of [{attempts: 1000}, {seed: [1]}, {chance_seed: [1, 2, 3, -1]}])
        assert.throws(() => constructOpeningWorld(view, {...SETTINGS, ...change}), /at most 32/);
    assert.deepEqual(Object.keys(require.cache).filter(file => /[\\/]random-battles[\\/]/.test(file)), []);
}));

