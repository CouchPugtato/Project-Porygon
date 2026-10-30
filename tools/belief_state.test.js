'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {TeamValidator} = require('../external/pokemon-showdown/dist/sim/team-validator');
const {playerObservation} = require('./position_simulator');
const {createFixture, SELF_BUFFS} = require('./fixtures/position_simulator');
const {publicHealth, observeBeliefs, sampleRevealedSets} = require('./belief_state');

const SETTINGS = {seed: [323, 1, 2, 3], samples: 8};

function withView(run, options) {
    const battle = createFixture(options);
    try { return run(playerObservation(battle, 'p1'), battle); }
    finally { battle.destroy(); }
}

function extended(view, ...lines) { return {...view, log: [...view.log, ...lines]}; }
function snorlax(beliefs) { return beliefs.opponent.pokemon.find(p => p.species === 'snorlax'); }

test('beliefs retain only visible opponent identities and leave unrevealed teammates unknown', () => withView(view => {
    const beliefs = observeBeliefs(view);
    assert.equal(beliefs.opponent.team_size, 3);
    assert.equal(beliefs.opponent.unrevealed_count, 1);
    assert.deepEqual(beliefs.opponent.pokemon.map(p => p.species).sort(), ['bronzong', 'snorlax']);
    assert(!JSON.stringify(beliefs).includes('clefable'));
    assert.equal(snorlax(beliefs).ability, null);
    assert.equal(snorlax(beliefs).item, null);
    assert.deepEqual(snorlax(beliefs).moves, []);
    assert.equal(beliefs.world_ready, false);
}));

test('different hidden teams, sets, stats and future RNG yield identical beliefs and proposals', () => {
    const a = createFixture(), b = createFixture({seed: [600, 500, 400, 300], changeTeams: selected => {
        selected[1][0].ability = 'Thick Fat'; selected[1][0].item = 'Choice Band';
        selected[1][0].moves = ['bellydrum', 'rest', 'bodyslam', 'crunch'];
        selected[1][0].evs = {hp: 252, atk: 252, spe: 4};
        selected[1][2].species = 'Alakazam'; selected[1][2].ability = 'Synchronize';
    }});
    try {
        const av = playerObservation(a, 'p1'), bv = playerObservation(b, 'p1');
        assert.deepEqual(av, bv);
        const ab = observeBeliefs(av), bb = observeBeliefs(bv);
        assert.deepEqual(ab, bb);
        assert.deepEqual(sampleRevealedSets(ab, SETTINGS), sampleRevealedSets(bb, SETTINGS));
    } finally { a.destroy(); b.destroy(); }
});

test('ordinary moves and damage update facts without inventing exact HP or damage-to-stat inference', () => withView((view, battle) => {
    assert(battle.choose('p1', 'move 1 1, move 2'));
    assert(battle.choose('p2', SELF_BUFFS));
    const next = observeBeliefs(playerObservation(battle, 'p1')), p = snorlax(next);
    assert.deepEqual(p.moves, ['amnesia']);
    assert(p.health.upper < 1); assert(p.health.lower < p.health.upper);
    assert.equal(p.health_history.at(-1).event, '-damage');
    assert.equal(p.max_hp, undefined); assert.equal(p.evs, undefined);
    assert.notEqual(next.information_sha256, observeBeliefs(view).information_sha256);
}));

test('HP percentage bounds handle full health, fainting and the special 99 percent display', () => {
    assert.equal(publicHealth('100/100').lower, 1);
    assert.equal(publicHealth('100/100').upper, 1);
    assert.deepEqual([publicHealth('99/100').lower, publicHealth('99/100').upper,
        publicHealth('99/100').upper_inclusive], [.98, 1, false]);
    assert.equal(publicHealth('50/100 par').status, 'par');
    assert.equal(publicHealth('0 fnt').fainted, true);
    assert.equal(publicHealth('211/211'), null);
});

test('legal completions preserve revealed moves, ability, item and Tera and pass the validator', () => withView(view => {
    const beliefs = observeBeliefs(extended(view,
        '|move|p2a: Snorlax|Belly Drum|p2a: Snorlax',
        '|-ability|p2a: Snorlax|Thick Fat', '|-item|p2a: Snorlax|Leftovers',
        '|-terastallize|p2a: Snorlax|Ghost'));
    const before = JSON.stringify(beliefs), samples = sampleRevealedSets(beliefs, SETTINGS);
    assert.equal(samples.particles.length, 8);
    const validator = new TeamValidator('gen9randomdoublesbattle');
    for (const particle of samples.particles) for (const {key, set} of particle.sets) {
        assert.equal(validator.validateSet(structuredClone(set), {}), null);
        if (key === 'p2: Snorlax') {
            assert(set.moves.includes('bellydrum'));
            assert.equal(set.ability, 'Thick Fat'); assert.equal(set.item, 'Leftovers');
            assert.equal(set.teraType, 'Ghost');
        }
    }
    assert.equal(JSON.stringify(beliefs), before);
    assert.equal(samples.world_ready, false);
    assert(samples.unsupported.some(i => i.code === 'unrevealed_teammates_not_sampled'));
    assert.equal(samples.proposal, 'broad_rule_based_completions_not_a_calibrated_posterior');
}));

test('consumed items stay absent even when their effect subsequently identifies them', () => withView(view => {
    const beliefs = observeBeliefs(extended(view,
        '|-enditem|p2a: Snorlax|Sitrus Berry|[eat]',
        '|-heal|p2a: Snorlax|80/100|[from] item: Sitrus Berry'));
    assert.deepEqual(snorlax(beliefs).item, {current: '', last_revealed: 'sitrusberry'});
    for (const particle of sampleRevealedSets(beliefs, SETTINGS).particles)
        assert.equal(particle.sets.find(p => p.key === 'p2: Snorlax').set.item, '');
}));

test('effect-source ownership does not assign an opposing item to the damaged player', () => withView(view => {
    const beliefs = observeBeliefs(extended(view,
        '|-damage|p1a: Pikachu|180/211|[from] item: Rocky Helmet|[of] p2a: Snorlax'));
    assert.equal(snorlax(beliefs).item.current, 'rockyhelmet');
}));

test('tailwind screens hazards and side activation name a team rather than an unknown pokemon', () => withView(view => {
    const beliefs = observeBeliefs(extended(view,
        '|move|p2b: Bronzong|Reflect|p2b: Bronzong',
        '|-sidestart|p2: RandomPolicy|move: Reflect',
        '|-sidestart|p2: RandomPolicy|move: Tailwind',
        '|-sidestart|p2: RandomPolicy|Spikes',
        '|-sideend|p2: RandomPolicy|move: Tailwind',
        '|-sideactivate|p2: RandomPolicy|move: Wide Guard',
        '|-swapsideconditions|p2: RandomPolicy'));
    assert.deepEqual(beliefs.issues, []);
    assert.equal(beliefs.opponent.pokemon.length, 2);
    assert(!beliefs.opponent.pokemon.some(mon => mon.key === 'p2: RandomPolicy'));
    assert(beliefs.opponent.pokemon.find(mon => mon.species === 'bronzong').moves.includes('reflect'));
}));

test('unknown pokemon evidence and deceptive identities still require explicit abstention', () => withView(view => {
    const unknown = observeBeliefs(extended(view, '|move|p2a: Unknown|Tackle|p1a: Pikachu'));
    assert(unknown.issues.some(issue => issue.code === 'evidence_before_identity' && issue.key === 'p2: Unknown'));
    const disguise = observeBeliefs(extended(view, '|replace|p2a: Bronzong|Zoroark-Hisui, L80, M'));
    assert(disguise.issues.some(issue => issue.code === 'changed_identity_or_ability'));
}));

test('native reflect activation and expiry do not introduce a fictitious roster identity', () => withView((view, battle) => {
    assert(battle.choose('p1', 'move 4, move 2'));
    assert(battle.choose('p2', 'move 3, move 3'));
    const next = playerObservation(battle, 'p1');
    assert(next.log.some(line => line.startsWith('|-sidestart|p2: FixtureB|')));
    assert.deepEqual(observeBeliefs(next).issues, []);
    const ended = extended(next, '|-sideend|p2: FixtureB|Reflect');
    assert.deepEqual(observeBeliefs(ended).issues, []);
}, {changeTeams: teams => { teams[1][1].moves[2] = 'reflect'; }}));

test('new unusual legal evidence regenerates support instead of consulting a set catalog', () => withView(view => {
    const first = observeBeliefs(view), next = observeBeliefs(extended(view,
        '|move|p2a: Snorlax|Belly Drum|p2a: Snorlax'));
    assert.deepEqual(snorlax(first).moves, []);
    const sampled = sampleRevealedSets(next, SETTINGS);
    assert.equal(sampled.particles.length, SETTINGS.samples);
    assert(sampled.particles.every(p => p.sets.find(s => s.key === 'p2: Snorlax').set.moves.includes('bellydrum')));
    assert.deepEqual(Object.keys(require.cache).filter(file => /[\\/]random-battles[\\/]/.test(file)), []);
}));

test('called moves, disguise, changing abilities and incompatible evidence abstain explicitly', () => withView(view => {
    const cases = [
        ['|move|p2a: Snorlax|Thunderbolt|p1a: Pikachu|[from] move: Metronome', 'called_or_forced_move_evidence'],
        ['|replace|p2a: Snorlax|Zoroark, M', 'changed_identity_or_ability'],
        ['|-ability|p2a: Snorlax|Levitate|[from] move: Skill Swap', 'changed_ability'],
        ['|move|p2a: Snorlax|Not A Real Move|p1a: Pikachu', 'revealed_evidence_outside_native_legality'],
        ['|-damage|p2a: Snorlax|400/461', 'unsupported_public_hp_display'],
    ];
    for (const [line, code] of cases) {
        const sampled = sampleRevealedSets(observeBeliefs(extended(view, line)), SETTINGS);
        assert.equal(sampled.particles.length, 0);
        assert(sampled.unsupported.some(i => i.code === code), code);
    }
}));

test('switches track observed roster slots without inventing an unrevealed identity', () => withView(view => {
    const beliefs = observeBeliefs(extended(view, '|switch|p2a: Clefable|Clefable, M|100/100'));
    assert.equal(beliefs.opponent.active.a, 'p2: Clefable');
    assert.equal(beliefs.opponent.unrevealed_count, 0);
    assert.deepEqual(snorlax(beliefs).active_slots, []);
}));

test('visible level 50 and opposite-side projections work without importing format spreads', () => {
    const battle = createFixture({changeTeams: selected => { selected[1][0].level = 50; }});
    try {
        const beliefs = observeBeliefs(playerObservation(battle, 'p1'));
        assert.equal(snorlax(beliefs).level, 50);
        const sampled = sampleRevealedSets(beliefs, SETTINGS);
        assert.equal(sampled.particles.length, SETTINGS.samples);
        assert(sampled.particles.every(p => p.sets.find(s => s.key === 'p2: Snorlax').set.level === 50));
        const other = observeBeliefs(playerObservation(battle, 'p2'));
        assert.equal(other.opponent.side, 'p1');
        assert.deepEqual(other.opponent.pokemon.map(p => p.species).sort(), ['blastoise', 'pikachu']);
        assert(!JSON.stringify(other.opponent).includes('venusaur'));
    } finally { battle.destroy(); }
});

test('the observation boundary rejects privileged snapshots and sampling rejects unbounded requests', () => withView(view => {
    assert.throws(() => observeBeliefs({...view, state_json: '{}'}), /player observation/);
    assert.throws(() => observeBeliefs({...view, request: {side: {id: 'p2'}}}), /player observation/);
    const beliefs = observeBeliefs(view);
    for (const changes of [{samples: 1000}, {attempts_per_set: 1000}, {seed: [1, 2, 3, -1]}])
        assert.throws(() => sampleRevealedSets(beliefs, {...SETTINGS, ...changes}), /bounded/);
    assert.deepEqual(sampleRevealedSets(beliefs, SETTINGS), sampleRevealedSets(beliefs, SETTINGS));
    assert.notDeepEqual(sampleRevealedSets(beliefs, SETTINGS).particles,
        sampleRevealedSets(beliefs, {...SETTINGS, seed: [323, 4, 5, 6]}).particles);
}));
