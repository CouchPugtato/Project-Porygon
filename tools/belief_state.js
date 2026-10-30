'use strict';

const crypto = require('node:crypto');
const {TeamValidator} = require('../external/pokemon-showdown/dist/sim/team-validator');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {toID} = require('../external/pokemon-showdown/dist/sim/dex');
const {validateHypotheticalSet} = require('./hypothesis_legality');

const FORMAT = 'gen9randomdoublesbattle';
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const CHANGED_MOVES = new Set(['transform', 'mimic', 'sketch', 'metronome', 'copycat',
    'mirrormove', 'naturepower', 'sleeptalk', 'instruct', 'assist', 'mefirst']);
const SIDE_EVENTS = new Set(['-sidestart', '-sideend', '-sideactivate', '-swapsideconditions']);

function identity(text, opponent) {
    const match = /^(p[12])([ab])?: (.+)$/.exec(text || '');
    return match && match[1] === opponent ? {key: `${opponent}: ${match[3]}`,
        slot: match[2] || null} : null;
}

function publicHealth(condition) {
    if (condition === '0 fnt') return {display: condition, lower: 0, upper: 0,
        lower_inclusive: true, upper_inclusive: true, fainted: true};
    const match = /^(\d+)\/100(?: (brn|par|slp|frz|psn|tox))?$/.exec(condition || '');
    if (!match || Number(match[1]) < 1 || Number(match[1]) > 100) return null;
    const percent = Number(match[1]);
    // rounded up percentages except a damaged pokemon cannot display full health
    return {display: condition, lower: percent === 100 ? 1 : (percent - 1) / 100,
        upper: percent >= 99 ? 1 : percent / 100, lower_inclusive: percent === 100,
        upper_inclusive: percent !== 99, fainted: false, status: match[2] || ''};
}

function observeBeliefs(view) {
    const keys = ['side', 'turn', 'phase', 'ended', 'winner', 'request', 'log'];
    if (!view || Object.keys(view).length !== keys.length || keys.some(k => !Object.hasOwn(view, k)) ||
            !['p1', 'p2'].includes(view.side) || !Array.isArray(view.log) ||
            view.log.some(line => typeof line !== 'string') ||
            (view.request && view.request.side?.id !== view.side))
        throw new Error('belief input must be a player observation, not a simulator snapshot');
    const opponent = view.side === 'p1' ? 'p2' : 'p1';
    const pokemon = new Map(), active = {a: null, b: null}, issues = [];
    let teamSize = null;
    const issue = (code, key = null) => {
        if (!issues.some(i => i.code === code && i.key === key)) issues.push({code, key});
    };
    for (const line of view.log) {
        const parts = line.split('|'), event = parts[1];
        // team conditions name the player rather than an active pokemon
        // source ownership still follows an explicit opposing pokemon when present
        const who = SIDE_EVENTS.has(event) ? null : identity(parts[2], opponent);
        if (event === 'teamsize' && parts[2] === opponent) {
            const size = Number(parts[3]);
            if (Number.isInteger(size) && size >= 2 && size <= 6) teamSize = size;
            else issue('unsupported_team_size');
        }
        if (['switch', 'drag'].includes(event) && who) {
            const details = (parts[3] || '').split(',').map(s => s.trim());
            const species = toID(details[0]), levelPart = details.find(s => /^L\d+$/.test(s));
            if (!pokemon.has(who.key)) pokemon.set(who.key, {key: who.key, species,
                level: levelPart ? Number(levelPart.slice(1)) : 100,
                gender: details.find(s => ['M', 'F'].includes(s)) || '',
                moves: [], used_moves: [], ability: null, item: null, tera_type: null,
                health: null, health_history: [], status: '', fainted: false});
            const known = pokemon.get(who.key);
            if (known.species !== species) issue('ambiguous_identity_or_disguise', who.key);
            if (Object.keys(active).some(slot => slot !== who.slot && active[slot] === who.key))
                issue('ambiguous_roster_identity', who.key);
            if (who.slot) active[who.slot] = who.key;
        }
        const source = parts.find(p => /^\[from\] (?:item|ability): /.test(p));
        const sourceOwner = parts.find(p => p.startsWith('[of] '));
        const owner = sourceOwner ? identity(sourceOwner.slice(5), opponent) : who;
        if (source && owner && pokemon.has(owner.key)) {
            const knownOwner = pokemon.get(owner.key), [, kind, name] = /^\[from\] (item|ability): (.+)$/.exec(source);
            if (kind === 'ability') knownOwner.ability = toID(name);
            else if (!(knownOwner.item?.current === '' && knownOwner.item.last_revealed === toID(name)))
                knownOwner.item = {current: toID(name), last_revealed: toID(name)};
        }
        if (!who) continue;
        const known = pokemon.get(who.key);
        if (!known) { issue('evidence_before_identity', who.key); continue; }
        if (['switch', 'drag', '-damage', '-heal'].includes(event)) {
            const health = publicHealth(parts[['switch', 'drag'].includes(event) ? 4 : 3]);
            if (!health) issue('unsupported_public_hp_display', who.key);
            else {
                known.health = health; known.health_history.push({event, ...health});
                known.fainted = health.fainted; known.status = health.status || '';
            }
        }
        if (event === 'move') {
            const move = toID(parts[3]);
            if (!known.used_moves.includes(move)) known.used_moves.push(move);
            if (parts.some(p => p.startsWith('[from]')) || move === 'struggle') {
                issue('called_or_forced_move_evidence', who.key);
            } else if (!known.moves.includes(move)) known.moves.push(move);
            if (CHANGED_MOVES.has(move)) issue('changed_or_indirect_move_slots', who.key);
            if (known.moves.length > 4) issue('more_than_four_observed_move_slots', who.key);
        }
        if (event === '-ability') {
            known.ability = toID(parts[3]);
            if (parts.some(p => p.startsWith('[from]')))
                issue('changed_ability', who.key);
        }
        if (event === '-activate' && (parts[3] || '').startsWith('ability: '))
            known.ability = toID(parts[3].slice(9));
        if (event === '-activate' && (parts[3] || '').startsWith('item: '))
            known.item = {current: toID(parts[3].slice(6)), last_revealed: toID(parts[3].slice(6))};
        if (event === '-item') known.item = {current: toID(parts[3]), last_revealed: toID(parts[3])};
        if (event === '-enditem') known.item = {current: '', last_revealed: toID(parts[3])};
        if (event === '-terastallize') known.tera_type = toID(parts[3]);
        if (event === '-status') known.status = parts[3];
        if (event === '-curestatus') known.status = '';
        if (event === 'faint') {
            known.fainted = true; known.health = publicHealth('0 fnt');
        }
        if (['replace', 'detailschange', '-formechange', '-transform', '-endability', 'swap'].includes(event))
            issue('changed_identity_or_ability', who.key);
    }
    const revealed = [...pokemon.values()].sort((a, b) => a.key.localeCompare(b.key));
    for (const p of revealed) {
        p.moves.sort(); p.used_moves.sort();
        p.active_slots = Object.keys(active).filter(slot => active[slot] === p.key);
    }
    if (new Set(revealed.map(p => p.species)).size !== revealed.length)
        issue('revealed_species_clause_conflict');
    if (teamSize === null) issue('team_size_not_observed');
    if (teamSize !== null && teamSize < revealed.length) issue('ambiguous_roster_identity');
    return {version: 1, kind: 'observation_beliefs', format: FORMAT, perspective: view.side,
        turn: view.turn, phase: view.phase, ended: view.ended,
        information_sha256: crypto.createHash('sha256').update(JSON.stringify(view)).digest('hex'),
        opponent: {side: opponent, team_size: teamSize, active, pokemon: revealed,
            unrevealed_count: teamSize === null ? null : Math.max(0, teamSize - revealed.length)},
        issues, world_ready: false, training_eligible: false,
        assumptions: ['Revealed identity may be deceptive until resolved.',
            'HP displays are fraction intervals, not exact opponent HP or stat estimates.',
            'This records visible evidence; it does not reconstruct all field/volatile state.']};
}

function sampleStats(rng) {
    const ivs = Object.fromEntries(STATS.map(stat => [stat, rng.random(32)]));
    // some native acquisition rules require several perfect inherited stats
    // mix those legal regions with unrestricted proposals without a set catalog
    const region = rng.random(4), perfect = [...STATS];
    for (let i = 0; i < (region === 0 ? 6 : region === 1 ? 3 : 0); ++i) {
        const index = rng.random(perfect.length); ivs[perfect.splice(index, 1)[0]] = 31;
    }
    const evs = Object.fromEntries(STATS.map(stat => [stat, 0]));
    const budget = rng.random(128);
    for (let i = 0; i < budget; ++i) {
        const available = STATS.filter(stat => evs[stat] < 252);
        evs[available[rng.random(available.length)]] += 4;
    }
    // odd investment to satisfy native intent checks without changing stats
    // no assumption about investment frequencies in generated teams
    ++evs[STATS.find(stat => evs[stat] < 252)];
    return {ivs, evs};
}

function sampleRevealedSets(beliefs, {seed, samples = 16, attempts_per_set = 32,
    include_unrevealed = false} = {}) {
    if (beliefs?.version !== 1 || beliefs.kind !== 'observation_beliefs' || beliefs.format !== FORMAT)
        throw new Error('expected observation beliefs');
    if (!Array.isArray(seed) || seed.length !== 4 ||
            seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535) ||
            !Number.isInteger(samples) || samples < 1 || samples > 64 ||
            !Number.isInteger(attempts_per_set) || attempts_per_set < 1 || attempts_per_set > 64 ||
            typeof include_unrevealed !== 'boolean')
        throw new Error('belief sampling requires an independent seed and bounded sample counts');
    const result = {version: 1, kind: include_unrevealed ? 'opponent_roster_hypotheses' : 'revealed_set_hypotheses',
        information_sha256: beliefs.information_sha256, seed: [...seed], samples_requested: samples,
        proposal: 'broad_rule_based_completions_not_a_calibrated_posterior',
        world_ready: false, training_eligible: false, particles: [], attempts: 0,
        unsupported: [...beliefs.issues], assumptions: [
            'Unknown moves, abilities, items, Tera, natures and stats use declared rule-based sampling.',
            include_unrevealed ? 'Unrevealed species and levels use broad legal proposals, not generator frequencies.' :
                'Unrevealed teammates are not sampled.',
            'No move-slot ordering, PP, exact current HP or dynamic-state reconstruction.',
            'No damage-to-stat or opponent-choice likelihood model; health evidence is retained only.']};
    if (beliefs.ended) result.unsupported.push({code: 'terminal_observation', key: null});
    if (!beliefs.opponent.pokemon.length) result.unsupported.push({code: 'no_revealed_pokemon', key: null});
    if (result.unsupported.length) return result;
    const validator = new TeamValidator(FORMAT), dex = validator.dex;
    const rng = new PRNG(seed.join(','));
    const pick = values => values[rng.random(values.length)];
    const items = ['', ...dex.items.all().filter(i => i.exists && !i.isNonstandard && i.gen <= 9)
        .map(i => i.id)].sort();
    const natures = dex.natures.all().map(n => n.name).sort();
    const types = dex.types.all().map(t => t.name).sort();
    const legalSpecies = include_unrevealed ? dex.species.all().filter(s => s.exists &&
        !s.isNonstandard && s.gen <= 9 && !s.battleOnly).sort((a, b) => a.id.localeCompare(b.id)) : [];
    function poolFor(evidence) {
        const species = dex.species.get(evidence.species);
        if (!species.exists || species.isNonstandard || species.gen > 9) {
            result.unsupported.push({code: 'unsupported_species', key: evidence.key}); return null;
        }
        const moves = [...dex.species.getMovePool(species.id)].filter(id => {
            const move = dex.moves.get(id); return move.exists && !move.isNonstandard && move.gen <= 9;
        }).sort();
        const abilities = [...new Set(Object.values(species.abilities).map(toID))].sort();
        if (evidence.moves.some(id => !moves.includes(id)) || evidence.moves.length > 4 ||
                (evidence.ability !== null && !abilities.includes(evidence.ability))) {
            result.unsupported.push({code: 'revealed_evidence_outside_native_legality', key: evidence.key}); return null;
        }
        if (!moves.length) result.unsupported.push({code: 'empty_legal_movepool', key: evidence.key});
        return {evidence, species, moves, abilities};
    }
    const pools = beliefs.opponent.pokemon.map(poolFor);
    if (result.unsupported.length) return result;
    for (let i = 0; i < samples; ++i) {
        const sets = [], selectedPools = [...pools];
        const usedSpecies = new Set(pools.map(p => p.species.baseSpecies));
        const usedKeys = new Set(pools.map(p => p.evidence.key));
        let placeholder = 1;
        for (let j = 0; include_unrevealed && j < beliefs.opponent.unrevealed_count; ++j) {
            const candidates = legalSpecies.filter(s => !usedSpecies.has(s.baseSpecies));
            if (!candidates.length) {
                result.particles = [];
                result.unsupported.push({code: 'unrevealed_species_pool_exhausted', key: null});
                return result;
            }
            const species = pick(candidates); usedSpecies.add(species.baseSpecies);
            // synthetic reserve identities distinct from observed nicknames
            while (usedKeys.has(`${beliefs.opponent.side}: Unrevealed ${placeholder}`)) ++placeholder;
            const key = `${beliefs.opponent.side}: Unrevealed ${placeholder++}`;
            usedKeys.add(key);
            const evidence = {key,
                species: species.id, level: rng.random(1, 101), gender: '', moves: [],
                ability: null, item: null, tera_type: null};
            selectedPools.push(poolFor(evidence));
        }
        if (result.unsupported.length) { result.particles = []; return result; }
        for (const {evidence, species, moves: pool, abilities} of selectedPools) {
            let accepted = null;
            for (let attempt = 0; attempt < attempts_per_set; ++attempt) {
                ++result.attempts;
                const moves = [...evidence.moves], available = pool.filter(id => !moves.includes(id));
                while (moves.length < Math.min(4, pool.length)) {
                    const index = rng.random(available.length); moves.push(available.splice(index, 1)[0]);
                }
                const set = {species: species.name, level: evidence.level,
                    gender: evidence.gender || species.gender || pick(['M', 'F']),
                    ability: evidence.ability || pick(abilities),
                    item: evidence.item === null ? pick(items) : evidence.item.current,
                    teraType: evidence.tera_type || pick(types), moves,
                    nature: pick(natures), ...sampleStats(rng)};
                const problems = validateHypotheticalSet(set);
                if (!problems) { accepted = set; break; }
            }
            if (!accepted) {
                result.particles = [];
                result.unsupported.push({code: 'legal_completion_budget_exhausted', key: evidence.key});
                return result;
            }
            sets.push({key: evidence.key, set: accepted});
        }
        result.particles.push({weight: 1 / samples, sets});
    }
    if (!include_unrevealed && beliefs.opponent.unrevealed_count)
        result.unsupported.push({code: 'unrevealed_teammates_not_sampled', key: null});
    return result;
}

module.exports = {publicHealth, observeBeliefs, sampleRevealedSets};
