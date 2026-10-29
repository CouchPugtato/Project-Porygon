'use strict';

const {isDeepStrictEqual} = require('node:util');
const {Battle} = require('../external/pokemon-showdown/dist/sim/battle');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {TeamValidator} = require('../external/pokemon-showdown/dist/sim/team-validator');
const {observeBeliefs, sampleRevealedSets} = require('./belief_state');
const {FORMAT, capturePosition, playerObservation} = require('./position_simulator');
const {validateHypotheticalSet, validateHypotheticalTeam} = require('./hypothesis_legality');

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];

function detailsOf(details) {
    const parts = String(details).split(',').map(s => s.trim());
    return {species: parts[0], level: Number(parts.find(s => /^L\d+$/.test(s))?.slice(1) || 100),
        gender: parts.find(s => ['M', 'F'].includes(s)) || ''};
}

function recoverOwnSet(entry, validator) {
    const details = detailsOf(entry.details), dex = validator.dex, species = dex.species.get(details.species);
    const hp = /^(\d+)\/(\d+)$/.exec(entry.condition || '');
    if (!species.exists || !hp || hp[1] !== hp[2] || entry.commanding || entry.reviving ||
            entry.terastallized || entry.baseAbility !== entry.ability ||
            !entry.stats || STATS.slice(1).some(stat => !Number.isInteger(entry.stats[stat]))) return null;
    const set = {...details, name: entry.ident.slice(4), ability: entry.baseAbility,
        item: entry.item, pokeball: entry.pokeball, moves: [...entry.moves], teraType: entry.teraType};
    const context = {trunc: dex.trunc, dex, ruleTable: validator.ruleTable};
    const target = {hp: Number(hp[2]), ...entry.stats};
    const natures = ['Serious', ...dex.natures.all().map(n => n.name).filter(n => n !== 'Serious')];
    // visible final stats rather than the original investment tuple
    // one legal equivalent build without claiming recovery of the true build
    for (const nature of natures) {
        set.nature = nature;
        set.ivs = Object.fromEntries(STATS.map(stat => [stat, 0]));
        set.evs = Object.fromEntries(STATS.map(stat => [stat, 0]));
        let fits = true;
        for (const stat of STATS) {
            if (stat === 'hp' && species.maxHP) {
                if (species.maxHP !== target.hp) fits = false;
                continue;
            }
            let found = false;
            for (let effective = 0; effective <= 94; ++effective) {
                set.ivs[stat] = Math.min(31, effective);
                set.evs[stat] = Math.max(0, effective - 31) * 4;
                if (Battle.prototype.statModify.call(context, species.baseStats, set, stat) === target[stat]) {
                    found = true; break;
                }
            }
            if (!found) { fits = false; break; }
        }
        if (!fits || Object.values(set.evs).reduce((a, b) => a + b, 0) > 508) continue;
        ++set.evs[STATS.find(stat => set.evs[stat] < 252)];
        const candidate = structuredClone(set);
        if (!validateHypotheticalSet(candidate)) return candidate;
    }
    return null;
}

function openingUnsupported(view, beliefs) {
    const reasons = [...beliefs.issues];
    const issue = code => reasons.push({code, key: null});
    if (view.ended || view.turn !== 1 || view.phase !== 'move') issue('opening_move_request_only');
    const actions = new Set(['move', 'cant', 'drag', 'faint', 'swap', 'replace', 'detailschange']);
    if (view.log.some(line => actions.has(line.split('|')[1]))) issue('history_after_initial_entry');
    if (!view.request?.side?.pokemon || !Array.isArray(view.request.active) || view.request.active.length !== 2)
        issue('two_friendly_active_requests_required');
    const active = beliefs.opponent.active;
    if (!active.a || !active.b || active.a === active.b) issue('two_revealed_opponent_leads_required');
    if (beliefs.opponent.pokemon.length !== 2) issue('opening_revealed_roster_only');
    return reasons;
}

function buildOpeningWorld(view, {seed, chance_seed, attempts = 16, proposal_sets = null} = {}, moveEvidence) {
    const validSeed = s => Array.isArray(s) && s.length === 4 &&
        s.every(n => Number.isInteger(n) && n >= 0 && n <= 65535);
    if (!validSeed(seed) || !validSeed(chance_seed) || !Number.isInteger(attempts) || attempts < 1 || attempts > 32)
        throw new Error('opening worlds require independent proposal/chance seeds and at most 32 attempts');
    const beliefs = observeBeliefs(view);
    const openingIssues = openingUnsupported(view, beliefs);
    if (moveEvidence) {
        for (const later of moveEvidence.opponent.pokemon) {
            const known = beliefs.opponent.pokemon.find(mon => mon.key === later.key);
            if (known) {
                known.moves = [...new Set([...known.moves, ...later.moves])].sort();
                // revealed tera type constrains replay proposals rather than earlier decisions
                known.tera_type = later.tera_type;
                known.ability = later.ability || known.ability;
                if (later.item) known.item = {current: later.item.last_revealed, last_revealed: later.item.last_revealed};
            }
            else {
                // later revealed reserves present from the opening in replay worlds
                // sampled unrevealed builds rather than copied hidden sets
                beliefs.opponent.pokemon.push(structuredClone(later));
            }
        }
        beliefs.opponent.unrevealed_count = Math.max(0,
            beliefs.opponent.team_size - beliefs.opponent.pokemon.length);
    }
    const result = {version: 1, kind: 'opening_hypothetical_world', scope: 'initial_move_request',
        information_sha256: beliefs.information_sha256, proposal_seed: [...seed], chance_seed: [...chance_seed],
        supported: false, simulation_ready: false, search_ready: false, training_eligible: false,
        position: null, attempts: 0, rejected: {}, unsupported: openingIssues,
        assumptions: ['Rule-based roster proposals are not calibrated hidden-state probabilities.',
            'Unknown reserves use legal species and levels, not random-generator frequencies.',
            'Own stats use one legal equivalent IV/EV/nature tuple, not a recovered true build.',
            'Opening startup is simulated; exact player projection must match before acceptance.']};
    if (result.unsupported.length) return result;
    const setValidator = new TeamValidator(FORMAT);
    const own = [];
    for (const entry of view.request.side.pokemon) {
        const set = recoverOwnSet(entry, setValidator);
        if (!set) {
            result.unsupported.push({code: 'own_request_not_reconstructable', key: entry.ident}); return result;
        }
        own.push(set);
    }
    const names = {};
    for (const line of view.log) {
        const p = line.split('|');
        if (p[1] === 'player' && ['p1', 'p2'].includes(p[2])) names[p[2]] = {name: p[3], avatar: p[4] || ''};
    }
    if (!names.p1 || !names.p2) {
        result.unsupported.push({code: 'player_metadata_missing', key: null}); return result;
    }
    const proposalRng = new PRNG(seed.join(',')), chanceRng = new PRNG(chance_seed.join(','));
    const nextSeed = rng => Array.from({length: 4}, () => rng.random(65536));
    const reject = reason => { result.rejected[reason] = (result.rejected[reason] || 0) + 1; };
    for (let i = 0; i < attempts; ++i) {
        ++result.attempts;
        const sampled = proposal_sets ? {particles: [{sets: structuredClone(proposal_sets)}]} : sampleRevealedSets(beliefs,
            {seed: nextSeed(proposalRng), samples: 1, attempts_per_set: 16, include_unrevealed: true});
        if (!sampled.particles.length) { reject('roster_proposal_failed'); continue; }
        const hypotheses = sampled.particles[0].sets;
        if (beliefs.opponent.pokemon.some(mon => !hypotheses.some(h => h.key === mon.key &&
                mon.moves.every(move => h.set.moves.includes(move)) &&
                (!mon.tera_type || h.set.teraType.toLowerCase() === mon.tera_type)))) {
            reject('proposal_evidence_conflict'); continue;
        }
        const order = [beliefs.opponent.active.a, beliefs.opponent.active.b,
            ...hypotheses.map(p => p.key).filter(key => !Object.values(beliefs.opponent.active).includes(key))];
        const opponent = order.map(key => {
            const hypothesis = hypotheses.find(p => p.key === key);
            return {...structuredClone(hypothesis.set), name: key.slice(4)};
        });
        const teams = view.side === 'p1' ? [structuredClone(own), opponent] : [opponent, structuredClone(own)];
        if (teams.some(team => team.some(set => validateHypotheticalSet(set)) || validateHypotheticalTeam(team))) {
            reject('team_legality'); continue;
        }
        let battle;
        try {
            battle = new Battle({formatid: FORMAT, seed: nextSeed(chanceRng).join(','), strictChoices: true,
                p1: {...names.p1, team: teams[0]}, p2: {...names.p2, team: teams[1]}});
            if (!isDeepStrictEqual(playerObservation(battle, view.side), view)) {
                reject('startup_projection_mismatch'); continue;
            }
            result.position = capturePosition(battle, 'belief-sample');
            result.supported = true; result.simulation_ready = true;
            result.opponent_hypotheses = hypotheses;
            return result;
        } finally { battle?.destroy(); }
    }
    result.unsupported.push({code: 'opening_world_budget_exhausted', key: null});
    return result;
}

function constructOpeningWorld(view, options) {
    return buildOpeningWorld(view, options);
}

function constructReplayOpening(initialView, latestView, options) {
    const latest = observeBeliefs(latestView);
    observeBeliefs(initialView);
    if (latestView.side !== initialView.side || latestView.turn < initialView.turn || latest.issues.length ||
            latestView.turn === initialView.turn && latestView.log.length <= initialView.log.length ||
            initialView.log.some((line, i) => latestView.log[i] !== line))
        throw new Error('replay opening requires later evidence extending the initial player observation');
    // past revealed moves used only for reconstruction at the later decision
    // unchanged opening actor interface without access to future evidence
    const result = buildOpeningWorld(initialView, options, latest);
    return {...result, scope: 'opening_conditioned_on_later_replay_evidence',
        evidence_information_sha256: latest.information_sha256};
}

module.exports = {recoverOwnSet, constructOpeningWorld, constructReplayOpening};
