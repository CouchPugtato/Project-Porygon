'use strict';

const {performance} = require('node:perf_hooks');
const {Dex} = require('../external/pokemon-showdown/dist/sim/dex');
const {restorePosition, advancePosition} = require('./position_simulator');
const {constructReplayOpening} = require('./opening_world');
const {firstDamageFeasibility} = require('./planner_damage_feasibility');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {nextSeed, choiceFailure} = require('./planner_beliefs');
const {isDeepStrictEqual} = require('node:util');

function defensiveGrid(sets, targetKey, defense) {
    const proposals = [];
    for (const hp of [0, 31, 63, 94]) for (const defensive of [0, 31, 63, 94]) {
        const proposal = structuredClone(sets), target = proposal.find(entry => entry.key === targetKey);
        if (!target) throw new Error('damage target missing from the sampled opposing roster');
        // bounded effective investment grid rather than an assumed competitive build
        // complete replay must still validate speed and every other observed effect
        target.set.evs = {hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0};
        for (const [stat, effective] of [['hp', hp], [defense, defensive]]) {
            target.set.ivs[stat] = Math.min(31, effective);
            target.set.evs[stat] = Math.max(0, effective - 31) * 4;
        }
        if (Object.values(target.set.evs).every(value => value === 0)) target.set.evs.atk = 1;
        proposals.push(proposal);
    }
    return proposals;
}

async function auditDefensiveProposals(failure, gate) {
    const trace = failure.simulation_trace, initial = failure.history.observations[0];
    if (!trace || failure.history.observations.length !== 2 || initial.turn !== 1)
        return {supported: false, reason: 'single_opening_transition_required'};
    const baseline = firstDamageFeasibility(trace), hit = baseline.outcomes?.[0];
    const opponent = initial.side === 'p1' ? 'p2' : 'p1';
    if (!baseline.supported || !baseline.context_matched || !hit?.target.startsWith(opponent))
        return {supported: false, reason: 'observed_first_hit_on_opposing_pokemon_required'};
    const move = Dex.moves.get(hit.move);
    if (!['Physical', 'Special'].includes(move.category) || move.overrideDefensiveStat || move.overrideDefensivePokemon)
        return {supported: false, reason: 'ordinary_defensive_stat_relationship_required'};
    const branch = restorePosition(trace.position, {mode: 'sample', seed: trace.seed});
    let sets;
    try { sets = branch.getSide(opponent).pokemon.map(mon => ({key: mon.fullname, set: structuredClone(mon.set)})); }
    finally { branch.destroy(); }
    const key = hit.target.replace(/^(p[12])[ab]: /, '$1: ');
    const proposals = defensiveGrid(sets, key, move.category === 'Physical' ? 'def' : 'spd');
    const started = performance.now(), rng = new PRNG('348,1,2,3');
    const report = {supported: true, calibrated: false, proposals: 0, startup_matches: 0,
        feasible_first_hits: 0, full_replay_samples: 0, exact_transition_matches: 0,
        budget_exhausted: false, training_eligible: false, strength_claim: false};
    for (const proposal of proposals) {
        await (gate?.waitForCapacity?.() ?? gate?.waitForLaunch?.());
        if (performance.now() - started >= 5000) { report.budget_exhausted = true; break; }
        ++report.proposals;
        const opening = constructReplayOpening(initial, trace.expected, {seed: nextSeed(rng),
            chance_seed: nextSeed(rng), attempts: 1, proposal_sets: proposal});
        if (!opening.supported) continue;
        ++report.startup_matches;
        const candidate = {...trace, position: opening.position};
        const feasibility = firstDamageFeasibility(candidate);
        if (feasibility.feasible_in_captured_context !== true) continue;
        ++report.feasible_first_hits;
        for (let sample = 0; sample < 8; ++sample) {
            if (performance.now() - started >= 5000) { report.budget_exhausted = true; break; }
            ++report.full_replay_samples;
            try {
                const next = advancePosition(opening.position, trace.choices, {mode: 'sample', seed: nextSeed(rng)});
                if (isDeepStrictEqual(next.observations[initial.side], trace.expected)) ++report.exact_transition_matches;
            } catch (error) { if (!choiceFailure(error)) throw error; }
        }
    }
    report.elapsed_seconds = (performance.now() - started) / 1000;
    report.validation_gate_passed = report.exact_transition_matches > 0;
    report.assessment = report.validation_gate_passed ? 'complete_transition_found_in_bounded_proposal_grid' :
        'first_hit_constraints_did_not_produce_a_verified_complete_transition';
    return report;
}

module.exports = {defensiveGrid, auditDefensiveProposals};
