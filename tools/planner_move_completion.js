'use strict';

const {isDeepStrictEqual} = require('node:util');
const {toID} = require('../external/pokemon-showdown/dist/sim/dex');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {observeBeliefs} = require('./belief_state');
const {restorePosition, advancePosition, playerObservation} = require('./position_simulator');
const {constructReplayOpening} = require('./opening_world');
const {observedResponses, nextSeed, choiceFailure} = require('./planner_beliefs');
const {UnsupportedPlannerRequest} = require('./planner_actions');

function moveCompletions(sets, facts) {
    let changed = false;
    const proposal = structuredClone(sets), replacements = [];
    for (const known of facts.opponent.pokemon) {
        const entry = proposal.find(mon => mon.key === known.key);
        if (!entry) return {supported: false, reason: 'new_reserve_requires_roster_completion', proposals: []};
        const moves = entry.set.moves.map(toID);
        if (known.moves.every(move => moves.includes(move))) continue;
        if (known.moves.length > 4) return {supported: false, reason: 'too_many_observed_moves', proposals: []};
        // retain every observed move and replace only unobserved hypothetical slots
        // the entire earlier history must survive the changed opening before adoption
        const missing = known.moves.filter(move => !moves.includes(move));
        const available = [...Array.from({length: 4 - moves.length}, (_, index) => moves.length + index),
            ...moves.map((move, index) => known.moves.includes(move) ? null : index).filter(index => index !== null)];
        if (available.length < missing.length) return {supported: false, reason: 'no_unobserved_move_slots', proposals: []};
        replacements.push({key: entry.key, moves, missing, available});
        changed = true;
    }
    if (!changed) return {supported: false, reason: 'no_new_move_completion_required', proposals: []};
    const proposals = [];
    // rotate the evicted slots so one arbitrary unseen move does not consume every attempt
    for (let offset = 0; offset < 4; ++offset) {
        const candidate = structuredClone(proposal);
        for (const {key, moves, missing, available} of replacements) {
            const completed = [...moves];
            missing.forEach((move, index) => { completed[available[(offset + index) % available.length]] = move; });
            candidate.find(mon => mon.key === key).set.moves = completed.filter(Boolean);
        }
        if (!proposals.some(saved => isDeepStrictEqual(saved, candidate))) proposals.push(candidate);
    }
    return {supported: true, proposals};
}

async function completeMoveHistory(path, history, seed, budget) {
    const observations = history.observations, own = history.own_choices;
    const report = {verified: false, reason: null, proposals: 0, witnesses: [], training_eligible: false};
    if (observations.length < 2 || observations.length > 16 || own.length !== observations.length - 1 ||
            own.some(choice => choice !== null && typeof choice !== 'string'))
        return {...report, reason: 'ordinary_bounded_history_required'};
    const facts = observeBeliefs(observations.at(-1));
    if (facts.issues.length) return {...report, reason: 'unsupported_visible_evidence'};
    const side = observations[0].side, opponent = side === 'p1' ? 'p2' : 'p1';
    const battle = restorePosition(path.opening, {mode: 'sample', seed});
    let sets;
    try {
        if (!isDeepStrictEqual(playerObservation(battle, side), observations[0]))
            throw new Error('move completion root differs from the recorded opening');
        sets = battle.getSide(opponent).pokemon.map(mon => ({key: mon.fullname, set: structuredClone(mon.set)}));
    } finally { battle.destroy(); }
    const completed = moveCompletions(sets, facts);
    if (!completed.supported) return {...report, reason: completed.reason};
    const rng = new PRNG(seed.join(','));
    for (const proposal of completed.proposals) {
        if (!budget.available()) break;
        await budget.wait?.();
        if (!budget.available()) break;
        ++report.proposals; budget.consume();
        const opening = constructReplayOpening(observations[0], observations.at(-1),
            {seed: nextSeed(rng), chance_seed: nextSeed(rng), attempts: 1, proposal_sets: proposal});
        if (!opening.supported) continue;
        for (let trial = 0; trial < 8 && budget.available(); ++trial) {
            let position = opening.position;
            const steps = [];
            for (let index = 1; index < observations.length && budget.available(); ++index) {
                const draw = index < observations.length - 1 && path.steps[index - 1]?.seed ||
                    (trial === 0 ? seed : nextSeed(rng));
                const branch = restorePosition(position, {mode: 'sample', seed: draw});
                let choices;
                try {
                    const responses = observedResponses(branch, observations[index - 1], observations[index]);
                    if (!responses.length) break;
                    const prior = path.steps[index - 1]?.choices?.[opponent];
                    const response = responses.includes(prior) ? prior : responses[trial % responses.length];
                    choices = {};
                    if (own[index - 1] !== null) choices[side] = own[index - 1];
                    if (response !== null) choices[opponent] = response;
                } catch (error) {
                    if (!(error instanceof UnsupportedPlannerRequest)) throw error;
                    break;
                } finally { branch.destroy(); }
                budget.consume();
                let next;
                try { next = advancePosition(position, choices, {mode: 'sample', seed: draw}); }
                catch (error) { if (choiceFailure(error)) break; throw error; }
                if (!isDeepStrictEqual(next.observations[side], observations[index])) break;
                steps.push({seed: draw, choices}); position = next.position;
            }
            if (steps.length !== observations.length - 1) continue;
            // a separate native replay validates every boundary without editing the retained parent
            let verified = opening.position;
            for (const [index, step] of steps.entries()) {
                if (!budget.available()) return {...report, reason: 'independent_replay_budget_exhausted'};
                budget.consume();
                const next = advancePosition(verified, step.choices, {mode: 'sample', seed: step.seed});
                if (!isDeepStrictEqual(next.observations[side], observations[index + 1]))
                    throw new Error('move completion failed independent full history replay');
                verified = next.position;
            }
            return {...report, verified: true, reason: 'revealed_moves_full_history_verified',
                witnesses: [{opening: opening.position, position: verified, steps}]};
        }
    }
    return {...report, reason: 'no_exact_move_completion_within_budget'};
}

module.exports = {moveCompletions, completeMoveHistory};
