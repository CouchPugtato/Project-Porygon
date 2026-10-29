'use strict';

const {isDeepStrictEqual} = require('node:util');
const {restorePosition, playerObservation, pendingSides} = require('./position_simulator');

function firstDamageFeasibility(trace) {
    const {position, choices, seed, expected} = trace;
    const outcomes = [], parent = position.state_json;
    for (let roll = 0; roll < 16; ++roll) {
        const battle = restorePosition(position, {mode: 'sample', seed});
        try {
            const before = playerObservation(battle, expected.side), required = pendingSides(battle);
            if (required.length !== Object.keys(choices).length || required.some(side => typeof choices[side] !== 'string'))
                throw new Error('damage feasibility requires choices for exactly the pending players');
            const observedIndex = expected.log.findIndex((line, i) => i >= before.log.length && line.startsWith('|-damage|'));
            if (observedIndex < 0) return {supported: false, reason: 'no_observed_direct_damage'};
            const originalDamage = battle.actions.getDamage, originalRandomizer = battle.randomizer;
            let call = 0, activeCall = 0, first = null, randomized = 0, selectedCall = null;
            battle.actions.getDamage = function (...args) {
                const previous = activeCall, current = ++call; activeCall = current;
                const logStart = playerObservation(battle, expected.side).log.length;
                try {
                    const damage = originalDamage.apply(this, args);
                    if (!first && damage > 0) first = {source: `${args[0].getSlot()}: ${args[0].name}`,
                        target: `${args[1].getSlot()}: ${args[1].name}`, move: args[2]?.id,
                        call: current, log_start: logStart, raw_damage: damage, critical: typeof args[2] === 'object' ?
                            !!args[1].getMoveHitData(args[2]).crit : null};
                    return damage;
                } finally { activeCall = previous; }
            };
            battle.randomizer = function (baseDamage) {
                // consume the native draw before varying this one damage factor
                // all other chance outcomes remain conditional on the captured seed
                const sampled = originalRandomizer.call(this, baseDamage);
                if (!first && activeCall && selectedCall === null) selectedCall = activeCall;
                if (activeCall !== selectedCall) return sampled;
                ++randomized;
                return this.trunc(this.trunc(baseDamage * (100 - roll)) / 100);
            };
            for (const side of required) if (!battle.choose(side, choices[side]))
                throw new Error('captured choices rejected during damage feasibility replay');
            if (!first || first.call !== selectedCall || randomized !== 1)
                return {supported: false, reason: 'first_hit_not_a_single_randomized_damage_calculation'};
            const actual = playerObservation(battle, expected.side);
            const index = actual.log.findIndex((line, i) => i >= before.log.length && line.startsWith('|-damage|'));
            if (index < first.log_start || actual.log[index].split('|')[2] !== first.target)
                return {supported: false, reason: 'first_logged_damage_not_the_first_calculated_hit'};
            const prefixMatched = isDeepStrictEqual(actual.log.slice(0, index), expected.log.slice(0, observedIndex));
            outcomes.push({roll, ...first, damage_event: actual.log[index], prefix_matched: prefixMatched,
                observed_damage_matched: prefixMatched && actual.log[index] === expected.log[observedIndex]});
        } finally { battle.destroy(); }
    }
    if (position.state_json !== parent) throw new Error('damage feasibility mutated its parent');
    const contextMatched = outcomes.some(outcome => outcome.prefix_matched);
    return {supported: true, scope: 'first_direct_hit_with_other_chance_draws_fixed',
        all_sixteen_damage_factors_enumerated: true, context_matched: contextMatched,
        feasible_in_captured_context: contextMatched ? outcomes.some(outcome => outcome.observed_damage_matched) : null,
        full_transition_verified: false, origin: position.origin, outcomes,
        limitation: 'Conditional hit feasibility is not a proof about all prior chance outcomes or all hidden builds.'};
}

module.exports = {firstDamageFeasibility};
