'use strict';

const {isDeepStrictEqual} = require('node:util');
const {Dex} = require('../external/pokemon-showdown/dist/sim/dex');
const {restorePosition, playerObservation, pendingSides} = require('./position_simulator');

function rollReplay(position, choices, seed, side, rolls) {
    const battle = restorePosition(position, {mode: 'sample', seed});
    let count = 0;
    try {
        const native = battle.randomizer;
        battle.randomizer = function (damage) {
            const roll = rolls[count++] ?? rolls.at(-1) ?? 0, random = this.random;
            // consume the native draw even when exploring a different legal damage roll
            // accuracy critical hits secondary effects and ordering keep their seeded draws
            this.random = function (...args) {
                const drawn = random.apply(this, args);
                return args.length === 1 && args[0] === 16 ? roll : drawn;
            };
            try { return native.call(this, damage); }
            finally { this.random = random; }
        };
        const required = pendingSides(battle);
        if (required.length !== Object.keys(choices).length || required.some(side => typeof choices[side] !== 'string'))
            throw new Error('roll diagnostic requires complete native choices');
        for (const id of required) if (!battle.choose(id, choices[id]))
            throw new Error(`roll diagnostic rejected ${id} choice`);
        return {observation: playerObservation(battle, side), count};
    } finally { battle.destroy(); }
}

async function diagnoseRolls(position, choices, seed, expected, budget, loss) {
    const report = {status: 'inconclusive', trials: 0, max_damage_events: 0, conditional: true,
        exhaustive: false, training_eligible: false, witness_eligible: false,
        scope: 'fixed build choices seed and all nondamage randomness', best_loss: null};
    let best = null;
    const seen = new Set();
    const evaluate = async rolls => {
        if (!budget.available() || report.trials >= 64) return;
        const key = rolls.join(',');
        if (seen.has(key)) return;
        seen.add(key); budget.consume(); ++report.trials;
        if (report.trials % 16 === 0) await budget.wait?.();
        const trial = rollReplay(position, choices, seed, expected.side, rolls);
        report.max_damage_events = Math.max(report.max_damage_events, trial.count);
        const distance = loss(trial.observation, expected);
        if (!best || distance < best.loss) best = {rolls, loss: distance, count: trial.count};
        if (isDeepStrictEqual(trial.observation, expected)) report.status = 'conditional_legal_roll_match';
    };
    // uniform roll sweeps are exhaustive only when no branch uses more than one roll
    // further coordinate trials guide proposals but cannot rule out a joint roll assignment
    for (let roll = 0; roll < 16 && budget.available(); ++roll) await evaluate([roll]);
    if (report.trials === 16 && report.max_damage_events <= 1) {
        report.exhaustive = true;
        if (report.status === 'inconclusive') report.status = 'no_roll_match_under_fixed_events';
    }
    for (let index = 0; index < Math.min(8, best?.count || 0) && report.status === 'inconclusive'; ++index) {
        const base = Array.from({length: best.count}, (_, i) => best.rolls[i] ?? best.rolls.at(-1));
        for (const value of [0, 4, 8, 12, 15]) {
            const proposal = [...base]; proposal[index] = value;
            await evaluate(proposal);
        }
    }
    report.best_loss = best?.loss ?? null;
    // forced rolls are diagnostic evidence only and never a natural seeded witness
    return report;
}

function mechanicalAlternatives(sets, facts, variables, speedFirst) {
    const proposals = [];
    const add = (entry, field, value) => {
        if (entry.set[field] === value) return;
        const changed = structuredClone(sets);
        changed.find(mon => mon.key === entry.key).set[field] = value;
        proposals.push(changed);
    };
    for (const entry of sets.filter(entry => variables.some(variable => variable.key === entry.key))) {
        const known = facts.opponent.pokemon.find(mon => mon.key === entry.key);
        if (!known) continue;
        if (!known.ability) for (const ability of Object.values(Dex.species.get(entry.set.species).abilities))
            add(entry, 'ability', ability);
        // alternatives selected by mechanics rather than published teams or frequency priors
        if (!known.item) for (const item of speedFirst ? ['Choice Scarf', 'Iron Ball', 'Lagging Tail', ''] :
            ['Life Orb', 'Choice Band', 'Choice Specs', 'Assault Vest', 'Eviolite', 'Leftovers', 'Sitrus Berry', ''])
            add(entry, 'item', item);
        const stats = variables.filter(variable => variable.key === entry.key).map(variable => variable.stat);
        for (const nature of Dex.natures.all()) if (stats.includes(nature.plus) || stats.includes(nature.minus))
            add(entry, 'nature', nature.name);
    }
    return proposals.slice(0, 96);
}

function investmentRanges(candidates, variables) {
    return variables.map(({key, stat}) => {
        const values = candidates.map(sets => sets.find(entry => entry.key === key)?.set)
            .filter(Boolean).map(set => (set.ivs?.[stat] ?? 31) + Math.floor((set.evs?.[stat] ?? 0) / 4));
        return {key, stat, sampled_values: [...new Set(values)].sort((a, b) => a - b),
            minimum: values.length ? Math.min(...values) : null, maximum: values.length ? Math.max(...values) : null};
    });
}

module.exports = {rollReplay, diagnoseRolls, mechanicalAlternatives, investmentRanges};
