'use strict';

const {Dex} = require('../external/pokemon-showdown/dist/sim/dex');

function mutateProposal(sets, rng) {
    const candidate = structuredClone(sets), chosen = candidate[rng.random(candidate.length)].set;
    const stats = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
    const stat = stats[rng.random(stats.length)];
    if (rng.random(4) === 0) {
        const natures = Dex.natures.all(); chosen.nature = natures[rng.random(natures.length)].name;
    } else {
        chosen.ivs = {...chosen.ivs, [stat]: rng.random(32)};
        chosen.evs = {...chosen.evs, [stat]: rng.random(64) * 4};
        if (Object.values(chosen.evs).reduce((a, b) => a + b, 0) > 508) chosen.evs[stat] = 0;
        if (Object.values(chosen.evs).every(n => n === 0)) chosen.evs.hp = 1;
    }
    return candidate;
}

module.exports = {mutateProposal};
