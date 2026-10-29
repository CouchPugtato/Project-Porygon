'use strict';

function sideValue(battle, side, enriched) {
    let value = 0;
    const statusCost = {brn: 0.12, par: 0.16, slp: 0.22, frz: 0.25, psn: 0.10, tox: 0.18};
    for (const mon of side.pokemon) {
        value += mon.hp / mon.maxhp + (mon.hp ? 2 : 0);
        if (!enriched || !mon.hp) continue;
        value -= statusCost[mon.status] || 0;
        const boosts = Object.values(mon.boosts).reduce((sum, n) => sum + n, 0);
        value += (mon.isActive ? 0.025 : 0.005) * boosts;
        if (mon.isActive) {
            const speed = mon.getStat('spe');
            const reversed = !!battle.field.pseudoWeather.trickroom;
            value += (reversed ? -1 : 1) * 0.025 * Math.log1p(speed) / 6;
            if (mon.volatiles.substitute) value += 0.10;
            if (mon.volatiles.confusion) value -= 0.05;
        }
        const usable = mon.moveSlots.filter(move => move.pp > 0).length;
        value += 0.005 * usable + (mon.item ? 0.02 : 0);
    }
    if (enriched) {
        const conditions = side.sideConditions;
        for (const screen of ['reflect', 'lightscreen', 'auroraveil', 'tailwind'])
            if (conditions[screen]) value += 0.10;
        for (const hazard of ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb'])
            if (conditions[hazard]) value -= 0.04 * (conditions[hazard].layers || 1) * side.pokemon.filter(mon => mon.hp && !mon.isActive).length;
        if (side.pokemon.some(mon => mon.hp && mon.canTerastallize)) value += 0.03;
    }
    return value;
}

function evaluatePosition(battle, perspective, mode = 'material') {
    const own = battle.getSide(perspective);
    return sideValue(battle, own, mode === 'positional') - sideValue(battle, own.foe, mode === 'positional');
}

module.exports = {evaluatePosition};
