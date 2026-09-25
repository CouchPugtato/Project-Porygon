#!/usr/bin/env node

'use strict';

const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.resolve(__dirname, '..');
const {Dex} = require(path.join(repoRoot, 'external', 'pokemon-showdown', 'dist', 'sim'));

function hasPositiveBoost(boosts) {
    return Boolean(boosts) && Object.values(boosts).some(value => Number(value) > 0);
}

function moveRow(id) {
    const move = Dex.moves.get(id);
    if (!move.exists) return [id, ...Array(19).fill(0)];
    const category = {Status: 1, Physical: 2, Special: 3}[move.category] || 0;
    const target = String(move.target || '');
    const spread = move.spreadHit || ['all', 'allAdjacent', 'allAdjacentFoes'].includes(target);
    const healing = Boolean(move.heal || move.drain || move.flags?.heal);
    const protection = Boolean(move.stallingMove);
    const setup = hasPositiveBoost(move.boosts) ||
        hasPositiveBoost(move.self?.boosts) || hasPositiveBoost(move.selfBoost?.boosts);
    const redirection = Boolean(move.flags?.redirect) ||
        move.volatileStatus === 'followme' || ['followme', 'ragepowder'].includes(move.id);
    const damaging = move.category !== 'Status' || move.damage !== undefined;
    const statusEffect = Boolean(move.status || move.volatileStatus);
    const fieldEffect = Boolean(move.sideCondition || move.slotCondition || move.pseudoWeather);
    const allySupport = ['adjacentAlly', 'adjacentAllyOrSelf', 'allySide'].includes(target) ||
        healing || redirection;
    return [
        id,
        Math.max(0, Number(move.basePower) || 0),
        move.accuracy === true ? 100 : Math.max(0, Number(move.accuracy) || 0),
        category,
        Number(move.priority) || 0,
        Number(Boolean(spread)),
        Number(healing),
        Number(protection),
        Number(setup),
        Number(redirection),
        Number(damaging),
        Number(Boolean(move.recoil || move.hasCrashDamage || move.mindBlownRecoil)),
        Number(Boolean(move.drain)),
        Number(Boolean(move.selfSwitch)),
        Number(Boolean(move.forceSwitch)),
        Number(statusEffect),
        Number(Boolean(move.weather)),
        Number(Boolean(move.terrain)),
        Number(fieldEffect),
        Number(allySupport),
    ];
}

function main() {
    const moveIdsPath = path.join(repoRoot, 'data', 'move_ids.txt');
    const outputPath = path.join(repoRoot, 'data', 'move_semantics.txt');
    const ids = fs.readFileSync(moveIdsPath, 'utf8')
        .split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    const header = '# move,base_power,accuracy,category,priority,spread,healing,protection,' +
        'setup,redirection,damaging,recoil,drain,self_switch,force_switch,status_effect,' +
        'weather,terrain,field_effect,ally_support';
    const output = [header, ...ids.map(id => moveRow(id).join(',')), ''].join('\n');
    fs.writeFileSync(outputPath, output, 'utf8');
    console.log(`wrote ${path.relative(repoRoot, outputPath)}: ${ids.length} entries`);
}

if (require.main === module) main();

module.exports = {hasPositiveBoost, moveRow};
