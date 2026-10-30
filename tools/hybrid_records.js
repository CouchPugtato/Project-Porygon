'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {writeJsonAtomic} = require('./counterfactual_rollout');
const {hashFile} = require('./tactical_teacher_smoke');

const DETAIL_LIMIT = 2 * 1024 ** 2;

async function detachDetails(game, directory, index) {
    const {first_belief_failure, battle_analysis, ...summary} = game;
    const belief_losses = game.hybrid_coverage?.belief_losses;
    if (!first_belief_failure && !battle_analysis && !belief_losses) return summary;
    if (game.hybrid_coverage) {
        const {belief_losses: ignored, ...coverage} = game.hybrid_coverage;
        summary.hybrid_coverage = coverage;
    }
    const details = {first_belief_failure, battle_analysis, belief_losses};
    const digest = createHash('sha256').update(JSON.stringify(details)).digest('hex');
    const relative = `details/${String(index).padStart(6, '0')}_${digest}.json`;
    const filename = path.join(directory, relative);
    const bytes = Buffer.byteLength(JSON.stringify(details, null, 2)) + 1;
    if (bytes > DETAIL_LIMIT) throw new Error('hybrid detail record exceeds the 2 MiB budget');
    await fs.mkdir(path.dirname(filename), {recursive: true});
    try { await fs.access(filename); }
    catch (error) { if (error.code !== 'ENOENT') throw error; await writeJsonAtomic(filename, details); }
    const pointer = {path: relative, sha256: await hashFile(filename), bytes: (await fs.stat(filename)).size};
    const saved = await loadDetails({...summary, details: pointer}, directory);
    if (JSON.stringify(saved.first_belief_failure) !== JSON.stringify(first_belief_failure) ||
            JSON.stringify(saved.battle_analysis) !== JSON.stringify(battle_analysis) ||
            JSON.stringify(saved.hybrid_coverage?.belief_losses) !== JSON.stringify(belief_losses))
        throw new Error('existing content addressed hybrid detail record differs');
    return {...summary, details: pointer};
}

async function loadDetails(game, directory) {
    if (!game.details) return game;
    const pointer = game.details;
    if (!/^details\/\d{6}_[a-f0-9]{64}\.json$/.test(pointer.path || '') ||
            !Number.isInteger(pointer.bytes) || pointer.bytes < 1 || pointer.bytes > DETAIL_LIMIT ||
            !/^[a-f0-9]{64}$/.test(pointer.sha256 || '')) throw new Error('invalid hybrid detail pointer');
    const filename = path.resolve(directory, pointer.path);
    if ((await fs.stat(filename)).size !== pointer.bytes || await hashFile(filename) !== pointer.sha256)
        throw new Error('hybrid detail integrity changed');
    const details = JSON.parse(await fs.readFile(filename, 'utf8'));
    const loaded = {...game, first_belief_failure: details.first_belief_failure, battle_analysis: details.battle_analysis};
    if (details.belief_losses && game.hybrid_coverage)
        loaded.hybrid_coverage = {...game.hybrid_coverage, belief_losses: details.belief_losses};
    return loaded;
}

module.exports = {detachDetails, loadDetails};
