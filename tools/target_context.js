'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {writeJsonAtomic, writeTextAtomic} = require('./counterfactual_rollout');

function validateContext(c) {
    if (c?.version !== 1 || !Number.isInteger(c.move_id) || c.move_id <= 0 || c.move_id > 65535 ||
            !Number.isInteger(c.move_type_id) || c.move_type_id < 0 || c.move_type_id >= 19 ||
            typeof c.move_type_known !== 'boolean' || c.move_type_known !== (c.move_type_id !== 0) ||
            !Array.isArray(c.targets) || c.targets.length !== 4) throw new Error('invalid selected-move context');
    for (const [i, p] of c.targets.entries()) {
        if (p.position !== i || typeof p.present !== 'boolean' || p.friendly !== (i < 2) ||
                ![p.type1_id, p.type2_id, p.type1_known_mode, p.type2_known_mode].every(Number.isInteger) ||
                p.type1_id < 0 || p.type1_id >= 19 || p.type2_id < 0 || p.type2_id >= 19 ||
                p.type1_known_mode < 0 || p.type1_known_mode > 2 || p.type2_known_mode < 0 || p.type2_known_mode > 2 ||
                (p.type1_known_mode !== 0) !== (p.type1_id !== 0) || (!p.type2_known_mode && p.type2_id !== 0) ||
                (!p.present && (p.type1_id || p.type2_id || p.type1_known_mode || p.type2_known_mode)))
            throw new Error('invalid or unredacted visible-target context');
    }
}

function writeContext(bytes, offset, c) {
    validateContext(c);
    bytes.writeUInt16LE(c.move_id, offset); offset += 2;
    bytes[offset++] = c.move_type_id; bytes[offset++] = +c.move_type_known;
    for (const p of c.targets) for (const key of
        ['position', 'present', 'friendly', 'type1_id', 'type1_known_mode', 'type2_id', 'type2_known_mode']) bytes[offset++] = +p[key];
    return offset;
}

async function compareContext(name, outputDirectory) {
    if (!/^[A-Za-z0-9_-]+$/.test(name)) throw new Error('a simple collection run name is required');
    const root = path.resolve(__dirname, '..'), directory = path.join(root, 'matches/runs', name);
    const {readLabels, packDataset} = require('./target_learning');
    const {isValid} = require('./tactical_teacher_benchmark');
    const {hashFile} = require('./tactical_teacher_smoke');
    const summaryPath = path.join(directory, `${name}_summary.json`);
    const summary = JSON.parse(await fs.readFile(summaryPath, 'utf8'));
    if (summary.status !== 'completed' || summary.valid_games !== 500 || summary.identity.config.target_context_version !== 1)
        throw new Error('comparison requires a completed 500-battle context collection');
    const battles = [];
    for (const game of summary.games.filter(isValid)) battles.push(await readLabels(directory, game.target_labels));
    const packed = packDataset(battles, 20261026, true);
    const output = path.resolve(root, outputDirectory);
    const reportPath = path.join(output, 'context_comparison.json');
    try { await fs.access(reportPath); throw new Error('comparison already exists; use a new output directory'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const parent = path.resolve(root, summary.identity.config.candidate_checkpoint || summary.identity.config.checkpoint);
    const executable = path.join(root, 'build-fresh/showdown_client.exe');
    if (await hashFile(parent) !== summary.identity.candidate_checkpoint_sha256) throw new Error('frozen collection parent changed');
    await fs.mkdir(output, {recursive: true});
    const dataset = path.join(output, 'target_context.bin');
    await writeTextAtomic(dataset, packed.bytes);
    const provenance = {counts: packed.counts, split_seed: 20261026, source_summary: summaryPath,
        source_summary_sha256: await hashFile(summaryPath), dataset_sha256: await hashFile(dataset),
        parent_sha256: await hashFile(parent), executable_sha256: await hashFile(executable),
        collection_executable_sha256: summary.identity.executable_sha256,
        wrapper_sha256: await hashFile(__filename), packing_source_sha256: await hashFile(path.join(__dirname, 'target_learning.js'))};
    await writeJsonAtomic(path.join(output, 'dataset.json'), provenance);
    const run = spawnSync(executable, ['--compare-target-context', dataset, parent],
        {cwd: root, windowsHide: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'],
            env: {...process.env, PORYGON_OMP_THREADS: '1', OMP_NUM_THREADS: '1'}});
    if (run.error || run.status !== 0) throw new Error(`context comparison failed: ${run.error || run.status}`);
    const report = JSON.parse(run.stdout);
    if (!report.parent_unchanged || report.weights_published !== false || report.gameplay_authorized !== false ||
            await hashFile(parent) !== provenance.parent_sha256) throw new Error('context comparison violated the frozen-model boundary');
    await writeJsonAtomic(reportPath, {...report, provenance});
    console.log(`[target-context] ${JSON.stringify({...report, counts: packed.counts})}`);
}

module.exports = {validateContext, writeContext, compareContext};
