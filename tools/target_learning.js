'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const {writeJsonAtomic, writeTextAtomic} = require('./counterfactual_rollout');

function validateLabels(labels) {
    if (!/^[0-9a-f]{16}$/.test(labels?.parent_fingerprint) || !Array.isArray(labels.examples) ||
            labels.examples.length > 2 || !Array.isArray(labels.hidden) ||
            (labels.examples.length ? labels.hidden.length < 1 || labels.hidden.length > 4096 : labels.hidden.length !== 0) ||
            labels.hidden.some(value => !Number.isFinite(value))) throw new Error('invalid target hidden-state cache');
    const slots = new Set();
    for (const e of labels.examples) {
        if (![e.slot, e.move, e.legal, e.best].every(Number.isInteger) || e.slot < 0 || e.slot > 1 ||
                e.move < 0 || e.move > 3 || e.legal < 1 || e.legal > 15 || e.best < 1 ||
                (e.best & ~e.legal) || e.best === e.legal || slots.has(e.slot))
            throw new Error('invalid or uninformative target label');
        slots.add(e.slot);
    }
}

function createTargetCollector() {
    let pending = null;
    const decisions = [];
    return {
        propose(action) {
            if (pending || action.training_eligible !== false) throw new Error('invalid target label proposal');
            validateLabels(action.target_labels);
            pending = action;
        },
        observe(line) {
            const message = JSON.parse(line);
            if (!['decision_accepted', 'decision_rejected'].includes(message.type)) return;
            if (!pending || pending.request_id !== message.request_id || pending.command !== message.command)
                throw new Error('target labels do not match accepted choice');
            if (message.type === 'decision_accepted') decisions.push({request_id: pending.request_id, ...pending.target_labels});
            pending = null;
        },
        finish(episode) {
            if (pending || episode.policy_tag !== 'target-labels-v1' || episode.training_eligible !== false ||
                    episode.label_decisions !== decisions.length || !decisions.length)
                throw new Error('target label terminal count mismatch');
            return decisions;
        },
    };
}

async function publishLabels(directory, game, opponent, index) {
    const filename = `target_labels/${opponent}_${index}.json`;
    const data = {battle_id: `target-${opponent}-${index}`, opponent, result: game.result,
        decisions: game.target_examples};
    await fs.mkdir(path.join(directory, 'target_labels'), {recursive: true});
    await writeJsonAtomic(path.join(directory, filename), data);
    const bytes = await fs.readFile(path.join(directory, filename));
    delete game.target_examples;
    return {path: filename, sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
        decisions: data.decisions.length, examples: data.decisions.reduce((n, row) => n + row.examples.length, 0)};
}

async function readLabels(directory, reference) {
    if (!reference || !/^target_labels\/(parent|random)_\d+\.json$/.test(reference.path))
        throw new Error('missing target label artifact');
    const bytes = await fs.readFile(path.join(directory, reference.path));
    if (crypto.createHash('sha256').update(bytes).digest('hex') !== reference.sha256)
        throw new Error('target label checksum mismatch');
    const data = JSON.parse(bytes);
    data.decisions.forEach(validateLabels);
    if (data.decisions.length !== reference.decisions ||
            data.decisions.reduce((n, row) => n + row.examples.length, 0) !== reference.examples)
        throw new Error('target label counts changed');
    return data;
}

function splitBattle(battleId, seed) {
    return crypto.createHash('sha256').update(`${seed}:${battleId}`).digest().readUInt32LE(0) % 5 === 0;
}

function packDataset(battles, seed) {
    const rows = [];
    const battleIds = new Set();
    let decisions = 0, informative = 0;
    let fingerprint, dimensions;
    for (const battle of battles) {
        if (typeof battle.battle_id !== 'string' || !battle.battle_id || battleIds.has(battle.battle_id))
            throw new Error('target dataset contains missing or duplicate battle IDs');
        battleIds.add(battle.battle_id);
        for (const decision of battle.decisions) {
            validateLabels(decision);
            ++decisions;
            informative += +(decision.examples.length > 0);
            if (!fingerprint) fingerprint = decision.parent_fingerprint;
            if (fingerprint !== decision.parent_fingerprint) throw new Error('target labels mix different parents');
            for (const example of decision.examples) {
                if (!dimensions) dimensions = decision.hidden.length;
                if (dimensions !== decision.hidden.length) throw new Error('target hidden dimensions changed');
                const key = `${battle.battle_id}:${decision.request_id}:${example.slot}`;
                rows.push({...example, hidden: decision.hidden, validation: +splitBattle(battle.battle_id, seed),
                    order: crypto.createHash('sha256').update(`${seed}:${key}`).digest('hex')});
            }
        }
    }
    if (!dimensions || rows.length < 300 || rows.length > 100000) throw new Error('insufficient or excessive target examples');
    rows.sort((a, b) => a.order.localeCompare(b.order));
    const bytes = Buffer.alloc(25 + rows.length * (5 + 4 * dimensions));
    bytes.write('PGTLABEL1'); bytes.writeUInt32LE(dimensions, 9); bytes.writeUInt32LE(rows.length, 13);
    bytes.writeBigUInt64LE(BigInt(`0x${fingerprint}`), 17);
    let offset = 25;
    for (const row of rows) {
        for (const key of ['validation', 'slot', 'move', 'legal', 'best']) bytes[offset++] = row[key];
        for (const value of row.hidden) { bytes.writeFloatLE(value, offset); offset += 4; }
    }
    return {bytes, counts: {battles: battles.length, examples: rows.length,
        train_battles: battles.filter(battle => !splitBattle(battle.battle_id, seed)).length,
        holdout_battles: battles.filter(battle => splitBattle(battle.battle_id, seed)).length,
        accepted_decisions: decisions, informative_decisions: informative, unlabelled_decisions: decisions - informative,
        train: rows.filter(row => !row.validation).length, holdout: rows.filter(row => row.validation).length}};
}

function targetFitEligible(metadata, report, exploratory = false) {
    if (!report.sidecar_published) return false;
    if (!exploratory) return report.passed === true && metadata.exploratory_evaluation !== true;
    return metadata.exploratory_evaluation === true && report.passed === false &&
        report.exploratory_exception === true && report.evaluation_only === true && report.parent_unchanged === true &&
        Number.isFinite(report.holdout_nll_after) && report.holdout_nll_after < report.holdout_nll_before &&
        JSON.stringify(report.failure_reasons) === '["holdout_accuracy_decreased"]';
}

async function main(argv) {
    const exploratory = argv.length === 3 && argv[2] === '--exploratory-evaluation';
    if (argv.length !== 2 && !exploratory) throw new Error('usage: node tools/target_learning.js COLLECTION_RUN OUTPUT_DIRECTORY [--exploratory-evaluation]');
    const root = path.resolve(__dirname, '..'), directory = path.join(root, 'matches/runs', argv[0]);
    const output = path.resolve(root, argv[1]);
    const {isValid} = require('./tactical_teacher_benchmark');
    const {hashFile} = require('./tactical_teacher_smoke');
    const summaryPath = path.join(directory, `${argv[0]}_summary.json`);
    const summary = JSON.parse(await fs.readFile(summaryPath, 'utf8'));
    if (summary.status !== 'completed' || summary.valid_games !== 500 || !summary.identity.config.collect_target_labels)
        throw new Error('fit requires the completed, fixed 500-battle collection');
    const battles = [];
    for (const game of summary.games.filter(isValid)) battles.push(await readLabels(directory, game.target_labels));
    const seed = 20261026, packed = packDataset(battles, seed);
    await fs.mkdir(output, {recursive: true});
    const dataset = path.join(output, 'target_labels.bin'), sidecar = path.join(output, 'target_sidecar.bin');
    const report = path.join(output, 'target_fit.json');
    const parent = path.resolve(root, summary.identity.config.candidate_checkpoint || summary.identity.config.checkpoint);
    if (await hashFile(parent) !== summary.identity.candidate_checkpoint_sha256) throw new Error('collection parent changed');
    const executable = path.join(root, 'build-fresh/showdown_client.exe');
    for (const filename of [sidecar, report, `${sidecar}.meta.json`]) {
        try { await fs.access(filename); } catch (error) { if (error.code === 'ENOENT') continue; throw error; }
        throw new Error('fit artifacts already exist; use a new output directory');
    }
    await writeTextAtomic(dataset, packed.bytes);
    await writeJsonAtomic(path.join(output, 'dataset.json'), {seed, split: 'sha256(seed:battle_id) mod 5 == 0',
        ...packed.counts, source_summary: summaryPath, source_sha256: await hashFile(summaryPath),
        dataset_sha256: await hashFile(dataset), parent_sha256: await hashFile(parent),
        collection_executable_sha256: summary.identity.executable_sha256,
        fit_executable_sha256: await hashFile(executable), fitter_source_sha256: await hashFile(__filename)});
    const run = spawnSync(executable, ['--fit-target-sidecar', dataset, parent, sidecar, report,
        ...(exploratory ? ['--exploratory-evaluation'] : [])],
        {cwd: root, windowsHide: true, env: {...process.env, PORYGON_OMP_THREADS: '1', OMP_NUM_THREADS: '1'}, stdio: 'inherit'});
    if (run.error || run.status !== 0) throw new Error(`target fit failed: ${run.error || run.status}`);
    const result = JSON.parse(await fs.readFile(report, 'utf8'));
    if (exploratory) {
        const original = JSON.parse(await fs.readFile(path.join(root,
            'models/runs/run_0286_move_conditioned_targets/target_fit.json'), 'utf8'));
        for (const key of ['best_epoch', 'train_nll_before', 'train_nll_after', 'holdout_nll_before',
            'holdout_nll_after', 'accuracy_before', 'accuracy_after', 'parent_fingerprint'])
            if (result[key] !== original[key]) throw new Error(`exploratory replay changed the fixed candidate: ${key}`);
    }
    if (targetFitEligible({exploratory_evaluation: exploratory}, result, exploratory))
        await writeJsonAtomic(`${sidecar}.meta.json`, {parent_sha256: await hashFile(parent),
        sidecar_sha256: await hashFile(sidecar), dataset_sha256: await hashFile(dataset),
        executable_sha256: await hashFile(executable), fit_report_sha256: await hashFile(report),
        exploratory_evaluation: exploratory});
    console.log(`[target-fit] ${JSON.stringify({...result, counts: packed.counts})}`);
    if (!targetFitEligible({exploratory_evaluation: exploratory}, result, exploratory)) process.exitCode = 2;
}

module.exports = {createTargetCollector, packDataset, publishLabels, readLabels, splitBattle, validateLabels, targetFitEligible};
if (require.main === module) main(process.argv.slice(2)).catch(error => {console.error(error); process.exitCode = 1;});
