'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {loadDetails} = require('./hybrid_records');

function renderAnalysis(record) {
    const value = number => Number.isFinite(number) ? number.toFixed(3) : 'unavailable';
    const lines = [`turn ${record.turn} ${record.phase}`, `chosen: ${record.selected_command}`,
        `baseline: ${record.reference_command}`, `reason: ${record.reason}`,
        `distinct active hypotheses: ${record.distinct_active_worlds}`, `units: ${record.units}`];
    for (const row of record.candidates) lines.push(`${row.command} | ${row.status} | score ${value(row.heuristic_score)}` +
        ` | mean gain ${value(row.gain_over_reference?.mean)} | worst sampled gain ${value(row.gain_over_reference?.worst)}`);
    if (record.budget) lines.push(`search: ${record.budget.nodes} nodes in ${value(record.budget.elapsed_seconds)}s`);
    lines.push(...record.assumptions.map(assumption => `assumption: ${assumption}`));
    return lines.join('\n');
}

async function main(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i += 2) {
        if (!['--input', '--game', '--turn'].includes(argv[i]) || !argv[i + 1]) throw new Error('expected --input FILE --game INDEX --turn TURN');
        args[argv[i]] = argv[i + 1];
    }
    const filename = path.resolve(args['--input'] || ''), index = Number(args['--game'] || 0), turn = Number(args['--turn']);
    if (!Number.isInteger(index) || index < 0 || !Number.isInteger(turn) || turn < 1)
        throw new Error('nonnegative game index and positive turn required');
    if ((await fs.stat(filename)).size > 16 * 1024 ** 2) throw new Error('analysis input exceeds the 16 MiB limit');
    const report = JSON.parse(await fs.readFile(filename, 'utf8')), game = report.games?.[index];
    const {recordHash} = require('./hybrid_benchmark');
    if (!game || game.record_hash !== recordHash(game)) throw new Error('battle analysis record integrity changed');
    const detailed = await loadDetails(game, path.dirname(filename));
    const records = detailed.battle_analysis?.records.filter(record => record.turn === turn) || [];
    if (!records.length) throw new Error('requested turn was not retained in this battle analysis');
    for (const record of records) console.log(renderAnalysis(record));
    if (detailed.battle_analysis.omitted) console.log(`${detailed.battle_analysis.omitted} decisions omitted by the capture budget`);
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[battle-analysis] ${error.message}`); process.exitCode = 1;
});

module.exports = {renderAnalysis};
