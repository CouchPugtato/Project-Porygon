'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {publishExperiment} = require('./position_simulator_experiment');

function auditCoverage(games) {
    const profiles = {};
    const add = (map, key, count = 1) => { map[key] = (map[key] || 0) + count; };
    for (const game of games) {
        if (!game.hybrid_coverage) continue;
        const stats = game.hybrid_coverage;
        const profile = profiles[game.profile] ||= {games: 0, invalid: 0, decisions: 0, search_completed: 0,
            games_beyond_turn_three: 0, elapsed_seconds: 0, recovery_attempts: 0, verified_recoveries: 0,
            recovery_seconds: 0, mismatch_counts: {}, proposal_rejections: {}, unsupported_evidence: {},
            stage_counts: {}, decision_failures: {}, lost_boundary_count: 0, chance_nodes: 0, fit_nodes: 0};
        ++profile.games;
        if (game.status !== 'completed' || game.choice_rejections?.length) ++profile.invalid;
        profile.decisions += stats.decisions || 0;
        profile.search_completed += stats.search_completed || 0;
        profile.elapsed_seconds += game.elapsed_seconds || 0;
        profile.games_beyond_turn_three += Number(stats.latest_verified_turn > 3);
        profile.recovery_seconds += stats.reconstruction_seconds || 0;
        for (const result of stats.reconstruction_results || []) {
            ++profile.recovery_attempts;
            profile.chance_nodes += result.chance_nodes || 0;
            profile.fit_nodes += result.fit_nodes || 0;
            if (result.verified) ++profile.verified_recoveries;
            else if (result.mismatch) add(profile.mismatch_counts, result.mismatch.kind);
            else for (const [kind, count] of Object.entries(result.reasons || {[result.reason]: 1}))
                add(profile.mismatch_counts, kind, count);
            add(profile.stage_counts, result.stage || 'legacy_history_fit');
        }
        for (const [key, count] of Object.entries(stats.belief_rejections || {})) add(profile.proposal_rejections, key, count);
        for (const [key, count] of Object.entries(stats.unsupported_evidence || {})) add(profile.unsupported_evidence, key, count);
        for (const [key, count] of Object.entries(stats.reasons || {})) add(profile.decision_failures, key, count);
        profile.lost_boundary_count += stats.belief_loss_count ?? stats.belief_losses?.length ?? 0;
    }
    for (const profile of Object.values(profiles)) {
        profile.search_rate = profile.decisions ? profile.search_completed / profile.decisions : 0;
        profile.late_coverage_rate = profile.games_beyond_turn_three / profile.games;
        profile.mean_game_seconds = profile.elapsed_seconds / profile.games;
    }
    return profiles;
}

function compareCoverage(games, control = 'guarded', candidate = 'persistent') {
    const profiles = auditCoverage(games), before = profiles[control], after = profiles[candidate];
    const ready = before?.games >= 8 && after?.games >= 8;
    return {profiles, control, candidate, enough_screen_games: !!ready,
        advancement_gate_passed: !!(ready && !before.invalid && !after.invalid && after.late_coverage_rate >= 0.25 &&
            after.search_rate > before.search_rate * 1.25 && after.mean_game_seconds <= before.mean_game_seconds * 1.2),
        meaning: 'engineering coverage screen rather than a strength or statistical confidence claim',
        strength_claim: false};
}

async function main(argv) {
    if (argv.length !== 4 || argv[0] !== '--run-name' || argv[2] !== '--input')
        throw new Error('expected --run-name NAME --input BENCHMARK_JSON');
    const filename = path.resolve(argv[3]);
    if ((await fs.stat(filename)).size > 16 * 1024 ** 2) throw new Error('coverage summary exceeds the 16 MiB limit');
    const report = JSON.parse(await fs.readFile(filename, 'utf8'));
    const {recordHash} = require('./hybrid_benchmark');
    if (report.games.some(game => game.record_hash !== recordHash(game))) throw new Error('coverage record integrity changed');
    return publishExperiment(argv[1], async () => ({diagnostic: 'belief_coverage_audit', passed: true,
        ...compareCoverage(report.games), historical_confirmed_random_win_rate: 0.597,
        limitation: 'Proposal rejection counts are repeated simulation attempts rather than independent lost decisions.'}),
    ['tools/belief_coverage_audit.js', path.relative(path.resolve(__dirname, '..'), filename)]);
}

if (require.main === module) main(process.argv.slice(2)).catch(error => {
    console.error(`[belief-coverage] ${error.stack || error.message}`); process.exitCode = 1;
});

module.exports = {auditCoverage, compareCoverage};
