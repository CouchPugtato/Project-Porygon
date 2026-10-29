'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const {performance} = require('node:perf_hooks');
const {publishExperiment} = require('./position_simulator_experiment');
const {playFixtureGame, validateSaved} = require('./hybrid_cycle_experiment');
const {writeJsonAtomic} = require('./counterfactual_rollout');

async function coverageExperiment(gate, context) {
    const started = performance.now(), report = {diagnostic: 'matched_belief_coverage_ablation',
        status: 'running', passed: false, strength_claim: false, training_eligible: false, profiles: [], failure: null};
    try {
        for (const recovery of [false, true]) {
            const profile = {recovery, games: []}; report.profiles.push(profile);
            for (const seed of [33401, 33901]) {
                const config = {games: 2, seed, max_boundaries: 160, experiment_budget_seconds: 60,
                    beliefs: {particles: 1, max_nodes: 64, max_elapsed_ms: 100, max_boundaries: 160,
                        recovery, filter_evidence: recovery},
                    search: {candidates: 4, responses: 2, chance_samples: 1, continuation_choices: 1,
                        max_nodes: 32, max_elapsed_ms: 100}};
                const filename = path.join(context.directory, `games_${recovery}_${seed}.json`);
                let games = [];
                try { games = validateSaved(JSON.parse(await fs.readFile(filename, 'utf8')), context.identity, config); }
                catch (error) { if (error.code !== 'ENOENT') throw error; }
                for (let index = games.length; index < config.games; ++index) {
                    const game = await playFixtureGame(config, index, gate, () => {
                        if (performance.now() - started > 60000) throw new Error('60s coverage experiment budget exhausted');
                    });
                    games.push(game);
                    await writeJsonAtomic(filename, {identity: context.identity, config, games});
                    if (game.status !== 'completed') throw new Error(game.failure);
                    console.log(`[hybrid-coverage] recovery=${recovery} seed=${seed} games=${games.length}/2 elapsed=${((performance.now() - started) / 1000).toFixed(1)}s`);
                }
                profile.games.push(...games);
            }
            profile.decisions = profile.games.reduce((sum, game) => sum + game.coverage.decisions, 0);
            profile.searched = profile.games.reduce((sum, game) => sum + game.coverage.searched, 0);
            profile.search_rate = profile.searched / profile.decisions;
            profile.rejections = {};
            for (const game of profile.games) for (const [kind, count] of Object.entries(game.coverage.belief_rejections))
                profile.rejections[kind] = (profile.rejections[kind] || 0) + count;
        }
        report.coverage_improved = report.profiles[1].search_rate > report.profiles[0].search_rate;
        report.status = 'completed'; report.passed = true;
    } catch (error) { report.status = 'failed'; report.failure = error.stack || error.message; }
    report.elapsed_seconds = (performance.now() - started) / 1000;
    report.observed_process_rss_mib = process.memoryUsage().rss / 1024 ** 2;
    report.limitations = ['Matched initial seeds do not keep trajectories identical after decisions diverge.',
        'Fresh collection seed is a coverage check rather than a strength confirmation.',
        'Budget exhaustion and overlapping diagnostic categories do not identify unique hidden stat values.'];
    return report;
}

if (require.main === module) {
    if (process.argv.length !== 4 || process.argv[2] !== '--run-name') throw new Error('expected --run-name');
    publishExperiment(process.argv[3], coverageExperiment, ['tools/hybrid_coverage_experiment.js',
        'tools/hybrid_cycle_experiment.js', 'tools/hybrid_planner.js', 'tools/planner_beliefs.js',
        'tools/planner_evidence.js', 'tools/planner_recovery.js', 'tools/planner_search.js', 'tools/planner_matrix.js',
        'tools/planner_value.js', 'tools/planner_actions.js', 'tools/opening_actions.js', 'tools/opening_world.js',
        'tools/opening_proposals.js', 'tools/belief_state.js', 'tools/replacement_actions.js',
        'tools/fixtures/hybrid_cycle.js', 'tools/hybrid_expansion.test.js']).catch(error => {
        console.error(error.stack || error.message); process.exitCode = 1;
    });
}

module.exports = {coverageExperiment};
