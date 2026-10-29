'use strict';

const {isDeepStrictEqual} = require('node:util');
const {performance} = require('node:perf_hooks');
const {Dex} = require('../external/pokemon-showdown/dist/sim/dex');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {restorePosition, advancePosition, playerObservation} = require('./position_simulator');
const {constructReplayOpening} = require('./opening_world');
const {nextSeed, choiceFailure} = require('./planner_beliefs');
const {mismatchEvidence} = require('./planner_evidence');

function projectionDifferences(actual, expected, path = '', output = []) {
    if (output.length >= 16 || isDeepStrictEqual(actual, expected)) return output;
    if (!actual || !expected || typeof actual !== 'object' || typeof expected !== 'object') {
        output.push({path, actual, expected}); return output;
    }
    for (const key of new Set([...Object.keys(actual), ...Object.keys(expected)]))
        projectionDifferences(actual[key], expected[key], `${path}/${key}`, output);
    return output;
}

function damageVariables(before, after) {
    const opponent = before.side === 'p1' ? 'p2' : 'p1', variables = new Map();
    const add = (ident, stat) => {
        if (ident?.startsWith(opponent)) {
            const key = ident.replace(/^(p[12])[ab]: /, '$1: ');
            variables.set(`${key}/${stat}`, {key, stat});
        }
    };
    let actor = null, move = null;
    for (const line of after.log.slice(before.log.length)) {
        const parts = line.split('|');
        if (parts[1] === 'move') { actor = parts[2]; move = Dex.moves.get(parts[3]); }
        if (parts[1] !== '-damage' || !move || !['Physical', 'Special'].includes(move.category) ||
                parts.length !== 4 || move.damage || move.damageCallback || move.overrideOffensiveStat ||
                move.overrideDefensiveStat || move.overrideOffensivePokemon || move.overrideDefensivePokemon) continue;
        add(actor, move.category === 'Physical' ? 'atk' : 'spa');
        add(parts[2], 'hp');
        add(parts[2], move.category === 'Physical' ? 'def' : 'spd');
    }
    // observed order motivates speed proposals without assuming equal move priority
    // the native queue still resolves abilities items ties and mid turn effects
    for (const line of after.log.slice(before.log.length)) {
        const parts = line.split('|');
        if (parts[1] === 'move') add(parts[2], 'spe');
    }
    return [...variables.values()];
}

function withInvestment(sets, {key, stat}, effective) {
    const proposal = structuredClone(sets), mon = proposal.find(entry => entry.key === key)?.set;
    if (!mon) throw new Error('joint constraint refers to a missing hypothetical pokemon');
    mon.ivs[stat] = Math.min(31, effective);
    mon.evs[stat] = Math.max(0, effective - 31) * 4;
    if (Object.values(mon.evs).reduce((sum, value) => sum + value, 0) > 508) return null;
    if (Object.values(mon.evs).every(value => value === 0)) mon.evs[stat] = 1;
    return proposal;
}

function transitionLoss(actual, expected, start) {
    if (isDeepStrictEqual(actual, expected)) return 0;
    // event alignment guides proposals without changing the acceptance comparison
    // separate damage occurrence counters retain multiple hits on the same target
    const events = view => {
        const counts = new Map(), result = new Map();
        for (const line of view.log.slice(start)) {
            const parts = line.split('|');
            const key = parts[1] === '-damage' ? parts.slice(0, 3).join('|') : line;
            const count = counts.get(key) || 0; counts.set(key, count + 1);
            result.set(`${key}/${count}`, line);
        }
        return result;
    };
    const a = events(actual), b = events(expected);
    let loss = 0;
    const order = view => view.log.slice(start).filter(line => line.startsWith('|move|'))
        .map(line => line.split('|').slice(2, 4).join('|'));
    const actualOrder = order(actual), expectedOrder = order(expected);
    for (let i = 0; i < expectedOrder.length; ++i)
        if (actualOrder[i] !== expectedOrder[i]) loss += 25;
    for (const key of new Set([...a.keys(), ...b.keys()])) {
        if (!a.has(key) || !b.has(key)) { loss += 100; continue; }
        if (a.get(key) === b.get(key)) continue;
        const health = line => {
            const condition = line.split('|')[3], match = /^(\d+)\/(\d+)/.exec(condition);
            return match ? Number(match[1]) / Number(match[2]) * 100 : condition === '0 fnt' ? 0 : null;
        };
        const ahp = health(a.get(key)), bhp = health(b.get(key));
        loss += ahp === null || bhp === null ? 100 : Math.abs(ahp - bhp);
    }
    // exact log order and private request equality remain mandatory even at a numeric optimum
    return Math.max(0.001, loss);
}

function jointSettings(options = {}) {
    const settings = {max_nodes: 2048, max_elapsed_ms: 5000, seeds: 16, particles: 1, seed: [351, 1, 2, 3], ...options};
    if (Object.keys(settings).some(key => !['max_nodes', 'max_elapsed_ms', 'seeds', 'particles', 'seed'].includes(key)) ||
            !Number.isInteger(settings.max_nodes) || settings.max_nodes < 1 || settings.max_nodes > 8192 ||
            !Number.isInteger(settings.max_elapsed_ms) || settings.max_elapsed_ms < 1 || settings.max_elapsed_ms > 60000 ||
            !Number.isInteger(settings.seeds) || settings.seeds < 1 || settings.seeds > 64 ||
            !Number.isInteger(settings.particles) || settings.particles < 1 || settings.particles > 4 ||
            !Array.isArray(settings.seed) || settings.seed.length !== 4 ||
            settings.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535))
        throw new Error('bounded joint transition settings required');
    return settings;
}

async function fitJointTransition(failure, gate, options = {}) {
    const settings = jointSettings(options);
    const trace = failure.simulation_trace, before = failure.history.observations[0];
    if (!trace || failure.history.observations.length !== 2 || before.turn !== 1)
        return {supported: false, reason: 'single_opening_transition_required'};
    if (!isDeepStrictEqual(trace.expected, failure.history.observations[1]) ||
            failure.history.own_choices[0] !== trace.choices[before.side])
        throw new Error('joint reconstruction must replay the recorded player boundary and accepted choice');
    const variables = damageVariables(before, trace.expected);
    if (!variables.length || variables.length > 8) return {supported: false, reason: 'bounded_ordinary_damage_variables_required'};
    const branch = restorePosition(trace.position, {mode: 'sample', seed: trace.seed});
    const opponent = before.side === 'p1' ? 'p2' : 'p1';
    let original;
    try {
        if (!isDeepStrictEqual(playerObservation(branch, before.side), before))
            throw new Error('joint reconstruction parent differs from the recorded initial observation');
        original = branch.getSide(opponent).pokemon.map(mon => ({key: mon.fullname, set: structuredClone(mon.set)}));
    }
    finally { branch.destroy(); }
    const started = performance.now(), rng = new PRNG(settings.seed.join(','));
    const report = {supported: true, settings, variables, proposals: 0, startup_matches: 0,
        native_replays: 0, seed_attempts: 0, best_loss: null, exact_transition_matches: 0,
        independently_verified: false, validation_gate_passed: false, witness: null, witnesses: [],
        calibrated: false, training_eligible: false, strength_claim: false};
    let best = null;
    const available = () => report.proposals < settings.max_nodes && performance.now() - started < settings.max_elapsed_ms;
    async function evaluate(sets, seed) {
        if (!available()) return null;
        if (report.proposals % 16 === 0) await (gate?.waitForCapacity?.() ?? gate?.waitForLaunch?.());
        if (!available()) return null;
        ++report.proposals;
        const opening = constructReplayOpening(before, trace.expected, {seed: nextSeed(rng),
            chance_seed: nextSeed(rng), attempts: 1, proposal_sets: sets});
        if (!opening.supported) return null;
        ++report.startup_matches;
        let next;
        try { ++report.native_replays; next = advancePosition(opening.position, trace.choices, {mode: 'sample', seed}); }
        catch (error) { if (choiceFailure(error)) return null; throw error; }
        const loss = transitionLoss(next.observations[before.side], trace.expected, before.log.length);
        const candidate = {sets, seed, position: opening.position, loss};
        if (!best || loss < best.loss) {
            best = candidate; report.best_loss = loss;
            report.best_mismatch = mismatchEvidence(next.observations[before.side], trace.expected);
            report.best_projection_differences = projectionDifferences(next.observations[before.side], trace.expected);
        }
        if (loss === 0) {
            // verify from the untouched opening with the ordinary native adapter again
            const verified = advancePosition(opening.position, trace.choices, {mode: 'sample', seed});
            if (!isDeepStrictEqual(verified.observations[before.side], trace.expected))
                throw new Error('joint constraint witness failed independent native replay');
            ++report.exact_transition_matches; report.independently_verified = true;
            const witness = {opening: opening.position, seed, choices: trace.choices,
                expected: trace.expected, position: verified.position};
            if (!report.witness) report.witness = witness;
            if (!report.witnesses.some(saved => saved.position.state_json === witness.position.state_json))
                report.witnesses.push(witness);
        }
        return candidate;
    }
    const complete = () => report.witnesses.length >= settings.particles;
    for (let attempt = 0; attempt < settings.seeds && available() && !complete(); ++attempt) {
        ++report.seed_attempts;
        const seed = attempt ? nextSeed(rng) : trace.seed;
        let current = await evaluate(best?.sets || original, seed);
        if (!current) continue;
        // hp and defense can sit on a shared knockout plateau
        // cross that plateau with joint proposals before the cheaper coordinate refinement
        for (const hp of variables.filter(variable => variable.stat === 'hp')) {
            const defense = variables.find(variable => variable.key === hp.key && ['def', 'spd'].includes(variable.stat));
            if (!defense) continue;
            const base = current.sets;
            for (const health of [0, 31, 63, 94]) for (const bulk of [0, 31, 63, 94]) {
                if (!available() || complete()) break;
                const first = withInvestment(base, hp, health);
                const proposal = first && withInvestment(first, defense, bulk);
                if (!proposal) continue;
                const candidate = await evaluate(proposal, seed);
                if (candidate && candidate.loss < current.loss) current = candidate;
            }
        }
        for (const step of [8, 2, 1]) {
            for (const variable of variables) {
                const mon = current.sets.find(entry => entry.key === variable.key).set;
                const center = mon.ivs[variable.stat] + Math.floor(mon.evs[variable.stat] / 4);
                const values = step === 8 ? [0, 16, 31, 40, 48, 56, 64, 72, 80, 88, 94] :
                    [center - 2 * step, center - step, center + step, center + 2 * step];
                for (const value of values) {
                    if (!available() || complete()) break;
                    if (value < 0 || value > 94) continue;
                    const proposal = withInvestment(current.sets, variable, value);
                    if (!proposal) continue;
                    const candidate = await evaluate(proposal, seed);
                    if (candidate && candidate.loss < current.loss) current = candidate;
                }
            }
        }
    }
    report.elapsed_seconds = (performance.now() - started) / 1000;
    report.budget_exhausted = !available();
    report.validation_gate_passed = report.independently_verified;
    report.assessment = report.validation_gate_passed ? 'complete_native_transition_verified' : 'no_exact_transition_within_joint_fit_budget';
    return report;
}

module.exports = {damageVariables, withInvestment, transitionLoss, fitJointTransition, jointSettings};
