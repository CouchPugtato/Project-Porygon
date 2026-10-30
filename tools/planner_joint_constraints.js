'use strict';

const {isDeepStrictEqual} = require('node:util');
const {performance} = require('node:perf_hooks');
const {Dex} = require('../external/pokemon-showdown/dist/sim/dex');
const {PRNG} = require('../external/pokemon-showdown/dist/sim/prng');
const {restorePosition, advancePosition, playerObservation} = require('./position_simulator');
const {constructReplayOpening} = require('./opening_world');
const {nextSeed, choiceFailure, observedResponses} = require('./planner_beliefs');
const {observeBeliefs, sampleRevealedSets} = require('./belief_state');
const {mismatchEvidence} = require('./planner_evidence');
const {UnsupportedPlannerRequest} = require('./planner_actions');

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
    const settings = {max_nodes: 2048, max_elapsed_ms: 5000, seeds: 16, particles: 1,
        max_boundaries: 8, max_attempts: 3, max_total_ms: 10000, persistent: false, seed: [351, 1, 2, 3], ...options};
    if (Object.keys(settings).some(key => !['max_nodes', 'max_elapsed_ms', 'seeds', 'particles', 'seed',
        'max_boundaries', 'max_attempts', 'max_total_ms', 'persistent'].includes(key)) || typeof settings.persistent !== 'boolean' ||
            !Number.isInteger(settings.max_nodes) || settings.max_nodes < 1 || settings.max_nodes > 8192 ||
            !Number.isInteger(settings.max_elapsed_ms) || settings.max_elapsed_ms < 1 || settings.max_elapsed_ms > 60000 ||
            !Number.isInteger(settings.seeds) || settings.seeds < 1 || settings.seeds > 64 ||
            !Number.isInteger(settings.particles) || settings.particles < 1 || settings.particles > 4 ||
            !Number.isInteger(settings.max_boundaries) || settings.max_boundaries < 2 || settings.max_boundaries > 16 ||
            !Number.isInteger(settings.max_attempts) || settings.max_attempts < 1 || settings.max_attempts > 8 ||
            !Number.isInteger(settings.max_total_ms) || settings.max_total_ms < 1 || settings.max_total_ms > 30000 ||
            !Array.isArray(settings.seed) || settings.seed.length !== 4 ||
            settings.seed.some(n => !Number.isInteger(n) || n < 0 || n > 65535))
        throw new Error('bounded joint transition settings required');
    return settings;
}

async function fitJointTransition(failure, gate, options = {}) {
    const settings = jointSettings(options);
    const trace = failure.simulation_trace, before = failure.history.observations[0];
    const {observations, own_choices: ownChoices} = failure.history;
    if (!trace || observations.length < 2 || observations.length > settings.max_boundaries || before.turn !== 1)
        return {supported: false, reason: 'bounded_opening_history_required'};
    if (!isDeepStrictEqual(trace.expected, observations.at(-1)) || ownChoices.length !== observations.length - 1 ||
            ownChoices.at(-1) !== (trace.choices[before.side] ?? null))
        throw new Error('joint reconstruction must replay the recorded player boundary and accepted choice');
    if (ownChoices.some(choice => choice !== null && typeof choice !== 'string'))
        return {supported: false, reason: 'rejected_choice_history_requires_disclosure_replay'};
    const variables = [...new Map(observations.slice(1).flatMap((view, i) => damageVariables(observations[i], view))
        .map(variable => [`${variable.key}/${variable.stat}`, variable])).values()];
    if (!variables.length || variables.length > 24) return {supported: false, reason: 'bounded_ordinary_damage_variables_required'};
    const source = failure.opening_position || trace.position;
    const branch = restorePosition(source, {mode: 'sample', seed: trace.seed});
    const opponent = before.side === 'p1' ? 'p2' : 'p1';
    let original;
    try {
        if (!isDeepStrictEqual(playerObservation(branch, before.side), before))
            throw new Error('joint reconstruction parent differs from the recorded initial observation');
        original = branch.getSide(opponent).pokemon.map(mon => ({key: mon.fullname, set: structuredClone(mon.set)}));
    }
    finally { branch.destroy(); }
    const facts = observeBeliefs(trace.expected);
    const missing = facts.opponent.pokemon.filter(known => !original.some(entry => entry.key === known.key));
    if (missing.length) {
        // replace only never revealed reserve proposals at the opening
        // replay of the entire history must validate the newly proposed roster
        const completion = sampleRevealedSets({...facts, opponent: {...facts.opponent, pokemon: missing,
            unrevealed_count: 0}}, {seed: settings.seed, samples: 1, attempts_per_set: 32});
        const sampled = completion.particles[0]?.sets;
        if (!sampled) return {supported: false, reason: 'new_reserve_legal_completion_failed'};
        for (const known of missing) {
            const index = original.findIndex(entry => !facts.opponent.pokemon.some(mon => mon.key === entry.key));
            if (index < 0) return {supported: false, reason: 'new_reserve_exceeds_opening_roster'};
            original[index] = sampled.find(entry => entry.key === known.key);
        }
    }
    for (const known of facts.opponent.pokemon) {
        const mon = original.find(entry => entry.key === known.key)?.set;
        mon.moves = [...new Set([...known.moves, ...mon.moves])].slice(0, 4);
        if (known.ability) mon.ability = known.ability;
        if (known.tera_type) mon.teraType = known.tera_type;
        if (known.item) mon.item = known.item.last_revealed;
    }
    const started = performance.now(), rng = new PRNG(settings.seed.join(','));
    const report = {supported: true, settings, variables, proposals: 0, startup_matches: 0,
        native_replays: 0, seed_attempts: 0, best_loss: null, exact_transition_matches: 0,
        independently_verified: false, validation_gate_passed: false, witness: null, witnesses: [],
        calibrated: false, training_eligible: false, strength_claim: false};
    report.history_boundaries = observations.length;
    const openings = new Map(), prefixes = new Map();
    let openingBytes = 0, prefixBytes = 0;
    const fixedPrefix = observations.slice(1, -1).every((_, i) => failure.replay_steps?.[i]?.seed);
    report.opening_cache_hits = 0; report.prefix_cache_hits = 0;
    let best = null;
    const available = () => report.proposals < settings.max_nodes && performance.now() - started < settings.max_elapsed_ms;
    async function evaluate(sets, seed) {
        if (!available()) return null;
        if (report.proposals % 16 === 0) await (gate?.waitForCapacity?.() ?? gate?.waitForLaunch?.());
        if (!available()) return null;
        ++report.proposals;
        const key = JSON.stringify(sets);
        let opening = settings.persistent && openings.get(key);
        if (opening) ++report.opening_cache_hits;
        else {
            opening = constructReplayOpening(before, trace.expected, {seed: nextSeed(rng),
                chance_seed: nextSeed(rng), attempts: 1, proposal_sets: sets});
            const bytes = opening.supported ? Buffer.byteLength(opening.position.state_json) + Buffer.byteLength(key) : 0;
            if (settings.persistent && opening.supported && openingBytes + bytes <= 1024 ** 2) {
                openings.set(key, opening); openingBytes += bytes;
            }
        }
        if (!opening.supported) return null;
        ++report.startup_matches;
        let next, position = opening.position, loss = 0, mismatch = null;
        const steps = [];
        const prefixKey = `${report.seed_attempts}/${key}`, cached = settings.persistent && fixedPrefix && prefixes.get(prefixKey);
        if (cached) {
            position = cached.position; steps.push(...cached.steps); ++report.prefix_cache_hits;
        }
        for (let i = steps.length + 1; i < observations.length; ++i) {
            const prior = observations[i - 1], expected = observations[i];
            const sampledSeed = i === observations.length - 1 ? seed : failure.replay_steps?.[i - 1]?.seed || seed;
            const battle = restorePosition(position, {mode: 'sample', seed: sampledSeed});
            let choices;
            try {
                const responses = observedResponses(battle, prior, expected);
                if (!responses.length) return null;
                const preferred = observations.length === 2 ? trace.choices[opponent] :
                    failure.replay_steps?.[i - 1]?.choices?.[opponent];
                const response = responses.includes(preferred) ? preferred : responses[report.seed_attempts % responses.length];
                choices = {};
                if (ownChoices[i - 1] !== null) choices[before.side] = ownChoices[i - 1];
                if (response !== null) choices[opponent] = response;
            } catch (error) {
                if (!(error instanceof UnsupportedPlannerRequest)) throw error;
                report.unsupported_opponent_requests = (report.unsupported_opponent_requests || 0) + 1;
                return null;
            } finally { battle.destroy(); }
            try { ++report.native_replays; next = advancePosition(position, choices, {mode: 'sample', seed: sampledSeed}); }
            catch (error) { if (choiceFailure(error)) return null; throw error; }
            steps.push({seed: sampledSeed, choices});
            loss = transitionLoss(next.observations[before.side], expected, prior.log.length);
            if (loss !== 0) {
                loss += (observations.length - 1 - i) * 10000;
                mismatch = {actual: next.observations[before.side], expected, boundary: i}; break;
            }
            position = next.position;
            if (settings.persistent && fixedPrefix && i === observations.length - 2 && !prefixes.has(prefixKey)) {
                // only unchanged builds and fixed earlier native draws share a verified prefix
                // final acceptance still replays the complete witness independently
                const bytes = Buffer.byteLength(position.state_json) + Buffer.byteLength(prefixKey);
                if (prefixBytes + bytes <= 2 * 1024 ** 2) {
                    prefixes.set(prefixKey, {position, steps: [...steps]}); prefixBytes += bytes;
                }
            }
        }
        const candidate = {sets, seed, position: opening.position, loss, steps};
        if (!best || loss < best.loss) {
            best = candidate; report.best_loss = loss;
            report.matched_boundaries = mismatch ? mismatch.boundary - 1 : observations.length - 1;
            report.best_mismatch = mismatch ? mismatchEvidence(mismatch.actual, mismatch.expected) : {kind: 'matched', distance: 0};
            report.best_projection_differences = mismatch ? projectionDifferences(mismatch.actual, mismatch.expected) : [];
        }
        if (loss === 0) {
            // verify from the untouched opening with the ordinary native adapter again
            let verified;
            position = opening.position;
            for (const [i, step] of steps.entries()) {
                verified = advancePosition(position, step.choices, {mode: 'sample', seed: step.seed});
                if (!isDeepStrictEqual(verified.observations[before.side], observations[i + 1]))
                    throw new Error('joint constraint witness failed independent native replay');
                position = verified.position;
            }
            ++report.exact_transition_matches; report.independently_verified = true;
            const witness = {opening: opening.position, seed, choices: steps.at(-1).choices, steps,
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
        let seed = attempt ? nextSeed(rng) : trace.seed;
        let current = await evaluate(best?.sets || original, seed);
        if (!current) continue;
        if (attempt === 0 && observations.length > 2 && !complete()) {
            // retain verified prefix draws while searching new chance outcomes at this boundary
            // this avoids spending an entire stat grid on an incompatible critical or protection draw
            report.chance_trials = 0;
            for (let trial = 0; trial < 32 && available() && !complete(); ++trial) {
                ++report.chance_trials;
                await evaluate(current.sets, nextSeed(rng));
            }
            if (best && best.loss < current.loss) { current = best; seed = best.seed; }
        }
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
