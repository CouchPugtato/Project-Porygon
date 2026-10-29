'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {createResourceGate} = require('./counterfactual_rollout');
const {searchBeliefWorlds} = require('./planner_search');
const {createHybridFixture} = require('./fixtures/hybrid_cycle');
const {capturePosition, playerObservation} = require('./position_simulator');

test('capacity checks do not consume launch staggering and actual launches remain staggered', async () => {
    const gate = createResourceGate({maxSystemCpuPercent: 100, maxSystemMemoryPercent: 100,
        resourceCheckSeconds: 0.01, launchStaggerMilliseconds: 100});
    await gate.waitForLaunch();
    const started = performance.now();
    await gate.waitForCapacity();
    assert(performance.now() - started < 80, 'capacity check unexpectedly waited for a launch');
    await gate.waitForLaunch();
    assert(performance.now() - started >= 75, 'actual launch stagger was lost');
});

test('in process planner simulations use capacity checks instead of launching more workers', async () => {
    const battle = createHybridFixture();
    let checks = 0;
    const gate = {async waitForCapacity() { ++checks; }, async waitForLaunch() {
        throw new Error('search attempted a process launch gate');
    }};
    try {
        const result = await searchBeliefWorlds(playerObservation(battle, 'p1'), [capturePosition(battle, 'fixture')],
            {seed: [340, 1, 2, 3], candidates: 2, responses: 1, chance_samples: 1,
                continuation_choices: 1, max_nodes: 32, max_elapsed_ms: 3000}, gate);
        assert(result.supported); assert(result.nodes > 0); assert(checks > 0);
        assert(result.capacity_wait_ms >= 0);
    } finally { battle.destroy(); }
});

test('real capacity pauses still exhaust the wall clock budget without promoting partial work', async () => {
    const battle = createHybridFixture();
    const gate = {async waitForCapacity() { await new Promise(resolve => setTimeout(resolve, 20)); }};
    try {
        const result = await searchBeliefWorlds(playerObservation(battle, 'p1'), [capturePosition(battle, 'fixture')],
            {seed: [340, 1, 2, 3], candidates: 2, responses: 1, chance_samples: 1,
                max_nodes: 32, max_elapsed_ms: 10}, gate);
        assert(!result.supported); assert.equal(result.nodes, 0);
        assert(result.budget_exhausted); assert.equal(result.budget_stage, 'capacity_check_or_elapsed_limit');
    } finally { battle.destroy(); }
});
