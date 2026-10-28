# Position simulation: first hybrid-engine experiment

This adapter operates on **explicit hypothetical complete states**. It is not a
live planner, belief model, or strength improvement. No checkpoint is loaded or
changed. Fixtures use hand-authored teams; no published sets or random-team
generator are used. Team builds in these fixtures are mechanics test inputs,
not a claim that they follow the random generator's distribution.

## Interface

`tools/position_simulator.js` wraps the pinned local Showdown implementation for
`gen9randomdoublesbattle`:

- `capturePosition(battle, origin)` saves a detached JSON state. Origin is
  `fixture` or `belief-sample`, never an inferred guarantee about its contents.
- `restorePosition(position, randomness)` creates an independent simulator.
  `fixture-replay` restores saved chance state only for fixtures; `sample`
  requires a separately supplied four-word seed.
- `advancePosition(position, choices, randomness)` takes every outstanding
  complete side choice upfront, simulates to the next request or terminal
  state, and returns a new hypothetical position and both player projections.
  Rejected or incomplete choices fail on a disposable branch, leaving the
  caller's position unchanged.
- `playerObservation(battle, side)` returns that side's request and filtered
  protocol history, not opposing requests, input logs, hidden teams, or RNG.

Snapshots are privileged backend objects. Do not pass them to a player or call
capture on the real opponent's hidden state to populate future search worlds.
The origin label is an API convention, not a security boundary. Enforcing world
construction from permitted observations is the next implementation stage.

Showdown normally regenerates requests during restoration. This adapter saves
and reinstates the exact requests instead, retaining same-phase updates and
avoiding request regeneration's possible trap/knowledge-cache mutations. It
also serializes immediately to detach live logs, sets, and other shared data.

The adapter preserves simulator PRNG state, pending choices, roster ordering,
PP and mid-turn queues. It does not yet own neural hidden states or policy RNG;
future player integration must fork those separately. It handles requests one
phase at a time rather than automatically choosing forced replacements.

## Small checks

From the repository root:

```powershell
node --max-old-space-size=256 --test --test-concurrency=1 .\tools\position_simulator.test.js
node --max-old-space-size=256 .\tools\position_simulator_experiment.js --run-name run_0322_position_simulator
```

The experiment refuses to overwrite an existing report; choose a new run name
for a rerun. Results are written atomically to
`models/diagnostics/<run-name>/report.json`.

Eleven deterministic scenarios compare restored branches against uninterrupted
Showdown execution: targeting/PP, Protect, Helping Hand, ally/spread damage,
Tera/priority/speed, voluntary switches, forced replacements, pivot queues,
pending choices, a lone surviving slot, and a terminal result. Unit tests also
cover request fidelity, branch isolation, missing targets, trapped switches,
independent chance seeds and player-specific projections.

The experiment additionally checks 192 independently seeded Thunder branches
against uninterrupted references and checks its 70% accuracy with a broad
six-standard-error sanity bound. These are repeated chance samples of one
fixture, not 192 independent battle positions or evidence of learning.

Execution is serial and uses the existing 65% CPU/memory scheduling gates.
Those gates are not OS-enforced resource limits. The active experiment is capped
at 30 seconds, checked between samples; resource waits can extend wall time.
Timings have explicit units. The report records loaded simulator/adapter hashes
and rejects loaded random-team/set-generation modules. This is not a complete
audit of arbitrary file reads or provenance of historical model weights.

The observation/fact and set-sampling boundary is documented in
[belief_state.md](belief_state.md). An initial-turn-only world constructor is
documented in [opening_world.md](opening_world.md). Later-turn construction and
a shallow simultaneous-choice planner remain gates; neither is claimed ready
by a successful fixture report.
