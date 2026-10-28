# Observation-based beliefs: initial boundary

`tools/belief_state.js` consumes the player projections from
`position_simulator.js`. It never receives a Battle object or complete-state
snapshot. It records visible facts and generates legal set completions for
revealed opposing Pokemon. It does **not** yet construct complete worlds or
provide a planner-ready posterior.

## Facts and uncertainty

`observeBeliefs(view)` rebuilds facts from the complete player-visible history.
Rebuilding makes repeated calls idempotent and avoids stale incremental state.
The input must be a player observation, with the request belonging to that
player; extra snapshot fields are rejected. This schema check is not a security
boundary against a caller deliberately fabricating observations.

Tracked evidence includes observed species, level, gender, active positions,
team size, ordinary used moves, revealed abilities/items, consumed items,
revealed Tera type, status, fainting, and public HP history. Effect-source
ownership matters: damage caused by an opponent's Rocky Helmet does not reveal
the damaged player's item.

HP is a public fraction interval, not exact opponent HP. The pinned Gen 9 rule
uses ceiling percentages, except damaged Pokemon cannot display 100%. Thus
`99/100` means more than 98% but less than 100%, while `100/100` means full HP.
The current implementation retains damage evidence but does not infer stat
spreads, damage rolls, or ability probabilities from it.

Unrevealed teammates remain unknown. Apparent identity can be deceptive.
Disguise/form changes, indirect move evidence, ability changes and other
unsupported cases produce explicit issues rather than a confident completion.
This is not a second complete battle-state reconstructor: field effects,
volatiles, move-slot ordering and exact PP remain outside this boundary.

## Rule-based proposals, not a calibrated posterior

`sampleRevealedSets(beliefs, {seed, samples, attempts_per_set})` generates set
hypotheses for the revealed Pokemon. Defaults are 16 particles and at most 32
attempts per set; both have hard upper bounds of 64. The four-word sampling seed
is caller-owned and independent of the live simulator's future RNG.

The initial proposal is deliberately broad:

- Unknown moves are drawn without replacement from the native Gen 9 learnset,
  retaining ordinary observed moves.
- Unknown abilities use the species' native abilities; unknown items use legal
  item definitions, including no held item.
- Unknown Tera types and natures are drawn from the rules' available types and
  natures. Observed level and gender are retained.
- IVs are drawn uniformly from 0 through 31. EVs use a uniform budget of 0
  through 127 four-EV increments, allocated among stats without exceeding 252.
  One additional, stat-neutral EV expresses intentional spreads to Showdown's
  validator, including the zero-investment and level-50 cases.
- Every complete proposed set passes Showdown's format set validator. The
  learnset union alone would not establish inter-move compatibility.

These choices define a proposal distribution, not learned opponent frequencies.
Equal particle weights do not mean every legal set is equally likely, and do
not establish a posterior. Damage and opponent action likelihoods do not yet
reweight particles. Newly revealed facts cause proposals to be regenerated;
there is no published-set catalog to lock out an unfamiliar legal move.

No published sets, generator-derived distributions, model checkpoints or
historical caches are loaded. Rules such as learnsets and ability definitions
are permitted inputs, not competitive build lists.

All results retain `world_ready: false` and `training_eligible: false`.
Unrevealed teammates, dynamic state and exact HP/PP must be handled before
building full hypothetical simulator positions. Unsupported evidence or an
exhausted completion budget yields no particles with an explicit reason.
Unknown teammates are separately reported even when partial set sampling works.
The explicit `include_unrevealed: true` option additionally proposes reserves
from native legal species and levels 1-100, avoiding already selected base
species. Each set still passes validation; complete team legality is checked
by the opening-world constructor. This option does not use generator rules or
claim calibrated species/level frequencies. Default partial sampling is unchanged.

## Checks and experiment

```powershell
node --max-old-space-size=256 --test --test-concurrency=1 .\tools\belief_state.test.js
node --max-old-space-size=256 .\tools\belief_state_experiment.js --run-name run_0323_observation_beliefs
```

The experiment checks 32 changed hidden worlds against the same player view,
using eight particles per observation. It then checks cumulative visible move,
ability, item and Tera updates, HP intervals, and five explicit abstentions.
Tests separately check consumed items, source ownership, both player sides,
unusual legal moves, level 50, deterministic independent sampling, validator
acceptance and rejection of privileged snapshot fields.

Reports reuse the position experiment's locking, source hashing, atomic writes
and 65% CPU/memory scheduling gates. Run names cannot overwrite prior reports.
Execution is serial, with a 30-second between-batch experiment check. Gates are
not OS percentage caps; this does not guarantee protection from PC crashes.
The loaded-module check detects random-team/set-generation modules, not every
possible file access or the provenance of unrelated historical weights.

The opening-only world constructor is documented in
[opening_world.md](opening_world.md). It builds a new simulation and requires
exact player-view agreement; it never copies the real hidden opponent state.
Later dynamic world reconstruction, search and strength testing remain gates.
