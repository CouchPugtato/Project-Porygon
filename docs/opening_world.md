# Observation-consistent opening worlds

`tools/opening_world.js` constructs complete hypothetical simulator positions
from a player projection, without taking a live Battle or snapshot as input.
The supported scope is the **initial turn-one move request**. Later turns and
already-played actions are rejected rather than reconstructed with missing PP,
volatile effects or hidden history.

## Construction

`constructOpeningWorld(view, {seed, chance_seed, attempts})`:

1. Extracts visible opponent facts through `observeBeliefs` and checks the
   opening-only boundary.
2. Reconstructs the friendly roster from the private request: names, species,
   level, gender, moves, item, ability, Tera type and displayed base stats.
3. Finds one legal IV/EV/nature combination yielding those displayed stats.
   Multiple original builds can be equivalent; this does not recover the true
   combination. Stat inversion uses Showdown's stat formula and validates the
   resulting set.
4. Samples opposing leads and unrevealed reserves using rules, not published
   sets. Revealed leads retain their observed species, level and gender.
   Reserves use broad species/level proposals, with species-clause exclusions.
5. Validates both complete teams under the same format rules. The validation
   copy disables only random-team delivery, so validation never invokes a
   random generator. It does not remove battle or obtainability rules.
6. Starts a fresh, independently seeded Showdown battle with those explicit
   teams. Startup abilities, weather, field effects and requests come from
   simulation, not patches copied from the real battle.
7. Accepts the world only if its original player's entire projection matches
   the supplied observation. Startup contradictions reject the proposal.

The proposal and startup-chance seeds are caller-owned four-word seeds, not
live future RNG. The constructor allows 1-32 world attempts, default 16.
Each attempted roster has a bounded legality-completion budget. Exhaustion
returns `supported: false`, a null position and explicit reasons/rejection
counts. There is no automatic fallback to the actual hidden opponent team.

## Result and limits

Accepted results have `simulation_ready: true`, an opening-only scope and a
`belief-sample` position for the existing adapter. `search_ready` and
`training_eligible` remain false. Origins are API conventions, not a sandbox.

The original player sees exactly its original startup view. A simulated
opponent receives its own hypothetical private team through Showdown's normal
player projection; it must never receive the omniscient snapshot as an actor
input. Future player-policy integration still needs that boundary tested.

Unknown reserves can be sampled even though their true identities were never
observed. These are legal proposals, not estimates of random-generator
frequencies. Startup rejection conditions the proposal on the observed startup,
but does not make its weights a calibrated posterior. No learned inference
model or published set list is used.

Strict matching can abstain for unsupported forms, private-request fields,
metadata variations or low proposal acceptance. Do not silently remove fields
from the comparison merely to improve coverage. Identity/disguise changes and
later dynamic positions remain unsupported. Equal displayed own stats also do
not prove that all possible later form/stat-recalculation mechanics are
equivalent across inferred latent builds.

## Bounded experiment

```powershell
node --max-old-space-size=256 --test --test-concurrency=1 .\tools\opening_world.test.js
node --max-old-space-size=256 .\tools\opening_world_experiment.js --run-name run_0324_opening_worlds
```

The experiment constructs and advances six opening cases: both perspectives
with three and six Pokemon, a non-neutral invested level-80 build, and startup
weather. Sixteen changed-hidden-world pairs must construct identical normalized
positions when the player views and proposal/chance seeds match. Three checks
verify explicit abstention on later history, incompatible private HP and
impossible startup evidence.

The fixture-only continuation helper supplies explicit legal targets to
Showdown's automatic choices. It is not a new live policy or complete search
action enumerator. Tests also cover input/root immutability, complete team
legality, independent continuation seeds and snapshot-field rejection.

Execution is serial, with the shared CPU/memory scheduling gates and a
30-second between-case budget check. Reports pin sources and loaded simulator
files, refuse overwrite, and retain `strength_claim: false`. No checkpoint is
loaded, trained, promoted or deployed.

First results: run 0324 passed all six opening/continuation cases, sixteen
invariance pairs and three expected abstentions. This is a correctness and
coverage experiment, not a battle win-rate measurement.
