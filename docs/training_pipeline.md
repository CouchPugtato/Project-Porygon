# Training Pipeline

The trainable path is staged:

1. Python communicator captures or streams Showdown battle traffic.
2. The C runtime reconstructs battle state from `request` payloads and battle log `event` lines.
3. The observation builder converts raw state into the full one-hot `Observation` format.
4. Each decision point is appended to an `Episode`.
5. Live inference uses the GRU model to choose an action and emits `/choose ...`.
6. Replay logs preserve raw protocol messages and agent `decision` records.
7. Offline supervised training replays those logs, rebuilds episodes, and updates the GRU over sequence windows.
8. Checkpoints serialize model weights and trainer state.
9. RL can either reuse reconstructed episodes offline or consume live-exported `episode_complete` batches directly.

Supervised validation reports `metrics_version=2`. Doubles turns are scored from the
legal 196-way joint policy used during live play, while single-action turns are scored
only against legal actions for their active slot. Slot accuracy comes from the joint
policy marginals, and target accuracy counts only decisions where a target choice
exists. The older `accuracy`, `top3_accuracy`, `action1_accuracy`, and
`action2_accuracy` fields remain as aliases for full-turn, legal decision top-3,
slot-0, and slot-1 accuracy respectively. The legacy `action_loss` field is the
full-turn negative log-likelihood, including any required target decisions.

## Trainer learnability checks

Supervised training uses SGD by default for checkpoint compatibility. Pass
`--supervised-optimizer adam` to either the C trainer or
`py/tools/train_batch_selfplay.py` when an experiment explicitly calls for it.

Before another full supervised run or PPO search, run the isolated overfit
diagnostic on an existing replay collection:

```powershell
.\build-fresh\showdown_client.exe --check-supervised-overfit .\matches\runs\collection\worker_1_a_raw.jsonl .\models\diagnostics\supervised_overfit.json
```

The command starts a fresh current-architecture model, deterministically chooses
two labelled battles with target decisions, and trains only those battles. It
does not load or publish a checkpoint. Adam, 200 epochs, a fixed seed, and a
0.001 learning rate are the diagnostic defaults; each can be overridden. The
JSON report includes before/after losses, accuracies, demonstrated action and
target probabilities, optimizer settings, and explicit failure reasons. A
failed criterion produces a nonzero exit code and blocks the remaining recovery
benchmark work.

The reconstruction test executable also contains deterministic PPO direction
checks. They cover direct positive and negative advantages, mixed winning and
losing episodes under production advantage normalization, and value learning
when the policy term is clipped.

`--check-critic-fit` tests whether recorded returns are learnable before more
PPO is attempted. It creates two temporary in-memory copies of an existing
checkpoint and does not publish either one by default. The first copy trains
only the value
head, leaving policy outputs unchanged. The second also trains the shared GRU
and entity representation using value loss. By default its policy-head
parameters remain frozen, while the report measures any policy-probability
drift caused by the shared representation changing. Setting
`--critic-policy-kl-coef` adds a distillation penalty against the original
policy; this may update policy heads as well as the shared representation to
preserve behavior.

Without early stopping, battle IDs use a fixed 90/10 train/holdout split.
Setting `--critic-early-stop-patience` changes this to independent 80/10/10
training, selection, and holdout splits. The best selection-loss epoch is
restored in memory before the untouched holdout is evaluated. All splits use
`--validation-seed`; both branches receive the same deterministic training
order and discounted Monte Carlo return targets.

The JSON report compares value loss, explained variance, correlation, and bias
overall and for wins and losses. The aggregate check requires holdout explained
variance of at least `0.05`, an improvement of at least `0.02`, and
return/value correlation of at least `0.20`. A recurrent fit also fails when
its absolute policy-probability drift exceeds `0.01`, its train-to-holdout
explained-variance gap exceeds `0.25`, or a sufficiently large win/loss
subgroup has negative return/value correlation. These are diagnostic
thresholds, not strength or promotion results.

`--check-critic-fit-manifest` is the stronger form of the diagnostic. Its
newline-delimited manifest supplies multiple training batches. A stable 90/10
split of those battles provides training and early-stopping selection data,
while the separately named episode batch is used only for the final holdout
evaluation. Blank manifest lines and lines beginning with `#` are ignored.
The report records all three sources explicitly.

Passing `--critic-output-checkpoint PATH` opts into publishing the selected
recurrent critic. Publication occurs only after every learnability and safety
gate passes, and uses the same atomic checkpoint replacement as normal
training. The report records whether publication was requested, rejected,
completed, or failed, along with the selected trainer step. A rejected or
failed publication returns a nonzero exit code without replacing the requested
checkpoint.

`--check-action-q-fit` is the next diagnostic after the state critic passes but
PPO and AWR remain flat. It freezes the checkpoint completely and trains a
small sidecar to predict return targets for the demonstrated action. The
sidecar has a 16-unit latent projection, separate values for all 196 ordered
doubles action pairs and 28 single-slot actions, and target-specific values for
selectable move targets. Illegal action pairs never enter its expectation or
gradient.

`--action-q-target td0|td_lambda|monte_carlo` selects the credit horizon. TD(0)
uses the next frozen state value, TD(lambda) uses `--gae-lambda` to blend
bootstrapped and later rewards, and Monte Carlo uses the complete discounted
episode return. TD(0) has the least variance but may leave too little action
signal; Monte Carlo has the most outcome noise; TD(lambda) is the intended
middle ground. Reports use `metrics_version = 2` and name the selected target
and lambda explicitly.

The learned action advantage is centered by its exact expectation under the
checkpoint's legal live-play policy. Its expected contribution is therefore
zero, leaving the existing state value as the baseline. A fit must improve
untouched holdout loss by at least 2%, correlate with the frozen critic's TD
or return residual by at least 0.10, agree with the residual sign at least 53%
of the time on at least half the samples, avoid a large train/holdout gap, and
keep finite bounded advantages. Zero advantages are excluded from direction
accuracy rather than being counted as positive predictions.
These thresholds detect a possible action-level signal; they do not establish
playing strength.

### Counterfactual rollout batches

When TD(0), TD(lambda), and Monte Carlo targets all fail on ordinary rollout
data, `tools/counterfactual_rollout.js` creates paired evidence instead of
tuning the same optimizer again. For each decision it starts repeated
in-process Pokemon Showdown branches with identical battle, team, and agent
seeds through the intervention. The candidate takes its highest-ranked legal
action in half of the branches and its second-ranked legal action in the other
half. Each matched rank-zero/rank-one repeat receives the same continuation
seed; different repeats receive different continuation seeds. This keeps the
decision state fixed while exposing outcome variance after the action.
`branch_concurrency` bounds simultaneous branch battles independently of the
number of repeats, so a larger noise audit does not multiply the collector's
peak process count.

Collection also has a hard agent-process ceiling and a shared launch gate.
Before starting another branch, the gate checks total system RAM and CPU use;
it pauses while either exceeds the configured percentage and staggers process
starts to avoid short launch spikes. The conservative defaults use one outer
pair, three concurrent branches, at most eight agents, 70% system memory, and
75% CPU. Each spawned agent is also pinned to one OpenMP thread so several
small inference processes cannot each create a machine-wide worker pool. These
controls limit new work rather than terminating an active battle. Failed-agent
messages are byte-limited, and failure summaries stream the log instead of
loading the entire file into memory.

The collector verifies that the frozen recurrent state and legal mask are
identical at the intervention. It rejects pairs whose decision was not reached
or whose actions are not distinct. Each accepted branch becomes one compact
`counterfactual_sample` record containing the 128-value recurrent state after
the current observation, legal mask, joint action, compact selected-action
descriptors, individual rollout returns, their variance, and the mean terminal
return used as the target. The descriptors cover move slot and type, targeting,
remaining PP, Tera use, and the selected switch's slot, types, status, HP, and
stats. They avoid copying the very large raw observation into every branch.

The default three repeats per action provide the control used to judge the
labels. Rank-zero versus rank-one outcomes under a shared continuation seed
measure the action treatment; repeated outcomes for the same action measure the
rollout noise floor. The separate `*_signal_audit.json` reports disagreement
rates, within-action variance, estimated action-effect variance,
signal-to-noise ratio, repeat-direction consistency, and the fraction of
decisions with a reliable preference. Fewer than 30 repeated decisions are
reported as insufficient evidence. The disagreement comparison is paired by
decision and reports a normal-approximation 95% confidence interval; its lower
bound must exceed zero. The audit calls the action effect separated from noise
only when that bound is positive, the estimated signal-to-noise ratio reaches
1.25, and at least 20% of decisions have a stable preference. These are
diagnostic thresholds, not a playing-strength claim.

Pass `counterfactual_action_batch.jsonl` to
`--check-counterfactual-q-fit`. Without another batch, the C diagnostic assigns
whole pairs to stable train, selection, and holdout splits. For an independent
test, pass `--counterfactual-holdout-batch PATH`; the training batch is then
split only into training and model-selection data, while every pair in `PATH`
is reserved from optimization for holdout metrics. The JSON report records both paths
and whether the holdout was external.

For more than one training collection, put one batch path per line in a local
manifest and use `--check-counterfactual-q-fit-manifest`. Blank lines and lines
beginning with `#` are ignored. The command requires a separate holdout batch;
that path must not also appear in the manifest. Duplicate manifest entries are
rejected, and stable train/selection assignment includes the source path so
pair IDs reused by independent collectors cannot become coupled. Reports list
every training source as well as the manifest path. Manifests remain local run
artifacts and are excluded from Git.

The diagnostic trains the small action-Q sidecar over the frozen recurrent
state. Counterfactual fitting regresses the predicted Q difference onto the
measured return difference with a Huber loss and omits tied pairs. This gives
the action gap a finite, calibrated target instead of rewarding an indefinitely
large preference margin. Repeated pairs are weighted continuously using the
mean return gap and its standard error. Clear, repeatable preferences approach
weight one; noisy small gaps approach zero. Older one-shot batches retain
weight one for compatibility. The report keeps both raw and
confidence-weighted metrics.

`--counterfactual-q-objective preference` is a diagnostic alternative for
datasets where action order is more repeatable than the measured size of the
return gap. It applies confidence-weighted logistic ranking updates and selects
the best epoch by pairwise preference loss. Its holdout gate therefore requires
preference-loss improvement rather than gap-loss improvement. The default is
`gap`, preserving calibrated sidecar training and existing commands. Preference
models are not publishable as live sidecars because their score margins are not
calibrated Q differences.

The current direction is recorded locally in `docs/project_direction.txt`.
Its first step compares the two objectives on the existing four development
splits before starting the tactical teacher work. Run the fixed comparison
from the repository root after building and copying the current trainer:

```powershell
& .\tools\run_counterfactual_preference_comparison.ps1
```

The settings live in
`config/experimental/counterfactual_preference_comparison.json`. The runner uses
`build-fresh/showdown_client.exe`, runs one split at a time with four OpenMP
threads, and reuses completed reports only when their recorded settings match.
It checks the original gap reports before starting training. Missing or
mismatched evidence stops the comparison. `-SummaryOnly` requires all reports
to exist and never invokes the trainer.

The JSON summary retains both objectives' preference-loss ratios and ranking
accuracies, plus the new fits' gap loss and advantage magnitudes. A lower
preference loss does not establish stronger play. These splits share one
collection; the summary explicitly requires fresh confirmation and makes no
promotion decision. Reports identify source paths, not content hashes, so
preserve the original batch and checkpoint when resuming.

The completed comparison is recorded locally in
`models/diagnostics/run_0261_0264_preference_comparison.json`. Preference fitting
improved holdout preference loss on two splits and worsened it on two, without
a consistent ranking gain over gap regression. This closes the fixed comparison;
the next direction step is the tactical reference player.

### Tactical reference player: scoring foundation

`src/tactical_scoring.c` provides the first component, `tactical_score_attack`.
It scores an ordinary attack as base power times nominal hit probability,
ordinary STAB, and type effectiveness against a supplied visible target. The
result exposes each multiplier and uses effective base-power units. It does
not estimate HP damage or knockout probability. It abstains when required type
information or ordinary power metadata is missing. A known absent secondary
type is represented by ID zero with a nonzero knowledge mode.

The scorer uses effective visible typing rather than reconstructing defensive
types from species. It reads no hidden opponent stats or moves. Callers must
handle legality, move-specific exceptions, abilities, Tera STAB, field effects,
and spread damage before using the score in a reference player. Live action
selection and student training do not consume this component yet.

`src/tactical_policy.c` now enumerates legal actions and explicit targets for
both active slots. It reuses the request mapper's shared pair constraints for
double Tera and duplicate switches. Explicit foe targets are restricted to
living active opponents. Forced-switch requests fill the maximum possible
number of slots, including a pass when only one bench Pokemon remains.

The selector accepts a joint-choice evaluator, keeping legal enumeration
separate from scoring. Its initial `tactical_evaluate_power` heuristic sums
power against foes and subtracts power against the partner. Spread attacks use
the 0.75 multiplier when multiple targets are present. A switching partner is
evaluated using the incoming Pokemon's visible typing. This is a static board
approximation: speed order, protection, abilities, field effects, and unusual
move mechanics are still omitted. Switches have no strategic value term;
status moves and Tera attacks contribute zero and are counted as unscored.
The selector enforces Tera legality independently of whether its evaluator can
estimate Tera value.

The parser now distinguishes `allAdjacent` from unknown targeting. The action
semantic encoder retains the previous UNKNOWN input for this target class,
preserving existing neural feature values and checkpoint layout.

Build and run `showdown_tactical_tests` for its focused fixtures, and re-run
reconstruction and legality tests when changing request parsing. The opt-in
runtime integration below is ready for coverage checks. The broader teacher
still needs a strength evaluation against the parent, random play, and simple
baselines before generating labels.

### Tactical runtime coverage smoke

Pass `--tactical-teacher 1` after the fallback checkpoint in `--battle-agent`
mode to enable the evaluation policy. It uses the tactical selector only when
the power heuristic reports no unscored moves and a positive score, or when
handling forced switches. Other decisions retain the checkpoint policy. This
is a hybrid of the tactical heuristic and its parent, rather than an independent
teacher whose strength has already been established.

Each action includes `tactical.applied`, `reason`, `supported_choices`, and,
when applied, the scored opponent and friendly power. These fields describe
coverage of the current approximation; they do not assert that omitted game
mechanics are modeled. Counts describe emitted attempts, including rejected
choices, rather than only accepted turns.

The runtime advertises `training: false`. At the end of a battle it emits an
evaluation summary under `episode_complete`, with `training_eligible: false`,
the distinct `tactical-eval-v1` policy tag, and attempt/applied/forced-switch/
fallback counts. It omits episode training arrays. The normal episode parser
rejects this record, preventing teacher choices from being mistaken for
on-policy PPO data. Counterfactual interventions and tactical mode cannot be
enabled together.

After rebuilding and copying the client, run:

```powershell
node .\tools\tactical_teacher_smoke.js --run-name run_0265_tactical_runtime_smoke
```

`config/experimental/tactical_teacher_smoke.json` defaults to six serial battles
against the same checkpoint without the tactical override, alternating sides.
Each process uses one OpenMP thread. Launches wait when observed system memory
or CPU usage exceeds 65%. The script reuses the collector's process bridge,
choice-rejection handling, resource gate, and atomic writes.

The summary is saved after every battle under `matches/runs/<run>/<run>_summary.json`.
It retains coverage, results, rejected commands, and bounded failure messages;
it does not export training episodes. Rerunning the same command resumes
unattempted games only if the configuration, checkpoint hash, and executable
hash match. Failed attempts remain recorded. Use a new run name after rebuilding
or changing inputs. This smoke is a legality/coverage check and makes no strength
or promotion claim.

### Frozen tactical strength benchmark

The six-game runtime smoke is not a strength test. Once it passes, keep the
rules and checkpoint unchanged and run:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0266_tactical_strength_100ps
```

`config/experimental/tactical_teacher_benchmark.json` requests 100 valid games
per side against each of two opponents: the same checkpoint without tactical
overrides, then Showdown's seeded `RandomPlayerAI`. Random play uses only its
player stream, not the simulator's hidden battle state. The runner shares the
smoke's single-battle path rather than maintaining a second protocol adapter.
Adjacent side-A/side-B attempts share battle and team seeds, swapping teams
and preserving policy-role inference seeds. Opponent matchups reuse the same
seed schedule. Replacement attempts can leave a seed pair incomplete; every
completed game records its seeds.

Only completed games with consistent terminal results, valid teacher coverage,
and no rejected choices count. Failed games and choice rejections stay in the
summary; each side stops after 125 attempts if it cannot reach its target.
Results include wins/losses/draws, score rate (draws count half), side splits,
invalid counts, tactical coverage, and approximate 95% Wilson score intervals.
These intervals treat games as independent, although seed-paired games can be
correlated. They are descriptive screening metrics, not an automatic promotion
or student-training gate. Review both opponents, side effects, and errors before
claiming a stronger teacher; use fresh seeds for confirmation.

The atomic `<run>_summary.json` is saved after every attempt. Resume checks the
config, checkpoint, executable, runner/bridge source hashes, and core simulator
source hashes. A truncated primary summary can recover its last atomic backup;
an interrupted game is replayed. Changed inputs require a new run name. The
summary retains a failed status and error if the attempt budget is exhausted.
Console progress reports attempts, valid games, invalid games, elapsed minutes,
and an estimated total remaining time in minutes, calibrated from observed
valid-game throughput. Early estimates and estimates across opponent changes
can move substantially.

Battles remain serial, with one OpenMP thread per native agent and system CPU
and memory launch gates at 65%. These gates pause new battles; they do not impose
an operating-system-wide hard limit on other applications or a running battle.
Neither benchmark opponent produces training data or publishes a checkpoint.

For fresh-seed confirmation without changing the teacher or game counts, run:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0267_tactical_confirmation_100ps --config .\config\experimental\tactical_teacher_confirmation.json
```

The confirmation config changes only seed 26601 to 26701. Its battle, team,
and random-player seed tuples do not overlap the screening schedule, including
replacement attempts. Keep the executable and checkpoint identical to the
screening run. Review each opponent and side separately rather than pooling
both runs to conceal a weaker confirmation. A successful result supports the
next teacher-imitation experiment; it does not demonstrate neural learning.

### Teacher demonstration pilot

After reviewing the fresh-seed confirmation, collect a small supervised pilot:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0268_tactical_demonstrations_200 --config .\config\experimental\tactical_teacher_collection.json
```

This opt-in collection mode uses the same battle bridge, scheduling, atomic
summary, and resource gates as the benchmark. It requests 50 valid games per
side against each opponent (200 total), with at most 65 attempts per side.
The collection seed is 26801; the random-opponent seed range adds 10,000,000.
Unlike the paired benchmark, collection uses a separate team/battle seed for
each attempt. Both teacher wins and losses are retained; there is no wins-only
selection.

Each valid game's `teacher_<opponent>_<index>_raw.jsonl` contains exactly the
messages sent to the teacher runtime: battle start, player-visible events and
requests, accepted action pairs and target-bearing commands, terminal result,
and battle end. The parent/random player's stream and the omniscient stream
are not recorded. Parent-policy fallback decisions remain in the hybrid
teacher's demonstrations. The existing replay reader can reconstruct these
sessions for supervised learning; native ingestion must still be checked on
the collected files before training.

Completed, rejection-free games are validated and written atomically before
their replay path, hash, battle ID, decision count, outcome, and policy tag are
added to the run summary. Resume verifies referenced hashes and refuses missing
or modified files. A crash between replay publication and summary publication
may leave an unreferenced replay; the deterministic retry replaces it rather
than counting it twice. Use only files referenced by a completed collection
summary as training inputs. The summary reports `demonstration_count` and
`demonstration_turns`, and each replay entry specifies
`purpose: supervised_imitation` and `ppo_eligible: false`.

No model is trained or checkpoint published by this command. Runtime
`training_eligible: false` remains the safeguard against treating heuristic
behavior as on-policy PPO data; it does not prohibit explicitly labelled
supervised imitation. The next reviewable step is native replay-ingestion
validation, followed by a controlled student fit with battle-level holdout
and a fresh playing-strength evaluation with tactical overrides disabled.

For the next dataset-size experiment, add 800 fresh teacher battles:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0273_tactical_demonstrations_add800 --config .\config\experimental\tactical_teacher_collection_scale_800.json
```

This requests 200 valid games per side against each opponent, with at most
250 attempts per side. The teacher, parent/random opponent mix, unpaired
collection, one-thread agents, and 65% launch gates match the pilot. Seed
27301 supplies fresh positions; tests check battle/team/random seed-tuple
separation from the pilot, including replacement budgets and both opponent
seed ranges. No checkpoint changes or student fitting occur in this step.

Keep the original 200 demonstrations and their collection summary intact.
The later 1,000-battle dataset must reference those 200 plus these 800, using
only checksummed references from completed summaries. Before fitting, plan
memory-bounded ingestion and re-evaluate the parent and pilot student on the
same expanded holdout as the larger-data student. Continue to exclude holdout
battle IDs using the fixed validation seed; do not compare validation scores
computed over different holdout sets as if they were the same test. Start the
larger-data student from the original parent, not the already-trained pilot.
With the same epoch count, this experiment increases both distinct training
battles and update count; it is not an equal-compute isolation of data diversity.
Use the existing strength results for development, reserving fresh evaluation
seeds for a later selected-model confirmation.

### Prepare and fit the 1,000-battle dataset

Prepare the derived dataset without changing either source collection:

```powershell
node .\tools\prepare_tactical_dataset.js --run-name run_0274_tactical_dataset_1000
```

`config/experimental/tactical_dataset_1000.json` fixes the two source summaries,
1,000 battles, 13,282 decision turns, split seed 20261024, and shard limits.
Preparation checks completed collections, matching teacher checkpoint/runtime
hashes, referenced replay hashes, complete accepted decisions, unique battle
IDs, and the expected totals. Only referenced valid games are included. Each
`shard_*_raw.jsonl` holds at most 50 whole battles and 8 MiB of replay text;
an individually oversized battle is rejected, not truncated. The native
trainer loads one such shard at a time. These bounds constrain replay input,
not an exact RAM percentage for the process or the whole computer.

The dataset manifest records source summary hashes, per-battle provenance,
split counts, and shard hashes. Its separate `holdout_*_raw.jsonl` files
contain exactly the validation battles from those same shards, using the C
trainer's seeded battle-ID hash. The JS implementation is tested against the
native/Python hash fixture. Rerunning preparation verifies completed output;
an interrupted preparation can be rebuilt with the same identity. Changed
inputs or damaged completed shards stop with an error rather than silently
replacing a different dataset.

Evaluate the parent and pilot student on this expanded holdout before fitting:

```powershell
python .\py\tools\train_batch_selfplay.py --config .\config\experimental\tactical_student_train_1000.toml --evaluate-only 1 --pattern "holdout_*_raw.jsonl" --checkpoint .\models\runs\run_0093_current_arch_full\current_arch_full.chk --evaluation-summary .\models\diagnostics\run_0274_parent_common_holdout.json
python .\py\tools\train_batch_selfplay.py --config .\config\experimental\tactical_student_train_1000.toml --evaluate-only 1 --pattern "holdout_*_raw.jsonl" --checkpoint .\models\runs\run_0269_tactical_student_pilot\tactical_student_best.chk --evaluation-summary .\models\diagnostics\run_0274_pilot_common_holdout.json
```

Evaluation-only reuses the weighted metric aggregator and does not initialize,
train, or publish a model. It rejects dropped ingestion messages/labels,
missing or changed prepared shards, a changed validation seed, an unexpected
holdout turn count, empty evaluation, skipped turns, illegal predictions, or
non-finite output. The baseline JSON files and subsequent epoch summaries
therefore refer to the same holdout, not the original 22-battle subset alone.

Start the new fit only after those commands succeed:

```powershell
python .\py\tools\train_batch_selfplay.py --config .\config\experimental\tactical_student_train_1000.toml
```

The profile writes to run_0275, initializing from the original parent rather
than the pilot student. It uses all shards, three outer epochs, one pass per
file, Adam at 0.0001, split seed 20261024, and one OpenMP thread. Optional
`shuffle_seed` fixes the outer shard order; the native within-shard RNG remains
unchanged. CLI flags override only the selected profile's defaults. Explicit
zero values for `sample_files` and `limit_files` now correctly mean all files.
The profile does not run a parameter search or change model architecture.

Adam state is retained across shards within an epoch, but the current wrapper
starts a new optimizer process for each outer epoch. This differs from the
single-process pilot. The larger-data experiment also has more updates, so it
does not isolate data diversity from optimizer scheduling or compute. Record
those limits when comparing results; use a matched-pipeline small-data control
if a later causal claim requires separating them.

The wrapper saves atomic `_dataset_epochNNN.chk` snapshots and weighted
`_dataset_epochNNN_validation.json` summaries. Per-shard auxiliary checkpoints
stay disabled. Resume is at completed outer-epoch boundaries: a crash within
an unfinished epoch reruns that epoch from its starting checkpoint. It does
not restore the in-memory Adam state midway through an epoch. After fitting,
review all epoch summaries and select by the predeclared lowest full-turn
holdout NLL before any new playing-strength test. Do not pick an epoch by its
later battle win rate. Root README and existing checkpoints remain unchanged.

### Student-only strength evaluation

The pilot's best checkpoint is selected by full-turn holdout loss, not by
whichever epoch later wins the most battles. Evaluate that fixed selection:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0270_tactical_student_strength_100ps --config .\config\experimental\tactical_student_benchmark.json
```

The config sets `candidate_checkpoint` to run_0269's
`tactical_student_best.chk`, keeps `checkpoint` as the original parent, and
sets `tactical_teacher` to false. Neither native player receives the tactical
override flag. The random opponent remains Showdown's seeded `RandomPlayerAI`,
as in the teacher benchmark. New seed 27001 is used for both matchups with
matched team seeds across sides, 100 valid games per side, and unchanged
serial resource gates. No demonstrations or PPO episodes are published.

The summary reports `candidate_policy_mode: neural`, hashes both checkpoints,
and retains wins, losses, draws, approximate intervals, side splits, and
invalid games. Per-game `teacher_side` is the existing scheduling key and now
means the candidate's side. Neural results have `coverage: null` rather than
claiming tactical coverage. Resume rejects a changed policy mode, checkpoint,
config, or recorded runner source. Imitation improvement alone is not grounds
to promote the student; first compare this run's actual playing results.

After the larger-data fit, screen its preselected lowest-loss epoch against
all three reference opponents in one resumable run:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0276_tactical_student_1000_screen_100ps --config .\config\experimental\tactical_student_1000_screen.json
```

The candidate is run_0275's immutable `tactical_student_dataset_epoch003.chk`.
`checkpoint` still points to the original parent; `pilot_checkpoint` points to
run_0269's loss-selected 200-battle student. Selecting
`opponents: ["parent", "random", "pilot"]` requests 100 valid games per side
against each, or 600 valid games total. The new seed is 27601, matched team
seeds alternate candidate sides, and resource limits remain serial battles
with one native thread per agent and 65% launch gates. No training occurs.

Results retain a separate `pilot` matchup rather than labelling the smaller
student as the original parent. The pilot checkpoint hash joins the candidate,
parent, executable, config, and runner hashes in resume identity. Neither
native player receives a tactical override flag. Review strength against all
three opponents, side splits, invalid games, and seed-pair outcomes; imitation
metrics alone are not the success criterion. Screen before spending another
long fit. Only promising screen results advance to 500-valid-games-per-side
confirmation on fresh seeds with this same candidate frozen. Do not switch to
another epoch after seeing screening win rates without treating that choice
as new development selection.

For a promising but mixed screen, confirm the same model on fresh seeds:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0277_tactical_student_1000_confirmation_500ps --config .\config\experimental\tactical_student_1000_confirmation.json
```

The confirmation profile changes only seed 27701 and the sample/replacement
budgets: 500 valid games per side against each of parent, random, and pilot,
or 3,000 valid games total, with at most 625 attempts per side. Do not train
further, change epochs, or rebuild the native executable during this run.
The three native checkpoints are unchanged and tactical overrides remain off.
Inspect confirmation separately from the screen. A parent-matchup gain alone
does not establish that the larger dataset beat the pilot; that question
requires the direct pilot matchup as well as checking for losses elsewhere.
Use side-pair outcomes and invalid-game records alongside the descriptive
per-game Wilson intervals. A near-50% pilot result is inconclusive, not proof
of equivalence and not an automatic reason for another larger training run.

### Sampled versus greedy inference

Completed runs 0278/0279 favored sampling. Over 200 valid games per matchup,
sampling scored 55% against the parent and 48.5% against random; greedy scored
37.5% and 24.5%. In their direct matchup greedy scored 22.5%. Checkpoint,
executable, and base seed matched. Keep sampling; this full-turn MAP decoder
does not justify a larger confirmation.

An imitation gain need not survive stochastic action selection. Before
another fit, compare inference policies on the frozen larger student. Rebuild
the client and run reconstruction, legality, and tactical tests first; copy
the resulting client to `build-fresh/showdown_client.exe` only if tests pass.
Then run both profiles with that same executable:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0278_student_decoding_sample_100ps --config .\config\experimental\tactical_student_decoding_sample.json
node .\tools\tactical_teacher_benchmark.js --run-name run_0279_student_decoding_greedy_100ps --config .\config\experimental\tactical_student_decoding_greedy.json
```

The sampling control requests 100 valid games per side against parent and
random (400 total). Greedy repeats those matchups on the same seed/team schedule
and adds a direct matchup against the same checkpoint using sampling (600
total). All checkpoints, seed 27801, one-thread agents, and serial 65% launch
gates are fixed. `sampled` means the candidate's own checkpoint, not its parent
or the smaller-data pilot. The candidate alone receives
`--inference-policy greedy`; all native opponents retain the sampling default.
Do not rebuild between these two runs or mix historical sampling results from
another seed/runtime into the matched comparison.

`policy_decoding.c` ranks complete legal choices by joint action probability
and conditional legal target probabilities, or the slot-local factorized
probability on single-action turns. It is not an independent argmax of each
head or simply an argmax over 196 action pairs. It shares legal enumeration
with the teacher, but never its damage heuristic. Forced replacements use
distinct bench slots and permit a pass only where the shared rules allow it.
Full-turn MAP can favor actions with fewer target branches; greedy is an
experiment, not a claim that deterministic play is always better.

Sampling remains the default everywhere. Greedy actions and terminal records
are tagged evaluation-only, carry no on-policy episode arrays, and exclude
tactical and counterfactual overrides. Startup capability checks reject an
old client, and resume checks decoding identity as well as model/executable
hashes. No training, checkpoint edit, or automatic default/promotion change
occurs. Compare valid shared seed pairs, the direct sampled matchup, side
splits, and excluded games. Confirm a promising decoding choice on fresh seeds
before changing deployment defaults or spending another long training budget.

If screening is promising but inconclusive, keep that same loss-selected
checkpoint frozen for a larger fresh-seed evaluation:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0271_tactical_student_confirmation_500ps --config .\config\experimental\tactical_student_confirmation.json
```

This profile requests 500 valid games per side against each opponent: 1,000
against the parent and 1,000 against random, with at most 625 attempts per
side. Only the seed and sample/replacement counts change. Review this run
separately from screening; do not pick another epoch using these results or
train further while confirmation is in progress. Matched side pairs can be
correlated, so the displayed per-game Wilson intervals remain descriptive.
Review the paired records and invalid games as well as point estimates before
claiming a verified gain.

If the student beats its parent but not random, measure the parent against
that same random policy rather than assuming the parent scored 50%:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0272_parent_random_control_500ps --config .\config\experimental\tactical_parent_control.json
```

This control changes the candidate back to the original parent and selects
only `opponents: ["random"]`. It retains run_0271's seed 27101, 500 valid games
per side, native inference seed roles, team seeds, random AI, and resource
limits. The target and ETA adjust to the selected opponents (1,000 valid
games total). Preserve the native executable between runs. Compare the two
candidates' outcomes on shared valid seeds and side pairs; replacement games
outside that intersection are not matched observations. This control is not
an independent confirmation or a new student fit.

The compatibility head remains `--counterfactual-q-head joint`: each legal
joint action has its own state-conditioned score. The experimental
`factorized` head reuses a slot-local action scorer across every partner action
and adds a small ordered interaction bias for combinations that are not purely
additive. This preserves doubles-specific combinations while avoiding a
separate state-conditioned head for all 196 joint actions. Reports record the
selected head and its parameter count so comparisons can hold the data split
and optimizer fixed.

Early stopping follows the confidence-weighted selection loss for the chosen
objective. Publication
requires at least a 1% weighted-loss improvement on the holdout, at least ten
holdout pairs with different outcomes, and at least 55% confidence-weighted
within-pair ranking accuracy. A `.qv` sidecar is written only when every gate
passes.

If a generalization run selects epoch zero, use
`--check-counterfactual-q-overfit` before collecting more data. It reuses a
deterministic prefix of real pairs for training and evaluation, cannot publish
a sidecar, and passes only when the selected objective loss falls by at least 50%
and weighted ranking accuracy reaches 90% on at least ten discordant pairs.
This is a trainer-capacity check, not evidence of generalization or playing
strength. Counterfactual reports also retain metrics from the last attempted
epoch before restoring the best selection checkpoint.

Counterfactual Q fitting accepts `--counterfactual-min-confidence` to apply
the same matched-rollout confidence filter to training, selection, and
holdout pairs. Batches without matched return arrays use the older
independent-variance confidence calculation.

The default `--counterfactual-q-representation hidden` keeps compatibility
with existing collections and fits from the current recurrent state alone.
Schema-5 collections also support `--counterfactual-q-representation semantic`,
which appends the compact description of the selected actions. This is a
diagnostic for whether the Q head needs explicit action meaning rather than
only recurrent state and action-slot numbers. Use it with
`--counterfactual-q-head selected`. That head produces one shared score from
the state and selected-action description instead of routing sparse examples
through hundreds of fixed joint-action rows. The semantic/selected path is
diagnostic-only and cannot publish a live sidecar; a positive result must first
be incorporated into a supported model path and confirmed on a fresh
collection. Start a new run rather than resuming an older collector manifest.

When the action-Q sidecar fits its training pairs but does not generalize, use
`--check-counterfactual-policy-preference` or its `-manifest` variant to test a
more direct update. This path compares the log probabilities of the two
matched actions and moves the policy toward the action with the better rollout
return. The return-gap confidence controls the update weight. The recurrent
encoder and value head remain frozen; only the existing factorized policy
heads change.

When matched `rollout_returns` are present, confidence uses the variance of
the per-seed action differences. Older batches without those arrays retain the
independent-variance estimate. `--preference-min-confidence` can exclude
uncertain pairs from every split; the default of zero preserves earlier runs.
Use a cutoff only when it was chosen before examining the final holdout.

The objective measures each policy margin relative to the parent checkpoint.
To prevent the model from satisfying a comparison by suppressing both sampled
actions, `--preference-winner-nll-coef` adds a confidence-weighted likelihood
term for the better rollout action. `--preference-anchor-kl-coef` adds an
explicit legal-policy KL penalty.
Early stopping uses the fixed selection split and rejects epochs whose mean KL
exceeds `--preference-max-mean-kl`. The report includes absolute pair ranking,
the direction of the update relative to the parent, preferred and rejected
log-probability changes, and mean and maximum legal-policy KL. A normal
checkpoint is published through `--policy-output` only when the untouched
holdout improves the confidence-weighted preference loss by at least 1%, the
weighted update direction reaches 55%, outputs stay finite, and mean KL remains
inside the configured limit. The preferred action must also become more likely
on a majority of the weighted holdout pairs. Passing this diagnostic warrants a balanced
battle evaluation; it does not itself establish improved playing strength.

Before interpreting a failed holdout run, use
`--check-counterfactual-policy-preference-overfit` on a small deterministic
subset of the real counterfactual batch. It trains and evaluates on the same
pairs, never publishes a checkpoint, and defaults to no anchor penalty with a
generous KL ceiling so the diagnostic measures optimization rather than
regularization. It passes only when at least ten discordant pairs contribute,
confidence-weighted reference-adjusted loss falls by 50%, weighted update
direction and preferred-probability increase rate reach 90%, and all outputs
remain finite. Failure points to the
direct preference objective or its gradients; passing while normal holdout
fails points instead to representation or data coverage.

An external batch can be used repeatedly for development with the default
`--counterfactual-final-confirmation 0`. Such reports explicitly require a
fresh holdout before making a final claim. Set
`--counterfactual-final-confirmation 1` only for the first evaluation of a newly
collected, untouched external batch.

This is a causal diagnostic, not a strength claim. Full-game continuations are
still noisy, and the first comparison covers only policy ranks zero and one.
The summary reports how often mean paired returns disagree and links the signal
audit. Collector manifests and per-pair files are updated
atomically, so `--resume true` continues after interruption without replaying
completed pairs. Elapsed time and ETA include time from earlier resumed runs.

`--check-action-q-fit-manifest` uses the same multiple-training-batch and
separate-holdout arrangement as the critic diagnostic. `--action-q-output`
writes a sidecar only when every holdout gate passes. The sidecar is not read by
live play and cannot be promoted as a normal checkpoint. A later actor update
should be added only after this diagnostic succeeds, so a failed action model
cannot silently steer the policy.

`--audit-ppo-update` compares the behavior checkpoint with an already-trained
PPO checkpoint on the episode batch that produced the update. It does not train
or publish a model. The audit reports value loss, explained variance, legal
policy KL, and demonstrated-action probability movement across five advantage
bands. It also checks the recorded behavior log probabilities and values
against the supplied before checkpoint, catching the wrong-parent comparison.

The advantage bands use raw GAE calculated from recorded behavior values before
per-minibatch normalization. This keeps the report reproducible without
claiming that a globally standardized band is the exact advantage seen in each
shuffled minibatch. Use the same `--episode-limit`, `--shuffle-seed`, `--gamma`,
and `--gae-lambda` as the training run to reproduce its selected episode set
and return calculation. The JSON records the standardized-advantage bin edges
used for the five bands.

PPO KL stopping uses the exact distribution over legal actions when the anchor
checkpoint is byte-identical to the behavior parent. The sampled action
log-ratio remains in `approx_kl` for diagnosis, but does not stop those runs.
Training summaries identify the active measurement in `target_kl_source`. Runs
without a matching frozen reference retain the sampled-log-ratio fallback.

Current implementation notes:

- The repository now includes the protocol/session/raw-state/trainer/checkpoint path.
- Supervised and policy-gradient updates now operate over sequence windows rather than only updating output heads.
- Replay logs preserve raw battle messages so the environment can be rebuilt offline.
- Live runtime sessions can also export completed episodes directly for `--train-live-rl` without replay reconstruction.

## Current operating model

As of July 24, 2026, the intended long-term path is:

1. keep a trusted supervised baseline checkpoint
2. warm-start RL from that checkpoint instead of retraining from scratch
3. capture provenance for collection, training, and eval runs
4. reject obviously collapsed RL candidates before promotion discussion

`g4_teacher_sup_pool_wins_sup` is the historical supervised baseline. The model
learning audit did not verify it above random, so it is not a trusted strength
baseline for promotion decisions.

## Warm-start training

`py/tools/train_batch_selfplay.py` now separates:

- `--checkpoint`
  - output checkpoint path
- `--init-checkpoint`
  - optional warm-start source

Behavior:

- if `checkpoint` does not exist and `init-checkpoint` is provided, the wrapper copies the init checkpoint into the output location before launching `showdown_client`
- if `checkpoint` already exists, the wrapper keeps it and ignores `init-checkpoint`
- this avoids manual checkpoint copying for RL experiments

The wrapper also writes a training manifest beside the checkpoint by default. The manifest records:

- replay source run
- init checkpoint
- output checkpoint
- RL hyperparameters such as `reward_mode`, `gamma`, and `entropy_coef`
- sample file count
- configured environment variables
- shard completion progress

Supervised validation uses a fixed holdout selected by a stable hash of battle ID
and `--validation-seed` (default `1337`). The same battle stays in the holdout
regardless of shard ordering. After every complete outer dataset epoch, the wrapper
evaluates the resulting model over the holdout in every source shard and writes one
label-count-weighted summary.

Sharded supervised runs write an ordered replay-path manifest and train every
selected shard for an outer epoch inside one C process. Each replay runtime is
freed before the next is loaded, while model weights and optimizer state stay
alive. This prevents per-shard Adam resets without loading the whole dataset
into memory. Adam moments are not stored in checkpoints, so they restart at an
outer-epoch boundary. Use one outer epoch when diagnosing uninterrupted Adam
training across a sampled shard set. The wrapper passes `--aux-checkpoints 0`;
the C trainer retains its compatible `--aux-checkpoints 1` default for direct
single-file use. The wrapper instead atomically publishes dataset-level artifacts:

- `<checkpoint>_dataset_epochNNN.chk`
- `<checkpoint>_dataset_epochNNN_validation.json`

Each manifest epoch record includes the exact training shard list, source shard
count, accepted label count, validation seed, and collection manifest path. With
`--resume true`, the wrapper persists each epoch's shuffled shard plan before
training and records each successful shard by epoch, index, and resolved path. An
interrupted run resumes that exact plan and evaluates or publishes an epoch only
after all of its shards are complete.

`current_arch_full_ep10.chk` and `current_arch_full_ep20.chk` predate this artifact
contract. They remain loadable legacy files, but automatic league pool selection,
historical-opponent selection, and promotion do not treat them as trustworthy
milestones.

## Live self-play RL

The repository now has a first round-based actor -> learner path for live RL:

1. live workers sample actions from the current checkpoint in battle
2. terminal battles emit `episode_complete` JSON records directly from the runtime
3. `py/tools/live_rl_orchestrator.py` collects those records from worker JSONLs
4. the orchestrator copies the parent checkpoint into a child checkpoint path
5. `showdown_client --train-live-rl <episode_batch.jsonl> <checkpoint>` applies RL updates from that batch
6. standalone live runs may use the child directly in the next round; league runs screen it first and retain the previously accepted parent on rejection

This is intentionally round-based and restart-based. Actors are relaunched between rounds so each rollout batch is tied to one frozen policy version.

League PPO is promotion-gated within the workflow. Each round runs as a one-update child capped by `--episode-limit` (default `256`), then receives a balanced screen (default 100 valid games per side). Safe candidates at or above `--round-expand-threshold` receive the full evaluation budget. Only a candidate that clears the full point, confidence, Tera, and collapse gates becomes the next round's parent. Rejected children remain recorded, the accepted parent is retained, the best observed child is preserved in the workflow manifest, and `--round-early-stop-patience` prevents a chain of degrading updates. Set `--round-gating false` only for deliberate legacy-style chained experiments.

Controlled recovery runs can supply a fixed JSON pool with `--opponent-pool` and
set `--round-eval-baseline random`. In that mode, a round advances only after a
balanced random screen and full confirmation clear the configured confidence
gate. `--eval-battle-seed-base` reuses one matched battle set across rounds so
candidate comparisons have less matchup noise. Use `--registry-update false`
while the starting checkpoint is provisional;
the workflow retains all artifacts and its final assessment without changing the
shared league registry.

Artifacts:

- workflow manifest:
  - `models/runs/<run_name>/<run_name>_live_rl_manifest.json`
- per-round manifest:
  - `models/runs/<run_name>/roundXX/<run_name>_roundXX_manifest.json`
- extracted live episode batch:
  - `matches/runs/<collect_run>/episode_batch_<side>.jsonl`

This path is the preferred long-term direction over replay RL because it removes the off-policy mismatch from training on older logged actions.

The preferred live optimizer is now PPO (`--train-live-ppo`). Live episode records include rollout-time log probabilities, values, factorized actions, explicit move-target choices and masks, and a frozen actor `policy_tag`. PPO uses GAE, policy clipping, value clipping, target-KL stopping, and accumulated Adam updates. Advantages are normalized once across all labelled turns in an optimizer minibatch, preserving the outcome difference between battles. Targetable moves learn among self, ally, left-foe, and right-foe choices; moves without a selectable target do not contribute target-head loss.

`--train-live-awr` is an experimental diagnostic for the current learning-recovery work. It uses the same GAE advantages and legal factorized/joint action probabilities as PPO, but replaces the clipped probability-ratio objective with advantage-weighted imitation: each demonstrated turn receives `exp(advantage / temperature)`, capped by `--awr-max-weight`. The default temperature is `1.0`, the cap is `20.0`, and weights are computed after minibatch-wide advantage normalization. This mode updates only the factorized policy heads; the recurrent representation and value head remain fixed. It therefore tests whether ranked rollout outcomes contain an actor-learning signal without changing the critic or introducing another architecture change.

Use `--awr-minibatch-episodes` to control accumulation. AWR retains the parent-checkpoint provenance check, anchor penalty, exact legal-policy KL guard, atomic checkpoint publication, and normal training summary. Its summary identifies `training_method = "advantage_weighted_imitation"` and reports mean/min/max weight plus effective label count. `value_loss` remains an observational measurement against the frozen critic; `critic_updated` is false. Treat an AWR checkpoint as an experiment until a matched-seed balanced evaluation beats its parent.

When both active slots act, the policy normalizes over legal action pairs instead of sampling the slots independently. Each slot retains its own move/switch/Tera score, while an unordered pair head adds a symmetric compatibility score. The interaction therefore values combinations without imposing an arbitrary slot-0-then-slot-1 direction. Double Tera and duplicate switch destinations are masked before sampling. Singles and turns where only one slot chooses continue to use the per-slot factorized heads directly.

All twelve Pokémon observation blocks also pass through one shared 32-unit entity encoder. Its decoded representation is added as a residual to each known Pokémon before the GRU, so concepts learned from one team position can transfer to every other position while the original slot-specific signal remains available. The same recurrent loss backpropagates through this shared path for self and opponent Pokémon. Unknown roster entries bypass it.

Every Pokémon block includes an explicit three-way board-role feature: non-active, left active slot, or right active slot. This connects each per-slot policy head and target choice to the correct current Pokémon after switches instead of forcing the model to infer board position from roster order.

## Eval and collapse guardrails

`py/tools/selfplay_server.py` summaries now include:

- `candidate_checkpoint`
- `parent_checkpoint`
- `group_stats`
- `group_collapse_flags`
- `collapse_thresholds`

Current default collapse checks:

- warning when one move slot exceeds `55%`
- hard collapse when one move slot exceeds `70%`
- warning when `switch_slot_6` exceeds `60%`
- warning when candidate tera rate falls below `50%` of the baseline side
- fail-fast warning when earned win rate is below `40%` after at least `150` games

Metric units:

- training summaries use `tera_action_rate`: tera move actions divided by all move actions
- evaluation summaries use `tera_battle_rate`: battles containing a tera divided by matches played
- `tera_rate` remains in both formats as a backward-compatible alias; do not compare the aliases across artifact types without checking the unit
- `baseline_tera_action_rate` is also action-level; the shared default is `0.03`, measured from the current pre-PPO actor, and must not be populated from a battle-level evaluation rate

PPO safety behavior:

- `--anchor-checkpoint` and `--anchor-kl-coef` apply to PPO as well as ordinary policy-gradient training
- PPO accumulates a configurable number of episodes before each Adam update (`--ppo-minibatch-episodes`, default `8`), reducing sensitivity to one unusually short or unusual battle
- the critic contributes once per decision turn, including dual-action turns, and continues learning when PPO clips only the policy objective
- `--shuffle-seed` makes the episode order reproducible across candidates trained from the same batch
- `--target-kl` uses the label-weighted running mean and does not stop until both `--target-kl-min-episodes` and `--target-kl-min-labels` have been processed
- `--target-kl-hard-multiplier` defines an extreme-minibatch threshold, and `--target-kl-hard-consecutive-updates` requires repeated breaches (default `2`) before the emergency stop fires; isolated outliers remain visible in the summary without discarding the candidate
- an emergency KL stop writes a rejected summary, returns a nonzero exit code, and does not overwrite or create the output checkpoint
- a missing live-PPO output is initialized from `--parent-checkpoint`; an existing output must be byte-identical to that parent before an update begins
- the training summary records the input parent, output checkpoint, anchor, shuffle seed, minibatch size, available/processed episode counts, whether the checkpoint was published, and whether the ordinary or emergency KL stop fired
- `critic_explained_variance`, `return_value_correlation`, and `value_bias` indicate whether the critic is ranking returns usefully; value loss alone cannot distinguish a useful critic from a nearly constant prediction

Live episode records carry their reward mode and dense-weight values. The self-play server forwards these settings to every model-backed battle agent, and the trainer rejects batches whose recorded rewards do not match the requested mode or weights. Older episode batches have no reward metadata and are treated as terminal-reward data. Dense rewards therefore require a newly collected `dense_additive` batch; changing only the training flag cannot relabel an existing terminal batch.

Stable algorithm and guardrail defaults are centralized in `config/rl_defaults.toml`. Reward weights remain in `config/reward_weights.toml` because both the C runtime and Python communicator consume them.
The current controlled league profile is in `config/league_rl_orchestrator.toml`; run-specific names and seeds remain explicit CLI arguments.

Use `py/tools/eval_collapse_check.py` to re-run those checks from an existing summary JSON.

## Standalone balanced checkpoint evaluation

`py/tools/balanced_checkpoint_eval.py` compares any existing candidate against a
checkpoint or `random` baseline without training, changing the league registry,
or creating a temporary league member. `--baseline-checkpoint` remains available
for compatibility; `--baseline random` is the clearer spelling for a random
baseline. It reuses the league evaluator's side balancing, valid-outcome rules,
replacement blocks, Wilson interval, collapse checks, resume provenance,
dashboard, and promotion assessment. Pass `--battle-seed-base` to reuse the same
random teams when the candidate changes sides.
Operational defaults are loaded from `config/balanced_checkpoint_eval.toml`.
Pass `--config <path>` to select another profile; explicit CLI flags override it.

The combined summary and manifest are written under `matches/runs/<run-name>/`. Raw self-play logs are retained in that run's `logs/` directory by default. `--games-per-side` is a valid-game target: disconnect and forfeit outcomes remain visible as invalid but do not satisfy it. `--block-games` defaults to 250. Completed blocks survive a later subprocess crash and are reused by `--resume true`; incomplete blocks restart under a clean retry run name.

Example:

```powershell
python .\py\tools\balanced_checkpoint_eval.py --run-name eval_round03_vs_champion --candidate-checkpoint .\models\runs\candidate\round03.chk --baseline-checkpoint .\models\runs\champion\champion.chk --pool-seed 10301 --battle-seed-base 1030100
```

Use this path to screen intermediate checkpoints or perform an independent comparison. Automatic league promotion remains the responsibility of `league_rl_orchestrator.py`; the standalone evaluator reports what the configured gates would decide but does not mutate registry state.

## Canonical strength baseline

`py/tools/strength_baseline_benchmark.py` runs the recovery audit defined in
`config/strength_baseline_benchmark.toml`. Before launching games, it requires a
passing supervised-overfit report and reruns the reconstruction test executable,
which contains the deterministic PPO-direction test. It then starts every named
checkpoint in the fixed `build-fresh/showdown_client.exe` runtime and requires a
ready message. A missing or incompatible checkpoint therefore fails before any
evaluation workers are launched.

The benchmark screens g4, the current-architecture final checkpoint, the two
labelled legacy epoch artifacts, run_0096 anchored PPO, and the top observed
run_0111 candidate against random. Every screen uses the same battle seed suite.
The two highest lower bounds from the 95% Wilson intervals advance to the larger
random evaluation and head-to-head comparison.

Child evaluations are ordinary resumable balanced-evaluation runs under
`matches/runs/`. The benchmark manifest and final learning audit are written to
`models/benchmarks/<run-name>/`. They retain side splits, invalid games, Tera
rates, collapse flags, checkpoint provenance, trainer checks, and confidence
intervals. The audit marks `strength_baseline_verified` only when the strongest
finalist's lower 95% bound against random is strictly above 50%.

```powershell
python .\py\tools\strength_baseline_benchmark.py
```

## Controlled PPO hyperparameter search

`py/tools/ppo_search.py` compares candidates without mixing data quality into the result. Every trial starts from the same `--init-checkpoint`, trains on the same required `--episode-batch`, and uses the same anchor and parent opponent. It searches learning rate, entropy coefficient, anchor strength, PPO clip, and deterministic shuffle seed.

Unless `--policy-tag-expected` is explicit, the search reads the exact tag spelling from the episode batch, verifies path-like tags identify the initial checkpoint, and passes that original spelling to the trainer. This keeps relative and absolute spellings from causing false provenance mismatches without weakening the parent-checkpoint check.

Search-grid, staging, worker, and dashboard defaults live in `config/ppo_search.toml`. Use `--config <path>` for a different experiment file. Explicit command-line flags take precedence over config values; run name, initial checkpoint, and episode batch remain required so a config cannot accidentally launch against stale artifacts. The trainer executable is fixed at `build-fresh/showdown_client.exe` and cannot be overridden by the config.

In an interactive terminal, the search dashboard shows separate progress bars for candidate training, screening games, final games, and the active training/evaluation subprocess. Its candidate table reports live PPO KL, anchor KL, clip fraction, safety status, win rate, and Wilson confidence intervals; an evaluated-candidate leaderboard uses the same confidence-aware ordering as finalist selection. Non-interactive output falls back to plain text. Live state is also written to the manifest's `progress` object, while raw child-process output is stored under `models/search/<run_prefix>/logs/` when `dashboard_write_raw_logs = true`.

The search is staged:

1. Train all sampled combinations and reject candidates that trip KL, anchor, or clipping safety checks.
2. Evaluate the safest candidates against the parent with the candidate on both sides of the matchup; these live games provide the candidate-specific Tera and move-slot collapse checks. Disconnect/forfeit outcomes are reported as invalid and excluded from ranking and confidence intervals. The evaluator automatically runs the remaining shortfall until each side has the requested number of normally completed games. When the finalist cutoff remains unresolved, schedule additional balanced blocks for the provisional finalists and statistically plausible challengers up to a configured cap.
3. Give only the leading candidates a larger final evaluation.
4. Rank results by the lower bound of the 95% Wilson interval, with collapse-free candidates first. With `replicate_ranking = true`, complete parameter settings are trained and evaluated across every configured shuffle seed, then ranked by their pooled valid-game result rather than their luckiest checkpoint. The manifest records grouped screening/final rankings and `best_hyperparameters`; `best_result` is only a representative checkpoint from a setting that passes the pooled point gate.

Screening starts at 100 valid games per side and adaptively grows in 100-game blocks to at most 200 valid games per side while the finalist cutoff is unresolved. In replicated searches, refinement operates on whole parameter settings and gives every shuffle-seed replicate the same additional budget. Clear losers stop at the initial allocation; unresolved leaders and challengers receive more evidence. Final evaluation defaults to 300 valid games per side. Each result retains raw win rate and earned-win rate for diagnostics, plus explicit `valid_games`, `invalid_games`, `valid_score`, and `valid_win_rate`; ranking and promotion use the valid-game fields. Existing candidate and evaluation artifacts are reused by default, so rerunning the same command resumes interrupted work. Set `adaptive_screen = false` to use only the initial screening allocation. `eval_max_replacement_attempts` bounds repeated shortfall collection and fails the search loudly if the valid target cannot be reached.

Search evaluations also use matched randomness. Candidates trained with the same shuffle-seed replicate receive the same deterministic battle and random-team sequence, including their player-side swap. Different training replicates use disjoint substreams, increasing matchup coverage and avoiding an artificially narrow pooled confidence interval. This makes candidate differences less sensitive to lucky team draws while retaining balanced A/B evaluation. Screening and adaptive allocation use one suite; finalists are evaluated and promoted only on a separate held-out confirmation suite that Bayesian selection never observes. By default both suites are derived independently from `run_prefix` and `search_seed`, so a new run name rotates the matchups. Set `screen_seed_suite` and `confirmation_seed_suite` to distinct positive integers to reproduce specific suites explicitly; zero keeps run-derived behavior. Both resolved values, their provenance, and the replicate-to-substream mapping are recorded in the manifest.

Held-out confirmation is also opponent-diverse. The evaluation parent is always protected. When `confirmation_registry` exists, the registry champion is added if it is a different checkpoint, followed by active historical snapshots prioritized by the champion's hardest recorded matchups and then recency. `confirmation_opponents` can add explicit comma-separated checkpoints. Paths are deduplicated and `confirmation_max_opponents` bounds the suite; the default current-architecture configuration evaluates at most three distinct opponents.

Each finalist/opponent matchup starts at 100 valid games per side. In replicated mode, only pooled matchups whose confidence interval still crosses the 50% non-regression threshold receive 100-game extensions, up to 300 per side. Clear regressions and clear non-regressions stop early. Final ranking uses the pooled lower confidence bound across opponents, but promotion is blocked if any protected opponent's upper confidence bound falls below the configured non-regression threshold. The manifest retains every per-opponent interval, the worst lower bound, matchup spread, and a `favorable_matchup_dominance` diagnostic when one matchup is disproportionately favorable.

For a promotion-grade result, pass a separately collected batch with `--fresh-confirmation-episode-batch`. After search and held-out finalist ranking finish, the tool takes only the top setting, starts again from the original parent, retrains it on the fresh batch with the independent `fresh_confirmation_shuffle_seeds`, and evaluates those checkpoints on a third matchup seed suite. These retrains never feed back into Bayesian selection. When this option is enabled, only the fresh-data assessment can populate the final promotion result; the search-batch assessment is retained as `preliminary_promotion_assessment`.

The fresh stage reports `reproduced` when the new pooled point result passes the same completeness, collapse, protected-opponent, and win-rate gates without weakening; `weakened` when it still passes but has a lower win rate; and `reversed` when it fails those gates. Confidence remains a separate promotion requirement. The fresh batch must identify the same original parent checkpoint, and its shuffle seeds must not overlap the search seeds. Leaving the batch path empty skips this stage so exploratory searches do not require collection of a second dataset.

The self-play launcher installs an opt-in hook in the ignored local Showdown checkout; ordinary self-play remains unseeded unless `--battle-seed-base` is supplied. Evaluation summaries record the seed and resumed searches reject older unmatched artifacts.

Example shape (run this only after building and copying the current client):

```powershell
python .\py\tools\ppo_search.py --run-prefix search_replace_me --init-checkpoint .\models\runs\parent\parent.chk --episode-batch .\matches\runs\collection\collection_episodes_a.jsonl --anchor-checkpoint .\models\runs\parent\parent.chk --eval-model-b .\models\runs\parent\parent.chk
```

The default search is a conservative 256-episode Bayesian optimization over learning rates `2.5e-6` through `1e-5`, anchor coefficients `0.01` through `0.05`, and PPO clips `0.1` and `0.2`. It uses shuffle seeds `101,202,303`, adaptive pooled screening, and pooled parameter-setting ranking. Four normalized space-filling settings establish initial coverage. After each initial setting has been trained and screened across all three seeds, a dependency-free Gaussian-process surrogate models pooled valid win rate and its finite-game noise; expected improvement selects the next untried coarse-grid setting.

After six coarse settings, the search centers a local refinement space on the highest confidence-safe coarse result. Learning rate, entropy, and anchor strength use log-space half steps; PPO clipping uses linear half steps. When the coarse winner is at an edge, the local space also extrapolates half a step beyond that configured boundary. The surrogate then spends four reserved setting evaluations on expected-improvement choices from this finer space. Unsafe settings are rejected when any seed fails training safety and are never allowed to become a favorable surrogate observation. If no safe coarse center exists, the reserved budget falls back to untried coarse-grid settings.

`max_trials` counts search checkpoints, so each three-seed setting consumes three trials; the default 30-checkpoint budget evaluates ten settings: six coarse and four refined. Optional fresh-confirmation retrains are recorded separately and added to live progress after selection. `bayes_refine_settings = 0` disables the second phase. `finalists` counts parameter settings in replicated mode. The manifest's `bayesian_optimization` section records the phase of every setting, refinement center and candidate count, boundary position, predicted mean, uncertainty, and expected improvement. The final manifest also reports whether the held-out top setting touches or exceeds an original configured boundary. It contains seed-level diagnostics, pooled group rankings, confidence intervals, `best_hyperparameters`, a representative `top_result`, preliminary and final promotion assessments, and the complete `fresh_data_confirmation` record when enabled. Use `selection_strategy = "random"` for the previous reproducibly sampled-grid behavior. `tentative_winner` means the pooled point estimate passed but its confidence interval did not clear the parent; only `confident_winner` supports automatic promotion. A search result is evidence for the tested parent, batches, opponents, parameter ranges, and surrogate assumptions—not a universally optimal PPO setting.

For a deterministic data-scale experiment, fix the PPO hyperparameters and pass nested episode limits such as `--episode-limits 128,256,512,1018` with multiple shuffle seeds. `0` means the full eligible batch. For each seed, the trainer deterministically permutes the full batch once and uses a nested prefix, avoiding copied multi-gigabyte batch files. Set `adaptive_screen = false`, and set `max_trials` and `screen_candidates` high enough to include every scale/seed combination; the tool rejects unequal or mixed-hyperparameter scale comparisons. When multiple limits are present, the manifest adds `data_scale_summary`, pooling the common screening allocation by scale while retaining seed-level trials. The training summary records the requested limit, full available count, and selected count.

`config/experimental/ppo_data_scale_sweep.toml` provides the 128/256/512/1018-by-three-seed experiment using the current leading PPO settings. Supply it with `--config` plus the usual run prefix, parent checkpoint, episode batch, anchor, and evaluation parent paths.

## Collection provenance

`py/tools/selfplay_server.py` now writes a manifest per run in addition to the existing summary.

The manifest records:

- pool source paths
- resolved model specs for both sides
- sampled member counts from pool-based runs
- worker settings such as `worker_pairs`, `worker_games`, and `ensure_shard_count`
- summary path linkage

This is the basis for reproducible collection -> train -> eval lineage.

## League metadata

`py/tools/league_manage.py` now supports optional lineage/eval metadata on members:

- `parent_id`
- `regime`
- `source_run`
- `experiment_id`
- `eval.summary_path`
- `eval.collapse_flags`
- `opponent_stats`
  - the most recent collection window against each sampled opponent
  - stored from the learner's perspective as wins, losses, draws, game count, and recent win rate

It also supports workflow statuses beyond the original active/inactive pair:

- `candidate`
- `active`
- `inactive`
- `rejected`
- `champion` remains derived from `champion_id`

Pool presets are available through `build-pool`:

- `champion-plus-random`
- `active-mixed`
- `top-k`

League RL pools preserve separate budgets for the champion, recent mains, historical snapshots, and exploiters. Within a category, sampling uses the parent's most recent direct matchup record and favors opponents near the configured 50% learner win-rate target. Confidence smoothing prevents small samples from causing abrupt weight changes, while a minimum difficulty weight keeps easy and hard opponents available.

Worker assignment also guarantees a configurable minimum number of starts for every active category before returning to adaptive weights. Within a category, members behind their expected weighted share are preferred. `league_min_category_starts` defaults to `1`, and `--min-category-starts 0` disables this quota. Collection summaries expose assignment coverage and completed-game coverage separately so failed or unfinished workers remain visible as shortfalls.

Collection summaries expose both configured and realized sampling shares plus per-opponent records. At workflow completion, `league_rl_orchestrator.py` aggregates those records from the learner's perspective and stores them on both the parent and candidate so the next pool closes the feedback loop.

During a multi-round live run, `live_rl_orchestrator.py` also closes the loop immediately. It snapshots the pool used for each round, recalculates within-category weights from the completed collection, and passes the refreshed snapshot to the next round. Both the used and next pool paths are stored in the round manifest, and resume restores the recorded chain.

Interactive live and league runs share a single terminal dashboard. It tracks round completion, current collection games, PPO episodes and optimization metrics, active ETA, and collapse warnings. During a league run the same display continues through valid-game evaluation on both sides, then reports the Wilson interval and promotion-gate result. Non-interactive terminals fall back to plain text; progress remains available in the workflow manifest and raw child output is retained in the workflow's `logs/` directory by default.

League candidates are evaluated against the current champion on both player sides. `--eval-games` is the required number of valid games per side. Each self-play subprocess is capped by `--eval-block-games`; disconnects and forfeits do not count, so the evaluator may add up to `--eval-max-replacement-attempts` blocks. Each block writes its own atomic summary. Resume aggregates matching completed blocks and skips failed partial directories. The process manifest for the active block also gives the parent evaluator a bounded list of child PIDs to clean up after a native subprocess failure. The combined evaluation summary records raw, valid, and invalid totals, per-side results, collapse metrics, and a 95% Wilson interval for the candidate's valid-game score.

Promotion now requires all of the following:

- the valid-game score meets `--promote-threshold`
- the lower Wilson bound is above `--promotion-confidence-threshold`
- no policy-collapse guard fires
- candidate Tera use meets the configured champion-relative minimum

A candidate that clears the point estimate but not the confidence bound is retained as a tentative winner rather than promoted. League PPO training also forwards the shared anchor, clipping, KL-guard, minibatch, and Adam settings to the trainer. `config/league_rl_orchestrator.toml` holds the current full workflow profile, while reusable algorithm defaults remain in `config/rl_defaults.toml`. Explicit CLI flags override either entrypoint config.

## Recommended first RL ladder

The ladder below is historical. The current recovery work follows
`docs/project_direction.txt`; it does not call for another RL sweep yet.

The current recommended first RL sweep is:

1. start from the trusted baseline checkpoint
2. use `wins-only postfaint` or another high-signal champion-centered replay source
3. keep:
   - `mode = rl`
   - `reward_mode = terminal`
   - `epochs = 1`
   - `epochs_per_file = 1`
   - `sample_files = 200`
   - `gamma = 1.0`
   - `advantage_norm = 1`
4. vary only `entropy_coef` first:
   - `0.0003`
   - `0.001`
   - `0.003`

Do not scale model size or broaden collection policy until this ladder stops collapsing.

## Student decision audit

After rebuilding/copying the native client, run:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0280_student_decision_audit --config .\config\experimental\tactical_student_decision_audit.json
```

This runs 200 valid games (50 per side against parent and random) on fresh
seed 28001, using the frozen run_0275 epoch-3 student with sampling. The teacher
scores the student's visited boards without controlling play. Nothing is
trained or published. Reuse the command to resume with unchanged inputs.

The normal summary retains accepted per-game `decision_audit` records and
opponent-level totals. Fully scored choices have three effective-power gaps:
teacher minus selected; best targets with the selected actions held fixed
minus selected; and teacher minus that retargeted choice. Target averages
include only choices with multiple targets. The teacher and target averages
have separate denominators; do not add them as if they covered the same rows.
All totals weight decisions rather than averaging battle percentages.

Protect/status, Tera, voluntary switches, friendly effective power, and move-ID
counts help locate behavior patterns. `consecutive_protect` means consecutive
accepted ordinary requests in the same slot, not verified consecutive battle
turns. Forced replacements reset that counter. The ten largest fully scored
gaps retain game/request indices and both commands for follow-up inspection.
Invalid games remain in the summary but are excluded from aggregate audits
and win rates, including all their accepted choices.

Unscored choices (including Protect and Tera) are counted separately and
excluded from gap averages. An effective-power gap is not an HP-damage loss or
proof of a worse battle decision: status, switching, abilities, speed, and
protection can make the simple attack heuristic wrong. Use this audit to pick
a concrete transfer problem, not to select another checkpoint on these games.

## Target-only causal screen

After the native regression tests pass, freeze one executable for both runs:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0281_target_control_100ps --config .\config\experimental\tactical_target_control.json
node .\tools\tactical_teacher_benchmark.js --run-name run_0282_target_intervention_100ps --config .\config\experimental\tactical_target_intervention.json
```

The profiles differ only in `tactical_targets`, request 400 valid games each,
and share seed 28101, checkpoints, and parent/random opponents. Keep the
executable and inputs unchanged between modes and on resume. Serial battles,
one agent thread, and the existing 65% launch gates remain in effect. This is
an evaluation-only hybrid; it does not train or promote a neural checkpoint.

Predeclared analysis: compare candidate win scores against each opponent on
shared valid seed pairs, retaining side splits, approximate pair-level score
difference intervals, invalid games, and accepted intervention counts. A
positive point difference against both opponents without a legality regression
is a reason to consider fresh-seed confirmation, not proof of improvement.
An uncertain result stays uncertain. Do not tune the retargeting rule on these
games or automatically start the 500-per-side confirmation. Flat or harmful
results close this intervention and redirect the next learning experiment
toward action selection. A positive confirmation instead motivates learning
move/target relationships on student-visited boards.

The approved fresh-seed confirmation uses the same frozen executable as the
screen, without rebuilding between control and treatment:

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0283_target_control_confirmation_500ps --config .\config\experimental\tactical_target_control_confirmation.json
node .\tools\tactical_teacher_benchmark.js --run-name run_0284_target_intervention_confirmation_500ps --config .\config\experimental\tactical_target_intervention_confirmation.json
```

Each mode requests 2,000 valid games on seed 28301. Keep control/treatment
paired analysis separate by opponent and separate from the consulted screen.
Intervals are approximate pair-level normal intervals, not independent-game
intervals. Improvement is supported against an opponent when the paired score
difference interval excludes zero in the positive direction. Describe a
positive but zero-crossing interval as uncertain, not confirmed. Retain side
splits and exclusions; a legality regression limits interpretation. Neither
an intervention gain nor beating random establishes a learned-policy gain.

Completed confirmation (runs 0283/0284): sampling versus intervention scored
53.2% versus 61.6% against parent and 49.9% versus 53.4% against random,
1,000 valid games per opponent per mode. Shared complete seed pairs give
paired differences of +8.23 points (approximate 95% interval +4.02 to +12.44;
498 pairs) and +3.51 points (-0.93 to +7.95; 499 pairs), respectively.
The parent gain confirmed; the random gain remains uncertain. Do not pool
the consulted development screen to call that uncertainty resolved.

All model/executable/runtime-source hashes matched screening. Ten control and
twelve intervention games were excluded and replaced. These include known
trapped/disabled-choice rejections, revival requests with no legal actions,
and two intervention-side one-survivor no-legal-action failures. Their retained
records require targeted runtime follow-up; no rule or runtime fixes were
made during confirmation. The result is a hybrid-policy gain against parent,
not evidence of improved neural weights or a clean legality certification.

The two no-legal-action cases now have minimized native/runtime fixtures and
Showdown choice-validation tests. The one-survivor request had Helping Hand
as its only enabled move: it is legal to select the fainted ally's position,
even though the move then fails. The mask restores enabled ally-only moves
only when the slot otherwise has no move or switch choice. Attacks and legal
switches retain priority over this fallback; disabled moves stay disabled.
Revival Blessing may select a fainted active teammate, unlike an ordinary
switch. Its target check now applies the revival rule before excluding active
positions. These repairs do not change checkpoint layout or the targeting
intervention rule. Existing confirmation artifacts describe the older runtime;
do not merge later games into those frozen runs after rebuilding.

### Move-conditioned target learning

The bounded experiment in `docs/project_direction.txt` freezes the 1,000-battle
student and fits a separate target head for each active slot and selected move
slot. The main checkpoint, recurrence, action selection, Tera and value heads
remain unchanged. Sidecars are optional and parent-bound; this is not PPO data.

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0285_target_learning_collection --config .\config\experimental\target_learning_collection.json
node .\tools\target_learning.js run_0285_target_learning_collection models/runs/run_0286_move_conditioned_targets
```

Collection uses unchanged sampled student play in 500 distinct battles, half
against parent and half against random. Only accepted choices in valid games
contribute cached recurrent states and informative teacher target labels.
Losses are retained. Unsupported turns are omitted; equally preferred targets
share their label mass. Per-battle files are checksummed and checked on resume.

The fit uses a fixed whole-battle 20% holdout, deterministic SHA-256 ordering,
Adam at 0.001, batches of 64 examples, at most ten epochs and three stale epochs
of patience. The lowest held-out target NLL selects the epoch. Publication
requires lower held-out NLL, non-decreasing tie-aware accuracy, finite outputs
and unchanged parent parameters. A failed gate stops the experiment before
gameplay; it does not trigger another learning-rate or architecture sweep.

The native fitting command accepts the prepared binary cache directly:
`showdown_client --fit-target-sidecar dataset.bin parent.chk sidecar.bin report.json`.
The JavaScript wrapper records the source summary, exact parent/executable and
dataset hashes, split seed and counts. No ordinary checkpoint is overwritten.
These held-out metrics measure target imitation, not playing strength.

Run 0286 stopped at this gate: validation NLL improved from 0.693295 to
0.632879, but tie-aware accuracy moved from 739/1,025 to 737/1,025. Epoch 6
was selected by NLL and stopping occurred after epoch 9. No sidecar was
published and the conditional three-mode gameplay screen was not run. A fixed-
settings diagnostic replay reproduced these metrics and records all eight
slot/move breakdowns in
`models/diagnostics/run_0286_target_gate_replay/target_fit.json`. Do not select
another observed epoch or relax the gate to turn this result into a pass.

An approved learned-target screen can use `target_sidecar` in the existing
benchmark config, alongside `tactical_teacher: false` and sampled inference.
The runner requires a passing fit report and checksummed metadata binding the
sidecar to the candidate checkpoint. It rejects teacher/target overrides,
label collection and other decoders in this mode. Native direct evaluation is
`showdown_client --battle-agent parent.chk --target-sidecar sidecar.bin`;
these episodes remain ineligible for PPO. None of this changes the default
policy or treats the failed run as deployable.

The subsequently approved exploratory screen explicitly waives only the
accuracy veto, without changing the original failed result. Publication uses
`--exploratory-evaluation`; its report keeps `passed: false`, the failure reason
and `exploratory_exception: true`. The benchmark additionally requires
`exploratory_target_evaluation: true`. Neither option bypasses finite-output,
parent-binding or improved-validation-NLL checks. This exception is for
evaluation, not promotion or PPO training.

```powershell
node .\tools\target_learning.js run_0285_target_learning_collection models/runs/run_0287_target_exploratory --exploratory-evaluation
node .\tools\tactical_teacher_benchmark.js --run-name run_0288_target_exploratory_control --config .\config\experimental\target_exploratory_control.json
node .\tools\tactical_teacher_benchmark.js --run-name run_0289_target_exploratory_learned --config .\config\experimental\target_exploratory_learned.json
node .\tools\tactical_teacher_benchmark.js --run-name run_0290_target_exploratory_heuristic --config .\config\experimental\target_exploratory_heuristic.json
```

Reproduction must match the original selected epoch and aggregate metrics
exactly. The three screen modes share seed 28801, models, executable and
resource settings, with 100 valid games per side against parent/random each.
Keep paired uncertainty separate by opponent; these are development results.

The approved screen completed cleanly: all 1,200 games were valid and all
three modes shared model/executable/source hashes. Sampling, learned targets
and heuristic targets scored 53.0%, 56.0% and 55.0% against parent; against
random they scored 44.5%, 41.5% and 53.0%. Learned-minus-sampling differences
were +3 points against parent (approximate paired 95% interval -7.41 to +13.41)
and -3 against random (-11.90 to +5.90), each over 100 complete seed pairs.
There is no demonstrated learned-strength gain and no automatic promotion.
See `models/diagnostics/run_0288_0290_target_exploratory_comparison.json` for
the side splits, coverage, full comparisons and provenance. Intervals are
nominal development intervals, without multiplicity adjustment; don't pool
opponents or reinterpret this as passing the original learning gate.

### Target-input audit

`docs/target_input_audit.md` traces the raw observation fields to the frozen
target sidecar and records what the collection actually retained. Reproduce
the checksum-verified, read-only cache inspection with:

```powershell
node .\tools\target_learning.js --audit-inputs run_0285_target_learning_collection models/diagnostics/run_0291_target_input_audit.json
```

The 500-battle cache contains frozen states and slot/move-index/target labels,
not the original move identity/type or target-board context. Those fields
exist upstream in the model's observations, but cannot be supplied directly
to a new target head from this cache. No focused explicit-input fit was run.
New context capture and collection require a separately approved bounded step;
do not fabricate features from the labels or overwrite the original data.

### Visible-context comparison

The separately approved context collection retains the actual selected move
ID/type and four relative target records (self, ally, foe-left, foe-right),
including presence, friendly/foe relation, visible types and knowledge flags.
Unknown or absent types are redacted. The cache accepts native move IDs beyond
the frozen model's 920-entry one-hot width; it does not change that model's
input dimensions or backfill the old cache.

```powershell
node .\tools\tactical_teacher_benchmark.js --run-name run_0292_target_context_collection --config .\config\experimental\target_context_collection.json
node .\tools\target_learning.js --compare-context run_0292_target_context_collection models/runs/run_0293_target_context_comparison
```

This is 500 fresh student-played battles against parent/random, with unchanged
serial resource gates. Capability/version checks reject a client that omits
the context. Accepted-choice filtering and per-battle checksums still apply.

The comparison trains the old frozen-state head and one head with an added
shared move-type/defender-type residual table. Friendly and opposing targets
have separate learned coefficients; dual types add two contributions. Missing
type knowledge contributes no residual. The 722 new coefficients start at
zero and contain no built-in type-chart or teacher-score values. Move IDs are
retained for audit but are not a further feature in this fixed comparison.

Both fits receive exactly the same examples, deterministic ordering and
whole-battle validation split (seed 20261026). Adam LR 0.001, batch size 64,
ten maximum epochs and patience three are shared. Each branch selects its
lowest validation-NLL epoch. Reports retain tie-aware accuracy and preferred-
target probability mass alongside NLL. The native command prints JSON; the
wrapper atomically saves it with dataset/model/executable provenance.
No weights are published and no gameplay is started by this diagnostic.

Run 0292 completed 500 valid battles from 504 attempts in 9.2 minutes. Four
games with trapped/disabled-choice rejections were excluded and retained.
All 5,482 labels have complete context; the split is 405/95 battles and
4,443/1,039 labels. On the same validation examples, the trained baseline
reached NLL 0.661749, accuracy 71.90% and preferred-target probability mass
69.87%. The context variant reached 0.550746, 90.28% and 75.79%, respectively.
Both selected epoch 10 without extending the predeclared budget. The report
is `models/runs/run_0293_target_context_comparison/context_comparison.json`.

This is a validation imitation improvement, not a win-rate result or proof
that all relevant information was absent from the frozen state. The variant
adds both explicit type interactions and 722 shared trainable coefficients.
Its 90.28% argmax agreement is not its sampled target success rate. Model
checkpoints are unchanged and no learned weights were published; live wiring
and strength evaluation need a separately approved step.
