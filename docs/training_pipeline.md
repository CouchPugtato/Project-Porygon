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

The current trusted baseline is `g4_teacher_sup_pool_wins_sup`.

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

## Live self-play RL

The repository now has a first round-based actor -> learner path for live RL:

1. live workers sample actions from the current checkpoint in battle
2. terminal battles emit `episode_complete` JSON records directly from the runtime
3. `py/tools/live_rl_orchestrator.py` collects those records from worker JSONLs
4. the orchestrator copies the parent checkpoint into a child checkpoint path
5. `showdown_client --train-live-rl <episode_batch.jsonl> <checkpoint>` applies RL updates from that batch
6. the child checkpoint becomes the actor policy for the next round

This is intentionally round-based and restart-based. Actors are relaunched between rounds so each rollout batch is tied to one frozen policy version.

Artifacts:

- workflow manifest:
  - `models/runs/<run_name>/<run_name>_live_rl_manifest.json`
- per-round manifest:
  - `models/runs/<run_name>/roundXX/<run_name>_roundXX_manifest.json`
- extracted live episode batch:
  - `matches/runs/<collect_run>/episode_batch_<side>.jsonl`

This path is the preferred long-term direction over replay RL because it removes the off-policy mismatch from training on older logged actions.

The preferred live optimizer is now PPO (`--train-live-ppo`). Live episode records include rollout-time log probabilities, values, factorized actions, explicit move-target choices and masks, and a frozen actor `policy_tag`. PPO uses GAE, policy clipping, value clipping, target-KL stopping, and accumulated Adam updates. Targetable moves learn among self, ally, left-foe, and right-foe choices; moves without a selectable target do not contribute target-head loss.

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
- `--shuffle-seed` makes the episode order reproducible across candidates trained from the same batch
- `--target-kl` uses the label-weighted running mean and does not stop until both `--target-kl-min-episodes` and `--target-kl-min-labels` have been processed
- `--target-kl-hard-multiplier` remains an emergency stop for an update whose KL is far beyond the target
- the training summary records the input parent, output checkpoint, anchor, shuffle seed, minibatch size, available/processed episode counts, and whether the ordinary or emergency KL stop fired

Stable algorithm and guardrail defaults are centralized in `config/rl_defaults.toml`. Reward weights remain in `config/reward_weights.toml` because both the C runtime and Python communicator consume them.

Use `py/tools/eval_collapse_check.py` to re-run those checks from an existing summary JSON.

## Controlled PPO hyperparameter search

`py/tools/ppo_search.py` compares candidates without mixing data quality into the result. Every trial starts from the same `--init-checkpoint`, trains on the same required `--episode-batch`, and uses the same anchor and parent opponent. It searches learning rate, entropy coefficient, anchor strength, PPO clip, and deterministic shuffle seed.

Search-grid, staging, and worker defaults live in `config/ppo_search.toml`. Use `--config <path>` for a different experiment file. Explicit command-line flags take precedence over config values; run name, initial checkpoint, and episode batch remain required so a config cannot accidentally launch against stale artifacts.

The search is staged:

1. Train all sampled combinations and reject candidates that trip KL, anchor, or clipping safety checks.
2. Evaluate the safest candidates against the parent with the candidate on both sides of the matchup; these live games provide the candidate-specific Tera and move-slot collapse checks.
3. Give only the leading candidates a larger final evaluation.
4. Rank results by the lower bound of the 95% Wilson interval, with collapse-free candidates first.

Screening defaults to 250 games per side and final evaluation defaults to 1,000 games per side. These values are intentionally much larger than a smoke test; reduce them only when validating that the workflow itself runs. Existing candidate and evaluation artifacts are reused by default, so rerunning the same command resumes interrupted work.

Example shape (run this only after building and copying the current client):

```powershell
python .\py\tools\ppo_search.py --run-prefix search_replace_me --init-checkpoint .\models\runs\parent\parent.chk --episode-batch .\matches\runs\collection\collection_episodes_a.jsonl --anchor-checkpoint .\models\runs\parent\parent.chk --eval-model-b .\models\runs\parent\parent.chk
```

For a robustness check, pass multiple deterministic orders such as `--shuffle-seeds 101,202,303`. The manifest under `models/search/<run-prefix>/` contains all training diagnostics, balanced evaluation results, confidence intervals, and the selected result. A search result is evidence for the tested parent, batch, opponent, and parameter ranges—not a universally optimal PPO setting.

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

## Recommended first RL ladder

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
