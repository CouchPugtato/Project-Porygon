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
6. standalone live runs may use the child directly in the next round; league runs screen it first and retain the previously accepted parent on rejection

This is intentionally round-based and restart-based. Actors are relaunched between rounds so each rollout batch is tied to one frozen policy version.

League PPO is promotion-gated within the workflow. Each round runs as a one-update child capped by `--episode-limit` (default `256`), then receives a balanced screen (default 100 valid games per side). Safe candidates at or above `--round-expand-threshold` receive the full evaluation budget. Only a candidate that clears the full point, confidence, Tera, and collapse gates becomes the next round's parent. Rejected children remain recorded, the accepted parent is retained, the best observed child is preserved in the workflow manifest, and `--round-early-stop-patience` prevents a chain of degrading updates. Set `--round-gating false` only for deliberate legacy-style chained experiments.

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
- `--target-kl-hard-multiplier` defines an extreme-minibatch threshold, and `--target-kl-hard-consecutive-updates` requires repeated breaches (default `2`) before the emergency stop fires; isolated outliers remain visible in the summary without discarding the candidate
- the training summary records the input parent, output checkpoint, anchor, shuffle seed, minibatch size, available/processed episode counts, and whether the ordinary or emergency KL stop fired

Stable algorithm and guardrail defaults are centralized in `config/rl_defaults.toml`. Reward weights remain in `config/reward_weights.toml` because both the C runtime and Python communicator consume them.

Use `py/tools/eval_collapse_check.py` to re-run those checks from an existing summary JSON.

## Standalone balanced checkpoint evaluation

`py/tools/balanced_checkpoint_eval.py` compares any existing candidate and baseline checkpoints without training, changing the league registry, or creating a temporary league member. It reuses the league evaluator's side balancing, valid-outcome rules, replacement blocks, Wilson interval, collapse checks, resume provenance, dashboard, and promotion assessment.

The combined summary and manifest are written under `matches/runs/<run-name>/`. Raw self-play logs are retained in that run's `logs/` directory by default. `--games-per-side` is a valid-game target: disconnect and forfeit outcomes remain visible as invalid but do not satisfy it.

Example:

```powershell
python .\py\tools\balanced_checkpoint_eval.py --run-name eval_round03_vs_champion --candidate-checkpoint .\models\runs\candidate\round03.chk --baseline-checkpoint .\models\runs\champion\champion.chk --games-per-side 250 --concurrent-games 70 --worker-pairs 125 --pool-seed 10301 --format gen9randomdoublesbattle --resume true
```

Use this path to screen intermediate checkpoints or perform an independent comparison. Automatic league promotion remains the responsibility of `league_rl_orchestrator.py`; the standalone evaluator reports what the configured gates would decide but does not mutate registry state.

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

The self-play launcher installs an opt-in hook in the ignored local Showdown checkout; ordinary self-play remains unseeded unless `--battle-seed-base` is supplied. Evaluation summaries record the seed and resumed searches reject older unmatched artifacts.

Example shape (run this only after building and copying the current client):

```powershell
python .\py\tools\ppo_search.py --run-prefix search_replace_me --init-checkpoint .\models\runs\parent\parent.chk --episode-batch .\matches\runs\collection\collection_episodes_a.jsonl --anchor-checkpoint .\models\runs\parent\parent.chk --eval-model-b .\models\runs\parent\parent.chk
```

The default search is a conservative 256-episode Bayesian optimization over learning rates `2.5e-6` through `1e-5`, anchor coefficients `0.01` through `0.05`, and PPO clips `0.1` and `0.2`. It uses shuffle seeds `101,202,303`, equal screening allocations, and pooled parameter-setting ranking. Four normalized space-filling settings establish initial coverage. After each initial setting has been trained and screened across all three seeds, a dependency-free Gaussian-process surrogate models pooled valid win rate and its finite-game noise; expected improvement selects the next untried grid setting. This continues sequentially until the configured checkpoint budget is exhausted. Unsafe settings are rejected when any seed fails training safety and are never allowed to become a favorable surrogate observation.

`max_trials` counts checkpoints, so each three-seed setting consumes three trials; the default 24-checkpoint budget evaluates eight settings. `finalists` counts parameter settings in replicated mode. The manifest's `bayesian_optimization` section records the initial settings, every sequential suggestion, predicted mean, uncertainty, and expected improvement. It also contains seed-level diagnostics, pooled group rankings, confidence intervals, `best_hyperparameters`, a representative `top_result`, and a separate promotion assessment. Use `selection_strategy = "random"` for the previous reproducibly sampled-grid behavior. `tentative_winner` means the pooled point estimate passed but its confidence interval did not clear the parent; only `confident_winner` supports automatic promotion. A search result is evidence for the tested parent, batch, opponent, parameter ranges, and surrogate assumptions—not a universally optimal PPO setting.

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

League candidates are evaluated against the current champion on both player sides. `--eval-games` is the required number of valid games per side; disconnects and forfeits do not count, and the orchestrator launches replacement blocks up to `--eval-max-replacement-attempts`. The combined evaluation summary records raw, valid, and invalid totals, per-side results, collapse metrics, and a 95% Wilson interval for the candidate's valid-game score.

Promotion now requires all of the following:

- the valid-game score meets `--promote-threshold`
- the lower Wilson bound is above `--promotion-confidence-threshold`
- no policy-collapse guard fires
- candidate Tera use meets the configured champion-relative minimum

A candidate that clears the point estimate but not the confidence bound is retained as a tentative winner rather than promoted. League PPO training also forwards the shared anchor, clipping, KL-guard, minibatch, and Adam settings to the trainer; the league-specific learning-rate and entropy defaults are the stable anchored values in `config/rl_defaults.toml`.

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
