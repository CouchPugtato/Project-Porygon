Default entrypoint config files.

Format:
- TOML
- Each entrypoint config uses top-level key/value defaults
- Keys map to CLI flags by converting `_` to `-`
- CLI args are appended after config-derived defaults, so explicit CLI args override config values

Files:
- `rl_defaults.toml`
  - shared stable defaults for policy-gradient/PPO training and collapse guardrails
  - consumed by the C trainer and Python orchestration tools
  - experiment TOMLs should only override values intentionally
- `showdown_client.toml`
  - consumed by `showdown_client` when launched with no CLI args
- `communicator.toml`
  - consumed by `py/communicator/main.py` when launched with no CLI args
- `selfplay_server.toml`
  - consumed by `py/tools/selfplay_server.py` when launched with no CLI args
  - forwards `reward_mode` and dense reward weights to every live battle agent and records them in run artifacts
- `train_batch_selfplay.toml`
  - consumed by `py/tools/train_batch_selfplay.py` before CLI args are applied
  - supports `--init-checkpoint <path>` for warm-start RL/supervised runs without manual checkpoint copying
  - supports `--experiment-id <token>` and `--manifest-path <path>` for experiment lineage/provenance
  - supports `--sample-files <N>` to train on a random subset of shards per epoch instead of the full run
  - supports `--reward-mode terminal|dense_additive` for RL reward shaping during replay reconstruction
  - supports `supervised_profile = true|false` to enable or fully disable per-episode supervised profiling work in `showdown_client`
  - uses `validation_seed` to keep the battle-ID-based supervised holdout fixed across shards and epochs
  - resumes the exact persisted epoch/shard plan when `resume = true`
  - supports a `Rich` live dashboard when `dashboard = true`
  - automatically falls back to plain-text output if `Rich` is unavailable or the terminal is not interactive
  - writes one raw trainer log per shard when `dashboard_write_raw_logs = true`
  - supports `env_<NAME> = <value>` entries to set subprocess environment variables for `showdown_client`
  - writes one batch-training stats JSON per shard under `models/runs/<run>/<checkpoint_stem>/<checkpoint_stem>_batch_training_stats/`
  - writes one training manifest JSON beside the checkpoint by default
  - disables trainer-side auxiliary shard checkpoints and atomically publishes one checkpoint plus one weighted validation summary per complete dataset epoch
- `live_rl_orchestrator.toml`
  - consumed by `py/tools/live_rl_orchestrator.py` before CLI args are applied
  - runs round-based live self-play RL using the current checkpoint on side `a`
  - collects episode rewards using the configured mode; episode-batch training rejects a different mode or dense-weight set
  - collects `episode_complete` records directly from worker JSONLs and trains with `showdown_client --train-live-rl`
  - copies the parent checkpoint into each round output path before the RL update so actor rollouts stay versioned by round
  - writes one workflow manifest under `models/runs/<run_name>/` plus one per-round manifest
  - renders collection, PPO training, round, KL/safety, and ETA progress when `dashboard = true`
  - writes live state to the workflow manifest and raw per-round subprocess logs under `models/runs/<run_name>/logs/`
- `league_rl_orchestrator.toml`
  - consumed by `py/tools/league_rl_orchestrator.py` before CLI arguments are applied
  - carries the current recovery pool, PPO safety settings, worker limits, parent gate, and read-only registry policy
  - leaves run names and battle/pool seeds on the command line so separate experiments cannot silently reuse them
- `balanced_checkpoint_eval.toml`
  - consumed by `py/tools/balanced_checkpoint_eval.py` before CLI arguments are applied
  - carries the balanced-game target, worker limits, resource guards, promotion thresholds, and dashboard settings
  - leaves the run name, candidate, baseline, and deterministic seeds on the command line
- `ppo_search.toml`
  - consumed by `py/tools/ppo_search.py` before CLI arguments are applied
  - controls the candidate grid, staged evaluation sizes, worker limits, and search dashboard
  - supports bounded adaptive screening blocks when the finalist cutoff remains statistically unresolved
  - excludes disconnect/forfeit outcomes from ranking and collects replacement games until each valid-game target is reached
  - supports deterministic nested data scales through `episode_limits`; `0` uses the full eligible batch
  - records the top-ranked result separately from the gated winner and confidence-based promotion assessment
  - renders training, active-operation, screening, and final-evaluation progress bars when `dashboard = true`
  - writes live progress into the search manifest and optional raw subprocess logs under `models/search/<run_prefix>/logs/`
  - always uses `build-fresh/showdown_client.exe`; the trainer path is intentionally not configurable
- `strength_baseline_benchmark.toml`
  - defines the six recovery-audit checkpoints and their provenance labels
  - requires the supervised-overfit report and reconstruction/PPO-direction tests before games start
  - screens every checkpoint against random with matched battle seeds, then advances two finalists
  - writes a resumable benchmark manifest and final learning audit under `models/benchmarks/`
- `reward_weights.toml`
  - runtime-loaded by both `showdown_client` and `py/communicator/main.py`
- `experimental/*.toml`
  - example templates for common collection/training regimes
  - intended as copy/rename starting points rather than auto-loaded defaults
  - `ppo_data_scale_sweep.toml` fixes PPO settings and compares 128/256/512/1018 episodes over three seeds

Example:

```toml
run = "run_0018_recorded_action_sp1000"
checkpoint = "run18_200shards.chk"
mode = "rl"
epochs = 1
supervised_profile = true
dashboard = true
dashboard_visible_shards = 5
dashboard_write_raw_logs = true
env_porygon_omp_threads = 8
```

Notes:

- strings should be quoted
- numbers can be written as numbers
- ordinary boolean options become `1`/`0` on the generated CLI
- `env_<NAME> = <value>` entries are exported to trainer subprocesses instead of becoming CLI flags
- `dashboard_visible_shards` accepts any positive integer; if even, the live window biases one extra row toward upcoming shards

Warm-start notes:

- `checkpoint` is still the output checkpoint path
- `init_checkpoint` is only used when the output checkpoint does not already exist
- if both are provided and the output checkpoint already exists, the wrapper keeps the existing output checkpoint and logs that `init_checkpoint` was ignored
- this is the intended entrypoint for warm-start RL from a trusted baseline such as `g4`

Training dashboard notes:

- install Python tooling dependencies from [requirements.txt](/abs/f:/Coding/Repositories/Project-Porygon/py/requirements.txt), including `rich`
- when `dashboard = true`, `py/tools/train_batch_selfplay.py` renders:
  - a top summary with ETA/checkpoint/throughput/context
  - an epoch progress bar
  - a rolling shard window centered around the current shard
- raw trainer stdout is written to:
  - `models/runs/<run>/<checkpoint_stem>/<checkpoint_stem>_batch_training_logs/`
- if `Rich` is missing or the output is not a TTY, the wrapper falls back to the current plain-text logging style

PPO-search dashboard notes:

- candidate rows show the searched hyperparameters, PPO/anchor KL, clip fraction, safety state, and evaluation confidence interval
- the leaderboard ranks evaluated candidates by the same confidence-aware search ordering used to choose finalists
- `dashboard_visible_trials` and `dashboard_leaderboard_size` control table sizes; `dashboard_refresh_per_second` controls terminal refresh rate
- disabling `dashboard_write_raw_logs` suppresses per-candidate trainer and evaluation logs but does not disable manifest progress updates

Live/league dashboard notes:

- standalone live RL shows round, collection, PPO-training, KL, anchor, clipping, and safety state
- league RL uses the same display across nested live training, balanced side-A/side-B evaluation, confidence intervals, and the final promotion gate
- the nested live process runs in plain-output mode during a league workflow so only one terminal dashboard owns the screen
- non-interactive runs retain plain-text output and still write the manifest `progress` object

Examples:

```powershell
.\build-fresh\showdown_client.exe
python -m py.communicator.main
python py/tools/selfplay_server.py
python py/tools/train_batch_selfplay.py
python py/tools/live_rl_orchestrator.py
python py/tools/league_rl_orchestrator.py --run-name <run> --pool-seed <seed> --eval-battle-seed-base <seed>
python py/tools/balanced_checkpoint_eval.py --run-name <run> --candidate-checkpoint <checkpoint> --baseline <checkpoint-or-random> --pool-seed <seed> --battle-seed-base <seed>
```

Those commands will use the tokens from these files automatically.
Pass `--config <path>` to either new entrypoint to use a different flat TOML file; later CLI arguments still win.

For the communicator, `--replay-save <run_name>` resolves to:

```text
matches/runs/<run_name>/<run_name>_raw.jsonl
```

It also accepts a nested worker token like:

```text
--replay-save run_0013_random_pool/worker_000_a
```

which resolves to:

```text
matches/runs/run_0013_random_pool/worker_000_a_raw.jsonl
```

`--server-uri <ws://...>` overrides the websocket endpoint directly. If omitted, the communicator falls back to `PS_URI`, then `PS_SERVER`, then the public default server.

### Reward weights

`config/reward_weights.toml` is a runtime-loaded declarative config for:

- terminal rewards
- dense additive shaping weights
- per-request reward clip

Current keys:

```toml
terminal_win = 1.0
terminal_loss = -1.0
terminal_draw = 0.0
terminal_disconnect_or_forfeit = 0.0

dense_additive_hp_swing_weight = 0.10
dense_additive_faint_swing_weight = 0.25
dense_additive_reward_clip = 0.40
```

Notes:

- `py/communicator/main.py` reads the terminal reward keys when writing replay terminal records
- `showdown_client` reads the dense additive keys when reconstructing RL rewards from replay
- keep this file flat and declarative; do not put inferred ratios, examples, or derived commentary into the config itself

`--reconnect-seconds <n>` controls how long the communicator waits before reconnecting after an unexpected websocket/network drop.

`--guest-refresh-seconds <n>` controls how long a guest session may live before the communicator intentionally reconnects between battles to get a fresh guest account. `0` disables this.

`py/tools/selfplay_server.py` also supports weighted model pools for collection:

- `--model-a-pool <path>`
- `--model-b-pool <path>`
- `--pool-seed <int>`
- `--ensure-shard-count true|false`

Pool files are JSON with this schema:

```json
{
  "coverage": {
    "enabled": true,
    "min_category_starts": 1,
    "prefer_under_sampled_members": true
  },
  "members": [
    {
      "name": "random",
      "kind": "random",
      "weight": 0.5
    },
    {
      "name": "best_v1",
      "kind": "checkpoint",
      "path": "models/runs/run_x/best_v1/best_v1.chk",
      "weight": 0.5,
      "category": "historical",
      "learner_win_rate": 0.52,
      "matchup_games": 100,
      "difficulty_weight": 0.96
    }
  ]
}
```

Rules:

- `members` must be a non-empty list
- member `name` values must be unique
- `kind` must be `random` or `checkpoint`
- `weight` must be greater than `0`
- `checkpoint` members require an existing `path`
- weights are normalized internally; they do not need to sum to `1.0`
- `category`, `learner_win_rate`, `matchup_games`, and `difficulty_weight` are optional provenance fields emitted by league workflows
- `coverage` is optional; pools without it retain ordinary weighted-random assignment
- `coverage.min_category_starts` must be non-negative; `0` disables category quotas

Sampling behavior:

- pool sampling happens per worker start, not per battle
- if a worker respawns, it samples again from that side's pool
- when coverage is enabled, categories below `min_category_starts` are assigned first
- after every category reaches its target, category selection returns to the pool's configured/adaptive weights
- within the selected category, `prefer_under_sampled_members` favors members behind their weighted assignment share
- existing `--model-a` / `--model-b` behavior is unchanged when pool args are omitted
- every run now writes both:
  - `<run_name>_summary.json`
  - `<run_name>_manifest.json`
- the manifest records pool provenance, sampled member counts, worker settings, and candidate/parent checkpoint info when available
- pool summaries report each member's configured weight, realized sample rate, category, result record, and score rate
- `group_pool_coverage` reports per-category assignment targets and shortfalls separately from completed-game shortfalls
- eval/collection summaries now include automatic collapse flags for each side using the current RL guardrail thresholds

Examples:

```powershell
python py/tools/selfplay_server.py --run-name run_pool_test --games 100 --concurrent-games 8 --model-a-pool config/pools/a_pool.json --model-b random
python py/tools/selfplay_server.py --run-name run_pool_vs_pool --games 100 --concurrent-games 8 --model-a-pool config/pools/a_pool.json --model-b-pool config/pools/b_pool.json --pool-seed 123
```

Experiment templates currently included:

- `config/experimental/supervised_baseline.toml`
- `config/experimental/warmstart_rl_safe.toml`
- `config/experimental/warmstart_rl_entropy_high.toml`
- `config/experimental/large_collection_active_pool.toml`
- `config/experimental/live_rl_dense_anchor_safe.toml`

### Replay diagnostics

`py/tools/replay_diagnose.py` analyzes replay JSONLs and produces:

- a compact terminal summary
- an optional JSON report for later review/tooling

Supported inputs:

- `--run <run_name>`
- `--path <jsonl_file_or_dir>`
- `--glob <glob_pattern>`

Notes:

- the analyzer is CLI-first and does not modify replay files
- it compares outcome buckets for a selected side, defaulting to side `a`
- it currently uses replay metadata and command-string parsing only; it does not require snapshot export

Examples:

```powershell
python py/tools/replay_diagnose.py --run run_0050_g1_best_loss_review --side a
python py/tools/replay_diagnose.py --run run_0050_g1_best_loss_review --side a --output matches/runs/run_0050_g1_best_loss_review/run_0050_g1_best_loss_review_diagnostics.json
python py/tools/replay_diagnose.py --glob "matches/runs/run_0050_g1_best_loss_review/worker_*_a_raw.jsonl" --side a
python py/tools/replay_diagnose.py --run run_0050_g1_best_loss_review --side a --include-outcomes loss --sample-losses 15
```

`py/tools/eval_collapse_check.py` reads a selfplay summary and emits a compact collapse report for RL guardrails.

Current default checks:

- warning when one move slot exceeds `55%`
- hard collapse when one move slot exceeds `70%`
- warning when `switch_slot_6` exceeds `60%`
- warning when candidate tera rate falls below `50%` of baseline side
- fail-fast warning when earned win rate is below `40%` after at least `150` games

Example:

```powershell
python py/tools/eval_collapse_check.py --summary matches/runs/run_0060_g9_teacher_rl_from_g4_vs_champion/run_0060_g9_teacher_rl_from_g4_vs_champion_summary.json --candidate-side a
```

### Live Self-Play RL

`py/tools/live_rl_orchestrator.py` is the first round-based actor -> learner loop for on-policy-style self-play.

Per round it does this:

1. snapshots the opponent pool that will be used for the round, when configured
2. launches `py/tools/selfplay_server.py` with the current checkpoint on side `a`
3. waits for the collection run to finish
4. refreshes adaptive opponent weights from that round's learner-perspective results
5. extracts `episode_complete` JSON records from `worker_*_<side>_raw.jsonl`
6. copies the parent checkpoint to a new child checkpoint path
7. runs `showdown_client --train-live-rl <episode_batch.jsonl> <child_checkpoint>`
8. uses the child checkpoint and refreshed pool for the next round

Artifacts:

- workflow manifest:
  - `models/runs/<run_name>/<run_name>_live_rl_manifest.json`
- per-round manifests:
  - `models/runs/<run_name>/roundXX/<run_name>_roundXX_manifest.json`
- per-round opponent pools:
  - `models/runs/<run_name>/roundXX/<run_name>_roundXX_opponent_pool_used.json`
  - `models/runs/<run_name>/roundXX/<run_name>_roundXX_opponent_pool_next.json`
- extracted episode batches:
  - `matches/runs/<collect_run>/episode_batch_<side>.jsonl`
- optional raw collection/training logs:
  - `models/runs/<run_name>/logs/`

Example:

```powershell
python py/tools/live_rl_orchestrator.py --run-name run_live_rl_g4_test --init-checkpoint "models/runs/run_0056_g4_teacher_sup_pool_collect_200shards_wins/g4_teacher_sup_pool_wins_sup/g4_teacher_sup_pool_wins_sup.chk" --rounds 2 --games 1000 --concurrent-games 70 --worker-pairs 200 --ensure-shard-count true --model-b random --pool-seed 71 --entropy-coef 0.003 --reward-mode terminal
```

Pools emitted by `league_rl_orchestrator.py` use the adaptive strategy automatically. Category totals stay fixed while members inside each category are reweighted toward the configured 50% learner win-rate target after every round. Static pool JSON files continue unchanged, though the exact pool used is still snapshotted for reproducibility.

### League registry workflow

`py/tools/league_manage.py` manages a filesystem-backed checkpoint league and emits pool JSON files for `selfplay_server.py`.

Default registry path:

```text
models/league/league_registry.json
```

Notes:

- the league registry is JSON, not TOML
- generated pool files use the same JSON schema already supported by `selfplay_server.py`
- `random` is not stored as a registry member; it is injected during pool build when requested

Supported commands:

- `init`
- `show`
- `add-checkpoint`
- `update-member`
- `promote`
- `deactivate`
- `build-pool`

Registry additions:

- members now support optional lineage/eval metadata:
  - `parent_id`
  - `regime`
  - `source_run`
  - `experiment_id`
  - `eval.summary_path`
  - `eval.collapse_flags`
  - `opponent_stats`, containing the most recent learner-perspective record for each sampled opponent
- public statuses now support:
  - `candidate`
  - `active`
  - `inactive`
  - `rejected`
  - `champion` is derived from `champion_id`
- `build-pool` also supports presets:
  - `champion-plus-random`
  - `active-mixed`
  - `top-k`

`league_rl_orchestrator.py` builds role-aware pools with separate champion, recent-main, historical, and exploiter budgets. Within each category it favors opponents whose recent learner win rate is closest to `league_matchup_target_win_rate` (default `0.50`). Low-sample results are shrunk toward the target using `league_matchup_confidence_games`, and `league_matchup_min_weight` keeps every eligible opponent in circulation. Generated pools require `league_min_category_starts` worker assignments per active category (default `1`); override it with `--min-category-starts`, or set it to `0` to disable quotas.

Pass `--opponent-pool <path>` to use an exact static pool instead of deriving one
from the registry. `experimental/model_learning_recovery_pool.json` mixes random,
the provisional ep20 checkpoint, and the strongest run 0111 candidate. Pair it
with `--round-eval-baseline random --registry-update false` so advancement means
beating random with the configured confidence bound while the shared registry
remains unchanged.

League evaluation is balanced across both player sides. `--eval-games` specifies the valid-game target for each side. `--eval-block-games` caps each durable self-play block, and `--eval-max-replacement-attempts` allows extra blocks when disconnects or forfeits leave a shortfall. Completed block summaries are reused after a crash; an incomplete block gets a separate retry directory so partial JSONL files cannot contaminate the result. Promotion requires the point estimate in `promotion_earned_win_rate`, a lower 95% Wilson bound above `promotion_confidence_threshold`, no collapse flags, and sufficient Tera use relative to the champion. The combined summary and workflow manifest retain per-side runs, invalid counts, confidence bounds, promotion-gate results, and the complete PPO training configuration. League PPO defaults use `league_ppo_learning_rate = 0.00001` and `league_ppo_entropy_coef = 0.0001`, and the orchestrator forwards the shared anchor, clipping, KL-guard, minibatch, and Adam settings.

Examples:

```powershell
python py/tools/league_manage.py init
python py/tools/league_manage.py add-checkpoint --id g1_seed --path models/runs/run_x/seed/seed.chk --status candidate --regime supervised
python py/tools/league_manage.py promote --id g1_seed
python py/tools/league_manage.py build-pool --output models/league/pools/active_pool.json --preset active-mixed
python py/tools/selfplay_server.py --run-name run_league_smoke --games 100 --concurrent-games 8 --model-a-pool models/league/pools/active_pool.json --model-b random
```
