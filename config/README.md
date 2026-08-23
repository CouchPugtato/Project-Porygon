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
- `train_batch_selfplay.toml`
  - consumed by `py/tools/train_batch_selfplay.py` before CLI args are applied
  - supports `--init-checkpoint <path>` for warm-start RL/supervised runs without manual checkpoint copying
  - supports `--experiment-id <token>` and `--manifest-path <path>` for experiment lineage/provenance
  - supports `--sample-files <N>` to train on a random subset of shards per epoch instead of the full run
  - supports `--reward-mode terminal|dense_additive` for RL reward shaping during replay reconstruction
  - supports `supervised_profile = true|false` to enable or fully disable per-episode supervised profiling work in `showdown_client`
  - supports a `Rich` live dashboard when `dashboard = true`
  - automatically falls back to plain-text output if `Rich` is unavailable or the terminal is not interactive
  - writes one raw trainer log per shard when `dashboard_write_raw_logs = true`
  - supports `env_<NAME> = <value>` entries to set subprocess environment variables for `showdown_client`
  - writes one batch-training stats JSON per shard under `models/runs/<run>/<checkpoint_stem>/<checkpoint_stem>_batch_training_stats/`
  - writes one training manifest JSON beside the checkpoint by default
- `live_rl_orchestrator.toml`
  - consumed by `py/tools/live_rl_orchestrator.py` before CLI args are applied
  - runs round-based live self-play RL using the current checkpoint on side `a`
  - collects `episode_complete` records directly from worker JSONLs and trains with `showdown_client --train-live-rl`
  - copies the parent checkpoint into each round output path before the RL update so actor rollouts stay versioned by round
  - writes one workflow manifest under `models/runs/<run_name>/` plus one per-round manifest
- `ppo_search.toml`
  - consumed by `py/tools/ppo_search.py` before CLI arguments are applied
  - controls the candidate grid, staged evaluation sizes, worker limits, and search dashboard
  - supports bounded adaptive screening blocks when the finalist cutoff remains statistically unresolved
  - records the top-ranked result separately from the gated winner and confidence-based promotion assessment
  - renders training, active-operation, screening, and final-evaluation progress bars when `dashboard = true`
  - writes live progress into the search manifest and optional raw subprocess logs under `models/search/<run_prefix>/logs/`
  - always uses `build-fresh/showdown_client.exe`; the trainer path is intentionally not configurable
- `reward_weights.toml`
  - runtime-loaded by both `showdown_client` and `py/communicator/main.py`
- `experimental/*.toml`
  - example templates for common collection/training regimes
  - intended as copy/rename starting points rather than auto-loaded defaults

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

Examples:

```powershell
.\build-fresh\showdown_client.exe
python -m py.communicator.main
python py/tools/selfplay_server.py
python py/tools/train_batch_selfplay.py
python py/tools/live_rl_orchestrator.py
```

Those commands will use the tokens from these files automatically.

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

`py/tools/replay_switch_diagnose.py` focuses specifically on switch-slot behavior and splits it by:

- outcome (`win`, `loss`, `draw`)
- `forced_switch`
- `reduced_active`
- `voluntary_switch`

Example:

```powershell
python py/tools/replay_switch_diagnose.py --run run_0060_g8_teacher_sup_pool_collect_200shards_postfaint_wins --side a
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

Example:

```powershell
python py/tools/live_rl_orchestrator.py --run-name run_live_rl_g4_test --init-checkpoint "models/runs/run_0056_g4_teacher_sup_pool_collect_200shards_wins/g4_teacher_sup_pool_wins_sup/g4_teacher_sup_pool_wins_sup.chk" --rounds 2 --games 1000 --concurrent-games 70 --worker-pairs 200 --ensure-shard-count true --model-b random --pool-seed 71 --entropy-coef 0.003 --reward-mode terminal
```

Pools emitted by `league_rl_orchestrator.py` use the adaptive strategy automatically. Category totals stay fixed while members inside each category are reweighted toward the configured 50% learner win-rate target after every round. Static pool JSON files continue unchanged, though the exact pool used is still snapshotted for reproducibility.

For reduced-board / post-faint curriculum extraction:

- `py/tools/replay_extract_postfaint.py` copies full battles into a new run folder when the selected side encounters:
  - a `forceSwitch` request, or
  - a request with fewer than two active mons
- it keeps the whole battle so the existing trainer can still reconstruct state from the start of the replay
- `--wins-only` further restricts the extracted set to earned wins for the selected side

Examples:

```powershell
python py/tools/replay_extract_postfaint.py --run run_0050_g1_best_loss_review --output-run run_0050_g1_best_postfaint --side a
python py/tools/replay_extract_postfaint.py --run run_0050_g1_best_loss_review --output-run run_0050_g1_best_postfaint_wins --side a --wins-only
python py/tools/train_batch_selfplay.py --run run_0050_g1_best_postfaint --mode supervised --pattern "worker_*_a_raw.jsonl" --checkpoint g1_best_postfaint_sup.chk --epochs 1
```

For wins-only full-battle extraction:

- `py/tools/replay_extract_wins.py` copies only full battles that ended in an earned `win` for the selected side
- it excludes disconnect/forfeit wins by requiring terminal `reward > 0.0`
- use this when you want a cleaner teacher-only supervised dataset without truncating battle context

Example:

```powershell
python py/tools/replay_extract_wins.py --run run_0044_teacher_sup_pool_collect --output-run run_0044_teacher_sup_pool_collect_wins --side a
python py/tools/train_batch_selfplay.py --run run_0044_teacher_sup_pool_collect_wins --mode supervised --pattern "worker_*_a_raw.jsonl" --checkpoint g1_teacher_sup_pool_wins_sup.chk --epochs 1
```

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

Examples:

```powershell
python py/tools/league_manage.py init
python py/tools/league_manage.py add-checkpoint --id g1_seed --path models/runs/run_x/seed/seed.chk --status candidate --regime supervised
python py/tools/league_manage.py promote --id g1_seed
python py/tools/league_manage.py build-pool --output models/league/pools/active_pool.json --preset active-mixed
python py/tools/selfplay_server.py --run-name run_league_smoke --games 100 --concurrent-games 8 --model-a-pool models/league/pools/active_pool.json --model-b random
```
