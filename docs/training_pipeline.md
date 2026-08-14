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

The preferred live optimizer is now PPO (`--train-live-ppo`). Live episode records include rollout-time log probabilities, values, factorized actions, and a frozen actor `policy_tag`. PPO uses GAE, policy clipping, value clipping, target-KL stopping, and accumulated Adam updates.

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

Stable algorithm and guardrail defaults are centralized in `config/rl_defaults.toml`. Reward weights remain in `config/reward_weights.toml` because both the C runtime and Python communicator consume them.

Use `py/tools/eval_collapse_check.py` to re-run those checks from an existing summary JSON.

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
