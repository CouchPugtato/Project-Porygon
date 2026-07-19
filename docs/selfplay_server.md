# Selfplay Server

`py/tools/selfplay_server.py` runs a managed local Pokemon Showdown clone and a pool of communicator workers against it.

Defaults:

- local clone dir: `external/pokemon-showdown`
- local client dir: `external/pokemon-showdown-client`
- format: `gen9randomdoublesbattle`
- matchmaking: `/search`
- same-model matches are allowed
- worker pool size: `2 * --concurrent-games`

Queued shard mode:

- by default, worker count still follows `2 * --concurrent-games`
- if you set `--worker-pairs`, the runner creates that many `A/B` worker pairs instead
- in queued shard mode, only `--concurrent-games` pairs are active at once
- each side-`A` worker file is a training shard, so `--worker-pairs 200` yields about `200` side-`A` shard files
- if `--worker-games` is unset, the runner auto-budgets per-pair games as `ceil(--games / --worker-pairs)`

What it manages:

- starts the local Showdown server
- starts a local static Showdown client server when available
- waits for the websocket endpoint to become reachable
- launches worker communicator processes
- counts completed battles from worker replay JSONL terminal records
- drains workers once `--games` is reached
- writes a summary JSON for the run

Run layout:

- `matches/runs/<run_name>/showdown_server.log`
- `matches/runs/<run_name>/showdown_client_static.log`
- `matches/runs/<run_name>/orchestrator.log`
- `matches/runs/<run_name>/<run_name>_summary.json`
- `matches/runs/<run_name>/worker_000_a_raw.jsonl`
- `matches/runs/<run_name>/worker_000_a_raw.stats.txt`
- `matches/runs/<run_name>/worker_000_a.stdout.log`

Example:

```powershell
python py/tools/selfplay_server.py `
  --run-name run_0013_random_pool `
  --games 200 `
  --concurrent-games 16 `
  --model-a fresh_supervised_a.chk `
  --model-b fresh_supervised_b.chk
```

You can also pass the literal value `random` for `--model-a` or `--model-b`. That worker group will use the existing communicator random agent instead of launching the learner runtime.

Example mixed pool:

```powershell
python py/tools/selfplay_server.py `
  --run-name run_0013_mixed_pool `
  --games 50 `
  --concurrent-games 8 `
  --model-a random `
  --model-b fresh_supervised_b.chk
```

Important flags:

- `--server-uri`
  - direct websocket target override
- `--showdown-dir`
  - local clone root
- `--client-dir`
  - local Showdown client root
- `--serve-client`
  - `1` or `0`
  - when enabled, the runner serves the local client and prints a local spectate URL
- `--worker-think-mode`
  - default mode for non-`random` model specs
  - if a model spec is exactly `random`, that worker group always uses random mode
- `--model-a-weight` / `--model-b-weight`
  - population split across the fixed worker pool
- `--worker-pairs`
  - explicit shard-producing pair count
  - enables queued shard mode
- `--worker-games`
  - per-worker max game budget in queued shard mode
  - defaults to `ceil(--games / --worker-pairs)` when omitted

Notes:

- the tool assumes the local Showdown server clone already exists
- you can set it up with:
  - `python py/tools/setup_showdown_local.py`
- that setup helper also clones and builds the local client
- if the client build step fails, the runner can still try serving `testclient-old.html` from the cloned client repo
- if the Showdown server exits unexpectedly, the run fails and workers are terminated
- worker disconnects are handled first by communicator reconnect logic; if a worker exits, the orchestrator respawns it before shutdown begins
- in queued shard mode, a worker that exits after logging `reached max games (...)` is treated as complete, not crashed
