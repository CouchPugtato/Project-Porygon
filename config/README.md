Default argument files for entrypoints.

Rules:
- One token per line.
- Blank lines are ignored.
- Lines starting with `#` are comments.

Files:
- `showdown_client.args`
  - consumed by `showdown_client` when launched with no CLI args
- `communicator.args`
  - consumed by `py/communicator/main.py` when launched with no CLI args
- `selfplay_server.args`
  - consumed by `py/tools/selfplay_server.py` when launched with no CLI args
- `train_batch_selfplay.args`
  - consumed by `py/tools/train_batch_selfplay.py` before CLI args are applied
  - supports `--sample-files <N>` to train on a random subset of shards per epoch instead of the full run
  - supports `--reward-mode terminal|dense_additive` for RL reward shaping during replay reconstruction

Examples:

```powershell
.\build-fresh\showdown_client.exe
python -m py.communicator.main
python py/tools/selfplay_server.py
python py/tools/train_batch_selfplay.py
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

`--reconnect-seconds <n>` controls how long the communicator waits before reconnecting after an unexpected websocket/network drop.

`--guest-refresh-seconds <n>` controls how long a guest session may live before the communicator intentionally reconnects between battles to get a fresh guest account. `0` disables this.

`py/tools/selfplay_server.py` also supports weighted model pools for collection:

- `--model-a-pool <path>`
- `--model-b-pool <path>`
- `--pool-seed <int>`

Pool files are JSON with this schema:

```json
{
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
      "weight": 0.5
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

Sampling behavior:

- pool sampling happens per worker start, not per battle
- if a worker respawns, it samples again from that side's pool
- existing `--model-a` / `--model-b` behavior is unchanged when pool args are omitted

Examples:

```powershell
python py/tools/selfplay_server.py --run-name run_pool_test --games 100 --concurrent-games 8 --model-a-pool config/pools/a_pool.json --model-b random
python py/tools/selfplay_server.py --run-name run_pool_vs_pool --games 100 --concurrent-games 8 --model-a-pool config/pools/a_pool.json --model-b-pool config/pools/b_pool.json --pool-seed 123
```
