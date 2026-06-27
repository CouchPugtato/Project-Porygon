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
