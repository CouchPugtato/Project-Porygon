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

Examples:

```powershell
.\build-fresh\showdown_client.exe
python -m py.communicator.main
```

Those commands will use the tokens from these files automatically.

For the communicator, `--replay-save <run_name>` resolves to:

```text
matches/runs/<run_name>/<run_name>_raw.jsonl
```
