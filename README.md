# ShowdownAI

## Build

Default build path:

```powershell
cmake -S . -B build-fresh
cmake --build build-fresh
```

The default build does **not** require `libwebsockets` or `libcurl`. Live Showdown play is expected to use the Python communicator.

Legacy native C websocket mode can still be enabled explicitly:

```powershell
cmake -S . -B build-fresh -DBUILD_LEGACY_NATIVE_SHOWDOWN=ON
cmake --build build-fresh
```

## Live Flow

Use the Python communicator for live battles:

```powershell
python py/communicator/main.py --mode live --learner-command .\build-fresh\showdown_client.exe --runtime
```
