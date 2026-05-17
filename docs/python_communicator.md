# Python Communicator

The communicator is a thin gateway:

- Python owns websocket connection, login, room lifecycle, and raw Showdown parsing.
- C owns state reconstruction, one-hot observation construction, GRU inference/training, replay storage, and checkpoints.

Implementation defaults:

- `asyncio`
- line-delimited JSON over `stdin/stdout`
- one learner subprocess
- optional replay capture without a learner subprocess

Runtime protocol details live in [runtime_protocol.md](/f:/Coding/Repositories/Project-Porygon/docs/runtime_protocol.md).

Recommended Python package layout:

- `py/communicator/main.py`
- `py/communicator/showdown_client.py`
- `py/communicator/ipc.py`
- `py/communicator/protocol.py`

Communicator modes:

- `live`
  - connect to Showdown
  - launch the C learner runtime
  - forward `battle_start`, `request`, `event`, `terminal`, and `battle_end`
  - send learner `action` messages back to Showdown

- `capture`
  - connect to Showdown
  - log raw protocol messages to a replay file
  - do not require a learner subprocess
