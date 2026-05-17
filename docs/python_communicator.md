# Python Communicator Options

Recommended structure:

- Python owns Showdown websocket/authentication/parsing.
- C owns battle-state reconstruction, observation encoding, GRU inference, training, and checkpoints.

Recommended IPC option:

- line-delimited JSON over `stdin/stdout`

Python to C messages:

```json
{"type":"battle_start","battle_id":"battle-gen9randombattle-123"}
{"type":"request","battle_id":"battle-gen9randombattle-123","payload":{"active":[...]}}
{"type":"event","battle_id":"battle-gen9randombattle-123","line":"|-weather|RainDance"}
{"type":"terminal","battle_id":"battle-gen9randombattle-123","reward":1.0,"result":"win"}
```

C to Python messages:

```json
{"type":"action","battle_id":"battle-gen9randombattle-123","choice":0}
{"type":"action","battle_id":"battle-gen9randombattle-123","command":"/choose move 1"}
```

Alternative transport options:

- local TCP socket if multiple battle workers are needed
- replay files for offline training/debugging
