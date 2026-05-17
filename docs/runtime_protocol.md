# Runtime Protocol

The Python communicator and C learner exchange line-delimited JSON on `stdin/stdout`.

Python to C:

- `battle_start`
- `request`
- `event`
- `terminal`
- `battle_end`
- `error`
- `heartbeat`

Replay-only records may additionally contain:

- `decision`

Canonical messages:

```json
{"type":"battle_start","battle_id":"battle-gen9randombattle-123","format":"gen9randombattle","is_doubles":true}
{"type":"request","battle_id":"battle-gen9randombattle-123","request_id":17,"payload":{"active":[]}}
{"type":"event","battle_id":"battle-gen9randombattle-123","seq":208,"line":"|-weather|RainDance"}
{"type":"terminal","battle_id":"battle-gen9randombattle-123","result":"win","reward":1.0}
{"type":"battle_end","battle_id":"battle-gen9randombattle-123"}
{"type":"decision","battle_id":"battle-gen9randombattle-123","request_id":17,"action":0,"command":"/choose move 1"}
```

C to Python:

```json
{"type":"ready","capabilities":{"doubles":true,"training":true}}
{"type":"action","battle_id":"battle-gen9randombattle-123","request_id":17,"command":"/choose move 1"}
{"type":"log","message":"runtime initialized"}
{"type":"error","battle_id":"battle-gen9randombattle-123","message":"failed to map action"}
```
