# Training Pipeline

The trainable path is staged:

1. Python communicator captures or streams Showdown battle traffic.
2. The C runtime reconstructs battle state from `request` payloads and battle log `event` lines.
3. The observation builder converts raw state into the full one-hot `Observation` format.
4. Each decision point is appended to an `Episode`.
5. Live inference uses the GRU model to choose an action and emits `/choose ...`.
6. Replay logs preserve raw protocol messages and agent `decision` records.
7. Offline supervised training replays those logs, rebuilds episodes, and updates the GRU heads.
8. Checkpoints serialize model weights and trainer state.
9. Online RL reuses the same episodes and applies policy-gradient head updates on terminal rewards.

Current implementation notes:

- The repository now includes the protocol/session/raw-state/trainer/checkpoint path.
- Supervised and policy-gradient updates currently operate on the GRU output heads using sequence hidden states.
- Replay logs preserve raw battle messages so the environment can be rebuilt offline.
