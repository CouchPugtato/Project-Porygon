# Target learner input audit

The full model receives move identity, move type, visible Pokemon types and
active positions. The experimental target sidecar does not receive these
fields directly: it receives a frozen recurrent state, active slot, selected
move-slot number and legal target mask. Missing direct inputs do not prove
that the recurrent state has lost the information.

## Source findings

| Information | Full model input | Target sidecar input |
|---|---|---|
| Selected move identity/type | Per-move one-hot fields | Indirectly through the frozen state; move index selects a head |
| Visible target types | Per-Pokemon type fields and knowledge flags | Indirectly through the frozen state |
| Target position | Per-Pokemon active-slot fields | Fixed self/ally/foe-left/foe-right output classes |
| Move-to-target relationship | May be represented by the trained recurrence | A move-slot-specific linear readout of the frozen state |

`flatten_pokemon` in `src/observation.c` writes the move/type/position fields.
`transform_observation_input` in `src/gru_model.c` adds an entity-encoder
residual to the original features; it does not replace them with only a
32-value embedding. The recurrence then produces the state used by the heads.

`target_sidecar_predict` in `src/target_sidecar.c` selects separate weights for
each active slot and move index. Its logits are `weights * hidden + bias`.
This can use relationships already represented in the frozen state, but does
not create new nonlinear move/type interactions from raw context. It is not
correct to say the whole network sees only move-slot numbers, nor to conclude
from the interface alone that the frozen state is insufficient.

## Retained data

The read-only audit verified all valid-battle cache checksums from run 0285:

- 500 battles, 10,316 accepted decisions and 5,444 informative slot labels.
- A 128-value hidden state for each informative decision.
- Decision fields: `examples`, `hidden`, `parent_fingerprint`, `request_id`.
- Example fields: `best`, `legal`, `move`, `slot`.
- No retained observation, request, move identity/type or target context.
- The original split remains 406 training and 94 validation battles, with
  4,419 and 1,025 examples respectively.

The collection saved labels, not native-runtime replay inputs. The prepared
binary repeats those same cached states and labels; it does not recover the
missing context. Simulator seeds are not saved observations: replaying games
would be another simulation run and must be identified as such. Preferred
target labels must not be used to manufacture input matchup features.

Reproduce the cache audit without fitting or playing games:

```powershell
node .\tools\target_learning.js --audit-inputs run_0285_target_learning_collection models/diagnostics/run_0291_target_input_audit.json
```

The JSON records observed schemas, dimensions, context-field counts, split
counts, source hashes and collection-summary identity. It does not estimate
how much information the hidden state retained.

## Consequence

No explicit-input fit was run: the required raw fields are absent from the
approved existing dataset. No model/checkpoint, cache, split or default
inference behavior changed. This audit identifies a data-retention gap and a
possible representation bottleneck, not the cause of the flat win-rate screen.

The next proposed step needs approval for new data work: retain compact
player-visible selected-move and target context, then collect a bounded dataset
that supports one frozen-model, whole-battle-split comparison. Keep the old
cache and results intact. Do not silently substitute teacher-played data,
infer hidden team information, add teacher scores as inputs, or start a model
size/optimizer sweep to work around the missing fields.
