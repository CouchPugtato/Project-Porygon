# Checkpoint Compatibility

Checkpoint compatibility has three independent requirements:

1. The stored observation input dimension must match `observation_flat_size()` in the current runtime, apart from the one explicitly supported pre-active-slot migration described below.
2. The stored action count and flat action layout must match `OBS_NUM_ACTIONS` and `enum ObsAction`.
3. The serialized parameter count must match the current model layout, the pre-entity or pre-joint factorized layouts, the earlier factorized layout without move-target heads, or the supported legacy flat-head layout.

Current checkpoints serialize:

- GRU recurrent parameters
- the legacy flat policy head
- factorized policy heads for each active slot:
  - move versus switch kind
  - move slot
  - switch target
  - tera choice
  - move target
- a symmetric joint-action compatibility head used when both active slots act
- one shared Pokémon entity encoder and residual decoder
- value head
- basic trainer state
- a CRC-32 integrity checksum

## Format versions and integrity

Version 2 appends a CRC-32 footer covering the complete serialized header and parameter payload. The loader computes the checksum before importing parameters and rejects any mismatch. This detects accidental byte corruption; it is not intended as protection against deliberate tampering.

Version 1 checkpoints remain loadable for migration and are reported as `checksum=unverified` because they have no integrity footer. Every newly saved checkpoint uses version 2 and is reported as `checksum=verified` after a successful load.

## Crash-safe saving

Checkpoint saves are staged in `<checkpoint>.tmp` in the same directory as the destination. The saver writes the complete header and parameter payload, flushes the C stream, forces the file data to disk, closes it, and only then atomically replaces the destination.

If serialization, flushing, closing, or replacement fails, the staging file is removed and an existing destination checkpoint is left unchanged. A process interruption before replacement can leave a partial `.tmp` file, but the previous checkpoint remains loadable and the next save safely overwrites the stale staging file.

Only one process should write a given checkpoint path at a time. Different training runs should continue to use distinct output paths.

## Legacy flat-head loading

`gru_model_import_parameters()` recognizes the older parameter layout that ends after the flat policy and value heads. When that layout is loaded, it bootstraps the factorized heads from the flat policy head.

Checkpoints from the factorized architecture before explicit move targeting also remain loadable. Their existing heads are restored unchanged and the new move-target heads are initialized neutrally, so every legal target starts with equal probability before further training.

Factorized checkpoints created before joint-action scoring also remain loadable. The compatibility head is initialized to zero, so the initial joint policy reduces to the existing slot-specific action scores until the new head is trained.

Checkpoints created before the shared entity encoder remain loadable. Both sides of the residual entity path initialize to zero, preserving the checkpoint's existing GRU input behavior exactly while allowing later training to activate the shared path.

The later addition of explicit non-active/left/right role features added three inputs to each of the 12 Pokémon blocks. A checkpoint from the older legacy flat-head layout can be migrated across that exact 36-input schema transition. The loader inserts zero-weight columns at the known locations, so the old policy and value outputs are unchanged until later training learns from the new inputs. This exception requires the exact old input size, action count, and legacy parameter count; other observation-dimension mismatches are still rejected.

This supports migration, but does not make every old checkpoint a valid baseline. Reconstruction semantics and observation contents may have changed even when the numeric input dimension happens to match.

## Operational rules

- Runtime starts a fresh model only when no checkpoint argument is supplied. An explicitly requested invalid checkpoint terminates runtime.
- Training starts fresh only when its output checkpoint does not exist. An existing but invalid checkpoint terminates training.
- Evaluation and anchor loading always require a valid compatible checkpoint.
- A migrated flat-head checkpoint should be evaluated before training or promotion.
- New serious training runs should start from a checkpoint produced by the current observation and reconstruction code.
- Promotion comparisons must identify the exact checkpoint paths in their manifests.
- A reported save failure leaves the last successfully published checkpoint in place.

The loader validates:

- magic header and complete header length
- supported format version (v1 or v2)
- nonzero, bounded dimensions
- observation dimension against `observation_flat_size()`, with an exact-layout exception for pre-active-slot legacy checkpoints
- action count against `OBS_NUM_ACTIONS`
- exact current-factorized, pre-entity-factorized, pre-joint-factorized, pre-target-factorized, or legacy-flat parameter count
- exact file length, rejecting truncation and trailing data
- complete and matching CRC-32 footer for v2 checkpoints
- successful parameter import

Failures report a specific reason plus stored and expected metadata, including stored and computed checksums when integrity verification fails. Successful loads report their format version and `checksum=verified` or `checksum=unverified`; legacy parameter-layout loads also report `layout=legacy_flat migrated_factorized_heads=1`. A migrated observation layout additionally reports `migrated_active_slot_inputs=1`.
