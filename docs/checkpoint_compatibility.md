# Checkpoint Compatibility

Checkpoint compatibility has three independent requirements:

1. The stored observation input dimension must match `observation_flat_size()` in the current runtime.
2. The stored action count and flat action layout must match `OBS_NUM_ACTIONS` and `enum ObsAction`.
3. The serialized parameter count must match either the current model layout or the supported legacy flat-head layout.

Current checkpoints serialize:

- GRU recurrent parameters
- the legacy flat policy head
- factorized policy heads for each active slot:
  - move versus switch kind
  - move slot
  - switch target
  - tera choice
- value head
- basic trainer state

## Legacy flat-head loading

`gru_model_import_parameters()` recognizes the older parameter layout that ends after the flat policy and value heads. When that layout is loaded, it bootstraps the factorized heads from the flat policy head.

This supports migration, but does not make every old checkpoint a valid baseline. Reconstruction semantics and observation contents may have changed even when the numeric input dimension happens to match.

## Operational rules

- A checkpoint that fails to load must not be used; runtime fallback to a fresh model is only suitable for smoke testing.
- A checkpoint that later causes `observation_flatten failed` is incompatible with the current observation schema.
- A migrated flat-head checkpoint should be evaluated before training or promotion.
- New serious training runs should start from a checkpoint produced by the current observation and reconstruction code.
- Promotion comparisons must identify the exact checkpoint paths in their manifests.

The checkpoint header currently records a format version, dimensions, action count, parameter count, and trainer state. Parameter-layout compatibility is enforced by the importer rather than by the header version alone.
