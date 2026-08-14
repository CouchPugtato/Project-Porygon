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

- Runtime starts a fresh model only when no checkpoint argument is supplied. An explicitly requested invalid checkpoint terminates runtime.
- Training starts fresh only when its output checkpoint does not exist. An existing but invalid checkpoint terminates training.
- Evaluation and anchor loading always require a valid compatible checkpoint.
- A migrated flat-head checkpoint should be evaluated before training or promotion.
- New serious training runs should start from a checkpoint produced by the current observation and reconstruction code.
- Promotion comparisons must identify the exact checkpoint paths in their manifests.

The loader validates:

- magic header and complete header length
- exact supported format version
- nonzero, bounded dimensions
- observation dimension against `observation_flat_size()`
- action count against `OBS_NUM_ACTIONS`
- exact current-factorized or legacy-flat parameter count
- exact file length, rejecting truncation and trailing data
- successful parameter import

Failures report a specific reason plus stored and expected metadata. Successful legacy loads report `layout=legacy_flat migrated_factorized_heads=1`.
