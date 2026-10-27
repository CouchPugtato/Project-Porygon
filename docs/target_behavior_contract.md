# Learned-target behavior contract

This is a diagnostic boundary, not permission to feed sidecar games into PPO.
The checkpoint, sidecar, target decoder and eligibility mode together define
the policy whose strength was measured in runs 0308-0310.

## Current path

`env_session.c` samples the legal joint action (or a single slot's factorized
action), then selects conditional targets. Supported slots use the sidecar;
unsupported slots keep the checkpoint's target distribution. Slot-local
eligibility can mix these two sources within one turn. Target-only argmax
consumes the usual sampling draw before replacing the target, preserving the
RNG schedule used by matched evaluations.

Accepted decisions copy `pending_old_log_prob` into the in-memory episode.
Ordinary episode JSON exports that value as `old_log_probs`; PPO reads it as
the behavior-policy denominator and evaluates the current checkpoint policy
for its numerator. That evaluator does not replay the sidecar or its visible
context. Sidecar terminal reports therefore omit training arrays and retain
`training_eligible: false`. Merely supplying a better denominator would not
make the current trainer consistent.

## Diagnostic records

Sidecar actions and terminal reports carry the same `behavior_policy` object:

- `version`: 1.
- `checkpoint_parameters` and `sidecar_parameters`: fingerprints of loaded
  parameter content, not filenames. The sidecar fingerprint includes its
  parent binding, dimensions and context mode.
- `parameter_fingerprint_algorithm`: `fnv1a64-native`. These are native-layout
  compatibility identifiers, not cryptographic hashes. Experiment manifests
  must still retain file SHA256 and executable provenance.
- `context_version`, `action_decoder`, `target_decoder`, and `eligibility`.
  Decoder or eligibility changes identify a different behavior policy even
  when both weight files are unchanged. `version` also versions the eligibility
  semantics; a rule change must not reuse the old descriptor.

Each action also carries `target_behavior`, version 1, with
`scope: "conditional_targets_only"` and `ppo_ready: false`. Its two slot entries
are null when no target is selected; otherwise they record source
(`checkpoint` or `sidecar`), legal mask, selected target, four effective
probabilities and selected probability. Target indices follow the existing
factorized target layout, not Showdown command target numbers.

Sampled targets use the legal renormalized distribution. Applied argmax uses a
point mass: selected probability 1, target log probability 0. An unsupported
slot remains sampled even when the configured sidecar decoder is argmax.
The zero-mass sampling fallback assigns probability 1 to the first legal
target. Serialization does not sample or advance recurrent state.

The runtime's pending log probability now accounts for applied argmax as a
point mass rather than scoring it with the pre-decoder softmax. This changes
diagnostic accounting only; action selection and RNG consumption are unchanged.
The target record deliberately does not claim an exact full-turn likelihood:
the older action-log helpers still floor very small probabilities.

## Training gate and remaining work

The native episode reader rejects explicit `training_eligible: false` records
and records carrying unsupported `behavior_policy` or `target_behavior`
metadata, even if complete episode arrays are attached. Legacy ordinary
episodes without these fields remain readable. No collector or trainer can
opt into this diagnostic schema yet.

Before resuming RL, choose a training-consistent policy and implement its
numerator as well as its denominator. A sampled sidecar preserves target
exploration; deterministic argmax has no support on alternative targets and
cannot simply be treated as the old softmax for PPO. Another option is to
distill the stronger bundle into the ordinary actor before ordinary PPO.
Neither route has been selected or enabled by this change.

A future episode schema needs accepted-only per-decision visible context,
eligibility/source, immutable bundle identity, complete legal action/target
support and matching likelihood reconstruction. It must distinguish rejected
choices from accepted retries and establish initial likelihood ratios of 1
for an unchanged policy. Keep evaluation-only historical games excluded.

## Bounded verification

Native regressions compare recorded probabilities with independent sidecar
predictions, check argmax's zero target log probability, and preserve sampled
checkpoint fallbacks on mixed turns. They also check reload/mutation identity,
decoder/eligibility identity differences, and rejection of ineligible records
with otherwise valid episode arrays. Existing tests continue checking choices,
hidden state and RNG consumption against the unchanged runtime path.

The run 0312 diagnostic additionally compares old/new executable responses on
one fixed public request with the frozen student and sidecar, in sampled,
whole-turn argmax and slot-local argmax modes. This is a protocol check, not a
strength evaluation or learning experiment. The deployed executable is not
replaced; the new build stays in `build-local/model-recovery`.
