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

The local run 0313 build extends this to version 2, with
`scope: "full_turn_diagnostic"`, `full_turn_log_probability` and
`likelihood_status`. The probability is unfloored and includes the legal
196-way joint action or single-slot factorization, followed by the actual
conditional target distributions. A legality-remapped or otherwise unsupported
decision has a null likelihood and `likelihood_status: "unsupported"`.
It must not be assigned the likelihood of its original proposal. Both versions
retain `ppo_ready: false`; run 0312's historical version-one records are not
rewritten or upgraded.

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

That limitation describes version 1 and the legacy in-memory `old_log_probs`.
Version 2 computes a separate diagnostic likelihood without those floors;
it does not silently replace legacy episode semantics. Terminal reports count
accepted supported/unsupported decisions in `likelihood_replay`. Rejected
proposals never increment either count, and rejection clears the pending
likelihood. Acceptance must match the pending action indices and command.

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

## Frozen-bundle replay proof and proposed update scope

`target_behavior_log_probability` reconstructs eligibility, selected-move
context and decoder-aware targets from an immutable post-forward hidden state,
public observation/request, final choice and fresh actor-policy snapshot.
It reuses the neutral probability factorization in `policy_decoding.c`.
Native tests replay accepted snapshots after pending runtime state is cleared,
with initial ratios of 1, unchanged choices/hidden state/RNG, independent
hand-calculated likelihoods, tiny probabilities and zero-probability alternative
targets. Single-slot status/revival choices and checkpoint fallbacks retain
their actual behavior. Altered target masks are rejected.

This is an in-memory snapshot proof, not a persistent accepted-record collector,
recurrent-history replay, crash-resume test or training result. Frozen hidden
snapshots are useful for the proposed head-only experiment: rejected requests
can advance live recurrent state even though they are not accepted training
examples. An accepted-only observation sequence must not be assumed to recreate
that state. A future collector must preserve exact frozen hidden states or all
state-advancing history, plus the public context and immutable bundle identity.

The narrow candidate update scope is kind/move/switch/Tera and joint-pair heads
only. Freeze the encoder, recurrent weights, sidecar, checkpoint target heads
(including unsupported-slot fallback), value head and eligibility/decoder rules.
This preserves the target function for any given state and proposed action,
without claiming that its usage frequency stays constant as actions change.

Two existing APIs are not safe substitutes:

- `gru_model_policy_gradient_update_heads` updates the legacy flat policy/value
  heads and bootstraps factorized heads; it is not this bounded joint-action
  optimizer.
- `gru_model_apply_accumulated_policy_adam_updates` freezes recurrence but also
  updates both checkpoint target heads. Zero current target gradients do not
  guarantee freezing when Adam moments already exist.

Implement an explicit action-head-only accumulator/updater and test both
positive/negative update direction and byte-identical frozen parameters before
collecting the pilot. Keep its optimizer state separate from broader training.
The sidecar's existing full-checkpoint binding also correctly rejects an actor
checkpoint after any weight update. A new experiment bundle must retain the
frozen reference checkpoint/sidecar identity, separately identify the updated
actor, and verify that only allowlisted action parameters differ. Do not weaken
the historical loader or simply rebind sidecar metadata to a new checkpoint.
Persistent collection and that update/compatibility boundary remain unfinished;
no learning pilot starts from the diagnostic schema alone.

The run 0314 native step implements that accumulator/updater as
`GruActionHeadOptimizer` in `gru_model.c`. It is an explicit C API, not a trainer
mode or collection flag. Its 18 destination arrays are the weights/biases for
both slots' kind/move/switch/Tera heads and the joint-pair head. It also refreshes
the existing derived flat-policy cache. It never registers target, value,
encoder or recurrent parameters as optimizer destinations.

`gru_action_head_accumulate_hidden` accumulates the signed negative action
log-likelihood gradient at a supplied frozen hidden state. Conditional targets
are excluded from this objective, including checkpoint fallback targets.
Joint and single-slot gradients reuse the existing head-gradient kernels;
the hidden gradient is discarded rather than backpropagated. This does not
validate context, acceptance, bundle compatibility or full-turn PPO support:
those remain the collector/trainer's responsibility.

The optimizer owns compact action-only gradient, first/second moment and
per-sample scratch buffers. It neither reads nor clears the model's broader
training gradients or Adam state. Failed samples/settings do not discard a
queued valid minibatch. Apply averages/clips action gradients and checks proposed
head updates for non-finite values before changing parameters/moments. The
model must outlive this optimizer; callers must not interleave external model
updates and claim the whole training interval remained frozen. Optimizer state
is not serialized, and existing checkpoint layouts/loaders remain unchanged.

Fresh-state positive/negative tests cover a joint move pair, a single-slot
switch and a single-slot Tera move. Joint-pair parameters demonstrably change.
An independent export-layout mask checks every frozen parameter byte, including
the entity encoder/decoder. A broader-optimizer control proves nonzero target
momentum still moves target weights with zero current target gradients; the
new optimizer does not inherit it. Scoped momentum and queued unrelated critic
gradients also leave frozen parameters intact. Mixed-turn tests on both slots
preserve target-source eligibility, learned argmax and sampled checkpoint
fallbacks, next recurrent states and value predictions while action probability
increases. Direction tests use fresh scoped moments; Adam momentum does not
guarantee every later step follows the current advantage's sign.

This is an update-kernel proof, not PPO training or a strength improvement.
PPO clipping, exploration/anchor objectives, persistent accepted records,
optimizer resume and frozen-reference/updated-actor compatibility remain
outside this API. No collection, training run or deployed executable change
was made in this step.

## Frozen reference and updated actor compatibility

Run 0315 adds a read-only compatibility boundary, not a runtime/training flag.
`gru_model_action_head_compatibility_reason(reference, actor)` returns null on
compatibility and a diagnostic reason otherwise. Input/hidden/action dimensions
and encoder mode must match. All loaded parameters must be finite. Recurrent
weights/biases, both checkpoint target heads, entity encoder/bias/decoder and
value weights/bias are compared byte-for-byte. Only the action heads listed
above may differ.

For a changed actor, the legacy flat-policy cache must exactly match the sum
of its factorized action heads, using the existing synchronization arithmetic.
It is not an unrestricted extra trainable head. An entirely unchanged reference
and actor are accepted without normalizing their historical cache. Neither
check mutates weights, advances hidden state or consumes RNG.

`target_sidecar_load_for_action_actor(path, reference, actor, expected_identity,
failure_reason)` first checks compatibility, then calls the existing strict
loader against the **reference**, and finally verifies the expected sidecar
fingerprint. This rejects both different-parent sidecars and valid but wrong
same-parent sidecars. The expected identity is required and must come from
trusted experiment provenance; fingerprints are not a substitute for manifest
file SHA256. The loaded sidecar keeps its original parent fingerprint. Calling
the historical `target_sidecar_load(path, actor)` on an updated actor still
fails exactly as before. No sidecar is rewritten or rebound.

Tests accept a real scoped-head update, reject mutations in each of 18 frozen
parameter regions through both the checker and bundle loader, and reject bad
dimensions, non-finite action parameters, arbitrary cache edits, missing identity
and wrong sidecars. They also verify non-mutating/RNG-preserving load behavior
and preservation of an unchanged legacy cache.

This API validates loaded members of a proposed experiment bundle. It does not
yet load a manifest, pin decoder/eligibility config, emit a new actor-aware
runtime identity, or enable that actor in `EnvRuntime`. Persistent accepted
records, optimizer resume and PPO integration remain separate work. Research
checkpoints, sidecars, historical loaders and the deployed executable remain
unchanged.

## Manifest-pinned runtime launch

Run 0316 adds `tools/target_bundle.js`, a single-process evaluation-only launcher.
Its JSON manifest pins the reference checkpoint, actor checkpoint, sidecar and
native executable by SHA256, plus the sidecar's native parameter fingerprint,
context version 1, sampled action decoding, target decoder, eligibility mode,
inference seed, one OMP thread and `training_eligible: false`. Paths are relative
to the manifest directory. Unknown fields and policy overrides are rejected.

The manifest itself must have a separately supplied expected SHA256. Editing
decoder/eligibility settings changes that hash; callers must explicitly approve
a new pin rather than silently reusing the old policy identity. The launcher
verifies file hashes, stages private verified copies of the three model members,
rechecks the executable and launches the native runtime. Normal exit, startup
failure and handled child termination clean up those copies. Abrupt process/OS
failure can leave only the temporary copies, not modify research artifacts.
Launch uses the existing 65% CPU/memory gates, a 750 ms stagger and one native
OMP thread. Gates can delay readiness and are not hard operating-system caps.

Use the committed reference smoke manifest with this local build:

```powershell
node .\tools\target_bundle.js --manifest .\config\experimental\target_bundle_reference_smoke.json --manifest-sha256 c201d9402ff4f880bb20ac452832e2ee054609565f689b49955e17cd091d1e1e
```

This is a stdin/stdout battle agent; it waits for protocol messages after
`ready`. It is not a self-play or training command. The smoke manifest uses the
same frozen student as reference and actor; it does not publish an updated
actor. Its executable hash pins `build-local/model-recovery/showdown_client.exe`.
A later recompilation may produce different bytes: make and review a new
manifest/pin rather than removing the executable check. The deployed
`build-fresh` executable remains unchanged.

The native path calls the strict reference/actor compatibility boundary and
keeps the original sidecar binding. It emits behavior-policy identity version
2 (`kind: "frozen-target-action-actor-v1"`) in readiness, every action and the
terminal summary, including distinct reference/actor parameter fingerprints
and the manifest SHA256. Standard sidecar loading keeps version-one identity
and its original strict full-parent check. All bundle episodes remain
PPO-ineligible and omit training arrays.

The Node launcher owns JSON/SHA256 validation; the native `--action-bundle-*`
arguments are its internal compatibility-routing interface, not a standalone
file-hash verifier. This is local experiment provenance checking, not a security
sandbox or protection against an adversary able to replace the launcher itself.
Decoder/eligibility and executable changes require new trusted manifest pins.
Loading the extra reference happens before resetting the actor inference seed,
so checkpoint construction does not shift its sampling draws.

Five JavaScript regressions cover schema/pins, changed settings/members, staging
races and override rejection. Native tests configure a genuinely scoped-updated
actor with its distinct identity. A six-decision protocol trace using the frozen
real student/sidecar produces identical actions and full-turn likelihoods under
direct and manifest loading; ready/action/terminal identities agree and all six
accepted decisions have supported likelihoods. This is not a battle-strength
test. Manifest routing is now available, but persistent accepted records,
optimizer persistence and learning integration are still separate work.

## Accepted bundle records (run 0317)

`tools/target_records.js` publishes records from one **complete saved native
protocol trace**. It is not connected to live self-play yet. Pass the pinned
manifest, a JSONL trace, and a dedicated output directory:

```powershell
node .\tools\target_records.js --manifest .\models\diagnostics\run_0317_accepted_records_bundle.json --manifest-sha256 8785064987785ffd0e3b54e27a83634f6305a21021c5059e1a04fbfcb6b76284 --trace .\models\diagnostics\run_0317_accepted_records_trace.jsonl --output .\models\diagnostics\run_0317_accepted_records
```

The native `--action-bundle-records 1` switch is internal to this collector and
requires an explicit bundle. Default battle-agent output is unchanged. Only
accepted decisions emit `bundle_decision` snapshots. Each contains the bundle
identity, request/decision IDs, exact frozen hidden state, legal action mask,
factorized choice, selected-move public context, target distributions and the
full-turn log likelihood. Hidden state uses eight hexadecimal digits per
float32 word; preserving bits avoids losing signed zero through JSON numbers.
Accepted choices whose likelihood is unsupported emit an audit-only
`bundle_decision_skipped`, without a usable hidden-state/likelihood snapshot.
Rejected choices emit neither frame.

The artifact retains the complete input trace, including rejected requests and
public updates: those requests can advance recurrent state even though they
are not training examples. It records input/content SHA256 hashes and is
published atomically under a battle-ID hash filename. A directory is pinned
to one manifest. Existing complete files are replayed and compared, not
appended or rewritten; a changed trace for the same battle ID is refused.
Uncommitted `.tmp` output receives no credit. Corrupt committed files fail
closed instead of silently starting the battle again.

Resume is **at battle boundaries**. An interrupted battle must be replayed
from the beginning using its original complete trace. There is no simulator
mid-battle restoration, streaming trace journal, or live capture integration.
Read-only verification reruns the artifact's retained input trace under its
pinned executable/model/sidecar and requires identical state, actions,
context, distributions and likelihoods:

```powershell
node .\tools\target_records.js --manifest .\models\diagnostics\run_0317_accepted_records_bundle.json --manifest-sha256 8785064987785ffd0e3b54e27a83634f6305a21021c5059e1a04fbfcb6b76284 --verify .\models\diagnostics\run_0317_accepted_records\84f61b610b206d51c65c405fb2664a2307e69cd93d998acb2175d2ce20a4308b.json
```

Run 0317 freezes a copy of the local executable in a hash-named `build-local`
path so later recompilation does not invalidate these saved records. Preserve
that executable to reproduce this diagnostic; the older run 0316 manifest
still pins its original build and must not be retagged. Replay is serial,
uses one OpenMP thread, retains the 65% CPU/memory launch gate, limits traces
to 8 MiB/1,024 requests, bounds native output to 16 MiB and allows 30 seconds
per native replay after the launch gate. These are bounded diagnostic limits,
not OS-enforced CPU/memory caps or settings for a large collection.

The real-model protocol fixture has six proposals, one rejected retry, five
supported accepted snapshots, no skips and exact saved replay/resume at ratio
one. It is not simulator gameplay or strength evidence. Seven JavaScript
regressions cover rejection/duplicate handling, bit preservation, target
support, interrupted publication, identity mixing and tamper detection;
native tests also record a genuinely action-head-updated actor. All three
native suites and 100 JavaScript tests pass. Artifacts remain explicitly
`training_eligible: false` and `ppo_ready: false`; optimizer persistence,
live capture and a scoped learner still require separate implementation.

## Live battle capture (run 0318)

`tools/target_record_collection.js` now connects the pinned bundle to the
existing `AgentProcess`/`drivePlayer` bridge and shared single-battle runner.
It captures exactly the native input messages and accepted snapshot outputs,
including rejected requests in the input history but not in the snapshot set.
The opponent is seeded random play, the candidate alternates sides, and each
new native process uses the manifest's fixed inference seed. That actual seed
is recorded instead of the ordinary runner's unused inference-seed schedule.

```powershell
node .\tools\target_record_collection.js
```

The command reads `config/experimental/target_record_live_smoke.json`. This is
a capped 1-6-game serial smoke collector, not a large collection command. It
uses one OpenMP thread, the existing 65% CPU/memory launch gates, and a maximum
180-second battle timeout. Those gates pause launches, not enforce OS resource
caps. Input/output byte limits remain the same as the saved-trace collector.

Each completed battle is replayed against the pinned native bundle before
publishing one atomic `battle_000N.json` file. This envelope holds the game
outcome, seeds, both players' rejection reports, the complete accepted-record
artifact, and a content hash. The artifact remains `training_eligible: false`.
These live envelopes contain an `artifact` member; they are not standalone
run 0317 artifact files or PPO episode batches.

Rerun the same command to resume. Committed files are verified and replayed,
not simulated again or rewritten. The summary can be rebuilt from battle files
if a crash loses the latest summary update. An interrupted battle has no commit
and restarts from its seeded setup at the same index. The runner does not save
or restore mid-battle simulator state. Explicit failures retain their available
input/output capture in `failure_000N.json`, outside the accepted battle set.
Configuration, source hashes and bundle pin must still match on resume; a code
or policy change needs a new reviewed run name rather than retagging old files.

The final six-game smoke saves 107 supported accepted decisions, with zero
skips and zero choice rejections. Every live snapshot reproduces exactly under
saved-trace replay. A six-game recheck after the startup-timeout cleanup fix
reproduces the same snapshots and outcomes; only Showdown's wall-clock events
differ between fresh simulations. Original raw timestamps are preserved, not
normalized in stored traces. Both smoke directories remain on disk. Resume of
the final run leaves all six committed files byte-identical and produces no
new simulator games.

Six new regressions cover bounded configuration, delayed-launch cleanup after
a timeout, trapped-switch rejection/retry capture, summary-loss recovery,
interrupted-battle restart, and changed/corrupt input refusal. Three native
suites and 106 JavaScript tests pass. The shared battle bridge only gains a
launch method and agent-factory hook; bundle-specific staging, capture and
publication stay outside the generic battle loop. No training, optimizer,
research checkpoint, deployed executable or root README changed. Action-only
optimizer persistence and scoped learning integration remain the next gate.

The run 0313 serial smoke compared six matched cases on both deployed and local
builds (12 games, seed 31301): sampled targets, whole-turn argmax, and slot-local
argmax, each on both sides against random. All completed without rejections;
native action sequences and outcomes were unchanged. The local build counted
157 supported accepted decisions and one unsupported decision. The latter was
the sampled case's request 349: both active Pokemon fainted with only one bench
survivor. The validator correctly emitted a switch/pass command; the diagnostic
abstained instead of assigning an independent two-switch sample's probability
to it. A native regression retains that case. Both argmax modes had all 108
accepted decisions supported. These counts test protocol coverage, not strength.
