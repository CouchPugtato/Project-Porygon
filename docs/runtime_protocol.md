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
- `episode_complete`

Live `episode_complete` records contain the full observation sequence, legal masks, paired flat actions, factorized action choices, per-slot move-target indices and masks, rewards/dones, rollout-time log probabilities and values, plus the frozen actor checkpoint in `policy_tag`. On two-action turns, the stored log probability includes one normalized joint-action probability plus any selectable target probabilities; one-action turns retain the per-slot factorized probability. Target indices represent self, ally, left foe, or right foe and are trained only when the selected move exposes a legal target choice. `--train-live-ppo` validates the policy tag before accepting the batch.

Canonical messages:

```json
{"type":"battle_start","battle_id":"battle-gen9randombattle-123","format":"gen9randombattle","is_doubles":true}
{"type":"request","battle_id":"battle-gen9randombattle-123","request_id":17,"payload":{"active":[]}}
{"type":"event","battle_id":"battle-gen9randombattle-123","seq":208,"line":"|-weather|RainDance"}
{"type":"terminal","battle_id":"battle-gen9randombattle-123","result":"win","reward":1.0}
{"type":"battle_end","battle_id":"battle-gen9randombattle-123"}
{"type":"decision","battle_id":"battle-gen9randombattle-123","request_id":17,"action":0,"command":"/choose move 1 2"}
```

C to Python:

```json
{"type":"ready","capabilities":{"doubles":true,"training":true}}
{"type":"action","battle_id":"battle-gen9randombattle-123","request_id":17,"command":"/choose move 1 2"}
{"type":"log","message":"runtime initialized"}
{"type":"error","battle_id":"battle-gen9randombattle-123","message":"failed to map action"}
```

## Tactical evaluation mode

`--battle-agent <checkpoint> --tactical-teacher 1` enables a tactical override
with the named checkpoint as its fallback. The ready record advertises
`capabilities.tactical_teacher: true` and `capabilities.training: false`.
Action records retain their usual fields and add a `tactical` object containing
the applied flag, reason, supported-choice count, and available power scores.

Terminal `episode_complete` records in this mode contain coverage counts and
`training_eligible: false` instead of episode arrays. Their policy tag is
`tactical-eval-v1`; they are evaluation summaries and cannot be parsed as normal
training episodes. Counters measure emitted decision attempts, including retries.

## Greedy neural evaluation

`--battle-agent <checkpoint> --inference-policy greedy` selects the
highest-probability complete legal turn, including conditional targets. The
default remains `--inference-policy sample`; training and ordinary collection
are unchanged. Greedy mode excludes tactical overrides and counterfactual
interventions, and non-sampling decoding is rejected outside runtime mode.

The ready record advertises `capabilities.inference_policy: "greedy"` and
`capabilities.training: false`. Actions add `inference_policy: "greedy"` and
`training_eligible: false`. Terminal records carry `policy_tag: "greedy-eval-v1"`,
the same decoding/eligibility fields, and `greedy_decisions` (accepted decisions).
They omit episode arrays and cannot be parsed as on-policy PPO episodes.

The decoder shares request action/target enumeration and pair legality with
the tactical selector but does not call its power-scoring rules. It consumes
the neural policy snapshot without advancing the model or RNG. Selection
includes joint action mass and each move's legally normalized target mass;
single-action turns use their slot's conditional factors. Ties keep the first
enumerated legal turn. A legality validator changing its selected actions is
an evaluation error, not a silent random fallback.

## Passive tactical audit

`--battle-agent <checkpoint> --tactical-audit 1` adds a `decision_audit` object
to action records without changing neural choices, RNG, or recurrent state.
It works with sampled or greedy decoding and excludes teacher overrides and
counterfactual interventions. Ready capabilities advertise `tactical_audit: true`
and `training: false`. Actions are marked `training_eligible: false`.

The object records selected effective power, unscored moves, the teacher's
supported best choice, and the best targets with the selected moves/switches/
Tera held fixed. Slot records identify move IDs, damaging/status classification,
Protect, Tera, and target-choice counts. These are visible-type heuristic scores,
not HP damage, action values, or loss explanations. Unsupported choices are
explicit; their zero contribution must not be called a tactical mistake.

Terminal records use `policy_tag: "tactical-audit-v1"`, `audit_decisions`
(accepted choices), decoding mode, and `training_eligible: false`. They omit
PPO arrays. The benchmark matches proposals to acceptance/rejection messages,
retains only accepted decisions, and verifies the terminal count.

## Target-only evaluation intervention

`--battle-agent <checkpoint> --tactical-targets 1` first samples and validates
the usual neural choice, then considers alternative targets for those exact
actions. Only a fully scored strict effective-power improvement changes target
indices. Forced replacements, unsupported turns (including Protect and Tera),
and ties remain unchanged. It consumes no extra RNG and preserves both flat
actions, target masks, and recurrent state at that decision. Later trajectories
can diverge because the resulting targets affect the battle.

This mode excludes greedy decoding, the passive audit, teacher overrides, and
counterfactual interventions. Ready capabilities include `tactical_targets: true`
and `training: false`. Actions add `training_eligible: false` and
`target_intervention: {applied, sampled_command}`, retaining the original
pre-intervention command. Terminal records use `policy_tag: "tactical-targets-v1"`
and contain `target_decisions` and `retargeted_decisions`, both counting accepted
choices only. They omit PPO arrays. No deployment default changes.
