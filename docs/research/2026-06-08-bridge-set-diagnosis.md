# Bridge Set Diagnosis

Date: 2026-06-08

## Question

After pair-count ablation found a six-pair SmolLM2 bridge threshold, what
separates failing five-pair bridge sets from passing six-pair bridge sets?

## Artifact Read

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_pair_bridge_direction_audit_target_exact/smol17_generated_control_pair_bridge_target_exact.json
```

This artifact is generated output and stays out of git.

## Target/Control Diagnosis

The target/control side is exact for the current 16-pair control:

- all `16,384` control-side bridge subsets were evaluated;
- every six-pair control bridge subset passes;
- every five-or-fewer-pair threshold still has at least one failure.

At bridge count `5`, there are `3,168` evaluated target/control subsets and
`35` failures. All `35` failures have the same held-out source family and
failed pair:

```text
held_out_group = hand_authored_incident_log_v1
failed_pair = privacy_exit::hand_authored_incident_log_v1
```

The weakest failing five-pair subset has all eight future-option paths:

```text
margin = -4.216725
bridge_path_values = appeal,dissent,evidence_access,exit,privacy_choice,proportional_review,refusal,repair
bridge_pairs =
  appeal_and_evidence::hand_authored_case_notes_v1
  appeal_and_evidence::hand_authored_meeting_minutes_v1
  harm_repair::hand_authored_case_notes_v1
  privacy_exit::hand_authored_meeting_minutes_v1
  voice_under_pressure::hand_authored_meeting_minutes_v1
```

This is the important regime update: all-eight future-option coverage is not
sufficient by itself. A bridge set can mention every procedural path and still
fail the incident-log privacy/exit holdout.

Pair frequencies among the `35` failing target/control five-pair subsets:

| Bridge pair | Failed-set count |
| --- | ---: |
| `appeal_and_evidence::hand_authored_case_notes_v1` | 34 |
| `privacy_exit::hand_authored_meeting_minutes_v1` | 30 |
| `voice_under_pressure::hand_authored_meeting_minutes_v1` | 30 |
| `harm_repair::hand_authored_case_notes_v1` | 19 |
| `appeal_and_evidence::hand_authored_meeting_minutes_v1` | 18 |
| `voice_under_pressure::hand_authored_case_notes_v1` | 9 |
| `appeal_and_evidence::hand_authored_policy_review_v1` | 8 |
| `harm_repair::hand_authored_policy_review_v1` | 8 |
| `privacy_exit::hand_authored_case_notes_v1` | 8 |
| `voice_under_pressure::hand_authored_policy_review_v1` | 8 |
| `privacy_exit::hand_authored_policy_review_v1` | 2 |
| `harm_repair::hand_authored_meeting_minutes_v1` | 1 |

Interpretation:

- the failure concentrates on the incident-log `privacy_exit` holdout;
- failing five-pair bridges overuse case-notes/meeting-minutes combinations;
- policy-review privacy/exit and meeting-minutes harm-repair rows are rare in
  failing sets and common in passing sets;
- source-style coverage inside the bridge set matters, not only procedural path
  coverage.

## Generated/Source Diagnosis

The generated/source side is not exact at six pairs because the combination
space is much larger, but the sampled threshold is stable across the `32`,
`128`, and target-exact stress passes.

At bridge count `5`, the target-exact run sampled `240` generated/source
subsets and found one failure:

```text
held_out_group = generated_fault_class_cross_fault
failed_pair = generated-fault::deliberative_speed__generated_neighborhood_forum_constrained_repair_cross
margin = -0.618218
missing bridge paths = appeal,proportional_review
```

The weakest passing six-pair generated/source subset adds the missing
appeal/proportional-review surface through `fair_allocation` and covers all
eight future-option paths:

```text
margin = +3.412271
bridge_pairs =
  generated-fault::autonomy_after_conflict__generated_neighborhood_forum_modal_hf
  generated-fault::belonging_norms__generated_neighborhood_forum_modal_hf
  generated-fault::care_boundary__generated_neighborhood_forum_modal_hf
  generated-fault::data_choice__generated_neighborhood_forum_modal_hf
  generated-fault::fair_allocation__generated_neighborhood_forum_modal_hf
  generated-fault::forgiveness_after_harm__generated_neighborhood_forum_modal_hf
```

For generated/source transfer, the five-pair sampled failure looks more like a
true procedural-path omission: appeal and proportional review are absent, and
the failed held-out pair is `deliberative_speed`.

## Regime Update

The bridge-count result is now sharpened:

- six pairs is the current minimal bridge-count baseline;
- control/target six-pair sufficiency is exact for the current control;
- generated/source six-pair sufficiency is sampled-stable but not exhaustive;
- future-option coverage is necessary-looking but not sufficient for
  target/control robustness;
- source-style and case-family composition must become part of the bridge-set
  gate.

## Next Operation

Build a bridge-set sufficiency audit that treats candidate six-pair sets as
first-class artifacts:

- construct path-complete six-pair bridge sets intentionally;
- add source-style coverage constraints, especially for control-side policy
  review and repair/privacy rows;
- compare passing six-pair sets against the exact failing five-pair boundary;
- rerun generated/control direction transfer with the selected bridge sets
  before adding new model families.

## Claim Boundary

This diagnosis is an activation-space analysis over generated and hand-authored
text benchmarks. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
