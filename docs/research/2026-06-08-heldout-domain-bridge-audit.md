# Held-Out Domain Bridge Audit

Date: 2026-06-08

## Question

Can the SmolLM2 generated/control mismatch be closed by bridge training, or do
held-out generated and hand-authored source families still fail when one source
family is excluded from training?

## Discovery-Regime Audit

Old regime:

- Source-only generated and control directions self-separated on SmolLM2 layer
  `-2`, but failed cross-benchmark direction transfer.
- A pooled generated+control joint direction separated both benchmarks, showing
  that a shared separable direction exists.

Transition:

- Added a held-out-domain bridge-training audit. For each held-out source
  family, it trains on the full opposite benchmark domain plus all remaining
  same-domain source families, then evaluates the held-out source family.
- This directly tests source-family generalization with some bridge examples
  from both domains.

## Artifact

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_heldout_domain_direction_audit/smol17_generated_control_heldout_domain.md
```

## Result

Model:

```text
HuggingFaceTB/SmolLM2-1.7B-Instruct
```

Layer:

```text
-2
```

Summary:

| Gate | Result |
| --- | ---: |
| readiness | `heldout_domain_ready` |
| source holdout folds | 4 |
| target holdout folds | 4 |
| source holdout minimum accuracy | 1.000 |
| source holdout minimum margin | +10.222 |
| target holdout minimum accuracy | 1.000 |
| target holdout minimum margin | +32.374 |
| failed pairs | 0 |

Target/control held-out source folds:

| Held-out control source | Test pairs | Accuracy | Min margin |
| --- | ---: | ---: | ---: |
| `hand_authored_case_notes_v1` | 4 | 1.000 | +39.793 |
| `hand_authored_incident_log_v1` | 4 | 1.000 | +32.374 |
| `hand_authored_meeting_minutes_v1` | 4 | 1.000 | +57.376 |
| `hand_authored_policy_review_v1` | 4 | 1.000 | +49.444 |

Source/generated held-out source folds:

| Held-out generated source | Test pairs | Accuracy | Min margin |
| --- | ---: | ---: | ---: |
| `generated_fault_class_cross_fault` | 10 | 1.000 | +52.031 |
| `generated_fault_class_lexical_adversarial` | 10 | 1.000 | +42.987 |
| `generated_fault_class_primary` | 10 | 1.000 | +10.222 |
| `generated_fault_class_source_diverse` | 10 | 1.000 | +75.630 |

## Residual Finding

Bridge training closes the SmolLM2 generated/control transfer failure at the
source-family level. The earlier failed source-only transfer is therefore not
evidence that generated and hand-authored procedural-justice examples are
incompatible in SmolLM2. It is evidence that a direction trained on only one
domain is too domain-specific.

The weakest passed fold is `generated_fault_class_primary`, with minimum margin
`+10.222`. That source remains the first source to watch in stricter future
audits.

## Next Operation

The next gate should be stricter than bridge training:

- leave one whole domain out is already the failing source-only transfer;
- bridge training with same-domain source-family holdout now passes;
- the next useful regime is minimal-bridge or bridge-ablation: how few
  opposite-domain examples are needed before SmolLM2 transfer becomes stable?

## Claim Boundary

This result is an activation-space diagnostic over generated and hand-authored
text benchmarks. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
