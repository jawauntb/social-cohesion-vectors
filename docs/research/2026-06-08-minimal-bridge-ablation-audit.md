# Minimal Bridge Ablation Audit

Date: 2026-06-08

## Question

After full bridge training closed the SmolLM2 generated/control transfer
failure, how many same-domain bridge source families are needed before held-out
source-family transfer becomes stable?

## Discovery-Regime Audit

Old regime:

- Source-only directions failed cross-benchmark transfer in SmolLM2.
- A pooled generated+control joint direction separated both benchmarks.
- Full bridge training passed when all non-held-out same-domain source families
  were available.

Transition:

- Added a minimal-bridge ablation audit. For each held-out source family, it
  trains on the full opposite benchmark domain plus every subset size of
  same-domain bridge source families, then evaluates the held-out source.

## Artifact

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_minimal_bridge_direction_audit/smol17_generated_control_minimal_bridge.md
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

| Held-out side | Minimum ready bridge source families | Failed ablation folds | Weakest ready margin |
| --- | ---: | ---: | ---: |
| generated/source | 1 | 1 | +6.162 |
| control/target | 2 | 6 | +12.384 |

Generated/source holdout by bridge count:

| Bridge source families | Folds | Min accuracy | Min margin | Ready |
| ---: | ---: | ---: | ---: | --- |
| 0 | 4 | 0.800 | -5.054 | False |
| 1 | 12 | 1.000 | +6.162 | True |
| 2 | 12 | 1.000 | +7.636 | True |
| 3 | 4 | 1.000 | +10.222 | True |

Control/target holdout by bridge count:

| Bridge source families | Folds | Min accuracy | Min margin | Ready |
| ---: | ---: | ---: | ---: | --- |
| 0 | 4 | 0.750 | -15.572 | False |
| 1 | 12 | 0.750 | -1.243 | False |
| 2 | 12 | 1.000 | +12.384 | True |
| 3 | 4 | 1.000 | +32.374 | True |

## Residual Finding

The bridge requirement is asymmetric:

- Holding out generated source families needs only one generated bridge source
  family when all control examples are available.
- Holding out hand-authored control source families needs two control bridge
  source families when all generated examples are available.

This says the control side is the stricter bridge-generalization surface in
SmolLM2. The weakest failed one-bridge control case is
`hand_authored_incident_log_v1`; it still fails when bridged only by
`hand_authored_case_notes_v1` or only by `hand_authored_meeting_minutes_v1`.

## Next Operation

Move from source-family-count ablation to pair-count ablation:

- find how many individual bridge pairs, not whole source families, are needed;
- stratify bridge pairs by procedural path so privacy/exit and repair are not
  accidentally omitted;
- keep the failed source-only and one-bridge control folds as regression cases.

## Claim Boundary

This result is an activation-space diagnostic over generated and hand-authored
text benchmarks. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
