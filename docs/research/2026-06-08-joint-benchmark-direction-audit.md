# Joint Benchmark Direction Audit

Date: 2026-06-08

## Question

After SmolLM2 generated/control direction transfer failed, is the problem that
the two benchmarks define incompatible activation axes, or that each
source-only direction is too domain-specific?

## Discovery-Regime Audit

Old regime:

- Cross-benchmark direction transfer trained one generated direction and one
  non-generated control direction, then evaluated each direction on the other
  benchmark.
- SmolLM2 layer `-2` self-separated both benchmarks, but generated-on-control
  reached only `0.750` accuracy and control-on-generated reached only `0.950`.

Transition:

- The cross-benchmark direction-transfer report now also trains a pooled
  `joint` direction over source and target activations in the same model
  coordinate space.
- The report records per-pair margins, failed-pair tables, and separate joint
  readiness gates.

## SmolLM2 Joint Direction Result

Model:

```text
HuggingFaceTB/SmolLM2-1.7B-Instruct
```

Layer:

```text
-2
```

Rerun artifact:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_cross_benchmark_direction_transfer_joint/smol17_generated_layer-2_to_control_v2_layer-2.md
```

| Direction | Evaluation set | Accuracy | Mean margin | Min margin |
| --- | --- | ---: | ---: | ---: |
| generated | generated | 1.000 | +118.050 | +13.567 |
| control | control | 1.000 | +156.851 | +65.028 |
| generated | control | 0.750 | +27.216 | -15.572 |
| control | generated | 0.950 | +20.484 | -5.054 |
| joint | generated | 1.000 | +106.457 | +21.173 |
| joint | control | 1.000 | +91.300 | +50.555 |

The pooled joint direction is accepted by the default transfer thresholds on
both benchmarks.

## Failed Source-Only Transfers

Generated direction on control:

| Control pair | Margin |
| --- | ---: |
| `appeal_and_evidence::hand_authored_meeting_minutes_v1` | -4.998 |
| `privacy_exit::hand_authored_case_notes_v1` | -0.337 |
| `privacy_exit::hand_authored_incident_log_v1` | -15.572 |
| `harm_repair::hand_authored_policy_review_v1` | -2.742 |

Control direction on generated:

| Generated pair | Margin |
| --- | ---: |
| `generated-fault::deliberative_speed__generated_neighborhood_forum_constrained_repair_cross` | -5.054 |
| `generated-fault::fair_allocation__generated_neighborhood_forum_constrained_repair_cross` | -0.400 |

## Residual Finding

The SmolLM2 residual is not "no shared direction exists." A pooled direction
over generated and hand-authored examples separates both datasets with positive
minimum margins. The failure is stricter and more useful: source-only
directions do not generalize across benchmark domains in SmolLM2.

That means the next bottleneck is domain-invariant direction learning, not
basic out-of-family separability.

## Next Operation

The next experiment should test leave-one-domain or bridge-training regimes:

- train on generated plus a subset of control sources and hold out one control
  source;
- train on control plus a subset of generated sources and hold out one generated
  source;
- use the failed SmolLM2 pairs as fixed regression cases;
- preserve Qwen7B and SmolLM2 accepted self-separation baselines.

## Claim Boundary

This is an activation-space diagnostic over text benchmarks. It does not
support human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
