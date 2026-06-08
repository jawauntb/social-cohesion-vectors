# Pair Bridge Ablation Audit

Date: 2026-06-08

## Question

After source-family bridge ablation passed, how many individual same-domain
bridge pairs are needed before SmolLM2 generated/control held-out source-family
transfer remains stable?

## Discovery-Regime Audit

Old regime:

- Source-only directions remained domain-specific in SmolLM2.
- Full held-out-domain bridge training passed.
- Source-family bridge ablation showed an asymmetric requirement: generated
  holdouts needed one bridge source family, while control holdouts needed two.

Transition:

- Added a pair-level bridge ablation audit. For each held-out source family, it
  trains on the full opposite benchmark domain plus path-stratified subsets of
  same-domain bridge pairs, then evaluates the held-out source family.
- The default bridge stratum is `slack_options_tested`, so subset sampling is
  organized around procedural future-option coverage rather than only pair IDs.

## Artifacts

Primary target-exact run:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_pair_bridge_direction_audit_target_exact/smol17_generated_control_pair_bridge_target_exact.md
```

Stress run with broader source-side sampling:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_pair_bridge_direction_audit_128/smol17_generated_control_pair_bridge.md
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

Target-exact summary:

| Held-out side | Minimum ready bridge pairs | Exhaustive at minimum | Failed subsets | Weakest ready margin |
| --- | ---: | --- | ---: | ---: |
| generated/source | 6 | False | 141 | +3.412 |
| control/target | 6 | True | 445 | +0.652 |

The target/control side is exact: with 12 candidate bridge pairs per held-out
control source family, the run evaluated all `16,384` control-side bridge
subsets. Every control-side subset with five or fewer bridge pairs still has at
least one held-out source-family failure; every six-pair subset passes.

Control/target holdout by bridge-pair count:

| Bridge pairs | Evaluated subsets | Min accuracy | Min margin | Failed subsets | Exhaustive | Ready |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 4 | 0.750 | -15.572 | 4 | True | False |
| 1 | 48 | 0.750 | -14.634 | 21 | True | False |
| 2 | 264 | 0.750 | -12.926 | 71 | True | False |
| 3 | 880 | 0.750 | -11.192 | 167 | True | False |
| 4 | 1980 | 0.750 | -8.280 | 147 | True | False |
| 5 | 3168 | 0.750 | -4.217 | 35 | True | False |
| 6 | 3696 | 1.000 | +0.652 | 0 | True | True |
| 7 | 3168 | 1.000 | +5.553 | 0 | True | True |
| 8 | 1980 | 1.000 | +10.389 | 0 | True | True |
| 9 | 880 | 1.000 | +15.368 | 0 | True | True |
| 10 | 264 | 1.000 | +20.271 | 0 | True | True |
| 11 | 48 | 1.000 | +26.091 | 0 | True | True |
| 12 | 4 | 1.000 | +32.374 | 0 | True | True |

Generated/source holdout by bridge-pair count in the same target-exact run:

| Bridge pairs | Evaluated subsets | Min accuracy | Min margin | Failed subsets | Exhaustive | Ready |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 4 | 0.800 | -5.054 | 1 | True | False |
| 1 | 120 | 0.800 | -6.059 | 26 | True | False |
| 2 | 1740 | 0.700 | -6.370 | 104 | True | False |
| 3 | 240 | 0.800 | -4.184 | 6 | False | False |
| 4 | 240 | 0.900 | -1.400 | 3 | False | False |
| 5 | 240 | 0.900 | -0.618 | 1 | False | False |
| 6 | 240 | 1.000 | +3.412 | 0 | False | True |
| 7 | 240 | 1.000 | +5.227 | 0 | False | True |
| 8 | 240 | 1.000 | +5.654 | 0 | False | True |
| 9 | 240 | 1.000 | +5.670 | 0 | False | True |
| 10 | 132 | 1.000 | +6.162 | 0 | False | True |
| 11 | 240 | 1.000 | +4.993 | 0 | False | True |
| 12 | 240 | 1.000 | +5.241 | 0 | False | True |

The broader 128-subset full-count stress run preserves the sampled generated
threshold at six bridge pairs. It also shows that an exhaustive generated-side
guarantee is much stricter: when all bridge-pair counts are included, the first
exhaustive generated/source pass appears only at 29 bridge pairs.

## Residual Finding

This narrows the bridge requirement from whole source families to a procedural
pair threshold:

- control-side transfer has an exact six-pair minimum under the current
  `slack_options_tested` bridge stratification;
- generated-side transfer has a stable sampled six-pair minimum, but not an
  exhaustive six-pair guarantee;
- below six bridge pairs, failures remain even when sampled subsets already
  cover several future-option paths.

The next bottleneck is therefore not whether a shared direction exists. It is
which six-pair bridge sets are robustly sufficient, and whether those sets can
be made intentionally path-complete rather than discovered by deterministic
sampling.

## Next Operation

Move from count ablation to bridge-set diagnosis:

- identify the six-pair subsets that pass with the weakest margins;
- compare their future-option coverage against failing five-pair subsets;
- add a path-complete bridge-set constructor or minimax subset search;
- keep exact control-side six-pair threshold and sampled generated-side six-pair
  threshold as regression baselines.

## Claim Boundary

This result is an activation-space diagnostic over generated and hand-authored
text benchmarks. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
