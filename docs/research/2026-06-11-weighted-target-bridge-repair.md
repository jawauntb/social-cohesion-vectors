# Weighted Target Bridge Repair

Date: 2026-06-11

## Question

Can the `dissent_after_mistake` target-bridge residual be repaired by changing
the constructed bridge geometry, rather than by running another generation
sweep?

## Discovery-Regime Audit

Current regime:

- Artifact types: fresh generated bridge diagnostics, bridge construction
  parameters, bridge-stability summaries, and dated research notes.
- Operations: expose bridge training-set repetition controls; keep source
  bridge folds at the default; increase only target-bridge secondary
  target/control bridge weight; rerun the fixed dissent perturbation diagnostic
  using cached Modal activations.
- Gate: all constructed bridge directions must separate both fresh generated
  dissent perturbations and fresh hand-authored controls in SmolLM2,
  Qwen2.5-0.5B, Qwen2.5-7B, and TinyLlama.

Action class:

- Selection/geometry-side repair. No new text generation and no new activation
  extraction were needed beyond the target bridge repair artifacts from the
  preceding run.

## Code Added

The fresh generated bridge diagnostic now supports explicit bridge repetition
controls:

```text
scripts/run_fresh_generated_bridge_diagnostic.py
src/social_cohesion_vectors/experiments/heldout_domain_direction_audit.py
```

New CLI flags:

```text
--target-bridge-primary-repetitions
--target-bridge-secondary-repetitions
--source-bridge-primary-repetitions
--source-bridge-secondary-repetitions
```

Defaults are `1`, preserving previous behavior. Reports now record the
repetition settings in `inputs`.

## Live Artifacts

Generated artifacts stay outside git:

```text
/tmp/social_cohesion_bridge_weight_audit_20260611/
```

Key outputs:

```text
/tmp/social_cohesion_bridge_weight_audit_20260611/bridge_diagnostic_count9_targetx3_smol17.json
/tmp/social_cohesion_bridge_weight_audit_20260611/bridge_diagnostic_count9_targetx3_qwen05b.json
/tmp/social_cohesion_bridge_weight_audit_20260611/bridge_diagnostic_count9_targetx3_qwen7b.json
/tmp/social_cohesion_bridge_weight_audit_20260611/bridge_diagnostic_count9_targetx3_tinyllama.json
/tmp/social_cohesion_bridge_weight_audit_20260611/bridge_stability_summary_count9_targetx3.md
```

## Regime

The accepted weighted bridge setting is:

| Parameter | Value |
| --- | ---: |
| Target bridge pair count | `9` |
| Target bridge primary repetitions | `1` |
| Target bridge secondary repetitions | `3` |
| Source bridge primary repetitions | `1` |
| Source bridge secondary repetitions | `1` |

This is intentionally asymmetric. A symmetric `3x` secondary weighting breaks
source-bridge rows, while target-bridge-only weighting addresses the failure
cluster localized by the previous bridge-stability audit.

## Result

The weighted count-9 target bridge repair clears the constructed fresh bridge
gate across all four model spaces:

| Model | Readiness | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `fresh_generated_bridge_ready` | `+3.151` | `0` | `+55.796` | `0` |
| Qwen2.5-0.5B layer `-2` | `fresh_generated_bridge_ready` | `+0.220` | `0` | `+2.026` | `0` |
| Qwen2.5-7B layer `-2` | `fresh_generated_bridge_ready` | `+6.866` | `0` | `+19.395` | `0` |
| TinyLlama-1.1B layer `-2` | `fresh_generated_bridge_ready` | `+0.513` | `0` | `+1.908` | `0` |

The bridge-stability summary reports:

| Metric | Result |
| --- | ---: |
| Models | `4` |
| Failing models | `0` |
| Constructed failure rows | `0` |

This is the first repair in this loop that clears the dissent constructed
target-bridge residual across every tested model space.

## Interpretation

The previous target/content repair showed that adding availability-correct
target/control examples was not enough at the default six-pair bridge budget.
The new result shows that the residual can be repaired by a targeted bridge
geometry change:

> Dissent transfer under pseudo-side shortcut removal needs enough
> target/control bridge evidence and enough target-side weight in constructed
> target bridges. With nine target bridge pairs and `3x` target-side bridge
> weighting, the residual disappears across SmolLM2, Qwen2.5-0.5B,
> Qwen2.5-7B, and TinyLlama.

This does not yet prove a universal method. It is, however, paper-shaped: the
project now has separate repair regimes for a generated-reference source pocket
and a constructed target-bridge geometry pocket.

## Next Move

Run one more audit before drafting a paper claim:

1. Check whether the weighted target-bridge setting preserves the earlier
   non-dissent bridge successes on the original source/control benchmarks.
2. Run the same weighted bridge setting on the accountability strict ladder as
   a negative-control style check; it should not be necessary there.
3. If those pass, draft the paper around residual taxonomy and bridge
   sufficiency, not around broad social-effect claims.

## Claim Boundary

This remains a generated/hand-authored text-benchmark activation diagnostic.
It does not establish causal steering, human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
