# Weighted Bridge Preservation Audit

Date: 2026-06-11

## Question

Does the weighted target-bridge repair preserve the original source/control
separations, and does it generalize to the strict accountability residual?

## Discovery-Regime Audit

Current regime:

- Artifact types: fresh bridge diagnostic JSON reports, constructed bridge
  direction evaluations, preservation summaries over source/target/fresh
  slices, stability summaries over perturbation failures, and dated research
  notes.
- Operations: summarize constructed bridge directions across `on_source`,
  `on_target`, `on_fresh_source`, and `on_fresh_target`; rerun cached
  accountability strict-ladder diagnostics with default and weighted target
  bridge settings.
- Gates: the accepted dissent repair must have zero constructed failures on
  all four evaluation slices. The accountability negative control should
  clarify whether weighted target bridges are a universal generated-pocket
  repair or a dissent/target-bridge-specific repair.

Action class:

- Preservation and negative-control audit. No new generation and no new Modal
  extraction were needed.

## Code Added

New preservation summarizer:

```text
src/social_cohesion_vectors/experiments/bridge_preservation_summary.py
```

New CLI:

```text
scripts/run_bridge_preservation_summary.py
```

New tests:

```text
tests/test_bridge_preservation_summary.py
```

The preservation summary checks constructed bridge rows across:

- `source`
- `target`
- `fresh_source`
- `fresh_target`

## Live Artifacts

Generated reports remain outside git:

```text
/tmp/social_cohesion_weighted_bridge_preservation_20260611/
```

Key reports:

```text
dissent_weighted_preservation.json
dissent_weighted_preservation.md
accountability_default_count9_stability.json
accountability_weighted_count9_targetx3_stability.json
accountability_weighted_count9_targetx3_preservation.json
```

## Dissent Preservation

The weighted dissent repair preserves all four evaluation slices across all
four model spaces:

| Metric | Result |
| --- | ---: |
| Models | `4` |
| Constructed direction rows | `32` |
| Evaluation rows | `16` |
| Failed pair evaluations | `0` |
| Worst margin | `+0.019` |
| Worst evaluation | `source` |
| Ready for preservation claims | `True` |

Model-level worst margins:

| Model | Worst margin | Failed pair evaluations |
| --- | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `+3.151` | `0` |
| Qwen2.5-0.5B layer `-2` | `+0.019` | `0` |
| Qwen2.5-7B layer `-2` | `+6.681` | `0` |
| TinyLlama-1.1B layer `-2` | `+0.388` | `0` |

This strengthens the weighted target-bridge result: it does not merely fix the
fresh dissent perturbations while damaging the original source/control slices.

## Accountability Negative Control

The strict accountability ladder was rerun as a fresh-source slice with the
same augmented target/control side. The default count-9 bridge setting fails in
all four model spaces:

| Model | Ready | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `False` | `-20.909` | `41` | `+27.330` | `0` |
| Qwen2.5-0.5B layer `-2` | `False` | `-2.229` | `50` | `-0.011` | `1` |
| Qwen2.5-7B layer `-2` | `False` | `-1.977` | `12` | `+11.245` | `0` |
| TinyLlama-1.1B layer `-2` | `False` | `-1.062` | `88` | `+0.818` | `0` |

The weighted count-9 target bridge setting improves the accountability strict
diagnostic but does not repair it except in Qwen2.5-7B:

| Model | Ready | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `False` | `-20.909` | `20` | `+55.796` | `0` |
| Qwen2.5-0.5B layer `-2` | `False` | `-1.206` | `43` | `+2.026` | `0` |
| Qwen2.5-7B layer `-2` | `True` | `+1.013` | `0` | `+19.395` | `0` |
| TinyLlama-1.1B layer `-2` | `False` | `-0.818` | `84` | `+1.908` | `0` |

The weighted accountability stability summary still has `147` constructed
failure rows across three failing models. The most failed perturbation remains
`positive_warmth_neutralized`, and failures include both `target_bridge` and
`source_bridge` rows.

## Finding

The preservation audit supports the weighted target-bridge repair as a real
dissent-specific bridge repair:

> Weighted target-bridge construction repairs the dissent target-bridge
> residual and preserves original source/control separations, but it does not
> globally repair generated-reference accountability pockets.

This is a useful separation. The current evidence now supports two distinct
residual classes:

- `accountability_after_harm`: generated-reference/source-pocket residual,
  repaired by local strict perturbation augmentation but not by weighted target
  bridges in small models.
- `dissent_after_mistake`: constructed target-bridge geometry residual,
  repaired by moderate target bridge coverage plus asymmetric target-side
  weighting.

## Next Move

The project is closer to a paper, but the next claim should stay narrow. The
right next move is a paper scaffold plus one more robustness check:

1. Draft a NeurIPS-style paper skeleton around residual taxonomy and bridge
   sufficiency.
2. Run a final report table that places the accepted and rejected repairs side
   by side.
3. Keep human/neural/real-world claims out of scope.

## Claim Boundary

This remains a generated/hand-authored text-benchmark activation diagnostic.
It does not establish causal steering, human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
