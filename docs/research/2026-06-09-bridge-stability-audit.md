# Bridge Stability Audit

Date: 2026-06-09

## Question

Can the constructed-bridge failures exposed by the
`dissent_after_mistake` perturbation ladder be localized by model, bridge
family, evaluation slice, and perturbation type?

## Discovery-Regime Audit

Current regime:

- Artifact types: fresh generated perturbation bridge diagnostics,
  perturbation pair metadata, constructed bridge direction rows, failed-pair
  rows, failure clusters, and durable research notes.
- Operations: load existing bridge diagnostic JSON reports from `/tmp`; join
  failed fresh-source pair IDs with perturbation metadata; aggregate failures
  by model, evaluation slice, bridge family, and perturbation id.
- Gates: the summary should identify a targetable cluster. If failures are
  diffuse across bridge families, perturbations, and models, the next repair
  step is underdetermined.

Action class:

- Diagnostic summarization inside the bridge regime. This does not add new
  activations; it converts existing live bridge outputs into a repair map.

## Code Added

New bridge-stability summarizer:

```text
src/social_cohesion_vectors/experiments/bridge_stability_summary.py
```

New CLI:

```text
scripts/run_bridge_stability_summary.py
```

New tests:

```text
tests/test_bridge_stability_summary.py
```

## Live Artifacts

The summary was run over the four dissent perturbation bridge diagnostics:

```text
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/bridge_stability_summary.json
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/bridge_stability_summary.md
```

Input bridge diagnostics:

```text
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/bridge_diagnostic_smol17.json
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/bridge_diagnostic_qwen05b.json
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/bridge_diagnostic_qwen7b.json
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/bridge_diagnostic_tinyllama.json
```

## Summary

| Gate | Result |
| --- | ---: |
| Models | `4` |
| Constructed failure rows | `39` |
| Failing models | `3` |
| Worst constructed margin | `-18.296` |
| Most failed fresh-source perturbation | `negative_shortcuts_neutralized` |
| Most failed bridge family | `target_bridge` |

All `39` constructed failure rows are `target_bridge` failures. That is the
main point: the residual is not diffuse across source bridges and target
bridges.

## Model Rows

| Model | Readiness | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `not_ready` | `-18.296` | `22` | `+23.644` | `0` |
| Qwen2.5-0.5B layer `-2` | `not_ready` | `-1.072` | `11` | `-0.257` | `2` |
| Qwen2.5-7B layer `-2` | `not_ready` | `-1.936` | `4` | `+10.213` | `0` |
| TinyLlama-1.1B layer `-2` | `fresh_generated_bridge_ready` | `+0.163` | `0` | `+0.683` | `0` |

## Failure Clusters

The top fresh-source clusters are:

| Model | Evaluation | Bridge family | Perturbation | Failures | Directions | Min margin |
| --- | --- | --- | --- | ---: | ---: | ---: |
| SmolLM2 | `fresh_source` | `target_bridge` | `negative_shortcuts_neutralized` | `4` | `4` | `-18.296` |
| SmolLM2 | `fresh_source` | `target_bridge` | `negative_opening_warmth_removed` | `4` | `4` | `-17.208` |
| SmolLM2 | `fresh_source` | `target_bridge` | `original_reference` | `4` | `4` | `-6.572` |
| SmolLM2 | `fresh_source` | `target_bridge` | `negative_conditions_explicit` | `4` | `4` | `-3.898` |
| Qwen2.5-7B | `fresh_source` | `target_bridge` | `negative_shortcuts_neutralized` | `4` | `4` | `-1.936` |
| Qwen2.5-0.5B | `fresh_source` | `target_bridge` | `negative_shortcuts_neutralized` | `4` | `4` | `-1.072` |
| Qwen2.5-0.5B | `fresh_source` | `target_bridge` | `negative_opening_warmth_removed` | `4` | `4` | `-0.891` |

This validates the manual read from the dissent perturbation ladder: the repair
target is pseudo-side shortcut/warmth removal under target-side constructed
bridges.

## Residual Finding

The bridge residual is now targetable:

> Constructed target bridges can depend on pseudo-side warmth/shortcut surface
> structure when classifying fresh generated dissent examples. Removing or
> neutralizing that pseudo-side warmth exposes bridge brittleness in SmolLM2,
> Qwen2.5-0.5B, and Qwen2.5-7B, while TinyLlama passes with thin margins.

This finding complements the accountability ladder rather than duplicating it.
The current residual taxonomy is:

- `accountability_after_harm`: source+target generated-reference pocket.
- `dissent_after_mistake`: target-bridge pseudo-side shortcut/warmth
  brittleness.
- `belonging_norms`: withheld content/availability failure.

## Next Move

Run a targeted bridge repair rather than a broad generation sweep:

1. Add a small set of pseudo-side shortcut-neutralized bridge rows to the
   target/control side.
2. Rerun constructed bridge diagnostics on dissent perturbations.
3. Gate on fixing target-bridge fresh-source failures while preserving original
   source/control and fresh hand-control margins.

If that passes, the project has a paper-shaped residual taxonomy with targeted
repairs for at least two distinct activation failure modes.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support causal
steering, human behavioral, neural, clinical, deployment, or real-world
social-effect claims.
