# Fresh-Prompt Bridge Transport Audit

Date: 2026-06-08

## Question

Do the accepted Qwen7B/SmolLM2 constructed bridge directions still transport
when evaluated on fresh generated and hand-authored prompts that were absent
from the alignment-map training rows?

## Discovery-Regime Audit

Current regime:

- Artifact types: source-diverse generated benchmark pairs, expanded
  hand-authored procedural-justice control pairs, fresh generated repair-v2
  pairs, fresh hand-authored procedural-justice control-v1 pairs, open-model
  activation NPZ files, constructed bridge directions, and cross-model
  alignment maps.
- Operations: extract Qwen7B and SmolLM2 activations, fit cross-model ridge
  maps on the accepted same-prompt generated/control rows, map constructed
  bridge directions into the other model space, and evaluate the mapped
  directions on fresh source and target prompt slices.
- Gate: mapped directions must preserve pairwise accuracy `1.000` and positive
  minimum margins on the original source, original target, leave-held-out map
  fold, fresh generated source, and fresh hand-authored control target.

Action class:

- Search inside the current activation diagnostic regime. The run adds fresh
  held-out evaluation rows and failure tables, but it does not introduce a new
  artifact type or support a human-facing claim.

## Artifacts

Fresh generated repair-v2 activation slice:

```text
/tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_000_001_repair_v2_000_limit20/
```

Fresh hand-authored procedural-justice control-v1 activation slice:

```text
/tmp/social_cohesion_procedural_justice_control_20260608/control_v1/
```

Fresh-prompt bridge transport report from the live run. Like the other
generated artifacts, this path is intentionally outside git and may not persist
between sessions:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/fresh_prompt_bridge_transport/qwen7b_smol17_fresh_prompt_bridge_transport.md
```

## Activation Extraction

The missing SmolLM2 activations for `procedural_justice_control_v1` were
extracted at layer `-2`. The existing Qwen7B control-v1 activations were
reused.

| Dataset | Model | Layer | Pair-LOO accuracy | Pair-LOO margin |
| --- | --- | ---: | ---: | ---: |
| `procedural_justice_control_v1` | Qwen2.5-7B | -2 | 1.000 | +34.684 |
| `procedural_justice_control_v1` | SmolLM2-1.7B | -2 | 1.000 | +98.185 |
| `fresh_generated_repair_v2_limit20` | Qwen2.5-7B | -2 | 0.800 | +18.297 |
| `fresh_generated_repair_v2_limit20` | SmolLM2-1.7B | -2 | 0.900 | +46.621 |

The fresh generated repair-v2 slice is therefore a real stress test: it is not
perfect under standalone leave-one-pair-out activation evaluation.

## Result

Readiness:

```text
not_ready
```

Baseline same-prompt transport remains accepted:

| Direction | Source min margin | Target min margin | Leave-held-out min margin |
| --- | ---: | ---: | ---: |
| Qwen7B -> SmolLM2 | +11.091 | +12.044 | +11.091 |
| SmolLM2 -> Qwen7B | +5.176 | +10.213 | +5.877 |

Fresh target control transport passes in both directions:

| Direction | Fresh target accuracy | Fresh target min margin |
| --- | ---: | ---: |
| Qwen7B -> SmolLM2 | 1.000 | +23.644 |
| SmolLM2 -> Qwen7B | 1.000 | +10.213 |

Fresh generated source transport is asymmetric:

| Direction | Fresh source accuracy | Fresh source min margin | Failed pair evaluations |
| --- | ---: | ---: | ---: |
| Qwen7B -> SmolLM2 | 0.700 | -11.273 | 18 |
| SmolLM2 -> Qwen7B | 1.000 | +0.178 | 0 |

The failed rows are all `model_a_to_b` fresh-source evaluations. They
concentrate in three fresh generated pairs:

| Pair | Failed folds | Worst margin |
| --- | ---: | ---: |
| `accountability_after_harm` | 8 | -9.459 |
| `belonging_norms` | 6 | -11.273 |
| `dissent_after_mistake` | 4 | -6.944 |

## Interpretation

Accepted:

- The cross-model bridge transport implementation now supports optional fresh
  source and fresh target evaluation slices.
- The original same-prompt source, target, and leave-held-out map checks still
  pass bidirectionally.
- Fresh hand-authored procedural-justice control prompts pass in both transport
  directions.
- SmolLM2-to-Qwen7B transport also passes on the fresh generated slice, though
  its minimum margin is thin at `+0.178`.

Rejected or withheld:

- Fresh-prompt bridge transport is not ready. Qwen7B-to-SmolLM2 mapped bridge
  directions fail on the fresh generated repair-v2 source slice.
- The failure is not a general alignment collapse, because the same maps and
  directions still pass original rows, leave-held-out rows, and fresh control
  rows.
- The failure is not merely "fresh prompts are impossible," because the
  opposite direction separates the same fresh generated pairs with accuracy
  `1.000`.

## Residual Content

The residual is now a fresh-generated source-regime mismatch under
Qwen7B-to-SmolLM2 direction transport. The failing pairs are concentrated in
`accountability_after_harm`, `belonging_norms`, and `dissent_after_mistake`.
This suggests the next loop should diagnose whether those fresh generated
repair-v2 prompts:

- encode the procedural-justice distinction differently from the four-source
  generated baseline;
- require fresh generated rows in the bridge/alignment support set;
- expose direction-map asymmetry between Qwen7B and SmolLM2; or
- still contain prompt-side lexical, length, or repair-path artifacts.

## Next Operation

Run a failure-focused bridge diagnostic before adding another model family:

- compare source-only, fresh-source-only, joint source+fresh-source, and
  constructed-bridge directions within Qwen7B and SmolLM2;
- keep the fresh control-v1 slice as the negative control that should remain
  transport-ready;
- focus per-pair tables on `accountability_after_harm`, `belonging_norms`, and
  `dissent_after_mistake`;
- decide whether the next fix is adding fresh generated rows to the bridge set,
  creating a fresh-generated bridge constructor, or repairing the fresh
  generated prompt slice itself.

## Claim Boundary

This run is a generated-text and hand-authored-text activation diagnostic over
two open-model activation spaces. It does not support human behavioral, neural,
clinical, deployment, or real-world social-effect claims.
