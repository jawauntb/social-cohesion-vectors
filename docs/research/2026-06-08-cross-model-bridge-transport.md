# Cross-Model Bridge Transport Audit

Date: 2026-06-08

## Question

Do the intentional six-pair bridge directions transport between the accepted
Qwen7B and SmolLM2 activation spaces, rather than merely working inside one
model?

## Discovery-Regime Audit

Current regime:

- Artifact types: source-diverse generated benchmark pairs, expanded
  hand-authored procedural-justice control pairs, open-model activation NPZ
  files, same-model bridge direction reports, and cross-model alignment reports.
- Operations: construct six-pair bridge directions, align same prompts between
  model spaces with a ridge map, map constructed bridge directions, and evaluate
  mapped directions on the other model's full generated/control benchmarks.
- Gate: mapped directions must preserve sign, reach pairwise accuracy `1.000`,
  keep positive minimum margins on both full benchmarks, and also pass a
  leave-held-out variant where the alignment map excludes the fold's held-out
  pair group before evaluating that group.

## Artifacts

Qwen7B same-model bridge direction comparison:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/qwen7b_bridge_direction_comparison/qwen7b_generated_control_bridge_direction_comparison.md
```

Qwen7B-to-SmolLM2 bridge transport:

```text
/tmp/social_cohesion_out_of_family_repl_20260608/cross_model_bridge_transport/qwen7b_smol17_bridge_transport.md
```

## Qwen7B Same-Model Bridge Direction Comparison

Model/layer:

```text
Qwen/Qwen2.5-7B-Instruct layer -2
```

The constructed bridge-direction comparison passes on Qwen7B:

| Metric | Result |
| --- | ---: |
| readiness | `constructed_bridge_direction_ready` |
| constructed directions | 8 |
| constructed minimum joint cosine | +0.794 |
| constructed source minimum accuracy | 1.000 |
| constructed source minimum margin | +5.176 |
| constructed target minimum accuracy | 1.000 |
| constructed target minimum margin | +10.213 |
| failed constructed directions | 0 |

Unlike SmolLM2, the Qwen7B source-only and target-only directions already
transfer across the generated/control benchmarks. This makes Qwen7B a
strong same-model replication, but not the hard direction-transfer case.

## Qwen7B-to-SmolLM2 Bridge Transport

Models/layers:

```text
Qwen/Qwen2.5-7B-Instruct layer -2
HuggingFaceTB/SmolLM2-1.7B-Instruct layer -2
```

Alignment was fit over the same generated and control prompts:

| Metric | Result |
| --- | ---: |
| shared generated samples | 80 |
| shared control samples | 32 |
| combined shared samples | 112 |
| combined linear CKA | 0.908 |
| combined mutual kNN overlap | 0.732 |

Bidirectional mapped bridge transport passes:

| Direction | Min cosine | Source min margin | Target min margin | Leave-held-out min margin | Failed directions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen7B -> SmolLM2 | +1.000 | +11.091 | +12.044 | +11.091 | 0 |
| SmolLM2 -> Qwen7B | +1.000 | +5.176 | +10.213 | +5.877 | 0 |

The leave-held-out column is the stricter diagnostic: for each constructed
bridge fold, the model-to-model alignment map excludes that fold's held-out
pair group and the mapped direction is then evaluated on that held-out group.
Both directions preserve accuracy `1.000` and positive minimum margins under
that stricter check.

## Interpretation

Accepted:

- The six-pair bridge construction now works in Qwen7B and SmolLM2.
- Constructed bridge directions map bidirectionally between Qwen7B and SmolLM2.
- The mapped directions separate both generated and hand-authored control
  benchmarks in the target model space with zero failed directions.
- The stricter leave-held-out map diagnostic remains positive, so the pass is
  not only an all-prompts in-sample alignment artifact.

Caveated:

- This is still a same-prompt cross-model transport audit. It does not yet test
  fresh generated/control prompts that were absent from the alignment map.
- The cosine values are saturated after ridge mapping, so the robust evidence is
  the signed pairwise margins and leave-held-out group evaluation, not the
  absolute cosine magnitude.
- The SmolLM2 generated benchmark still has a separate metadata-held-out
  residual on `privacy_bypass::data_choice`; bridge transport narrows the
  failure mode but does not erase that audit record.

## Next Operation

Move from same-prompt cross-model transport to fresh-prompt bridge transport.
Add a small held-out generated/control prompt slice that preserves the same
procedural-justice paths, keep it out of the alignment-map training rows, and
evaluate whether Qwen7B/SmolLM2 mapped bridge directions preserve sign and
positive margins there.

## Claim Boundary

This run supports only a generated-text and hand-authored-text activation
diagnostic across two open-model activation spaces. It does not support human
behavioral, neural, clinical, deployment, or real-world social-effect claims.
