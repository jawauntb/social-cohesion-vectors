# Residual Taxonomy Repair Comparison

Date: 2026-06-11

## Question

What is the smallest honest paper claim supported by the completed
availability/control/bridge repair loop?

## Discovery-Regime Audit

Current regime:

- Artifact types: generated benchmark bundles, hand-authored procedural-justice
  controls, practical-availability reports, lexical reports, Modal activation
  payloads, perturbation ladders, constructed bridge diagnostics, preservation
  summaries, and dated research notes.
- Operations: compare source+target directions, local perturbation
  augmentation, constructed source/target bridge directions, bridge-pair count
  ablations, asymmetric bridge weighting, preservation checks, and
  negative-control reruns.
- Gates: accepted repairs must clear the intended residual across SmolLM2,
  Qwen2.5-0.5B, Qwen2.5-7B, and TinyLlama at layer `-2`, and must preserve
  source/control or fresh-control slices relevant to the claim. Rejected repairs
  remain in the ledger.
- Known limitations: all evidence is generated or hand-authored text-benchmark
  activation evidence. No causal steering, human behavioral, neural, clinical,
  deployment, or real-world social-effect claim is supported.

Action class:

- Consolidation. This note does not add a new experiment; it turns the
  accepted and rejected repairs into a paper-ready residual taxonomy.

## Claim Candidate

The strongest current claim is a narrow residual taxonomy:

> Procedural-justice activation directions fail in at least two distinguishable
> ways under generated and hand-authored text controls: a generated-reference
> source pocket that is repaired by local perturbation support, and a
> constructed target-bridge geometry pocket that is repaired by asymmetric
> target-bridge weighting. The repairs do not transfer universally across the
> two residual classes.

This is stronger than a single benchmark score because it includes:

- a positive repair for `accountability_after_harm`;
- a positive repair for `dissent_after_mistake`;
- a rejected content-only target repair;
- a rejected overgeneralized bridge-weighting repair;
- a preservation check for the accepted weighted dissent repair.

## Side-By-Side Repair Table

| Residual | Failure before repair | Accepted repair | Gate cleared | Rejected or partial repairs | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `accountability_after_harm` generated reference | SmolLM2, Qwen2.5-0.5B, and TinyLlama invert the recovered generated reference while clean hand-authored accountability controls pass. | Strict local perturbation augmentation over twelve deterministic variants. | Full augmentation and leave-one-perturbation-out pass in all four model spaces: SmolLM2 fresh LOO `+47.185`, Qwen2.5-0.5B `+1.376`, Qwen2.5-7B `+23.059`, TinyLlama `+0.800`. | Weighted target bridges are not a universal repair: default count-9 constructed bridges have `192` failures; weighted target bridges still have `147` failures and repair only Qwen2.5-7B. | A generated-reference/source-pocket residual, not broad inability to encode accountability or procedure. |
| `dissent_after_mistake` constructed target bridge | Broad source+target directions already score the generated dissent reference positive, but constructed target bridges fail fresh perturbations, especially pseudo-side warmth/shortcut neutralization. | Count-9 constructed target bridges with target bridge repetitions `1:3` and source repetitions `1:1`. | All four model spaces reach `fresh_generated_bridge_ready`; the bridge-stability summary has `0` constructed failure rows. Preservation passes across source, target, fresh-source, and fresh-target slices: `32` constructed rows, `0` failed pair evaluations, worst margin `+0.019`. | Content-only shortcut-neutralized target/control rows pass scoped text gates but do not repair six-pair bridge transfer: failures move `39 -> 40`. Increasing target bridge count to `15` is partial: failures drop to `13`, Qwen2.5-7B and TinyLlama pass, SmolLM2 and Qwen2.5-0.5B fail. | A constructed target-bridge geometry pocket, not the same generated-reference pocket as accountability. |
| `belonging_norms` and other withheld residuals | Earlier availability failures or incomplete control evidence. | None accepted. | Withheld from activation claims. | Do not use for publication claims yet. | Useful future stress tests, not part of the current positive result. |

## Dissent Repair Detail

The accepted weighted bridge setting is:

| Parameter | Value |
| --- | ---: |
| Target bridge pair count | `9` |
| Target bridge primary repetitions | `1` |
| Target bridge secondary repetitions | `3` |
| Source bridge primary repetitions | `1` |
| Source bridge secondary repetitions | `1` |

Fresh generated bridge diagnostics:

| Model | Fresh-source min | Fresh-source failures | Fresh-target min | Fresh-target failures |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `+3.151` | `0` | `+55.796` | `0` |
| Qwen2.5-0.5B layer `-2` | `+0.220` | `0` | `+2.026` | `0` |
| Qwen2.5-7B layer `-2` | `+6.866` | `0` | `+19.395` | `0` |
| TinyLlama-1.1B layer `-2` | `+0.513` | `0` | `+1.908` | `0` |

Preservation audit:

| Metric | Result |
| --- | ---: |
| Models | `4` |
| Constructed direction rows | `32` |
| Evaluation rows | `16` |
| Failed pair evaluations | `0` |
| Worst preservation margin | `+0.019` |
| Worst evaluation | `source` |

## Accountability Negative Control

The same weighted target-bridge move does not repair the strict
`accountability_after_harm` residual in three of four model spaces.

| Model | Default count-9 fresh-source min / failures | Weighted count-9 target x3 fresh-source min / failures |
| --- | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `-20.909` / `41` | `-20.909` / `20` |
| Qwen2.5-0.5B layer `-2` | `-2.229` / `50` | `-1.206` / `43` |
| Qwen2.5-7B layer `-2` | `-1.977` / `12` | `+1.013` / `0` |
| TinyLlama-1.1B layer `-2` | `-1.062` / `88` | `-0.818` / `84` |

This is the main guard against overclaiming the weighted bridge result. It
shows that the dissent repair is not a universal generated-pocket repair.

## Paper Readiness

Ready to share as a working paper scaffold:

- a crisp phenomenon: residuals survive practical-availability, lexical,
  control, and transport gates;
- two positive repairs with different mechanisms;
- one preservation audit;
- one negative control that blocks overgeneralization;
- explicit claim boundaries.

Not ready for a final NeurIPS submission without more work:

- broader residual replication beyond two named residual classes;
- cleaner bibliography and formal related-work positioning;
- full reproducibility manifest from committed scripts to regenerated reports;
- optional larger-model or layer sweep replication;
- steering-side experiments, if the paper wants to claim controllability rather
  than diagnostic separability.

## Live Artifact Pointers

Generated and activation artifacts remain outside git:

```text
/tmp/social_cohesion_accountability_strict_perturbation_ladder_20260609/
/tmp/social_cohesion_dissent_perturbation_ladder_20260609/
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/
/tmp/social_cohesion_bridge_weight_audit_20260611/
/tmp/social_cohesion_weighted_bridge_preservation_20260611/
```

Durable notes:

```text
docs/research/2026-06-09-strict-accountability-perturbation-ladder.md
docs/research/2026-06-09-dissent-perturbation-ladder.md
docs/research/2026-06-11-target-bridge-shortcut-repair.md
docs/research/2026-06-11-weighted-target-bridge-repair.md
docs/research/2026-06-11-weighted-bridge-preservation-audit.md
```

## Claim Boundary

This comparison is about generated/hand-authored text benchmark construction
and layer `-2` activation diagnostics in four language-model spaces. It does
not establish human behavioral, neural, clinical, deployment, causal steering,
or real-world social-effect claims.
