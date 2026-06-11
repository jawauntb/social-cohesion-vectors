# Residual Taxonomy and Bridge Sufficiency for Procedural-Justice Activation Directions

Status: NeurIPS-style working draft for collaborator review

Rendered PDF: `docs/papers/neurips_residual_taxonomy_bridge_sufficiency.pdf`

PDF renderer: `scripts/render_residual_taxonomy_pdf.py`

Date: 2026-06-11

## Abstract

Activation-space studies of social constructs risk confusing semantic
structure with surface shortcuts, generated-text artifacts, or fragile
selection effects. We study this problem in a procedural-justice benchmark
where positive examples preserve usable voice, refusal, evidence access,
appeal, non-retaliatory exit, and proportionate repair, while pseudo-cohesive
negative examples preserve warm or orderly language but tax those future
options. Across generated and hand-authored controls, four model spaces, and
constructed source/target bridge directions, we find that residual failures are
not monolithic. The strongest current evidence supports two distinct residual
classes. First, `accountability_after_harm` exposes a generated-reference
source pocket: small models invert one recovered generated reference while
clean hand-authored accountability controls pass, and strict local perturbation
augmentation repairs the slice with positive leave-one-perturbation-out margins
in all four model spaces. Second, `dissent_after_mistake` exposes a constructed
target-bridge geometry pocket: broad source+target directions already score
the generated reference correctly, but constructed target bridges fail when
pseudo-side warmth shortcuts are removed. This residual is repaired by
asymmetric target-bridge weighting and preserved across source, target,
fresh-source, and fresh-target evaluations. A strict accountability negative
control rejects the overgeneralized claim that weighted target bridges repair
all generated pockets. The result is a narrow diagnostic contribution: a
claim-bounded residual taxonomy and bridge-sufficiency protocol for
procedural-justice activation directions, not evidence of human social effects
or causal steering.

## 1. Introduction

The project asks whether language-model activations can represent distinctions
that matter for agency-preserving social cohesion. The target is deliberately
narrower than "make the model more prosocial." In the current benchmark, a
positive procedural-justice example must keep future options genuinely usable:
voice, refusal, evidence access, appeal, non-retaliatory exit, and repair. A
pseudo-cohesive negative example may sound warm, calm, or cooperative, but it
taxes those options through approval requirements, privacy loss, alignment
pressure, proof burdens, or delayed remedy.

The central problem is not only whether a direction separates labels. A
direction can separate a benchmark for the wrong reason: lexical cues, generated
style, source-family duplication, target-side shortcut phrases, or bridge-set
selection. Our research loop therefore moved from simple benchmark construction
to a stricter question:

> When activation directions fail after practical-availability, lexical,
> source-diversity, hand-authored control, and out-of-family gates, what kind of
> failure remains, and what kind of repair actually fixes it?

The answer so far is paper-shaped but still bounded. The failures we observed
are not one generic generated-text problem. They separate into at least two
residual classes with different repairs:

1. `accountability_after_harm`: a generated-reference source pocket repaired by
   local perturbation support.
2. `dissent_after_mistake`: a constructed target-bridge geometry pocket
   repaired by asymmetric target-bridge weighting.

The repairs are not interchangeable. Weighted target bridges fix the dissent
bridge residual and preserve the original evaluation slices, but the same move
does not repair the strict accountability residual in three of four model
spaces. That negative control is part of the result.

## 2. Contributions

This draft makes four contributions.

1. A claim-bounded procedural-justice activation benchmark pipeline that treats
   practical availability as a path-level gate rather than a mention-level
   property.
2. A residual taxonomy for activation failures that survive lexical,
   availability, hand-authored control, and model-family checks.
3. Two accepted repairs matched to distinct residual classes:
   perturbation-ladder augmentation for a generated-reference source pocket,
   and asymmetric target-bridge weighting for a constructed target-bridge
   geometry pocket.
4. Rejected alternatives and negative controls showing that content-only
   target repair is insufficient for the dissent bridge residual, and weighted
   target bridges are not a universal generated-pocket repair.

## 3. Benchmark And Gates

The benchmark is organized around paired positive and pseudo-cohesive examples.
The positive side must make future options usable now. The pseudo side must
retain procedural taxes even when it avoids obvious villain language.

The main gates are:

| Gate | Purpose |
| --- | --- |
| Practical availability | Verify that future-option paths are usable in positive examples and taxed in pseudo examples. |
| Lexical diagnostics | Detect whether labels can be solved by shallow terms, lengths, or label-aligned phrases. |
| Source diversity | Prevent duplicated generated phrasings from masquerading as semantic coverage. |
| Hand-authored controls | Test whether the procedural distinction survives outside generated text. |
| Out-of-family replication | Test whether the result survives additional model spaces. |
| Constructed bridge diagnostics | Train directions from limited source/target bridge subsets and evaluate transfer to held-out fresh slices. |
| Preservation and negative controls | Confirm a repair does not damage earlier slices and does not overgeneralize to unrelated residuals. |

All activation diagnostics in this draft use layer `-2` reports for four model
spaces:

- SmolLM2-1.7B
- Qwen2.5-0.5B
- Qwen2.5-7B
- TinyLlama-1.1B

Generated/model artifacts are kept out of git. Durable summaries are committed
under `docs/research/`.

## 4. Residual 1: Generated-Reference Source Pocket

The hardest accountability residual was the recovered generated
`accountability_after_harm` reference. The broad source+target direction
inverted this reference in SmolLM2, Qwen2.5-0.5B, and TinyLlama, while
Qwen2.5-7B remained positive. A source-style intervention narrowed the issue:
clean hand-authored accountability variants passed, including a generated-like
paragraph, so the residual was not broad accountability content failure.

A strict perturbation ladder split the reference into twelve deterministic
variants, including opening-frame edits, neutral replacements, length controls,
explicit refusal, pseudo-side condition edits, and shortcut neutralization.
Scoped practical availability remained intact over all `72/72` tested paths.
Lexical diagnostics stayed caveated because the artifact was intentionally the
known leaky generated reference, so this experiment is interpreted as a local
mechanism stress test rather than a clean benchmark.

The accepted repair was strict perturbation augmentation:

| Model | Full fresh-source min | Fresh LOO min | Full source min | Full target min |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `+54.525` | `+47.185` | `+20.317` | `+52.012` |
| Qwen2.5-0.5B layer `-2` | `+2.228` | `+1.376` | `+1.512` | `+1.549` |
| Qwen2.5-7B layer `-2` | `+25.406` | `+23.059` | `+5.052` | `+22.233` |
| TinyLlama-1.1B layer `-2` | `+1.029` | `+0.800` | `+0.593` | `+1.194` |

This supports a narrow interpretation: small model directions can encode the
accountability distinction, but one generated reference lies in a local pocket
that requires nearby support. It does not support a broad claim about human
accountability perception or real-world repair.

## 5. Residual 2: Constructed Target-Bridge Geometry Pocket

The `dissent_after_mistake` residual behaved differently. The unchanged
generated dissent reference was already positive under broad source+target
directions in all four model spaces. Positive-side path and neutral replacement
edits strengthened rather than repaired the result. The failure appeared only
when constructed bridge directions were trained from limited bridge subsets and
evaluated on fresh perturbations.

The bridge-stability summary localized the initial failure:

| Metric | Result |
| --- | ---: |
| Constructed failure rows | `39` |
| Most failed bridge family | `target_bridge` |
| Most failed perturbation | `negative_shortcuts_neutralized` |
| Worst margin | `-18.296` |

The first repair added shortcut-neutralized target/control content. It passed
the scoped text gate: `20/20` repaired availability paths and minimum margin
`+0.640`. But the default six-pair constructed bridge gate did not improve:
constructed failures moved from `39` to `40`, and all failures remained
target-bridge failures. Increasing target bridge count to `15` helped but did
not finish the repair: failures dropped to `13`, Qwen2.5-7B and TinyLlama
passed, and SmolLM2 plus Qwen2.5-0.5B still failed.

The accepted repair was asymmetric weighted target-bridge construction:

| Parameter | Value |
| --- | ---: |
| Target bridge pair count | `9` |
| Target bridge primary repetitions | `1` |
| Target bridge secondary repetitions | `3` |
| Source bridge primary repetitions | `1` |
| Source bridge secondary repetitions | `1` |

This repair cleared the fresh constructed bridge gate:

| Model | Fresh-source min | Fresh-source failures | Fresh-target min | Fresh-target failures |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `+3.151` | `0` | `+55.796` | `0` |
| Qwen2.5-0.5B layer `-2` | `+0.220` | `0` | `+2.026` | `0` |
| Qwen2.5-7B layer `-2` | `+6.866` | `0` | `+19.395` | `0` |
| TinyLlama-1.1B layer `-2` | `+0.513` | `0` | `+1.908` | `0` |

## 6. Preservation And Negative Control

A repair is only useful if it preserves the earlier distinctions. The weighted
dissent bridge repair passed preservation across source, target, fresh-source,
and fresh-target slices:

| Metric | Result |
| --- | ---: |
| Models | `4` |
| Constructed direction rows | `32` |
| Evaluation rows | `16` |
| Failed pair evaluations | `0` |
| Worst margin | `+0.019` |
| Worst evaluation | `source` |

The same weighted target-bridge setting was then tested against the strict
accountability residual as a negative control. It improved some margins but did
not repair the class:

| Model | Default count-9 fresh-source min / failures | Weighted count-9 target x3 fresh-source min / failures |
| --- | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `-20.909` / `41` | `-20.909` / `20` |
| Qwen2.5-0.5B layer `-2` | `-2.229` / `50` | `-1.206` / `43` |
| Qwen2.5-7B layer `-2` | `-1.977` / `12` | `+1.013` / `0` |
| TinyLlama-1.1B layer `-2` | `-1.062` / `88` | `-0.818` / `84` |

At the summary level, default count-9 constructed bridges had `192` failure
rows. Weighted target bridges reduced this to `147`, but three model spaces
still failed. This is the main reason the paper should claim residual taxonomy,
not a universal repair recipe.

## 7. Discussion

The useful scientific object here is not a single "cohesion vector." It is a
typed repair ledger: what failed, which gate caught it, which repair was
accepted, which plausible repair failed, and which claim boundary remains.

The accountability result says local generated references can create
off-manifold source pockets. The dissent result says limited bridge
construction can create target-side geometry pockets even when the broad
direction already scores the generated reference correctly. The negative
control says these pockets are not interchangeable.

This matters for activation-based social-construct work because a benchmark
score alone would hide all three facts. A system could appear successful on a
generated benchmark while failing clean controls, or appear broken on a
generated residual while still encoding the intended procedural distinction
under hand-authored controls. Conversely, adding more positive content is not
always enough; the bridge geometry itself can be the failure surface.

## 8. Limitations

The current evidence remains limited.

- The experiments are text-benchmark activation diagnostics, not human studies.
- The results are layer `-2` diagnostics over four model spaces; they are not a
  full architecture or layer sweep.
- The benchmark is about procedural-justice distinctions, not social cohesion
  as a complete construct.
- The accountability perturbation ladder is a local mechanism stress test over
  a known leaky generated reference, not a clean benchmark by itself.
- The current repairs are diagnostic repairs. They do not establish causal
  steering, deployment safety, clinical outcomes, neural correlates, or
  real-world social effects.
- The draft still needs a formal bibliography, clean figures, and a
  reproducibility manifest before it can be treated as submission-ready.

## 9. Reproducibility Ledger

Durable research notes:

```text
docs/research/2026-06-08-literature-foundation-audit.md
docs/research/2026-06-08-procedural-justice-control-expansion-run.md
docs/research/2026-06-08-out-of-family-replication-run.md
docs/research/2026-06-09-strict-accountability-perturbation-ladder.md
docs/research/2026-06-09-dissent-perturbation-ladder.md
docs/research/2026-06-11-target-bridge-shortcut-repair.md
docs/research/2026-06-11-weighted-target-bridge-repair.md
docs/research/2026-06-11-weighted-bridge-preservation-audit.md
docs/research/2026-06-11-residual-taxonomy-repair-comparison.md
```

Live artifacts used for the latest tables:

```text
/tmp/social_cohesion_accountability_strict_perturbation_ladder_20260609/
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/
/tmp/social_cohesion_bridge_weight_audit_20260611/
/tmp/social_cohesion_weighted_bridge_preservation_20260611/
```

Generated text and activation payloads intentionally remain outside git.

## 10. Next Experiments

The next work should improve publication readiness rather than broadening
search blindly.

1. Add a regeneration manifest that maps every table in this draft to committed
   scripts and expected external artifact paths.
2. Run one narrow robustness check over either another withheld residual with
   fixed practical availability, or a small layer sweep for the accepted
   weighted dissent repair.
3. Convert this draft into a formal paper with related work, figures, and a
   reproducible appendix.

## Claim Boundary

This draft does not claim human behavioral effects, neural alignment, clinical
relevance, deployment readiness, or causal steering. It claims only a
generated/hand-authored text-benchmark activation diagnostic: two residual
classes, two matched repairs, preservation for the accepted dissent repair, and
a negative control blocking universal repair claims.
