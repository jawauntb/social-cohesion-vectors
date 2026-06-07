---
title: 2026-06-03 Transition Record Schema
status: draft
date: 2026-06-03
origin: CK-4 parallel-lane bridge from cocktail reports to perturbation records
---

# 2026-06-03 Transition Record Schema

This schema converts CK-3/CK-4 cocktail report records and toy substrate sweep
reports into lane-agnostic perturbation transition records. It is deliberately
lightweight: each transition pairs a non-baseline cocktail generation with the
prompt-matched baseline generation, or a non-baseline toy sweep result with the
coefficient-1.0 toy baseline, then records the observed movement.

The record is compute-only. It does not claim biological, pharmacological,
neural, human, therapeutic, or receptor-level effects.

## Regime Transition Records

Regime-transition records are adjacent to, but separate from, CK perturbation
records. They describe changes in the scientific contract itself: benchmark
upgrades, scorer components, audit gates, new artifact types, and claim
boundaries. They are intended for the self-revising discovery layer where old
artifacts are preserved, new verifiers are introduced, rejected alternatives
remain visible, and residual content records what the new regime exposed beyond
transporting old evidence.

Regime records use `src/social_cohesion_vectors/regime_records.py` and contain:

- `record_id`: stable deterministic id for the regime change.
- `title`: short human-readable transition title.
- `status`: `proposed`, `accepted`, `rejected`, or `superseded`.
- `source_id`: plan, report, or artifact path that motivated the record.
- `claim_boundary`: compute-only scope statement. By default, regime records do
  not claim human, neural, clinical, deployment, or real-world social effects.
- `old_regime` and `new_regime`: JSON objects describing the old and new
  scientific schema, benchmark contract, scorer components, or audit posture.
- `preserved_artifacts`: old artifacts carried forward into the new regime.
- `new_artifact_types`: artifact types that the new regime can now express.
- `new_verifiers`: gates or checks introduced by the new regime.
- `rejected_alternatives`: explicit alternatives rejected by the transition.
- `gates`: gate records with `gate_id`, `status`, optional metric/threshold,
  observed value, and note.
- `residual_content`: findings not explained by simply transporting old
  artifacts into the new regime. Legacy input named `residual_findings` is still
  accepted and normalized to `residual_content`.
- `notes`: optional additional provenance notes.

Use:

```bash
uv run python scripts/export_regime_transition_records.py \
  docs/research/regime_records.json \
  --output data/reports/regime_transition_records.jsonl \
  --markdown-output data/reports/regime_transition_records.md
```

The Markdown report lists gates, rejected alternatives, residual content, and
the claim boundary for each transition so benchmark/scorer/audit changes do not
silently become human or neural validation claims.

## Fields

- `transition_id`: deterministic `{prompt_id}::{baseline}->{perturbation}` id.
- `baseline_state`: prompt, baseline recipe, generated text, CK-1 score, and
  score components before perturbation.
- `perturbation`: recipe id, label, and component list.
- `dose`: component count, per-component strengths, and absolute strength sum.
- `site`: component layer, hook site, and steering position.
- `timing`: component steering timing and schedule, including CK-4 schedules
  such as `first-N`, `after-N`, `decay-N`, and `ramp-A-B` when present.
- `effect_class`: one of `beneficial_transition`, `mixed_transition`,
  `adverse_transition`, or `neutral_transition`, based on CK-1 score delta and
  pseudo-attunement-risk delta.
- `observed_transition`: from/to recipe ids, CK-1 score delta,
  safe-attunement-signal delta, pseudo-attunement-risk delta, and paired output
  text.
- `side_effects`: pseudo-attunement-risk level, pseudo-risk delta, and whether
  pseudo-risk increased.
- `antagonist`: inferred guardrail/control component ids, with an explicit note
  that these are not biological receptor antagonists.
- `washout`: marked `not_measured` because cocktail reports are single-pass
  generation assays.
- `replication_context`: source report metadata plus prompt, recipe, model, seed,
  generation parameters, and component steering schedules when present.

CK-4 scheduled components preserve `steering_schedule` in both
`perturbation.components` and `timing`. The replication context also carries
`component_steering_schedules` so provenance summaries retain scheduled
components even when consumers do not inspect the full component payload.

## Toy Substrate Records

Toy substrate reports with `results` are supported without changing the
Drosophila/toy-substrate exporter. Results are grouped by graph, transmitter,
and source-class first; within each group, `coefficient == 1.0` is treated as
the baseline, and every other result becomes a transition record.

Toy records reuse the same top-level fields:

- `baseline_state`: graph id, transmitter mechanism, and baseline toy metrics.
- `perturbation`: edge-scaling intervention id, coefficient, transmitter,
  source class, and scaled-edge count.
- `dose`: one component with coefficient and distance from vehicle
  (`abs(coefficient - 1.0)`) as `absolute_strength_sum`.
- `site`: graph id, target nodes, and scaled-edge count.
- `timing`: constant `simulation_step` timing for the static sweep.
- `effect_class`: `beneficial_transition` when target movement occurs without
  off-target movement, washout residue, or instability; `mixed_transition` when
  target movement co-occurs with off-target movement or residue;
  `adverse_transition` for instability or off-target/residue without target
  movement; otherwise `neutral_transition`.
- `side_effects`: off-target movement and instability status.
- `washout`: `recovered` when the toy washout value is zero, otherwise
  `residual`.

Toy records remain compute-only and do not claim real Drosophila biology,
behavior, pharmacology, neural effects, or social effects.

## Summaries

`summarize_transition_records(records)` returns three count maps:

- `effect_class`: count by transition effect class.
- `side_effect_status`: `observed` versus `none`.
- `washout_status`: count by washout status such as `not_measured`,
  `recovered`, or `residual`.

## Export

Use:

```bash
uv run python scripts/export_ck3_transition_records.py \
  /tmp/ck3_cocktail/ck3_cocktail_report.json \
  --output /tmp/ck3_cocktail/ck3_transition_records.jsonl
```

The exporter also accepts generation JSONL or a JSON list of records. Records
without a prompt-matched baseline are skipped so the output remains a true
baseline-to-perturbation transition set.

The same exporter auto-detects toy substrate JSON reports that contain
`results`, writes transition JSONL, and prints effect-class, side-effect, and
washout summary counts.
