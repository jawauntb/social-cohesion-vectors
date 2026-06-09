# Activation-Manifest Recovery Fresh-Bridge Diagnostic

Date: 2026-06-09

## Question

Was the fresh-generated bridge-transport failure caused by cross-model mapping,
or was it already present inside the destination model's own activation space?

## Discovery-Regime Audit

Current regime:

- Artifact types: activation NPZ payloads, recovered `PairwiseExample`
  manifests, generated source-family prompts, hand-authored
  procedural-justice controls, fresh generated repair-v2 prompts, constructed
  bridge directions, and failed-pair margin tables.
- Operations: recover pair manifests from activation payload text/labels/scores
  plus stable taxonomy/control ids; train source-only, fresh-source-only,
  source+fresh-source, and constructed bridge directions inside one model space;
  evaluate on original generated source, procedural-control target, fresh
  generated source, and fresh procedural-control target.
- Gates: recovered manifests must preserve all activation pair ids and expose
  `source`, `primary_fault_class`, and `slack_options_tested`; constructed
  bridge directions must reach pairwise accuracy `1.000` and positive minimum
  margins on both fresh generated and fresh procedural-control prompt slices.
- Known limitation: recovered manifests are not original generation logs. Text,
  labels, and target scores come from activation NPZ files; taxonomy and source
  metadata are inferred from stable pair ids.

Action class:

- Diagnostic discovery inside the activation regime. The new artifact type is a
  provenance-marked recovered manifest that can unblock activation diagnostics
  when `/tmp` generation outputs have been cleaned.

## Code Added

New recovery module:

```text
src/social_cohesion_vectors/experiments/activation_pair_manifest_recovery.py
```

New CLI:

```text
scripts/recover_activation_pair_manifest.py
```

The recovered pair metadata includes `metadata_recovery` and an explicit warning
that the original generation manifest was unavailable.

## Recovered Manifests

Artifacts were written outside git under:

```text
/tmp/social_cohesion_manifest_recovery_20260609/
```

Recovered coverage:

| Dataset | Pairs | Sources | Notes |
| --- | ---: | ---: | --- |
| `qwen7b_replication_four_source_v1` | 40 | 4 | generated four-source benchmark |
| `procedural_justice_control_v2` | 16 | 4 | expanded hand-authored control |
| `fresh_generated_repair_v2_limit20` | 10 | 1 | fresh repair-v2 tournament slice |
| `procedural_justice_control_v1` | 8 | 2 | fresh hand-authored control slice |

## Live Diagnostic Results

Reports:

```text
/tmp/social_cohesion_manifest_recovery_20260609/fresh_generated_bridge_diagnostic/qwen7b.json
/tmp/social_cohesion_manifest_recovery_20260609/fresh_generated_bridge_diagnostic/smol17.json
```

Summary:

| Model space | Ready | Constructed fresh generated | Constructed fresh control | Failed-pair pattern |
| --- | --- | ---: | ---: | --- |
| Qwen2.5-7B layer `-2` | yes | `1.000` / `+0.178` | `1.000` / `+10.213` | constructed bridges have zero fresh failures |
| SmolLM2-1.7B layer `-2` | no | `0.700` / `-11.273` | `1.000` / `+23.644` | constructed bridges fail fresh generated only |

The earlier cross-model fresh-transport asymmetry is now explained more
precisely. Qwen's same-model constructed bridges pass the fresh generated slice
with a thin minimum margin `+0.178`. SmolLM2's same-model constructed bridges
fail the same slice with minimum margin `-11.273`. Those match the prior
cross-model transport margins, which means the failure is not primarily an
alignment-map artifact. It is a destination-model fresh-generated geometry
failure: SmolLM2 separates the original generated/control benchmarks and the
fresh hand-authored controls, but not the repaired generated tournament prompts
under constructed bridge directions.

SmolLM2 constructed-bridge fresh-generated failures concentrate on:

- `accountability_after_harm`
- `belonging_norms`
- `dissent_after_mistake`

Baseline rows add two useful controls:

- `source_only` fails fresh generated prompts in both models, so broad source
  training alone is insufficient.
- `fresh_source_only` separates fresh generated prompts but fails multiple
  original generated source-family pairs, so the fresh slice is not a drop-in
  replacement for the four-source benchmark.

## Residual Finding

The residual is now narrower than "fresh generated bridge transport fails." The
new residual is:

> Repaired generated prompts define a fresh-source subdirection that is only
> weakly aligned with the original generated/procedural bridge direction in
> Qwen and is actively non-robust in SmolLM2, while fresh hand-authored
> procedural controls remain robust in both spaces.

This is a meaningful activation-benchmark finding, not yet a causal steering,
human behavioral, or neural finding.

## Next Move

Attack the three-pair SmolLM2 residual directly:

1. Build a fresh-generated residual diagnostic that compares failed and passing
   fresh repair-v2 pairs against original four-source pairs by
   `primary_fault_class`, option path, length, and lexical/procedural cue
   margins.
2. Test whether adding the fresh-source tournament slice as an explicit bridge
   stratum repairs SmolLM2 without breaking original generated source-family
   margins.
3. If the repaired direction is stable, run a second out-of-family model
   (`Llama-3.2-3B-Instruct` or another locally extractable instruct model)
   before drafting strong paper claims.

## Claim Boundary

This supports a text-benchmark activation diagnostic over generated and
hand-authored prompts. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
