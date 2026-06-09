# Fresh-Generated Residual Diagnostic

Date: 2026-06-09

## Question

Are the SmolLM2 fresh repair-v2 failures content failures, lexical artifacts,
or genuine activation-geometry residuals relative to the original four-source
benchmark?

## Discovery-Regime Audit

Current regime:

- Artifact types: recovered generated source pair manifests, recovered fresh
  repair-v2 pair manifests, Qwen7B and SmolLM2 fresh bridge reports,
  constructed bridge margins, practical-availability rows, lexical cue margins,
  and source same-base activation margins.
- Operations: compare each fresh generated pair's constructed-bridge margins
  against the same base contrast in the original four-source generated
  benchmark; add practical-availability, length, and fixed lexical cue features.
- Gate: a fresh failure is an activation residual only if same-base original
  source pairs still have positive constructed margins and the fresh pair's
  practical-availability audit is positive. Content failures are withheld from
  activation-residual claims.

Action class:

- Diagnostic search inside the activation regime. This narrows a known failure
  surface; it does not create a causal or human-facing claim.

## Code Added

New diagnostic module:

```text
src/social_cohesion_vectors/experiments/fresh_generated_residual_diagnostic.py
```

New CLI:

```text
scripts/run_fresh_generated_residual_diagnostic.py
```

The report ranks fresh generated pairs by constructed-direction failure count,
then records fresh margin, reference-model fresh margin, same-base source
margin, availability margin, lexical cue margin, and length delta.

## Live Results

Reports were written outside git under:

```text
/tmp/social_cohesion_manifest_recovery_20260609/fresh_generated_residual_diagnostic/
```

Summary:

| Model | Status | Failing fresh pairs | Min fresh margin |
| --- | --- | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `fresh_generated_residual_present` | 3/10 | `-11.273` |
| Qwen2.5-7B layer `-2` | `no_residual_detected` | 0/10 | `+0.178` |

SmolLM2 residual rows:

| Pair | Failure count | Smol min | Qwen min | Same-base source min | Availability min | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `accountability_after_harm` | 8/8 | `-9.459` | `+0.178` | `+19.384` | `+0.150` | fresh-only activation failure |
| `belonging_norms` | 6/8 | `-11.273` | `+1.972` | `+36.016` | `-0.150` | activation failure with content/availability failure |
| `dissent_after_mistake` | 4/8 | `-6.944` | `+5.701` | `+35.314` | `+0.310` | fresh-only activation failure |

## Residual Finding

The SmolLM2 residual is mixed:

- `accountability_after_harm` and `dissent_after_mistake` are clean activation
  residuals under this diagnostic. Their fresh prompts fail SmolLM2 constructed
  bridges, but Qwen separates them, same-base original source pairs separate
  strongly in SmolLM2, and practical availability remains positive.
- `belonging_norms` is not a clean activation-only residual because the fresh
  pair's practical-availability margin is negative. It should be repaired or
  withheld before using it as evidence about activation geometry.
- Lexical cues do not explain the residual as a simple fixed-lexicon shortcut:
  `dissent_after_mistake` has lexical cue margin `0.000`, and
  `accountability_after_harm` fails despite a positive cue margin favoring the
  genuine label.

## Next Move

Run a SmolLM2 fresh-augmented direction audit:

1. Train directions on original generated source + procedural control plus
   fresh repair-v2 prompts.
2. Use leave-one-fresh-pair-out evaluation so the repair is not just memorizing
   the three failures.
3. Gate on preserving original generated source, procedural target, fresh
   generated source, and fresh hand-authored control margins.

If fresh augmentation repairs `accountability_after_harm` and
`dissent_after_mistake` without breaking original/control margins, the next
out-of-family model should test whether this residual is Smol-specific or a
small-model geometry pattern.

## Claim Boundary

This remains a text-benchmark activation diagnostic. It does not support human
behavioral, neural, clinical, deployment, or real-world social-effect claims.
