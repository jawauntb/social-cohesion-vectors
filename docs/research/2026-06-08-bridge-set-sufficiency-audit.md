# Bridge Set Sufficiency Audit

Date: 2026-06-08

## Question

Can the six-pair bridge threshold be reached intentionally, with explicit
procedural-path and source-style coverage, instead of being found only by
sampled subset search?

## Discovery-Regime Audit

Current regime:

- Source-only directions remain domain-specific in SmolLM2.
- Full bridge training, source-family bridge ablation, and pair-count bridge
  ablation all pass.
- Pair-count diagnosis showed that all-eight future-option coverage alone is
  not sufficient: target/control five-pair failures can already cover every
  future-option path.

Action:

- Added an intentional bridge-set sufficiency audit. For each held-out source
  family, it constructs one fixed-size same-domain bridge set by greedily
  maximizing new procedural paths, source-style coverage, and
  `primary_fault_class` coverage, then evaluates the held-out source family.

## Artifact

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_bridge_set_sufficiency_audit/smol17_generated_control_bridge_set_sufficiency.md
```

## Result

Model:

```text
HuggingFaceTB/SmolLM2-1.7B-Instruct
```

Layer:

```text
-2
```

Configuration:

```text
bridge_pair_count = 6
bridge_stratum_key = slack_options_tested
composition_keys = source,primary_fault_class
```

Summary:

| Held-out side | Folds | Min accuracy | Min margin | Path-complete folds | Failed folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| generated/source | 4 | 1.000 | +11.090 | 4/4 | 0 |
| control/target | 4 | 1.000 | +12.044 | 4/4 | 0 |

Every constructed bridge set covers all eight procedural future-option paths.
Every target/control bridge set covers all three available non-held-out source
families and all four control case/fault families:

```text
appeal_and_evidence
harm_repair
privacy_exit
voice_under_pressure
```

Every generated/source bridge set covers all three available non-held-out source
families and six generated fault families:

```text
deliberation_bypass
dissent_suppression
fairness_bypass
forced_forgiveness
privacy_bypass
punitive_accountability
```

## Interpretation

This closes the immediate bridge-set sufficiency bottleneck for the current
SmolLM2 diagnostic:

- six-pair bridge count is not merely a lucky sampled threshold;
- a metadata-only constructor can intentionally choose passing six-pair sets;
- the constructor succeeds where exact five-pair target/control sets still have
  failures;
- source-style and case/fault-family coverage are useful gates alongside
  future-option coverage.

This is still a text-benchmark activation result. It supports the claim that
the current generated/control benchmark pair has an intentionally constructible,
out-of-family, six-pair bridge direction on SmolLM2 layer `-2`; it does not
support human, neural, clinical, deployment, or social-effect claims.

## Next Operation

Use the constructed six-pair bridge sets as the active bridge baseline, then
rerun the broader generated/control direction-transfer checks around this
bridge setting:

- preserve the selected bridge-set manifests in generated reports;
- compare constructed bridge-set directions against full-domain joint
  directions;
- test whether the constructed bridge-set direction transfers across Qwen7B and
  SmolLM2 comparable layers;
- only then consider adding new model families.

## Claim Boundary

This result is an activation-space diagnostic over generated and hand-authored
text benchmarks. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
