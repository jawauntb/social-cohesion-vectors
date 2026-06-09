# Bridge Direction Comparison

Date: 2026-06-08

## Question

Do intentionally constructed six-pair bridge directions merely pass held-out
bridge-set folds, or do they behave like useful approximations of the full
generated/control joint direction?

## Artifact

```text
/tmp/social_cohesion_out_of_family_repl_20260608/smol17_bridge_direction_comparison/smol17_generated_control_bridge_direction_comparison.md
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

Baseline directions:

| Direction | Evaluation set | Accuracy | Min margin | Failed pairs |
| --- | --- | ---: | ---: | ---: |
| source-only | source | 1.000 | +13.567 | 0 |
| source-only | target | 0.750 | -15.572 | 4 |
| target-only | source | 0.950 | -5.054 | 2 |
| target-only | target | 1.000 | +65.028 | 0 |
| joint | source | 1.000 | +21.173 | 0 |
| joint | target | 1.000 | +50.555 | 0 |

Constructed bridge directions:

| Constructed direction family | Folds | Min joint cosine | Source min margin | Target min margin | Failed folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| target/control bridge directions | 4 | +0.965 | +20.945 | +12.044 | 0 |
| generated/source bridge directions | 4 | +0.756 | +11.090 | +72.168 | 0 |
| all constructed bridge directions | 8 | +0.756 | +11.090 | +12.044 | 0 |

## Interpretation

This is the strongest bridge result so far:

- source-only directions still fail cross-benchmark transfer, so the test is
  not trivial;
- the full joint direction separates both benchmarks with large positive
  margins;
- every intentional six-pair bridge direction separates both full benchmarks;
- all constructed directions have positive cosine with the full joint direction;
- target/control bridge directions are especially close to joint direction
  geometry, with cosine at least `+0.965`.

The generated/source constructed bridge directions are less joint-like, with
minimum cosine `+0.756`, but they still rank all generated and control pairs
correctly. That makes the current residual narrower: the six-pair bridge sets
are sufficient for SmolLM2 layer `-2`; the next question is cross-model
transport, not whether the constructed bridge direction exists in this model.

## Next Operation

Run the constructed bridge direction comparison on comparable accepted Qwen7B
layers, then compare Qwen7B and SmolLM2 constructed bridge directions using the
existing same-prompt alignment/mapped-transfer machinery.

## Claim Boundary

This result is an activation-space diagnostic over generated and hand-authored
text benchmarks. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
