# Target Bridge Shortcut Repair

Date: 2026-06-11

## Question

Can pseudo-side shortcut-neutralized target/control bridge rows repair the
`dissent_after_mistake` constructed target-bridge failures without weakening
source/control or fresh hand-control margins?

## Discovery-Regime Audit

Current regime:

- Artifact types: hand-authored target/control bridge repair rows, scoped
  availability and lexical reports, Modal activation NPZs, constructed bridge
  diagnostics, bridge-count ablations, and dated research notes.
- Operations: add a small target/control repair bundle; extract augmented
  target activations on Modal; rerun the fixed fresh-generated bridge
  diagnostic over SmolLM2, Qwen2.5-0.5B, Qwen2.5-7B, and TinyLlama.
- Gate: default six-pair constructed target bridges must recover the dissent
  fresh-source perturbation slice while preserving original target/control and
  fresh hand-control margins.

Action class:

- Targeted bridge-regime repair. This is not a broad generation sweep and does
  not add human or neural evidence.

## Code Added

New target/control repair bundle:

```text
src/social_cohesion_vectors/experiments/target_bridge_shortcut_repair.py
```

New exporter:

```text
scripts/export_target_bridge_shortcut_repair.py
```

New tests:

```text
tests/test_target_bridge_shortcut_repair.py
```

## Live Artifacts

Generated artifacts remain outside git:

```text
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/
```

Key files:

```text
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/repair_pairs.jsonl
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/augmented_target_pairs.jsonl
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/augmented_target_activation_prompts.jsonl
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/repair_availability_scoped.md
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/bridge_stability_summary_augmented.md
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/bridge_stability_summary_augmented_count15.md
```

Modal activation outputs:

```text
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/augmented_target_smol17_layer-2.npz
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/augmented_target_qwen05b_layer-2.npz
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/augmented_target_qwen7b_layer-2.npz
/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/augmented_target_tinyllama_layer-2.npz
```

## Repair Bundle

The repair adds four hand-authored target/control rows, one for each existing
procedural-control source family:

- `hand_authored_case_notes_v1`
- `hand_authored_meeting_minutes_v1`
- `hand_authored_policy_review_v1`
- `hand_authored_incident_log_v1`

Each positive row makes refusal, evidence access, exit, dissent, and repair
usable now. Each negative row preserves procedural taxes through private review,
approval, permission, and alignment while avoiding the earlier pseudo-side
warmth shortcuts.

## Text Gates

Scoped practical availability over the five repaired options passes:

| Gate | Result |
| --- | ---: |
| Repair pairs | `4` |
| Tested paths | `20` |
| Required options covered | `5/5` |
| Paths preferring genuine | `20/20` |
| Mean availability margin | `+0.710` |
| Min availability margin | `+0.640` |
| Activation readiness | `availability_ready` |

Lexical diagnostics remain acceptable for a targeted repair:

| Diagnostic | Result |
| --- | ---: |
| Lexical cue-solved rate | `0.250` |
| Mean cue margin | `+0.250` |
| Label-aligned terms | `1` |
| Aggregate term polarized | `False` |

## Default Six-Pair Bridge Result

The default six-pair constructed bridge gate does not pass.

| Model | Readiness | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |
| --- | --- | ---: | ---: | ---: | ---: |
| SmolLM2-1.7B layer `-2` | `not_ready` | `-18.128` | `19` | `+20.403` | `0` |
| Qwen2.5-0.5B layer `-2` | `not_ready` | `-1.455` | `12` | `-0.302` | `4` |
| Qwen2.5-7B layer `-2` | `not_ready` | `-2.803` | `4` | `+9.343` | `0` |
| TinyLlama-1.1B layer `-2` | `not_ready` | `-0.063` | `1` | `+0.614` | `0` |

Compared with the previous dissent bridge-stability baseline, the default
repair changes the shape but does not solve the problem:

| Metric | Before repair | Six-pair repair |
| --- | ---: | ---: |
| Constructed failure rows | `39` | `40` |
| Failing models | `3` | `4` |
| Worst constructed margin | `-18.296` | `-18.128` |
| Most failed perturbation | `negative_shortcuts_neutralized` | `negative_shortcuts_neutralized` |
| Most failed bridge family | `target_bridge` | `target_bridge` |

The selected bridge subset contains only one new repair row per target holdout,
so the small additive content repair is underpowered at the original six-pair
bridge budget.

## Bridge-Budget Ablation

No extra Modal extraction was needed for the bridge-count ablation. The same
augmented target activations were reused while increasing the constructed
bridge-pair count.

| Bridge pairs | Model | Ready | Fresh source min | Fresh source failures | Fresh target min | Fresh target failures |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| `6` | SmolLM2 | `False` | `-18.128` | `19` | `+20.403` | `0` |
| `6` | Qwen2.5-0.5B | `False` | `-1.455` | `12` | `-0.302` | `4` |
| `6` | Qwen2.5-7B | `False` | `-2.803` | `4` | `+9.343` | `0` |
| `6` | TinyLlama | `False` | `-0.063` | `1` | `+0.614` | `0` |
| `9` | SmolLM2 | `False` | `-10.213` | `12` | `+27.330` | `0` |
| `9` | Qwen2.5-0.5B | `False` | `-1.095` | `8` | `-0.011` | `1` |
| `9` | Qwen2.5-7B | `False` | `-1.094` | `4` | `+11.245` | `0` |
| `9` | TinyLlama | `True` | `+0.057` | `0` | `+0.818` | `0` |
| `12` | SmolLM2 | `False` | `-8.737` | `8` | `+34.311` | `0` |
| `12` | Qwen2.5-0.5B | `False` | `-1.055` | `8` | `+0.239` | `0` |
| `12` | Qwen2.5-7B | `False` | `-0.791` | `1` | `+14.190` | `0` |
| `12` | TinyLlama | `True` | `+0.010` | `0` | `+1.060` | `0` |
| `15` | SmolLM2 | `False` | `-6.310` | `5` | `+40.406` | `0` |
| `15` | Qwen2.5-0.5B | `False` | `-0.538` | `8` | `+0.933` | `0` |
| `15` | Qwen2.5-7B | `True` | `+0.893` | `0` | `+16.376` | `0` |
| `15` | TinyLlama | `True` | `+0.166` | `0` | `+1.338` | `0` |

At the all-candidate count-15 target bridge budget, failures drop to `13`
constructed rows. Qwen2.5-7B and TinyLlama pass; SmolLM2 and Qwen2.5-0.5B
still fail on the pseudo-side warmth/shortcut perturbations.

## Finding

The targeted content repair is not sufficient at the default six-pair bridge
budget. The stronger result is a bridge-sufficiency finding:

> Constructed target-bridge transfer on the dissent perturbation slice is
> bridge-budget sensitive and model-space dependent. Increasing target/control
> bridge coverage repairs Qwen2.5-7B and TinyLlama, but SmolLM2 and
> Qwen2.5-0.5B retain a pseudo-side warmth/shortcut residual even when all
> available target bridge candidates are used.

This is meaningful progress, but it is not a NeurIPS-grade endpoint yet. The
paper-shaped claim needs either a principled small-bridge selector that clears
the gate without all-candidate target coverage, or a sharper residual account
explaining why small-model spaces retain shortcut-sensitive target-bridge
geometry.

## Next Move

Follow-up completed in
`docs/research/2026-06-11-weighted-target-bridge-repair.md`: asymmetric
weighted target-bridge construction with bridge count `9` and target bridge
repetitions `1:3` clears the constructed bridge gate across all four tested
model spaces.

Do not run a broad generation sweep next. The next efficient move is
selection-side:

1. Add a first-class bridge-pair count sweep or bridge-selector audit so the
   six-pair vs all-candidate result is reproducible from the CLI.
2. Test selector variants that prioritize target-side dissent/shortcut repair
   rows before generic path coverage.
3. If small-model failures persist under the strongest selector, document the
   remaining SmolLM2/Qwen0.5B residual as a model-space limitation and move the
   publication path toward bridge-sufficiency laws rather than a universal
   repair claim.

## Claim Boundary

This remains a text-benchmark activation diagnostic over generated and
hand-authored prompt artifacts. It does not establish causal steering, human
behavioral, neural, clinical, deployment, or real-world social-effect claims.
