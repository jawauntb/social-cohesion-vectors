# Fresh-Generated Bridge Diagnostic Scaffold

Date: 2026-06-09

## Question

Can we diagnose the fresh-generated bridge transport asymmetry without adding a
new model family or another broad generation sweep?

## Discovery-Regime Audit

Current regime:

- Artifact types: original generated source pairs, original hand-authored
  procedural-justice control pairs, fresh generated repair-v2 pairs, fresh
  hand-authored control pairs, open-model activation NPZ files, constructed
  bridge directions, and per-pair margin tables.
- Operations: train source-only, fresh-source-only, source+fresh-source, and
  constructed bridge directions inside one activation space; evaluate each
  direction on original source, original target, fresh source, and fresh
  target prompt slices.
- Gate: constructed bridge directions must preserve pairwise accuracy `1.000`
  and positive minimum margins on both fresh generated source prompts and fresh
  hand-authored control prompts. Source-only and fresh-source-only rows are
  diagnostic baselines and may fail.

Action class:

- Discovery tooling inside the existing activation-diagnostic regime. This adds
  a new verifier/report, not a new evidence claim.

## Code Added

New audit entry point:

```text
run_fresh_generated_bridge_diagnostic_from_files
```

New CLI:

```text
scripts/run_fresh_generated_bridge_diagnostic.py
```

The report emits:

- source-only, fresh-source-only, source+fresh-source, and constructed bridge
  direction evaluations;
- original source, original target, fresh source, and fresh target accuracy and
  minimum margin for each direction;
- cosine to source-only, fresh-source-only, source+fresh-source, and original
  source+target joint directions;
- failed-pair tables across every direction/evaluation set.

## Live-Run Boundary

No accepted live Qwen7B/SmolLM2 diagnostic was run in this scaffold branch.
The available local activation NPZ files preserve prompt text and pair IDs, but
the exact generated pair manifests from the previous `/tmp` runs were cleaned
before this 2026-06-09 follow-up. Those manifests contain the real
`slack_options_tested` and bridge-composition metadata needed to reproduce the
accepted constructed bridge sets.

Synthesizing pair manifests from activation IDs alone would be useful as a
smoke test, but it would not be the accepted fresh-generated bridge gate.

## Next Live Commands

After regenerating or preserving the exact pair manifests, run the diagnostic
once per model/layer. The Qwen7B shape is:

```text
.venv/bin/python scripts/run_fresh_generated_bridge_diagnostic.py \
  --source-activation-npz data/features/open_llm/layer_sweep/qwen7b_replication_four_source_v1__Qwen__Qwen2.5-7B-Instruct__layer-2.npz \
  --source-pairs /tmp/social_cohesion_source_family_bundle_20260608/qwen7b_replication_four_source_v1/pairs.jsonl \
  --target-activation-npz data/features/open_llm/layer_sweep/procedural_justice_control_v2__Qwen__Qwen2.5-7B-Instruct__layer-2.npz \
  --target-pairs /tmp/social_cohesion_out_of_family_repl_20260608/control_v2/pairs.jsonl \
  --fresh-source-activation-npz data/features/open_llm/layer_sweep/fresh_generated_repair_v2_limit20__Qwen__Qwen2.5-7B-Instruct__layer-2.npz \
  --fresh-source-pairs /tmp/social_cohesion_authorship_tournament_20260608/availability_targeted_plus_repair_000_001_repair_v2_000_limit20/pairs.jsonl \
  --fresh-target-activation-npz data/features/open_llm/layer_sweep/procedural_justice_control_v1__Qwen__Qwen2.5-7B-Instruct__layer-2.npz \
  --fresh-target-pairs /tmp/social_cohesion_procedural_justice_control_20260608/control_v1/pairs.jsonl \
  --source-name generated_four_source \
  --target-name procedural_control_v2 \
  --fresh-source-name fresh_generated_repair_v2_limit20 \
  --fresh-target-name procedural_control_v1 \
  --bridge-stratum-key slack_options_tested \
  --composition-keys source,primary_fault_class \
  --bridge-pair-count 6
```

Repeat with the corresponding SmolLM2 layer `-2` activation NPZ files. The
diagnostic should decide whether the fresh-generated failure is:

- present inside Qwen7B before transport;
- present inside SmolLM2 before transport;
- repaired by source+fresh-source training but not constructed bridges;
- concentrated only in constructed bridges; or
- traceable to specific failed pairs such as `accountability_after_harm`,
  `belonging_norms`, and `dissent_after_mistake`.

## Claim Boundary

This scaffold supports only generated-text and hand-authored-text activation
diagnostics. It does not support human behavioral, neural, clinical,
deployment, or real-world social-effect claims.
