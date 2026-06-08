# Availability-Targeted Generation Contract

Date: 2026-06-08

## Question

Can we revise the generated fault-class authoring regime so limited Modal
shards are aimed directly at the practical-availability verifier, rather than
only balancing lexical future-option words?

## Current Regime

- Artifact types: prompt records, raw provider outputs, generated examples,
  pairwise examples, tournament selections, audit bundles.
- Operations: build prompt records, generate/replay provider outputs, select
  candidates by tournament gates, run generated benchmark audits.
- Gates: component margin, slack preservation, lexical leakage, lexical
  baseline, source diversity, future-path availability, optional activation
  metadata transfer.
- Limitation: `lexical_negative_v1` asks for comparable future-option words but
  only requires at least two listed paths. Availability failures can survive
  because some declared paths are missing, weakly expressed, or equally usable
  on both labels.

## Regime Transition

Added a selectable prompt contract:

- `lexical_negative_v1`: preserved as the baseline and default;
- `availability_targeted_v1`: every listed future path must appear in both
  labels; genuine keeps each path practically usable; pseudo keeps comparable
  path words while making those same paths weaker through delay, private-only
  routing, permission, tone/loyalty tests, retaliation risk, or evidence
  restriction.

Added limited-shard ordering:

- `prioritize_prompt_records_for_future_options(...)`;
- `--availability-priority` for API generation, Modal generation, and the
  authorship tournament.

This is a search/regime-revision step, not an activation claim. It changes the
authoring schema and prompt-ledger ordering so the next generation run can test
the new verifier.

## Local Smoke Check

```bash
.venv/bin/python - <<'PY'
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    prioritize_prompt_records_for_future_options,
)
records = prioritize_prompt_records_for_future_options(
    build_fault_prompt_records(
        variants=DEFAULT_VARIANTS[:1],
        prompt_contract_version=API_AVAILABILITY_TARGETED_CONTRACT_VERSION,
    )
)[:20]
pairs = {}
for record in records:
    pairs.setdefault(f"{record.base_contrast_id}__{record.variant}", set()).update(
        str(record.metadata["future_options_tested"]).split(",")
    )
print(len(records), len(pairs), sorted(set().union(*pairs.values())))
PY
```

Result:

- prompt records: `20`;
- pairs: `10`;
- covered paths:
  `appeal,dissent,evidence_access,exit,privacy_choice,proportional_review,refusal,repair`;
- first prioritized pairs:
  `fair_allocation`, `data_choice`, `accountability_after_harm`,
  `autonomy_after_conflict`, `belonging_norms`;
- contract versions: `availability_targeted_v1`.

## Next Live Run

Generate the next first-20 Modal candidate batch with:

```bash
.venv/bin/python scripts/run_fault_class_modal_generation.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --variants neighborhood_forum \
  --limit 20 \
  --availability-priority \
  --prompt-contract-version availability_targeted_v1 \
  --raw-outputs /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/raw_outputs.jsonl \
  --examples-output /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/examples.jsonl \
  --scored-runs-output /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/scored_runs.jsonl \
  --pairs-output /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/pairs.jsonl \
  --prompts-output /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/activation_prompts.jsonl \
  --json-report-output /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/dataset.json \
  --markdown-report-output /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/dataset.md \
  --audit-output-dir /tmp/social_cohesion_modal_hf_qwen7_availability_targeted_20260608/chunk_000/audit_bundle
```

Any tournament using that batch must use the same prompt-ledger flags:

```bash
--availability-priority \
--prompt-contract-version availability_targeted_v1
```

## Gate

The next generated shard improves the research state only if it beats the
current selected baseline:

- availability pass rate above `1/10`;
- core gate pass rate above `0/10`;
- `evidence_access` and `privacy_choice` covered;
- no regression in score/slack/lexical readiness relative to the current
  selected tournament.

## Claim Boundary

This is a generated-text benchmark construction step. It makes no human
behavioral, neural, deployment, or real-world social-effect claim.
