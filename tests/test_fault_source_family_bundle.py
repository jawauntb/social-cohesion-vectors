from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from social_cohesion_vectors.datasets import read_jsonl, write_jsonl
from social_cohesion_vectors.experiments.fault_constrained_repair import (
    LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
    SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION,
    compose_constrained_repair_output_records,
)
from social_cohesion_vectors.experiments.fault_generation import (
    API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
    DEFAULT_VARIANTS,
    build_fault_prompt_records,
    filter_prompt_records_for_repair_targets,
    repair_targets_from_specs,
)
from social_cohesion_vectors.experiments.fault_source_family_bundle import (
    RawOutputSourceFamily,
    parse_raw_output_source_family_spec,
    render_fault_source_family_bundle_pipeline_markdown,
    run_fault_source_family_bundle_pipeline,
)

_SOURCE_FAMILY_REPAIR_TARGETS = (
    "accountability_after_harm=repair",
    "autonomy_after_conflict=dissent",
    "fair_allocation=refusal,appeal,repair",
)


def test_fault_source_family_bundle_exports_shared_raw_output_sources(
    tmp_path: Path,
) -> None:
    paths = _source_family_paths(tmp_path)
    primary_raw, diverse_raw = _write_source_family_raw_outputs(tmp_path)

    manifest = _run_source_family_bundle_pipeline(
        paths,
        sources=[
            RawOutputSourceFamily("primary", "source_family_primary", primary_raw),
            RawOutputSourceFamily("diverse", "source_family_diverse", diverse_raw),
        ],
    )
    markdown = render_fault_source_family_bundle_pipeline_markdown(manifest)
    pairs = read_jsonl(paths["pairs_output"])
    prompts = read_jsonl(paths["prompts_output"])

    assert manifest["summary"]["sources"] == 2
    assert manifest["summary"]["pairwise_examples"] == 6
    assert manifest["summary"]["activation_prompts"] == 12
    assert manifest["summary"]["prompt_records"] == 6
    assert _step(manifest, "source_diversity_audit")["ready"] is True
    assert _step(manifest, "fault_heldout_transfer")["ready"] is True
    assert _step(manifest, "activation_metadata_transfer")["status"] == "skipped"
    assert len(pairs) == 6
    assert len(prompts) == 12
    assert {row["metadata"]["provider"] for row in pairs} == {
        "source_family_diverse",
        "source_family_primary",
    }
    assert {row["metadata"]["source"] for row in pairs} == {
        "generated_fault_class_diverse",
        "generated_fault_class_primary",
    }
    assert "Fault Source-Family Bundle Pipeline" in markdown
    assert paths["dataset_markdown_report"].exists()
    assert paths["pipeline_markdown_report"].exists()


def test_parse_raw_output_source_family_spec_requires_source_provider_path() -> None:
    parsed = parse_raw_output_source_family_spec("primary=provider_a=/tmp/raw.jsonl")

    assert parsed.source_id == "primary"
    assert parsed.provider == "provider_a"
    assert parsed.raw_outputs_path == Path("/tmp/raw.jsonl")


def test_fault_source_family_bundle_cli_writes_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    script = _load_script()
    paths = _source_family_paths(tmp_path)
    primary_raw, diverse_raw = _write_source_family_raw_outputs(tmp_path)

    exit_code = script.main(
        [
            "--source-raw-outputs",
            f"primary=source_family_primary={primary_raw}",
            "--source-raw-outputs",
            f"diverse=source_family_diverse={diverse_raw}",
            "--scored-runs-output",
            str(paths["scored_runs_output"]),
            "--pairs-output",
            str(paths["pairs_output"]),
            "--prompts-output",
            str(paths["prompts_output"]),
            "--prompt-records-output",
            str(paths["prompt_records_output"]),
            "--dataset-json-report",
            str(paths["dataset_json_report"]),
            "--dataset-markdown-report",
            str(paths["dataset_markdown_report"]),
            "--audit-output-dir",
            str(paths["audit_output_dir"]),
            "--pipeline-json-report",
            str(paths["pipeline_json_report"]),
            "--pipeline-markdown-report",
            str(paths["pipeline_markdown_report"]),
            "--variants",
            DEFAULT_VARIANTS[0].name,
            "--availability-priority",
            "--prompt-contract-version",
            API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            "--repair-target",
            _SOURCE_FAMILY_REPAIR_TARGETS[0],
            "--repair-target",
            _SOURCE_FAMILY_REPAIR_TARGETS[1],
            "--repair-target",
            _SOURCE_FAMILY_REPAIR_TARGETS[2],
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "fault source-family bundle pipeline" in captured.out
    loaded = json.loads(paths["pipeline_json_report"].read_text(encoding="utf-8"))
    assert loaded["summary"]["sources"] == 2
    assert loaded["summary"]["pairwise_examples"] == 6
    assert loaded["audit_bundle"]["summary"]["skipped_steps"] == 1


def _write_source_family_raw_outputs(tmp_path: Path) -> tuple[Path, Path]:
    records = _source_family_records()
    primary_result = compose_constrained_repair_output_records(
        records,
        provider="source_family_primary",
        model=LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
        composer_version=LEXICAL_BALANCED_REPAIR_COMPOSER_VERSION,
    )
    diverse_result = compose_constrained_repair_output_records(
        records,
        provider="source_family_diverse",
        model=SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION,
        composer_version=SOURCE_DIVERSE_REPAIR_COMPOSER_VERSION,
    )
    primary_raw = tmp_path / "raw" / "primary.jsonl"
    diverse_raw = tmp_path / "raw" / "diverse.jsonl"
    write_jsonl(primary_result.output_records, primary_raw)
    write_jsonl(diverse_result.output_records, diverse_raw)
    return primary_raw, diverse_raw


def _source_family_records():
    repair_targets = repair_targets_from_specs(_SOURCE_FAMILY_REPAIR_TARGETS)
    return filter_prompt_records_for_repair_targets(
        build_fault_prompt_records(
            variants=DEFAULT_VARIANTS[:1],
            prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
            repair_focus_options_by_contrast=repair_targets,
        ),
        repair_targets,
    )


def _source_family_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "scored_runs_output": tmp_path / "processed" / "scored.jsonl",
        "pairs_output": tmp_path / "training" / "pairs.jsonl",
        "prompts_output": tmp_path / "training" / "prompts.jsonl",
        "prompt_records_output": tmp_path / "raw" / "prompt_records.jsonl",
        "dataset_json_report": tmp_path / "reports" / "dataset.json",
        "dataset_markdown_report": tmp_path / "reports" / "dataset.md",
        "audit_output_dir": tmp_path / "reports" / "audit_bundle",
        "pipeline_json_report": tmp_path / "reports" / "pipeline.json",
        "pipeline_markdown_report": tmp_path / "reports" / "pipeline.md",
    }


def _run_source_family_bundle_pipeline(
    paths: Mapping[str, Path],
    **kwargs: Any,
) -> dict[str, Any]:
    return run_fault_source_family_bundle_pipeline(
        scored_runs_output=paths["scored_runs_output"],
        pairs_output=paths["pairs_output"],
        prompts_output=paths["prompts_output"],
        prompt_records_output=paths["prompt_records_output"],
        dataset_json_report=paths["dataset_json_report"],
        dataset_markdown_report=paths["dataset_markdown_report"],
        audit_output_dir=paths["audit_output_dir"],
        pipeline_json_report=paths["pipeline_json_report"],
        pipeline_markdown_report=paths["pipeline_markdown_report"],
        variants=DEFAULT_VARIANTS[:1],
        prompt_contract_version=API_AVAILABILITY_REPAIR_STRICT_CONTRACT_VERSION,
        repair_target_specs=_SOURCE_FAMILY_REPAIR_TARGETS,
        availability_priority=True,
        **kwargs,
    )


def _step(manifest: dict[str, object], step_id: str) -> dict[str, object]:
    audit_bundle = manifest["audit_bundle"]
    assert isinstance(audit_bundle, dict)
    steps = audit_bundle["steps"]
    assert isinstance(steps, list)
    return next(step for step in steps if step["step_id"] == step_id)


def _load_script() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_fault_source_family_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_fault_source_family_bundle",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
