from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.lexical_leakage import (
    run_lexical_leakage_report,
)
from social_cohesion_vectors.experiments.social_state_modulators import (
    activation_prompts_from_social_state_modulators,
    canonical_social_state_modulators,
    export_social_state_modulator_artifacts,
    normalize_modulator_id,
    render_markdown_summary,
    render_social_state_modulator_markdown,
    shape_social_state_modulator_report,
    social_state_modulator_pairwise_examples,
    social_state_modulator_scored_runs,
)
from social_cohesion_vectors.schemas import ActivationPrompt

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_modulator_ids_are_normalized_and_unique() -> None:
    modulators = canonical_social_state_modulators()

    assert normalize_modulator_id(" CK-1 Attunement Amplifier ") == (
        "ck_1_attunement_amplifier"
    )
    assert len({modulator.modulator_id for modulator in modulators}) == len(
        modulators
    )
    assert all(
        modulator.modulator_id == normalize_modulator_id(modulator.modulator_id)
        for modulator in modulators
    )
    assert all(modulator.vector_terms for modulator in modulators)
    assert all(modulator.phase_contrasts for modulator in modulators)
    assert all(modulator.contraindications for modulator in modulators)


def test_social_state_modulator_prompts_are_schema_valid_and_counted() -> None:
    modulators = canonical_social_state_modulators()
    scored_runs = social_state_modulator_scored_runs(modulators)
    pairs = social_state_modulator_pairwise_examples(modulators)
    prompts = activation_prompts_from_social_state_modulators(modulators)
    contrast_count = sum(
        len(modulator.phase_contrasts) for modulator in modulators
    )

    assert len(scored_runs) == contrast_count * 2
    assert len(pairs) == contrast_count
    assert len(prompts) == contrast_count * 2
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert all(prompt.target_score >= 0.0 for prompt in prompts)
    assert len({prompt.sample_id for prompt in prompts}) == len(prompts)
    assert all(
        prompt.pair_id.startswith("social-state-modulator::")
        for prompt in prompts
    )
    assert all("Social-state modulator:" in prompt.text for prompt in prompts)
    assert all("Contraindications:" in prompt.text for prompt in prompts)
    assert all(pair.positive_score > pair.negative_score for pair in pairs)
    assert all(float(pair.metadata["score_margin"]) > 0.0 for pair in pairs)
    assert all(
        float(pair.metadata["safe_attunement_margin"]) > 0.0 for pair in pairs
    )
    assert all(
        float(pair.metadata["pseudo_attunement_risk_margin"]) > 0.0
        for pair in pairs
    )
    assert all(
        ActivationPrompt.model_validate(prompt.model_dump()) for prompt in prompts
    )


def test_markdown_summary_names_recipe_and_guardrails() -> None:
    modulators = canonical_social_state_modulators()
    prompts = activation_prompts_from_social_state_modulators(modulators)
    markdown = render_markdown_summary(modulators=modulators, prompts=prompts)

    assert "# Social-State Modulator Activation Prompts" in markdown
    assert f"- Modulators: {len(modulators)}" in markdown
    assert f"- Activation prompts: {len(prompts)}" in markdown
    assert "`ck1_attunement_amplifier`" in markdown
    assert "`sycophancy` (inhibit)" in markdown
    assert "`boundary_collapse` (monitor)" in markdown
    assert "### Contraindications" in markdown


def test_social_state_modulator_report_and_leakage_shape() -> None:
    pairs = social_state_modulator_pairwise_examples()
    report = shape_social_state_modulator_report()
    markdown = render_social_state_modulator_markdown(report)
    leakage = run_lexical_leakage_report(
        pairs=pairs,
        group_metadata_key="phase",
    )

    assert report["summary"]["modulators"] == 1
    assert report["summary"]["phase_contrasts"] == 4
    assert report["summary"]["pairwise_examples"] == 4
    assert report["summary"]["activation_prompts"] == 8
    assert report["summary"]["scorer_pairwise_accuracy"] == 1.0
    assert report["summary"]["min_safe_attunement_margin"] > 0.0
    assert "Social-State Modulator Benchmark" in markdown
    assert leakage["summary"]["pairs"] == 4
    assert {row["group"] for row in leakage["groups"]} == {
        "intake",
        "repair",
        "shared_attention",
        "verification",
    }


def test_export_social_state_modulator_artifacts(tmp_path: Path) -> None:
    counts = export_social_state_modulator_artifacts(
        scored_runs_output=tmp_path / "runs.jsonl",
        pairs_output=tmp_path / "pairs.jsonl",
        prompts_output=tmp_path / "prompts.jsonl",
        json_report_output=tmp_path / "report.json",
        markdown_report_output=tmp_path / "report.md",
    )

    assert counts == {
        "scored_runs": 8,
        "pairwise_examples": 4,
        "activation_prompts": 8,
    }
    assert len(read_jsonl(tmp_path / "runs.jsonl")) == 8
    assert len(read_jsonl(tmp_path / "pairs.jsonl")) == 4
    assert len(read_jsonl(tmp_path / "prompts.jsonl")) == 8
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("#")


def test_cli_writes_jsonl_and_optional_markdown_summary(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    scored_runs = tmp_path / "social_state_modulator_scored_runs.jsonl"
    pairs = tmp_path / "social_state_modulator_pairwise_probe_dataset.jsonl"
    prompts_output = tmp_path / "social_state_modulator_activation_prompts.jsonl"
    json_report = tmp_path / "social_state_modulator_benchmark.json"
    markdown_report = tmp_path / "social_state_modulator_benchmark.md"

    assert (
        cli_module.main(
            [
                "--scored-runs-output",
                str(scored_runs),
                "--pairs-output",
                str(pairs),
                "--prompts-output",
                str(prompts_output),
                "--json-report-output",
                str(json_report),
                "--markdown-report-output",
                str(markdown_report),
            ]
        )
        == 0
    )

    records = read_jsonl(prompts_output)
    prompts = [ActivationPrompt.model_validate(record) for record in records]
    markdown = markdown_report.read_text(encoding="utf-8")

    assert len(prompts) == len(activation_prompts_from_social_state_modulators())
    assert len(read_jsonl(scored_runs)) == 8
    assert len(read_jsonl(pairs)) == 4
    assert prompts[0].sample_id == (
        "social-state-modulator::ck1_attunement_amplifier::"
        "intake::reflective_intake:positive"
    )
    assert "Social-State Modulator Benchmark" in markdown
    assert f"- Activation prompts: {len(prompts)}" in markdown


def _load_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "export_social_state_modulator_prompts.py"
    spec = importlib.util.spec_from_file_location(
        "export_social_state_modulator_prompts",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
