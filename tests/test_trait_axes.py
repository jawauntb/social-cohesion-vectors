from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.trait_axes import (
    activation_prompts_from_trait_axes,
    canonical_trait_axes,
    normalize_axis_id,
    render_markdown_summary,
)
from social_cohesion_vectors.schemas import ActivationPrompt

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_axis_ids_are_normalized_and_unique() -> None:
    axes = canonical_trait_axes()

    assert normalize_axis_id(" Repair vs Harm Denial ") == "repair_vs_harm_denial"
    assert len({axis.axis_id for axis in axes}) == len(axes)
    assert all(axis.axis_id == normalize_axis_id(axis.axis_id) for axis in axes)
    assert all(axis.positive_pole and axis.negative_pole for axis in axes)
    assert all(axis.description for axis in axes)
    assert all(axis.contrasts for axis in axes)


def test_trait_axis_activation_prompts_are_schema_valid_and_counted() -> None:
    axes = canonical_trait_axes()
    prompts = activation_prompts_from_trait_axes(axes)
    contrast_count = sum(len(axis.contrasts) for axis in axes)

    assert len(prompts) == contrast_count * 2
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert {prompt.target_score for prompt in prompts} == {0.0, 1.0}
    assert len({prompt.sample_id for prompt in prompts}) == len(prompts)
    assert all(prompt.pair_id.startswith("trait-axis::") for prompt in prompts)
    assert all("Trait axis:" in prompt.text for prompt in prompts)
    assert all(
        ActivationPrompt.model_validate(prompt.model_dump()) for prompt in prompts
    )


def test_markdown_summary_names_axes_and_counts_prompts() -> None:
    axes = canonical_trait_axes()
    prompts = activation_prompts_from_trait_axes(axes)
    markdown = render_markdown_summary(axes=axes, prompts=prompts)

    assert "# Trait-Axis Activation Prompts" in markdown
    assert f"- Axes: {len(axes)}" in markdown
    assert f"- Activation prompts: {len(prompts)}" in markdown
    assert "`autonomy_vs_coercion`" in markdown
    assert "principled_respect_vs_sycophancy" in markdown
    assert "## Seed Contrasts" in markdown


def test_cli_writes_jsonl_and_optional_markdown_summary(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    output = tmp_path / "trait_axis_activation_prompts.jsonl"
    markdown_summary = tmp_path / "trait_axis_prompt_summary.md"

    assert (
        cli_module.main(
            [
                "--output",
                str(output),
                "--markdown-summary",
                str(markdown_summary),
            ]
        )
        == 0
    )

    records = read_jsonl(output)
    prompts = [ActivationPrompt.model_validate(record) for record in records]
    markdown = markdown_summary.read_text(encoding="utf-8")

    assert len(prompts) == len(activation_prompts_from_trait_axes())
    assert prompts[0].sample_id == (
        "trait-axis::repair_vs_harm_denial::missed_deadline:positive"
    )
    assert "Trait-Axis Activation Prompts" in markdown
    assert f"- Activation prompts: {len(prompts)}" in markdown


def _load_cli_module() -> ModuleType:
    module_path = _REPO_ROOT / "scripts" / "export_trait_axis_prompts.py"
    spec = importlib.util.spec_from_file_location(
        "export_trait_axis_prompts",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
