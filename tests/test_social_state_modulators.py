from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from social_cohesion_vectors.datasets import read_jsonl
from social_cohesion_vectors.experiments.social_state_modulators import (
    activation_prompts_from_social_state_modulators,
    canonical_social_state_modulators,
    normalize_modulator_id,
    render_markdown_summary,
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
    prompts = activation_prompts_from_social_state_modulators(modulators)
    contrast_count = sum(
        len(modulator.phase_contrasts) for modulator in modulators
    )

    assert len(prompts) == contrast_count * 2
    assert {prompt.label for prompt in prompts} == {"positive", "negative"}
    assert {prompt.target_score for prompt in prompts} == {0.0, 1.0}
    assert len({prompt.sample_id for prompt in prompts}) == len(prompts)
    assert all(
        prompt.pair_id.startswith("social-state-modulator::")
        for prompt in prompts
    )
    assert all("Social-state modulator:" in prompt.text for prompt in prompts)
    assert all("Contraindications:" in prompt.text for prompt in prompts)
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


def test_cli_writes_jsonl_and_optional_markdown_summary(tmp_path: Path) -> None:
    cli_module = _load_cli_module()
    output = tmp_path / "social_state_modulator_activation_prompts.jsonl"
    markdown_summary = tmp_path / "social_state_modulator_prompt_summary.md"

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

    assert len(prompts) == len(activation_prompts_from_social_state_modulators())
    assert prompts[0].sample_id == (
        "social-state-modulator::ck1_attunement_amplifier::"
        "reflective_intake:positive"
    )
    assert "Social-State Modulator Activation Prompts" in markdown
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
