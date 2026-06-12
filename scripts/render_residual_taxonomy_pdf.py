"""Render the residual-taxonomy working paper as a vector PDF."""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
MARGIN = 44.0


@dataclass(frozen=True)
class Color:
    red: float
    green: float
    blue: float


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    color: Color


@dataclass(frozen=True)
class GroupedBar:
    label: str
    first: float
    second: float


INK = Color(0.10, 0.14, 0.15)
MUTED = Color(0.38, 0.43, 0.43)
PAPER = Color(0.97, 0.97, 0.93)
PANEL = Color(1.00, 1.00, 0.98)
LINE = Color(0.82, 0.84, 0.79)
TEAL = Color(0.08, 0.45, 0.48)
TEAL_LIGHT = Color(0.70, 0.88, 0.86)
GOLD = Color(0.82, 0.55, 0.16)
GOLD_LIGHT = Color(0.96, 0.83, 0.54)
CORAL = Color(0.77, 0.25, 0.22)
CORAL_LIGHT = Color(0.96, 0.72, 0.65)
GREEN = Color(0.20, 0.52, 0.34)
GREEN_LIGHT = Color(0.72, 0.88, 0.74)
BLUE = Color(0.20, 0.33, 0.58)
BLUE_LIGHT = Color(0.70, 0.78, 0.93)


class Canvas:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: Color | None = None,
        stroke: Color | None = None,
        stroke_width: float = 1.0,
    ) -> None:
        bottom = PAGE_HEIGHT - y - height
        parts = ["q", f"{stroke_width:.2f} w"]
        if fill is not None:
            parts.append(_fill(fill))
        if stroke is not None:
            parts.append(_stroke(stroke))
        parts.append(f"{x:.2f} {bottom:.2f} {width:.2f} {height:.2f} re")
        parts.append(_paint(fill, stroke))
        parts.append("Q")
        self.commands.append(" ".join(parts))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: Color = LINE,
        width: float = 1.0,
    ) -> None:
        y1_pdf = PAGE_HEIGHT - y1
        y2_pdf = PAGE_HEIGHT - y2
        self.commands.append(
            " ".join(
                [
                    "q",
                    f"{width:.2f} w",
                    _stroke(color),
                    f"{x1:.2f} {y1_pdf:.2f} m",
                    f"{x2:.2f} {y2_pdf:.2f} l",
                    "S",
                    "Q",
                ]
            )
        )

    def text(
        self,
        value: str,
        x: float,
        y: float,
        *,
        size: float = 10.0,
        font: str = "F1",
        color: Color = INK,
        align: str = "left",
    ) -> None:
        text_width = _measure(value, size, font)
        adjusted_x = _aligned_x(x, text_width, align)
        baseline = PAGE_HEIGHT - y - size
        self.commands.append(
            " ".join(
                [
                    "BT",
                    f"/{font} {size:.2f} Tf",
                    _fill(color),
                    f"{adjusted_x:.2f} {baseline:.2f} Td",
                    f"({_escape(value)}) Tj",
                    "ET",
                ]
            )
        )

    def wrapped_text(
        self,
        value: str,
        x: float,
        y: float,
        width: float,
        *,
        size: float = 10.0,
        font: str = "F1",
        color: Color = INK,
        leading: float | None = None,
    ) -> float:
        line_height = leading if leading is not None else size * 1.36
        max_chars = max(18, int(width / (size * 0.52)))
        current_y = y
        for paragraph in value.split("\n"):
            for line in textwrap.wrap(paragraph, width=max_chars):
                self.text(line, x, current_y, size=size, font=font, color=color)
                current_y += line_height
            current_y += line_height * 0.35
        return current_y

    def stream(self) -> bytes:
        return ("\n".join(self.commands) + "\n").encode("ascii")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    pages = [
        _title_page(),
        _abstract_page(),
        _introduction_page(),
        _benchmark_page(),
        _taxonomy_page(),
        _accountability_page(),
        _dissent_text_page(),
        _repair_funnel_page(),
        _dissent_page(),
        _preservation_text_page(),
        _negative_control_page(),
        _discussion_page(),
        _reproducibility_page(),
    ]
    _write_pdf(pages, args.output)
    print(f"wrote {args.output}")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/neurips_residual_taxonomy_bridge_sufficiency.pdf"),
    )
    return parser.parse_args(argv)


def _title_page() -> Canvas:
    canvas = _page_base(1, "Working paper PDF")
    canvas.text("Residual Taxonomy & Bridge Sufficiency", 54, 72, size=22, font="F2")
    canvas.text(
        "for Procedural-Justice Activation Directions",
        54,
        104,
        size=18,
        font="F2",
        color=TEAL,
    )
    canvas.wrapped_text(
        "A NeurIPS-style working draft summarizing generated and hand-authored "
        "text-benchmark activation evidence. The supported claim is narrow: two "
        "different residual classes, two matched repairs, and a negative control "
        "blocking universal repair claims.",
        56,
        150,
        500,
        size=12,
        color=MUTED,
    )
    _metric_card(canvas, 54, 242, "Residual classes", "2", "source pocket + bridge geometry", TEAL)
    _metric_card(canvas, 224, 242, "Model spaces", "4", "SmolLM2, Qwen0.5B, Qwen7B, Tiny", GOLD)
    _metric_card(canvas, 394, 242, "Dissent bridge failures", "0", "after weighted target bridges", GREEN)
    _claim_boundary(canvas, 54, 384, 504)
    _mini_taxonomy(canvas, 72, 560)
    return canvas


def _abstract_page() -> Canvas:
    canvas = _page_base(2, "Abstract")
    _section_title(canvas, "Abstract", "Full textual paper content, not only charts.", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        126,
        486,
        [
            "Activation-space studies of social constructs risk confusing semantic structure with surface shortcuts, generated-text artifacts, or fragile selection effects. We study this problem in a procedural-justice benchmark where positive examples preserve usable voice, refusal, evidence access, appeal, non-retaliatory exit, and proportionate repair, while pseudo-cohesive negative examples preserve warm or orderly language but tax those future options.",
            "Across generated and hand-authored controls, four model spaces, and constructed source/target bridge directions, residual failures are not monolithic. The strongest current evidence supports two distinct residual classes.",
            "First, accountability_after_harm exposes a generated-reference source pocket: small models invert one recovered generated reference while clean hand-authored accountability controls pass, and strict local perturbation augmentation repairs the slice with positive leave-one-perturbation-out margins in all four model spaces.",
            "Second, dissent_after_mistake exposes a constructed target-bridge geometry pocket: broad source+target directions already score the generated reference correctly, but constructed target bridges fail when pseudo-side warmth shortcuts are removed. This residual is repaired by asymmetric target-bridge weighting and preserved across source, target, fresh-source, and fresh-target evaluations.",
            "A strict accountability negative control rejects the overgeneralized claim that weighted target bridges repair all generated pockets. The result is a narrow diagnostic contribution: a claim-bounded residual taxonomy and bridge-sufficiency protocol for procedural-justice activation directions, not evidence of human social effects or causal steering.",
        ],
        size=10.2,
        leading=13.8,
    )
    _claim_boundary(canvas, 64, y + 18, 486)
    return canvas


def _introduction_page() -> Canvas:
    canvas = _page_base(3, "Introduction")
    _section_title(canvas, "1. Introduction", "The target is agency-preserving procedural justice.", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        122,
        486,
        [
            "The project asks whether language-model activations can represent distinctions that matter for agency-preserving social cohesion. The target is deliberately narrower than making a model more prosocial. In the current benchmark, a positive procedural-justice example must keep future options genuinely usable: voice, refusal, evidence access, appeal, non-retaliatory exit, and repair.",
            "A pseudo-cohesive negative example may sound warm, calm, or cooperative, but it taxes those options through approval requirements, privacy loss, alignment pressure, proof burdens, or delayed remedy.",
            "The central problem is not only whether a direction separates labels. A direction can separate a benchmark for the wrong reason: lexical cues, generated style, source-family duplication, target-side shortcut phrases, or bridge-set selection. The research loop therefore moved from simple benchmark construction to a stricter question.",
        ],
        size=10.2,
        leading=13.6,
    )
    _quote_box(
        canvas,
        74,
        y + 8,
        464,
        "When activation directions fail after practical-availability, lexical, source-diversity, hand-authored control, and out-of-family gates, what kind of failure remains, and what kind of repair actually fixes it?",
    )
    y += 120
    _subheading(canvas, "Contributions", 64, y)
    y = _numbered_list(
        canvas,
        [
            "A claim-bounded procedural-justice activation benchmark pipeline that treats practical availability as a path-level gate rather than a mention-level property.",
            "A residual taxonomy for activation failures that survive lexical, availability, hand-authored control, and model-family checks.",
            "Two accepted repairs matched to distinct residual classes: perturbation-ladder augmentation for a generated-reference source pocket, and asymmetric target-bridge weighting for a constructed target-bridge geometry pocket.",
            "Rejected alternatives and negative controls showing that content-only target repair is insufficient for the dissent bridge residual, and weighted target bridges are not a universal generated-pocket repair.",
        ],
        74,
        y + 28,
        464,
    )
    return canvas


def _benchmark_page() -> Canvas:
    canvas = _page_base(4, "Benchmark and gates")
    _section_title(canvas, "2. Benchmark And Gates", "The benchmark is built to reject shortcut cohesion.", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        122,
        486,
        [
            "The benchmark is organized around paired positive and pseudo-cohesive examples. The positive side must make future options usable now. The pseudo side must retain procedural taxes even when it avoids obvious villain language.",
            "All activation diagnostics in this draft use layer -2 reports for four model spaces: SmolLM2-1.7B, Qwen2.5-0.5B, Qwen2.5-7B, and TinyLlama-1.1B. Generated/model artifacts are kept out of git; durable summaries are committed under docs/research/.",
        ],
        size=10.2,
        leading=13.6,
    )
    _subheading(canvas, "Main gates", 64, y + 6)
    _simple_table(
        canvas,
        64,
        y + 42,
        486,
        ["Gate", "Purpose"],
        [
            ["Practical availability", "Future-option paths are usable in positives and taxed in pseudo examples."],
            ["Lexical diagnostics", "Detect shallow terms, lengths, or label-aligned phrase shortcuts."],
            ["Source diversity", "Prevent duplicated generated phrasings from masquerading as semantic coverage."],
            ["Hand-authored controls", "Test whether the distinction survives outside generated text."],
            ["Out-of-family replication", "Test whether the result survives additional model spaces."],
            ["Constructed bridge diagnostics", "Train from limited bridge subsets and evaluate held-out fresh slices."],
            ["Preservation / negative controls", "Confirm repairs do not damage earlier slices or overgeneralize."],
        ],
        row_height=45,
        column_widths=[150, 336],
    )
    return canvas


def _taxonomy_page() -> Canvas:
    canvas = _page_base(5, "Residual taxonomy")
    _section_title(canvas, "Two Residual Classes", "Failures are not monolithic.", 54, 58)
    _lane(
        canvas,
        54,
        126,
        "Generated-reference source pocket",
        "accountability_after_harm",
        "Small models invert one recovered generated reference while clean "
        "hand-authored accountability controls pass.",
        "Accepted repair: strict perturbation augmentation.",
        TEAL,
    )
    _lane(
        canvas,
        54,
        318,
        "Constructed target-bridge geometry pocket",
        "dissent_after_mistake",
        "Broad directions score the reference correctly, but target bridges "
        "fail when pseudo-side warmth shortcuts are removed.",
        "Accepted repair: count-9 target bridges with 1:3 target weighting.",
        GOLD,
    )
    _ledger_row(canvas, 74, 548, "Accepted", "Two matched repairs clear all four model spaces.", GREEN)
    _ledger_row(canvas, 74, 604, "Rejected", "Content-only target repair and universal bridge repair.", CORAL)
    _ledger_row(canvas, 74, 660, "Withheld", "belonging_norms and other incomplete availability controls.", BLUE)
    return canvas


def _repair_funnel_page() -> Canvas:
    canvas = _page_base(8, "Repair funnel")
    _section_title(canvas, "Dissent Bridge Repair Funnel", "Rejected paths make the result sharper.", 54, 58)
    bars = [
        Bar("Initial target-bridge failures", 39, CORAL),
        Bar("Content repair, six pairs", 40, CORAL_LIGHT),
        Bar("All target candidates, count 15", 13, GOLD),
        Bar("Weighted target bridge, count 9", 0, GREEN),
    ]
    _horizontal_bars(canvas, 74, 142, 460, 230, bars, max_value=42)
    _callout(
        canvas,
        72,
        438,
        468,
        "The accepted repair changes bridge construction, not generation. "
        "Target bridge pair count stays moderate at 9, but the target-side "
        "secondary bridge evidence is repeated 3x while source bridge weights "
        "stay at 1:1.",
        GREEN_LIGHT,
        GREEN,
    )
    _bridge_recipe(canvas, 84, 594)
    return canvas


def _dissent_page() -> Canvas:
    canvas = _page_base(9, "Dissent margins")
    _section_title(canvas, "Accepted Dissent Repair", "Fresh slices and preservation both pass.", 54, 58)
    bars = [
        Bar("SmolLM2", 3.151, TEAL),
        Bar("Qwen0.5B", 0.220, GOLD),
        Bar("Qwen7B", 6.866, GREEN),
        Bar("TinyLlama", 0.513, BLUE),
    ]
    _vertical_bars(canvas, 64, 146, 496, 220, bars, max_value=7.0, label="Fresh-source margin")
    preservation = [
        Bar("SmolLM2", 3.151, TEAL),
        Bar("Qwen0.5B", 0.019, GOLD),
        Bar("Qwen7B", 6.681, GREEN),
        Bar("TinyLlama", 0.388, BLUE),
    ]
    _vertical_bars(canvas, 64, 442, 496, 190, preservation, max_value=7.0, label="Worst preservation margin")
    _footer_note(
        canvas,
        "Preservation audit: 32 constructed direction rows, 16 model/evaluation rows, 0 failed pair evaluations, worst margin +0.019.",
    )
    return canvas


def _negative_control_page() -> Canvas:
    canvas = _page_base(11, "Negative control")
    _section_title(canvas, "Accountability Negative Control", "The bridge repair is not universal.", 54, 58)
    bars = [
        GroupedBar("SmolLM2", 41, 20),
        GroupedBar("Qwen0.5B", 50, 43),
        GroupedBar("Qwen7B", 12, 0),
        GroupedBar("TinyLlama", 88, 84),
    ]
    _grouped_bars(canvas, 64, 146, 496, 250, bars, max_value=90)
    _failure_total(canvas, 76, 452, "Default count-9", 192, CORAL)
    _failure_total(canvas, 314, 452, "Weighted target x3", 147, GOLD)
    _callout(
        canvas,
        72,
        594,
        468,
        "Interpretation: weighted target bridges repair the dissent geometry "
        "pocket and preserve its source/control slices. They do not repair the "
        "strict accountability generated-reference pocket in three small-model "
        "spaces, so the paper claim stays a residual taxonomy rather than a "
        "universal recipe.",
        BLUE_LIGHT,
        BLUE,
    )
    return canvas


def _accountability_page() -> Canvas:
    canvas = _page_base(6, "Residual 1")
    _section_title(canvas, "3. Residual 1: Source Pocket", "accountability_after_harm", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        122,
        486,
        [
            "The hardest accountability residual was the recovered generated accountability_after_harm reference. The broad source+target direction inverted this reference in SmolLM2, Qwen2.5-0.5B, and TinyLlama, while Qwen2.5-7B remained positive.",
            "A source-style intervention narrowed the issue: clean hand-authored accountability variants passed, including a generated-like paragraph. The residual was therefore not broad accountability content failure.",
            "A strict perturbation ladder split the reference into twelve deterministic variants, including opening-frame edits, neutral replacements, length controls, explicit refusal, pseudo-side condition edits, and shortcut neutralization. Scoped practical availability remained intact over all 72/72 tested paths.",
            "Lexical diagnostics stayed caveated because the artifact was intentionally the known leaky generated reference. This experiment is therefore interpreted as a local mechanism stress test rather than a clean benchmark.",
        ],
        size=10.0,
        leading=13.4,
    )
    _subheading(canvas, "Accepted repair: strict perturbation augmentation", 64, y + 4)
    _simple_table(
        canvas,
        64,
        y + 38,
        486,
        ["Model", "Full fresh", "Fresh LOO", "Source", "Target"],
        [
            ["SmolLM2", "+54.525", "+47.185", "+20.317", "+52.012"],
            ["Qwen0.5B", "+2.228", "+1.376", "+1.512", "+1.549"],
            ["Qwen7B", "+25.406", "+23.059", "+5.052", "+22.233"],
            ["TinyLlama", "+1.029", "+0.800", "+0.593", "+1.194"],
        ],
        row_height=34,
        column_widths=[130, 88, 88, 88, 92],
    )
    _callout(
        canvas,
        72,
        636,
        468,
        "Interpretation: small model directions can encode the accountability distinction, but one generated reference lies in a local source pocket that requires nearby perturbation support.",
        TEAL_LIGHT,
        TEAL,
    )
    return canvas


def _dissent_text_page() -> Canvas:
    canvas = _page_base(7, "Residual 2")
    _section_title(canvas, "4. Residual 2: Bridge Geometry", "dissent_after_mistake", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        122,
        486,
        [
            "The dissent residual behaved differently. The unchanged generated dissent reference was already positive under broad source+target directions in all four model spaces. Positive-side path and neutral replacement edits strengthened rather than repaired the result.",
            "The failure appeared only when constructed bridge directions were trained from limited bridge subsets and evaluated on fresh perturbations. The bridge-stability summary localized the initial failure to target_bridge rows, especially the negative_shortcuts_neutralized perturbation.",
            "The first repair added shortcut-neutralized target/control content. It passed the scoped text gate with 20/20 repaired availability paths and minimum margin +0.640, but the default six-pair constructed bridge gate did not improve. Constructed failures moved from 39 to 40, and all failures remained target-bridge failures.",
            "Increasing target bridge count to 15 helped but did not finish the repair: failures dropped to 13, Qwen2.5-7B and TinyLlama passed, and SmolLM2 plus Qwen2.5-0.5B still failed.",
        ],
        size=10.0,
        leading=13.4,
    )
    _subheading(canvas, "Accepted repair: asymmetric target-bridge weighting", 64, y + 8)
    _simple_table(
        canvas,
        64,
        y + 42,
        486,
        ["Parameter", "Value"],
        [
            ["Target bridge pair count", "9"],
            ["Target bridge primary repetitions", "1"],
            ["Target bridge secondary repetitions", "3"],
            ["Source bridge primary repetitions", "1"],
            ["Source bridge secondary repetitions", "1"],
        ],
        row_height=34,
        column_widths=[340, 146],
    )
    return canvas


def _preservation_text_page() -> Canvas:
    canvas = _page_base(10, "Preservation")
    _section_title(canvas, "5. Preservation And Control", "A repair must preserve earlier distinctions.", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        122,
        486,
        [
            "A repair is only useful if it preserves the earlier distinctions. The weighted dissent bridge repair passed preservation across source, target, fresh-source, and fresh-target slices.",
            "The same weighted target-bridge setting was then tested against the strict accountability residual as a negative control. It improved some margins but did not repair the class.",
        ],
        size=10.2,
        leading=13.6,
    )
    _subheading(canvas, "Dissent preservation audit", 64, y + 14)
    _simple_table(
        canvas,
        64,
        y + 50,
        486,
        ["Metric", "Result"],
        [
            ["Models", "4"],
            ["Constructed direction rows", "32"],
            ["Evaluation rows", "16"],
            ["Failed pair evaluations", "0"],
            ["Worst margin", "+0.019"],
            ["Worst evaluation", "source"],
        ],
        row_height=31,
        column_widths=[310, 176],
    )
    _callout(
        canvas,
        72,
        620,
        468,
        "The negative control is central: default count-9 accountability bridges had 192 failure rows; weighted target bridges reduced this to 147, but three model spaces still failed.",
        BLUE_LIGHT,
        BLUE,
    )
    return canvas


def _discussion_page() -> Canvas:
    canvas = _page_base(12, "Discussion")
    _section_title(canvas, "6. Discussion And Limits", "The result is a typed repair ledger.", 54, 58)
    y = _paragraphs(
        canvas,
        64,
        122,
        486,
        [
            "The useful scientific object here is not a single cohesion vector. It is a typed repair ledger: what failed, which gate caught it, which repair was accepted, which plausible repair failed, and which claim boundary remains.",
            "The accountability result says local generated references can create off-manifold source pockets. The dissent result says limited bridge construction can create target-side geometry pockets even when the broad direction already scores the generated reference correctly. The negative control says these pockets are not interchangeable.",
            "This matters for activation-based social-construct work because a benchmark score alone would hide all three facts. A system could appear successful on a generated benchmark while failing clean controls, or appear broken on a generated residual while still encoding the intended procedural distinction under hand-authored controls. Conversely, adding more positive content is not always enough; bridge geometry itself can be the failure surface.",
        ],
        size=10.0,
        leading=13.4,
    )
    _subheading(canvas, "Limitations", 64, y + 10)
    _bullet_list(
        canvas,
        [
            "The experiments are text-benchmark activation diagnostics, not human studies.",
            "The results are layer -2 diagnostics over four model spaces, not a full architecture or layer sweep.",
            "The benchmark is about procedural-justice distinctions, not social cohesion as a complete construct.",
            "The accountability perturbation ladder is a local mechanism stress test over a known leaky generated reference.",
            "The repairs are diagnostic repairs; they do not establish causal steering, deployment safety, clinical outcomes, neural correlates, or real-world social effects.",
        ],
        74,
        y + 44,
        452,
    )
    return canvas


def _reproducibility_page() -> Canvas:
    canvas = _page_base(13, "Reproducibility")
    _section_title(canvas, "7. Reproducibility And Next Steps", "What to share and what to do next.", 54, 58)
    _subheading(canvas, "Durable research notes", 64, 124)
    y = _bullet_list(
        canvas,
        [
            "2026-06-09-strict-accountability-perturbation-ladder.md",
            "2026-06-09-dissent-perturbation-ladder.md",
            "2026-06-11-target-bridge-shortcut-repair.md",
            "2026-06-11-weighted-target-bridge-repair.md",
            "2026-06-11-weighted-bridge-preservation-audit.md",
            "2026-06-11-residual-taxonomy-repair-comparison.md",
        ],
        74,
        158,
        452,
        size=9.4,
        leading=12.4,
    )
    _subheading(canvas, "Live artifact roots", 64, y + 12)
    y = _bullet_list(
        canvas,
        [
            "/tmp/social_cohesion_accountability_strict_perturbation_ladder_20260609/",
            "/tmp/social_cohesion_target_bridge_shortcut_repair_20260611/",
            "/tmp/social_cohesion_bridge_weight_audit_20260611/",
            "/tmp/social_cohesion_weighted_bridge_preservation_20260611/",
        ],
        74,
        y + 46,
        452,
        size=8.9,
        leading=12.0,
        font="F3",
    )
    _subheading(canvas, "Next experiments", 64, y + 18)
    _numbered_list(
        canvas,
        [
            "Add a regeneration manifest mapping every paper table to committed scripts and expected external artifact paths.",
            "Run one narrow robustness check: either a fixed-availability third residual or a small layer sweep for the accepted weighted dissent repair.",
            "Convert this working draft into a formal paper with related work, figures, and a reproducibility appendix.",
        ],
        74,
        y + 52,
        452,
        size=9.6,
        leading=12.8,
    )
    _claim_boundary(canvas, 64, 610, 486)
    return canvas


def _paragraphs(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    paragraphs: Sequence[str],
    *,
    size: float = 9.8,
    leading: float = 13.0,
) -> float:
    current_y = y
    for paragraph in paragraphs:
        current_y = canvas.wrapped_text(
            paragraph,
            x,
            current_y,
            width,
            size=size,
            leading=leading,
        )
        current_y += 2
    return current_y


def _subheading(canvas: Canvas, text: str, x: float, y: float) -> None:
    canvas.text(text, x, y, size=13, font="F2", color=TEAL)


def _quote_box(canvas: Canvas, x: float, y: float, width: float, text: str) -> None:
    canvas.rect(x, y, width, 92, fill=Color(0.93, 0.97, 0.97), stroke=TEAL)
    canvas.rect(x, y, 8, 92, fill=TEAL)
    canvas.wrapped_text(text, x + 22, y + 20, width - 42, size=10.4, font="F2", color=INK)


def _bullet_list(
    canvas: Canvas,
    items: Sequence[str],
    x: float,
    y: float,
    width: float,
    *,
    size: float = 9.8,
    leading: float = 13.2,
    font: str = "F1",
) -> float:
    current_y = y
    for item in items:
        canvas.text("-", x, current_y, size=size, font="F2", color=TEAL)
        current_y = canvas.wrapped_text(
            item,
            x + 16,
            current_y,
            width - 16,
            size=size,
            font=font,
            leading=leading,
        )
        current_y += 1
    return current_y


def _numbered_list(
    canvas: Canvas,
    items: Sequence[str],
    x: float,
    y: float,
    width: float,
    *,
    size: float = 9.8,
    leading: float = 13.2,
) -> float:
    current_y = y
    for index, item in enumerate(items, start=1):
        canvas.text(f"{index}.", x, current_y, size=size, font="F2", color=TEAL)
        current_y = canvas.wrapped_text(
            item,
            x + 20,
            current_y,
            width - 20,
            size=size,
            leading=leading,
        )
        current_y += 1
    return current_y


def _simple_table(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    row_height: float,
    column_widths: Sequence[float],
) -> None:
    table_height = row_height * (len(rows) + 1)
    canvas.rect(x, y, width, table_height, fill=Color(0.99, 0.99, 0.96), stroke=LINE)
    canvas.rect(x, y, width, row_height, fill=Color(0.91, 0.95, 0.94), stroke=LINE)
    current_x = x
    for index, header in enumerate(headers):
        if index > 0:
            canvas.line(current_x, y, current_x, y + table_height, color=LINE)
        canvas.text(header, current_x + 8, y + 12, size=8.7, font="F2", color=TEAL)
        current_x += column_widths[index]
    for row_index, row in enumerate(rows):
        row_y = y + row_height * (row_index + 1)
        canvas.line(x, row_y, x + width, row_y, color=LINE)
        current_x = x
        for column_index, value in enumerate(row):
            canvas.wrapped_text(
                value,
                current_x + 8,
                row_y + 9,
                column_widths[column_index] - 16,
                size=8.3,
                leading=10.2,
            )
            current_x += column_widths[column_index]


def _page_base(page_number: int, label: str) -> Canvas:
    canvas = Canvas()
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=PAPER)
    canvas.rect(MARGIN, 34, PAGE_WIDTH - MARGIN * 2, PAGE_HEIGHT - 68, fill=PANEL, stroke=LINE)
    canvas.text(label.upper(), MARGIN + 12, 48, size=8.5, font="F2", color=MUTED)
    canvas.text(f"2026-06-11  |  Page {page_number}", PAGE_WIDTH - MARGIN - 12, 48, size=8.5, color=MUTED, align="right")
    canvas.line(MARGIN + 12, 66, PAGE_WIDTH - MARGIN - 12, 66, color=LINE)
    return canvas


def _section_title(canvas: Canvas, title: str, subtitle: str, x: float, y: float) -> None:
    canvas.text(title, x, y, size=22, font="F2")
    canvas.text(subtitle, x, y + 30, size=11.5, color=MUTED)


def _metric_card(
    canvas: Canvas,
    x: float,
    y: float,
    title: str,
    value: str,
    detail: str,
    color: Color,
) -> None:
    canvas.rect(x, y, 146, 108, fill=PANEL, stroke=color, stroke_width=1.2)
    canvas.rect(x, y, 146, 10, fill=color)
    canvas.text(title, x + 14, y + 24, size=9.5, font="F2", color=MUTED)
    canvas.text(value, x + 14, y + 45, size=30, font="F2", color=color)
    canvas.wrapped_text(detail, x + 14, y + 74, 116, size=8.2, color=MUTED, leading=10.0)


def _claim_boundary(canvas: Canvas, x: float, y: float, width: float) -> None:
    canvas.rect(x, y, width, 128, fill=Color(0.99, 0.94, 0.89), stroke=CORAL)
    canvas.text("Claim Boundary", x + 18, y + 20, size=16, font="F2", color=CORAL)
    text = (
        "This PDF claims only generated/hand-authored text-benchmark activation "
        "diagnostics. It does not claim human behavioral effects, neural "
        "alignment, clinical relevance, deployment readiness, or causal steering."
    )
    canvas.wrapped_text(text, x + 18, y + 50, width - 36, size=11, color=INK)


def _mini_taxonomy(canvas: Canvas, x: float, y: float) -> None:
    _node(canvas, x, y, "Accountability", "source pocket", TEAL)
    _node(canvas, x + 180, y, "Dissent", "bridge geometry", GOLD)
    _node(canvas, x + 360, y, "Withheld", "availability gaps", BLUE)
    canvas.line(x + 126, y + 34, x + 170, y + 34, color=LINE, width=1.4)
    canvas.line(x + 306, y + 34, x + 350, y + 34, color=LINE, width=1.4)


def _node(canvas: Canvas, x: float, y: float, title: str, subtitle: str, color: Color) -> None:
    canvas.rect(x, y, 126, 68, fill=Color(0.99, 0.99, 0.96), stroke=color)
    canvas.text(title, x + 12, y + 18, size=11, font="F2", color=color)
    canvas.text(subtitle, x + 12, y + 42, size=9, color=MUTED)


def _lane(
    canvas: Canvas,
    x: float,
    y: float,
    title: str,
    residual: str,
    failure: str,
    repair: str,
    color: Color,
) -> None:
    canvas.rect(x, y, 504, 150, fill=Color(0.99, 0.99, 0.96), stroke=color)
    canvas.rect(x, y, 12, 150, fill=color)
    canvas.text(title, x + 28, y + 20, size=16, font="F2", color=color)
    canvas.text(residual, x + 28, y + 48, size=10, font="F3", color=MUTED)
    canvas.wrapped_text(failure, x + 28, y + 74, 300, size=10, color=INK)
    canvas.rect(x + 350, y + 55, 128, 54, fill=Color(0.96, 0.98, 0.95), stroke=GREEN)
    canvas.wrapped_text(repair, x + 364, y + 70, 100, size=9.2, font="F2", color=GREEN)


def _ledger_row(canvas: Canvas, x: float, y: float, label: str, detail: str, color: Color) -> None:
    canvas.rect(x, y, 464, 42, fill=Color(0.99, 0.99, 0.96), stroke=LINE)
    canvas.rect(x, y, 76, 42, fill=color)
    canvas.text(label, x + 12, y + 16, size=11, font="F2", color=Color(1, 1, 1))
    canvas.text(detail, x + 94, y + 16, size=10.2, color=INK)


def _horizontal_bars(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    bars: Sequence[Bar],
    *,
    max_value: float,
) -> None:
    canvas.rect(x, y, width, height, fill=Color(0.99, 0.99, 0.96), stroke=LINE)
    row_height = height / len(bars)
    for index, bar in enumerate(bars):
        row_y = y + index * row_height + 18
        bar_width = (width - 220) * (bar.value / max_value)
        canvas.text(bar.label, x + 18, row_y, size=10.2, font="F2")
        canvas.rect(x + 210, row_y - 3, width - 250, 16, fill=Color(0.91, 0.92, 0.88))
        if bar_width > 0:
            canvas.rect(x + 210, row_y - 3, bar_width, 16, fill=bar.color)
        canvas.text(f"{bar.value:.0f}", x + width - 22, row_y, size=10, font="F2", align="right")


def _vertical_bars(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    bars: Sequence[Bar],
    *,
    max_value: float,
    label: str,
) -> None:
    canvas.rect(x, y, width, height, fill=Color(0.99, 0.99, 0.96), stroke=LINE)
    chart_x = x + 52
    chart_y = y + 38
    chart_w = width - 86
    chart_h = height - 80
    canvas.text(label, x + 18, y + 18, size=11, font="F2", color=MUTED)
    canvas.line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, color=LINE)
    slot = chart_w / len(bars)
    for index, bar in enumerate(bars):
        bar_h = chart_h * max(0.0, bar.value / max_value)
        bx = chart_x + index * slot + slot * 0.23
        by = chart_y + chart_h - bar_h
        canvas.rect(bx, by, slot * 0.54, bar_h, fill=bar.color)
        canvas.text(f"+{bar.value:.3f}", bx + slot * 0.27, by - 18, size=8.5, font="F2", color=bar.color, align="center")
        canvas.text(bar.label, bx + slot * 0.27, chart_y + chart_h + 18, size=8.5, color=MUTED, align="center")


def _grouped_bars(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    bars: Sequence[GroupedBar],
    *,
    max_value: float,
) -> None:
    canvas.rect(x, y, width, height, fill=Color(0.99, 0.99, 0.96), stroke=LINE)
    canvas.text("Fresh-source failure counts", x + 18, y + 18, size=11, font="F2", color=MUTED)
    _legend(canvas, x + width - 210, y + 18)
    chart_x = x + 52
    chart_y = y + 48
    chart_w = width - 90
    chart_h = height - 92
    slot = chart_w / len(bars)
    canvas.line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, color=LINE)
    for index, bar in enumerate(bars):
        bx = chart_x + index * slot + slot * 0.18
        _one_failure_bar(canvas, bx, chart_y, chart_h, slot * 0.24, bar.first, max_value, CORAL)
        _one_failure_bar(canvas, bx + slot * 0.30, chart_y, chart_h, slot * 0.24, bar.second, max_value, GOLD)
        canvas.text(bar.label, bx + slot * 0.28, chart_y + chart_h + 18, size=8.5, color=MUTED, align="center")


def _one_failure_bar(
    canvas: Canvas,
    x: float,
    y: float,
    chart_height: float,
    width: float,
    value: float,
    max_value: float,
    color: Color,
) -> None:
    bar_height = chart_height * (value / max_value)
    canvas.rect(x, y + chart_height - bar_height, width, bar_height, fill=color)
    canvas.text(f"{value:.0f}", x + width / 2, y + chart_height - bar_height - 16, size=8.3, font="F2", color=color, align="center")


def _legend(canvas: Canvas, x: float, y: float) -> None:
    canvas.rect(x, y, 10, 10, fill=CORAL)
    canvas.text("Default", x + 16, y + 1, size=8.2, color=MUTED)
    canvas.rect(x + 82, y, 10, 10, fill=GOLD)
    canvas.text("Weighted", x + 98, y + 1, size=8.2, color=MUTED)


def _failure_total(canvas: Canvas, x: float, y: float, label: str, value: int, color: Color) -> None:
    canvas.rect(x, y, 210, 90, fill=Color(0.99, 0.99, 0.96), stroke=color)
    canvas.text(label, x + 18, y + 20, size=11, font="F2", color=MUTED)
    canvas.text(str(value), x + 18, y + 46, size=28, font="F2", color=color)
    canvas.text("constructed failure rows", x + 78, y + 58, size=9, color=MUTED)


def _bridge_recipe(canvas: Canvas, x: float, y: float) -> None:
    labels = [
        ("target pairs", "9", TEAL),
        ("target reps", "1:3", GREEN),
        ("source reps", "1:1", BLUE),
    ]
    for index, (label, value, color) in enumerate(labels):
        bx = x + index * 150
        canvas.rect(bx, y, 122, 74, fill=Color(0.99, 0.99, 0.96), stroke=color)
        canvas.text(value, bx + 61, y + 18, size=24, font="F2", color=color, align="center")
        canvas.text(label, bx + 61, y + 52, size=9.2, color=MUTED, align="center")


def _callout(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    text: str,
    fill: Color,
    stroke: Color,
) -> None:
    canvas.rect(x, y, width, 98, fill=fill, stroke=stroke)
    canvas.wrapped_text(text, x + 18, y + 22, width - 36, size=10.5, font="F2", color=INK)


def _footer_note(canvas: Canvas, text: str) -> None:
    canvas.rect(64, 676, 496, 52, fill=Color(0.94, 0.97, 0.95), stroke=GREEN)
    canvas.wrapped_text(text, 82, 692, 460, size=9.6, font="F2", color=GREEN)


def _write_pdf(pages: Sequence[Canvas], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    objects = _pdf_objects(pages)
    payload = _serialize_pdf(objects)
    output.write_bytes(payload)


def _pdf_objects(pages: Sequence[Canvas]) -> list[bytes]:
    page_count = len(pages)
    page_ids = [6 + index * 2 for index in range(page_count)]
    content_ids = [7 + index * 2 for index in range(page_count)]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    for page_id, content_id, page in zip(page_ids, content_ids, pages, strict=True):
        del page_id
        stream = page.stream()
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH:.0f} {PAGE_HEIGHT:.0f}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream")
    return objects


def _serialize_pdf(objects: Sequence[bytes]) -> bytes:
    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_position = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f\n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n\n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode("ascii")
    return pdf


def _fill(color: Color) -> str:
    return f"{color.red:.3f} {color.green:.3f} {color.blue:.3f} rg"


def _stroke(color: Color) -> str:
    return f"{color.red:.3f} {color.green:.3f} {color.blue:.3f} RG"


def _paint(fill: Color | None, stroke: Color | None) -> str:
    if fill is not None and stroke is not None:
        return "B"
    if fill is not None:
        return "f"
    return "S"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _measure(value: str, size: float, font: str) -> float:
    factor = 0.56 if font == "F2" else 0.51
    if font == "F3":
        factor = 0.60
    return len(value) * size * factor


def _aligned_x(x: float, text_width: float, align: str) -> float:
    if align == "center":
        return x - text_width / 2
    if align == "right":
        return x - text_width
    return x


if __name__ == "__main__":
    raise SystemExit(main())
