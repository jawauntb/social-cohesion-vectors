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
        _taxonomy_page(),
        _repair_funnel_page(),
        _dissent_page(),
        _negative_control_page(),
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


def _taxonomy_page() -> Canvas:
    canvas = _page_base(2, "Residual taxonomy")
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
    canvas = _page_base(3, "Repair funnel")
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
    canvas = _page_base(4, "Dissent margins")
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
    canvas = _page_base(5, "Negative control")
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
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
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
