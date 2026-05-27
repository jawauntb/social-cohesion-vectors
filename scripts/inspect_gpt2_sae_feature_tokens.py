"""Inspect GPT-2 SAE candidate features at token and example level."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import read_jsonl

DEFAULT_RELEASE = "gpt2-small-resid-post-v5-32k"
DEFAULT_SAE_ID = "blocks.11.hook_resid_post"
DEFAULT_MODEL_ID = "gpt2-small"
DEFAULT_FEATURES = (3056, 24555, 28005, 20249, 11999, 11737, 703)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_inspection(
        prompts_path=args.prompts,
        features=args.features,
        release=args.release,
        sae_id=args.sae_id,
        model_id=args.model_id,
        top_k_tokens=args.top_k_tokens,
        top_k_examples=args.top_k_examples,
        top_k_pairs=args.top_k_pairs,
        context_window=args.context_window,
        include_bos=args.include_bos,
    )
    write_reports(report, json_path=args.json_output, markdown_path=args.markdown_output)
    print(
        "gpt2 sae token inspection: "
        f"prompts={report['n_prompts']} features={len(report['features'])}"
    )
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


def run_inspection(
    *,
    prompts_path: str | Path,
    features: Sequence[int] = DEFAULT_FEATURES,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    top_k_tokens: int = 12,
    top_k_examples: int = 6,
    top_k_pairs: int = 6,
    context_window: int = 4,
    include_bos: bool = False,
) -> dict[str, Any]:
    """Run token-level feature inspection for selected SAE features."""

    records = read_jsonl(prompts_path)
    feature_ids = [int(feature) for feature in features]
    if not feature_ids:
        raise ValueError("At least one SAE feature id is required.")

    model = HookedTransformer.from_pretrained(model_id, device="cpu")
    sae, cfg, _ = SAE.from_pretrained(release, sae_id, device="cpu")

    token_rows: dict[int, list[dict[str, Any]]] = {
        feature: [] for feature in feature_ids
    }
    example_rows: dict[int, list[dict[str, Any]]] = {
        feature: [] for feature in feature_ids
    }
    feature_index = torch.tensor(feature_ids, dtype=torch.long)

    for record in records:
        text = str(record["text"])
        tokens = cast(list[str], model.to_str_tokens(text, prepend_bos=True))
        _, cache = model.run_with_cache(text, names_filter=[sae_id], prepend_bos=True)
        residual = cache[sae_id][0]
        with torch.no_grad():
            activations = sae.encode_standard(residual).index_select(
                dim=1,
                index=feature_index,
            )

        start_index = 0 if include_bos else _first_content_token_index(tokens)
        for column_index, feature in enumerate(feature_ids):
            values = activations[:, column_index]
            content_values = values[start_index:]
            example_rows[feature].append(
                _example_row(
                    record=record,
                    feature=feature,
                    values=content_values,
                    tokens=tokens,
                    start_index=start_index,
                    text=text,
                )
            )
            for token_index in range(start_index, len(tokens)):
                token_rows[feature].append(
                    _token_row(
                        record=record,
                        feature=feature,
                        activation=float(values[token_index].item()),
                        token_index=token_index,
                        token=tokens[token_index],
                        tokens=tokens,
                        context_window=context_window,
                    )
                )

    return {
        "experiment": "gpt2_sae_token_feature_inspection",
        "prompts_path": str(prompts_path),
        "model_id": model_id,
        "release": release,
        "sae_id": sae_id,
        "d_sae": int(cfg.get("d_sae", 0)),
        "features": feature_ids,
        "n_prompts": len(records),
        "include_bos": include_bos,
        "top_k_tokens": top_k_tokens,
        "top_k_examples": top_k_examples,
        "top_k_pairs": top_k_pairs,
        "feature_reports": [
            summarize_feature(
                feature=feature,
                token_rows=token_rows[feature],
                example_rows=example_rows[feature],
                top_k_tokens=top_k_tokens,
                top_k_examples=top_k_examples,
                top_k_pairs=top_k_pairs,
            )
            for feature in feature_ids
        ],
    }


def summarize_feature(
    *,
    feature: int,
    token_rows: Sequence[Mapping[str, Any]],
    example_rows: Sequence[Mapping[str, Any]],
    top_k_tokens: int,
    top_k_examples: int,
    top_k_pairs: int,
) -> dict[str, Any]:
    """Summarize token, example, and pair-level behavior for one feature."""

    positive_tokens = [row for row in token_rows if row.get("label") == "positive"]
    negative_tokens = [row for row in token_rows if row.get("label") == "negative"]
    positive_examples = [
        row for row in example_rows if row.get("label") == "positive"
    ]
    negative_examples = [
        row for row in example_rows if row.get("label") == "negative"
    ]
    pair_deltas = pair_delta_rows(example_rows)
    active_tokens = [
        row for row in token_rows if float(row.get("activation", 0.0)) > 0.0
    ]
    active_positive_examples = [
        row
        for row in positive_examples
        if float(row.get("max_activation", 0.0)) > 0.0
    ]
    active_negative_examples = [
        row
        for row in negative_examples
        if float(row.get("max_activation", 0.0)) > 0.0
    ]
    summary = {
        "feature": feature,
        "token_mean_positive": _mean_float(
            float(row["activation"]) for row in positive_tokens
        ),
        "token_mean_negative": _mean_float(
            float(row["activation"]) for row in negative_tokens
        ),
        "token_rate_positive": _activation_rate(positive_tokens),
        "token_rate_negative": _activation_rate(negative_tokens),
        "example_max_mean_positive": _mean_float(
            float(row["max_activation"]) for row in positive_examples
        ),
        "example_max_mean_negative": _mean_float(
            float(row["max_activation"]) for row in negative_examples
        ),
    }
    summary["token_mean_pos_minus_neg"] = round(
        float(summary["token_mean_positive"])
        - float(summary["token_mean_negative"]),
        6,
    )
    summary["example_max_pos_minus_neg"] = round(
        float(summary["example_max_mean_positive"])
        - float(summary["example_max_mean_negative"]),
        6,
    )
    summary["direction_hint"] = direction_hint(
        float(summary["token_mean_pos_minus_neg"])
    )

    return {
        "feature": feature,
        "summary": summary,
        "top_tokens": top_rows(active_tokens, key="activation", limit=top_k_tokens),
        "top_positive_examples": top_rows(
            active_positive_examples,
            key="max_activation",
            limit=top_k_examples,
        ),
        "top_negative_examples": top_rows(
            active_negative_examples,
            key="max_activation",
            limit=top_k_examples,
        ),
        "largest_genuine_minus_pseudo_pairs": top_rows(
            pair_deltas,
            key="mean_delta_positive_minus_negative",
            limit=top_k_pairs,
        ),
        "largest_pseudo_minus_genuine_pairs": top_rows(
            pair_deltas,
            key="mean_delta_negative_minus_positive",
            limit=top_k_pairs,
        ),
    }


def pair_delta_rows(example_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare positive and negative example means within each pair."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in example_rows:
        grouped.setdefault(str(row["pair_id"]), {})[str(row["label"])] = row

    rows: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        positive = pair.get("positive")
        negative = pair.get("negative")
        if positive is None or negative is None:
            continue
        mean_delta = float(positive["mean_activation"]) - float(
            negative["mean_activation"]
        )
        max_delta = float(positive["max_activation"]) - float(
            negative["max_activation"]
        )
        rows.append(
            {
                "pair_id": pair_id,
                "positive_sample_id": positive["sample_id"],
                "negative_sample_id": negative["sample_id"],
                "mean_delta_positive_minus_negative": round(mean_delta, 6),
                "mean_delta_negative_minus_positive": round(-mean_delta, 6),
                "max_delta_positive_minus_negative": round(max_delta, 6),
                "positive_mean": positive["mean_activation"],
                "negative_mean": negative["mean_activation"],
                "positive_max": positive["max_activation"],
                "negative_max": negative["max_activation"],
            }
        )
    return rows


def top_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return rows sorted descending by a numeric key."""

    ordered = sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=True)
    return [dict(row) for row in ordered[:limit]]


def direction_hint(delta: float) -> str:
    """Name which side has higher aggregate feature activation."""

    if delta > 0:
        return "genuine_higher"
    if delta < 0:
        return "pseudo_higher"
    return "tied"


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown inspection reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render token-level feature inspection as markdown."""

    lines = [
        "# GPT-2 SAE Token Feature Inspection",
        "",
        f"- Model: `{report['model_id']}`",
        f"- SAE release: `{report['release']}`",
        f"- SAE id: `{report['sae_id']}`",
        f"- Prompts: {report['n_prompts']}",
        f"- Features: {', '.join(str(feature) for feature in report['features'])}",
        f"- Includes BOS token: {_yes_no(bool(report['include_bos']))}",
        "",
        "## Feature Summary",
        "",
        (
            "| Feature | Direction hint | Token pos-neg | Pos token mean | "
            "Neg token mean | Pos rate | Neg rate | Example max pos-neg |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for raw_feature_report in _sequence(report.get("feature_reports")):
        feature_report = _mapping(raw_feature_report)
        summary = _mapping(feature_report.get("summary"))
        lines.append(
            "| "
            f"{summary.get('feature', '')} | "
            f"{summary.get('direction_hint', '')} | "
            f"{float(summary.get('token_mean_pos_minus_neg', 0.0)):+.4f} | "
            f"{float(summary.get('token_mean_positive', 0.0)):.4f} | "
            f"{float(summary.get('token_mean_negative', 0.0)):.4f} | "
            f"{float(summary.get('token_rate_positive', 0.0)):.3f} | "
            f"{float(summary.get('token_rate_negative', 0.0)):.3f} | "
            f"{float(summary.get('example_max_pos_minus_neg', 0.0)):+.4f} |"
        )

    for raw_feature_report in _sequence(report.get("feature_reports")):
        feature_report = _mapping(raw_feature_report)
        summary = _mapping(feature_report.get("summary"))
        feature = int(summary.get("feature", 0))
        lines.extend(
            [
                "",
                f"## Feature {feature}",
                "",
                (
                    f"Direction hint: `{summary.get('direction_hint', '')}` "
                    f"(token pos-neg "
                    f"{float(summary.get('token_mean_pos_minus_neg', 0.0)):+.4f})."
                ),
                "",
                "### Top Token Activations",
                "",
                "| Activation | Label | Pair | Token | Context |",
                "| ---: | --- | --- | --- | --- |",
            ]
        )
        for row in _sequence(feature_report.get("top_tokens")):
            item = _mapping(row)
            lines.append(
                "| "
                f"{float(item.get('activation', 0.0)):.4f} | "
                f"{_cell(str(item.get('label', '')))} | "
                f"{_cell(short_pair_id(str(item.get('pair_id', ''))))} | "
                f"`{_cell(str(item.get('token', '')))}` | "
                f"{_cell(str(item.get('context', '')))} |"
            )

        lines.extend(
            [
                "",
                "### Top Genuine Examples",
                "",
                "| Max | Mean | Pair | Token | Excerpt |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in _sequence(feature_report.get("top_positive_examples")):
            lines.append(_example_markdown_row(_mapping(row)))

        lines.extend(
            [
                "",
                "### Top Pseudo Examples",
                "",
                "| Max | Mean | Pair | Token | Excerpt |",
                "| ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in _sequence(feature_report.get("top_negative_examples")):
            lines.append(_example_markdown_row(_mapping(row)))

        lines.extend(
            [
                "",
                "### Pair Deltas",
                "",
                "Largest genuine-minus-pseudo mean deltas:",
                "",
                "| Delta | Pair | Positive mean | Negative mean |",
                "| ---: | --- | ---: | ---: |",
            ]
        )
        for row in _sequence(
            feature_report.get("largest_genuine_minus_pseudo_pairs")
        ):
            item = _mapping(row)
            lines.append(
                "| "
                f"{float(item.get('mean_delta_positive_minus_negative', 0.0)):+.4f} | "
                f"{_cell(short_pair_id(str(item.get('pair_id', ''))))} | "
                f"{float(item.get('positive_mean', 0.0)):.4f} | "
                f"{float(item.get('negative_mean', 0.0)):.4f} |"
            )

        lines.extend(
            [
                "",
                "Largest pseudo-minus-genuine mean deltas:",
                "",
                "| Delta | Pair | Positive mean | Negative mean |",
                "| ---: | --- | ---: | ---: |",
            ]
        )
        for row in _sequence(
            feature_report.get("largest_pseudo_minus_genuine_pairs")
        ):
            item = _mapping(row)
            lines.append(
                "| "
                f"{float(item.get('mean_delta_negative_minus_positive', 0.0)):+.4f} | "
                f"{_cell(short_pair_id(str(item.get('pair_id', ''))))} | "
                f"{float(item.get('positive_mean', 0.0)):.4f} | "
                f"{float(item.get('negative_mean', 0.0)):.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def short_pair_id(pair_id: str) -> str:
    """Remove the pseudo-cohesion namespace in reports."""

    return pair_id.removeprefix("pseudo-cohesion::")


def _example_markdown_row(row: Mapping[str, Any]) -> str:
    return (
        "| "
        f"{float(row.get('max_activation', 0.0)):.4f} | "
        f"{float(row.get('mean_activation', 0.0)):.4f} | "
        f"{_cell(short_pair_id(str(row.get('pair_id', ''))))} | "
        f"`{_cell(str(row.get('max_token', '')))}` | "
        f"{_cell(str(row.get('text_excerpt', '')))} |"
    )


def _example_row(
    *,
    record: Mapping[str, Any],
    feature: int,
    values: torch.Tensor,
    tokens: Sequence[str],
    start_index: int,
    text: str,
) -> dict[str, Any]:
    max_value, max_index = values.max(dim=0)
    token_index = start_index + int(max_index.item())
    mean_value = values.mean()
    active_rate = (values > 0).float().mean()
    return {
        "feature": feature,
        "sample_id": str(record["sample_id"]),
        "pair_id": str(record["pair_id"]),
        "label": str(record["label"]),
        "target_score": float(record["target_score"]),
        "mean_activation": round(float(mean_value.item()), 6),
        "max_activation": round(float(max_value.item()), 6),
        "activation_rate": round(float(active_rate.item()), 6),
        "max_token_index": token_index,
        "max_token": _clean_token(tokens[token_index]),
        "text_excerpt": _excerpt(text),
    }


def _token_row(
    *,
    record: Mapping[str, Any],
    feature: int,
    activation: float,
    token_index: int,
    token: str,
    tokens: Sequence[str],
    context_window: int,
) -> dict[str, Any]:
    return {
        "feature": feature,
        "sample_id": str(record["sample_id"]),
        "pair_id": str(record["pair_id"]),
        "label": str(record["label"]),
        "target_score": float(record["target_score"]),
        "activation": round(activation, 6),
        "token_index": token_index,
        "token": _clean_token(token),
        "context": token_context(tokens, token_index, window=context_window),
    }


def _first_content_token_index(tokens: Sequence[str]) -> int:
    if tokens and tokens[0] == "<|endoftext|>":
        return 1
    return 0


def token_context(tokens: Sequence[str], token_index: int, *, window: int) -> str:
    """Return a compact token context around an activation."""

    start = max(token_index - window, 0)
    end = min(token_index + window + 1, len(tokens))
    rendered = []
    for index, token in enumerate(tokens[start:end], start=start):
        cleaned = _clean_token(token)
        if index == token_index:
            rendered.append(f"[{cleaned}]")
        else:
            rendered.append(cleaned)
    return " ".join(rendered)


def _activation_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    active = sum(1 for row in rows if float(row["activation"]) > 0.0)
    return round(active / len(rows), 6)


def _mean_float(values: Sequence[float] | Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(float(value) for value in collected) / len(collected), 6)


def _clean_token(token: str) -> str:
    cleaned = token.replace("\n", "\\n").replace("\t", "\\t")
    return cleaned if cleaned else "<empty>"


def _excerpt(text: str, *, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else []


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=paths.training / "pseudo_cohesion_activation_prompts.jsonl",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        default=list(DEFAULT_FEATURES),
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--top-k-tokens", type=int, default=12)
    parser.add_argument("--top-k-examples", type=int, default=6)
    parser.add_argument("--top-k-pairs", type=int, default=6)
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--include-bos", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "gpt2_sae_token_feature_inspection.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "gpt2_sae_token_feature_inspection.md",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
