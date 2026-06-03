"""Run a tiny GPT-2 SAE probe on pseudo-cohesion activation prompts."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_cohesion_vectors.config import get_config
from social_cohesion_vectors.datasets import read_jsonl

DEFAULT_RELEASE = "gpt2-small-resid-post-v5-32k"
DEFAULT_SAE_ID = "blocks.11.hook_resid_post"
DEFAULT_MODEL_ID = "gpt2-small"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_probe(
        prompts_path=args.prompts,
        release=args.release,
        sae_id=args.sae_id,
        model_id=args.model_id,
        top_k=args.top_k,
    )
    write_reports(report, json_path=args.json_output, markdown_path=args.markdown_output)
    print(
        "gpt2 sae pseudo probe: "
        f"prompts={report['n_prompts']} features={len(report['top_features'])}"
    )
    print(f"wrote {args.json_output}")
    print(f"wrote {args.markdown_output}")
    return 0


def run_probe(
    *,
    prompts_path: str | Path,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    model_id: str = DEFAULT_MODEL_ID,
    top_k: int = 20,
) -> dict[str, Any]:
    """Load GPT-2/SAE, encode prompt activations, and compare poles."""

    records = read_jsonl(prompts_path)
    texts = [str(record["text"]) for record in records]
    labels = [str(record["label"]) for record in records]
    pair_ids = [str(record["pair_id"]) for record in records]

    ml = _load_optional_ml()
    torch = ml["torch"]
    model = ml["HookedTransformer"].from_pretrained(model_id, device="cpu")
    sae, cfg, _ = ml["SAE"].from_pretrained(release, sae_id, device="cpu")
    _, cache = model.run_with_cache(texts, names_filter=[sae_id], prepend_bos=True)
    activations = cache[sae_id].mean(dim=1)
    with torch.no_grad():
        features = sae.encode_standard(activations)

    positive_mask = torch.tensor([label == "positive" for label in labels])
    negative_mask = torch.tensor([label == "negative" for label in labels])
    positive = features[positive_mask]
    negative = features[negative_mask]
    difference = positive.mean(dim=0) - negative.mean(dim=0)
    values, indices = torch.topk(difference.abs(), k=min(top_k, features.shape[-1]))

    return {
        "experiment": "gpt2_sae_pseudo_cohesion_probe",
        "prompts_path": str(prompts_path),
        "model_id": model_id,
        "release": release,
        "sae_id": sae_id,
        "d_sae": int(cfg.get("d_sae", features.shape[-1])),
        "n_prompts": len(records),
        "n_positive": int(positive_mask.sum().item()),
        "n_negative": int(negative_mask.sum().item()),
        "leave_one_pair_out": {
            "residual": evaluate_leave_one_pair_out(
                activations,
                labels=labels,
                pair_ids=pair_ids,
            ),
            "sae_features": evaluate_leave_one_pair_out(
                features,
                labels=labels,
                pair_ids=pair_ids,
            ),
        },
        "top_features": [
            {
                "feature": int(index),
                "absolute_difference": round(float(value), 6),
                "difference_positive_minus_negative": round(
                    float(difference[index]),
                    6,
                ),
                "positive_mean": round(float(positive[:, index].mean()), 6),
                "negative_mean": round(float(negative[:, index].mean()), 6),
                "positive_activation_rate": round(
                    float((positive[:, index] > 0).float().mean()),
                    6,
                ),
                "negative_activation_rate": round(
                    float((negative[:, index] > 0).float().mean()),
                    6,
                ),
            }
            for value, index in zip(values, indices, strict=True)
        ],
    }


def evaluate_leave_one_pair_out(
    activations: Any,
    *,
    labels: Sequence[str],
    pair_ids: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a positive-minus-negative direction by held-out pair."""

    rows: list[dict[str, Any]] = []
    for pair_id in sorted(set(pair_ids)):
        test_indices = [index for index, value in enumerate(pair_ids) if value == pair_id]
        if len(test_indices) != 2:
            continue
        train_indices = [
            index for index, value in enumerate(pair_ids) if value != pair_id
        ]
        train_labels = [labels[index] for index in train_indices]
        positive_train_indices = [
            index
            for index, label in zip(train_indices, train_labels, strict=True)
            if label == "positive"
        ]
        negative_train_indices = [
            index
            for index, label in zip(train_indices, train_labels, strict=True)
            if label == "negative"
        ]
        if not positive_train_indices or not negative_train_indices:
            continue

        direction = (
            activations[positive_train_indices].mean(dim=0)
            - activations[negative_train_indices].mean(dim=0)
        )
        norm = float(direction.norm().item())
        if norm == 0.0:
            continue
        direction = direction / norm
        scores = {
            labels[index]: float((activations[index] @ direction).item())
            for index in test_indices
        }
        if "positive" not in scores or "negative" not in scores:
            continue
        margin = scores["positive"] - scores["negative"]
        rows.append(
            {
                "pair_id": pair_id,
                "margin": round(margin, 6),
                "correct": margin > 0.0,
                "positive_projection": round(scores["positive"], 6),
                "negative_projection": round(scores["negative"], 6),
            }
        )

    failures = [row for row in rows if not row["correct"]]
    return {
        "pairs": len(rows),
        "accuracy": round((len(rows) - len(failures)) / len(rows), 6)
        if rows
        else 0.0,
        "mean_margin": round(
            sum(float(row["margin"]) for row in rows) / len(rows),
            6,
        )
        if rows
        else 0.0,
        "failures": failures,
    }


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write JSON and markdown probe reports."""

    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise markdown report."""

    lines = [
        "# GPT-2 SAE Pseudo-Cohesion Probe",
        "",
        f"- Model: `{report['model_id']}`",
        f"- SAE release: `{report['release']}`",
        f"- SAE id: `{report['sae_id']}`",
        f"- SAE width: {report['d_sae']}",
        f"- Prompts: {report['n_prompts']}",
        f"- Positive prompts: {report['n_positive']}",
        f"- Negative prompts: {report['n_negative']}",
        "",
        "## Leave-One-Pair-Out",
        "",
        "| Representation | Pairs | Accuracy | Mean margin | Failures |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    loo = _mapping(report.get("leave_one_pair_out"))
    for key, label in [
        ("residual", "Residual stream"),
        ("sae_features", "SAE features"),
    ]:
        result = _mapping(loo.get(key))
        failures = result.get("failures", [])
        failure_count = len(failures) if isinstance(failures, list) else 0
        lines.append(
            "| "
            f"{label} | "
            f"{int(result.get('pairs', 0))} | "
            f"{float(result.get('accuracy', 0.0)):.3f} | "
            f"{float(result.get('mean_margin', 0.0)):+.4f} | "
            f"{failure_count} |"
        )

    lines.extend(
        [
            "",
            "## Top Differentiating Features",
            "",
            (
                "| Feature | Pos - neg | Pos mean | Neg mean | "
                "Pos rate | Neg rate | Abs diff |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["top_features"]:
        item = _mapping(row)
        lines.append(
            "| "
            f"{item['feature']} | "
            f"{float(item['difference_positive_minus_negative']):+.4f} | "
            f"{float(item['positive_mean']):.4f} | "
            f"{float(item['negative_mean']):.4f} | "
            f"{float(item['positive_activation_rate']):.3f} | "
            f"{float(item['negative_activation_rate']):.3f} | "
            f"{float(item['absolute_difference']):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=paths.training / "pseudo_cohesion_activation_prompts.jsonl",
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=paths.reports / "gpt2_sae_pseudo_probe.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=paths.reports / "gpt2_sae_pseudo_probe.md",
    )
    return parser.parse_args(argv)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_optional_ml() -> dict[str, Any]:
    """Import optional GPT-2/SAE dependencies only for model execution."""

    missing: list[str] = []
    modules: dict[str, Any] = {}
    for module_name in ("torch", "sae_lens", "transformer_lens"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        joined = ", ".join(missing)
        raise ImportError(
            "GPT-2 SAE pseudo probe requires optional ML dependencies missing from "
            f"this environment: {joined}. Install the project's ml and sae extras "
            "to run model-backed probing."
        )
    return {
        "torch": modules["torch"],
        "SAE": modules["sae_lens"].SAE,
        "HookedTransformer": modules["transformer_lens"].HookedTransformer,
    }


if __name__ == "__main__":
    raise SystemExit(main())
