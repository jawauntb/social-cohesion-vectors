"""Run a tiny GPT-2 SAE probe on pseudo-cohesion activation prompts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

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

    model = HookedTransformer.from_pretrained(model_id, device="cpu")
    sae, cfg, _ = SAE.from_pretrained(release, sae_id, device="cpu")
    _, cache = model.run_with_cache(texts, names_filter=[sae_id], prepend_bos=True)
    activations = cache[sae_id].mean(dim=1)
    with torch.no_grad():
        features = sae.encode_standard(activations)

    positive_mask = torch.tensor([label == "positive" for label in labels])
    negative_mask = torch.tensor([label == "negative" for label in labels])
    positive = features[positive_mask]
    negative = features[negative_mask]
    difference = positive.mean(dim=0) - negative.mean(dim=0)
    values, indices = torch.topk(difference.abs(), k=top_k)

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
            }
            for value, index in zip(values, indices, strict=True)
        ],
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
        "## Top Differentiating Features",
        "",
        "| Feature | Pos - neg | Pos mean | Neg mean | Abs diff |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["top_features"]:
        item = _mapping(row)
        lines.append(
            "| "
            f"{item['feature']} | "
            f"{float(item['difference_positive_minus_negative']):+.4f} | "
            f"{float(item['positive_mean']):.4f} | "
            f"{float(item['negative_mean']):.4f} | "
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


if __name__ == "__main__":
    raise SystemExit(main())
