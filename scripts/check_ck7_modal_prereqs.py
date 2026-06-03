"""Check local prerequisites for a CK-7 Modal cocktail candidate run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

DEFAULT_GUARDRAIL_AXES = (
    "principled_respect_vs_sycophancy",
    "truth_vs_deception",
    "manipulation_resistance_vs_persuasion_capture",
    "privacy_exit_vs_surveillance_lock_in",
)


@dataclass(frozen=True)
class CheckResult:
    """One readiness check with a machine-readable status."""

    name: str
    status: str
    detail: str


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    artifact_overrides = _parse_artifact_overrides(args.artifact)
    artifacts = (
        {}
        if args.only_artifact_overrides
        else _default_required_artifacts(
            repo_root=repo_root,
            direction_root=args.direction_root,
            ck1_direction=args.ck1_direction,
        )
    )
    artifacts.update(artifact_overrides)

    results = [
        *_script_checks(repo_root),
        *_environment_checks(repo_root),
        *_artifact_checks(
            artifacts,
            expected_hidden_size=args.expected_hidden_size,
        ),
    ]
    blocked = [result for result in results if result.status == "BLOCKED"]
    payload = {
        "experiment": "ck7_modal_prereq_check",
        "description": (
            "Local, no-network prerequisite check for CK-7 boundary-preserving "
            "activation-cocktail candidate runs."
        ),
        "expected_hidden_size": args.expected_hidden_size,
        "can_run_modal_now": not blocked,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "checks": [result.__dict__ for result in results],
    }

    _print_report(payload)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 1 if args.fail_on_blocker and blocked else 0


def _default_required_artifacts(
    *,
    repo_root: Path,
    direction_root: Path | None,
    ck1_direction: Path | None,
) -> dict[str, Path]:
    root = direction_root or repo_root / "data" / "models" / "vectors" / "ck45_guardrails"
    artifacts = {
        "ck1_boundary": ck1_direction
        or repo_root / "data" / "models" / "vectors" / "ck1_boundary_l2_direction.npz",
    }
    artifacts.update(
        {
            axis: root / f"{axis}_direction.npz"
            for axis in DEFAULT_GUARDRAIL_AXES
        }
    )
    return artifacts


def _script_checks(repo_root: Path) -> list[CheckResult]:
    required = (
        "scripts/run_ck3_modal_cocktail.py",
        "scripts/run_ck4_scheduled_modal_cocktail.py",
        "scripts/export_ck45_guardrail_vector_grid.py",
        "src/social_cohesion_vectors/modal_app/functions/activation_steering.py",
    )
    return [
        CheckResult(
            name=f"path:{path}",
            status="OK" if (repo_root / path).exists() else "BLOCKED",
            detail=str(repo_root / path),
        )
        for path in required
    ]


def _environment_checks(repo_root: Path) -> list[CheckResult]:
    env_keys = _env_keys(repo_root / ".env")
    hf_present = bool(
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or {"HUGGINGFACE_TOKEN", "HF_TOKEN"}.intersection(env_keys)
    )
    modal_markers = _modal_credential_markers()
    checks = [
        CheckResult(
            "uv",
            "OK" if shutil.which("uv") else "BLOCKED",
            shutil.which("uv") or "uv executable not found",
        ),
        CheckResult(
            "python-package:social_cohesion_vectors",
            "OK"
            if importlib.util.find_spec("social_cohesion_vectors") is not None
            else "BLOCKED",
            "importable in current Python environment",
        ),
        CheckResult(
            "python-package:modal",
            "OK" if importlib.util.find_spec("modal") is not None else "BLOCKED",
            "Modal package import check only; no remote call made",
        ),
        CheckResult(
            "modal-credentials-marker",
            "OK" if modal_markers else "WARN",
            ", ".join(str(path) for path in modal_markers)
            if modal_markers
            else "no local Modal credential marker found",
        ),
        CheckResult(
            "huggingface-token-marker",
            "OK" if hf_present else "WARN",
            "HF token key present in environment or .env"
            if hf_present
            else "no HUGGINGFACE_TOKEN or HF_TOKEN marker found",
        ),
    ]
    return checks


def _artifact_checks(
    artifacts: dict[str, Path],
    *,
    expected_hidden_size: int,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, path in sorted(artifacts.items()):
        results.extend(
            _direction_artifact_checks(
                name,
                path,
                expected_hidden_size=expected_hidden_size,
            )
        )
    return results


def _direction_artifact_checks(
    name: str,
    path: Path,
    *,
    expected_hidden_size: int,
) -> list[CheckResult]:
    if not path.exists():
        return [
            CheckResult(
                f"artifact:{name}:exists",
                "BLOCKED",
                f"missing {path}",
            )
        ]
    try:
        with np.load(path, allow_pickle=False) as data:
            if "direction" not in data.files:
                return [
                    CheckResult(
                        f"artifact:{name}:direction-key",
                        "BLOCKED",
                        f"{path} lacks a 'direction' array",
                    )
                ]
            direction = np.asarray(data["direction"], dtype=np.float32)
    except (OSError, ValueError, KeyError, BadZipFile, EOFError) as exc:
        return [
            CheckResult(
                f"artifact:{name}:load",
                "BLOCKED",
                f"{type(exc).__name__}: {exc}",
            )
        ]

    results = [
        CheckResult(
            f"artifact:{name}:exists",
            "OK",
            str(path),
        )
    ]
    if direction.ndim != 1:
        results.append(
            CheckResult(
                f"artifact:{name}:shape",
                "BLOCKED",
                f"direction must be 1-D, got shape {tuple(direction.shape)}",
            )
        )
    elif int(direction.shape[0]) != expected_hidden_size:
        results.append(
            CheckResult(
                f"artifact:{name}:hidden-size",
                "BLOCKED",
                f"expected {expected_hidden_size}, got {int(direction.shape[0])}",
            )
        )
    else:
        results.append(
            CheckResult(
                f"artifact:{name}:hidden-size",
                "OK",
                f"shape={tuple(direction.shape)}",
            )
        )

    norm = float(np.linalg.norm(direction))
    results.append(
        CheckResult(
            f"artifact:{name}:norm",
            "OK" if norm > 0.0 else "BLOCKED",
            f"norm={norm:.6f}",
        )
    )
    return results


def _parse_artifact_overrides(values: Sequence[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError("Artifact overrides must look like name=path.")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Artifact override name cannot be empty.")
        overrides[name] = Path(path.strip()).expanduser()
    return overrides


def _env_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _value = stripped.split("=", 1)
        keys.add(key.strip())
    return keys


def _modal_credential_markers() -> list[Path]:
    home = Path.home()
    candidates = (
        home / ".modal.toml",
        home / ".modal",
        home / ".config" / "modal.toml",
    )
    return [path for path in candidates if path.exists()]


def _print_report(payload: dict[str, Any]) -> None:
    print("CK-7 Modal prerequisite check")
    print(f"can_run_modal_now={payload['can_run_modal_now']}")
    print("")
    for check in payload["checks"]:
        print(f"[{check['status']}] {check['name']} - {check['detail']}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        help=(
            "Override or add a direction artifact as name=path. Defaults check "
            "CK-1 boundary plus CK-4.5 per-axis guardrail vector targets."
        ),
    )
    parser.add_argument(
        "--only-artifact-overrides",
        action="store_true",
        help="Check only artifacts supplied with --artifact.",
    )
    parser.add_argument(
        "--direction-root",
        type=Path,
        default=None,
        help="Root containing {axis_id}_direction.npz guardrail vectors.",
    )
    parser.add_argument(
        "--ck1-direction",
        type=Path,
        default=None,
        help="CK-1 boundary direction path. Defaults to data/models/vectors.",
    )
    parser.add_argument("--expected-hidden-size", type=int, default=896)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--fail-on-blocker",
        action="store_true",
        help="Return exit code 1 if any required check is blocked.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
