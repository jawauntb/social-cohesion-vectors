"""The MVP gate: does the brain trajectory add predictive power?

`run_ablation` trains the same model twice per target — once on baseline
multimodal features, once on baseline + brain-derived features — under grouped
cross-validation (group = creator, so we test generalization to held-out
creators, not memorization). The uplift is the whole ballgame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score


@dataclass
class TargetAblation:
    target: str
    r2_baseline: float
    r2_with_brain: float

    @property
    def uplift(self) -> float:
        return self.r2_with_brain - self.r2_baseline


@dataclass
class AblationResult:
    per_target: list[TargetAblation] = field(default_factory=list)

    @property
    def mean_uplift(self) -> float:
        return float(np.mean([t.uplift for t in self.per_target])) if self.per_target else 0.0

    def verdict(self, threshold: float = 0.02) -> str:
        up = self.mean_uplift
        if up >= threshold:
            return (
                f"GO — brain features add mean R² uplift of {up:+.3f}. "
                "TRIBE trajectories are capturing something the multimodal baseline misses."
            )
        return (
            f"NO-GO (yet) — mean R² uplift only {up:+.3f}. "
            "Ship the multimodal predictor; the brain signal isn't earning its keep."
        )


def _cv_r2(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int) -> float:
    """Grouped-CV out-of-fold R². Groups = creators → held-out-creator test."""
    mask = ~np.isnan(y)
    X, y, groups = X[mask], y[mask], groups[mask]
    n_groups = len(np.unique(groups))
    if n_groups < 2 or len(y) < 8:
        return float("nan")
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)
    preds = np.full_like(y, np.nan, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        model = GradientBoostingRegressor(random_state=0)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    ok = ~np.isnan(preds)
    return float(r2_score(y[ok], preds[ok])) if ok.sum() > 1 else float("nan")


def run_ablation(
    baseline_features: np.ndarray,   # n x d_base  (transcript+visual+audio+rhythm)
    brain_features: np.ndarray,      # n x d_brain (TRIBE trajectory summaries)
    targets: np.ndarray,             # n x t
    target_names: list[str],
    groups: np.ndarray,              # n, creator id per row
    n_splits: int = 5,
) -> AblationResult:
    """Compare baseline vs. baseline+brain for every target under grouped CV."""
    X_base = np.nan_to_num(baseline_features, nan=0.0)
    X_brain = np.nan_to_num(brain_features, nan=0.0)
    X_both = np.hstack([X_base, X_brain])

    result = AblationResult()
    for j, name in enumerate(target_names):
        y = targets[:, j]
        r2_base = _cv_r2(X_base, y, groups, n_splits)
        r2_both = _cv_r2(X_both, y, groups, n_splits)
        result.per_target.append(
            TargetAblation(target=name, r2_baseline=r2_base, r2_with_brain=r2_both)
        )
    return result
