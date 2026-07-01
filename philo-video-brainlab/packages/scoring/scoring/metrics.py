"""Engagement targets — kept deliberately separate.

A great philosophy video may be high-save / high-comment, not merely high-like.
Different targets plausibly correspond to different cognitive states (discussion
vs. sharing vs. completion), so we preserve each target and only *optionally*
combine them later (see `latent.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The targets we predict, in a fixed order. Retention is 0..1 already.
TARGET_NAMES = ["likes", "comments", "shares", "saves", "retention"]

# Counts that should be normalized by reach before comparing across videos.
_COUNT_TARGETS = ["likes", "comments", "shares", "saves"]


@dataclass
class EngagementTargets:
    """A tidy, per-video view of the (normalized) targets."""

    frame: pd.DataFrame  # index = video_id, columns = TARGET_NAMES

    def matrix(self) -> np.ndarray:
        return self.frame[TARGET_NAMES].to_numpy(dtype=float)


def rate_per_reach(counts: pd.Series, reach: pd.Series, eps: float = 1.0) -> pd.Series:
    """Engagement *rate*: count per view (or per follower if views missing).

    Raw counts conflate reach with resonance. A rate isolates how strongly the
    audience that saw it reacted.
    """
    denom = reach.fillna(0).clip(lower=0) + eps
    return counts.fillna(0).clip(lower=0) / denom


def normalize_targets(
    df: pd.DataFrame,
    *,
    reach_col: str = "views",
    fallback_reach_col: str = "followersAtPost",
    log1p: bool = True,
) -> EngagementTargets:
    """Turn raw metric columns into comparable, roughly-Gaussian targets.

    Steps: count -> rate-per-reach -> log1p -> z-score. Retention is passed
    through (already a 0..1 fraction), then z-scored so every target shares a
    scale for latent combination.
    """
    reach = df[reach_col] if reach_col in df else pd.Series(index=df.index, dtype=float)
    if reach.isna().all() and fallback_reach_col in df:
        reach = df[fallback_reach_col]

    out = pd.DataFrame(index=df.index)
    for name in _COUNT_TARGETS:
        raw = df[name] if name in df else pd.Series(0, index=df.index)
        rate = rate_per_reach(raw, reach)
        out[name] = np.log1p(rate) if log1p else rate

    out["retention"] = df["retention"] if "retention" in df else np.nan

    # z-score each column (ignoring NaNs) so targets are comparable.
    z = (out - out.mean(skipna=True)) / out.std(skipna=True).replace(0, 1.0)
    return EngagementTargets(frame=z[TARGET_NAMES])
