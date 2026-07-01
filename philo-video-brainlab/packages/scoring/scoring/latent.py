"""Optional latent structure over the separate targets.

We don't invent a single "engagement" number up front. Instead we offer several
lenses and let the data say whether engagement is one axis or (more likely) a
few modes — e.g. curiosity vs. identity/social-signaling.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def latent_weighted(targets: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Simple weighted linear combination of z-scored targets."""
    n_targets = targets.shape[1]
    w = np.ones(n_targets) / n_targets if weights is None else np.asarray(weights, float)
    w = w / (np.abs(w).sum() or 1.0)
    filled = np.nan_to_num(targets, nan=0.0)
    return filled @ w


def latent_pca(targets: np.ndarray, n_components: int = 2):
    """Factor the target space. If the first component dominates, engagement is
    ~one axis; if two components matter, there are distinct modes worth naming.

    Returns (scores, explained_variance_ratio, components).
    """
    filled = np.nan_to_num(targets, nan=0.0)
    k = min(n_components, filled.shape[1])
    pca = PCA(n_components=k)
    scores = pca.fit_transform(filled)
    return scores, pca.explained_variance_ratio_, pca.components_


def pareto_front(targets: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated videos (no single score).

    A video is on the front if no other video is >= on every target and > on at
    least one. Useful when you refuse to trade comments for likes prematurely.
    """
    filled = np.nan_to_num(targets, nan=-np.inf)
    n = filled.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        dominated = np.all(filled >= filled[i], axis=1) & np.any(filled > filled[i], axis=1)
        if dominated.any():
            on_front[i] = False
    return on_front
