"""Engagement scoring and the brain-vs-baseline ablation.

The whole package is organized around one honest question:

    Do TRIBE-derived brain trajectories improve engagement prediction beyond
    ordinary video/audio/text features?

`metrics`  — normalize the raw targets, keep them separate.
`latent`   — optional latent structure (weighted / PCA / Pareto).
`models`   — train predictors and run the with-brain vs. without-brain ablation.
"""

from .latent import latent_pca, latent_weighted, pareto_front
from .metrics import TARGET_NAMES, EngagementTargets, normalize_targets
from .models import AblationResult, run_ablation

__all__ = [
    "EngagementTargets",
    "TARGET_NAMES",
    "normalize_targets",
    "latent_weighted",
    "latent_pca",
    "pareto_front",
    "AblationResult",
    "run_ablation",
]
