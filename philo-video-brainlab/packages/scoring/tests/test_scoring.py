import numpy as np
import pandas as pd

from scoring import normalize_targets, latent_pca, pareto_front, run_ablation, TARGET_NAMES


def test_normalize_targets_shapes():
    df = pd.DataFrame(
        {
            "views": [1000, 5000, 200],
            "likes": [100, 250, 40],
            "comments": [10, 5, 20],
            "shares": [3, 40, 1],
            "saves": [5, 5, 30],
            "retention": [0.4, 0.7, 0.5],
        },
        index=["a", "b", "c"],
    )
    t = normalize_targets(df)
    assert list(t.frame.columns) == TARGET_NAMES
    assert t.matrix().shape == (3, 5)


def test_pareto_front_marks_nondominated():
    targets = np.array([[1.0, 1.0], [0.5, 0.5], [2.0, 0.1]])
    front = pareto_front(targets)
    assert front[0] and front[2]      # non-dominated
    assert not front[1]               # dominated by row 0


def test_ablation_detects_informative_brain_features():
    rng = np.random.default_rng(0)
    n = 120
    groups = rng.integers(0, 6, size=n)          # 6 creators
    base = rng.normal(size=(n, 4))
    brain = rng.normal(size=(n, 3))
    # target genuinely depends on a brain feature -> ablation should show uplift
    y = (2.0 * brain[:, 0] + 0.3 * base[:, 0] + rng.normal(scale=0.1, size=n)).reshape(-1, 1)
    res = run_ablation(base, brain, y, ["likes"], groups)
    assert res.per_target[0].uplift > 0
