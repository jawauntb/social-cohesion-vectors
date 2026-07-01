# @brainlab/scoring

Engagement metrics, latent structure, and the **brain-vs-baseline ablation** that
decides whether the rest of the project is worth building.

## Modules

- `metrics.py` — normalize raw counts into comparable, rate-per-reach, z-scored
  targets. Keeps `likes / comments / shares / saves / retention` **separate**.
- `latent.py` — optional combination: `latent_weighted`, `latent_pca` (is engagement
  one axis or two modes?), `pareto_front` (no single score).
- `models.py` — `run_ablation(...)` trains baseline vs. baseline+brain per target under
  **grouped CV by creator**, so we measure generalization to held-out creators. The
  `AblationResult.verdict()` is the go/no-go gate.

## The gate

```python
from scoring import run_ablation

res = run_ablation(baseline_feats, brain_feats, targets, TARGET_NAMES, creator_groups)
print(res.verdict())   # "GO — brain features add mean R² uplift of +0.05 ..."
```

## Test

```bash
uv pip install -e ".[dev]"
pytest
```
