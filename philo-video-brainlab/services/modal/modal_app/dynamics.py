"""Trajectory dynamics — the part we actually care about.

Success is likely caused less by *where* the brain is than by *how it moves*.
These summaries turn a steps x dim trajectory into compact, model-ready features
and are also what the editor-note landmarks are derived from.
"""

from __future__ import annotations

import numpy as np


def velocity(traj: np.ndarray) -> np.ndarray:
    """Per-step speed ||b(t+1) - b(t)||. High velocity ~ rapid representational change."""
    if traj.shape[0] < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(traj, axis=0), axis=1)


def curvature(traj: np.ndarray) -> np.ndarray:
    """Turning of the path (angle between successive velocity vectors)."""
    d = np.diff(traj, axis=0)
    if d.shape[0] < 2:
        return np.zeros(0)
    a, b = d[:-1], d[1:]
    na = np.linalg.norm(a, axis=1) + 1e-8
    nb = np.linalg.norm(b, axis=1) + 1e-8
    cos = np.clip(np.sum(a * b, axis=1) / (na * nb), -1.0, 1.0)
    return np.arccos(cos)


def novelty(traj: np.ndarray) -> np.ndarray:
    """Distance of each state from the running mean of prior states.

    Semantic novelty that plateaus is a classic disengagement signature.
    """
    out = np.zeros(traj.shape[0])
    for t in range(1, traj.shape[0]):
        prior = traj[:t].mean(axis=0)
        out[t] = np.linalg.norm(traj[t] - prior)
    return out


def surprise(traj: np.ndarray) -> np.ndarray:
    """Prediction error under a constant-velocity forecast: ||b(t) - b_hat(t)||."""
    if traj.shape[0] < 3:
        return np.zeros(traj.shape[0])
    out = np.zeros(traj.shape[0])
    for t in range(2, traj.shape[0]):
        forecast = traj[t - 1] + (traj[t - 1] - traj[t - 2])
        out[t] = np.linalg.norm(traj[t] - forecast)
    return out


def novelty_decay(nov: np.ndarray) -> float:
    """Slope of novelty over time (negative = plateauing/dropping)."""
    if nov.shape[0] < 3:
        return 0.0
    t = np.arange(nov.shape[0])
    return float(np.polyfit(t, nov, 1)[0])


def summarize(traj: np.ndarray) -> dict:
    """Compact dynamics features for the engagement model."""
    v, c, nov, sur = velocity(traj), curvature(traj), novelty(traj), surprise(traj)
    return {
        "velocity_mean": float(v.mean()) if v.size else 0.0,
        "curvature_mean": float(c.mean()) if c.size else 0.0,
        "novelty_decay": novelty_decay(nov),
        "surprise_mean": float(sur.mean()) if sur.size else 0.0,
    }


def landmarks(traj: np.ndarray, fps: float) -> list[dict]:
    """Human-readable trajectory events for editor notes.

    Returns anchored notes like curiosity collapse (velocity crash) and novelty
    plateau. These are heuristics over the dynamics, not clinical claims.
    """
    notes: list[dict] = []
    v, nov = velocity(traj), novelty(traj)
    if v.size >= 4:
        # sustained velocity drop after an early peak -> curiosity collapse
        peak = int(np.argmax(v[: max(2, v.size // 2)]))
        tail = v[peak + 1 :]
        if tail.size and tail.mean() < 0.5 * v[peak]:
            t_sec = round((peak + 1) / max(fps, 1e-6), 1)
            notes.append(
                {
                    "tSec": t_sec,
                    "kind": "curiosity",
                    "message": f"Curiosity collapses after second {t_sec:g} (representational change flattens).",
                    "severity": 2,
                }
            )
    if nov.size >= 4 and novelty_decay(nov) < 0:
        t_sec = round(int(np.argmax(nov)) / max(fps, 1e-6), 1)
        notes.append(
            {
                "tSec": t_sec,
                "kind": "novelty",
                "message": f"Semantic novelty peaks near second {t_sec:g}, then plateaus.",
                "severity": 1,
            }
        )
    return notes
