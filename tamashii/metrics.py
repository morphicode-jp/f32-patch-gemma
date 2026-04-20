"""Structure-level metrics for tamashii — measures the ARCHITECTURE'S
operation quality, not task performance.

Metrics:
  - shell_coupling: correlation of per-tick deltas between shell pairs
    (off-diagonal = shells interacting through S)
  - state_diversity: how many distinct basins S visits (PCA variance, n_clusters)
  - novelty_response: information content of S delta relative to input
  - behavioral_coherence: motor trajectory autocorrelation / smoothness
  - additive_scaling: complexity metric vs n_shells (should grow monotonically)
"""
from __future__ import annotations

from typing import Any

import numpy as np


def shell_coupling(delta_log: list[dict[str, float]]) -> dict:
    """Compute pairwise correlation of shell delta norms over time.

    delta_log: list of dicts {shell_name: delta_norm} per tick.
    Returns: dict with
      - shell_names: list
      - correlation: N×N np.ndarray
      - off_diag_mean: average of |corr[i,j]| for i≠j (higher = more coupling)
    """
    if not delta_log:
        return {"shell_names": [], "correlation": np.array([[]]), "off_diag_mean": 0.0}
    names = sorted(delta_log[0].keys())
    series = np.array([[d.get(n, 0.0) for d in delta_log] for n in names])
    n = len(names)
    if n == 0 or series.shape[1] < 2:
        return {"shell_names": names, "correlation": np.zeros((n, n)),
                "off_diag_mean": 0.0}

    # Guard against zero-variance series
    stds = series.std(axis=1)
    valid = stds > 1e-10
    if not valid.any():
        return {"shell_names": names, "correlation": np.zeros((n, n)),
                "off_diag_mean": 0.0}

    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if stds[i] > 1e-10 and stds[j] > 1e-10:
                corr[i, j] = float(np.corrcoef(series[i], series[j])[0, 1])
            elif i == j:
                corr[i, j] = 1.0
    # Off-diagonal mean (abs, excluding self-correlation)
    mask = ~np.eye(n, dtype=bool)
    off_diag = np.abs(corr[mask])
    off_mean = float(off_diag.mean()) if off_diag.size else 0.0
    return {
        "shell_names": names,
        "correlation": corr,
        "off_diag_mean": off_mean,
    }


def state_diversity(S_trajectory: np.ndarray, n_pca_dims: int = 4) -> dict:
    """Measure how much of S-space is explored.

    S_trajectory: (T, D) array of S snapshots over time.
    Returns:
      - explained_variance: top-n_pca_dims fraction
      - trajectory_length: sum of |ΔS| per step (total path length)
      - unique_bins: count of distinct coarse bins visited
    """
    T, D = S_trajectory.shape
    if T < 2:
        return {"explained_variance": [], "trajectory_length": 0.0, "unique_bins": 0}

    # Center
    mean = S_trajectory.mean(axis=0, keepdims=True)
    centered = S_trajectory - mean
    # Covariance + eigendecomp (limit to first few dims for speed)
    cov = (centered.T @ centered) / max(1, T - 1)
    # Use np.linalg.eigvalsh on a smaller eff matrix if D large
    if D <= 64:
        eigs = np.linalg.eigvalsh(cov)
    else:
        # Economy: use SVD of centered
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigs = (s ** 2) / max(1, T - 1)
        except np.linalg.LinAlgError:
            eigs = np.zeros(D)
    eigs = np.sort(eigs)[::-1]
    total = eigs.sum()
    if total <= 1e-10:
        explained = [0.0] * n_pca_dims
    else:
        top_k = min(n_pca_dims, len(eigs))
        explained = [float(e / total) for e in eigs[:top_k]]
        # Pad
        while len(explained) < n_pca_dims:
            explained.append(0.0)

    # Trajectory length
    deltas = np.diff(S_trajectory, axis=0)
    traj_len = float(np.linalg.norm(deltas, axis=1).sum())

    # Unique bins (coarse quantization of S to detect basin visits)
    bin_size = 0.5
    bins = np.round(S_trajectory / bin_size).astype(int)
    unique_bins = len({tuple(b) for b in bins})

    return {
        "explained_variance": explained,
        "trajectory_length": traj_len,
        "unique_bins": int(unique_bins),
    }


def behavioral_coherence(motor_series: np.ndarray) -> dict:
    """Motor trajectory analysis.

    motor_series: (T, 3) array of (nav, speed, voice) per step.
    """
    T = motor_series.shape[0]
    if T < 3:
        return {"autocorr_lag1": 0.0, "smoothness": 0.0, "std": 0.0}

    # Autocorrelation at lag 1 (smoothness — high = coherent; 0 = random)
    autocorrs = []
    for dim in range(motor_series.shape[1]):
        x = motor_series[:, dim]
        mean = x.mean()
        var = x.var()
        if var > 1e-10:
            ac = np.mean((x[:-1] - mean) * (x[1:] - mean)) / var
        else:
            ac = 0.0
        autocorrs.append(float(ac))

    # Smoothness: inverse of mean |2nd difference|
    second_diff = np.diff(motor_series, 2, axis=0)
    smoothness = float(1.0 / (1.0 + np.abs(second_diff).mean()))

    motor_std = float(motor_series.std())

    return {
        "autocorr_lag1": float(np.mean(autocorrs)),
        "smoothness": smoothness,
        "std": motor_std,
    }


def novelty_response(S_trajectory: np.ndarray, input_series: np.ndarray) -> dict:
    """Measure S's information-theoretic response to input novelty.

    Approximates: how much does ΔS vary when input varies?
    """
    T = min(len(S_trajectory), len(input_series))
    if T < 3:
        return {"response_ratio": 0.0}

    S_deltas = np.diff(S_trajectory[:T], axis=0)
    input_deltas = np.diff(input_series[:T], axis=0)
    s_var = float(np.var(S_deltas.flatten()))
    inp_var = float(np.var(input_deltas.flatten()))
    if inp_var < 1e-10:
        return {"response_ratio": 0.0}
    return {"response_ratio": s_var / inp_var}


def summarize(
    delta_log: list[dict[str, float]],
    S_trajectory: np.ndarray,
    motor_series: np.ndarray,
    input_series: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run all metrics in one call."""
    out = {}
    out["shell_coupling"] = shell_coupling(delta_log)
    out["state_diversity"] = state_diversity(S_trajectory)
    out["behavioral_coherence"] = behavioral_coherence(motor_series)
    if input_series is not None:
        out["novelty_response"] = novelty_response(S_trajectory, input_series)
    return out
