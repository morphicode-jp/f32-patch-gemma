"""shell_batched.py — batched (N agents × D) versions of common shells.

Used by GPUBatchRunner to process many agents' shell deltas in single
vectorized calls. Trivially-batchable shells (brainstem, cerebellum EMA part,
salience baseline part) become numpy array ops across batch dim.

Complex shells with per-agent variable state (hippocampus slots, prefrontal
goals, dmn engagement EMA) remain per-agent for now — they can be batched
later if hot, but their compute is already cheap.

API: each function takes
  S_batch: (N, D) np array
  shells: list[N] of Shell instances (for param lookup)
Returns:
  delta_batch: (N, D) np array
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def brainstem_batched(S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Vectorized brainstem: pull toward range for each agent.

    Assumes all shells share same structural params (s_min, s_max, pull_strength,
    output_indices) — which is true in practice (same brainstem.json).
    """
    delta = np.zeros_like(S_batch)
    if not shells:
        return delta
    s0 = shells[0]
    s_min = float(s0.params.get("s_min", -3.0))
    s_max = float(s0.params.get("s_max", 3.0))
    pull = float(s0.params.get("pull_strength", 0.1))
    out_idx = s0.output_indices
    if out_idx.size == 0:
        return delta
    view = S_batch[:, out_idx]                # (N, K)
    below = np.minimum(view - s_min, 0.0)
    above = np.maximum(view - s_max, 0.0)
    delta[:, out_idx] = -pull * (below + above)
    return delta


def cerebellum_batched(S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Vectorized cerebellum: motor smoothing + prediction error.

    Per-agent state (history deque, EMA) is kept per shell. We extract
    all EMAs as (N, 3), update vectorized, write back.
    """
    delta = np.zeros_like(S_batch)
    N = S_batch.shape[0]
    if not shells:
        return delta
    s0 = shells[0]
    motor_dims = list(s0.params.get("motor_dims", [16, 17, 18]))
    ws_start = int(s0.params.get("ws_start", 35))
    ws_end = int(s0.params.get("ws_end", 51))
    ema_alpha = float(s0.params.get("ema_alpha", 0.6))
    smoothing_strength = float(s0.params.get("smoothing_strength", 0.3))
    prediction_weight = float(s0.params.get("prediction_weight", 0.5))
    pred_err_idx = int(s0.params.get("prediction_error_idx", ws_end - 1))

    motor = S_batch[:, motor_dims]            # (N, 3)
    # Collect per-agent EMA
    ema_arr = np.zeros((N, len(motor_dims)), dtype=np.float64)
    prev_motor = np.zeros((N, len(motor_dims)), dtype=np.float64)
    for i, sh in enumerate(shells):
        if sh._ema is None:
            ema_arr[i] = motor[i]
            sh._ema = motor[i].copy()
        else:
            ema_arr[i] = sh._ema
        if len(sh._motor_history) >= 1:
            prev_motor[i] = sh._motor_history[-1]
        else:
            prev_motor[i] = motor[i]

    # Update EMAs vectorized
    new_ema = ema_alpha * motor + (1 - ema_alpha) * ema_arr  # (N, 3)

    # Prediction error: |actual - ema|
    pred_error = np.mean(np.abs(motor - new_ema), axis=1)  # (N,)

    # Smoothing delta on motor dims: push toward new_ema
    for k, dim_idx in enumerate(motor_dims):
        delta[:, dim_idx] = smoothing_strength * (new_ema[:, k] - motor[:, k])

    # Write pred_error to dedicated slot (absolute-write)
    delta[:, pred_err_idx] = pred_error - S_batch[:, pred_err_idx]

    # Write back per-agent EMAs and histories
    for i, sh in enumerate(shells):
        sh._ema = new_ema[i].copy()
        sh._motor_history.append(motor[i].copy())
        sh._state["last_prediction_error"] = float(pred_error[i])

    return delta


def salience_batched(S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Vectorized salience: attention map from novelty.

    Baseline EMA per-agent, novelty computed vs baseline, attention written
    to attention slice.
    """
    delta = np.zeros_like(S_batch)
    N = S_batch.shape[0]
    if not shells:
        return delta
    s0 = shells[0]
    watched_slice = slice(
        int(s0.params.get("watched_start", 0)),
        int(s0.params.get("watched_end", 35)))
    attention_slice = slice(
        int(s0.params.get("attention_start", 51)),
        int(s0.params.get("attention_end", 83)))
    baseline_alpha = float(s0.params.get("baseline_alpha", 0.05))
    novelty_threshold = float(s0.params.get("novelty_threshold", 0.15))
    boost_strength = float(s0.params.get("boost_strength", 0.1))

    watched = S_batch[:, watched_slice]      # (N, K)
    K = watched.shape[1]

    # Per-agent baseline
    baseline_arr = np.zeros_like(watched)
    for i, sh in enumerate(shells):
        if sh._baseline is None:
            baseline_arr[i] = watched[i]
            sh._baseline = watched[i].copy()
        else:
            baseline_arr[i] = sh._baseline
    new_baseline = (1 - baseline_alpha) * baseline_arr + baseline_alpha * watched

    novelty = np.abs(watched - new_baseline)                 # (N, K)
    # Normalize per-agent to [0, 1]
    max_n = novelty.max(axis=1, keepdims=True)               # (N, 1)
    max_n = np.where(max_n > 1e-6, max_n, 1.0)
    attention = novelty / max_n                              # (N, K)

    # Write attention map
    att_width = attention_slice.stop - attention_slice.start
    att_to_write = attention[:, :att_width] if K >= att_width else attention
    a_start = attention_slice.start
    cols = min(att_to_write.shape[1], att_width)
    for k in range(cols):
        delta[:, a_start + k] = att_to_write[:, k] - S_batch[:, a_start + k]

    # Boost novel dims (threshold gated)
    boost_mask = novelty > novelty_threshold                 # (N, K) bool
    sign = np.where(watched >= new_baseline, 1.0, -1.0)       # (N, K)
    boost_val = boost_strength * sign * novelty * boost_mask.astype(np.float64)
    w_start = watched_slice.start
    for k in range(K):
        delta[:, w_start + k] += boost_val[:, k]

    # Write back baselines
    for i, sh in enumerate(shells):
        sh._baseline = new_baseline[i].copy()
    return delta


# Name-based dispatch
BATCHED_IMPLS = {
    "brainstem": brainstem_batched,
    "cerebellum": cerebellum_batched,
    "salience": salience_batched,
}


def has_batched_impl(shell_name: str) -> bool:
    return shell_name in BATCHED_IMPLS


def apply_batched(shell_name: str, S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Apply batched step for named shell type, return (N, D) delta."""
    return BATCHED_IMPLS[shell_name](S_batch, shells)
