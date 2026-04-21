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


def hippocampus_batched(S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Vectorized hippocampus: cosine similarity across N agents × n_slots.

    Per-agent slots vary in count (0..n_slots) — pad to fixed n_slots tensor,
    mask empty slots. Recall / store decisions still per-agent (Python) but
    cosine similarity computation is fully vectorized.
    """
    delta = np.zeros_like(S_batch)
    N = S_batch.shape[0]
    if not shells:
        return delta
    s0 = shells[0]
    query_slice = slice(
        int(s0.params.get("query_start", 19)),
        int(s0.params.get("query_end", 35)))
    trace_slice = slice(
        int(s0.params.get("trace_start", 83)),
        int(s0.params.get("trace_end", 147)))
    n_slots = int(s0.params.get("n_slots", 8))
    slot_dim = int(s0.params.get("slot_dim", 8))
    recall_thr = float(s0.params.get("recall_threshold", 0.8))
    store_thr = float(s0.params.get("store_threshold", 0.6))
    recall_strength = float(s0.params.get("recall_strength", 0.2))

    # Extract queries (N, Q) and current trace_slice state (N, trace_dim)
    queries = S_batch[:, query_slice].astype(np.float64)       # (N, Q)
    trace_cur = S_batch[:, trace_slice].astype(np.float64)
    q_dim = queries.shape[1]

    # Build padded slot tensor (N, n_slots, q_dim). Missing slots filled with zeros + mask.
    slot_storage = np.zeros((N, n_slots, q_dim), dtype=np.float64)
    slot_payload = np.zeros((N, n_slots, slot_dim), dtype=np.float64)
    slot_valid = np.zeros((N, n_slots), dtype=bool)
    for i, sh in enumerate(shells):
        sh._step_count += 1
        n = len(sh._slot_queries)
        for k in range(min(n, n_slots)):
            slot_storage[i, k] = sh._slot_queries[k]
            slot_payload[i, k] = sh._slot_payloads[k]
            slot_valid[i, k] = True

    # Cosine similarity (N, n_slots)
    # normalize queries
    q_norm = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-6  # (N,1)
    q_unit = queries / q_norm
    s_norm = np.linalg.norm(slot_storage, axis=2, keepdims=True) + 1e-6  # (N, n_slots, 1)
    s_unit = slot_storage / s_norm
    # sims[i, k] = dot(q_unit[i], s_unit[i, k])
    sims = (s_unit * q_unit[:, None, :]).sum(axis=2)  # (N, n_slots)
    # Mask invalid slots (sim = -inf)
    sims = np.where(slot_valid, sims, -np.inf)
    max_sim = sims.max(axis=1)  # (N,)
    best_idx = sims.argmax(axis=1)  # (N,)

    t_start = trace_slice.start
    t_end = trace_slice.stop

    # Recall: per-agent (hard to fully vectorize due to payload writing)
    for i, sh in enumerate(shells):
        if max_sim[i] > recall_thr and slot_valid[i, best_idx[i]]:
            bi = int(best_idx[i])
            sh._usage[bi] = sh._step_count
            sh._recall_events += 1
            payload = slot_payload[i, bi]
            # Write payload to first slot of trace region
            for j in range(min(slot_dim, t_end - t_start)):
                delta[i, t_start + j] = recall_strength * (
                    payload[j] - S_batch[i, t_start + j])

        # Store: if novel enough
        if max_sim[i] < store_thr:
            pyl = queries[i, :slot_dim].copy()
            n_stored = len(sh._slot_queries)
            if n_stored < n_slots:
                sh._slot_queries.append(queries[i].copy())
                sh._slot_payloads.append(pyl)
                sh._usage.append(sh._step_count)
            else:
                lru = int(np.argmin(sh._usage))
                sh._slot_queries[lru] = queries[i].copy()
                sh._slot_payloads[lru] = pyl
                sh._usage[lru] = sh._step_count

    # Mirror all slots to trace_slice for external visibility
    # (N, n_slots * slot_dim) reshaped and written
    for i, sh in enumerate(shells):
        for s_i, payload in enumerate(sh._slot_payloads):
            off = t_start + s_i * slot_dim
            for j in range(slot_dim):
                if off + j < t_end:
                    delta[i, off + j] = 0.3 * (payload[j] - S_batch[i, off + j])

    return delta


def dmn_batched(S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Vectorized DMN: engagement EMA + replay when idle."""
    delta = np.zeros_like(S_batch)
    N = S_batch.shape[0]
    if not shells:
        return delta
    s0 = shells[0]
    hippo_slice = slice(
        int(s0.params.get("hippo_start", 83)),
        int(s0.params.get("hippo_end", 147)))
    scratch_slice = slice(
        int(s0.params.get("scratch_start", 176)),
        int(s0.params.get("scratch_end", 192)))
    motor_dims = list(s0.params.get("motor_dims", [16, 17, 18]))
    engagement_threshold = float(s0.params.get("engagement_threshold", 0.3))
    replay_strength = float(s0.params.get("replay_strength", 0.15))
    decay = float(s0.params.get("decay", 0.9))
    slot_dim = int(s0.params.get("slot_dim", 8))

    motor = S_batch[:, motor_dims]  # (N, 3)
    motor_mag = np.linalg.norm(motor, axis=1)  # (N,)

    # Collect engagement per agent
    eng_arr = np.array([sh._engagement_ema for sh in shells], dtype=np.float64)
    alpha = 0.1  # hardcoded in original
    new_eng = (1 - alpha) * eng_arr + alpha * motor_mag  # (N,)

    sc_start = scratch_slice.start
    sc_end = scratch_slice.stop
    scratch_width = sc_end - sc_start

    # Always decay scratch (multiplicative)
    cur_scratch = S_batch[:, sc_start:sc_end]
    delta[:, sc_start:sc_end] = (decay - 1.0) * cur_scratch

    # Replay for agents whose engagement < threshold
    idle_mask = new_eng < engagement_threshold  # (N,) bool
    hp_traces = S_batch[:, hippo_slice]  # (N, trace_dim)
    n_slots_in_trace = max(1, hp_traces.shape[1] // slot_dim)
    for i, sh in enumerate(shells):
        if idle_mask[i]:
            trace_off = (sh._replay_idx % n_slots_in_trace) * slot_dim
            trace = hp_traces[i, trace_off:trace_off + slot_dim]
            n_write = min(trace.size, scratch_width)
            for j in range(n_write):
                delta[i, sc_start + j] += replay_strength * float(trace[j])
            sh._replay_idx += 1
        sh._engagement_ema = float(new_eng[i])
    return delta


def prefrontal_batched(S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Vectorized prefrontal: working memory slots + top-down bias."""
    delta = np.zeros_like(S_batch)
    N = S_batch.shape[0]
    if not shells:
        return delta
    s0 = shells[0]
    attention_slice = slice(
        int(s0.params.get("attention_start", 51)),
        int(s0.params.get("attention_end", 83)))
    watched_slice = slice(
        int(s0.params.get("watched_start", 0)),
        int(s0.params.get("watched_end", 35)))
    wm_slice = slice(
        int(s0.params.get("wm_start", 163)),
        int(s0.params.get("wm_end", 176)))
    n_slots = int(s0.params.get("n_slots", 3))
    decay_rate = float(s0.params.get("decay_rate", 0.05))
    lock_threshold = float(s0.params.get("lock_threshold", 0.5))
    bias_strength = float(s0.params.get("bias_strength", 0.08))

    attention = S_batch[:, attention_slice]
    watched = S_batch[:, watched_slice]
    w_start = watched_slice.start
    w_width = watched.shape[1]
    att_width = attention.shape[1]

    # Per-agent slot management — can't easily vectorize due to variable state
    for i, sh in enumerate(shells):
        # Decay
        for slot in sh._slots:
            slot["activation"] *= (1.0 - decay_rate)
        # Lock candidates
        att_trim = attention[i, :w_width] if att_width > w_width else attention[i]
        lock_candidates = np.where(att_trim > lock_threshold)[0]
        for c_idx in lock_candidates:
            exists = any(s["source_idx"] == int(c_idx) for s in sh._slots)
            if not exists and len(sh._slots) < n_slots:
                sh._slots.append({
                    "source_idx": int(c_idx),
                    "target_value": float(watched[i, c_idx]),
                    "activation": 1.0,
                })
        # Remove expired
        sh._slots = [s for s in sh._slots if s["activation"] > 0.05]
        sh._slots.sort(key=lambda s: -s["activation"])
        sh._slots = sh._slots[:n_slots]
        # Push bias
        for slot in sh._slots:
            idx = slot["source_idx"]
            cur = watched[i, idx]
            tgt = slot["target_value"]
            gain = bias_strength * slot["activation"]
            delta[i, w_start + idx] += gain * (tgt - cur)
        # Write slot summary to wm
        wm_start = wm_slice.start
        wm_width = wm_slice.stop - wm_start
        if wm_width >= 2 * n_slots:
            for k in range(n_slots):
                if k < len(sh._slots):
                    act = sh._slots[k]["activation"]
                    src = float(sh._slots[k]["source_idx"])
                else:
                    act, src = 0.0, 0.0
                delta[i, wm_start + k] = act - S_batch[i, wm_start + k]
                delta[i, wm_start + n_slots + k] = src - S_batch[i, wm_start + n_slots + k]
    return delta


# Name-based dispatch
BATCHED_IMPLS = {
    "brainstem": brainstem_batched,
    "cerebellum": cerebellum_batched,
    "salience": salience_batched,
    "hippocampus": hippocampus_batched,
    "prefrontal": prefrontal_batched,
    "dmn": dmn_batched,
}


def has_batched_impl(shell_name: str) -> bool:
    return shell_name in BATCHED_IMPLS


def apply_batched(shell_name: str, S_batch: np.ndarray, shells: list) -> np.ndarray:
    """Apply batched step for named shell type, return (N, D) delta."""
    return BATCHED_IMPLS[shell_name](S_batch, shells)
