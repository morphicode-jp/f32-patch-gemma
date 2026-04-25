"""Opt-in practical control-law gate for ODIN results.

This module keeps the confirmed experiment-side law small and dependency-light:
strict global deployability is measured separately from a per-dimension control
contract. The optimizer output is not changed here; callers decide how to use
the diagnostics.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def _bounds_arrays(param_ranges: Sequence[tuple[float, float]], dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.asarray(param_ranges, dtype=float)
    if bounds.shape != (dim, 2):
        raise ValueError(f"param_ranges shape mismatch: {bounds.shape} vs dim={dim}")
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    span = hi - lo
    if np.any(~np.isfinite(bounds)) or np.any(span <= 0.0):
        raise ValueError("param_ranges must be finite increasing pairs")
    return lo, hi, span


def _robustness(base: float, mean: float, std: float) -> float:
    drop = base - mean
    if base > 1e-9:
        robust = 1.0 - max(drop, 0.0) / max(base, 1e-9)
    else:
        scale = max(abs(base), abs(mean), std, 1e-9)
        robust = 1.0 / (1.0 + max(drop, 0.0) / scale)
    return float(min(1.0, max(0.0, robust)))


def _evaluate_points(
    points: Sequence[np.ndarray],
    eval_fn: Callable[[np.ndarray], float],
    batch_eval_fn: Callable | None,
) -> tuple[list[float], int]:
    """Evaluate points either one-by-one or through a caller-provided batch hook."""
    if len(points) == 0:
        return [], 0
    if batch_eval_fn is None:
        return [float(eval_fn(p)) for p in points], 0

    payload = [np.asarray(p, dtype=float).tolist() for p in points]
    raw_scores = batch_eval_fn(payload)
    scores = list(raw_scores)
    if len(scores) != len(payload):
        raise ValueError(
            f"batch_eval_fn returned {len(scores)} scores for {len(payload)} points"
        )
    return [float(s) for s in scores], 1


def probe_control_noise(
    params: Sequence[float],
    eval_fn: Callable[[np.ndarray], float],
    param_ranges: Sequence[tuple[float, float]],
    *,
    sigma_by_dim: Sequence[float],
    n_trials: int = 31,
    seed: int = 41,
    min_robustness: float = 0.8,
    early_stop: bool = True,
    min_trials: int = 32,
    decision_margin: float = 0.08,
    batch_eval_fn: Callable | None = None,
    batch_size: int = 64,
) -> dict:
    """Measure candidate quality under per-dimension normalized Gaussian noise."""
    x = np.asarray(params, dtype=float)
    dim = int(x.size)
    lo, hi, span = _bounds_arrays(param_ranges, dim)
    sigmas = np.asarray(sigma_by_dim, dtype=float)
    if sigmas.shape != (dim,):
        raise ValueError(f"sigma_by_dim shape mismatch: {sigmas.shape} vs dim={dim}")
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas < 0.0):
        raise ValueError("sigma_by_dim must be finite non-negative values")

    trials = max(1, int(n_trials))
    min_checks = max(1, min(trials, int(min_trials)))
    margin = max(0.0, float(decision_margin))
    chunk_size = 1 if batch_eval_fn is None else max(1, int(batch_size))
    rng = np.random.default_rng(seed)
    base_scores, base_batch_calls = _evaluate_points(
        [np.minimum(np.maximum(x, lo), hi)],
        eval_fn,
        batch_eval_fn,
    )
    base = float(base_scores[0])
    scores: list[float] = []
    early_stopped = False
    stop_reason = None
    attempted_trials = 0
    batch_calls = base_batch_calls
    for start in range(0, trials, chunk_size):
        n_chunk = min(chunk_size, trials - start)
        noise = rng.normal(0.0, 1.0, size=(n_chunk, dim)) * sigmas * span
        points = np.minimum(np.maximum(x + noise, lo), hi)
        chunk_scores, chunk_batch_calls = _evaluate_points(points, eval_fn, batch_eval_fn)
        batch_calls += chunk_batch_calls
        attempted_trials += n_chunk
        for score in chunk_scores:
            if math.isfinite(score):
                scores.append(float(score))
        if early_stop and len(scores) >= min_checks and attempted_trials < trials:
            arr_now = np.asarray(scores, dtype=float)
            robust_now = _robustness(
                base,
                float(arr_now.mean()),
                float(arr_now.std()),
            )
            if robust_now >= min_robustness + margin:
                early_stopped = True
                stop_reason = "robustness_above_margin"
                break
            if robust_now < min_robustness - margin:
                early_stopped = True
                stop_reason = "robustness_below_margin"
                break

    eval_calls = 1 + attempted_trials
    if not math.isfinite(base) or not scores:
        return {
            "base": base,
            "mean": float("-inf"),
            "std": float("inf"),
            "min": float("-inf"),
            "p05": float("-inf"),
            "utility": float("-inf"),
            "robustness": 0.0,
            "deployable": False,
            "finite_trials": len(scores),
            "n_trials": trials,
            "eval_calls": eval_calls,
            "batch_calls": batch_calls,
            "batch_enabled": batch_eval_fn is not None,
            "batch_size": int(batch_size),
            "early_stopped": early_stopped,
            "stop_reason": stop_reason,
            "sigma_by_dim": sigmas.tolist(),
        }

    arr = np.asarray(scores, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std())
    p05 = float(np.percentile(arr, 5))
    robust = _robustness(base, mean, std)
    return {
        "base": base,
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "p05": p05,
        "utility": p05,
        "robustness": robust,
        "deployable": bool(robust >= min_robustness),
        "finite_trials": len(scores),
        "n_trials": trials,
        "eval_calls": eval_calls,
        "batch_calls": batch_calls,
        "batch_enabled": batch_eval_fn is not None,
        "batch_size": int(batch_size),
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "sigma_by_dim": sigmas.tolist(),
    }


def evaluate_control_law_contract(
    params: Sequence[float],
    eval_fn: Callable[[np.ndarray], float],
    param_ranges: Sequence[tuple[float, float]],
    *,
    sigma: float = 0.1,
    n_trials: int = 31,
    seed: int = 41,
    min_robustness: float = 0.8,
    joint_budget: bool = True,
    early_stop: bool = True,
    min_trials: int = 32,
    decision_margin: float = 0.08,
    skip_contract_when_global_deployable: bool = True,
    batch_eval_fn: Callable | None = None,
    batch_size: int = 64,
) -> dict:
    """Return strict global deployability and the safer joint-budget contract."""
    x = np.asarray(params, dtype=float)
    dim = int(x.size)
    if dim == 0:
        raise ValueError("params must not be empty")
    target_sigma = float(sigma)
    if not math.isfinite(target_sigma) or target_sigma < 0.0:
        raise ValueError("sigma must be a finite non-negative value")

    global_sigmas = [target_sigma] * dim
    divisor = math.sqrt(dim) if joint_budget else 1.0
    contract_sigmas = [target_sigma / divisor] * dim

    global_probe = probe_control_noise(
        x,
        eval_fn,
        param_ranges,
        sigma_by_dim=global_sigmas,
        n_trials=n_trials,
        seed=seed,
        min_robustness=min_robustness,
        early_stop=early_stop,
        min_trials=min_trials,
        decision_margin=decision_margin,
        batch_eval_fn=batch_eval_fn,
        batch_size=batch_size,
    )
    contract_skipped = bool(skip_contract_when_global_deployable and global_probe["deployable"])
    if contract_skipped:
        contract_probe = dict(global_probe)
        contract_probe["skipped"] = True
        contract_probe["skip_reason"] = "global_already_deployable"
        contract_probe["sigma_by_dim"] = contract_sigmas
        contract_probe["eval_calls"] = 0
        contract_probe["batch_calls"] = 0
        contract_probe["batch_enabled"] = batch_eval_fn is not None
        contract_probe["batch_size"] = int(batch_size)
        contract_probe["early_stopped"] = True
        contract_probe["stop_reason"] = "global_already_deployable"
    else:
        contract_probe = probe_control_noise(
            x,
            eval_fn,
            param_ranges,
            sigma_by_dim=contract_sigmas,
            n_trials=n_trials,
            seed=seed + 1,
            min_robustness=min_robustness,
            early_stop=early_stop,
            min_trials=min_trials,
            decision_margin=decision_margin,
            batch_eval_fn=batch_eval_fn,
            batch_size=batch_size,
        )

    final_deployable = bool(global_probe["deployable"] or contract_probe["deployable"])
    eval_calls = int(global_probe.get("eval_calls", 0)) + int(contract_probe.get("eval_calls", 0))
    batch_calls = int(global_probe.get("batch_calls", 0)) + int(contract_probe.get("batch_calls", 0))
    return {
        "enabled": True,
        "law": "joint_noise_budget",
        "target_sigma": target_sigma,
        "min_robustness": float(min_robustness),
        "joint_budget_enabled": bool(joint_budget),
        "global_sigma_by_dim": global_sigmas,
        "sigma_by_dim": contract_sigmas,
        "early_stop_enabled": bool(early_stop),
        "early_stop_min_trials": int(min_trials),
        "early_stop_decision_margin": float(decision_margin),
        "contract_skipped": contract_skipped,
        "eval_calls": eval_calls,
        "batch_eval_enabled": batch_eval_fn is not None,
        "batch_size": int(batch_size),
        "batch_calls": batch_calls,
        "global_probe": global_probe,
        "contract_probe": contract_probe,
        "practical_deployable": bool(global_probe["deployable"]),
        "deployable_under_dim_sigma": bool(contract_probe["deployable"]),
        "final_deployable": final_deployable,
        "contract": {
            "type": "per_dimension_sigma",
            "source": "joint_noise_budget",
            "meaning": (
                "The candidate is separately checked under strict global sigma and "
                "under a per-dimension control contract; the contract does not "
                "claim unconstrained global deployability."
            ),
            "target_sigma": target_sigma,
            "sigma_by_dim": contract_sigmas,
            "min_robustness": float(min_robustness),
        },
    }


__all__ = ["evaluate_control_law_contract", "probe_control_noise"]
