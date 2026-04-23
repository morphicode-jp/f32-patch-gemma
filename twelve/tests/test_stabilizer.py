"""Tests for stabilizer.py (peak→plateau) + mimir_odin_stable integration."""
import math
import os
import sys

import pytest

# Project root for stabilizer import
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _ROOT)

import numpy as np

from stabilizer import (
    StabilizerResult,
    ensemble_score,
    measure_robustness,
    stabilize,
)
from twelve.agent.mimir_odin_stable import mimir_odin_stable


# -----------------------------------------------------------------
# module-level eval_fns (picklable)
# -----------------------------------------------------------------

def sharp_peak_fn(x):
    """2d 鋭い peak (崩壊しやすい)."""
    x = np.asarray(x)
    center = np.array([0.8, 0.8])
    d = np.linalg.norm(x - center)
    return float(math.exp(-50 * d ** 2))


def broad_plateau_fn(x):
    """2d 広い plateau (崩壊しにくい)."""
    x = np.asarray(x)
    center = np.array([-0.5, -0.5])
    d = np.linalg.norm(x - center)
    return float(0.7 * math.exp(-2 * d ** 2))


def mixed_fn(x):
    """sharp + broad の混合."""
    return sharp_peak_fn(x) + broad_plateau_fn(x)


def quadratic_max(x):
    """負 quadratic (最大化)."""
    return -sum((v - 0.5) ** 2 for v in x)


# -----------------------------------------------------------------
# StabilizerResult 構造 + stabilize() 基本動作
# -----------------------------------------------------------------

def test_stabilize_basic_returns_valid_result():
    """stabilize() が正しい dataclass を返す."""
    r = stabilize(
        x0=[0.8, 0.8],
        eval_fn=sharp_peak_fn,
        param_ranges=[(-1.5, 1.5)] * 2,
        n_particles=8,
        n_gens=5,
        verbose=False,
    )
    assert isinstance(r, StabilizerResult)
    assert r.particles.shape == (8, 2)
    assert r.fitnesses.shape == (8,)
    assert r.centroid.shape == (2,)
    assert r.width.shape == (2,)
    # width は non-negative
    assert np.all(r.width >= 0)
    # history キー確認
    assert "fitness_mean" in r.history
    assert "T" in r.history
    assert "sigma" in r.history
    assert len(r.history["T"]) == 5


def test_stabilize_centroid_within_bounds():
    """centroid が bounds 内に収まる."""
    r = stabilize(
        x0=[0.9, 0.9],
        eval_fn=sharp_peak_fn,
        param_ranges=[(-1.0, 1.0)] * 2,
        n_particles=8,
        n_gens=5,
        verbose=False,
    )
    assert -1.0 <= r.centroid[0] <= 1.0
    assert -1.0 <= r.centroid[1] <= 1.0


def test_stabilize_improves_mixed_peak_robustness():
    """sharp + broad 混合で stabilize すると摂動耐性が大幅改善 (demo の 8%→96% パターン)."""
    bounds = [(-1.5, 1.5)] * 2
    r_raw = measure_robustness([0.8, 0.8], mixed_fn, bounds,
                                n_trials=15, sigma=0.1)
    r_stab = stabilize(
        x0=[0.8, 0.8],
        eval_fn=mixed_fn,
        param_ranges=bounds,
        n_particles=12,
        n_gens=15,
        verbose=False,
    )
    r_stab_rob = measure_robustness(r_stab.centroid, mixed_fn, bounds,
                                     n_trials=15, sigma=0.1)
    # sharp raw robust は低く (< 30%)、stabilized は高い (> 50%) 想定
    assert r_raw["robustness"] < 0.30, f"sharp raw not sharp: {r_raw['robustness']}"
    assert r_stab_rob["robustness"] > 0.50, (
        f"stabilize failed to improve robustness: {r_stab_rob['robustness']}"
    )


# -----------------------------------------------------------------
# measure_robustness 単体
# -----------------------------------------------------------------

def test_measure_robustness_returns_keys():
    r = measure_robustness([0.5, 0.5], quadratic_max, [(0, 1)] * 2, n_trials=5)
    for k in ("base", "perturbed_mean", "perturbed_std", "drop",
              "robustness", "n_trials", "sigma"):
        assert k in r


def test_measure_robustness_broad_is_robust():
    """broad plateau は摂動でもあまり下がらない."""
    r = measure_robustness([-0.5, -0.5], broad_plateau_fn, [(-1.5, 1.5)] * 2,
                            n_trials=15, sigma=0.1)
    assert r["robustness"] > 0.50


# -----------------------------------------------------------------
# ensemble_score
# -----------------------------------------------------------------

def test_ensemble_score_reductions():
    r = stabilize(
        x0=[0.5, 0.5], eval_fn=quadratic_max,
        param_ranges=[(-1, 1)] * 2, n_particles=6, n_gens=4, verbose=False,
    )
    mean_v = ensemble_score(r, quadratic_max, reduction="mean")
    max_v = ensemble_score(r, quadratic_max, reduction="max")
    med_v = ensemble_score(r, quadratic_max, reduction="median")
    # max >= median >= mean (概ね、noise 逆転もあるが同オーダー)
    assert max_v >= mean_v - 0.5
    assert isinstance(mean_v, float)


def test_ensemble_score_invalid_reduction():
    r = stabilize(
        x0=[0.5, 0.5], eval_fn=quadratic_max,
        param_ranges=[(-1, 1)] * 2, n_particles=4, n_gens=2, verbose=False,
    )
    with pytest.raises(ValueError):
        ensemble_score(r, quadratic_max, reduction="invalid_mode")


# -----------------------------------------------------------------
# mimir_odin_stable 統合
# -----------------------------------------------------------------

def test_mimir_odin_stable_basic():
    """mimir_odin_stable が期待キーを全部返す."""
    r = mimir_odin_stable(
        quadratic_max,
        [(-1, 1)] * 2,
        time_budget=10,
        stabilize_particles=6,
        stabilize_gens=5,
        executor="thread",   # picklable issue 回避
        experience_id="test_odin_stable",
        verbose=False,
    )
    # odin の元キー
    assert "specialist" in r
    assert "council" in r
    # stable 固有キー
    assert "peak_params" in r
    assert "peak_robustness" in r
    assert "best_params" in r   # ← plateau centroid に上書き
    assert "plateau_robustness" in r
    assert "plateau_width" in r
    assert "plateau_particles" in r
    assert "robustness_improvement" in r
    assert "stabilize_elapsed_s" in r
    assert "stage1_elapsed_s" in r
    assert r["stabilize_applied"] is True


def test_mimir_odin_stable_peak_vs_plateau_present():
    """peak と plateau が別 array として返る (壊れず分離されてる)."""
    r = mimir_odin_stable(
        quadratic_max,
        [(-1, 1)] * 3,
        time_budget=8,
        stabilize_particles=6,
        stabilize_gens=5,
        executor="thread",
        experience_id="test_odin_stable_vs",
        verbose=False,
    )
    assert len(r["peak_params"]) == 3
    assert len(r["best_params"]) == 3   # plateau centroid
    assert len(r["plateau_width"]) == 3
    # robustness は 0-1 の範囲
    assert 0.0 <= r["peak_robustness"] <= 1.01
    assert 0.0 <= r["plateau_robustness"] <= 1.01


def test_mimir_odin_stable_passes_odin_kwargs():
    """extra kwargs が mimir_odin に passthrough される."""
    r = mimir_odin_stable(
        quadratic_max,
        [(-1, 1)] * 2,
        time_budget=8,
        stabilize_particles=4,
        stabilize_gens=3,
        curated_measurements=[
            {"params": [0.5, 0.5], "score": 0.0},
            {"params": [0.4, 0.5], "score": -0.01},
            {"params": [0.5, 0.6], "score": -0.01},
            {"params": [0.6, 0.5], "score": -0.01},
            {"params": [0.5, 0.4], "score": -0.01},
        ],
        executor="thread",
        experience_id="test_odin_stable_pass",
        verbose=False,
    )
    # curated は expensive_single ルートに切替 → tool_used が "owl" に近い
    assert r.get("tool_used") is not None
