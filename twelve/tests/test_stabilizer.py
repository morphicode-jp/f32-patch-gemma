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


def control_contract_fn(x):
    """同時摂動には弱いが次元別制御なら通る ridge."""
    x = np.asarray(x)
    return float(1.0 - x[0] ** 2 - 8.0 * x[1] ** 2 - x[2] ** 2)


def negative_loss_like(x):
    """常に負の -loss 系目的関数."""
    return -1.0 - sum(v ** 2 for v in x)


def zero_top_loss_like(x):
    """最良点が 0 になる -loss 系目的関数."""
    return -sum(v ** 2 for v in x)


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


def test_measure_robustness_negative_objective_is_not_forced_zero():
    """-loss 系の負 base でも摂動耐性を 0..1 で評価する."""
    r = measure_robustness([0.0, 0.0], negative_loss_like, [(-1, 1)] * 2,
                            n_trials=10, sigma=0.05)
    assert r["base"] < 0
    assert 0.0 <= r["robustness"] <= 1.0
    assert r["robustness"] > 0.0


def test_measure_robustness_zero_base_objective_is_meaningful():
    """最良 score=0 の -loss 系でも旧実装のように常時 0 へ潰さない."""
    r = measure_robustness([0.0, 0.0], zero_top_loss_like, [(-1, 1)] * 2,
                            n_trials=10, sigma=0.05)
    assert r["base"] == pytest.approx(0.0)
    assert 0.0 <= r["robustness"] <= 1.0
    assert r["robustness"] > 0.0


def test_control_law_contract_separates_global_and_dim_sigma():
    """strict global と dim別制御契約を別判定する."""
    from twelve.agent.control_law_gate import evaluate_control_law_contract

    r = evaluate_control_law_contract(
        [0.0, 0.0, 0.0],
        control_contract_fn,
        [(-1, 1)] * 3,
        sigma=0.1,
        n_trials=256,
        seed=17,
        min_robustness=0.8,
    )
    assert r["practical_deployable"] is False
    assert r["deployable_under_dim_sigma"] is True
    assert len(r["sigma_by_dim"]) == 3
    assert all(0.0 <= s <= 0.1 for s in r["sigma_by_dim"])


def test_control_law_skips_contract_when_global_passes():
    """global が十分 robust なら contract 再測定を省く."""
    from twelve.agent.control_law_gate import evaluate_control_law_contract

    calls = {"n": 0}

    def broad_counting_fn(x):
        calls["n"] += 1
        x = np.asarray(x)
        return float(1.0 - 0.01 * np.sum(x * x))

    r = evaluate_control_law_contract(
        [0.0, 0.0, 0.0],
        broad_counting_fn,
        [(-1, 1)] * 3,
        sigma=0.1,
        n_trials=256,
        seed=17,
        min_robustness=0.8,
        min_trials=16,
    )
    assert r["practical_deployable"] is True
    assert r["deployable_under_dim_sigma"] is True
    assert r["contract_skipped"] is True
    assert r["contract_probe"]["skip_reason"] == "global_already_deployable"
    assert r["eval_calls"] == r["global_probe"]["eval_calls"]
    assert calls["n"] < 256


def test_control_law_does_not_skip_contract_when_global_fails():
    """global が落ちる場合は contract 判定まで進む."""
    from twelve.agent.control_law_gate import evaluate_control_law_contract

    r = evaluate_control_law_contract(
        [0.0, 0.0, 0.0],
        control_contract_fn,
        [(-1, 1)] * 3,
        sigma=0.1,
        n_trials=256,
        seed=17,
        min_robustness=0.8,
        min_trials=16,
    )
    assert r["practical_deployable"] is False
    assert r["deployable_under_dim_sigma"] is True
    assert r["contract_skipped"] is False
    assert r["contract_probe"]["eval_calls"] > 0


def test_control_law_uses_batch_eval_fn():
    """batch_eval_fn がある場合は制御診断をまとめて評価する."""
    from twelve.agent.control_law_gate import evaluate_control_law_contract

    scalar_calls = {"n": 0}
    batch_calls = {"n": 0}

    def scalar_fn(x):
        scalar_calls["n"] += 1
        x = np.asarray(x)
        return float(1.0 - 0.01 * np.sum(x * x))

    def batch_fn(points):
        batch_calls["n"] += 1
        return [
            float(1.0 - 0.01 * np.sum(np.asarray(p) * np.asarray(p)))
            for p in points
        ]

    r = evaluate_control_law_contract(
        [0.0, 0.0, 0.0],
        scalar_fn,
        [(-1, 1)] * 3,
        sigma=0.1,
        n_trials=256,
        seed=17,
        min_robustness=0.8,
        min_trials=16,
        batch_eval_fn=batch_fn,
        batch_size=64,
    )
    assert scalar_calls["n"] == 0
    assert batch_calls["n"] == r["batch_calls"]
    assert r["batch_eval_enabled"] is True
    assert r["global_probe"]["batch_enabled"] is True
    assert r["contract_skipped"] is True
    assert r["batch_calls"] <= 2


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


def test_mimir_odin_stable_marks_fallback_when_stabilizer_fails(monkeypatch):
    """stabilizer 例外時は peak fallback を明示する."""
    import stabilizer as stabilizer_mod
    import twelve.agent.mimir_odin as odin_mod

    def fake_odin(eval_fn, param_ranges, **kwargs):
        return {
            "best_params": [0.0],
            "best_score": 1.0,
            "tool_used": "fake_odin",
        }

    def broken_stabilize(*args, **kwargs):
        raise RuntimeError("intentional stabilizer failure")

    monkeypatch.setattr(odin_mod, "mimir_odin", fake_odin)
    monkeypatch.setattr(stabilizer_mod, "stabilize", broken_stabilize)

    def eval_fn(p):
        return 1.0 - abs(p[0])

    r = mimir_odin_stable(
        eval_fn,
        [(-1, 1)],
        time_budget=1,
        robustness_trials=2,
        verbose=False,
    )
    assert r["best_params"] == [0.0]
    assert r["peak_params"] == [0.0]
    assert r["stabilize_applied"] is False
    assert "RuntimeError: intentional stabilizer failure" == r["stabilize_error"]


def test_mimir_odin_stable_control_law_is_opt_out(monkeypatch):
    """control law は default で有効。明示的に False を渡した時のみ抑止される (Rule 16b、2026-04-25 以降)。"""
    import stabilizer as stabilizer_mod
    import twelve.agent.mimir_odin as odin_mod

    def fake_odin(eval_fn, param_ranges, **kwargs):
        return {
            "best_params": [0.0, 0.0, 0.0],
            "best_score": 1.0,
            "tool_used": "fake_odin",
        }

    def fake_stabilize(*args, **kwargs):
        x0 = np.asarray([0.0, 0.0, 0.0], dtype=float)
        particles = np.tile(x0, (2, 1))
        return StabilizerResult(
            particles=particles,
            fitnesses=np.asarray([1.0, 1.0]),
            centroid=x0,
            width=np.zeros(3),
            x0=x0,
            x0_fitness=1.0,
            history={},
        )

    monkeypatch.setattr(odin_mod, "mimir_odin", fake_odin)
    monkeypatch.setattr(stabilizer_mod, "stabilize", fake_stabilize)

    r = mimir_odin_stable(
        control_contract_fn,
        [(-1, 1)] * 3,
        time_budget=1,
        robustness_trials=2,
        control_law_enabled=False,
        verbose=False,
    )
    assert "control_law" not in r
    assert "control_law_applied" not in r


def test_mimir_odin_stable_control_law_opt_in(monkeypatch):
    """有効化時だけ practical/control-law 診断を追加する."""
    import stabilizer as stabilizer_mod
    import twelve.agent.mimir_odin as odin_mod

    def fake_odin(eval_fn, param_ranges, **kwargs):
        return {
            "best_params": [0.0, 0.0, 0.0],
            "best_score": 1.0,
            "tool_used": "fake_odin",
        }

    def fake_stabilize(*args, **kwargs):
        x0 = np.asarray([0.0, 0.0, 0.0], dtype=float)
        particles = np.tile(x0, (2, 1))
        return StabilizerResult(
            particles=particles,
            fitnesses=np.asarray([1.0, 1.0]),
            centroid=x0,
            width=np.zeros(3),
            x0=x0,
            x0_fitness=1.0,
            history={},
        )

    monkeypatch.setattr(odin_mod, "mimir_odin", fake_odin)
    monkeypatch.setattr(stabilizer_mod, "stabilize", fake_stabilize)

    batch_calls = {"n": 0}

    def batch_eval_fn(points):
        batch_calls["n"] += 1
        return [control_contract_fn(p) for p in points]

    r = mimir_odin_stable(
        control_contract_fn,
        [(-1, 1)] * 3,
        time_budget=1,
        robustness_trials=2,
        control_law_enabled=True,
        control_law_trials=256,
        control_law_seed=17,
        control_law_min_robustness=0.8,
        control_law_batch_size=64,
        batch_eval_fn=batch_eval_fn,
        verbose=False,
    )
    assert r["control_law_applied"] is True
    assert r["control_law"]["law"] == "joint_noise_budget"
    assert r["practical_deployable"] is False
    assert r["deployable_under_dim_sigma"] is True
    assert len(r["control_sigma_by_dim"]) == 3
    assert r["practical_control_constrained_deployable"] is True
    assert r["control_law_eval_calls"] > 0
    assert r["control_law_contract_skipped"] is False
    assert r["control_law_batch_eval_enabled"] is True
    assert r["control_law_batch_calls"] == batch_calls["n"]


def test_mimir_odin_stable_auto_check_default_passes_continuous():
    """auto_check=True default で純連続 eval_fn は問題なく通過する."""
    r = mimir_odin_stable(
        lambda p: -(p[0] ** 2 + p[1] ** 2),
        [(-1.0, 1.0)] * 2,
        time_budget=6.0,
        executor="thread",
        verbose=False,
    )
    assert "auto_check" in r
    diag = r["auto_check"]
    assert diag is not None
    assert diag.get("severity") in ("ok", "warn")
    assert diag.get("recommended_optimizer") == "mimir_odin_stable"


def test_mimir_odin_stable_auto_check_raises_on_constant():
    """auto_check が constant eval_fn を fatal 検出して raise する."""
    def constant_fn(p):
        return 0.42

    with pytest.raises(ValueError, match="auto_check detected fatal"):
        mimir_odin_stable(
            constant_fn,
            [(-1.0, 1.0)] * 2,
            time_budget=6.0,
            executor="thread",
            verbose=False,
        )


def test_mimir_odin_stable_auto_check_false_bypasses():
    """auto_check=False で diagnostic 完全 skip、constant でも raise しない."""
    def constant_fn(p):
        return 0.42

    r = mimir_odin_stable(
        constant_fn,
        [(-1.0, 1.0)] * 2,
        time_budget=4.0,
        executor="thread",
        auto_check=False,
        verbose=False,
    )
    assert r.get("auto_check") is None
