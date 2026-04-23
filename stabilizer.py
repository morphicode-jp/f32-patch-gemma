#!/usr/bin/env python3
"""
Stabilizer — 最適化器の peak を plateau に変換する汎用ラッパー
================================================================
ablation 実測 (stabilizer_ablation.py) で判明:
  Metropolis単独で robust 78.5% 獲得。Darwinian は plateau構築に寄与せず。

設計思想:
  最適化器が見つける peak は「一点」から摂動で崩壊する。
  Metropolis + cooling で peak 周辺を探索し、高fitness を維持できる
  近傍点の集合 = plateau を empirical に獲得する。

インターフェース:
  stabilize(x0, eval_fn, param_ranges) → StabilizerResult
    任意の (np.ndarray, Callable, bounds) に対して動作。
    ODIN/OWL/scipy/self-built optimizer どれとでも組み合わせ可能。

使用例:
  result = stabilize(x0=odin_best, eval_fn=my_score, param_ranges=bounds)
  robust_peak = result.centroid    # plateau中心 (単一点が欲しい時)
  ensemble    = result.particles   # 全粒子 (アンサンブル評価)
  plateau_w   = result.width       # plateau幅 (信頼区間的に使える)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional
import numpy as np


@dataclass
class StabilizerResult:
    particles: np.ndarray          # (n_particles, dim) 最終位置
    fitnesses: np.ndarray          # (n_particles,) 最終fitness
    centroid: np.ndarray           # (dim,) fitness重み付き重心
    width: np.ndarray              # (dim,) 次元ごとのstd (plateau幅)
    x0: np.ndarray                 # 入力seed
    x0_fitness: float              # 入力seedのfitness
    history: dict = field(default_factory=dict)  # fitness/T/accept_rate per gen


def _clip(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, bounds[:, 0]), bounds[:, 1])


def stabilize(
    x0: Sequence[float],
    eval_fn: Callable[[np.ndarray], float],
    param_ranges: Sequence[tuple[float, float]],
    *,
    n_particles: int = 12,
    n_gens: int = 20,
    sigma0: float = 0.08,
    sigma_decay: float = 0.95,
    T0: float = 1.0,
    cooling: float = 0.92,
    jitter: float = 0.02,
    seed: int = 31,
    verbose: bool = True,
    maximize: bool = True,
) -> StabilizerResult:
    """
    Peak → Plateau 変換 (Metropolis + cooling).

    Args:
        x0: 最適化器が見つけた peak 点 (dim,)
        eval_fn: fitness関数. eval_fn(x) → float. maximize=Trueなら高いほど良い.
        param_ranges: 各次元の (min, max) list
        n_particles: 粒子数 (ODIN実測の12推奨)
        n_gens: 世代数
        sigma0: 初期摂動スケール (bounds幅の0.08倍相当)
        sigma_decay: 世代ごとにσを乗算 (0.95だと20gen後 ~36%)
        T0: 初期温度 (fitness単位で解釈される)
        cooling: 温度減衰率
        jitter: x0の周りに撒く初期散乱 (bounds幅の比)
        seed: RNG seed
        verbose: 進捗print
        maximize: Trueなら高fitness採択. False(最小化)なら符号反転して扱う.

    Returns:
        StabilizerResult: particles / centroid / width / history
    """
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed + 1)

    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)
    bounds = np.asarray(param_ranges, dtype=float)
    assert bounds.shape == (dim, 2), f"param_ranges shape mismatch: {bounds.shape} vs dim={dim}"

    span = bounds[:, 1] - bounds[:, 0]
    sign = 1.0 if maximize else -1.0

    # ── 初期化: x0 周辺にjitterで撒く ──
    X = np.tile(x0, (n_particles, 1))
    X += rng.normal(0.0, jitter, size=X.shape) * span
    X = _clip(X, bounds)

    # 初期fitness
    F = np.array([sign * eval_fn(x) for x in X])
    x0_fit = float(sign * eval_fn(x0))

    history = {"fitness_mean": [], "fitness_max": [], "T": [], "sigma": [],
               "accept_rate": [], "width_mean": []}

    if verbose:
        print(f"[stabilizer] dim={dim} n={n_particles} gens={n_gens} "
              f"x0_fit={x0_fit:+.4f}")

    for gen in range(n_gens):
        T = T0 * (cooling ** gen)
        sigma = sigma0 * (sigma_decay ** gen)
        accepts = 0

        for i in range(n_particles):
            # 提案 = 現在位置 + N(0, sigma*span)
            prop = X[i] + rng.normal(0.0, sigma, size=dim) * span
            prop = _clip(prop, bounds)
            f_prop = sign * eval_fn(prop)

            # Metropolis (高fitness=低"energy")
            dF = F[i] - f_prop   # >0 なら proposal が改善
            if dF < 0 or pyrng.random() < math.exp(-dF / max(T, 1e-9)):
                X[i] = prop
                F[i] = f_prop
                accepts += 1

        # 次元ごとの広がり
        width = X.std(axis=0)
        history["fitness_mean"].append(float(F.mean()))
        history["fitness_max"].append(float(F.max()))
        history["T"].append(T)
        history["sigma"].append(sigma)
        history["accept_rate"].append(accepts / n_particles)
        history["width_mean"].append(float(width.mean()))

        if verbose and (gen % max(1, n_gens // 6) == 0 or gen == n_gens - 1):
            print(f"  gen{gen:>2}: f_mean={F.mean():+.3f} f_max={F.max():+.3f} "
                  f"T={T:.3f} σ={sigma:.3f} acc={accepts/n_particles:.0%} "
                  f"width_mean={width.mean():.3f}")

    # ── 後処理: fitness重み付き重心 + 幅 ──
    # 負 fitness混在時に備えて softmax 重み
    f_display = sign * F  # 表示は元の符号に戻す
    w = np.exp((F - F.max()) / max(F.std(), 1e-9))
    w = w / w.sum()
    centroid = (X * w[:, None]).sum(axis=0)
    width = X.std(axis=0)

    return StabilizerResult(
        particles=X,
        fitnesses=f_display,
        centroid=centroid,
        width=width,
        x0=x0,
        x0_fitness=sign * x0_fit,
        history=history,
    )


# ─────────────────────────────────────────────────────────────
# 摂動耐性の汎用テスト (評価は同じ eval_fn を使う)
# ─────────────────────────────────────────────────────────────

def measure_robustness(
    x: Sequence[float],
    eval_fn: Callable[[np.ndarray], float],
    param_ranges: Sequence[tuple[float, float]],
    *,
    n_trials: int = 5,
    sigma: float = 0.1,
    seed: int = 42,
    maximize: bool = True,
) -> dict:
    """単一点 x の摂動耐性を測る. sigma は bounds幅に対する比."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(param_ranges, dtype=float)
    span = bounds[:, 1] - bounds[:, 0]
    sign = 1.0 if maximize else -1.0

    base = float(sign * eval_fn(x))
    perturbed = []
    for _ in range(n_trials):
        xp = x + rng.normal(0.0, sigma, size=x.shape) * span
        xp = np.minimum(np.maximum(xp, bounds[:, 0]), bounds[:, 1])
        perturbed.append(float(sign * eval_fn(xp)))

    p_mean = float(np.mean(perturbed))
    p_std = float(np.std(perturbed))
    drop = base - p_mean
    robust = 1.0 - drop / max(base, 1e-9) if base > 0 else 0.0
    return {
        "base": base,
        "perturbed_mean": p_mean,
        "perturbed_std": p_std,
        "drop": drop,
        "robustness": robust,
        "n_trials": n_trials,
        "sigma": sigma,
    }


# ─────────────────────────────────────────────────────────────
# アンサンブル評価: particlesの集団を1つのpredictorとして使う
# ─────────────────────────────────────────────────────────────

def ensemble_score(
    result: StabilizerResult,
    eval_fn: Callable[[np.ndarray], float],
    *,
    maximize: bool = True,
    reduction: str = "mean",
) -> float:
    """plateau粒子集団の評価. reduction=mean/max/median."""
    sign = 1.0 if maximize else -1.0
    fs = np.array([sign * eval_fn(x) for x in result.particles])
    fs = sign * fs
    if reduction == "mean":
        return float(fs.mean())
    elif reduction == "max":
        return float(fs.max())
    elif reduction == "median":
        return float(np.median(fs))
    else:
        raise ValueError(f"unknown reduction: {reduction}")


# ─────────────────────────────────────────────────────────────
# 動作確認: 合成関数で smoke test
# ─────────────────────────────────────────────────────────────

def _demo():
    """鋭い peak (sharp) + 広い plateau (broad) の混合関数で動作確認."""
    # f(x) = max(sharp_peak, broad_plateau)
    #   sharp: 高いが細い → 摂動で崩壊
    #   broad: 低いが広い → 摂動耐性
    def f(x):
        x = np.asarray(x)
        sharp_center = np.array([0.8, 0.8])
        broad_center = np.array([-0.5, -0.5])
        d_sharp = np.linalg.norm(x - sharp_center)
        d_broad = np.linalg.norm(x - broad_center)
        sharp = 1.0 * math.exp(-50 * d_sharp ** 2)   # 鋭い
        broad = 0.7 * math.exp(-2 * d_broad ** 2)    # 広い
        return sharp + broad

    bounds = [(-1.5, 1.5), (-1.5, 1.5)]
    x_sharp = [0.8, 0.8]
    x_broad = [-0.5, -0.5]

    print("=" * 60)
    print("Stabilizer demo: sharp vs broad peak")
    print("=" * 60)

    # まず生の摂動耐性 (各 peak 単体)
    r_sharp = measure_robustness(x_sharp, f, bounds, n_trials=10, sigma=0.1)
    r_broad = measure_robustness(x_broad, f, bounds, n_trials=10, sigma=0.1)
    print(f"\n[raw] sharp  base={r_sharp['base']:.3f} "
          f"pert={r_sharp['perturbed_mean']:.3f} robust={r_sharp['robustness']:.0%}")
    print(f"[raw] broad  base={r_broad['base']:.3f} "
          f"pert={r_broad['perturbed_mean']:.3f} robust={r_broad['robustness']:.0%}")

    # sharp を stabilize
    print("\n-- stabilize(sharp) --")
    res = stabilize(x_sharp, f, bounds, n_gens=15, n_particles=12, verbose=True)

    # stabilized centroid の耐性
    r_stab = measure_robustness(res.centroid, f, bounds, n_trials=10, sigma=0.1)
    print(f"\n[stabilized] centroid={res.centroid.round(3).tolist()}")
    print(f"[stabilized] base={r_stab['base']:.3f} "
          f"pert={r_stab['perturbed_mean']:.3f} robust={r_stab['robustness']:.0%}")
    print(f"[stabilized] width={res.width.round(3).tolist()}")

    print(f"\n→ sharp raw robust={r_sharp['robustness']:.0%} → stabilized={r_stab['robustness']:.0%}")


if __name__ == "__main__":
    _demo()
