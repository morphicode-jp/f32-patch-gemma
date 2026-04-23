"""mimir_odin_stable — odin → stabilizer 自動連結 (peak → plateau、Rule 16).

背景:
  mimir_odin は 4 specialist 並列で**最良 peak** を見つける。しかしこの peak は
  「尖った山」のことがあり、実世界で使う際に param の微小ドリフトで性能暴落する。
  Stabilizer (Metropolis + cooling) で peak 周辺を探索し、plateau 中心に変換
  すると摂動耐性が 8% → 96% に跳ね上がる (sharp 関数での実測)。

Pipeline:
  Stage 1: mimir_odin で最適 peak 発見 (time_budget × (1 - stabilize_ratio))
  Stage 2: stabilizer.stabilize() で peak → plateau 変換
  Stage 3: plateau centroid + 12 particle ensemble を返す

使い方:
  from twelve.agent.mimir_odin_stable import mimir_odin_stable

  r = mimir_odin_stable(eval_fn, ranges, time_budget=300)
  # 実用推奨:
  print(r["best_params"])         # plateau centroid (robust)
  # 比較用:
  print(r["peak_params"])         # odin の sharp peak
  print(r["peak_robustness"])     # peak の摂動耐性 (%)
  print(r["plateau_robustness"])  # plateau の摂動耐性 (%)
  print(r["plateau_width"])       # 各次元の信頼区間的幅
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Optional, Sequence

# Project root を sys.path に追加 (stabilizer.py を import するため)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def mimir_odin_stable(
    eval_fn: Callable,
    param_ranges: Sequence[tuple],
    *,
    time_budget: float = 300.0,
    stabilize_budget_ratio: float = 0.25,
    stabilize_particles: int = 12,
    stabilize_gens: int = 20,
    stabilize_sigma0: float = 0.08,
    stabilize_sigma_decay: float = 0.95,
    stabilize_T0: float = 1.0,
    stabilize_cooling: float = 0.92,
    stabilize_jitter: float = 0.02,
    stabilize_seed: int = 31,
    robustness_sigma: float = 0.10,
    robustness_trials: int = 5,
    verbose: bool = False,
    **odin_kwargs: Any,
) -> dict:
    """odin → stabilizer 自動連結。peak (理論最適) + plateau (実用推奨) 両方返す。

    Args:
      eval_fn: f(params: list[float]) -> float (higher=better、mimir 規約)
      param_ranges: [(lo, hi), ...]
      time_budget: 総時間予算 (秒)
      stabilize_budget_ratio: 総予算のうち stabilize に使う比率 (default 0.25)
      stabilize_particles: plateau 探索粒子数 (default 12)
      stabilize_gens: plateau 探索世代数 (default 20)
      stabilize_sigma0 〜 jitter: Metropolis パラメータ
      stabilize_seed: stabilizer RNG seed
      robustness_sigma: 摂動耐性測定の sigma (bounds 比、default 0.10)
      robustness_trials: 摂動耐性測定の試行数 (default 5)
      verbose: 進捗 print
      odin_kwargs: mimir_odin に passthrough (specialists / executor 等)

    Returns:
      mimir_odin の全キー (specialist / council / council_variance_std / ...) +
        peak_params       : odin が見つけた sharp peak (list)
        peak_score        : peak の実測 score
        peak_robustness   : peak の摂動耐性 (0-1)
        best_params       : ★ plateau centroid (list、実用推奨)
        plateau_score     : centroid の実測 score
        plateau_robustness: plateau の摂動耐性 (0-1、期待 >0.7)
        plateau_width     : 各次元の std (list、信頼区間的)
        plateau_particles : 最終 12 粒子の位置 (list of list、アンサンブル用)
        robustness_improvement: plateau_robustness - peak_robustness
        stabilize_elapsed_s: Stage 2 所要時間
        stage1_elapsed_s   : Stage 1 (odin) 所要時間
    """
    from twelve.agent.mimir_odin import mimir_odin
    from stabilizer import stabilize, measure_robustness

    t_all = time.time()

    # ---- Stage 1: mimir_odin で peak 発見 ----
    odin_budget = time_budget * (1.0 - stabilize_budget_ratio)
    if verbose:
        print(f"[odin_stable] Stage 1: odin search ({odin_budget:.1f}s budget)")
    t1 = time.time()
    odin_result = mimir_odin(
        eval_fn, param_ranges,
        time_budget=odin_budget,
        verbose=verbose,
        **odin_kwargs,
    )
    stage1_elapsed = time.time() - t1

    peak_params = odin_result.get("best_params")
    if peak_params is None:
        raise RuntimeError("mimir_odin returned no best_params")
    peak_score = odin_result.get("verified_score")
    if peak_score is None:
        peak_score = odin_result.get("best_score")

    # ---- peak の摂動耐性測定 (比較用) ----
    peak_rob = measure_robustness(
        peak_params, eval_fn, param_ranges,
        n_trials=robustness_trials,
        sigma=robustness_sigma,
        seed=stabilize_seed + 1,
        maximize=True,
    )

    # ---- Stage 2: stabilizer で peak → plateau 変換 ----
    if verbose:
        print(f"[odin_stable] Stage 2: stabilize "
              f"({stabilize_particles} particles × {stabilize_gens} gens)")
    t2 = time.time()
    try:
        stab_result = stabilize(
            x0=peak_params,
            eval_fn=eval_fn,
            param_ranges=list(param_ranges),
            n_particles=stabilize_particles,
            n_gens=stabilize_gens,
            sigma0=stabilize_sigma0,
            sigma_decay=stabilize_sigma_decay,
            T0=stabilize_T0,
            cooling=stabilize_cooling,
            jitter=stabilize_jitter,
            seed=stabilize_seed,
            verbose=verbose,
            maximize=True,
        )
        centroid = stab_result.centroid.tolist()
        width = stab_result.width.tolist()
        particles = stab_result.particles.tolist()
    except Exception as e:
        # stabilize 失敗時: odin peak をそのまま best_params に fallback
        if verbose:
            print(f"[odin_stable] stabilize failed ({type(e).__name__}: {e}); "
                  f"falling back to peak")
        centroid = list(peak_params)
        width = [0.0] * len(peak_params)
        particles = [list(peak_params)]
    stabilize_elapsed = time.time() - t2

    # ---- plateau centroid の実測 + 摂動耐性 ----
    try:
        plateau_score = float(eval_fn(centroid))
    except Exception:
        plateau_score = float("-inf")

    plateau_rob = measure_robustness(
        centroid, eval_fn, param_ranges,
        n_trials=robustness_trials,
        sigma=robustness_sigma,
        seed=stabilize_seed + 2,
        maximize=True,
    )

    # ---- 結果 merge ----
    result = dict(odin_result)
    # peak (odin の結果を保持)
    result["peak_params"] = list(peak_params)
    result["peak_score"] = peak_score
    result["peak_robustness"] = peak_rob["robustness"]
    # plateau (stabilizer の結果、これが best_params として返る)
    result["best_params"] = centroid
    result["best_score"] = plateau_score
    result["plateau_score"] = plateau_score
    result["plateau_robustness"] = plateau_rob["robustness"]
    result["plateau_width"] = width
    result["plateau_particles"] = particles
    # diagnostics
    result["robustness_improvement"] = plateau_rob["robustness"] - peak_rob["robustness"]
    result["stabilize_elapsed_s"] = stabilize_elapsed
    result["stage1_elapsed_s"] = stage1_elapsed
    result["total_elapsed_s"] = time.time() - t_all
    result["stabilize_applied"] = True

    return result


__all__ = ["mimir_odin_stable"]
