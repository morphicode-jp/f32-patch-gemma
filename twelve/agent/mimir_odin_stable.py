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
    control_law_enabled: bool = True,
    control_law_sigma: Optional[float] = None,
    control_law_trials: Optional[int] = None,
    control_law_min_robustness: float = 0.80,
    control_law_joint_budget: bool = True,
    control_law_seed: Optional[int] = None,
    control_law_early_stop: bool = True,
    control_law_min_trials: int = 32,
    control_law_decision_margin: float = 0.08,
    control_law_skip_contract_when_global_deployable: bool = True,
    control_law_batch_eval_fn: Optional[Callable] = None,
    control_law_batch_size: int = 64,
    auto_check: bool = True,
    auto_check_budget_max: float = 15.0,
    auto_check_budget_ratio: float = 0.05,
    auto_check_declared_structural: bool = False,
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
      control_law_enabled: True なら実用投入用の制御契約を診断として追加 (default True、Rule 16b)
      control_law_sigma: 制御契約の目標 sigma。None なら robustness_sigma
      control_law_trials: 制御契約の測定回数。None なら max(robustness_trials, 31)
      control_law_min_robustness: deployable 判定の最低 robustness
      control_law_joint_budget: 次元ごとの sigma を sqrt(dim) で共同予算化する
      control_law_seed: 制御契約診断の RNG seed。None なら stabilize_seed + 101
      control_law_early_stop: 判定が固まった時に摂動測定を途中終了する
      control_law_min_trials: 早期終了を許可する最低 trial 数
      control_law_decision_margin: robustness 閾値からこの幅だけ離れたら早期終了
      control_law_skip_contract_when_global_deployable: global が通ったら contract 測定を省く
      control_law_batch_eval_fn: 制御契約診断用 batch_eval_fn。None なら odin_kwargs の batch_eval_fn を再利用
      control_law_batch_size: batch_eval_fn にまとめて渡す摂動点数
      auto_check: True で本番前に check_eval_fn() を自動実行 (Rule 0.5 強制)。
        fatal なら ValueError raise、structure_policy 推奨なら warning。
        慣れた eval_fn では auto_check=False で skip 可能。
      auto_check_budget_max: auto_check の最大予算秒 (default 15s)
      auto_check_budget_ratio: time_budget の何割を auto_check に割くか (default 5%)
        実 budget = max(5, min(auto_check_budget_max, time_budget * ratio))
      auto_check_declared_structural: True で混合問題 hint を check_eval_fn に渡す。
        この場合 stable じゃなく structure_policy 推奨が確定するので
        通常は意図的に False のまま (stable で押し通すなら warning 受ける)。
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
        control_law         : control_law_enabled=True の時だけ返る実用投入診断
    """
    from twelve.agent.mimir_odin import mimir_odin
    from stabilizer import stabilize, measure_robustness

    t_all = time.time()

    # ---- Stage 0: auto_check (Rule 0.5 を実装で強制) ----
    auto_check_diag = None
    if auto_check:
        from twelve.agent.eval_check import check_eval_fn
        check_budget = max(5.0, min(auto_check_budget_max,
                                    time_budget * auto_check_budget_ratio))
        try:
            auto_check_diag = check_eval_fn(
                eval_fn, param_ranges,
                time_budget=check_budget,
                declared_structural=auto_check_declared_structural,
                experience_id=odin_kwargs.get("experience_id", "stable_auto_check") + "_check",
                verbose=False,
            )
        except Exception as e:
            if verbose:
                print(f"[odin_stable] auto_check raised {type(e).__name__}: {e}; "
                      f"continuing without diagnostic")
            auto_check_diag = {"severity": "skipped", "error": str(e)}
        if auto_check_diag.get("severity") == "fatal":
            raise ValueError(
                f"[odin_stable] auto_check detected fatal eval_fn issues: "
                f"{auto_check_diag.get('issues', [])}. "
                f"Run check_eval_fn() manually to diagnose, "
                f"or pass auto_check=False to skip this safety check."
            )
        rec = auto_check_diag.get("recommended_optimizer")
        if rec == "mimir_odin_structure_policy":
            print(
                f"[odin_stable] WARNING: auto_check recommends "
                f"mimir_odin_structure_policy ({auto_check_diag.get('recommended_reason')}). "
                f"Continuing with stable as requested. "
                f"Pass auto_check=False to suppress this warning, or switch to "
                f"mimir_odin_structure_policy(param_decoder=...) for proper handling.",
                flush=True,
            )

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
    stabilize_applied = False
    stabilize_error = None
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
        stabilize_applied = True
    except Exception as e:
        # stabilize 失敗時: odin peak をそのまま best_params に fallback
        if verbose:
            print(f"[odin_stable] stabilize failed ({type(e).__name__}: {e}); "
                  f"falling back to peak")
        stabilize_error = f"{type(e).__name__}: {e}"
        centroid = list(peak_params)
        width = [0.0] * len(peak_params)
        particles = [list(peak_params)]
    stabilize_elapsed = time.time() - t2

    # ---- plateau centroid の実測 + 摂動耐性 ----
    try:
        plateau_score = float(eval_fn(centroid))
    except Exception:
        plateau_score = (
            float(peak_score) if not stabilize_applied and peak_score is not None
            else float("-inf")
        )

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
    result["stabilize_applied"] = stabilize_applied
    result["stabilize_error"] = stabilize_error
    result["auto_check"] = auto_check_diag

    if control_law_enabled:
        try:
            from twelve.agent.control_law_gate import evaluate_control_law_contract

            effective_control_batch_eval_fn = control_law_batch_eval_fn
            if effective_control_batch_eval_fn is None:
                candidate_batch_eval_fn = odin_kwargs.get("batch_eval_fn")
                if callable(candidate_batch_eval_fn):
                    effective_control_batch_eval_fn = candidate_batch_eval_fn

            control_law = evaluate_control_law_contract(
                centroid,
                eval_fn,
                param_ranges,
                sigma=robustness_sigma if control_law_sigma is None else control_law_sigma,
                n_trials=(
                    max(int(robustness_trials), 31)
                    if control_law_trials is None
                    else int(control_law_trials)
                ),
                seed=stabilize_seed + 101 if control_law_seed is None else int(control_law_seed),
                min_robustness=control_law_min_robustness,
                joint_budget=control_law_joint_budget,
                early_stop=control_law_early_stop,
                min_trials=control_law_min_trials,
                decision_margin=control_law_decision_margin,
                skip_contract_when_global_deployable=(
                    control_law_skip_contract_when_global_deployable
                ),
                batch_eval_fn=effective_control_batch_eval_fn,
                batch_size=control_law_batch_size,
            )
            result["control_law_applied"] = True
            result["control_law"] = control_law
            result["practical_deployable"] = control_law["practical_deployable"]
            result["deployable_under_dim_sigma"] = control_law["deployable_under_dim_sigma"]
            result["control_sigma_by_dim"] = control_law["sigma_by_dim"]
            result["control_contract"] = control_law["contract"]
            result["practical_control_constrained_deployable"] = (
                control_law["deployable_under_dim_sigma"]
            )
            result["practical_control_constrained_utility"] = (
                control_law["contract_probe"]["utility"]
            )
            result["practical_control_constrained_robustness"] = (
                control_law["contract_probe"]["robustness"]
            )
            result["practical_control_constrained_sigma_by_dim"] = (
                control_law["sigma_by_dim"]
            )
            result["control_law_eval_calls"] = control_law["eval_calls"]
            result["control_law_contract_skipped"] = control_law["contract_skipped"]
            result["control_law_batch_eval_enabled"] = control_law["batch_eval_enabled"]
            result["control_law_batch_calls"] = control_law["batch_calls"]
        except Exception as e:
            result["control_law_applied"] = False
            result["control_law_error"] = f"{type(e).__name__}: {e}"

    return result


__all__ = ["mimir_odin_stable"]
