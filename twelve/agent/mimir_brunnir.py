"""mimir_brunnir — 4 specialist mimir を並列実行し、最良結果を返す.

Mímisbrunnr (北欧神話): ミーミル (知恵の神) の泉。オーディンが片目を捧げて
知恵を汲んだ聖なる泉。本 module は 4 つの specialist mimir を「4 本の泉」と
見立て、それぞれから汲み上げた最適解候補の最良を採用する。

背景:
  単一 mimir は config によって得意分野が違う。低 noise / 高 noise / 多峰 /
  高次元 gradient など、それぞれに特化した config がある。ユーザーが問題性質を
  事前に判断するのは難しいので、**全 specialist を並列で走らせて最良を採用**する。

比較: mimir 単体 vs brunnir (4 泉の合議)
  | | 単体 mimir | mimir_brunnir (本 module) |
  |---|---|---|
  | 時間 | 1×time_budget | 1×time_budget (並列) |
  | CPU | 1 core | 4 core |
  | 当たり外れ | 弱点問題で全敗 | 1 specialist が当てれば勝ち |
  | 問題難易度検出 | 不可 | council variance で可視化 |

使い方:
    from twelve.agent.mimir_brunnir import mimir_brunnir
    r = mimir_brunnir(eval_fn, ranges, time_budget=60)
    print(r["best_params"], r["council"])  # 勝者 + 全員の結果
"""
from __future__ import annotations

import time
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed
)
from typing import Any, Callable, Optional, Sequence


DEFAULT_SPECIALISTS: list[dict] = [
    {
        "name": "default",
        "kwargs": {},
        "role": "低 noise / symbolic 多峰 (reigen 効く決定論問題)",
    },
    {
        "name": "lad",
        "kwargs": {"n_samples_per_eval": 20, "stochastic_aggregator": "median"},
        "role": "stochastic / 高 noise (LLM 生成 / RL rollout / Monte Carlo)",
    },
    {
        "name": "expensive",
        "kwargs": {"eval_cost_hint": 2.0},
        "role": "owl 全力 (L-BFGS + multi-start + random restart 5)",
    },
    {
        "name": "scipy-forced",
        "kwargs": {"eval_cost_hint": 2.0, "scipy_cascade_dim_threshold": 3},
        "role": "高次元 gradient (Rosenbrock 系 / basin-hopping 有利)",
    },
]


def _score_of(r: dict) -> float:
    """verified_score 優先、なければ best_score、どちらもなければ -inf."""
    v = r.get("verified_score")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    b = r.get("best_score")
    if b is not None:
        try:
            return float(b)
        except (TypeError, ValueError):
            pass
    return float("-inf")


def mimir_brunnir(
    eval_fn: Callable,
    param_ranges: Sequence[tuple],
    *,
    time_budget: float = 300.0,
    specialists: Optional[list[dict]] = None,
    executor: str = "process",
    experience_id: str = "council",
    verbose: bool = False,
    **extra_kwargs: Any,
) -> dict:
    """4 specialist mimir 並列実行、最良結果を返す.

    Args:
      eval_fn: f(params) -> float (pickleable 推奨、process 並列用)
      param_ranges: [(lo, hi), ...]
      time_budget: 各 specialist に渡す秒数 (並列なので wall time と同じ)
      specialists: custom specialist list。None で DEFAULT_SPECIALISTS (4 個)。
      executor: "process" (CPU 4 core 並列、デフォルト) /
                "thread" (pickle 不可 eval_fn / thread-safe 時のみ)
      experience_id: 名札 (内部で {specialist}_{id} に展開)
      extra_kwargs: 全 specialist に共通 kwargs (curated_measurements 等)

    Returns:
      勝者 specialist の mimir 返り値に以下キー追加:
        "specialist": 勝った specialist 名 (e.g. "lad")
        "specialist_role": その specialist の役割説明
        "council": [(name, score), ...] 降順、全員の結果
        "council_variance_std": best_score の std (問題難易度シグナル)
        "council_elapsed_s": wall time
        "n_specialists_ran": 成功した specialist 数
        "n_specialists_failed": 失敗した数
        "council_errors": 失敗 specialist の詳細

    Raises:
      RuntimeError: 全 specialist 失敗時
      ValueError: executor が "process"/"thread" 以外
    """
    from twelve.agent.mimir import mimir

    specs = specialists if specialists is not None else DEFAULT_SPECIALISTS

    if executor == "process":
        Executor = ProcessPoolExecutor
    elif executor == "thread":
        Executor = ThreadPoolExecutor
    else:
        raise ValueError(
            f"executor must be 'process' or 'thread', got {executor!r}"
        )

    t0 = time.time()
    council: list[dict] = []
    errors: list[dict] = []

    with Executor(max_workers=len(specs)) as ex:
        futures = {}
        for s in specs:
            name = s["name"]
            kwargs = dict(s.get("kwargs", {}))
            kwargs.update(extra_kwargs)
            kwargs["time_budget"] = time_budget
            kwargs["experience_id"] = f"{name}_{experience_id}"
            fut = ex.submit(mimir, eval_fn, param_ranges, **kwargs)
            futures[fut] = s

        for fut in as_completed(futures):
            s = futures[fut]
            try:
                r = fut.result()
                if not isinstance(r, dict):
                    raise TypeError(f"mimir returned non-dict: {type(r)}")
                r = dict(r)  # copy
                r["specialist"] = s["name"]
                r["specialist_role"] = s.get("role", "")
                council.append(r)
                if verbose:
                    print(f"[council] {s['name']:13} "
                          f"best={r.get('best_score'):.4f} "
                          f"verified={r.get('verified_score')} "
                          f"tool={r.get('tool_used')}")
            except Exception as e:
                errors.append({
                    "specialist": s["name"],
                    "error": f"{type(e).__name__}: {e}",
                })
                if verbose:
                    print(f"[council] {s['name']:13} FAILED: {e}")

    if not council:
        raise RuntimeError(
            f"All {len(specs)} specialists failed: {errors}"
        )

    # 勝者 = verified/best score 最大
    best = max(council, key=_score_of)

    # Council summary
    scores = [(r["specialist"], _score_of(r)) for r in council]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    # Variance signal: 問題難易度の自動検出
    vals = [s for _, s in scores if s != float("-inf")]
    if len(vals) >= 2:
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = variance ** 0.5
    else:
        std = 0.0

    best = dict(best)
    best["council"] = scores_sorted
    best["council_variance_std"] = float(std)
    best["council_elapsed_s"] = float(time.time() - t0)
    best["n_specialists_ran"] = len(council)
    best["n_specialists_failed"] = len(errors)
    best["council_errors"] = errors

    return best


__all__ = ["mimir_brunnir", "DEFAULT_SPECIALISTS"]
