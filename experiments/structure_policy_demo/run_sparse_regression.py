"""Sparse Regression: 30 個のセンサーから本当に効く 5 個 + 係数を当てる.

問題:
  30 dim 中、真の active = 5 dim ([3, 7, 12, 19, 25])、各 +1.0 の係数。
  他 25 dim は無効、入力ノイズが乗る。
  目的: target = sum(coef[i] * x[i] for i in active) を最大化。
  制約: 累積 |x_i| ≤ 5.0 (sparsity 圧迫、5 以上に並べると penalty)。

比較:
  [A] mimir_odin_stable (純連続 30d 最適化)
       30 次元の連続値を直接探索。proxy 構築困難。
  [B] mimir_odin_structure_policy (構造 + 連続 同時探索)
       「上位 5 dim を keep + その値」を policy 化、構造 + 値の混合最適化。

期待:
  B が「真の active dim 5 個」を発見、A は 30d 全体を漫然と探索して劣る。
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from twelve.agent.mimir_odin_stable import mimir_odin_stable
from twelve.agent.mimir_odin_structure_policy import mimir_odin_structure_policy


DIM = 30
TRUE_ACTIVE = [3, 7, 12, 19, 25]
TRUE_COEF = +1.0
SPARSITY_BUDGET = 5.0  # |x_i| 累積上限
BOUNDS = [(-1.5, 1.5)] * DIM


def f_sparse(p):
    """目的: active dim の和を最大化、ただし sparsity_budget 越えで penalty."""
    p = np.asarray(p, dtype=float)
    main = sum(p[i] * TRUE_COEF for i in TRUE_ACTIVE)
    l1 = float(np.sum(np.abs(p)))
    penalty = max(0.0, l1 - SPARSITY_BUDGET) ** 2
    return float(main - penalty)


def policy_decode(p):
    """連続 params → 構造 policy (top-k by abs, keep, others zero).

    keep_k は p[0] の符号で動的決定 (3, 5, 7 のいずれか)。
    """
    p = np.asarray(p, dtype=float)
    abs_p = np.abs(p)
    # keep_k を symbolically: |p[0]| > 0.5 なら 7, > 0.2 なら 5, else 3
    keep_k = 7 if abs_p[0] > 0.5 else (5 if abs_p[0] > 0.2 else 3)
    top_idx = list(np.argsort(abs_p)[::-1][:keep_k])
    keep_mask = sorted(int(i) for i in top_idx)
    return {
        "keep_k": int(keep_k),
        "keep_dims": keep_mask,
        "values": [float(p[i]) for i in keep_mask],
    }


def policy_detail(p):
    """detail_fn: 構造 policy 用の practical metric (precision / recall) も返す."""
    policy = policy_decode(p)
    score = f_sparse(p)
    keep_set = set(policy["keep_dims"])
    true_set = set(TRUE_ACTIVE)
    tp = len(keep_set & true_set)
    fp = len(keep_set - true_set)
    fn = len(true_set - keep_set)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "score": score,
        "policy": policy,
        "true_active_recall": recall,
        "active_precision": precision,
        "n_kept": policy["keep_k"],
        "true_positive": tp,
    }


def evaluate(label, params, t):
    p = np.asarray(params, dtype=float)
    score = f_sparse(p)
    abs_p = np.abs(p)
    # トップ 5 を見る (active 推定)
    top5 = sorted(np.argsort(abs_p)[::-1][:5].tolist())
    tp = len(set(top5) & set(TRUE_ACTIVE))
    print(f"  {label}", flush=True)
    print(f"    score={score:+.3f}  t={t:.1f}s", flush=True)
    print(f"    top-5 dims={top5}", flush=True)
    print(f"    true active={TRUE_ACTIVE}", flush=True)
    print(f"    true positive={tp}/5  (recall {tp/5:.0%})", flush=True)
    print(f"    L1 norm={np.sum(abs_p):.2f} (budget {SPARSITY_BUDGET})", flush=True)
    return {"label": label, "score": score, "t": t, "top5": top5,
            "true_positive": tp, "params": list(map(float, p))}


def main():
    print("=" * 78, flush=True)
    print(f"Sparse Regression: 30d 中 真の active 5 dim を発見せよ", flush=True)
    print("=" * 78, flush=True)
    print(f"true active dims: {TRUE_ACTIVE}, coef={TRUE_COEF}", flush=True)
    print(f"sparsity budget: |x_i|累積 ≤ {SPARSITY_BUDGET}", flush=True)
    print(f"problem dim: {DIM}, bounds: {BOUNDS[0]}", flush=True)

    BUDGET = 60.0

    # [A] 純連続オーディン
    print(f"\n" + "=" * 78, flush=True)
    print(f"[A] mimir_odin_stable (30d 連続、構造発見なし)", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    r_a = mimir_odin_stable(
        f_sparse, BOUNDS,
        time_budget=BUDGET,
        executor="thread",
        verbose=False,
    )
    t_a = time.time() - t0
    res_a = evaluate("A: mimir_odin_stable", r_a["best_params"], t_a)

    # [B] structure_policy オーディン
    print(f"\n" + "=" * 78, flush=True)
    print(f"[B] mimir_odin_structure_policy (構造 + 連続 同時)", flush=True)
    print("=" * 78, flush=True)

    # seed policies: 何個か候補を渡して probe させる
    rng = np.random.default_rng(42)
    seed_policies = []
    for _ in range(8):
        p = rng.uniform(-1.0, 1.0, size=DIM)
        # 適度に sparse にする
        threshold = np.percentile(np.abs(p), 80)
        p[np.abs(p) < threshold] = 0.0
        seed_policies.append(list(map(float, p)))

    t0 = time.time()
    r_b = mimir_odin_structure_policy(
        f_sparse, BOUNDS,
        policy_points=seed_policies,
        param_decoder=policy_decode,
        detail_fn=policy_detail,
        objective_schema={"primary": "score", "penalty": "n_kept"},
        time_budget=BUDGET,
        executor="thread",
        verbose=False,
    )
    t_b = time.time() - t0

    if r_b.get("aborted"):
        print(f"  ABORTED: {r_b.get('abort_reason')}", flush=True)
        res_b = None
    else:
        best_pol = r_b.get("best_policy", {})
        best_pol_data = best_pol.get("policy") if isinstance(best_pol, dict) else None
        res_b = evaluate("B: structure_policy", r_b["best_params"], t_b)
        if best_pol_data:
            print(f"    decoded policy: keep_k={best_pol_data.get('keep_k')}  "
                  f"keep_dims={best_pol_data.get('keep_dims')}", flush=True)
            tp_pol = len(set(best_pol_data.get("keep_dims", [])) & set(TRUE_ACTIVE))
            print(f"    policy true positive: {tp_pol}/5", flush=True)
            res_b["policy_keep_dims"] = best_pol_data.get("keep_dims")
            res_b["policy_true_positive"] = tp_pol
        probe = r_b.get("policy_response_probe", {})
        print(f"    policy probe ok={probe.get('ok')}  "
              f"unique_policies={probe.get('unique_policies', 'n/a')}", flush=True)

    # VERDICT
    print(f"\n" + "=" * 78, flush=True)
    print(f"VERDICT", flush=True)
    print(f"=" * 78, flush=True)
    print(f"  [A] 純連続オーディン     : score={res_a['score']:+.3f}  "
          f"top-5 hit={res_a['true_positive']}/5  t={res_a['t']:.0f}s", flush=True)
    if res_b:
        print(f"  [B] structure_policy   : score={res_b['score']:+.3f}  "
              f"top-5 hit={res_b['true_positive']}/5  t={res_b['t']:.0f}s", flush=True)
        if "policy_keep_dims" in res_b:
            print(f"      policy decoded     : kept dims = {res_b['policy_keep_dims']}", flush=True)
            print(f"      policy true positive: {res_b['policy_true_positive']}/5", flush=True)
        diff = res_b["score"] - res_a["score"]
        print(f"\n  score 差 (B - A): {diff:+.3f}", flush=True)
        winner = "B (structure_policy)" if diff > 0.05 else (
                 "A (純連続)" if diff < -0.05 else "tie")
        print(f"  winner: {winner}", flush=True)

    import json
    out = {"A": res_a, "B": res_b, "true_active": TRUE_ACTIVE}
    out_path = os.path.join(_HERE, "sparse_regression_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
