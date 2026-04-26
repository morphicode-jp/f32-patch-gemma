"""Sparse Regression 100d: 純連続オーディンが proxy 崩壊する規模で差を露出.

問題: 100 dim 中、真の active 10 dim をランダム選択。
  目的: target = sum(x[i] for i in active) - 0.1 * sum(|x_j| for j not active)
        + sparsity 予算超過 penalty
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


DIM = 100
_RNG_SETUP = np.random.default_rng(2026)
TRUE_ACTIVE = sorted(_RNG_SETUP.choice(DIM, 10, replace=False).tolist())
SPARSITY_BUDGET = 10.0
BOUNDS = [(-1.5, 1.5)] * DIM


def f_sparse(p):
    p = np.asarray(p, dtype=float)
    main = sum(p[i] for i in TRUE_ACTIVE)
    l1 = float(np.sum(np.abs(p)))
    penalty = max(0.0, l1 - SPARSITY_BUDGET) ** 2
    noise_drag = 0.05 * sum(abs(p[j]) for j in range(DIM) if j not in TRUE_ACTIVE)
    return float(main - penalty - noise_drag)


def policy_decode(p):
    p = np.asarray(p, dtype=float)
    abs_p = np.abs(p)
    # 動的 keep_k: |p[0]| signal で 5/10/15 を選ぶ
    if abs_p[0] > 0.7:
        keep_k = 15
    elif abs_p[0] > 0.3:
        keep_k = 10
    else:
        keep_k = 5
    top_idx = list(np.argsort(abs_p)[::-1][:keep_k])
    keep_dims = sorted(int(i) for i in top_idx)
    return {
        "keep_k": int(keep_k),
        "keep_dims": keep_dims,
        "values": [float(p[i]) for i in keep_dims],
    }


def policy_detail(p):
    score = f_sparse(p)
    policy = policy_decode(p)
    keep_set = set(policy["keep_dims"])
    true_set = set(TRUE_ACTIVE)
    tp = len(keep_set & true_set)
    return {
        "score": score,
        "policy": policy,
        "true_positive": tp,
        "n_kept": policy["keep_k"],
    }


def evaluate(label, params, t):
    p = np.asarray(params, dtype=float)
    score = f_sparse(p)
    abs_p = np.abs(p)
    top10 = sorted(np.argsort(abs_p)[::-1][:10].tolist())
    tp = len(set(top10) & set(TRUE_ACTIVE))
    print(f"  {label}", flush=True)
    print(f"    score={score:+.3f}  t={t:.1f}s", flush=True)
    print(f"    top-10 dims={top10}", flush=True)
    print(f"    true positive={tp}/10  (recall {tp/10:.0%})", flush=True)
    print(f"    L1 norm={np.sum(abs_p):.2f} (budget {SPARSITY_BUDGET})", flush=True)
    return {"label": label, "score": score, "t": t, "top10": top10,
            "true_positive": tp}


def main():
    print("=" * 78, flush=True)
    print(f"Sparse Regression 100d: active 10 dim 発見、純連続 vs structure_policy",
          flush=True)
    print("=" * 78, flush=True)
    print(f"true active dims: {TRUE_ACTIVE}", flush=True)
    print(f"sparsity budget: {SPARSITY_BUDGET}", flush=True)
    print(f"problem dim: {DIM}", flush=True)

    BUDGET = 120.0  # 100 dim は時間要る

    # [A] 純連続
    print(f"\n" + "=" * 78, flush=True)
    print(f"[A] mimir_odin_stable (100d 連続)", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    r_a = mimir_odin_stable(
        f_sparse, BOUNDS,
        time_budget=BUDGET,
        executor="thread",
        verbose=False,
    )
    t_a = time.time() - t0
    res_a = evaluate("A", r_a["best_params"], t_a)

    # [B] structure_policy with curated sparse seeds
    print(f"\n" + "=" * 78, flush=True)
    print(f"[B] mimir_odin_structure_policy (sparse 仮定で warm-start)", flush=True)
    print("=" * 78, flush=True)

    rng = np.random.default_rng(42)
    seed_policies = []
    for _ in range(12):
        p = np.zeros(DIM)
        # ランダム 10 dim を選んで [-1, 1] 値、他 0
        idx = rng.choice(DIM, 10, replace=False)
        p[idx] = rng.uniform(-1.0, 1.0, size=10)
        seed_policies.append(list(map(float, p)))

    t0 = time.time()
    r_b = mimir_odin_structure_policy(
        f_sparse, BOUNDS,
        policy_points=seed_policies,
        param_decoder=policy_decode,
        detail_fn=policy_detail,
        objective_schema={"primary": "score", "side": "true_positive"},
        time_budget=BUDGET,
        executor="thread",
        verbose=False,
    )
    t_b = time.time() - t0

    if r_b.get("aborted"):
        print(f"  ABORTED: {r_b.get('abort_reason')}", flush=True)
        res_b = None
    else:
        res_b = evaluate("B", r_b["best_params"], t_b)
        best_pol = r_b.get("best_policy", {})
        if isinstance(best_pol, dict):
            pol_data = best_pol.get("policy")
            if pol_data:
                tp_pol = len(set(pol_data.get("keep_dims", [])) & set(TRUE_ACTIVE))
                print(f"    decoded policy: keep_k={pol_data.get('keep_k')}  "
                      f"keep_dims={pol_data.get('keep_dims')}", flush=True)
                print(f"    policy true positive: {tp_pol}/10", flush=True)
                res_b["policy_keep_dims"] = pol_data.get("keep_dims")
                res_b["policy_true_positive"] = tp_pol

    print(f"\n" + "=" * 78, flush=True)
    print(f"VERDICT", flush=True)
    print(f"=" * 78, flush=True)
    print(f"  [A] 純連続オーディン     : score={res_a['score']:+.3f}  "
          f"top-10 hit={res_a['true_positive']}/10  t={res_a['t']:.0f}s", flush=True)
    if res_b:
        print(f"  [B] structure_policy   : score={res_b['score']:+.3f}  "
              f"top-10 hit={res_b['true_positive']}/10  t={res_b['t']:.0f}s",
              flush=True)
        if "policy_keep_dims" in res_b:
            print(f"      policy decoded     : kept dims = {res_b['policy_keep_dims']}",
                  flush=True)
            print(f"      policy true positive: {res_b['policy_true_positive']}/10",
                  flush=True)
        diff = res_b["score"] - res_a["score"]
        winner = "B" if diff > 0.05 else ("A" if diff < -0.05 else "tie")
        print(f"\n  score 差 (B - A): {diff:+.3f}  winner: {winner}", flush=True)

    import json
    out = {"A": res_a, "B": res_b, "true_active": TRUE_ACTIVE,
           "dim": DIM, "budget_each": BUDGET}
    out_path = os.path.join(_HERE, "sparse_100d_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
