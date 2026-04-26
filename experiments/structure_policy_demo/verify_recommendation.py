"""diagnostic API 拡張の徹底検証: 6 ケースで recommended_optimizer の正解率測定.

ケース毎に「この問題は X が正解」と先に書き、check_eval_fn が同じ推奨を
出すかをテスト。間違いがあれば判定ロジックを修正する材料にする。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from twelve.agent.eval_check import check_eval_fn, format_report


# ====================================================================
# Case 1: 純連続 smooth (LLM scale 模倣) — 期待: stable
# ====================================================================
def f_continuous_smooth(p):
    """10d 滑らかな Gaussian peak."""
    p = np.asarray(p)
    return float(math.exp(-0.5 * np.linalg.norm(p - 0.3) ** 2))


# ====================================================================
# Case 2: 整数化 (mask、丸めで潰れる) — 期待: structure_policy
# ====================================================================
def f_integer_mask(p):
    """連続 p を round して 5 個の dim を keep mask 化、その和を返す."""
    p = np.asarray(p)
    abs_p = np.abs(p)
    # 上位 round(p[0] * 3 + 5) 個を keep (整数化、丸めで潰れる)
    k = int(round(p[0] * 3 + 5))  # 2..8 の整数
    k = max(2, min(8, k))
    top_idx = np.argsort(abs_p)[::-1][:k]
    return float(sum(p[i] for i in top_idx))


# ====================================================================
# Case 3: stochastic (LLM eval 模倣) — 期待: stable + lad warning
# ====================================================================
_STOCH_RNG = np.random.default_rng(0)
def f_stochastic(p):
    p = np.asarray(p)
    base = math.exp(-0.5 * np.linalg.norm(p - 0.3) ** 2)
    return float(base + _STOCH_RNG.normal(0, 0.15))


# ====================================================================
# Case 4: TSP 風 argsort decode — 期待: structure_policy
# ====================================================================
_CITIES = np.random.default_rng(2026).uniform(0, 1, size=(8, 2))
def f_tsp(p):
    """連続 p を argsort で順列化、TSP 距離."""
    order = np.argsort(p)
    dist = sum(np.linalg.norm(_CITIES[order[i]] - _CITIES[order[(i+1) % 8]])
               for i in range(8))
    return float(-dist)


# ====================================================================
# Case 5: 純連続 sparse (sparse regression) — 期待: stable
# ====================================================================
TRUE_ACTIVE = [3, 7, 12, 19, 25]
def f_sparse(p):
    p = np.asarray(p)
    main = sum(p[i] for i in TRUE_ACTIVE)
    return float(main - 0.05 * sum(abs(p[j]) for j in range(len(p)) if j not in TRUE_ACTIVE))


# ====================================================================
# Case 6: constant (broken eval_fn) — 期待: fatal、推奨なし
# ====================================================================
def f_constant(p):
    return 0.42  # state leak / apply漏れ模倣


# ====================================================================
# Run
# ====================================================================
CASES = [
    ("Case 1: 純連続 smooth (10d Gaussian)",
     f_continuous_smooth, [(-1.0, 1.5)] * 10,
     "mimir_odin_stable", "純連続"),
    ("Case 2: 整数 mask (丸めで潰れる)",
     f_integer_mask, [(-1.5, 1.5)] * 10,
     "mimir_odin_structure_policy", "離散 mask"),
    ("Case 3: stochastic (Gaussian noise σ=0.15)",
     f_stochastic, [(-1.0, 1.5)] * 10,
     "mimir_odin_stable", "stochastic + lad warning"),
    ("Case 4: TSP argsort (8 cities)",
     f_tsp, [(-1.0, 1.0)] * 8,
     "mimir_odin_structure_policy", "順列構造"),
    ("Case 5: 純連続 sparse (30d)",
     f_sparse, [(-1.5, 1.5)] * 30,
     "mimir_odin_stable", "純連続 sparse"),
    ("Case 6: constant (broken)",
     f_constant, [(-1.0, 1.0)] * 5,
     None, "fatal、推奨保留"),
]


def main():
    print("=" * 80, flush=True)
    print("check_eval_fn 拡張 (recommended_optimizer) 徹底検証", flush=True)
    print("=" * 80, flush=True)

    results = []
    correct = 0
    total = 0
    for label, fn, ranges, expected, note in CASES:
        print(f"\n--- {label} ---", flush=True)
        print(f"  期待: {expected}  ({note})", flush=True)

        diag = check_eval_fn(fn, ranges, time_budget=15, verbose=False)
        actual = diag.get("recommended_optimizer")
        ok = (actual == expected)
        marker = "✓" if ok else "✗"
        print(f"  実測: {actual}  {marker}", flush=True)
        print(f"  reason: {diag.get('recommended_reason')}", flush=True)
        if diag.get("discreteness"):
            d = diag["discreteness"]
            print(f"  discreteness: n_unique={d['n_unique']}/{d['n_probes']}  "
                  f"spread={d['spread']:.2e}  likely={d['likely_discrete']}",
                  flush=True)
        if diag.get("stochastic_cv") is not None:
            print(f"  stochastic_cv: {diag['stochastic_cv']:.3f}  "
                  f"detected={diag['stochastic_detected']}", flush=True)
        print(f"  severity: {diag['severity']}  proxy_r2: {diag['proxy_r2']:.2f}",
              flush=True)

        if ok:
            correct += 1
        total += 1
        results.append({
            "case": label, "expected": expected, "actual": actual, "ok": ok,
            "severity": diag["severity"],
            "stochastic_detected": diag.get("stochastic_detected"),
            "likely_discrete": diag.get("likely_discrete"),
            "reason": diag.get("recommended_reason"),
        })

    # SUMMARY
    print("\n" + "=" * 80, flush=True)
    print(f"SUMMARY: {correct}/{total} 正解 ({100*correct/total:.0f}%)", flush=True)
    print("=" * 80, flush=True)
    for r in results:
        marker = "✓" if r["ok"] else "✗"
        print(f"  {marker} {r['case']}", flush=True)
        if not r["ok"]:
            print(f"      期待={r['expected']}  実測={r['actual']}", flush=True)
            print(f"      reason: {r['reason']}", flush=True)

    import json
    out = {"correct": correct, "total": total, "results": results}
    out_path = os.path.join(_HERE, "verify_recommendation_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
