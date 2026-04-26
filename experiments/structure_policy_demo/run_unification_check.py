"""structure_policy 一本化検証: 4 シナリオで stable と公平比較.

目的: structure_policy の速度優位が「curated seed 効果」なら、stable に同じ
seed を渡せば差が消える。差が残るなら structure_policy 固有の利点がある。
逆に structure_policy が overhead で stable に明確に負けるシナリオを探す。

シナリオ:
  A) 純連続 + curated なし     stable のテリトリー
  B) 純連続 + curated あり     公平条件
  C) eval_fn 重い (sleep 0.3s) probe overhead が visible になる
  D) 高次元 純連続 (50d)        stable の構造発見能力テスト

各で stable vs structure_policy 比較。
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


def f_hard_10d(p):
    p = np.asarray(p)
    sharp = math.exp(-100 * np.linalg.norm(p - 0.9) ** 2)
    broad = 0.55 * math.exp(-2.8 * np.linalg.norm(p - 0.3) ** 2)
    return float(sharp + broad)


def f_hard_50d(p):
    p = np.asarray(p)
    sharp = math.exp(-30 * np.linalg.norm(p - 0.9) ** 2)
    broad = 0.55 * math.exp(-1.0 * np.linalg.norm(p - 0.3) ** 2)
    return float(sharp + broad)


def make_slow(fn, sleep_s=0.3):
    def slow_fn(p):
        time.sleep(sleep_s)
        return fn(p)
    return slow_fn


def identity_decode(p):
    return {"values": list(map(float, p))}


def identity_detail(p, fn):
    return {"score": float(fn(p)), "policy": identity_decode(p)}


def random_seeds(n, dim, lo=-1.0, hi=1.5, seed=42):
    rng = np.random.default_rng(seed)
    return [list(map(float, rng.uniform(lo, hi, size=dim))) for _ in range(n)]


def run_pair(label, fn, bounds, dim, *, budget, curated=None, optimal_fit):
    print(f"\n  --- {label} ---", flush=True)

    # [stable]
    kw = {"executor": "thread", "verbose": False}
    if curated is not None:
        kw["curated_measurements"] = [
            {"params": s, "score": float(fn(s))} for s in curated
        ]
    t0 = time.time()
    r_a = mimir_odin_stable(fn, bounds, time_budget=budget, **kw)
    t_a = time.time() - t0
    fit_a = float(fn(r_a["best_params"]))

    # [structure_policy]
    seeds = curated if curated is not None else random_seeds(8, dim)
    detail = lambda p: identity_detail(p, fn)
    t0 = time.time()
    r_b = mimir_odin_structure_policy(
        fn, bounds,
        policy_points=seeds,
        param_decoder=identity_decode,
        detail_fn=detail,
        time_budget=budget,
        executor="thread",
        verbose=False,
    )
    t_b = time.time() - t0
    if r_b.get("aborted"):
        fit_b = float("nan")
        aborted = True
    else:
        fit_b = float(fn(r_b["best_params"]))
        aborted = False

    delta_fit = fit_b - fit_a
    delta_t = t_b - t_a
    print(f"    stable           : fit={fit_a:+.4f}  t={t_a:5.1f}s", flush=True)
    print(f"    structure_policy : fit={fit_b:+.4f}  t={t_b:5.1f}s "
          f"(Δfit={delta_fit:+.3f}  Δt={delta_t:+5.1f}s)",
          flush=True)
    return {
        "label": label, "optimal": optimal_fit,
        "stable": {"fit": fit_a, "t": t_a},
        "structure_policy": {"fit": fit_b, "t": t_b, "aborted": aborted},
        "delta_fit": delta_fit, "delta_t_s": delta_t,
    }


def main():
    print("=" * 78, flush=True)
    print(f"structure_policy 一本化検証: 4 シナリオで公平比較", flush=True)
    print("=" * 78, flush=True)

    bounds_10 = [(-1.0, 1.5)] * 10
    bounds_50 = [(-1.0, 1.5)] * 50
    OPTIMAL = 0.55  # broad center

    results = []

    # シナリオ A: 純連続 + curated なし
    print(f"\n[A] 純連続 10d + curated なし (stable のテリトリー)", flush=True)
    print(f"    structure_policy は内部で 8 random seed 自動生成", flush=True)
    r = run_pair("A: 純連続 curated なし", f_hard_10d, bounds_10, 10,
                 budget=30, curated=None, optimal_fit=OPTIMAL)
    results.append(r)

    # シナリオ B: 純連続 + curated あり (公平条件)
    print(f"\n[B] 純連続 10d + curated 8 seed (両者に同じ seed 渡す = 公平条件)",
          flush=True)
    seeds_10 = random_seeds(8, 10, seed=42)
    r = run_pair("B: 純連続 curated 公平", f_hard_10d, bounds_10, 10,
                 budget=30, curated=seeds_10, optimal_fit=OPTIMAL)
    results.append(r)

    # シナリオ C: eval_fn 重い (sleep 0.3s)
    print(f"\n[C] eval_fn slow (sleep 0.3s = LLM 模倣)、probe overhead 露出条件",
          flush=True)
    slow_fn = make_slow(f_hard_10d, sleep_s=0.3)
    r = run_pair("C: 重い eval_fn", slow_fn, bounds_10, 10,
                 budget=45, curated=None, optimal_fit=OPTIMAL)
    results.append(r)

    # シナリオ D: 50d 純連続
    print(f"\n[D] 高次元 50d 純連続 (proxy 構築困難な規模)",
          flush=True)
    r = run_pair("D: 50d 純連続", f_hard_50d, bounds_50, 50,
                 budget=60, curated=None, optimal_fit=OPTIMAL)
    results.append(r)

    # SUMMARY
    print("\n" + "=" * 78, flush=True)
    print(f"SUMMARY", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'scenario':<28} {'stable':>14} {'structure_pol':>14} {'Δt':>10}",
          flush=True)
    for r in results:
        sa, sb = r["stable"], r["structure_policy"]
        print(f"  {r['label']:<28} "
              f"fit={sa['fit']:+.3f} t={sa['t']:4.0f}s "
              f"fit={sb['fit']:+.3f} t={sb['t']:4.0f}s "
              f"{r['delta_t_s']:+5.0f}s", flush=True)

    print(f"\n  解釈:", flush=True)
    print(f"    Δt < 0   = structure_policy 速い (curated seed の warm-start 効果)",
          flush=True)
    print(f"    Δt ≈ 0   = ツール差なし (curated 効果が同等)", flush=True)
    print(f"    Δt > 0   = structure_policy が overhead 損 (probe 無駄)", flush=True)
    print(f"    Δfit ≠ 0 = ツール固有の探索能力差", flush=True)

    import json
    out_path = os.path.join(_HERE, "unification_check_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
