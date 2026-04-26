"""TSP (純離散) + 純連続を両 ODIN で解いて、限界と利得を確認.

問題 1: TSP 10 cities (純離散順列、3.6M 順列)
  連続パラメータ → argsort で順列化 (DARTS 流の relaxation)。
  stable と structure_policy 両方で解けるが、structure_policy が利く期待。

問題 2: 10d sharp+broad trap (純連続)
  前の hard test と同じ。stable のテリトリー。
  structure_policy で解こうとすると overhead 損するかも。
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


# ---------- 問題 1: TSP ----------
N_CITIES = 10
_TSP_RNG = np.random.default_rng(2026)
CITIES = _TSP_RNG.uniform(0, 1, size=(N_CITIES, 2))
TSP_BOUNDS = [(-1.0, 1.0)] * N_CITIES


def tour_distance(order):
    return sum(np.linalg.norm(CITIES[order[i]] - CITIES[order[(i + 1) % N_CITIES]])
               for i in range(N_CITIES))


def f_tsp(p):
    p = np.asarray(p)
    order = np.argsort(p)
    return float(-tour_distance(order))  # 距離最小化 → score 最大化


def tsp_policy_decode(p):
    p = np.asarray(p)
    order = list(map(int, np.argsort(p)))
    return {
        "order": order,
        "distance": float(tour_distance(order)),
    }


def tsp_policy_detail(p):
    p = np.asarray(p)
    order = list(map(int, np.argsort(p)))
    dist = tour_distance(order)
    return {
        "score": float(-dist),
        "distance": float(dist),
        "policy": {"order": order},
        "n_cities": N_CITIES,
    }


def brute_force_optimal_tsp():
    """10 cities なら 9!/2 = 181440 経路で全探索可能 (city 0 固定 + 反転対称除去)."""
    from itertools import permutations
    best_dist = float("inf")
    best_order = None
    for perm in permutations(range(1, N_CITIES)):
        order = [0] + list(perm)
        d = tour_distance(order)
        if d < best_dist:
            best_dist = d
            best_order = order
    return best_order, best_dist


# ---------- 問題 2: 純連続 (10d sharp+broad trap) ----------
DIM_CONT = 10
SHARP = np.full(DIM_CONT, 0.9)
BROAD = np.full(DIM_CONT, 0.3)
CONT_BOUNDS = [(-1.0, 1.5)] * DIM_CONT


def f_hard(p):
    p = np.asarray(p)
    d_s = np.linalg.norm(p - SHARP)
    d_b = np.linalg.norm(p - BROAD)
    return float(math.exp(-100 * d_s ** 2) + 0.55 * math.exp(-2.8 * d_b ** 2))


def cont_policy_decode(p):
    """純連続を policy にしないので identity 的、全 dim keep."""
    p = np.asarray(p)
    return {"keep_dims": list(range(DIM_CONT)), "values": list(map(float, p))}


def cont_policy_detail(p):
    return {"score": float(f_hard(p)), "policy": cont_policy_decode(p)}


# ---------- 実行 ----------
def run_tsp():
    print("=" * 78, flush=True)
    print(f"問題 1: TSP {N_CITIES} cities (純離散順列、9!/2 = 181440 経路)", flush=True)
    print("=" * 78, flush=True)

    print("brute force で真の最適解を計算中...", flush=True)
    best_order, best_dist = brute_force_optimal_tsp()
    print(f"  optimal: order={best_order}  distance={best_dist:.4f}  score={-best_dist:+.4f}",
          flush=True)

    BUDGET = 60.0

    # [A] stable で順列発見できるか
    print(f"\n[A] mimir_odin_stable (連続値の argsort)", flush=True)
    t0 = time.time()
    r_a = mimir_odin_stable(f_tsp, TSP_BOUNDS, time_budget=BUDGET,
                            executor="thread", verbose=False)
    t_a = time.time() - t0
    a_score = float(f_tsp(r_a["best_params"]))
    a_dist = -a_score
    a_order = list(map(int, np.argsort(r_a["best_params"])))
    print(f"  score={a_score:+.4f}  distance={a_dist:.4f}  t={t_a:.1f}s",
          flush=True)
    print(f"  order={a_order}", flush=True)
    a_gap = a_dist - best_dist
    print(f"  gap from optimal: {a_gap:+.4f} (relative {a_gap/best_dist:+.1%})",
          flush=True)

    # [B] structure_policy with curated 順列 seeds
    print(f"\n[B] mimir_odin_structure_policy (順列 policy として扱う)", flush=True)
    rng = np.random.default_rng(7)
    seeds = []
    for _ in range(8):
        # ランダム 連続パラメータを seed として渡す
        seeds.append(list(map(float, rng.uniform(-1, 1, size=N_CITIES))))
    # 真値でない random だが、structure probe 用に十分

    t0 = time.time()
    r_b = mimir_odin_structure_policy(
        f_tsp, TSP_BOUNDS,
        policy_points=seeds,
        param_decoder=tsp_policy_decode,
        detail_fn=tsp_policy_detail,
        objective_schema={"primary": "score", "side": "distance"},
        time_budget=BUDGET,
        executor="thread",
        verbose=False,
    )
    t_b = time.time() - t0
    if r_b.get("aborted"):
        print(f"  ABORTED: {r_b.get('abort_reason')}", flush=True)
        b_score, b_dist, b_order, b_gap = float("nan"), float("nan"), None, float("nan")
    else:
        b_score = float(f_tsp(r_b["best_params"]))
        b_dist = -b_score
        b_order = list(map(int, np.argsort(r_b["best_params"])))
        b_gap = b_dist - best_dist
        print(f"  score={b_score:+.4f}  distance={b_dist:.4f}  t={t_b:.1f}s",
              flush=True)
        print(f"  order={b_order}", flush=True)
        print(f"  gap from optimal: {b_gap:+.4f} (relative {b_gap/best_dist:+.1%})",
              flush=True)

    return {
        "optimal_dist": best_dist, "optimal_order": best_order,
        "A": {"score": a_score, "dist": a_dist, "order": a_order, "t": t_a, "gap": a_gap},
        "B": {"score": b_score, "dist": b_dist, "order": b_order, "t": t_b, "gap": b_gap},
    }


def run_continuous():
    print("\n" + "=" * 78, flush=True)
    print(f"問題 2: 10d sharp+broad trap (純連続、stable のテリトリー)", flush=True)
    print("=" * 78, flush=True)
    print(f"  optimal: BROAD center 0.3 → fit 0.55 (sharp 0.9 は針穴)", flush=True)

    BUDGET = 40.0

    # [A] stable
    print(f"\n[A] mimir_odin_stable (本職)", flush=True)
    t0 = time.time()
    r_a = mimir_odin_stable(f_hard, CONT_BOUNDS, time_budget=BUDGET,
                            executor="thread", verbose=False)
    t_a = time.time() - t0
    a_fit = float(f_hard(r_a["best_params"]))
    print(f"  fit={a_fit:+.4f}  t={t_a:.1f}s", flush=True)

    # [B] structure_policy で純連続を解く
    print(f"\n[B] mimir_odin_structure_policy (純連続を policy として無理やり解く)",
          flush=True)
    rng = np.random.default_rng(7)
    seeds = []
    for _ in range(8):
        seeds.append(list(map(float, rng.uniform(-1, 1.5, size=DIM_CONT))))

    t0 = time.time()
    r_b = mimir_odin_structure_policy(
        f_hard, CONT_BOUNDS,
        policy_points=seeds,
        param_decoder=cont_policy_decode,
        detail_fn=cont_policy_detail,
        objective_schema={"primary": "score"},
        time_budget=BUDGET,
        executor="thread",
        verbose=False,
    )
    t_b = time.time() - t0
    if r_b.get("aborted"):
        print(f"  ABORTED: {r_b.get('abort_reason')}", flush=True)
        b_fit = float("nan")
    else:
        b_fit = float(f_hard(r_b["best_params"]))
        print(f"  fit={b_fit:+.4f}  t={t_b:.1f}s", flush=True)

    return {
        "A": {"fit": a_fit, "t": t_a},
        "B": {"fit": b_fit, "t": t_b},
    }


def main():
    print("=" * 78, flush=True)
    print(f"TSP (純離散) + 純連続 で stable と structure_policy を比較", flush=True)
    print("=" * 78, flush=True)

    tsp_res = run_tsp()
    cont_res = run_continuous()

    print("\n" + "=" * 78, flush=True)
    print(f"VERDICT", flush=True)
    print("=" * 78, flush=True)
    print(f"\n[TSP 純離散 (10 cities, optimal dist={tsp_res['optimal_dist']:.4f})]", flush=True)
    print(f"  [A] stable           : dist={tsp_res['A']['dist']:.4f}  "
          f"gap={tsp_res['A']['gap']:+.4f} ({tsp_res['A']['gap']/tsp_res['optimal_dist']:+.1%})  "
          f"t={tsp_res['A']['t']:.0f}s", flush=True)
    if tsp_res['B']['order']:
        print(f"  [B] structure_policy : dist={tsp_res['B']['dist']:.4f}  "
              f"gap={tsp_res['B']['gap']:+.4f} ({tsp_res['B']['gap']/tsp_res['optimal_dist']:+.1%})  "
              f"t={tsp_res['B']['t']:.0f}s", flush=True)

    print(f"\n[純連続 10d sharp+broad trap (optimal fit ≈ 0.55)]", flush=True)
    print(f"  [A] stable           : fit={cont_res['A']['fit']:+.4f}  "
          f"t={cont_res['A']['t']:.0f}s", flush=True)
    print(f"  [B] structure_policy : fit={cont_res['B']['fit']:+.4f}  "
          f"t={cont_res['B']['t']:.0f}s", flush=True)

    import json
    out_path = os.path.join(_HERE, "tsp_and_continuous_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tsp": tsp_res, "continuous": cont_res}, f, indent=2,
                  ensure_ascii=False, default=str)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
