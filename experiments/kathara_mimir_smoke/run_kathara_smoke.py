"""kathara_mimir 実機 smoke test.

12 task の似た 5d Rastrigin (center が円周上に並ぶ) を解く。
share-加速が効くか確認するため、Kathara 隣接で center が近い設定にする。

比較:
  [A] kathara_mimir (share あり、3 rounds)
  [B] kathara_mimir (share weight=0、share なしの 12 並列、reference)

期待:
  [A] のほうが mean fit が高い (share 加速 + 構造共有)。
  Wall time は同程度 (両方 12 並列)。
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

from twelve.agent.kathara_mimir import kathara_mimir, KATHARA_NEIGHBORS


DIM = 5
BOUNDS = [(-3.0, 3.0)] * DIM


def make_eval_fns():
    """12 task: each rastrigin centered at a 5d point on a ring.

    隣接 (Kathara 30 edges) は近い center、遠い node は遠い center。
    share が効くなら、隣接 task の解が互いに seed として有用になる。
    """
    rng = np.random.default_rng(7)
    centers = []
    # 12 角形を 5d に embed: 最初 2 dim で円周、残り 3 dim はランダム共通シフト
    base_extra = rng.uniform(-0.5, 0.5, size=DIM - 2)
    for i in range(12):
        theta = 2 * math.pi * i / 12
        c = np.zeros(DIM)
        c[0] = 1.5 * math.cos(theta)
        c[1] = 1.5 * math.sin(theta)
        c[2:] = base_extra + rng.uniform(-0.1, 0.1, size=DIM - 2)
        centers.append(c)
    centers = np.stack(centers)

    def make_fn(c):
        def f(p):
            x = np.asarray(p) - c
            return -(10 * DIM + np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)))
        return f
    return [make_fn(c) for c in centers], centers


def measure_run(label, eval_fns, **kwargs):
    print("=" * 78, flush=True)
    print(f"[{label}]", flush=True)
    print("=" * 78, flush=True)
    t0 = time.time()
    results = kathara_mimir(eval_fns, BOUNDS, verbose=False, **kwargs)
    elapsed = time.time() - t0
    fits = [r["best_score"] for r in results]
    print(f"  wall: {elapsed:.1f}s", flush=True)
    print(f"  mean fit: {np.mean(fits):+.3f}  "
          f"min: {np.min(fits):+.3f}  max: {np.max(fits):+.3f}", flush=True)
    print(f"  shares received per node (mean): "
          f"{np.mean([r.get('kathara_shares_received', 0) for r in results]):.1f}", flush=True)
    return {
        "label": label,
        "elapsed_s": elapsed,
        "fits": [float(f) for f in fits],
        "mean_fit": float(np.mean(fits)),
        "min_fit": float(np.min(fits)),
        "max_fit": float(np.max(fits)),
    }


def main():
    eval_fns, centers = make_eval_fns()

    print("=" * 78, flush=True)
    print(f"kathara_mimir smoke: 12 × 5d Rastrigin on 12-ring", flush=True)
    print(f"DIM={DIM}, bounds={BOUNDS[0]}, expected ideal fit -> 0", flush=True)
    print("=" * 78, flush=True)
    print(f"\nNeighbors of node 0: {KATHARA_NEIGHBORS[0]}", flush=True)
    print(f"  (Kathara 30 edges = Circulant(12, {{1,4,6}}))", flush=True)

    # quick sanity: midpoint score per task (baseline)
    midpoint = [(b[0] + b[1]) / 2 for b in BOUNDS]
    mid_scores = [eval_fns[i](midpoint) for i in range(12)]
    print(f"\n[reference] midpoint mean fit: {np.mean(mid_scores):+.2f}", flush=True)

    # [A] share あり (default、max_curated_per_node=6 で mimir の最低 5 件要件をクリア)
    a = measure_run("share=0.3, rounds=3", eval_fns,
                    time_budget=36.0, share_rounds=3, share_weight=0.3,
                    max_curated_per_node=6)

    # [B] share なし (rounds=1, share_weight=0)
    b = measure_run("share=0.0, rounds=1 (parallel only)", eval_fns,
                    time_budget=36.0, share_rounds=1, share_weight=0.0,
                    max_curated_per_node=6)

    print("\n" + "=" * 78, flush=True)
    print("VERDICT", flush=True)
    print("=" * 78, flush=True)
    diff = a["mean_fit"] - b["mean_fit"]
    print(f"  [A] share=0.3: mean_fit={a['mean_fit']:+.3f}  t={a['elapsed_s']:.1f}s", flush=True)
    print(f"  [B] share=0.0: mean_fit={b['mean_fit']:+.3f}  t={b['elapsed_s']:.1f}s", flush=True)
    print(f"  share gain   : {diff:+.3f}  ({'A wins' if diff > 0 else 'B wins'})", flush=True)

    import json
    out_path = os.path.join(_HERE, "kathara_smoke_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"share_on": a, "share_off": b,
                   "share_gain": float(diff)}, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
