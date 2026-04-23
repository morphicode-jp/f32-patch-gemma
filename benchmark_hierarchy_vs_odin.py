"""Hierarchy (Pattern 3) vs Council benchmark — sparse Rastrigin.

20-dim Rastrigin where only first 5 dims matter (other 15 are dead).
Tests whether hierarchy's Council-find-active → GA-in-reduced-space beats
full-space Council.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.mimir_cardinal import mimir_cardinal_hierarchy
from twelve.agent.mimir_odin import mimir_odin


DIM = 20
ACTIVE_DIM = 5
RANGE_LO, RANGE_HI = -5.12, 5.12
TIME_BUDGET = 60.0
SEEDS = [0, 1, 2]


def sparse_rastrigin(p):
    """Only first ACTIVE_DIM dims contribute; rest are dead."""
    active = p[:ACTIVE_DIM]
    base = 10 * ACTIVE_DIM + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in active)
    return -base


def run_council(seed: int) -> dict:
    t0 = time.time()
    r = mimir_odin(
        sparse_rastrigin,
        [(RANGE_LO, RANGE_HI)] * DIM,
        time_budget=TIME_BUDGET,
        executor="thread",
        experience_id=f"bench_council_s{seed}",
    )
    return {
        "method": "council",
        "seed": seed,
        "best_score": r.get("best_score"),
        "best_params": r.get("best_params"),
        "elapsed_s": time.time() - t0,
        "gap": -r.get("best_score", float("-inf")),
    }


def run_hierarchy(seed: int) -> dict:
    t0 = time.time()
    r = mimir_cardinal_hierarchy(
        sparse_rastrigin,
        [(RANGE_LO, RANGE_HI)] * DIM,
        time_budget=TIME_BUDGET,
        council_budget_share=0.25,
        ga_population=16,
        ga_generations=50,
        ga_mutation_sigma=0.15,
        ga_seed=seed,
        experience_id=f"bench_hier_s{seed}",
    )
    return {
        "method": "hierarchy",
        "seed": seed,
        "best_score": r["best_score"],
        "best_params": r["best_params"],
        "active_dims": r["active_dims"],
        "dead_dims": r["dead_dims"],
        "n_active": r["n_active"],
        "ga_generations_run": r["ga_generations_run"],
        "council_elapsed_s": r["council_elapsed_s"],
        "ga_elapsed_s": r["ga_elapsed_s"],
        "elapsed_s": r["elapsed_s"],
        "gap": -r["best_score"],
    }


def main():
    results = []
    for method_name, runner in [("council", run_council), ("hierarchy", run_hierarchy)]:
        for seed in SEEDS:
            print(f"[{method_name} seed={seed}] running ...", flush=True)
            res = runner(seed)
            print(f"  gap={res['gap']:.2f}  t={res['elapsed_s']:.1f}s  "
                  f"{'n_active=%d' % res.get('n_active', 0) if method_name == 'hierarchy' else ''}",
                  flush=True)
            results.append(res)

    print()
    print("=" * 60)
    print(f"Sparse Rastrigin 20d (only first {ACTIVE_DIM} active)")
    print("=" * 60)
    for method in ("council", "hierarchy"):
        rs = [r for r in results if r["method"] == method]
        gaps = [r["gap"] for r in rs]
        print(f"  {method:10s}: gap={statistics.mean(gaps):7.2f} ± "
              f"{statistics.stdev(gaps) if len(gaps) > 1 else 0:.2f}  "
              f"best={min(gaps):.2f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_hierarchy_vs_council.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"problem": f"sparse Rastrigin 20d, {ACTIVE_DIM} active",
                   "time_budget_s": TIME_BUDGET,
                   "seeds": SEEDS,
                   "results": results}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
