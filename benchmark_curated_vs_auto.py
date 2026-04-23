"""A/B benchmark: mimir auto vs curated vs legacy — stochastic Rastrigin 10d.

Settles the question: does curated actually help enough to change the default
workflow, or is auto mode (with LaD n_samples_per_eval) already sufficient?

Conditions:
  A  = mimir(fn, ranges, n_samples_per_eval=20)               # auto + LaD
  B  = mimir(fn, ranges, curated_measurements=past,
             n_samples_per_eval=20)                            # curated + LaD
  C  = mimir(fn, ranges)                                        # legacy auto (no LaD)

Problem: 10d Rastrigin + additive gauss noise(0, 2.0).
True optimum: x=[0]*10, f=0 (unnoised).
Range: [-5.12, 5.12]^10.
Curated for B: 20 points placed in [-2, 2]^10 ("expert knows bowl is near origin"),
each with N=20 aggregation pre-measured.

Output: JSON with per-run best_score, verified_score, proxy_r2, tool_used,
        eval_count, elapsed_s. Summary with mean ± std per condition.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.mimir import mimir


# -----------------------------------------------------------------
# problem: stochastic Rastrigin 10d
# -----------------------------------------------------------------

DIM = 10
RANGE_LO, RANGE_HI = -5.12, 5.12
NOISE_STD = 2.0
TIME_BUDGET = 60.0
SEEDS = [0, 1, 2]


def rastrigin_noisy_factory(seed: int):
    """Create a stochastic Rastrigin with seeded RNG. mimir maximizes → negate."""
    rng = random.Random(seed * 9973 + 31)

    def eval_fn(p):
        base = 10 * len(p) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in p)
        noise = rng.gauss(0, NOISE_STD)
        return -(base + noise)  # higher = better

    return eval_fn


# -----------------------------------------------------------------
# curated: 20 informed points in [-2, 2]^10, each N=20 aggregated
# -----------------------------------------------------------------

def build_curated(eval_fn, seed: int, n_points: int = 20, n_samples: int = 20,
                  curated_range: tuple = (-2.0, 2.0)) -> list[dict]:
    """Simulate a domain expert: sample in reduced range, aggregate N calls per point."""
    rng = random.Random(seed + 7777)
    lo, hi = curated_range
    curated = []
    for _ in range(n_points):
        p = [rng.uniform(lo, hi) for _ in range(DIM)]
        vals = [eval_fn(p) for _ in range(n_samples)]
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        median = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
        curated.append({"params": list(p), "score": float(median)})
    return curated


# -----------------------------------------------------------------
# true-fn verifier (no noise) — post-hoc evaluate how good the found params are
# -----------------------------------------------------------------

def true_rastrigin(p):
    base = 10 * len(p) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in p)
    return -base


# -----------------------------------------------------------------
# run one condition
# -----------------------------------------------------------------

def run_condition(cond: str, seed: int) -> dict:
    eval_fn = rastrigin_noisy_factory(seed)
    ranges = [(RANGE_LO, RANGE_HI)] * DIM
    experience_id = f"curated_bench_{cond}_s{seed}"

    t0 = time.time()
    if cond == "A":
        r = mimir(eval_fn, ranges,
                  n_samples_per_eval=20,
                  time_budget=TIME_BUDGET,
                  experience_id=experience_id)
    elif cond == "B":
        past = build_curated(eval_fn, seed)
        r = mimir(eval_fn, ranges,
                  curated_measurements=past,
                  n_samples_per_eval=20,
                  time_budget=TIME_BUDGET,
                  experience_id=experience_id)
    elif cond == "C":
        r = mimir(eval_fn, ranges,
                  time_budget=TIME_BUDGET,
                  experience_id=experience_id)
    else:
        raise ValueError(f"unknown cond: {cond}")
    elapsed = time.time() - t0

    best_p = r.get("best_params")
    true_score = true_rastrigin(best_p) if best_p is not None else float("-inf")

    return {
        "cond": cond,
        "seed": seed,
        "best_params": list(best_p) if best_p else None,
        "best_score_noisy": float(r.get("best_score", float("nan"))),
        "true_score": float(true_score),
        "gap_to_optimum": float(-true_score),  # optimum is 0, so gap = -true
        "proxy_r2": float(r.get("proxy_r2") or 0.0),
        "tool_used": r.get("tool_used"),
        "route": r.get("route"),
        "n_measurements": int(r.get("n_measurements") or 0),
        "elapsed_s": float(elapsed),
        "dead_dims": len(r.get("dead_dims") or []),
        "active_dims": len(r.get("active_dims") or []),
    }


# -----------------------------------------------------------------
# summarize
# -----------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    summary: dict[str, Any] = {}
    for cond in ("A", "B", "C"):
        rows = [r for r in results if r["cond"] == cond]
        gaps = [r["gap_to_optimum"] for r in rows]
        r2s = [r["proxy_r2"] for r in rows]
        ns = [r["n_measurements"] for r in rows]
        ts = [r["elapsed_s"] for r in rows]
        summary[cond] = {
            "gap_mean": statistics.mean(gaps),
            "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            "gap_best": min(gaps),
            "proxy_r2_mean": statistics.mean(r2s),
            "n_meas_mean": statistics.mean(ns),
            "elapsed_mean": statistics.mean(ts),
            "n_runs": len(rows),
        }
    return summary


def main():
    results = []
    total = len(SEEDS) * 3
    i = 0
    for cond in ("A", "B", "C"):
        for seed in SEEDS:
            i += 1
            print(f"[{i}/{total}] cond={cond} seed={seed} ...", flush=True)
            try:
                res = run_condition(cond, seed)
                print(f"  gap={res['gap_to_optimum']:.2f} "
                      f"proxy_r2={res['proxy_r2']:.3f} "
                      f"n_meas={res['n_measurements']} "
                      f"tool={res['tool_used']} "
                      f"t={res['elapsed_s']:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                res = {"cond": cond, "seed": seed, "error": str(e),
                       "gap_to_optimum": float("inf"), "proxy_r2": 0.0,
                       "n_measurements": 0, "elapsed_s": 0.0}
            results.append(res)

    summary = summarize(results)

    out = {
        "problem": "stochastic Rastrigin 10d, noise=gauss(0, 2.0)",
        "range": [RANGE_LO, RANGE_HI],
        "time_budget_s": TIME_BUDGET,
        "seeds": SEEDS,
        "conditions": {
            "A": "mimir(fn, ranges, n_samples_per_eval=20)",
            "B": "mimir(fn, ranges, curated_measurements=past_20, n_samples_per_eval=20)",
            "C": "mimir(fn, ranges)  # legacy auto, no LaD",
        },
        "summary": summary,
        "runs": results,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_curated_vs_auto.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("SUMMARY (lower gap = closer to optimum, optimum=0)")
    print("=" * 70)
    for cond, s in summary.items():
        label = {"A": "auto+LaD", "B": "curated+LaD", "C": "legacy"}[cond]
        print(f"  {cond} ({label:12}): "
              f"gap={s['gap_mean']:>7.2f} ± {s['gap_std']:.2f}  "
              f"proxy_r2={s['proxy_r2_mean']:.3f}  "
              f"n_meas={s['n_meas_mean']:.0f}  "
              f"t={s['elapsed_mean']:.1f}s")

    print()
    print(f"Full results: {out_path}")


if __name__ == "__main__":
    main()
