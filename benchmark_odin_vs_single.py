"""Council vs single-mimir benchmark — same problem as A/B/C, same seeds.

Adds condition D = mimir_odin (4 specialists parallel) to the previous
A/B/C runs on stochastic Rastrigin 10d.

Runs both noise regimes (sigma=2 low, sigma=30 high) to see if council fixes
the weakness each was showing:
  - Low noise: legacy was winning — does council still win or does LaD drag it?
  - High noise: LaD had high variance — does council stabilize?

If council mean ≤ min(A, B, C) mean on BOTH, it's a clean win.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.mimir_odin import mimir_odin


DIM = 10
RANGE_LO, RANGE_HI = -5.12, 5.12
TIME_BUDGET = 60.0
SEEDS = [0, 1, 2]

# Module-level RNG state (pickle-safe for ProcessPool)
_RNG_STATES: dict = {}


def rastrigin_noisy(p, seed_id=None, noise_std=2.0):
    """Module-level fn, picklable. Uses per-seed RNG from shared dict."""
    if seed_id not in _RNG_STATES:
        _RNG_STATES[seed_id] = random.Random(seed_id * 9973 + 31)
    rng = _RNG_STATES[seed_id]
    base = 10 * len(p) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in p)
    noise = rng.gauss(0, noise_std)
    return -(base + noise)


# Use functools.partial-equivalent via closures for each seed
# But closures aren't picklable — use thread executor instead of process
def make_noisy(seed: int, noise_std: float):
    rng = random.Random(seed * 9973 + 31)

    def f(p):
        base = 10 * len(p) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in p)
        return -(base + rng.gauss(0, noise_std))

    return f


def true_rastrigin(p):
    return -(10 * len(p) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in p))


def run_council(seed: int, noise_std: float) -> dict:
    eval_fn = make_noisy(seed, noise_std)
    ranges = [(RANGE_LO, RANGE_HI)] * DIM

    t0 = time.time()
    # Thread executor because eval_fn is closure (not picklable)
    r = mimir_odin(
        eval_fn, ranges,
        time_budget=TIME_BUDGET,
        executor="thread",
        experience_id=f"council_bench_n{int(noise_std)}_s{seed}",
    )
    elapsed = time.time() - t0

    best_p = r.get("best_params")
    true_score = true_rastrigin(best_p) if best_p is not None else float("-inf")

    return {
        "seed": seed,
        "noise_std": noise_std,
        "best_params": list(best_p) if best_p else None,
        "best_score_noisy": float(r.get("best_score", float("nan"))),
        "true_score": float(true_score),
        "gap_to_optimum": float(-true_score),
        "winner_specialist": r.get("specialist"),
        "council_scores": r.get("council"),
        "council_variance_std": r.get("council_variance_std"),
        "council_elapsed_s": r.get("council_elapsed_s"),
        "n_specialists_ran": r.get("n_specialists_ran"),
        "elapsed_s": float(elapsed),
    }


def main():
    results: dict[str, list] = {"low_noise": [], "high_noise": []}
    i = 0
    total = len(SEEDS) * 2
    for noise_std, label in [(2.0, "low_noise"), (30.0, "high_noise")]:
        for seed in SEEDS:
            i += 1
            print(f"[{i}/{total}] noise={noise_std} seed={seed} ...", flush=True)
            try:
                res = run_council(seed, noise_std)
                print(f"  gap={res['gap_to_optimum']:.2f} "
                      f"winner={res['winner_specialist']} "
                      f"variance_std={res['council_variance_std']:.2f} "
                      f"t={res['elapsed_s']:.1f}s",
                      flush=True)
                print(f"  council: {res['council_scores']}", flush=True)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
                res = {"seed": seed, "noise_std": noise_std, "error": str(e),
                       "gap_to_optimum": float("inf"),
                       "winner_specialist": None}
            results[label].append(res)

    out = {
        "problem": "stochastic Rastrigin 10d",
        "range": [RANGE_LO, RANGE_HI],
        "time_budget_s": TIME_BUDGET,
        "seeds": SEEDS,
        "council_specialists": ["default", "lad", "expensive", "scipy-forced"],
        "results": results,
    }

    # Summary
    out["summary"] = {}
    for label, runs in results.items():
        gaps = [r["gap_to_optimum"] for r in runs]
        out["summary"][label] = {
            "gap_mean": statistics.mean(gaps),
            "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            "gap_best": min(gaps),
            "n_runs": len(runs),
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_council_vs_single.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 70)
    print("COUNCIL RESULTS (compare to A=auto, B=curated, C=default)")
    print("=" * 70)
    print(f"  low noise (σ=2):")
    print(f"    A previous: gap=  86.44 ± 8.72")
    print(f"    B previous: gap= 102.62 ± 89.64")
    print(f"    C previous: gap=  31.04 ± 53.76")
    s = out["summary"]["low_noise"]
    print(f"    D council : gap= {s['gap_mean']:7.2f} ± {s['gap_std']:.2f}  "
          f"best={s['gap_best']:.2f}")
    print(f"  high noise (σ=30):")
    print(f"    A previous: gap=  66.61 ± 63.30")
    print(f"    B previous: gap= 201.21 ± 44.21")
    print(f"    C previous: gap=  89.53 ± 21.38")
    s = out["summary"]["high_noise"]
    print(f"    D council : gap= {s['gap_mean']:7.2f} ± {s['gap_std']:.2f}  "
          f"best={s['gap_best']:.2f}")
    print()
    print(f"Full results: {out_path}")


if __name__ == "__main__":
    main()
