"""phase7_parallel_search.py - Simple parallel random search + local refine.

Design:
  No Sentinel, no Reigen, no owl. Just raw parallel evaluation of Dense-Prune
  configurations. For slow eval_fn (8 min each), framework overhead dominates,
  so direct search is cleaner.

Phases:
  1. Random sample 24 configs, eval in parallel (4 batches × 6 workers = 4 rounds)
  2. Take top-5, mutate around them → 12 more configs (2 rounds)
  3. Report best config

Total: ~6 rounds × 8 min = 48 min (vs Reigen's 13 hours).
"""
import os
import sys
import json
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import worker from phase7_mega (already pickleable)
from phase7_mega import worker_dense_prune


USER_RANGES = [
    (0.0, 1.0),    # dale_law
    (0.0, 1.0),    # hebbian
    (0.3, 0.9),    # reach_w
    (0.3, 0.8),    # prune_fraction
    (20.0, 80.0),  # stage1_budget
    (20.0, 80.0),  # stage3_budget
    (1.0, 3.0),    # n_cycles_s1
    (1.0, 3.0),    # n_cycles_s3
]
USER_NAMES = [
    "dale_law", "hebbian", "reach_w", "prune_fraction",
    "stage1_budget", "stage3_budget", "n_cycles_s1", "n_cycles_s3",
]


def sample_random(n, seed=42):
    rng = random.Random(seed)
    configs = []
    for _ in range(n):
        configs.append(tuple(rng.uniform(lo, hi) for lo, hi in USER_RANGES))
    return configs


def mutate(config, sigma=0.15, seed=7):
    rng = random.Random(seed)
    out = []
    for (v, (lo, hi)) in zip(config, USER_RANGES):
        step = rng.gauss(0, (hi - lo) * sigma)
        new_v = max(lo, min(hi, v + step))
        out.append(new_v)
    return tuple(out)


def batch_evaluate(configs, workers=6):
    """Evaluate configs in parallel. Returns list of result dicts (same order)."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_dense_prune, c) for c in configs]
        results = [f.result() for f in futures]
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        if "error" in r:
            print(f"    [{i}] ERROR: {r['error'][:100]}")
        else:
            print(f"    [{i}] d={int(r['dale'])}/h={int(r['hebb'])}/"
                  f"rw={r['reach_w']:.2f}/pf={r['prune_frac']:.2f}/"
                  f"s1={r['s1_budget']}/s3={r['s3_budget']}/"
                  f"c1={r['n_cyc1']}/c3={r['n_cyc3']} "
                  f"→ reach={r['reach_n3']*100:.0f}% "
                  f"tom={r['tom_pair']*100:.0f}% "
                  f"scales={r['scales_n5']*100:.0f}% "
                  f"C={r['composite']:.3f}")
    print(f"    batch elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return results


def main():
    print("=" * 70)
    print("  PHASE 7 PARALLEL SEARCH (no framework, raw batch×6)")
    print("=" * 70)

    all_results = []
    t_all = time.time()

    # ============================================================
    # Phase 1: Random exploration (24 configs = 4 rounds × 6)
    # ============================================================
    print(f"\n[Phase 1] Random exploration: 24 configs")
    random_configs = sample_random(24)
    BATCH = 6
    for batch_idx in range(0, len(random_configs), BATCH):
        print(f"\n  Round {batch_idx//BATCH + 1}/{len(random_configs)//BATCH}:")
        batch = random_configs[batch_idx:batch_idx + BATCH]
        results = batch_evaluate(batch, workers=BATCH)
        all_results.extend(results)

    # Take top-5 so far
    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda r: -r.get("composite", 0))
    top5 = valid[:5]
    print(f"\n[Phase 1 top-5]")
    for i, r in enumerate(top5):
        print(f"  #{i+1}: reach={r['reach_n3']*100:.0f}% "
              f"tom={r['tom_pair']*100:.0f}% C={r['composite']:.3f} "
              f"(dale={int(r['dale'])} hebb={int(r['hebb'])} "
              f"rw={r['reach_w']:.2f} pf={r['prune_frac']:.2f})")

    # ============================================================
    # Phase 2: Mutate top-5 → 15 refined configs (3 rounds × 6 = ~18 but we do 12)
    # ============================================================
    print(f"\n[Phase 2] Mutate top-5 × 3 each = 15 refined configs")
    refined = []
    for i, r in enumerate(top5):
        base = (r["dale"], r["hebb"], r["reach_w"], r["prune_frac"],
                r["s1_budget"], r["s3_budget"], r["n_cyc1"], r["n_cyc3"])
        base_tuple = tuple(float(v) for v in base)
        for j in range(3):
            refined.append(mutate(base_tuple, sigma=0.12, seed=100*i + j))

    for batch_idx in range(0, len(refined), BATCH):
        print(f"\n  Refine round {batch_idx//BATCH + 1}/{(len(refined)+BATCH-1)//BATCH}:")
        batch = refined[batch_idx:batch_idx + BATCH]
        results = batch_evaluate(batch, workers=BATCH)
        all_results.extend(results)

    # ============================================================
    # Final: best overall
    # ============================================================
    total = time.time() - t_all
    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda r: -r.get("composite", 0))
    best = valid[0] if valid else None

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS (total {total:.0f}s = {total/60:.1f} min)")
    print(f"  Total evals: {len(all_results)}, valid: {len(valid)}")
    print(f"{'='*70}")

    if best:
        print(f"\n  BEST CONFIG:")
        print(f"    dale_law     = {'ON' if best['dale'] else 'OFF'}")
        print(f"    hebbian      = {'ON' if best['hebb'] else 'OFF'}")
        print(f"    reach_w      = {best['reach_w']:.2f}")
        print(f"    prune_frac   = {best['prune_frac']:.2f}")
        print(f"    s1_budget    = {best['s1_budget']}s")
        print(f"    s3_budget    = {best['s3_budget']}s")
        print(f"    n_cycles     = s1:{best['n_cyc1']}  s3:{best['n_cyc3']}")
        print(f"    n_edges_kept = {best['n_kept']}")
        print(f"\n  METRICS:")
        print(f"    reach (N=3):     {best['reach_n3']*100:.0f}%")
        print(f"    ToM pairs:       {best['tom_pair']*100:.0f}%")
        print(f"    scales to N=5:   {best['scales_n5']*100:.0f}%")
        print(f"    composite:       {best['composite']:.3f}")

        print(f"\n  COMPARISON TO BASELINES:")
        print(f"    Phase 7 original N=5:            reach 7%")
        print(f"    Dense-Prune N=3 (yesterday):     reach 20%, C 0.388")
        print(f"    Reigen-batch peak (yesterday):   C 0.401")
        print(f"    This search best:                reach {best['reach_n3']*100:.0f}%, C {best['composite']:.3f}")

    # Save
    out = {
        "total_elapsed_s": round(total, 1),
        "n_evals": len(all_results),
        "n_valid": len(valid),
        "best": best,
        "top5": top5,
        "all_results": all_results,
    }
    with open("phase7_parallel_search_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_parallel_search_result.json")


if __name__ == "__main__":
    main()
