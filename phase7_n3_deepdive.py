"""phase7_n3_deepdive.py - Focused N=3 refinement around parallel-search best.

Previous best (2026-04-19 parallel search, reach 28%):
  dale=ON, hebb=OFF, reach_w=0.44, prune_frac=0.44
  s1=25s, s3=34s, n_cycles=(1,2), 67/120 edges

Deepdive strategy:
  - Fix dale=ON, hebb=OFF (winner dims) → drop 2 user-params
  - Narrow reach_w, prune_frac around 0.44 (±0.1)
  - WIDEN budgets: s1/s3 50-120s (2-4x longer → more training)
  - Cycles 2-3 (top configs favored more cycles)
  - Goal: break 28% ceiling via longer training

Phases:
  1. Verify: re-run current best 3 times (robust estimate)
  2. Narrow-search: 24 configs around best, 4 rounds × 6
  3. Refine: top-3 mutated, 12 configs, 2 rounds
  Total: ~39 evals, 90-120 min expected
"""
import os
import sys
import json
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase7_mega import worker_dense_prune


# Fixed winners from parallel-search
DALE = 1.0   # ON
HEBB = 0.0   # OFF

# Narrow ranges around parallel-search best
DEEPDIVE_RANGES = [
    (0.35, 0.55),   # reach_w (was 0.44)
    (0.35, 0.55),   # prune_frac (was 0.44)
    (50.0, 120.0),  # stage1_budget (was 25 → WIDEN upward for more training)
    (50.0, 120.0),  # stage3_budget (was 34 → WIDEN upward)
    (2.0, 3.0),     # n_cycles_s1 (was 1; push higher)
    (2.0, 3.0),     # n_cycles_s3 (was 2)
]
RANGE_NAMES = [
    "reach_w", "prune_frac",
    "stage1_budget", "stage3_budget",
    "n_cycles_s1", "n_cycles_s3",
]


def make_config(user6):
    """Expand 6-dim user space to 8-tuple for worker_dense_prune."""
    rw, pf, s1, s3, c1, c3 = user6
    return (DALE, HEBB, rw, pf, s1, s3, c1, c3)


def sample_random(n, seed=42):
    rng = random.Random(seed)
    configs = []
    for _ in range(n):
        u = tuple(rng.uniform(lo, hi) for lo, hi in DEEPDIVE_RANGES)
        configs.append(make_config(u))
    return configs


def mutate(config_8, sigma=0.10, seed=7):
    """Mutate only the user 6 dims (indices 2-7), preserve DALE/HEBB."""
    rng = random.Random(seed)
    out = list(config_8)
    for idx, (lo, hi) in zip(range(2, 8), DEEPDIVE_RANGES):
        v = out[idx]
        step = rng.gauss(0, (hi - lo) * sigma)
        out[idx] = max(lo, min(hi, v + step))
    return tuple(out)


def batch_evaluate(configs, workers=6, tag=""):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_dense_prune, c) for c in configs]
        results = [f.result() for f in futures]
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        if "error" in r:
            print(f"    [{tag}{i}] ERROR: {r['error'][:100]}", flush=True)
        else:
            print(f"    [{tag}{i}] d={int(r['dale'])}/h={int(r['hebb'])}/"
                  f"rw={r['reach_w']:.2f}/pf={r['prune_frac']:.2f}/"
                  f"s1={r['s1_budget']}/s3={r['s3_budget']}/"
                  f"c1={r['n_cyc1']}/c3={r['n_cyc3']} "
                  f"→ reach={r['reach_n3']*100:.0f}% "
                  f"tom={r['tom_pair']*100:.0f}% "
                  f"scales={r['scales_n5']*100:.0f}% "
                  f"C={r['composite']:.3f}", flush=True)
    print(f"    batch elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    return results


def main():
    print("=" * 70, flush=True)
    print("  PHASE 7 N=3 DEEPDIVE (dale=ON, hebb=OFF fixed)", flush=True)
    print(f"  Narrow around best + WIDEN training budgets (50-120s)", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    t_all = time.time()

    # ============================================================
    # Phase 0: Verify current best (3 replications = different RNG)
    # ============================================================
    BEST = (DALE, HEBB, 0.44, 0.44, 25.0, 34.0, 1.0, 2.0)
    print(f"\n[Phase 0] Verify current best × 3 replications", flush=True)
    verify_configs = [BEST] * 3
    results_v = batch_evaluate(verify_configs, workers=3, tag="V")
    all_results.extend(results_v)
    valid_v = [r for r in results_v if "error" not in r]
    if valid_v:
        reaches = [r["reach_n3"] for r in valid_v]
        print(f"  Best replications: reach = {[f'{r*100:.0f}%' for r in reaches]} "
              f"(mean {np.mean(reaches)*100:.1f}%, std {np.std(reaches)*100:.1f}%)",
              flush=True)

    # ============================================================
    # Phase 1: Narrow random search (24 configs, 4 rounds × 6)
    # ============================================================
    print(f"\n[Phase 1] Narrow random: 24 configs in 6-dim user space", flush=True)
    random_configs = sample_random(24, seed=42)
    BATCH = 6
    for batch_idx in range(0, len(random_configs), BATCH):
        print(f"\n  Round {batch_idx//BATCH + 1}/{len(random_configs)//BATCH}:", flush=True)
        batch = random_configs[batch_idx:batch_idx + BATCH]
        results = batch_evaluate(batch, workers=BATCH, tag=f"R{batch_idx//BATCH+1}-")
        all_results.extend(results)

    # Take top-5 so far (including verify)
    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda r: -r.get("composite", 0))
    top5 = valid[:5]
    print(f"\n[Phase 1 top-5]", flush=True)
    for i, r in enumerate(top5):
        print(f"  #{i+1}: reach={r['reach_n3']*100:.0f}% "
              f"tom={r['tom_pair']*100:.0f}% C={r['composite']:.3f} "
              f"(rw={r['reach_w']:.2f} pf={r['prune_frac']:.2f} "
              f"s1={r['s1_budget']} s3={r['s3_budget']} "
              f"c1={r['n_cyc1']} c3={r['n_cyc3']})", flush=True)

    # ============================================================
    # Phase 2: Refine top-3 × 4 mutations = 12 (2 rounds × 6)
    # ============================================================
    print(f"\n[Phase 2] Tight mutate top-3 × 4 = 12 refined configs", flush=True)
    refined = []
    for i, r in enumerate(top5[:3]):
        base = (DALE, HEBB, r["reach_w"], r["prune_frac"],
                float(r["s1_budget"]), float(r["s3_budget"]),
                float(r["n_cyc1"]), float(r["n_cyc3"]))
        for j in range(4):
            refined.append(mutate(base, sigma=0.08, seed=100*i + j))

    for batch_idx in range(0, len(refined), BATCH):
        print(f"\n  Refine round {batch_idx//BATCH + 1}/{(len(refined)+BATCH-1)//BATCH}:",
              flush=True)
        batch = refined[batch_idx:batch_idx + BATCH]
        results = batch_evaluate(batch, workers=BATCH, tag=f"F{batch_idx//BATCH+1}-")
        all_results.extend(results)

    # ============================================================
    # Final
    # ============================================================
    total = time.time() - t_all
    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda r: -r.get("composite", 0))
    best = valid[0] if valid else None

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL RESULTS (total {total:.0f}s = {total/60:.1f} min)", flush=True)
    print(f"  Total evals: {len(all_results)}, valid: {len(valid)}", flush=True)
    print(f"{'='*70}", flush=True)

    if best:
        print(f"\n  BEST CONFIG:", flush=True)
        print(f"    dale_law     = ON (fixed)", flush=True)
        print(f"    hebbian      = OFF (fixed)", flush=True)
        print(f"    reach_w      = {best['reach_w']:.3f}", flush=True)
        print(f"    prune_frac   = {best['prune_frac']:.3f}", flush=True)
        print(f"    s1_budget    = {best['s1_budget']}s", flush=True)
        print(f"    s3_budget    = {best['s3_budget']}s", flush=True)
        print(f"    n_cycles     = s1:{best['n_cyc1']}  s3:{best['n_cyc3']}", flush=True)
        print(f"    n_edges_kept = {best['n_kept']}", flush=True)
        print(f"\n  METRICS:", flush=True)
        print(f"    reach (N=3):     {best['reach_n3']*100:.0f}%", flush=True)
        print(f"    ToM pairs:       {best['tom_pair']*100:.0f}%", flush=True)
        print(f"    scales to N=5:   {best['scales_n5']*100:.0f}%", flush=True)
        print(f"    composite:       {best['composite']:.3f}", flush=True)

        print(f"\n  COMPARISON:", flush=True)
        print(f"    parallel-search best:    reach 28%, C 0.449", flush=True)
        print(f"    this deepdive best:      reach {best['reach_n3']*100:.0f}%, "
              f"C {best['composite']:.3f}", flush=True)

    out = {
        "total_elapsed_s": round(total, 1),
        "n_evals": len(all_results),
        "n_valid": len(valid),
        "fixed": {"dale": True, "hebb": False},
        "ranges": dict(zip(RANGE_NAMES, DEEPDIVE_RANGES)),
        "best": best,
        "top5": top5,
        "all_results": all_results,
    }
    with open("phase7_n3_deepdive_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_n3_deepdive_result.json", flush=True)


if __name__ == "__main__":
    main()
