"""phase7_seed_avg.py - Settle the noise question in 15min.

Question: Is reach 28% observed peak real, or is it seed outlier?
  (Deepdive Phase 0 with 3 replications gave 17%, 14%, 6% = mean 12%+-5%.)

Test: Run TOP-3 deepdive configs, 8 replications each in parallel batches.
  → 24 evals / 6 workers = 4 rounds × ~8min = ~32 min total.

If all 3 configs converge to similar mean (~12-18%), ceiling confirmed.
If any config significantly higher → that config is truly better, push it.
"""
import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase7_mega import worker_dense_prune


# Top-3 from deepdive Phase 1 narrow search (reach 28% each in single eval)
TOP3_CONFIGS = [
    # #1 deepdive best: rw=0.44 pf=0.54 s1=111 s3=68 c1=3 c3=2
    (1.0, 0.0, 0.441, 0.541, 111.0, 68.0, 3.0, 2.0),
    # #2: rw=0.52 pf=0.47 s1=110 s3=90 c1=3 c3=2
    (1.0, 0.0, 0.520, 0.470, 110.0, 90.0, 3.0, 2.0),
    # #4 (skipping #3 w/ reach=25%): rw=0.48 pf=0.42 s1=76 s3=65 c1=2 c3=3
    (1.0, 0.0, 0.480, 0.420, 76.0, 65.0, 2.0, 3.0),
]
CONFIG_LABELS = ["best#1", "#2", "#4"]
N_REPLICATIONS = 8


def batch_evaluate(configs_with_tags, workers=6):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [(tag, ex.submit(worker_dense_prune, c))
                   for tag, c in configs_with_tags]
        results = [(tag, f.result()) for tag, f in futures]
    elapsed = time.time() - t0
    for tag, r in results:
        if "error" in r:
            print(f"    [{tag}] ERROR: {r['error'][:80]}", flush=True)
        else:
            print(f"    [{tag}] reach={r['reach_n3']*100:.0f}% "
                  f"tom={r['tom_pair']*100:.0f}% "
                  f"scales={r['scales_n5']*100:.0f}% "
                  f"C={r['composite']:.3f}", flush=True)
    print(f"    batch elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    return results


def main():
    print("=" * 70, flush=True)
    print(f"  PHASE 7 SEED AVERAGING: top-3 × {N_REPLICATIONS} replications", flush=True)
    print("=" * 70, flush=True)

    # Build flat list: each config × N_REPLICATIONS
    all_tasks = []
    for label, cfg in zip(CONFIG_LABELS, TOP3_CONFIGS):
        for rep in range(N_REPLICATIONS):
            all_tasks.append((f"{label}-r{rep}", cfg))

    print(f"\n  Total: {len(all_tasks)} evals in {len(all_tasks)//6} rounds × 6 workers",
          flush=True)

    all_results = []
    t_all = time.time()
    BATCH = 6
    for batch_idx in range(0, len(all_tasks), BATCH):
        print(f"\n  Round {batch_idx//BATCH + 1}/{(len(all_tasks)+BATCH-1)//BATCH}:",
              flush=True)
        batch = all_tasks[batch_idx:batch_idx + BATCH]
        results = batch_evaluate(batch, workers=BATCH)
        all_results.extend(results)

    # Aggregate per config
    total = time.time() - t_all
    print(f"\n{'='*70}", flush=True)
    print(f"  SEED-AVERAGED RESULTS (total {total:.0f}s = {total/60:.1f} min)",
          flush=True)
    print(f"{'='*70}", flush=True)

    summary = {}
    for label in CONFIG_LABELS:
        config_results = [r for tag, r in all_results
                          if tag.startswith(label + "-") and "error" not in r]
        if not config_results:
            summary[label] = {"error": "all failed"}
            continue
        reaches = [r["reach_n3"] for r in config_results]
        toms = [r["tom_pair"] for r in config_results]
        scales = [r["scales_n5"] for r in config_results]
        composites = [r["composite"] for r in config_results]

        summary[label] = {
            "n": len(config_results),
            "reach_mean": float(np.mean(reaches)),
            "reach_std": float(np.std(reaches)),
            "reach_min": float(np.min(reaches)),
            "reach_max": float(np.max(reaches)),
            "tom_mean": float(np.mean(toms)),
            "scales_mean": float(np.mean(scales)),
            "composite_mean": float(np.mean(composites)),
            "composite_std": float(np.std(composites)),
        }

        print(f"\n  {label} (n={len(config_results)}):", flush=True)
        print(f"    reach:     mean {np.mean(reaches)*100:.1f}% "
              f"± {np.std(reaches)*100:.1f}% "
              f"(min {np.min(reaches)*100:.0f}% / max {np.max(reaches)*100:.0f}%)",
              flush=True)
        print(f"    tom:       mean {np.mean(toms)*100:.1f}%", flush=True)
        print(f"    scales:    mean {np.mean(scales)*100:.1f}%", flush=True)
        print(f"    composite: mean {np.mean(composites):.3f} "
              f"± {np.std(composites):.3f}", flush=True)

    # Conclusion
    best_label = max(summary.keys(),
                     key=lambda k: summary[k].get("reach_mean", -1))
    best_mean = summary[best_label]["reach_mean"]
    print(f"\n  BEST BY MEAN REACH: {best_label} ({best_mean*100:.1f}%)", flush=True)
    reaches_by_label = {k: summary[k].get("reach_mean", 0) for k in CONFIG_LABELS}
    print(f"  All means: {reaches_by_label}", flush=True)

    out = {
        "total_elapsed_s": round(total, 1),
        "configs": [{"label": l, "params": list(c)}
                    for l, c in zip(CONFIG_LABELS, TOP3_CONFIGS)],
        "n_replications_per_config": N_REPLICATIONS,
        "summary": summary,
        "all_results": [{"tag": tag, **r} for tag, r in all_results],
    }
    with open("phase7_seed_avg_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_seed_avg_result.json", flush=True)


if __name__ == "__main__":
    main()
