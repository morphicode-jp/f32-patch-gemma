"""Phase 11.1 — Quality function ablation for Zenron^∞ escalation.

Test: does mut_rate escalation depend on the quality function's DNA-diversity
reward (γ)? If γ=0 and escalation STILL happens, something else drives it;
if γ dominates, confirmed mechanism.

5 quality conditions (all 4u × 15 agents × 1500 step × 3 epoch GPU):

  A. BALANCED (control)        — default composite
  B. ALIVE_ONLY                — quality = log(n_alive) — survive-only pressure
  C. BIRTHS_ONLY               — quality = log(n_births) — creativity-only
  D. DIVERSITY_ONLY            — quality = dna_diversity × 10 — pure variety
  E. FOOD_ONLY                 — quality = log(mean_food) — pure productivity

Expected results if Zenron^∞ → escalation holds:
  A: clear escalation (as Ch22)
  B: NO escalation (or even suppression)
  C: moderate escalation (births favor some mutation)
  D: STRONGEST escalation (direct selection for variety)
  E: NO escalation (food doesn't depend on mutation)

If this prediction pattern holds → Zenron^∞ is REAL (env-signal sensitive).
If escalation happens uniformly → artifact.
If no pattern → weak correlation, need more investigation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.phase_10_3_cardinal_gpu import run_cardinal_gpu
from tamashii.phase_10_3_cardinal import Universe


QUALITY_CONDITIONS = {
    "A_balanced": {
        "alive": 1.0, "births": 1.5, "dna_diversity": 10.0,
        "food": 0.5, "generation": 0.8,
    },
    "B_alive_only": {
        "alive": 10.0, "births": 0.0, "dna_diversity": 0.0,
        "food": 0.0, "generation": 0.0,
    },
    "C_births_only": {
        "alive": 0.0, "births": 10.0, "dna_diversity": 0.0,
        "food": 0.0, "generation": 0.0,
    },
    "D_diversity_only": {
        "alive": 0.0, "births": 0.0, "dna_diversity": 30.0,
        "food": 0.0, "generation": 0.0,
    },
    "E_food_only": {
        "alive": 0.0, "births": 0.0, "dna_diversity": 0.0,
        "food": 10.0, "generation": 0.0,
    },
}


def run_single_condition(name: str, weights: dict, seed: int = 42,
                          n_universes: int = 4, agents_per_universe: int = 15,
                          epoch_steps: int = 1500, n_epochs: int = 3):
    """Run Cardinal with specific quality weights, return mut_rate trajectory."""
    print(f"\n{'='*72}", flush=True)
    print(f"  CONDITION {name}: weights = {weights}", flush=True)
    print(f"{'='*72}", flush=True)
    # Set class-level weights before running
    Universe.quality_weights = weights
    out_path = f"tamashii_phase_11_1_{name}.json"
    run_cardinal_gpu(
        n_universes=n_universes,
        agents_per_universe=agents_per_universe,
        epoch_steps=epoch_steps,
        n_epochs=n_epochs,
        base_seed=seed,
        output=out_path,
        device="cuda",
    )
    Universe.quality_weights = None  # reset
    # Load results and extract mut_rate trajectory of winning universe
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mut_rate_traj = []
    for h in data["history"]:
        # Universe ranked #1 this epoch
        best = h["universes"][0]
        mut_rate_traj.append({
            "epoch": h["epoch"],
            "best_u_id": best["id"],
            "best_mut_rate": best["world_params"]["mutation_rate"],
            "best_quality": best["quality"],
            "best_dna_div": best["stats"]["dna_diversity"],
        })
    # Final across all universes: mean & max mut_rate, dna_div
    all_mut = [u["world_params"]["mutation_rate"] for u in data["final_ranking"]]
    all_dna = [u["stats"]["dna_diversity"] for u in data["final_ranking"]]
    return {
        "name": name,
        "weights": weights,
        "trajectory": mut_rate_traj,
        "final_mean_mut_rate": float(np.mean(all_mut)),
        "final_max_mut_rate": float(np.max(all_mut)),
        "final_mean_dna_div": float(np.mean(all_dna)),
        "final_max_dna_div": float(np.max(all_dna)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents_per_universe", type=int, default=15)
    ap.add_argument("--epoch_steps", type=int, default=1500)
    ap.add_argument("--n_epochs", type=int, default=3)
    ap.add_argument("--conditions", type=str, default="A_balanced,B_alive_only,"
                                                       "C_births_only,D_diversity_only,"
                                                       "E_food_only")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_11_1_ablation_summary.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("  PHASE 11.1 — Quality Function Ablation", flush=True)
    print(f"  5 conditions × 4u × 15 agents × {args.epoch_steps}step × "
          f"{args.n_epochs}epoch", flush=True)
    print("=" * 72, flush=True)

    results = []
    conditions_to_run = [c.strip() for c in args.conditions.split(",")]
    t_all = time.time()
    for name in conditions_to_run:
        if name not in QUALITY_CONDITIONS:
            print(f"  Unknown condition: {name}, skipping", flush=True)
            continue
        weights = QUALITY_CONDITIONS[name]
        t0 = time.time()
        result = run_single_condition(
            name, weights, seed=args.seed,
            n_universes=args.n_universes,
            agents_per_universe=args.agents_per_universe,
            epoch_steps=args.epoch_steps,
            n_epochs=args.n_epochs,
        )
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)
        print(f"\n  [{name}] ELAPSED {elapsed:.0f}s "
              f"final_mean_mut_rate={result['final_mean_mut_rate']:.4f} "
              f"final_mean_dna={result['final_mean_dna_div']:.4f}",
              flush=True)

    total = time.time() - t_all
    print(f"\n{'='*72}", flush=True)
    print(f"  ABLATION SUMMARY ({total:.0f}s = {total/60:.1f}min)", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"\n  {'Condition':22s} {'final_mean_mut':>14s} {'final_max_mut':>14s} "
          f"{'final_mean_dna':>14s}", flush=True)
    for r in results:
        print(f"  {r['name']:22s} {r['final_mean_mut_rate']:>14.4f} "
              f"{r['final_max_mut_rate']:>14.4f} "
              f"{r['final_mean_dna_div']:>14.4f}", flush=True)

    print(f"\n  Mut_rate trajectories (best of each epoch):", flush=True)
    for r in results:
        traj = " → ".join(f"{t['best_mut_rate']:.3f}" for t in r['trajectory'])
        print(f"    {r['name']:22s} {traj}", flush=True)

    out = {
        "total_elapsed_s": round(total, 1),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
