"""brain_evolve.py — Multi-generation Sentinel evolution at a fixed scale

Runs Sentinel N times at the same scale with the same experience_id.
Each generation carries forward:
  - Fossil record (best params ever seen) via optimize()'s ExperienceStore
  - dead_dims hint + warm_start via UnifiedExperience (owl path)

Hypothesis: ISS should monotonically improve (or stay flat) across generations
as the optimizer accumulates knowledge.

Previous best at 12L (4L × 12N = 144 nodes):
  brain_growth single-run: ISS 107.6 (approved)

Usage:
    python brain_evolve.py --n-layers 12 --n-gens 5 --time-per-gen 300
"""
import os
import sys
import time
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from brain_growth import (
    make_iss_eval,
    make_connectivity_guard,
    get_ranges,
    build_layered_graph,
    measure_graph,
    _decode_params,
    PARAM_NAMES,
    N_NODES_PER_LAYER,
)


def run_generation(gen, n_layers, time_budget, experience_id, verbose=True):
    """One Sentinel run. Returns entry dict."""
    eval_fn = make_iss_eval(n_layers)
    guard_fn = make_connectivity_guard(n_layers)
    ranges = get_ranges(n_layers)

    # Known-good Kathara seed — used on gen 1, but optimize()'s fossil record
    # will supersede this on later generations
    skip_max = max(3, min(n_layers // 2, 24))
    initial = [0.0, 1.0, 4.0, 6.0, 8.0,
               float(min(2, skip_max)),
               float(min(4, skip_max)),
               float(min(8, skip_max))]

    print(f"\n{'#' * 70}")
    print(f"#  GENERATION {gen}  —  {n_layers}L x 12N "
          f"({n_layers * N_NODES_PER_LAYER} nodes)")
    print(f"#  experience_id={experience_id}  |  budget={time_budget}s")
    print(f"{'#' * 70}\n")

    t_start = time.time()
    result = Sentinel(
        eval_fn=eval_fn,
        guard_fn=guard_fn,
        param_ranges=ranges,
        param_names=PARAM_NAMES,
        experience_id=experience_id,
        initial_params=initial,
        learn=True,
    ).run(time_budget=time_budget, verbose=verbose)
    elapsed = time.time() - t_start

    best = result["best_params"]
    offsets, skips = _decode_params(best)
    G = build_layered_graph(n_layers, offsets, skips)
    m = measure_graph(G)

    entry = {
        "gen": gen,
        "verdict": result["verdict"],
        "iss_v2": round(float(result["eval_score"]), 2),
        "guard": round(float(result["guard_score"]), 2),
        "baseline_guard": round(float(result["baseline_guard"]), 2),
        "optimization_mode": result.get("optimization_mode", "owl"),
        "L": round(m[0], 3) if m else None,
        "C": round(m[1], 3) if m else None,
        "hub": round(m[2], 1) if m else None,
        "n_edges": G.number_of_edges(),
        "offsets": offsets,
        "skips": skips,
        "elapsed_s": round(elapsed, 1),
    }
    print(f"\n  Gen {gen}: ISS={entry['iss_v2']}  guard={entry['guard']}  "
          f"verdict={entry['verdict']}  mode={entry['optimization_mode']}  "
          f"({elapsed:.0f}s)")
    return entry


def main():
    ap = argparse.ArgumentParser(description="Multi-gen brain evolution")
    ap.add_argument("--n-layers", type=int, default=12,
                    help="Layers (12 = the best-so-far scale)")
    ap.add_argument("--n-gens", type=int, default=5)
    ap.add_argument("--time-per-gen", type=int, default=300)
    ap.add_argument("--experience-id", default=None,
                    help="Experience ID (default: brain_evolve_{n_layers}L)")
    args = ap.parse_args()

    exp_id = args.experience_id or f"brain_evolve_{args.n_layers}L"

    history = []
    progress_file = f"brain_evolve_{args.n_layers}L_progress.json"

    # Reset progress file
    with open(progress_file, "w") as f:
        json.dump({
            "n_layers": args.n_layers,
            "n_gens": args.n_gens,
            "time_per_gen": args.time_per_gen,
            "experience_id": exp_id,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gens": [],
        }, f, indent=2)

    total_start = time.time()

    for gen in range(1, args.n_gens + 1):
        entry = run_generation(gen, args.n_layers,
                               args.time_per_gen, exp_id, verbose=True)
        history.append(entry)

        # Append to progress file
        with open(progress_file, "r") as f:
            data = json.load(f)
        data["gens"] = history
        data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(progress_file, "w") as f:
            json.dump(data, f, indent=2)

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 78)
    print(f"  MULTI-GEN EVOLUTION — {args.n_layers}L ({args.n_layers * 12} nodes)")
    print(f"  experience_id={exp_id}")
    print("=" * 78)
    print(f"{'Gen':>4} {'ISS':>8} {'Guard':>8} {'Mode':>10} {'L':>6} {'C':>6} "
          f"{'Edges':>7} {'Time':>7} {'Verdict':>10}")
    print("-" * 78)
    for h in history:
        L = f"{h['L']:.2f}" if h['L'] else "-"
        C = f"{h['C']:.2f}" if h['C'] else "-"
        print(f"{h['gen']:>4} {h['iss_v2']:>8.2f} {h['guard']:>8.2f} "
              f"{h['optimization_mode']:>10} {L:>6} {C:>6} "
              f"{h['n_edges']:>7} {h['elapsed_s']:>7.0f} {h['verdict']:>10}")
    print("-" * 78)

    # ISS progression check
    iss_vals = [h["iss_v2"] for h in history]
    best_so_far = iss_vals[0]
    improvements = 0
    for v in iss_vals[1:]:
        if v > best_so_far:
            improvements += 1
            best_so_far = v
    print(f"\n  ISS progression: {iss_vals}")
    print(f"  Best: {max(iss_vals):.2f}  (gen {iss_vals.index(max(iss_vals)) + 1})")
    print(f"  Improvements across gens: {improvements}/{args.n_gens - 1}")
    print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Save final
    out_file = f"brain_evolve_{args.n_layers}L_result.json"
    with open(out_file, "w") as f:
        json.dump({
            "n_layers": args.n_layers,
            "n_gens": args.n_gens,
            "time_per_gen": args.time_per_gen,
            "experience_id": exp_id,
            "total_elapsed_s": round(total_elapsed, 1),
            "iss_progression": iss_vals,
            "best_iss": max(iss_vals),
            "best_gen": iss_vals.index(max(iss_vals)) + 1,
            "improvements": improvements,
            "gens": history,
        }, f, indent=2)
    print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
