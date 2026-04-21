"""Phase 11.5 — 16N Kathara vs 60N Cortical in 3D Cardinal.

Compare 16N vs 60N core brain on the SAME 3D Cardinal setup.
Both get N_universes × N_epochs × N_steps to evolve. Best universe's final
quality + meta-learned world params are the benchmark.

Hypothesis: cortical 60N reaches higher quality faster, AND speciates more
(varied gravity/jump regimes each universe's cortical DNA adapts to).
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

from tamashii.phase_11_cardinal_3d import (
    GravityUniverse, CorticalGravityUniverse,
    make_3d_universe_params, mutate_3d_universe_params,
)
from tamashii.phase_9_ecology import SHELLS_DEFAULT


def run_cardinal(U_class, n_universes, n_epochs, epoch_steps, agents_per_u,
                   seed, label):
    """Run Cardinal with given Universe class. Return history."""
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)
    shells = SHELLS_DEFAULT
    configs_dir = "tamashii/configs"
    trained_dir = "tamashii/configs"

    # Build universes
    t_build = time.time()
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = U_class(u_id, wp, agents_per_u, seed + u_id * 101,
                     shells, configs_dir, trained_dir)
        universes.append(u)
    print(f"  built {n_universes} universes in {time.time() - t_build:.1f}s",
          flush=True)

    history = []
    t0 = time.time()
    for epoch in range(n_epochs):
        t_ep = time.time()
        for u in universes:
            u.run_epoch(epoch_steps, log_every=epoch_steps // 2)

        qualities = [(u.id, u.quality()) for u in universes]
        qualities.sort(key=lambda x: -x[1])
        best_id = qualities[0][0]; worst_id = qualities[-1][0]

        # Snapshot per universe
        snapshots = []
        for u in universes:
            s = u.world.stats()
            snapshots.append({
                "id": u.id,
                "quality": float(u.quality()),
                "n_alive": s["n_alive"],
                "n_births": s["n_births"],
                "n_deaths": s["n_deaths"],
                "mean_agent_z": float(s.get("mean_agent_z", 0)),
                "max_agent_z": float(s.get("max_agent_z", 0)),
                "gravity": float(u.world_params.get("gravity", 0)),
                "vert_food": float(u.world_params.get("vertical_food_frac", 0)),
                "max_gen": s["max_generation"],
                "dna_div": float(s["dna_diversity"]),
            })
        history.append({"epoch": epoch, "snapshots": snapshots,
                         "best": best_id, "worst": worst_id})

        print(f"  ep{epoch}: best u{best_id} q={qualities[0][1]:+.2f} | "
              f"worst u{worst_id} q={qualities[-1][1]:+.2f} "
              f"({time.time() - t_ep:.0f}s)",
              flush=True)
        for snap in sorted(snapshots, key=lambda x: -x["quality"]):
            print(f"    u{snap['id']}: q={snap['quality']:+.2f} alive={snap['n_alive']} "
                  f"b/d={snap['n_births']}/{snap['n_deaths']} "
                  f"mz={snap['mean_agent_z']:.2f} g={snap['gravity']:+.3f} "
                  f"vf={snap['vert_food']:.2f}", flush=True)

        # Meta-evolve
        if epoch < n_epochs - 1:
            best_u = next(u for u in universes if u.id == best_id)
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == worst_id)
            old_id = universes[worst_idx].id
            universes[worst_idx] = U_class(
                old_id, new_params, agents_per_u,
                seed + old_id * 101 + epoch * 7,
                shells, configs_dir, trained_dir)

    total = time.time() - t0
    print(f"\n  {label} total: {total:.0f}s", flush=True)
    return {
        "label": label,
        "total_elapsed_s": round(total, 1),
        "history": history,
        "final_snapshots": history[-1]["snapshots"],
        "final_best_quality": max(s["quality"] for s in history[-1]["snapshots"]),
        "final_mean_quality": float(np.mean(
            [s["quality"] for s in history[-1]["snapshots"]])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--n_epochs",    type=int, default=3)
    ap.add_argument("--epoch_steps", type=int, default=400)
    ap.add_argument("--agents",      type=int, default=5)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_11_5_cortical_vs_16n.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  PHASE 11.5: 16N Kathara vs 60N Cortical in 3D Cardinal", flush=True)
    print(f"  {args.n_universes}u × {args.n_epochs}ep × {args.epoch_steps}step × {args.agents}agents",
          flush=True)
    print("=" * 72, flush=True)

    result_16n = run_cardinal(
        GravityUniverse, args.n_universes, args.n_epochs,
        args.epoch_steps, args.agents, args.seed, "A. 16N Kathara (3D)")

    result_60n = run_cardinal(
        CorticalGravityUniverse, args.n_universes, args.n_epochs,
        args.epoch_steps, args.agents, args.seed, "B. 60N Cortical (3D)")

    print(f"\n{'=' * 72}\n  COMPARISON\n{'=' * 72}", flush=True)
    print(f"  {'metric':30s} {'16N':>12s} {'60N cortical':>14s}", flush=True)
    print(f"  {'final_best_quality':30s} {result_16n['final_best_quality']:>12.2f} "
          f"{result_60n['final_best_quality']:>14.2f}", flush=True)
    print(f"  {'final_mean_quality':30s} {result_16n['final_mean_quality']:>12.2f} "
          f"{result_60n['final_mean_quality']:>14.2f}", flush=True)
    print(f"  {'total_elapsed_s':30s} {result_16n['total_elapsed_s']:>12.1f} "
          f"{result_60n['total_elapsed_s']:>14.1f}", flush=True)
    delta = result_60n['final_best_quality'] - result_16n['final_best_quality']
    print(f"\n  Quality diff (60N - 16N): {delta:+.2f} "
          f"({'60N WINS' if delta > 0.5 else '16N WINS' if delta < -0.5 else 'TIE'})",
          flush=True)

    out = {"A_16N": result_16n, "B_60N_cortical": result_60n,
            "diff_60n_minus_16n": delta, "config": vars(args)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
