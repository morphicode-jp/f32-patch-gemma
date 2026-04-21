"""Phase 11.6 — Long-term Cardinal 3D run with 16N Kathara.

Goal: observe what emerges when Cardinal runs for MANY more steps in 3D.
- Does mut_rate escalate like in 2D Ch22? (Theorem ZR5d re-test in 3D)
- Does z-axis specialization emerge per universe? (species divergence)
- Do generations deepen (gen_max > 4)?
- Does DNA diversity grow sustainably?

Setup: 4 universes × 5 agents × LONG_STEPS × N_EPOCHS = total ~200K agent-steps.
Detailed per-epoch snapshot + final analysis.

NOT FOR BENCHMARKING — for EMERGENCE OBSERVATION.
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
    GravityUniverse, make_3d_universe_params, mutate_3d_universe_params,
)
from tamashii.phase_9_ecology import SHELLS_DEFAULT


def snapshot_universe(u: GravityUniverse, epoch: int) -> dict:
    """Extract detailed universe state at end of epoch."""
    s = u.world.stats()
    alive = [i for i in range(u.world.n_agents) if u.world.agent_alive[i]]
    # Per-agent stats
    z_values = [float(u.world.agent_positions[i][2]) for i in alive]
    gens = [int(u.world.agent_generation[i]) for i in alive]
    energies = [float(u.world.agent_energy[i]) for i in alive]
    # DNA divergence from founder
    dna_samples = []
    for i in alive[:8]:
        try:
            dna = np.asarray(u.world.agents_external[i].shells[0].kathara_params)
            dna_samples.append(dna.tolist())
        except Exception:
            pass

    return {
        "epoch": epoch,
        "universe_id": u.id,
        "quality": float(u.quality()),
        "world_params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                          for k, v in u.world_params.items()},
        "stats": {
            "n_alive": s["n_alive"],
            "n_births": s["n_births"],
            "n_deaths": s["n_deaths"],
            "max_generation": s["max_generation"],
            "dna_diversity": float(s["dna_diversity"]),
            "mean_energy": float(s["mean_energy"]),
            "mean_agent_z": float(s.get("mean_agent_z", 0.5)),
            "max_agent_z": float(s.get("max_agent_z", 0.5)),
            "n_elevated_food": s.get("n_elevated_food", 0),
        },
        "alive_z_distribution": z_values,
        "alive_generations": gens,
        "alive_energies": energies,
        "dna_samples_n": len(dna_samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=5)
    ap.add_argument("--epoch_steps", type=int, default=4000)
    ap.add_argument("--n_epochs",    type=int, default=6)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_11_6_longrun_3d.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  PHASE 11.6: LONG-TERM 3D Cardinal with 16N Kathara", flush=True)
    print(f"  {args.n_universes} universes × {args.agents} agents × "
          f"{args.n_epochs} epochs × {args.epoch_steps} steps", flush=True)
    total_agent_steps = (args.n_universes * args.agents *
                          args.n_epochs * args.epoch_steps)
    print(f"  Total agent-steps: {total_agent_steps:,}", flush=True)
    print("=" * 72, flush=True)

    shells = SHELLS_DEFAULT
    configs_dir = "tamashii/configs"
    trained_dir = "tamashii/configs"

    # Build universes
    t0 = time.time()
    universes = []
    for u_id in range(args.n_universes):
        wp = make_3d_universe_params(args.seed + u_id * 17)
        u = GravityUniverse(u_id, wp, args.agents, args.seed + u_id * 101,
                             shells, configs_dir, trained_dir)
        universes.append(u)
        print(f"  built u{u_id}: gravity={wp['gravity']:+.3f} "
              f"vert_food={wp['vertical_food_frac']:.2f} "
              f"jump={wp['jump_impulse']:.2f}", flush=True)
    print(f"  setup: {time.time() - t0:.1f}s", flush=True)

    history = []
    t_all = time.time()

    for epoch in range(args.n_epochs):
        t_ep = time.time()
        print(f"\n  === EPOCH {epoch+1}/{args.n_epochs} ===", flush=True)
        for u in universes:
            u.run_epoch(args.epoch_steps, log_every=args.epoch_steps // 2)

        # Snapshot all universes
        snaps = [snapshot_universe(u, epoch) for u in universes]
        ranked = sorted(snaps, key=lambda x: -x["quality"])
        best_id = ranked[0]["universe_id"]
        worst_id = ranked[-1]["universe_id"]

        history.append({
            "epoch": epoch,
            "best": best_id,
            "worst": worst_id,
            "snapshots": snaps,
        })

        # Print epoch summary
        print(f"\n  Epoch {epoch} summary ({time.time() - t_ep:.0f}s):",
              flush=True)
        for snap in ranked:
            wp = snap["world_params"]
            st = snap["stats"]
            print(f"    u{snap['universe_id']}: q={snap['quality']:+.2f} "
                  f"alive={st['n_alive']} b/d={st['n_births']}/{st['n_deaths']} "
                  f"gen_max={st['max_generation']} DNA_div={st['dna_diversity']:.3f} "
                  f"mean_z={st['mean_agent_z']:.2f} | "
                  f"mut_rate={wp.get('mutation_rate', 0):.3f} "
                  f"g={wp.get('gravity', 0):+.3f} vf={wp.get('vertical_food_frac', 0):.2f}",
                  flush=True)

        # Meta-evolve: worst gets variant of best
        if epoch < args.n_epochs - 1:
            best_u = next(u for u in universes if u.id == best_id)
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=args.seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == worst_id)
            old_id = universes[worst_idx].id
            universes[worst_idx] = GravityUniverse(
                old_id, new_params, args.agents,
                args.seed + old_id * 101 + epoch * 7,
                shells, configs_dir, trained_dir)
            print(f"  META: u{worst_id} ← variant of u{best_id}", flush=True)

    total_elapsed = time.time() - t_all

    # Final analysis
    print(f"\n{'=' * 72}", flush=True)
    print(f"  DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"{'=' * 72}", flush=True)

    # Emergence indicators
    print(f"\n  EMERGENCE INDICATORS:", flush=True)

    # 1. mut_rate trajectory
    print(f"\n  1. Mutation rate trajectory (Theorem ZR5d re-test):", flush=True)
    for u_id in range(args.n_universes):
        traj = [h["snapshots"][idx]["world_params"].get("mutation_rate", 0)
                for h in history
                for idx, s in enumerate(h["snapshots"])
                if s["universe_id"] == u_id]
        if traj:
            delta = traj[-1] - traj[0]
            print(f"    u{u_id}: {traj[0]:.3f} → {traj[-1]:.3f} "
                  f"(Δ={delta:+.3f}) {'↑ ESCALATED' if delta > 0.02 else ''}",
                  flush=True)

    # 2. Generation depth
    print(f"\n  2. Generational depth (multi-gen observation):", flush=True)
    max_gen_ever = 0
    for h in history:
        for s in h["snapshots"]:
            max_gen_ever = max(max_gen_ever, s["stats"]["max_generation"])
    print(f"    Deepest generation reached: {max_gen_ever}", flush=True)
    if max_gen_ever >= 5:
        print(f"    ★ Multi-generational evolution observed (>4 gens)",
              flush=True)
    elif max_gen_ever >= 3:
        print(f"    ~ 3-gen wall (consistent with Phase 11.3 finding)",
              flush=True)
    else:
        print(f"    × Gen wall < 3 (needs tighter food selection)", flush=True)

    # 3. Z-axis specialization (per-universe final mean z)
    print(f"\n  3. Z-axis specialization (species divergence):", flush=True)
    final_snapshots = history[-1]["snapshots"]
    z_values = [s["stats"]["mean_agent_z"] for s in final_snapshots
                 if s["stats"]["n_alive"] > 0]
    if len(z_values) >= 2:
        z_range = max(z_values) - min(z_values)
        z_std = float(np.std(z_values))
        print(f"    Mean z range across universes: {min(z_values):.2f} - {max(z_values):.2f} "
              f"(spread {z_range:.2f}, std {z_std:.2f})", flush=True)
        if z_std > 0.3:
            print(f"    ★ Species divergence (z-axis strategies differ)", flush=True)
        else:
            print(f"    ~ Similar z strategies (no clear specialization)", flush=True)

    # 4. Quality evolution
    print(f"\n  4. Quality evolution:", flush=True)
    for h in history:
        qs = [s["quality"] for s in h["snapshots"]]
        print(f"    ep{h['epoch']}: best={max(qs):.2f}, "
              f"mean={np.mean(qs):.2f}, min={min(qs):.2f}",
              flush=True)

    # Save
    out = {
        "config": vars(args),
        "total_elapsed_s": round(total_elapsed, 1),
        "history": history,
        "emergence_indicators": {
            "max_gen_ever": max_gen_ever,
            "final_z_values": [s["stats"]["mean_agent_z"]
                                for s in final_snapshots],
            "final_qualities": [s["quality"] for s in final_snapshots],
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
