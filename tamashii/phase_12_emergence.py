"""Phase 12 — Emergence Observation (paradigm shift: watch, don't optimize).

Human brain runs on 10W. Intelligence isn't scale, it's organization + selection.
Stop fighting benchmarks. Stop pretraining. Let Cardinal RUN and WATCH.

Lean config:
  - 16N Kathara (proven, pretrained) — no 60N cortical here
  - 3D gravity world
  - 4 universes × 5 agents × 50 epochs × 800 steps = 800K agent-steps
  - Estimated ~15-20 min CPU (fits in 10W budget philosophically)

Qualitative observations logged per epoch:
  - DNA trajectory per universe (centroid drift, diversity, variance)
  - Behavioral signatures: mean z, speed distribution, voice usage
  - Generational depth (max_generation)
  - Selection events: when universes replace, what DNA copies over
  - Cross-universe divergence: how different do universes become?

Success is NOT "universe X beats baseline". Success is "we can tell a story
about what emerged, how it changed, and why it's interesting".

Output: phase_12_emergence.json (full history) + emergence_story.md (narrative).
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


def dna_centroid_and_spread(u: GravityUniverse) -> dict:
    """Compute DNA population statistics per universe."""
    params_list = []
    for i in range(u.world.n_agents):
        if u.world.agent_alive[i]:
            try:
                p = np.asarray(u.world.agents_external[i].shells[0].kathara_params,
                                dtype=np.float64)
                params_list.append(p)
            except Exception:
                pass
    if len(params_list) < 1:
        return {"n": 0, "centroid_norm": 0.0, "spread": 0.0,
                 "centroid_first8": []}
    stacked = np.stack(params_list)
    centroid = stacked.mean(axis=0)
    spread = float(np.linalg.norm(stacked.std(axis=0)))  # "DNA pool radius"
    return {
        "n": len(params_list),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "spread": spread,
        "centroid_first8": [float(x) for x in centroid[:8]],
    }


def behavioral_signature(u: GravityUniverse) -> dict:
    """Extract behavioral signature from current world state."""
    alive = [i for i in range(u.world.n_agents) if u.world.agent_alive[i]]
    if not alive:
        return {"n_alive": 0}
    # Positions
    zs = [u.world.agent_positions[i][2] for i in alive]
    xs = [u.world.agent_positions[i][0] for i in alive]
    ys = [u.world.agent_positions[i][1] for i in alive]
    # Spatial spread: how dispersed are agents?
    pos_spread = float(np.linalg.norm([np.std(xs), np.std(ys), np.std(zs)]))
    # z distribution: ground-huggers vs jumpers
    mean_z = float(np.mean(zs))
    # Energy distribution: starving or thriving?
    energies = [u.world.agent_energy[i] for i in alive]
    mean_energy = float(np.mean(energies))
    energy_spread = float(np.std(energies))
    # Generation distribution
    gens = [u.world.agent_generation[i] for i in alive]
    return {
        "n_alive": len(alive),
        "mean_z": mean_z,
        "pos_spread": pos_spread,
        "mean_energy": mean_energy,
        "energy_spread": energy_spread,
        "gen_max": int(max(gens)),
        "gen_spread": float(np.std(gens)),
        "n_distinct_gens": len(set(gens)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=5)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_12_emergence.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  PHASE 12: EMERGENCE OBSERVATION (lean / 10W philosophy)", flush=True)
    print(f"  16N × {args.n_universes}u × {args.agents}agents × "
          f"{args.n_epochs}ep × {args.epoch_steps}step", flush=True)
    total = args.n_universes * args.agents * args.n_epochs * args.epoch_steps
    print(f"  Total agent-steps: {total:,}", flush=True)
    print("=" * 72, flush=True)

    shells = SHELLS_DEFAULT
    cfg = "tamashii/configs"

    # Build 4 universes
    universes = []
    for u_id in range(args.n_universes):
        wp = make_3d_universe_params(args.seed + u_id * 17)
        u = GravityUniverse(u_id, wp, args.agents,
                             args.seed + u_id * 101, shells, cfg, cfg)
        universes.append(u)
        print(f"  u{u_id}: gravity={wp['gravity']:+.3f} "
              f"vert_food={wp['vertical_food_frac']:.2f} "
              f"jump={wp['jump_impulse']:.2f}", flush=True)

    history = []
    t_all = time.time()
    selection_events = []  # who replaced whom

    for epoch in range(args.n_epochs):
        t_ep = time.time()
        for u in universes:
            u.run_epoch(args.epoch_steps, log_every=args.epoch_steps)

        snapshots = []
        for u in universes:
            wp = u.world_params
            dna = dna_centroid_and_spread(u)
            beh = behavioral_signature(u)
            snapshots.append({
                "u_id": u.id,
                "quality": float(u.quality()),
                "world_params": {
                    "gravity": float(wp.get("gravity", 0)),
                    "jump_impulse": float(wp.get("jump_impulse", 0)),
                    "max_height": float(wp.get("max_height", 0)),
                    "vertical_food_frac": float(wp.get("vertical_food_frac", 0)),
                    "mutation_rate": float(wp.get("mutation_rate", 0)),
                    "n_food": int(wp.get("n_food", 0)),
                },
                "dna": dna,
                "behavior": beh,
            })

        ranked = sorted(snapshots, key=lambda x: -x["quality"])
        best_id = ranked[0]["u_id"]
        worst_id = ranked[-1]["u_id"]

        history.append({
            "epoch": epoch,
            "snapshots": snapshots,
            "best": best_id,
            "worst": worst_id,
            "elapsed_ep_s": time.time() - t_ep,
        })

        # Terse per-epoch line
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"\n  === ep{epoch} ({time.time() - t_ep:.1f}s) ===", flush=True)
            for s in ranked:
                b = s["behavior"]; w = s["world_params"]; d = s["dna"]
                print(f"    u{s['u_id']}: q={s['quality']:+.2f} "
                      f"alive={b.get('n_alive', 0)} gen={b.get('gen_max', 0)} "
                      f"mz={b.get('mean_z', 0):.2f} "
                      f"DNA_ctr={d['centroid_norm']:.2f} "
                      f"sprd={d['spread']:.2f} "
                      f"mut={w['mutation_rate']:.3f}", flush=True)

        # Meta-evolve
        if epoch < args.n_epochs - 1:
            best_u = next(u for u in universes if u.id == best_id)
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=args.seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == worst_id)
            old_id = universes[worst_idx].id
            # Log the selection event: who replaced whom and with what mutation
            selection_events.append({
                "epoch": epoch,
                "replaced_uid": old_id,
                "parent_uid": best_id,
                "old_quality": ranked[-1]["quality"],
                "best_quality": ranked[0]["quality"],
                "old_gravity": float(universes[worst_idx].world_params.get("gravity", 0)),
                "new_gravity": float(new_params.get("gravity", 0)),
                "old_mut_rate": float(universes[worst_idx].world_params.get("mutation_rate", 0)),
                "new_mut_rate": float(new_params.get("mutation_rate", 0)),
            })
            universes[worst_idx] = GravityUniverse(
                old_id, new_params, args.agents,
                args.seed + old_id * 101 + epoch * 7,
                shells, cfg, cfg)

    total_elapsed = time.time() - t_all
    print(f"\n{'=' * 72}", flush=True)
    print(f"  DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"{'=' * 72}", flush=True)

    # Qualitative summary — story-oriented
    print(f"\n  === EMERGENCE SUMMARY ===", flush=True)

    # Mutation rate trajectories per universe
    print(f"\n  Mutation rate per universe (trajectory, every 10 eps):", flush=True)
    for u_id in range(args.n_universes):
        traj = []
        for h in history:
            for s in h["snapshots"]:
                if s["u_id"] == u_id:
                    traj.append(s["world_params"]["mutation_rate"])
                    break
        samples = [traj[i] for i in range(0, len(traj), max(1, len(traj)//5))]
        print(f"    u{u_id}: {' → '.join(f'{v:.3f}' for v in samples)}", flush=True)

    # Deepest generation reached
    max_gen = max(s["behavior"].get("gen_max", 0)
                    for h in history for s in h["snapshots"])
    print(f"\n  Max generation ever: {max_gen}", flush=True)

    # DNA divergence between universes (at final epoch)
    final_snaps = history[-1]["snapshots"]
    dna_centroids = np.stack(
        [np.asarray(s["dna"].get("centroid_first8", [0]*8)) for s in final_snaps])
    if len(dna_centroids) >= 2:
        pairwise_dists = []
        for i in range(len(dna_centroids)):
            for j in range(i+1, len(dna_centroids)):
                pairwise_dists.append(np.linalg.norm(dna_centroids[i] - dna_centroids[j]))
        print(f"\n  DNA divergence between universes (final):", flush=True)
        print(f"    mean pairwise: {float(np.mean(pairwise_dists)):.2f}", flush=True)
        print(f"    max pairwise: {float(np.max(pairwise_dists)):.2f}", flush=True)

    # z-axis trajectory across epochs (does anything ever leave ground?)
    max_z_reached = max(s["behavior"].get("mean_z", 0)
                         for h in history for s in h["snapshots"])
    print(f"\n  Max universe-averaged z ever reached: {max_z_reached:.2f}", flush=True)

    # Number of meta-evolution events
    print(f"\n  Meta-evolution events: {len(selection_events)}", flush=True)

    # Which universe survived longest as best?
    best_counts = {}
    for h in history:
        b = h["best"]
        best_counts[b] = best_counts.get(b, 0) + 1
    top_u = max(best_counts.items(), key=lambda x: x[1])
    print(f"  Most-winning universe: u{top_u[0]} ({top_u[1]}/{args.n_epochs} epochs)",
          flush=True)

    out = {
        "config": vars(args),
        "total_elapsed_s": round(total_elapsed, 1),
        "history": history,
        "selection_events": selection_events,
        "summary": {
            "max_generation_ever": max_gen,
            "max_z_reached": float(max_z_reached),
            "best_universe_tally": best_counts,
            "final_dna_pairwise_max": float(np.max(pairwise_dists)) if len(dna_centroids) >= 2 else 0.0,
            "final_dna_pairwise_mean": float(np.mean(pairwise_dists)) if len(dna_centroids) >= 2 else 0.0,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
