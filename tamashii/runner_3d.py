"""runner_3d.py — tamashii Fluctlight in 3D voxel world (Underworld Phase 4).

Couples Tamashii (7 trained shells + optional Hebbian core) with VoxelWorld3D.

Experimental baseline: can the 7-shell Fluctlight, trained only on 2D
multi-agent tasks, function in a 3D voxel environment with different
sensor/action semantics? If yes, we have substrate portability (a key
Underworld Fluctlight property — Alice's brain works whether in Axiom
Cathedral or Dark Territory).

Reports:
  - food_eaten_total         (primitive "task success")
  - survival_steps           (agent moved steadily, not frozen)
  - exploration_area         (unique 2D cells visited)
  - final_hebbian_stats      (if hebbian_core used)
  - S_trajectory_variance    (brain state diversity)
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

from tamashii.core import Tamashii
from tamashii.runner_multiagent import SHELL_REGISTRY, build_agent, extract_action
from tamashii.shell_base import load_shell_from_config
from tamashii.shells.hebbian_core import HebbianCoreBrain


def build_fluctlight(shells: list[str], configs_dir: str,
                     trained_dir: str | None = None,
                     use_hebbian_core: bool = False) -> Tamashii:
    """Build a 7-shell Fluctlight. Optionally replace core with Hebbian variant."""
    # Clone normal builder, then swap core if requested
    if not use_hebbian_core:
        return build_agent(shells, configs_dir, trained_dir=trained_dir, D=192)

    # Hebbian version: hand-build
    shell_instances = []
    for name in shells:
        if name == "core_brain":
            base = os.path.join(configs_dir, "hebbian_core.json")
            with open(base, "r", encoding="utf-8") as f:
                hcfg = json.load(f)
            if trained_dir:
                tc = os.path.join(trained_dir, "core_brain_trained.json")
                if os.path.exists(tc):
                    with open(tc, "r", encoding="utf-8") as f:
                        td = json.load(f)
                    kp = td.get("params", {}).get("kathara_params")
                    if kp is not None:
                        hcfg["params"]["kathara_params"] = kp
            shell_instances.append(HebbianCoreBrain(hcfg))
        else:
            cls = SHELL_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"Unknown shell: {name}")
            base_json = os.path.join(configs_dir, f"{name}.json")
            trained_json = None
            if trained_dir:
                cand = os.path.join(trained_dir, f"{name}_trained.json")
                if os.path.exists(cand):
                    trained_json = cand
            shell_instances.append(
                load_shell_from_config(cls, base_json, trained_json))
    return Tamashii(shells=shell_instances, D=192)


def run_3d_episode(agent: Tamashii, world,
                   n_steps: int = 500, ticks_per_step: int = 3,
                   verbose_every: int = 100) -> dict:
    sensors = world.reset()
    agent.reset_episode()

    # Inject initial sensors (writing into S[0:16] via agent._lock)
    def _write_sensors(s):
        with agent._lock:
            agent.S[0:16] = np.asarray(s, dtype=np.float64)

    food_events: list[tuple[int, int]] = []  # (step, total_after)
    pos_visited: set[tuple[int, int]] = set()
    S_snapshots = []
    heading_series = []

    for step in range(n_steps):
        _write_sensors(sensors)

        for _ in range(ticks_per_step):
            agent.tick_once()

        S = agent.read_state()
        nav = float(np.clip(S[16], 0.0, 1.0))
        speed = float(np.clip(S[17], 0.0, 1.0))
        voice = float(np.clip(S[18], 0.0, 1.0))

        sensors, ate_this, done = world.step(nav, speed, voice)
        if ate_this > 0:
            food_events.append((step, world.food_eaten))

        pos_visited.add((int(world.agent_pos[0]), int(world.agent_pos[1])))
        heading_series.append(float(world.agent_heading))

        if step % 5 == 0:
            S_snapshots.append(S.copy())

        if verbose_every and step % verbose_every == 0 and step > 0:
            print(f"    step {step}: pos=({world.agent_pos[0]:.1f},"
                  f"{world.agent_pos[1]:.1f}) "
                  f"heading={world.agent_heading:.2f} "
                  f"food_eaten={world.food_eaten} "
                  f"explored={len(pos_visited)}cells", flush=True)

        if done:
            break

    # Metrics
    S_traj = np.array(S_snapshots) if S_snapshots else np.zeros((1, 192))
    S_var = float(np.var(S_traj, axis=0).mean()) if S_traj.shape[0] > 1 else 0.0
    # Heading autocorrelation (how coherent is direction choice)
    h_arr = np.array(heading_series)
    if len(h_arr) > 3:
        h_mean = h_arr.mean()
        h_var = h_arr.var()
        h_autocorr = (float(np.mean((h_arr[:-1] - h_mean) * (h_arr[1:] - h_mean))
                            / h_var) if h_var > 1e-10 else 0.0)
    else:
        h_autocorr = 0.0

    heb_stats = None
    if len(agent.shells) > 0 and isinstance(agent.shells[0], HebbianCoreBrain):
        heb_stats = agent.shells[0].hebbian_stats()

    return {
        "food_eaten": int(world.food_eaten),
        "food_events": food_events,
        "n_cells_explored": len(pos_visited),
        "total_cells": world.size * world.size,
        "exploration_ratio": len(pos_visited) / (world.size * world.size),
        "heading_autocorr": h_autocorr,
        "S_traj_variance": S_var,
        "steps_run": step + 1,
        "final_state": world.state_summary(),
        "hebbian_stats": heb_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", type=str,
                    default="core_brain,brainstem,cerebellum,salience,hippocampus,prefrontal,dmn")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=500)
    ap.add_argument("--world_size", type=int, default=16)
    ap.add_argument("--n_food", type=int, default=5)
    ap.add_argument("--n_walls", type=int, default=30)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--use_hebbian", action="store_true",
                    help="Replace core_brain with HebbianCoreBrain")
    ap.add_argument("--trained_dir", type=str, default="tamashii/configs")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_4_3d_result.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from world_3d import VoxelWorld3D

    shells = [s.strip() for s in args.shells.split(",") if s.strip()]
    print("=" * 70, flush=True)
    print(f"  UNDERWORLD Phase 4 - Fluctlight in 3D voxel world", flush=True)
    print(f"  shells: {shells}", flush=True)
    print(f"  hebbian_core: {args.use_hebbian}", flush=True)
    print(f"  world: {args.world_size}^3, food={args.n_food}, "
          f"walls={args.n_walls}", flush=True)
    print(f"  eps={args.episodes}, steps_per_ep={args.n_steps}", flush=True)
    print("=" * 70, flush=True)

    configs_dir = os.path.join(THIS_DIR, "configs")
    # ONE agent, episodes share it (so Hebbian w_adapt accumulates if enabled)
    agent = build_fluctlight(
        shells, configs_dir, trained_dir=args.trained_dir,
        use_hebbian_core=args.use_hebbian,
    )

    episodes = []
    t0 = time.time()
    for ep in range(args.episodes):
        world = VoxelWorld3D(
            size=args.world_size, n_food=args.n_food, n_walls=args.n_walls,
            seed=args.seed_base + ep * 7,
        )
        print(f"\n[Episode {ep} seed={args.seed_base + ep * 7}]", flush=True)
        result = run_3d_episode(
            agent, world, n_steps=args.n_steps,
            verbose_every=100 if args.verbose else 0)
        episodes.append(result)
        print(f"  END: food_eaten={result['food_eaten']} "
              f"explored={result['n_cells_explored']}/{result['total_cells']}"
              f" ({result['exploration_ratio']*100:.1f}%) "
              f"heading_coherence={result['heading_autocorr']:.3f} "
              f"S_var={result['S_traj_variance']:.3f}", flush=True)
        if result["hebbian_stats"]:
            h = result["hebbian_stats"]
            print(f"  Hebbian: |w|={h['w_adapt_mean_abs']:.4f} "
                  f"changed={h['n_changed_edges']}/48 "
                  f"cumul_reward={h['cumulative_reward']:.2f}", flush=True)

    elapsed = time.time() - t0
    total_food = sum(e["food_eaten"] for e in episodes)
    total_explored = sum(e["n_cells_explored"] for e in episodes)
    mean_coh = float(np.mean([e["heading_autocorr"] for e in episodes]))

    print(f"\n{'='*70}", flush=True)
    print(f"  UNDERWORLD Phase 4 SUMMARY ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Total food eaten:     {total_food} across {args.episodes} eps", flush=True)
    print(f"  Total unique cells:   {total_explored}", flush=True)
    print(f"  Mean heading coh:     {mean_coh:.3f}", flush=True)
    if episodes[-1]["hebbian_stats"]:
        h = episodes[-1]["hebbian_stats"]
        print(f"  Final Hebbian |w|:    {h['w_adapt_mean_abs']:.4f}  "
              f"(edges changed {h['n_changed_edges']}/48)", flush=True)

    # Minimum viable Fluctlight: did it eat at all?
    if total_food > 0:
        print(f"\n  ✓ FLUCTLIGHT ALIVE: agent ate food in 3D world", flush=True)
    else:
        print(f"\n  × FLUCTLIGHT DORMANT: no food eaten — tune sensor/motor map",
              flush=True)

    out = {
        "shells": shells,
        "use_hebbian": args.use_hebbian,
        "world_config": {
            "size": args.world_size, "n_food": args.n_food,
            "n_walls": args.n_walls,
        },
        "episodes": args.episodes,
        "n_steps": args.n_steps,
        "elapsed_s": round(elapsed, 1),
        "per_episode": episodes,
        "aggregate": {
            "total_food": total_food,
            "total_cells_explored": total_explored,
            "mean_heading_coherence": mean_coh,
        },
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
