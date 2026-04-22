"""Phase 13b — Behavior regression test: did hierarchy break the agent?

After M1 (tick_once layer-ordered) and M3 (inhibition shells), the DNA was
trained under the PRE-M1 flat parallel-additive tick. Same DNA now runs
under layered execution. Does behavior degrade?

Test design (minimal & direct):
  A. Pre-M1 shells (flat, 7 shells, no mimir/inhibition)
  B. Post-M4 shells (hierarchical, 12 shells with guards)
  Same 3D universe params, same DNA pool (pristine 3D)
  Measure: food eaten, births, survival in 500-step episodes × 5 seeds

If B >> A: hierarchy helped (unlikely since DNA not re-trained)
If B ≈ A: hierarchy is neutral (ISS up, behavior same)
If B << A: hierarchy BROKE behavior (DNA-execution mismatch)

Run: ~2-3 min
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

from tamashii.runner_3d import build_fluctlight
from world_3d_gravity import GravityVoxelWorld3D
from tamashii.phase_11_cardinal_3d import make_3d_universe_params


FLAT_SHELLS = ["core_brain", "brainstem", "cerebellum", "salience",
                "hippocampus", "prefrontal", "dmn"]
HIERARCHICAL_SHELLS = FLAT_SHELLS + ["taboo", "mimir_shell",
                                     "inhibition_L1", "inhibition_L3",
                                     "inhibition_L4"]


def run_episode(shells, n_agents=5, n_steps=500, seed=42,
                world_params_seed=42):
    """Run one episode, return food, births, deaths, final alive."""
    cfg = "tamashii/configs"
    wp = make_3d_universe_params(world_params_seed)
    # Build N agents
    agents = [build_fluctlight(shells, cfg, cfg, use_3d_brain=True)
              for _ in range(n_agents)]
    # Build world
    world = GravityVoxelWorld3D(
        n_agents=n_agents, seed=seed, **wp)
    world.register_agents(agents)
    world.reset()
    for a in agents:
        a.reset_episode()

    sensors = [world.get_sensors(i) for i in range(n_agents)]
    for step in range(n_steps):
        N = world.n_agents
        # Sensor → S
        for i in range(N):
            if world.agent_alive[i]:
                with agents[i]._lock:
                    agents[i].S[0:16] = np.asarray(
                        sensors[i] if isinstance(sensors, list) else sensors,
                        dtype=np.float64)
        # Tick × 3
        for _ in range(3):
            for i in range(N):
                if world.agent_alive[i]:
                    agents[i].tick_once()
        # Actions (4-tuple for 3D world: nav/speed/voice/jump)
        actions = []
        for i in range(N):
            if world.agent_alive[i]:
                S = agents[i].read_state()
                actions.append((
                    float(np.clip(S[16], 0, 1)),
                    float(np.clip(S[17], 0, 1)),
                    float(np.clip(S[18], 0, 1)),
                    float(np.clip(S[19], 0, 1)) if len(S) > 19 else 0.0,
                ))
            else:
                actions.append((0.5, 0.0, 0.0, 0.0))
        sensors, _, _ = world.step(actions)
        while len(world.agents_external) > len(agents):
            nc = world.agents_external[len(agents)]
            nc.reset_episode()
            agents.append(nc)
        if world.stats()["n_alive"] == 0:
            break

    stats = world.stats()
    total_food = sum(world.agent_food_eaten[i]
                      for i in range(world.n_agents))
    return {
        "food_total": total_food,
        "births": stats["n_births"],
        "deaths": stats["n_deaths"],
        "final_alive": stats["n_alive"],
        "final_steps": stats["steps"],
        "max_gen": stats["max_generation"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_agents", type=int, default=5)
    ap.add_argument("--n_steps", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--output", type=str,
                     default="phase_13b_behavior_regression.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  Phase 13b: Behavior Regression Test", flush=True)
    print(f"  Flat (7 shells) vs Hierarchical (12 shells, M3)", flush=True)
    print(f"  {args.n_agents} agents × {args.n_steps} steps × {args.n_seeds} seeds",
          flush=True)
    print("=" * 72, flush=True)

    results = {"flat": [], "hierarchical": []}
    for seed in range(args.n_seeds):
        print(f"\n  seed {seed}:", flush=True)
        # A. Flat
        t0 = time.time()
        r_flat = run_episode(FLAT_SHELLS, n_agents=args.n_agents,
                              n_steps=args.n_steps, seed=seed,
                              world_params_seed=42 + seed)
        r_flat["elapsed_s"] = time.time() - t0
        results["flat"].append(r_flat)
        print(f"    FLAT: food={r_flat['food_total']} "
              f"births={r_flat['births']} deaths={r_flat['deaths']} "
              f"alive={r_flat['final_alive']} gen={r_flat['max_gen']} "
              f"({r_flat['elapsed_s']:.1f}s)", flush=True)
        # B. Hierarchical
        t0 = time.time()
        r_hier = run_episode(HIERARCHICAL_SHELLS, n_agents=args.n_agents,
                              n_steps=args.n_steps, seed=seed,
                              world_params_seed=42 + seed)
        r_hier["elapsed_s"] = time.time() - t0
        results["hierarchical"].append(r_hier)
        print(f"    HIER: food={r_hier['food_total']} "
              f"births={r_hier['births']} deaths={r_hier['deaths']} "
              f"alive={r_hier['final_alive']} gen={r_hier['max_gen']} "
              f"({r_hier['elapsed_s']:.1f}s)", flush=True)

    # Aggregate
    def agg(rs, k):
        return float(np.mean([r[k] for r in rs]))

    print(f"\n{'=' * 72}\n  AGGREGATE (mean over {args.n_seeds} seeds)\n{'=' * 72}",
          flush=True)
    print(f"  {'metric':15s} {'FLAT':>12s} {'HIER':>12s} {'delta':>12s}",
          flush=True)
    for k in ["food_total", "births", "deaths", "final_alive", "max_gen"]:
        f_v = agg(results["flat"], k)
        h_v = agg(results["hierarchical"], k)
        delta = h_v - f_v
        sign = "+" if delta >= 0 else ""
        print(f"  {k:15s} {f_v:>12.2f} {h_v:>12.2f} {sign}{delta:>11.2f}",
              flush=True)

    # Interpretation
    food_delta = agg(results["hierarchical"], "food_total") - agg(results["flat"], "food_total")
    birth_delta = agg(results["hierarchical"], "births") - agg(results["flat"], "births")
    verdict_lines = []
    if food_delta > 0 and birth_delta >= 0:
        verdict_lines.append("  HIER >= FLAT: hierarchy did NOT break agent.")
        if food_delta > 3:
            verdict_lines.append("  HIER > FLAT: hierarchy may HELP (DNA generalized)")
    elif food_delta < -2 or birth_delta < -1:
        verdict_lines.append("  HIER < FLAT: hierarchy BROKE agent (DNA mismatch)")
        verdict_lines.append("  → Need to re-train DNA under hierarchical execution")
    else:
        verdict_lines.append("  HIER ~= FLAT: hierarchy is approximately neutral")
        verdict_lines.append("  (ISS improved but behavior unchanged - as predicted)")
    print("\n" + "\n".join(verdict_lines), flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "flat": results["flat"],
            "hierarchical": results["hierarchical"],
            "summary": {
                "flat_mean_food": agg(results["flat"], "food_total"),
                "hier_mean_food": agg(results["hierarchical"], "food_total"),
                "flat_mean_births": agg(results["flat"], "births"),
                "hier_mean_births": agg(results["hierarchical"], "births"),
                "food_delta": food_delta,
                "birth_delta": birth_delta,
            },
        }, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
