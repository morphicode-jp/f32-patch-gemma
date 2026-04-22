"""Phase 13g — Does higher ISS actually mean better behavior?

This is THE question: ISS jumped 7.7 → 81.87, but does the agent actually
act smarter in the 3D world, or is ISS a structural score disconnected
from function?

Compares 4 configurations in 3D GravityVoxelWorld3D:
  1. FLAT (7 shells, baseline, pre-hierarchy, ISS ≈ 35)
  2. M4 13 hierarchical (12 shells, ISS 22.6) — earlier test showed -21% food
  3. 13d PEAK (20 shells best genome, ISS 81.87)
  4. 13e best (30 shells best genome with L=2.01)

Each config × N seeds × 500 steps × 5 agents.
Measures: food eaten, births, survival, total quality.

If ISS is real → higher ISS config performs better behaviorally.
If ISS is just structural → all perform similarly (or PEAK fails due to
                           DNA mismatch being worse).

Honest test. Either way, we learn.
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
M4_SHELLS = FLAT_SHELLS + ["taboo", "mimir_shell",
                            "inhibition_L1", "inhibition_L3", "inhibition_L4"]
SHELLS_20 = M4_SHELLS + [
    "inhibition_L0_a", "inhibition_L0_b",
    "inhibition_L2_a", "inhibition_L2_b",
    "inhibition_L3_b",
    "inhibition_L4_a", "inhibition_L4_b",
    "inhibition_L5",
]
SHELLS_30 = SHELLS_20 + [
    "inhibition_L1_b", "inhibition_L1_c",
    "inhibition_L2_c", "inhibition_L2_d",
    "inhibition_L3_c", "inhibition_L3_d",
    "inhibition_L4_c", "inhibition_L4_d",
    "inhibition_L5_b", "inhibition_L5_c",
]


def apply_genome_overrides(agent, genome: dict):
    """Apply genome from 13c/13d/13e JSON to shells."""
    for shell in agent.shells:
        name = shell.name
        if name in genome.get("layer_assignments", {}):
            shell.layer = int(genome["layer_assignments"][name])
        if name in genome.get("shell_signs", {}):
            shell.shell_sign = int(genome["shell_signs"][name])
        if name in genome.get("gain_scales", {}):
            shell.gain = float(genome["gain_scales"][name])
        # Slot overrides don't affect runtime behavior directly (those are
        # used only for ISS graph extraction). But store them for completeness.
        if name in genome.get("reads_overrides", {}):
            shell._reads_override = list(genome["reads_overrides"][name])
        if name in genome.get("writes_overrides", {}):
            shell._writes_override = list(genome["writes_overrides"][name])


def run_episode(shells: list[str], genome: dict | None,
                n_agents: int = 5, n_steps: int = 500, seed: int = 42,
                world_params_seed: int = 42) -> dict:
    cfg = "tamashii/configs"
    wp = make_3d_universe_params(world_params_seed)
    agents = []
    for _ in range(n_agents):
        a = build_fluctlight(shells, cfg, cfg, use_3d_brain=True)
        if genome is not None:
            apply_genome_overrides(a, genome)
        agents.append(a)
    world = GravityVoxelWorld3D(n_agents=n_agents, seed=seed, **wp)
    world.register_agents(agents)
    world.reset()
    for a in agents:
        a.reset_episode()

    sensors = [world.get_sensors(i) for i in range(n_agents)]
    for step in range(n_steps):
        N = world.n_agents
        for i in range(N):
            if world.agent_alive[i]:
                with agents[i]._lock:
                    agents[i].S[0:16] = np.asarray(
                        sensors[i] if isinstance(sensors, list) else sensors,
                        dtype=np.float64)
        for _ in range(3):
            for i in range(N):
                if world.agent_alive[i]:
                    agents[i].tick_once()
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
    food_total = sum(world.agent_food_eaten[i] for i in range(world.n_agents))
    # Cardinal-style quality
    q = (np.log1p(stats["n_alive"]) * 1.0 +
         np.log1p(stats["n_births"]) * 1.5 +
         stats.get("dna_diversity", 0) * 10.0 +
         np.log1p(stats["max_generation"]) * 0.8)
    return {
        "food_total": int(food_total),
        "births": int(stats["n_births"]),
        "deaths": int(stats["n_deaths"]),
        "final_alive": int(stats["n_alive"]),
        "max_gen": int(stats["max_generation"]),
        "cardinal_quality": float(q),
    }


def load_best_genome(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("final_best_genome", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_agents", type=int, default=5)
    ap.add_argument("--n_steps", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--output", type=str,
                     default="phase_13g_behavior_by_iss.json")
    args = ap.parse_args()

    # Load genomes from peak runs
    g_13d = load_best_genome("phase_13d_scaled.json")
    g_13e = load_best_genome("phase_13e_30shell.json")

    configs = [
        ("FLAT (7sh, pre-M1)", FLAT_SHELLS, None,              "ISS~35"),
        ("M4 (12sh, ISS 22.6)",   M4_SHELLS, None,              "ISS 22.6"),
        ("13d (20sh, ISS 81.9 PEAK)", SHELLS_20, g_13d if g_13d else None, "ISS 81.9"),
        ("13e (30sh, L=2.01)", SHELLS_30, g_13e if g_13e else None,     "ISS 69.8"),
    ]

    print("=" * 76, flush=True)
    print(f"  Phase 13g: Does higher ISS = better behavior?", flush=True)
    print(f"  {args.n_agents} agents x {args.n_steps} steps x {args.n_seeds} seeds",
          flush=True)
    print("=" * 76, flush=True)

    all_results = {}
    for label, shells, genome, iss_note in configs:
        print(f"\n  {label}  [{iss_note}]", flush=True)
        rs = []
        for seed in range(args.n_seeds):
            t0 = time.time()
            r = run_episode(shells, genome,
                             n_agents=args.n_agents, n_steps=args.n_steps,
                             seed=seed, world_params_seed=42 + seed)
            r["elapsed_s"] = round(time.time() - t0, 2)
            rs.append(r)
            print(f"    seed{seed}: food={r['food_total']:3d} "
                  f"births={r['births']:2d} alive={r['final_alive']} "
                  f"gen={r['max_gen']} q={r['cardinal_quality']:.2f} "
                  f"({r['elapsed_s']:.1f}s)", flush=True)
        all_results[label] = rs

    # Aggregate
    def agg(rs, k):
        return float(np.mean([r[k] for r in rs]))

    print(f"\n{'=' * 76}", flush=True)
    print(f"  AGGREGATE (mean over {args.n_seeds} seeds)", flush=True)
    print(f"{'=' * 76}", flush=True)
    print(f"  {'config':30s} {'food':>7s} {'births':>8s} {'alive':>7s} "
          f"{'q':>7s} {'time':>7s}", flush=True)
    for label, _, _, _ in configs:
        rs = all_results[label]
        print(f"  {label:30s} {agg(rs, 'food_total'):>7.1f} "
              f"{agg(rs, 'births'):>8.1f} "
              f"{agg(rs, 'final_alive'):>7.1f} "
              f"{agg(rs, 'cardinal_quality'):>7.2f} "
              f"{agg(rs, 'elapsed_s'):>7.2f}", flush=True)

    # Verdict
    flat_food = agg(all_results[configs[0][0]], "food_total")
    peak_food = agg(all_results[configs[2][0]], "food_total")
    flat_q = agg(all_results[configs[0][0]], "cardinal_quality")
    peak_q = agg(all_results[configs[2][0]], "cardinal_quality")

    print(f"\n  ANSWER TO USER QUESTION:", flush=True)
    print(f"  FLAT (ISS~35) food={flat_food:.1f} q={flat_q:.2f}", flush=True)
    print(f"  PEAK (ISS 82) food={peak_food:.1f} q={peak_q:.2f}", flush=True)
    if peak_food > flat_food + 1 and peak_q > flat_q + 0.2:
        print(f"  ★ ISS UP → BEHAVIOR UP. Structure IS function.", flush=True)
    elif peak_food < flat_food - 1:
        print(f"  ✗ ISS UP → BEHAVIOR DOWN. Structural gains DON'T transfer.",
              flush=True)
        print(f"    (Expected — DNA trained for flat. Cardinal needed to re-train.)",
              flush=True)
    else:
        print(f"  ~ ISS UP → BEHAVIOR FLAT. Structural gain not yet realized.",
              flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": all_results}, f,
                   indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
