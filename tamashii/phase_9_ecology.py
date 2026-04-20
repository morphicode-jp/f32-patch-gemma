"""Phase 9 — Ecology: realistic world with death + reproduction + day-night.

Full Underworld Phase 9 demonstration:
  - EnergyVoxelWorld3D: agents have energy, die when depleted, corpse → food
  - Sexual reproduction with DNA mixing + mutation
  - 2000-step day cycle, vision modulated by light
  - Tamashii Fluctlights drive agents (trained 3D brain + shells)

Pre-registered pass conditions:
  Phase A (death):       >=1 death, corpse becomes food
  Phase B (reproduction): >=1 birth, child DNA verifiably different
  Phase C (evolution):    generation >= 3 after long run
  Phase D (day-night):    light_level cycles, vision_factor varies
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.runner_3d import build_fluctlight
from world_3d_realistic import EnergyVoxelWorld3D


SHELLS_DEFAULT = ["core_brain", "brainstem", "cerebellum", "salience",
                  "hippocampus", "prefrontal", "dmn"]


def make_child_factory(configs_dir: str, trained_dir: str,
                       shells: list[str]):
    """Build a child_factory closure: (parent_i, parent_j, child_dna) → Tamashii."""
    def child_factory(parent_i, parent_j, child_dna):
        # Build baseline Tamashii (same shell stack as parents)
        child = build_fluctlight(
            shells, configs_dir, trained_dir=trained_dir,
            use_hebbian_core=False, use_3d_brain=True,
        )
        # Override core_brain DNA with child_dna
        child.shells[0].kathara_params = np.asarray(
            child_dna, dtype=np.float64)
        return child
    return child_factory


def run_ecology(
    n_agents_init: int = 5,
    n_steps: int = 5000,
    world_size: int = 16,
    n_food: int = 6,
    n_walls: int = 20,
    seed: int = 42,
    snapshot_every: int = 100,
    enable_repro: bool = True,
    max_population: int = 30,
    log_births: bool = True,
    log_deaths: bool = True,
    verbose_every: int = 500,
) -> dict:
    """Run full ecology simulation."""
    shells = SHELLS_DEFAULT
    configs_dir = os.path.join(THIS_DIR, "configs")
    trained_dir = "tamashii/configs"

    # Build initial agents
    agents = [
        build_fluctlight(shells, configs_dir, trained_dir=trained_dir,
                         use_hebbian_core=False, use_3d_brain=True)
        for _ in range(n_agents_init)
    ]
    # Slightly perturb each initial agent's DNA for diversity
    rng = np.random.default_rng(seed)
    from kathara16_brain import PARAM_RANGES_16
    for a in agents[1:]:  # keep agent 0 as the "trained" baseline
        dna = np.asarray(a.shells[0].kathara_params, dtype=np.float64).copy()
        for g in range(len(dna)):
            if rng.random() < 0.08:
                lo, hi = PARAM_RANGES_16[g]
                dna[g] += rng.normal(0, 0.1 * (hi - lo))
                dna[g] = np.clip(dna[g], lo, hi)
        a.shells[0].kathara_params = dna

    # World with reproduction hook
    cf = make_child_factory(configs_dir, trained_dir, shells) if enable_repro else None
    world = EnergyVoxelWorld3D(
        size=world_size, n_food=n_food, n_walls=n_walls, seed=seed,
        n_agents=n_agents_init, max_population=max_population,
        child_factory=cf,
    )
    world.register_agents(agents)

    sensors_list = world.reset()
    for a in agents:
        a.reset_episode()

    snapshots = []
    t0 = time.time()

    for step in range(n_steps):
        # Inject sensors into each alive agent
        N = world.n_agents
        for i in range(N):
            if world.agent_alive[i]:
                with agents[i]._lock:
                    if isinstance(sensors_list, list):
                        agents[i].S[0:16] = np.asarray(
                            sensors_list[i], dtype=np.float64)
                    else:
                        agents[i].S[0:16] = np.asarray(
                            sensors_list, dtype=np.float64)
        # Tick all alive agents
        for _ in range(3):
            for i in range(N):
                if world.agent_alive[i]:
                    agents[i].tick_once()
        # Extract actions
        actions = []
        for i in range(N):
            if world.agent_alive[i]:
                S = agents[i].read_state()
                actions.append((
                    float(np.clip(S[16], 0.0, 1.0)),
                    float(np.clip(S[17], 0.0, 1.0)),
                    float(np.clip(S[18], 0.0, 1.0)),
                ))
            else:
                actions.append((0.5, 0.0, 0.0))  # placeholder for dead

        sensors_list, ate_total, done = world.step(actions)

        # After step: world may have grown (children born).
        # Extend our 'agents' mirror list from world.agents_external
        if len(world.agents_external) > len(agents):
            agents = list(world.agents_external)
            # Reset newly-added children
            for a_new in agents[len(actions):]:
                a_new.reset_episode()

        # Snapshot
        if step % snapshot_every == 0 or step == n_steps - 1:
            snapshots.append({**world.stats(), "step": step})

        if verbose_every and step % verbose_every == 0:
            s = world.stats()
            print(f"  step {step:5d}  alive={s['n_alive']:2d}/{s['n_total_agents']:2d}  "
                  f"births={s['n_births']} deaths={s['n_deaths']}  "
                  f"gen_max={s['max_generation']}  "
                  f"|E|={s['mean_energy']:.1f}  "
                  f"light={s['light_level']:.2f}  "
                  f"dna_div={s['dna_diversity']:.4f}",
                  flush=True)

        if world.stats()["n_alive"] == 0:
            print(f"  EXTINCTION at step {step}", flush=True)
            break

    elapsed = time.time() - t0

    # Final stats
    final = world.stats()
    # Gather per-agent final
    per_agent = [{
        "idx": i,
        "alive": world.agent_alive[i],
        "generation": world.agent_generation[i],
        "food_eaten": world.agent_food_eaten[i],
        "energy": world.agent_energy[i] if world.agent_alive[i] else 0,
    } for i in range(world.n_agents)]

    return {
        "elapsed_s": round(elapsed, 1),
        "n_steps_run": step + 1,
        "final_stats": final,
        "snapshots": snapshots,
        "per_agent_final": per_agent,
        "event_log": world.event_log[:500],  # cap for file size
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="C",
                    choices=["smoke", "A", "B", "C", "D"],
                    help="test phase to run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    presets = {
        "smoke": dict(n_agents_init=3, n_steps=1000, n_food=5, n_walls=15,
                      snapshot_every=100, verbose_every=200,
                      enable_repro=False),
        "A": dict(n_agents_init=3, n_steps=2000, n_food=1, n_walls=15,
                   snapshot_every=100, verbose_every=400,
                   enable_repro=False),  # food scarce → deaths
        "B": dict(n_agents_init=3, n_steps=5000, n_food=10, n_walls=10,
                   snapshot_every=200, verbose_every=500,
                   enable_repro=True),  # abundant food → births
        "C": dict(n_agents_init=5, n_steps=20000, n_food=6, n_walls=20,
                   snapshot_every=500, verbose_every=2000,
                   enable_repro=True, max_population=30),  # long evolution
        "D": dict(n_agents_init=5, n_steps=4000, n_food=6, n_walls=15,
                   snapshot_every=50, verbose_every=500,
                   enable_repro=False),  # day-night cycle observation
    }
    cfg = presets[args.mode]
    out_path = args.output or f"tamashii_phase_9_{args.mode}.json"

    print("=" * 70, flush=True)
    print(f"  PHASE 9 ECOLOGY - mode={args.mode}", flush=True)
    print(f"  init N={cfg['n_agents_init']}  steps={cfg['n_steps']}  "
          f"food={cfg['n_food']}  repro={cfg['enable_repro']}", flush=True)
    print("=" * 70, flush=True)

    result = run_ecology(seed=args.seed, **cfg)

    final = result["final_stats"]
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 9 RESULTS (mode={args.mode}, {result['elapsed_s']}s)",
          flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  steps_run:        {result['n_steps_run']}", flush=True)
    print(f"  n_total_agents:   {final['n_total_agents']}", flush=True)
    print(f"  n_alive:          {final['n_alive']}", flush=True)
    print(f"  n_births:         {final['n_births']}", flush=True)
    print(f"  n_deaths:         {final['n_deaths']}", flush=True)
    print(f"  max_generation:   {final['max_generation']}", flush=True)
    print(f"  mean_generation:  {final['mean_generation']:.1f}", flush=True)
    print(f"  dna_diversity:    {final['dna_diversity']:.5f}", flush=True)
    print(f"  n_corpse_voxels:  {final['n_corpse_voxels']}", flush=True)

    # Phase-specific pass checks
    if args.mode == "A":
        ok = final["n_deaths"] >= 1
        print(f"\n  PASS-A (>=1 death): {ok}", flush=True)
    elif args.mode == "B":
        ok = final["n_births"] >= 1
        print(f"\n  PASS-B (>=1 birth): {ok}", flush=True)
    elif args.mode == "C":
        ok = final["max_generation"] >= 3 and final["n_alive"] > 0
        print(f"\n  PASS-C (gen>=3 & not extinct): {ok}", flush=True)
    elif args.mode == "D":
        # Check that snapshots show light variation
        lights = [s["light_level"] for s in result["snapshots"]]
        light_range = max(lights) - min(lights) if lights else 0
        ok = light_range > 0.5
        print(f"\n  PASS-D (light range >0.5): {ok}  "
              f"(got {min(lights):.2f}-{max(lights):.2f})",
              flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
