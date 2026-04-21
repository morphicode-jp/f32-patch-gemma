"""Phase 10.3 — Cardinal: multiverse meta-evolution.

"宇宙自体が自分を最適化し続けてる" の実装。

Architecture:
  N parallel universes, each = (EnergyVoxelWorld3D with different world params) + M agents.
  Each epoch:
    1. Run each universe for T steps (agents live/die/breed within)
    2. Measure "universe quality" = diversity × longevity × emergence
    3. Meta-evolve: bottom worst universes replaced by variants (mutations) of
       the best ones. This applies Zenron's x_i ← best(perturb, share) at the
       WORLD level.

Run settings (default):
  N_universes = 4
  agents per universe start = 15
  epoch steps = 2500
  epochs = 4

Observation outputs:
  - Each universe's trajectory (n_alive, n_births, gen_max, dna_div, food_eaten)
  - World config parameters over time (which params thrive, which die)
  - Cross-epoch quality evolution
"""
from __future__ import annotations

import argparse
import copy
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
from tamashii.phase_9_ecology import make_child_factory, SHELLS_DEFAULT
from world_3d_realistic import EnergyVoxelWorld3D


# Parameters of a universe (= "physical laws" of that world)
# These are what meta-evolution perturbs
DEFAULT_WORLD_PARAMS = {
    "size": 14,
    "n_food": 6,
    "n_walls": 15,
    "energy_decay_per_step": 0.15,
    "food_energy_gain": 25.0,
    "corpse_food_gain": 35.0,
    "repro_energy_threshold": 60.0,
    "mutation_rate": 0.05,
    "mutation_sigma": 0.15,
    "day_cycle_steps": 2000,
    "vision_min_factor": 0.3,
    "max_population": 25,
}

# Which params can meta-evolve, and their ranges
PARAM_META_RANGES = {
    "n_food": (3, 10, "int"),
    "energy_decay_per_step": (0.08, 0.25, "float"),
    "food_energy_gain": (15.0, 35.0, "float"),
    "repro_energy_threshold": (50.0, 75.0, "float"),
    "mutation_rate": (0.02, 0.12, "float"),
    "mutation_sigma": (0.08, 0.25, "float"),
    "day_cycle_steps": (1000, 4000, "int"),
    "vision_min_factor": (0.1, 0.6, "float"),
}


def make_universe_params(base_seed: int = 42, variation: float = 0.3) -> dict:
    """Randomly sample a universe's parameters around defaults."""
    rng = np.random.default_rng(base_seed)
    params = dict(DEFAULT_WORLD_PARAMS)
    for key, (lo, hi, typ) in PARAM_META_RANGES.items():
        val = rng.uniform(lo, hi)
        if typ == "int":
            val = int(round(val))
        params[key] = val
    return params


def mutate_universe_params(parent: dict, sigma: float = 0.2,
                           seed: int = 0) -> dict:
    """Mutate a universe's parameters (for meta-evolution)."""
    rng = np.random.default_rng(seed)
    child = dict(parent)
    # Per-param Gaussian perturb within range
    for key, (lo, hi, typ) in PARAM_META_RANGES.items():
        if rng.random() < 0.5:  # mutate ~50% of meta-params
            continue
        delta = rng.normal(0, sigma * (hi - lo))
        val = parent[key] + delta
        val = np.clip(val, lo, hi)
        if typ == "int":
            val = int(round(val))
        child[key] = val
    return child


class Universe:
    """One universe = world config + agent pool + full state."""

    def __init__(self, universe_id: int, world_params: dict,
                 agents_init: int, world_seed: int,
                 shells: list[str], configs_dir: str, trained_dir: str):
        self.id = universe_id
        self.world_params = dict(world_params)
        self.shells = shells
        self.configs_dir = configs_dir
        self.trained_dir = trained_dir
        self.world_seed = world_seed
        self.agents_init = agents_init
        # Trajectory log
        self.trajectory = []
        self.initialize_agents()

    def initialize_agents(self):
        """Build initial agent pool (with small DNA diversity)."""
        from kathara16_brain import PARAM_RANGES_16
        agents = []
        rng = np.random.default_rng(self.world_seed + 1000)
        for i in range(self.agents_init):
            a = build_fluctlight(self.shells, self.configs_dir,
                                  trained_dir=self.trained_dir,
                                  use_hebbian_core=False, use_3d_brain=True)
            # Perturb DNA for diversity
            dna = np.asarray(a.shells[0].kathara_params,
                             dtype=np.float64).copy()
            for g in range(len(dna)):
                if rng.random() < 0.1:
                    lo, hi = PARAM_RANGES_16[g]
                    dna[g] += rng.normal(0, 0.12 * (hi - lo))
                    dna[g] = np.clip(dna[g], lo, hi)
            a.shells[0].kathara_params = dna
            agents.append(a)
        self.agents = agents
        child_factory = make_child_factory(self.configs_dir, self.trained_dir,
                                            self.shells)
        self.world = EnergyVoxelWorld3D(
            n_agents=self.agents_init,
            seed=self.world_seed,
            child_factory=child_factory,
            **self.world_params,
        )
        self.world.register_agents(agents)
        self.world.reset()
        for a in agents:
            a.reset_episode()

    def run_epoch(self, n_steps: int, log_every: int = 500):
        """Run this universe for n_steps."""
        sensors_list = [self.world.get_sensors(i)
                        for i in range(self.world.n_agents)]

        for step in range(n_steps):
            # Some agents may have died
            N = self.world.n_agents
            # Inject sensors
            for i in range(N):
                if self.world.agent_alive[i]:
                    with self.agents[i]._lock:
                        self.agents[i].S[0:16] = np.asarray(
                            sensors_list[i] if isinstance(sensors_list, list)
                            else sensors_list, dtype=np.float64)
            # Tick 3x per step (matches phase_9 pattern)
            for _ in range(3):
                for i in range(N):
                    if self.world.agent_alive[i]:
                        self.agents[i].tick_once()
            # Extract actions
            actions = []
            for i in range(N):
                if self.world.agent_alive[i]:
                    S = self.agents[i].read_state()
                    actions.append((
                        float(np.clip(S[16], 0.0, 1.0)),
                        float(np.clip(S[17], 0.0, 1.0)),
                        float(np.clip(S[18], 0.0, 1.0)),
                    ))
                else:
                    actions.append((0.5, 0.0, 0.0))
            sensors_list, ate, done = self.world.step(actions)
            # Children may have been added
            if len(self.world.agents_external) > len(self.agents):
                self.agents = list(self.world.agents_external)
                for new_a in self.agents[len(actions):]:
                    new_a.reset_episode()

            if step % log_every == 0:
                self.trajectory.append({"step": step, **self.world.stats()})

            if self.world.stats()["n_alive"] == 0:
                break

        # Final snapshot
        self.trajectory.append({"step": n_steps, **self.world.stats(), "final": True})

    def quality(self) -> float:
        """Universe quality metric.

        Composite:
          - longevity: log(n_alive + 1)
          - creativity: log(n_births + 1)
          - diversity: dna_diversity (0-1)
          - food_productivity: mean food_eaten per alive agent

        Higher = richer, more interesting universe (more likely to be
        preserved in meta-evolution).
        """
        s = self.world.stats()
        n_alive = s["n_alive"]
        if n_alive == 0:
            return 0.0
        n_births = s["n_births"]
        dna_div = s["dna_diversity"]
        # Food per alive
        alive_food = [self.world.agent_food_eaten[i]
                      for i in range(self.world.n_agents)
                      if self.world.agent_alive[i]]
        mean_food = float(np.mean(alive_food)) if alive_food else 0.0
        gen_max = s["max_generation"]

        quality = (
            np.log1p(n_alive) * 1.0 +
            np.log1p(n_births) * 1.5 +
            dna_div * 10.0 +
            np.log1p(mean_food) * 0.5 +
            gen_max * 0.8
        )
        return float(quality)


def run_cardinal(
    n_universes: int = 4,
    agents_per_universe: int = 15,
    epoch_steps: int = 2500,
    n_epochs: int = 4,
    base_seed: int = 42,
    output: str = "tamashii_phase_10_3_cardinal.json",
):
    shells = SHELLS_DEFAULT
    configs_dir = os.path.join(THIS_DIR, "configs")
    trained_dir = "tamashii/configs"

    print("=" * 72, flush=True)
    print(f"  CARDINAL — Multiverse meta-evolution", flush=True)
    print(f"  {n_universes} universes × {agents_per_universe} agents init × "
          f"{epoch_steps} steps/epoch × {n_epochs} epochs", flush=True)
    print("=" * 72, flush=True)

    # Initial N universes with random world params
    universes = []
    for u_id in range(n_universes):
        wp = make_universe_params(base_seed + u_id * 17, variation=0.4)
        u = Universe(
            universe_id=u_id,
            world_params=wp,
            agents_init=agents_per_universe,
            world_seed=base_seed + u_id * 101,
            shells=shells, configs_dir=configs_dir, trained_dir=trained_dir,
        )
        universes.append(u)

    # Meta-evolution history
    history = []
    t_total = time.time()

    for epoch in range(n_epochs):
        print(f"\n{'='*72}", flush=True)
        print(f"  EPOCH {epoch+1}/{n_epochs}", flush=True)
        print(f"{'='*72}", flush=True)

        epoch_start = time.time()
        # Run each universe
        for u_idx, u in enumerate(universes):
            t_u = time.time()
            u.run_epoch(epoch_steps, log_every=epoch_steps // 4)
            t_elapsed = time.time() - t_u
            s = u.world.stats()
            q = u.quality()
            print(f"  universe {u.id}: alive={s['n_alive']}/{s['n_total_agents']}  "
                  f"births={s['n_births']}  deaths={s['n_deaths']}  "
                  f"gen_max={s['max_generation']}  dna_div={s['dna_diversity']:.4f}  "
                  f"quality={q:.2f}  ({t_elapsed:.0f}s)", flush=True)

        # Rank universes by quality
        ranked = sorted(universes, key=lambda u: -u.quality())
        qualities = [u.quality() for u in ranked]
        print(f"\n  Ranking: {[u.id for u in ranked]}  qualities: "
              f"{[round(q,2) for q in qualities]}", flush=True)

        # Save snapshot
        history.append({
            "epoch": epoch,
            "universes": [{
                "id": u.id,
                "world_params": u.world_params,
                "stats": u.world.stats(),
                "quality": u.quality(),
                "trajectory": list(u.trajectory),
            } for u in ranked],
        })

        # Meta-evolve: bottom 1 replaced by mutation of best
        if epoch < n_epochs - 1 and len(ranked) >= 2:
            best = ranked[0]
            worst = ranked[-1]
            new_params = mutate_universe_params(
                best.world_params, sigma=0.25, seed=base_seed + epoch * 31)
            print(f"\n  META-EVOLVE: universe {worst.id} replaced by variant of "
                  f"universe {best.id}", flush=True)
            print(f"  New params: "
                  f"n_food={new_params['n_food']} "
                  f"decay={new_params['energy_decay_per_step']:.3f} "
                  f"food_gain={new_params['food_energy_gain']:.1f} "
                  f"mut_rate={new_params['mutation_rate']:.3f} "
                  f"day_cycle={new_params['day_cycle_steps']}",
                  flush=True)
            # Replace worst in-place
            new_universe = Universe(
                universe_id=worst.id,
                world_params=new_params,
                agents_init=agents_per_universe,
                world_seed=base_seed + worst.id * 101 + epoch * 7,
                shells=shells, configs_dir=configs_dir, trained_dir=trained_dir,
            )
            idx = universes.index(worst)
            universes[idx] = new_universe

        epoch_elapsed = time.time() - epoch_start
        print(f"  Epoch elapsed: {epoch_elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - t_total

    # Summary
    print(f"\n{'='*72}", flush=True)
    print(f"  CARDINAL SUMMARY ({total_elapsed:.0f}s = {total_elapsed/60:.1f}min)",
          flush=True)
    print(f"{'='*72}", flush=True)

    final_ranked = sorted(universes, key=lambda u: -u.quality())
    print(f"\n  Final ranking (highest quality first):", flush=True)
    for u in final_ranked:
        s = u.world.stats()
        print(f"    universe {u.id}: quality={u.quality():.2f}  "
              f"alive={s['n_alive']}  births={s['n_births']}  "
              f"gen_max={s['max_generation']}  "
              f"DNA div={s['dna_diversity']:.4f}", flush=True)

    # Quality trajectory across epochs
    print(f"\n  Quality evolution (best universe per epoch):", flush=True)
    for h in history:
        u_best = h["universes"][0]
        print(f"    epoch {h['epoch']}: best universe {u_best['id']} "
              f"quality={u_best['quality']:.2f}", flush=True)

    out = {
        "total_elapsed_s": round(total_elapsed, 1),
        "n_universes": n_universes,
        "agents_per_universe": agents_per_universe,
        "epoch_steps": epoch_steps,
        "n_epochs": n_epochs,
        "history": history,
        "final_ranking": [{
            "id": u.id,
            "quality": u.quality(),
            "world_params": u.world_params,
            "stats": u.world.stats(),
        } for u in final_ranked],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents_per_universe", type=int, default=15)
    ap.add_argument("--epoch_steps", type=int, default=2500)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_10_3_cardinal.json")
    args = ap.parse_args()

    run_cardinal(
        n_universes=args.n_universes,
        agents_per_universe=args.agents_per_universe,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
