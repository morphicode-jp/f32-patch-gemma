"""Phase 11 — Cardinal with REAL 3D physics (gravity + jump + vertical food).

Extends tamashii.phase_10_3_cardinal Universe to use GravityVoxelWorld3D.
Adds gravity/jump/max_height as meta-evolvable parameters so Cardinal can
speciate universes (low-gravity flyers vs high-gravity ground-dwellers).

Usage:
  from tamashii.phase_11_cardinal_3d import GravityUniverse, make_3d_universe_params
  u = GravityUniverse(u_id=0, world_params=make_3d_universe_params(42),
                       agents_init=6, world_seed=42, shells=..., configs_dir=..., trained_dir=...)
  u.run_epoch(1500)
"""
from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from world_3d_gravity import GravityVoxelWorld3D
from tamashii.phase_10_3_cardinal import (
    Universe, DEFAULT_WORLD_PARAMS as BASE_DEFAULTS,
    PARAM_META_RANGES as BASE_RANGES,
)
from tamashii.phase_9_ecology import make_child_factory


# Extended defaults with 3D physics params
DEFAULT_3D_WORLD_PARAMS = {
    **BASE_DEFAULTS,
    "gravity": -0.08,
    "jump_impulse": 0.6,
    "max_height": 6.0,
    "vertical_food_frac": 0.4,
}

# Extended meta ranges — gravity/jump/height EVOLVE → speciation possible
# Narrower ranges so agents can survive initial universes; mutation explores edges
PARAM_META_RANGES_3D = {
    **BASE_RANGES,
    "gravity": (-0.06, -0.02, "float"),          # gentle; mutation can push harder
    "jump_impulse": (0.4, 0.9, "float"),          # modest to strong
    "max_height": (3.0, 7.0, "float"),            # world ceiling
    "vertical_food_frac": (0.05, 0.35, "float"),  # mostly ground with some elevation
}


def make_3d_universe_params(base_seed: int = 42) -> dict:
    """Sample a 3D universe's parameters (includes gravity, jump, height)."""
    rng = np.random.default_rng(base_seed)
    params = dict(DEFAULT_3D_WORLD_PARAMS)
    for key, (lo, hi, typ) in PARAM_META_RANGES_3D.items():
        val = rng.uniform(lo, hi)
        if typ == "int":
            val = int(round(val))
        params[key] = val
    return params


def mutate_3d_universe_params(parent: dict, sigma: float = 0.2,
                                 seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    child = dict(parent)
    for key, (lo, hi, typ) in PARAM_META_RANGES_3D.items():
        if rng.random() < 0.5:
            continue
        delta = rng.normal(0, sigma * (hi - lo))
        val = parent.get(key, (lo + hi) / 2) + delta
        val = np.clip(val, lo, hi)
        if typ == "int":
            val = int(round(val))
        child[key] = val
    return child


class GravityUniverse(Universe):
    """Universe using GravityVoxelWorld3D — real z-axis physics."""

    def initialize_agents(self):
        """Rebuild using GravityVoxelWorld3D instead of EnergyVoxelWorld3D."""
        from kathara16_brain import PARAM_RANGES_16
        from tamashii.runner_3d import build_fluctlight
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
        # Strip keys that are specific to GravityVoxelWorld3D and pass them as kwargs
        self.world = GravityVoxelWorld3D(
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
        """Run with 4-action (nav, speed, voice, jump) read from S[16..19]."""
        sensors_list = [self.world.get_sensors(i)
                        for i in range(self.world.n_agents)]
        for step in range(n_steps):
            N = self.world.n_agents
            for i in range(N):
                if self.world.agent_alive[i]:
                    with self.agents[i]._lock:
                        self.agents[i].S[0:16] = np.asarray(
                            sensors_list[i] if isinstance(sensors_list, list)
                            else sensors_list, dtype=np.float64)
            for _ in range(3):
                for i in range(N):
                    if self.world.agent_alive[i]:
                        self.agents[i].tick_once()
            # Extract 4 actions (nav, speed, voice, jump)
            actions = []
            for i in range(N):
                if self.world.agent_alive[i]:
                    S = self.agents[i].read_state()
                    actions.append((
                        float(np.clip(S[16], 0.0, 1.0)),
                        float(np.clip(S[17], 0.0, 1.0)),
                        float(np.clip(S[18], 0.0, 1.0)),
                        float(np.clip(S[19], 0.0, 1.0)) if len(S) > 19 else 0.0,
                    ))
                else:
                    actions.append((0.5, 0.0, 0.0, 0.0))
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
        self.trajectory.append({"step": n_steps, **self.world.stats(), "final": True})


if __name__ == "__main__":
    # Smoke test: 1 universe × 3 agents × 200 steps
    import time
    print("Building GravityUniverse...")
    wp = make_3d_universe_params(42)
    print(f"Universe params (3D): gravity={wp['gravity']:.3f} jump={wp['jump_impulse']:.2f} "
          f"max_height={wp['max_height']:.1f} vert_food={wp['vertical_food_frac']:.2f}")
    u = GravityUniverse(
        universe_id=0, world_params=wp, agents_init=3,
        world_seed=42,
        shells=["core_brain", "brainstem", "cerebellum"],
        configs_dir="tamashii/configs",
        trained_dir="tamashii/configs",
    )
    t0 = time.time()
    u.run_epoch(100, log_every=50)
    elapsed = time.time() - t0
    stats = u.world.stats()
    print(f"\n100 steps in {elapsed:.1f}s")
    print(f"n_alive: {stats['n_alive']}")
    print(f"mean_agent_z: {stats['mean_agent_z']:.2f}")
    print(f"max_agent_z:  {stats['max_agent_z']:.2f}")
    print(f"mean_food_z:  {stats['mean_food_z']:.2f}")
    print(f"n_elevated_food: {stats['n_elevated_food']}")
    print(f"quality: {u.quality():.2f}")
