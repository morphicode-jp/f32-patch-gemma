"""world_3d_gravity.py — Underworld Phase 11 REAL 3D physics.

Adds to EnergyVoxelWorld3D:
  1. Gravity: agents fall (z velocity integrates down)
  2. Jump action (4th action channel, read from S[19])
  3. Food at varying z (trees / ground / both)
  4. 3D vision: new sensor dim = relative food z direction
  5. Height-based effects (higher = more vision, but harder to reach)

Sensor layout (16D, backwards compatible):
  [0]     = left food direction
  [1]     = heading sin
  [2]     = right food direction
  [3]     = forward raycast (wall proximity)
  [4-9]   = 6 horizontal raycasts
  [10]    = olfactory (food strength)
  [11-14] = ToM channel (peer voice/nav/speed/proximity)
  [15]    = NEW: vertical food direction (0.5 = level, 1 = food above, 0 = below)

Action layout (4D):
  [0] nav    (from S[16])
  [1] speed  (from S[17])
  [2] voice  (from S[18])
  [3] jump   (from S[19], NEW) — values > 0.5 trigger jump if on ground

Compatibility: existing 3-tuple actions still work (jump defaults 0).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from world_3d_realistic import EnergyVoxelWorld3D


class GravityVoxelWorld3D(EnergyVoxelWorld3D):
    """Energy world + real z-axis dynamics."""

    def __init__(
        self,
        # z-axis params
        gravity: float = -0.08,          # per step, m/s^2 equivalent
        jump_impulse: float = 0.6,       # upward velocity on jump trigger
        max_height: float = 6.0,         # world z ceiling
        ground_z: float = 0.5,           # ground plane
        food_z_range: tuple = (0.5, 4.0),  # food spawn z distribution
        food_z_bimodal: bool = True,     # True = food at ground OR high; False = uniform
        vertical_food_frac: float = 0.4, # fraction of food at elevated z
        **kwargs,
    ):
        # Set z-axis attrs BEFORE super().__init__ (because super calls _spawn_food)
        self.gravity = gravity
        self.jump_impulse = jump_impulse
        self.max_height = max_height
        self.ground_z = ground_z
        self.food_z_range = food_z_range
        self.food_z_bimodal = food_z_bimodal
        self.vertical_food_frac = vertical_food_frac
        # Energy and base world init (calls _spawn_food)
        super().__init__(**kwargs)
        # z velocity per agent (after super so n_agents is set)
        self.agent_z_velocity = [0.0] * self.n_agents

    def _sample_food_z(self, idx: int = 0) -> float:
        """Sample z for a new food voxel."""
        use_rng = self.rng
        if self.food_z_bimodal:
            # Bimodal: ground (0.5) or canopy (high)
            if use_rng.random() < self.vertical_food_frac:
                z_lo, z_hi = self.food_z_range[0] + 1.5, self.food_z_range[1]
                return float(use_rng.uniform(z_lo, z_hi))
            else:
                return float(self.ground_z)
        else:
            return float(use_rng.uniform(*self.food_z_range))

    def _spawn_food(self, n_food: int):
        """Override to give food 3D position."""
        placed = 0
        attempts = 0
        while placed < n_food and attempts < n_food * 20:
            attempts += 1
            x, y = self.rng.integers(0, self.size, 2)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 2
                z = self._sample_food_z()
                self.food_positions.append(
                    np.array([x + 0.5, y + 0.5, z], dtype=np.float64))
                placed += 1

    def reset(self):
        super().reset()
        self.agent_z_velocity = [0.0] * self.n_agents
        # Ensure agent z is on ground at reset
        for i in range(self.n_agents):
            self.agent_positions[i][2] = self.ground_z

    def _apply_gravity_step(self):
        """Update z for alive agents: gravity + jump + clip."""
        # Children may have been added; pad z_velocity list
        while len(self.agent_z_velocity) < self.n_agents:
            self.agent_z_velocity.append(0.0)
        for a in range(self.n_agents):
            if not self.agent_alive[a]:
                continue
            # Apply gravity to velocity
            self.agent_z_velocity[a] += self.gravity
            # Update z
            new_z = self.agent_positions[a][2] + self.agent_z_velocity[a]
            # Ground collision (stop on ground)
            if new_z <= self.ground_z:
                new_z = self.ground_z
                # Fall damage: if landing velocity high, cost energy
                if self.agent_z_velocity[a] < -0.5:
                    fall_cost = abs(self.agent_z_velocity[a]) * 5.0
                    self.agent_energy[a] -= fall_cost
                self.agent_z_velocity[a] = 0.0
            # Ceiling
            elif new_z >= self.max_height:
                new_z = self.max_height
                self.agent_z_velocity[a] = 0.0
            self.agent_positions[a][2] = new_z

    def _apply_jump(self, a: int, jump_intent: float):
        """If agent on ground and jump_intent > 0.5, apply upward impulse."""
        if jump_intent <= 0.5:
            return
        # Extend z_velocity if children added
        while len(self.agent_z_velocity) < self.n_agents:
            self.agent_z_velocity.append(0.0)
        # Only jump if on/near ground
        if self.agent_positions[a][2] <= self.ground_z + 0.05:
            # Scale impulse by intent strength
            strength = (jump_intent - 0.5) * 2.0  # 0 to 1
            self.agent_z_velocity[a] = self.jump_impulse * (0.5 + 0.5 * strength)
            # Small energy cost
            self.agent_energy[a] -= 1.0

    def step(self, nav, speed=None, voice=None, jump=None):
        """Override step to handle 4th action (jump) + gravity."""
        # Extract jump intent from action tuples
        jumps = None
        if self.n_agents == 1 and not isinstance(nav, (list, tuple)):
            # Single agent scalar API
            jumps = [float(jump) if jump is not None else 0.0]
        else:
            actions_in = list(nav) if isinstance(nav, list) else nav
            jumps = []
            for act in actions_in:
                if len(act) >= 4:
                    jumps.append(float(act[3]))
                else:
                    jumps.append(0.0)
            if len(jumps) < self.n_agents:
                jumps += [0.0] * (self.n_agents - len(jumps))

        # Apply jumps BEFORE calling super (so z velocity set before gravity integrates)
        for a in range(self.n_agents):
            if self.agent_alive[a]:
                self._apply_jump(a, jumps[a])

        # Call parent step (handles xy motion, food, death, reproduction)
        result = super().step(nav, speed, voice)

        # Apply gravity AFTER parent step (integrate z)
        self._apply_gravity_step()

        # Food 3D reach: recompute food consumption w/ z-distance
        # (parent's 2D reach already happened; here we add z-gated re-check
        # for food that was missed due to being elevated)
        self._consume_elevated_food()

        return result

    def _consume_elevated_food(self):
        """Give agents chance to eat food within 3D reach radius."""
        ate_extra = 0
        for a in range(self.n_agents):
            if not self.agent_alive[a]:
                continue
            agent_pos = self.agent_positions[a]
            to_remove = []
            for i, food_pos in enumerate(self.food_positions):
                d3 = float(np.linalg.norm(agent_pos - food_pos))
                # Only re-check elevated food we missed
                if abs(food_pos[2] - self.ground_z) < 0.1:
                    continue
                if d3 < self.food_reach_radius:
                    fx, fy = int(food_pos[0]), int(food_pos[1])
                    self.grid[fx, fy] = 0
                    to_remove.append(i)
                    self.agent_food_eaten[a] += 1
                    ate_extra += 1
                    corpse_gain = self.corpse_voxels.pop((fx, fy), None)
                    gain = corpse_gain if corpse_gain is not None else self.food_energy_gain
                    self.agent_energy[a] = min(
                        self.energy_max, self.agent_energy[a] + gain)
            for i in reversed(to_remove):
                self.food_positions.pop(i)
        if self.respawn_food and ate_extra > 0:
            self._spawn_food(ate_extra)

    def get_sensors(self, agent_idx: int = 0) -> np.ndarray:
        """Extend base sensors with vertical food direction in inputs[15]."""
        inputs = super().get_sensors(agent_idx)
        if not self.agent_alive[agent_idx]:
            return inputs
        # Vertical food direction: 0.5 = level, 1 = nearest food above, 0 = below
        if not self.food_positions:
            inputs[15] = 0.5
            return inputs
        my_pos = self.agent_positions[agent_idx]
        # Find nearest food (by 3D distance)
        nearest = None
        nearest_d = float("inf")
        for fp in self.food_positions:
            d = float(np.linalg.norm(my_pos - fp))
            if d < nearest_d:
                nearest_d = d
                nearest = fp
        if nearest is not None:
            dz = float(nearest[2]) - float(my_pos[2])
            # Normalize: +max_height → ~1, -max_height → ~0
            inputs[15] = float(np.clip(0.5 + dz / (2.0 * self.max_height),
                                         0.0, 1.0))
        return inputs

    def stats(self) -> dict:
        """Extend stats with z-axis metrics."""
        s = super().stats()
        alive = [i for i in range(self.n_agents) if self.agent_alive[i]]
        z_values = [self.agent_positions[i][2] for i in alive]
        s["mean_agent_z"] = float(np.mean(z_values)) if z_values else 0.0
        s["max_agent_z"] = float(max(z_values)) if z_values else 0.0
        food_z_values = [f[2] for f in self.food_positions]
        s["mean_food_z"] = float(np.mean(food_z_values)) if food_z_values else 0.0
        s["n_elevated_food"] = sum(1 for z in food_z_values
                                     if z > self.ground_z + 0.3)
        return s


if __name__ == "__main__":
    # Quick smoke test
    w = GravityVoxelWorld3D(size=12, n_agents=2, n_food=6, seed=42)
    print(f"Initial food z values: {[f[2] for f in w.food_positions]}")
    print(f"Agent 0 initial pos: {w.agent_positions[0]}")

    # 10 steps with jump on step 2
    for step in range(10):
        actions = [(0.5, 0.3, 0.0, 1.0 if step == 2 else 0.0),
                     (0.5, 0.3, 0.0, 0.0)]
        w.step(actions)
        a0_z = w.agent_positions[0][2]
        a0_vz = w.agent_z_velocity[0]
        print(f"step {step}: a0 z={a0_z:.3f} vz={a0_vz:+.3f} "
              f"e={w.agent_energy[0]:.1f}")

    print(f"\nFinal stats: {w.stats()}")
