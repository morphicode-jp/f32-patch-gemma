"""world_3d_realistic.py — Underworld Phase 9 realistic physics world.

EnergyVoxelWorld3D(VoxelWorld3D): adds to base VoxelWorld3D the 4 biological
primitives that turn "sandbox" into "ecosystem":

  1. Energy + metabolism: agents have energy[0-100], decays 0.15/step, food +25
  2. Death: energy <= 0 -> alive=False, corpse becomes food voxel (recycle)
  3. Sexual reproduction: 2 adjacent alive agents + energy>60 each -> child
     Child DNA = per-gene random pick from parent1/parent2 + 5% Gaussian mutation
  4. Day-night cycle: 2000-step sinusoidal light_level, modulates raycast range

Population size is DYNAMIC (agents can be born/die). Index stability preserved
by using alive[i] flags rather than list shrinkage. multi-agent list length
may GROW as children are born.

API compatibility: all VoxelWorld3D methods retained. Dead agents:
  - get_sensors(i) returns 16D zero vector
  - step actions for dead agents are ignored
  - position/heading held frozen at death
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from world_3d import VoxelWorld3D


class EnergyVoxelWorld3D(VoxelWorld3D):
    """Voxel world with energy, death, reproduction, and day-night cycle."""

    def __init__(
        self,
        size: int = 16,
        n_food: int = 5,
        n_walls: int = 30,
        seed: int = 0,
        max_turn: float = 0.6,
        max_speed: float = 3.0,
        food_reach_radius: float = 1.0,
        respawn_food: bool = True,
        n_agents: int = 1,
        # Energy
        init_energy: float = 50.0,
        energy_max: float = 100.0,
        energy_decay_per_step: float = 0.15,
        food_energy_gain: float = 25.0,
        corpse_food_gain: float = 35.0,
        # Reproduction
        repro_energy_threshold: float = 60.0,
        repro_cost: float = 15.0,
        child_init_energy: float = 40.0,
        repro_distance: float = 2.0,
        repro_cooldown_steps: int = 200,
        mutation_rate: float = 0.05,
        mutation_sigma: float = 0.15,
        max_population: int = 50,
        # Reproduction hook: caller provides a function that builds a child
        # Tamashii given (parent_i_agent, parent_j_agent, child_dna)
        # If None, reproduction logic is disabled (for world-only tests)
        child_factory: Optional[Callable] = None,
        # Day-night
        day_cycle_steps: int = 2000,
        vision_min_factor: float = 0.3,
    ):
        super().__init__(
            size=size, n_food=n_food, n_walls=n_walls, seed=seed,
            max_turn=max_turn, max_speed=max_speed,
            food_reach_radius=food_reach_radius,
            respawn_food=respawn_food, n_agents=n_agents,
        )
        # Energy per agent
        self.init_energy = init_energy
        self.energy_max = energy_max
        self.energy_decay = energy_decay_per_step
        self.food_energy_gain = food_energy_gain
        self.corpse_food_gain = corpse_food_gain
        self.agent_energy = [init_energy] * n_agents
        self.agent_alive = [True] * n_agents
        self.agent_generation = [0] * n_agents
        # Corpse tracking (food voxels that came from death): position → gain
        self.corpse_voxels: dict[tuple[int, int], float] = {}

        # Reproduction
        self.repro_energy_threshold = repro_energy_threshold
        self.repro_cost = repro_cost
        self.child_init_energy = child_init_energy
        self.repro_distance = repro_distance
        self.repro_cooldown_steps = repro_cooldown_steps
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.max_population = max_population
        self.child_factory = child_factory
        self.agent_repro_cooldown = [0] * n_agents
        self.agents_external: list = []  # parallel list of Tamashii objects
        # filled by caller via register_agents()

        # Day-night
        self.day_cycle_steps = day_cycle_steps
        self.vision_min_factor = vision_min_factor

        # Event log for analysis
        self.event_log: list[dict] = []
        self.n_births = 0
        self.n_deaths = 0

    # ------------------------------------------------------------
    # External agent registration (required for reproduction)
    # ------------------------------------------------------------
    def register_agents(self, agents: list):
        """Register external Tamashii instances so world can spawn children.

        agents: list of Tamashii, len must equal self.n_agents.
        """
        assert len(agents) == self.n_agents, (
            f"register_agents: expected {self.n_agents}, got {len(agents)}")
        self.agents_external = list(agents)

    # ------------------------------------------------------------
    # Light level
    # ------------------------------------------------------------
    def light_level(self) -> float:
        """[0, 1], cosine cycle. 1=noon, 0=midnight. Period = day_cycle_steps."""
        phase = 2 * np.pi * self.steps_taken / max(1, self.day_cycle_steps)
        return float(0.5 + 0.5 * np.cos(phase))

    def vision_factor(self) -> float:
        """Multiplier for raycast max_dist based on light."""
        return self.vision_min_factor + (1.0 - self.vision_min_factor) * self.light_level()

    # ------------------------------------------------------------
    # Override raycast to apply vision factor
    # ------------------------------------------------------------
    def raycast_2d(self, yaw_offset: float, max_dist: float = 5.0,
                   step: float = 0.25, agent_idx: int = 0) -> float:
        effective_dist = max_dist * self.vision_factor()
        return super().raycast_2d(yaw_offset, effective_dist, step, agent_idx)

    # ------------------------------------------------------------
    # Override get_sensors: dead agents feel nothing, add light channel
    # ------------------------------------------------------------
    def get_sensors(self, agent_idx: int = 0) -> np.ndarray:
        if not self.agent_alive[agent_idx]:
            return np.zeros(16, dtype=np.float64)
        inputs = super().get_sensors(agent_idx)
        # Channel 15: light level (so agent has internal clock via sensor)
        # Does not conflict; 15 was scratchpad
        inputs[15] = self.light_level()
        return inputs

    # ------------------------------------------------------------
    # Override reset: restore all energy/alive state
    # ------------------------------------------------------------
    def reset(self):
        # Parent reset places living agents + food
        result = super().reset()
        # Restore energy/alive for existing agents (keep generation)
        for i in range(self.n_agents):
            self.agent_energy[i] = self.init_energy
            self.agent_alive[i] = True
            self.agent_repro_cooldown[i] = 0
        self.corpse_voxels.clear()
        self.event_log.clear()
        self.n_births = 0
        self.n_deaths = 0
        return result

    # ------------------------------------------------------------
    # Override step: energy dynamics + death + reproduction
    # ------------------------------------------------------------
    def step(self, nav, speed=None, voice=None):
        """Same signature as parent.

        For alive agents: apply actions normally, eat food = +energy.
        For dead agents: action ignored, no energy change.
        After motion: check death. Dead → corpse voxel.
        After death-check: attempt reproduction for adjacent high-energy pairs.
        """
        self.steps_taken += 1

        # Normalize actions to list form
        if self.n_agents == 1 and not isinstance(nav, (list, tuple)):
            actions = [(float(nav), float(speed) if speed is not None else 0.0,
                        float(voice) if voice is not None else 0.0)]
        else:
            actions_in = list(nav) if isinstance(nav, list) else nav
            actions = [(float(a[0]),
                        float(a[1]) if len(a) > 1 else 0.0,
                        float(a[2]) if len(a) > 2 else 0.0) for a in actions_in]

        if len(actions) < self.n_agents:
            # Pad with no-op for missing (e.g. children born mid-step)
            actions = list(actions) + [(0.5, 0.0, 0.0)] * (
                self.n_agents - len(actions))

        # 1. Apply motion for alive agents
        for a, (nav_a, speed_a, voice_a) in enumerate(actions[: self.n_agents]):
            if not self.agent_alive[a]:
                continue
            turn = (nav_a - 0.5) * 2.0 * self.max_turn
            self.agent_headings[a] = (self.agent_headings[a] + turn) % (2 * np.pi)
            spd = float(np.clip(speed_a, 0.0, 1.0)) * self.max_speed
            dx = np.cos(self.agent_headings[a]) * spd
            dy = np.sin(self.agent_headings[a]) * spd
            new_pos = self.agent_positions[a] + np.array([dx, dy, 0.0])
            new_pos[0] = np.clip(new_pos[0], 0.5, self.size - 0.5)
            new_pos[1] = np.clip(new_pos[1], 0.5, self.size - 0.5)
            cx, cy = int(new_pos[0]), int(new_pos[1])
            if self.grid[cx, cy] != 1:
                self.agent_positions[a] = new_pos

        # 2. Food consumption (alive agents only)
        ate_this_total = 0
        for a in range(self.n_agents):
            if not self.agent_alive[a]:
                continue
            to_remove = []
            for i, food_pos in enumerate(self.food_positions):
                if np.linalg.norm(self.agent_positions[a] - food_pos) < self.food_reach_radius:
                    fx, fy = int(food_pos[0]), int(food_pos[1])
                    self.grid[fx, fy] = 0
                    to_remove.append(i)
                    self.agent_food_eaten[a] += 1
                    ate_this_total += 1
                    # Is this a corpse (higher energy gain)?
                    corpse_gain = self.corpse_voxels.pop((fx, fy), None)
                    gain = corpse_gain if corpse_gain is not None else self.food_energy_gain
                    self.agent_energy[a] = min(
                        self.energy_max, self.agent_energy[a] + gain)
            for i in reversed(to_remove):
                self.food_positions.pop(i)
        # Respawn only natural food (corpse doesn't respawn)
        if self.respawn_food and ate_this_total > 0:
            self._spawn_food(ate_this_total)

        # 3. Metabolism: decay energy for alive agents
        for a in range(self.n_agents):
            if self.agent_alive[a]:
                self.agent_energy[a] -= self.energy_decay
                # Cooldown countdown
                if self.agent_repro_cooldown[a] > 0:
                    self.agent_repro_cooldown[a] -= 1

        # 4. Death check: alive with energy <= 0 becomes corpse
        for a in range(self.n_agents):
            if self.agent_alive[a] and self.agent_energy[a] <= 0:
                self.agent_alive[a] = False
                self.n_deaths += 1
                # Convert body to corpse food voxel
                px, py = int(self.agent_positions[a][0]), int(self.agent_positions[a][1])
                if 0 <= px < self.size and 0 <= py < self.size and self.grid[px, py] == 0:
                    self.grid[px, py] = 2
                    corpse_pos = np.array([px + 0.5, py + 0.5, 0.5], dtype=np.float64)
                    self.food_positions.append(corpse_pos)
                    self.corpse_voxels[(px, py)] = self.corpse_food_gain
                self.event_log.append({
                    "step": self.steps_taken, "type": "death", "agent": a,
                    "generation": self.agent_generation[a],
                    "food_eaten": self.agent_food_eaten[a],
                })

        # 5. Reproduction: attempt pairs
        self._attempt_reproduction(actions)

        # Record last actions (only for current-range agents, children added below are fresh)
        while len(self.agent_last_actions) < self.n_agents:
            self.agent_last_actions.append((0.5, 0.0, 0.0))
        for a in range(min(self.n_agents, len(actions))):
            if self.agent_alive[a]:
                self.agent_last_actions[a] = actions[a]

        done = False
        if self.n_agents == 1:
            return self.get_sensors(0), ate_this_total, done
        return [self.get_sensors(i) for i in range(self.n_agents)], \
            ate_this_total, done

    # ------------------------------------------------------------
    # Reproduction internals
    # ------------------------------------------------------------
    def _attempt_reproduction(self, actions):
        """Check all alive pairs for repro conditions, trigger child_factory."""
        if self.child_factory is None:
            return
        if not self.agents_external:
            return
        if self.n_agents >= self.max_population:
            return
        alive_idx = [i for i in range(self.n_agents) if self.agent_alive[i]]
        for ii, i in enumerate(alive_idx):
            for j in alive_idx[ii + 1:]:
                if self.n_agents >= self.max_population:
                    return
                if (self.agent_energy[i] > self.repro_energy_threshold and
                        self.agent_energy[j] > self.repro_energy_threshold and
                        self.agent_repro_cooldown[i] == 0 and
                        self.agent_repro_cooldown[j] == 0):
                    d = float(np.linalg.norm(
                        self.agent_positions[i] - self.agent_positions[j]))
                    if d < self.repro_distance:
                        self._reproduce(i, j)

    def _reproduce(self, pi: int, pj: int):
        """Create a child via DNA mixing + mutation from parents pi, pj."""
        from kathara16_brain import (
            PARAM_RANGES_16, TOTAL_PARAMS_16,
        )
        parent_i_obj = self.agents_external[pi]
        parent_j_obj = self.agents_external[pj]
        # Extract parent DNA (core_brain kathara_params)
        try:
            dna_i = parent_i_obj.shells[0].kathara_params
            dna_j = parent_j_obj.shells[0].kathara_params
        except Exception:
            return
        dna_i = np.asarray(dna_i, dtype=np.float64)
        dna_j = np.asarray(dna_j, dtype=np.float64)
        if dna_i.shape[0] != TOTAL_PARAMS_16 or dna_j.shape[0] != TOTAL_PARAMS_16:
            return
        # Per-gene random choice
        mask = self.rng.random(TOTAL_PARAMS_16) < 0.5
        child_dna = np.where(mask, dna_i, dna_j)
        # Mutation
        mutate_mask = self.rng.random(TOTAL_PARAMS_16) < self.mutation_rate
        for g_idx in np.where(mutate_mask)[0]:
            lo, hi = PARAM_RANGES_16[g_idx]
            child_dna[g_idx] += self.rng.normal(0, self.mutation_sigma * (hi - lo))
            child_dna[g_idx] = np.clip(child_dna[g_idx], lo, hi)
        # Child spawn position
        mid_xy = (self.agent_positions[pi][:2] + self.agent_positions[pj][:2]) / 2.0
        child_pos = np.array([mid_xy[0], mid_xy[1], 0.5], dtype=np.float64)
        child_pos[0] = np.clip(child_pos[0], 0.5, self.size - 0.5)
        child_pos[1] = np.clip(child_pos[1], 0.5, self.size - 0.5)
        # Parent cost
        self.agent_energy[pi] -= self.repro_cost
        self.agent_energy[pj] -= self.repro_cost
        self.agent_repro_cooldown[pi] = self.repro_cooldown_steps
        self.agent_repro_cooldown[pj] = self.repro_cooldown_steps
        # Ask caller to build child Tamashii with child_dna
        try:
            child_tamashii = self.child_factory(parent_i_obj, parent_j_obj, child_dna)
        except Exception as e:
            self.event_log.append({
                "step": self.steps_taken, "type": "repro_fail",
                "error": str(e)[:120],
            })
            return
        # Register child in world
        new_idx = self.n_agents
        self.n_agents += 1
        self.agent_positions.append(child_pos)
        self.agent_headings.append(float(self.rng.uniform(0, 2 * np.pi)))
        self.agent_food_eaten.append(0)
        self.agent_last_actions.append((0.5, 0.0, 0.0))
        self.agent_energy.append(self.child_init_energy)
        self.agent_alive.append(True)
        self.agent_repro_cooldown.append(self.repro_cooldown_steps)
        child_gen = max(self.agent_generation[pi], self.agent_generation[pj]) + 1
        self.agent_generation.append(child_gen)
        self.agents_external.append(child_tamashii)
        self.n_births += 1
        self.event_log.append({
            "step": self.steps_taken, "type": "birth",
            "parent_i": pi, "parent_j": pj, "child": new_idx,
            "generation": child_gen,
        })

    # ------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------
    def alive_indices(self) -> list[int]:
        return [i for i in range(self.n_agents) if self.agent_alive[i]]

    def stats(self) -> dict:
        alive = self.alive_indices()
        n_alive = len(alive)
        mean_energy = (float(np.mean([self.agent_energy[i] for i in alive]))
                       if alive else 0.0)
        gens = [self.agent_generation[i] for i in alive]
        mean_gen = float(np.mean(gens)) if gens else 0.0
        max_gen = max(gens) if gens else 0
        # DNA diversity: variance across alive agents' kathara_params
        dna_div = 0.0
        if len(alive) >= 2 and self.agents_external:
            params = []
            for i in alive:
                try:
                    p = self.agents_external[i].shells[0].kathara_params
                    params.append(np.asarray(p, dtype=np.float64))
                except Exception:
                    pass
            if len(params) >= 2:
                stacked = np.stack(params)
                dna_div = float(np.var(stacked, axis=0).mean())
        return {
            "steps": self.steps_taken,
            "n_total_agents": self.n_agents,
            "n_alive": n_alive,
            "n_births": self.n_births,
            "n_deaths": self.n_deaths,
            "mean_energy": mean_energy,
            "mean_generation": mean_gen,
            "max_generation": max_gen,
            "dna_diversity": dna_div,
            "light_level": self.light_level(),
            "vision_factor": self.vision_factor(),
            "n_food_voxels": len(self.food_positions),
            "n_corpse_voxels": len(self.corpse_voxels),
        }


if __name__ == "__main__":
    # Smoke test: world runs for N=3, no reproduction (no child_factory)
    w = EnergyVoxelWorld3D(size=12, n_food=4, n_walls=10, n_agents=3, seed=7)
    w.reset()
    for step in range(1000):
        actions = [(0.5 + 0.2 * np.sin(step * 0.1 + i), 0.6, 0.1)
                   for i in range(w.n_agents)]
        out = w.step(actions)
    s = w.stats()
    print("Smoke OK:", s)
