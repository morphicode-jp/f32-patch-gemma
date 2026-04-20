"""world_3d.py — minimal 3D voxel world for tamashii Fluctlight.

Underworld Phase 4 MVP: 3D body + 3D vision + physics + multiple entities.

Design constraints (MVP, not final):
  - 16x16x16 voxel grid (4096 cells, fast sim)
  - Agent is a point in continuous 3D space with yaw heading
  - Agent stays on z=0 ground for MVP (no jump/fly yet)
  - Voxels: 0=empty, 1=wall, 2=food
  - Vision: 7 raycasts (forward + 6 directional), returns [0,1] proximity
  - Olfactory: 1/dist to nearest food (biological smell gradient)
  - Collision: wall block prevents movement
  - Food reached = food_eaten counter increment + voxel cleared

Sensor output (16D, matches tamashii S[0:16]):
  [0] left_food_direction      (sin of angle-to-food, positive side)
  [1] head_orient_y            (sin of heading, useful for disambiguation)
  [2] right_food_direction     (sin of angle-to-food, negative side)
  [3] forward_wall_proximity   (1/dist to wall directly ahead)
  [4-9] 6 vision rays          (proximity in ±35°, ±70°, ±90°)
  [10] olfactory               (1/dist to nearest food)
  [11-13] touch (front, below, side) — reserved ToM in multi-agent
  [14-15] free / scratch

Motor input (3):
  nav   in [0,1] → turn rate = (nav - 0.5) * max_turn
  speed in [0,1] → forward velocity
  voice in [0,1] → vocal signal (multi-agent later)
"""
from __future__ import annotations

import numpy as np


class VoxelWorld3D:
    """Minimal 3D voxel world with optional multi-agent support.

    n_agents >= 1. When n_agents > 1, each agent has its own position/heading
    and sees the others via extra ToM channels in sensor vector.
    """

    def __init__(
        self,
        size: int = 16,
        n_food: int = 5,
        n_walls: int = 30,
        seed: int = 0,
        max_turn: float = 0.6,       # rad/step
        max_speed: float = 3.0,      # voxels/step (boosted: tamashii motor nodes output small values)
        food_reach_radius: float = 1.0,
        respawn_food: bool = True,
        n_agents: int = 1,           # multi-agent support
    ):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.max_turn = max_turn
        self.max_speed = max_speed
        self.food_reach_radius = food_reach_radius
        self.respawn_food = respawn_food
        self.n_food_target = n_food
        self.n_walls_target = n_walls
        self.n_agents = int(n_agents)

        # Grid: 0=empty, 1=wall, 2=food (z=0 is ground plane for walls/food)
        self.grid = np.zeros((size, size), dtype=int)
        self.food_positions: list[np.ndarray] = []
        self._place_walls(n_walls)
        self._spawn_food(n_food)

        # Agents state (multi-agent). For n_agents=1, behavior matches old API.
        self.agent_positions = [
            np.array([size / 2.0, size / 2.0, 0.5], dtype=np.float64)
            for _ in range(self.n_agents)
        ]
        self.agent_headings = [0.0] * self.n_agents
        self.agent_food_eaten = [0] * self.n_agents
        # Previous-step actions (for ToM channel: peers see neighbor's last action)
        self.agent_last_actions = [(0.5, 0.0, 0.0)] * self.n_agents
        self.steps_taken = 0

    # Backward-compat properties (1-agent view)
    @property
    def agent_pos(self):
        return self.agent_positions[0]

    @agent_pos.setter
    def agent_pos(self, val):
        self.agent_positions[0] = val

    @property
    def agent_heading(self):
        return self.agent_headings[0]

    @agent_heading.setter
    def agent_heading(self, val):
        self.agent_headings[0] = val

    @property
    def food_eaten(self):
        # Sum across all agents for aggregate "world ate count"
        return sum(self.agent_food_eaten)

    @food_eaten.setter
    def food_eaten(self, val):
        # Used only for reset (val=0); distribute zero
        if val == 0:
            self.agent_food_eaten = [0] * self.n_agents

    def _place_walls(self, n_walls: int):
        placed = 0
        attempts = 0
        while placed < n_walls and attempts < n_walls * 10:
            attempts += 1
            x, y = self.rng.integers(0, self.size, 2)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 1
                placed += 1

    def _spawn_food(self, n_food: int):
        placed = 0
        attempts = 0
        while placed < n_food and attempts < n_food * 20:
            attempts += 1
            x, y = self.rng.integers(0, self.size, 2)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 2
                self.food_positions.append(
                    np.array([x + 0.5, y + 0.5, 0.5], dtype=np.float64))
                placed += 1

    def reset(self):
        # Place each agent at distinct random empty positions
        for a in range(self.n_agents):
            placed = False
            for _ in range(100):
                x, y = self.rng.integers(1, self.size - 1, 2)
                if self.grid[x, y] == 0:
                    # Don't overlap existing agents
                    conflict = False
                    for prev in range(a):
                        if (int(self.agent_positions[prev][0]) == x and
                                int(self.agent_positions[prev][1]) == y):
                            conflict = True
                            break
                    if conflict:
                        continue
                    self.agent_positions[a] = np.array(
                        [x + 0.5, y + 0.5, 0.5], dtype=np.float64)
                    placed = True
                    break
            if not placed:
                self.agent_positions[a] = np.array(
                    [self.size / 2 + a, self.size / 2, 0.5], dtype=np.float64)
            self.agent_headings[a] = float(self.rng.uniform(0, 2 * np.pi))
        self.agent_food_eaten = [0] * self.n_agents
        self.agent_last_actions = [(0.5, 0.0, 0.0)] * self.n_agents
        self.steps_taken = 0
        if self.n_agents == 1:
            return self.get_sensors(0)
        return [self.get_sensors(i) for i in range(self.n_agents)]

    def step(self, nav, speed=None, voice=None):
        """Advance one tick.

        Single-agent API (back-compat): step(nav, speed, voice)
          Returns (sensors, ate_this_step, done)
        Multi-agent API: step([(nav,speed,voice), ...])
          Returns (sensors_list, ate_total, done)
        """
        self.steps_taken += 1

        # Normalize action input to list-of-tuples
        if self.n_agents == 1 and not isinstance(nav, (list, tuple)):
            actions = [(float(nav), float(speed) if speed is not None else 0.0,
                        float(voice) if voice is not None else 0.0)]
        else:
            actions = list(nav) if isinstance(nav, list) else nav
            # Coerce to (f,f,f)
            actions = [(float(a[0]),
                        float(a[1]) if len(a) > 1 else 0.0,
                        float(a[2]) if len(a) > 2 else 0.0) for a in actions]

        assert len(actions) == self.n_agents, (
            f"expected {self.n_agents} actions, got {len(actions)}")

        # Apply each agent's motion, resolve wall collision per agent
        for a, (nav_a, speed_a, voice_a) in enumerate(actions):
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

        # Food eaten? (per-agent attribution)
        ate_this_total = 0
        for a in range(self.n_agents):
            to_remove = []
            for i, food_pos in enumerate(self.food_positions):
                if np.linalg.norm(self.agent_positions[a] - food_pos) < self.food_reach_radius:
                    fx, fy = int(food_pos[0]), int(food_pos[1])
                    self.grid[fx, fy] = 0
                    to_remove.append(i)
                    self.agent_food_eaten[a] += 1
                    ate_this_total += 1
            for i in reversed(to_remove):
                self.food_positions.pop(i)
        if self.respawn_food and ate_this_total > 0:
            self._spawn_food(ate_this_total)

        # Record last actions (for ToM channel)
        self.agent_last_actions = list(actions)

        done = False
        if self.n_agents == 1:
            return self.get_sensors(0), ate_this_total, done
        return [self.get_sensors(i) for i in range(self.n_agents)], \
            ate_this_total, done

    def raycast_2d(self, yaw_offset: float, max_dist: float = 5.0,
                   step: float = 0.25, agent_idx: int = 0) -> float:
        """Proximity [0,1] to wall in (heading+offset) from agent_idx."""
        yaw = self.agent_headings[agent_idx] + yaw_offset
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        start = self.agent_positions[agent_idx]
        for t in np.arange(step, max_dist, step):
            x = start[0] + cos_y * t
            y = start[1] + sin_y * t
            if not (0 <= x < self.size and 0 <= y < self.size):
                return 1.0 - t / max_dist
            cx, cy = int(x), int(y)
            if self.grid[cx, cy] == 1:
                return 1.0 - t / max_dist
        return 0.0

    def food_direction(self, agent_idx: int = 0) -> tuple[float, float, float]:
        if not self.food_positions:
            return 0.0, 0.0, 0.0
        pos = self.agent_positions[agent_idx]
        heading = self.agent_headings[agent_idx]
        dists = [float(np.linalg.norm(pos - f)) for f in self.food_positions]
        nearest_idx = int(np.argmin(dists))
        nearest = self.food_positions[nearest_idx]
        diff = nearest - pos
        angle_to = float(np.arctan2(diff[1], diff[0]))
        rel = (angle_to - heading + np.pi) % (2 * np.pi) - np.pi
        dist = dists[nearest_idx]
        left = max(0.0, float(np.sin(rel))) / (dist * 0.3 + 1)
        right = max(0.0, -float(np.sin(rel))) / (dist * 0.3 + 1)
        olfactory = 1.0 / (dist * 0.5 + 0.1)
        return left, right, olfactory

    def get_sensors(self, agent_idx: int = 0) -> np.ndarray:
        """16D sensor vector. Channels 11-13 = ToM (nearest peer voice/nav/speed)."""
        inputs = np.zeros(16, dtype=np.float64)
        left, right, olf = self.food_direction(agent_idx)
        inputs[0] = left
        inputs[2] = right
        inputs[10] = min(olf, 2.0)
        inputs[1] = 0.5 + 0.5 * float(np.sin(self.agent_headings[agent_idx]))

        angles = [-1.22, -0.61, 0.61, 1.22, 1.57, -1.57]
        for i, off in enumerate(angles):
            inputs[4 + i] = self.raycast_2d(off, max_dist=5.0, agent_idx=agent_idx)
        inputs[3] = self.raycast_2d(0.0, max_dist=3.0, agent_idx=agent_idx)

        # ToM channel: nearest OTHER agent's last action (voice/nav/speed)
        if self.n_agents > 1:
            me_pos = self.agent_positions[agent_idx]
            nearest_peer = None
            nearest_d = float("inf")
            for j in range(self.n_agents):
                if j == agent_idx:
                    continue
                d = float(np.linalg.norm(self.agent_positions[j] - me_pos))
                if d < nearest_d:
                    nearest_d = d
                    nearest_peer = j
            if nearest_peer is not None:
                nav_p, speed_p, voice_p = self.agent_last_actions[nearest_peer]
                inputs[11] = voice_p
                inputs[12] = nav_p
                inputs[13] = speed_p
                # 14: peer proximity (1/dist, biological "someone's here")
                inputs[14] = 1.0 / (nearest_d * 0.5 + 1.0)

        return inputs

    def get_food_dist(self, agent_idx: int = 0) -> float:
        if not self.food_positions:
            return float("inf")
        pos = self.agent_positions[agent_idx]
        return float(min(np.linalg.norm(pos - f) for f in self.food_positions))

    def state_summary(self) -> dict:
        return {
            "agent_pos": self.agent_pos.tolist(),
            "agent_heading": float(self.agent_heading),
            "food_eaten": int(self.food_eaten),
            "n_food_remaining": len(self.food_positions),
            "steps": int(self.steps_taken),
        }


if __name__ == "__main__":
    # Smoke: random actions, see if world is sane
    world = VoxelWorld3D(size=16, n_food=5, n_walls=30, seed=42)
    sensors = world.reset()
    print(f"Sensors init: {sensors.round(3)}")
    rng = np.random.default_rng(7)
    for s in range(200):
        nav = float(rng.uniform(0, 1))
        speed = float(rng.uniform(0, 1))
        sensors, ate, done = world.step(nav, speed)
        if ate:
            print(f"  step {s}: ate {ate}, total={world.food_eaten}")
    print(f"Final: {world.state_summary()}")
