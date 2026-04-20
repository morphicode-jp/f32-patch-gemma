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
    """Minimal 3D voxel world, 1 agent (multi-agent extension later)."""

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
    ):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.max_turn = max_turn
        self.max_speed = max_speed
        self.food_reach_radius = food_reach_radius
        self.respawn_food = respawn_food
        self.n_food_target = n_food
        self.n_walls_target = n_walls

        # Grid: 0=empty, 1=wall, 2=food (z=0 is ground plane for walls/food)
        self.grid = np.zeros((size, size), dtype=int)
        self.food_positions: list[np.ndarray] = []
        self._place_walls(n_walls)
        self._spawn_food(n_food)

        # Agent state
        self.agent_pos = np.array([size / 2.0, size / 2.0, 0.5], dtype=np.float64)
        self.agent_heading = 0.0
        self.food_eaten = 0
        self.steps_taken = 0

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
        # Place agent at random empty position
        placed = False
        for _ in range(100):
            x, y = self.rng.integers(1, self.size - 1, 2)
            if self.grid[x, y] == 0:
                self.agent_pos = np.array(
                    [x + 0.5, y + 0.5, 0.5], dtype=np.float64)
                placed = True
                break
        if not placed:
            self.agent_pos = np.array(
                [self.size / 2, self.size / 2, 0.5], dtype=np.float64)
        self.agent_heading = float(self.rng.uniform(0, 2 * np.pi))
        self.food_eaten = 0
        self.steps_taken = 0
        return self.get_sensors()

    def step(self, nav: float, speed: float, voice: float = 0.0):
        """Advance one tick. Returns (sensors, food_eaten_this_step, done)."""
        self.steps_taken += 1

        # Turn
        turn = (float(nav) - 0.5) * 2.0 * self.max_turn
        self.agent_heading = (self.agent_heading + turn) % (2 * np.pi)

        # Move forward
        spd = float(np.clip(speed, 0.0, 1.0)) * self.max_speed
        dx = np.cos(self.agent_heading) * spd
        dy = np.sin(self.agent_heading) * spd
        new_pos = self.agent_pos + np.array([dx, dy, 0.0])

        # Wall collision (block at target 2D cell)
        ate_this = 0
        # Always clamp to valid range first
        new_pos[0] = np.clip(new_pos[0], 0.5, self.size - 0.5)
        new_pos[1] = np.clip(new_pos[1], 0.5, self.size - 0.5)
        cx, cy = int(new_pos[0]), int(new_pos[1])
        # Then block if wall
        if self.grid[cx, cy] != 1:
            self.agent_pos = new_pos

        # Food reached? Iterate by index to avoid np array equality issue
        to_remove = []
        for i, food_pos in enumerate(self.food_positions):
            if np.linalg.norm(self.agent_pos - food_pos) < self.food_reach_radius:
                fx, fy = int(food_pos[0]), int(food_pos[1])
                self.grid[fx, fy] = 0
                to_remove.append(i)
                self.food_eaten += 1
                ate_this += 1
        for i in reversed(to_remove):
            self.food_positions.pop(i)
        if self.respawn_food and ate_this > 0:
            self._spawn_food(ate_this)

        done = False  # infinite episode until caller stops
        return self.get_sensors(), ate_this, done

    def raycast_2d(self, yaw_offset: float, max_dist: float = 5.0,
                   step: float = 0.25) -> float:
        """Return proximity [0,1] to nearest wall in (heading+offset) direction."""
        yaw = self.agent_heading + yaw_offset
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        for t in np.arange(step, max_dist, step):
            x = self.agent_pos[0] + cos_y * t
            y = self.agent_pos[1] + sin_y * t
            if not (0 <= x < self.size and 0 <= y < self.size):
                return 1.0 - t / max_dist  # world boundary
            cx, cy = int(x), int(y)
            if self.grid[cx, cy] == 1:
                return 1.0 - t / max_dist  # wall hit
        return 0.0

    def food_direction(self) -> tuple[float, float, float]:
        """Return (left, right, olfactory) sensor triad for nearest food."""
        if not self.food_positions:
            return 0.0, 0.0, 0.0
        dists = [float(np.linalg.norm(self.agent_pos - f))
                 for f in self.food_positions]
        nearest_idx = int(np.argmin(dists))
        nearest = self.food_positions[nearest_idx]
        diff = nearest - self.agent_pos
        angle_to = float(np.arctan2(diff[1], diff[0]))
        rel = (angle_to - self.agent_heading + np.pi) % (2 * np.pi) - np.pi
        dist = dists[nearest_idx]
        left = max(0.0, float(np.sin(rel))) / (dist * 0.3 + 1)
        right = max(0.0, -float(np.sin(rel))) / (dist * 0.3 + 1)
        olfactory = 1.0 / (dist * 0.5 + 0.1)
        return left, right, olfactory

    def get_sensors(self) -> np.ndarray:
        """Return 16D sensor vector matching tamashii S[0:16] layout."""
        inputs = np.zeros(16, dtype=np.float64)
        left, right, olf = self.food_direction()
        inputs[0] = left
        inputs[2] = right
        inputs[10] = min(olf, 2.0)  # cap

        inputs[1] = 0.5 + 0.5 * float(np.sin(self.agent_heading))

        # 6 vision rays (deg: -70, -35, +35, +70, +90, -90)
        angles = [-1.22, -0.61, 0.61, 1.22, 1.57, -1.57]
        for i, off in enumerate(angles):
            inputs[4 + i] = self.raycast_2d(off, max_dist=5.0)

        # Forward wall proximity (close obstacle = survival-relevant)
        inputs[3] = self.raycast_2d(0.0, max_dist=3.0)

        # 11-13: reserved for ToM slots (other agents) — zero in single-agent MVP
        # 14-15: scratchpad

        return inputs

    def get_food_dist(self) -> float:
        if not self.food_positions:
            return float("inf")
        return float(min(np.linalg.norm(self.agent_pos - f)
                         for f in self.food_positions))

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
