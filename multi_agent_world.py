"""multi_agent_world.py - 2-agent shared FlyWorld for co-evolution.

Phase 2 MVP: Two Kathara brains share a world with a single food source.
Each agent sees both food location AND opponent location. First-to-food wins.

Sensor layout (12D, preserves v8 conventions):
    [0-2]   reserved
    [3]     obstacle (NOT used here, = 0)
    [4-6]   reserved
    [7]     left_vis  -> signal towards food (from left side of head)
    [8]     danger    -> proximity to OPPONENT (reused as 'social threat')
    [9]     right_vis -> signal towards food (from right)
    [10]    olfactory -> 1/distance to food (smell)
    [11]    reserved (output side)

Key design choice:
  Node 8 is reinterpreted as "opponent proximity" (close opponent = danger-ish).
  This creates an incentive to either avoid or chase the opponent depending
  on learned behavior.
"""
import numpy as np


class MultiAgentFlyWorld:
    """Shared 2D world with one food and two competing agents."""

    def __init__(self, world_size=10.0, seed=None):
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        self.agent_pos = [None, None]
        self.agent_angle = [0.0, 0.0]
        self.food_pos = None
        self.winner = None  # None, 0, or 1
        self.step_count = 0

    def reset(self):
        # Agents spawn at opposite corners
        self.agent_pos[0] = np.array([self.world_size * 0.25, self.world_size * 0.5])
        self.agent_pos[1] = np.array([self.world_size * 0.75, self.world_size * 0.5])
        self.agent_angle[0] = self.rng.uniform(0, 2 * np.pi)
        self.agent_angle[1] = self.rng.uniform(0, 2 * np.pi)
        # Food spawns randomly
        margin = self.world_size * 0.2
        self.food_pos = self.rng.uniform(margin, self.world_size - margin, size=2)
        self.winner = None
        self.step_count = 0
        return self._get_sensors(0), self._get_sensors(1)

    def _get_sensors(self, agent_idx):
        """12D sensor for agent_idx (sees food + opponent)."""
        inputs = np.zeros(12)
        me = self.agent_pos[agent_idx]
        opp = self.agent_pos[1 - agent_idx]
        angle = self.agent_angle[agent_idx]

        # Food signal (nodes 7, 9, 10)
        diff = self.food_pos - me
        dist = np.linalg.norm(diff) + 1e-6
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel_angle = angle_to_food - angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        inputs[7] = max(0.0, np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[9] = max(0.0, -np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)

        # Opponent signal (node 8 = "danger/social")
        opp_diff = opp - me
        opp_dist = np.linalg.norm(opp_diff) + 1e-6
        # Strong when opponent is close (within 3 units)
        inputs[8] = 1.0 / (opp_dist * 0.5 + 0.5)

        return inputs

    def get_food_dist(self, agent_idx):
        return float(np.linalg.norm(self.food_pos - self.agent_pos[agent_idx]))

    def step(self, nav0, cen0, nav1, cen1):
        """Advance both agents simultaneously. Returns (sensors0, sensors1, done, reached_by)."""
        self.step_count += 1
        actions = [(nav0, cen0), (nav1, cen1)]
        for i in range(2):
            nav, cen = actions[i]
            turn = (nav - 0.5) * 0.6
            self.agent_angle[i] += turn
            speed = cen * 0.4
            self.agent_pos[i][0] += np.cos(self.agent_angle[i]) * speed
            self.agent_pos[i][1] += np.sin(self.agent_angle[i]) * speed
            self._handle_walls(i)

        # Check food reach (simultaneous; if both reach, closer wins)
        d0 = self.get_food_dist(0)
        d1 = self.get_food_dist(1)
        reached0 = d0 < 0.5
        reached1 = d1 < 0.5

        done = False
        reached_by = None
        if reached0 and reached1:
            reached_by = 0 if d0 <= d1 else 1
            done = True
        elif reached0:
            reached_by = 0
            done = True
        elif reached1:
            reached_by = 1
            done = True

        self.winner = reached_by
        return self._get_sensors(0), self._get_sensors(1), done, reached_by

    def _handle_walls(self, i):
        for dim in range(2):
            if self.agent_pos[i][dim] < 0:
                self.agent_pos[i][dim] = 0.0
                if dim == 0:
                    self.agent_angle[i] = np.pi - self.agent_angle[i]
                else:
                    self.agent_angle[i] = -self.agent_angle[i]
            elif self.agent_pos[i][dim] > self.world_size:
                self.agent_pos[i][dim] = self.world_size
                if dim == 0:
                    self.agent_angle[i] = np.pi - self.agent_angle[i]
                else:
                    self.agent_angle[i] = -self.agent_angle[i]


if __name__ == "__main__":
    # Smoke test
    world = MultiAgentFlyWorld(seed=42)
    s0, s1 = world.reset()
    print(f"Agent 0 pos: {world.agent_pos[0]}, sensors s7/s8/s9/s10: "
          f"{s0[7]:.2f} {s0[8]:.2f} {s0[9]:.2f} {s0[10]:.2f}")
    print(f"Agent 1 pos: {world.agent_pos[1]}, sensors s7/s8/s9/s10: "
          f"{s1[7]:.2f} {s1[8]:.2f} {s1[9]:.2f} {s1[10]:.2f}")
    print(f"Food pos: {world.food_pos}")

    # 30 steps, both random actions
    rng = np.random.RandomState(0)
    for step in range(30):
        nav0, cen0 = rng.uniform(0, 1), rng.uniform(0, 1)
        nav1, cen1 = rng.uniform(0, 1), rng.uniform(0, 1)
        s0, s1, done, winner = world.step(nav0, cen0, nav1, cen1)
        if done:
            print(f"Step {step}: Agent {winner} won! d0={world.get_food_dist(0):.2f} "
                  f"d1={world.get_food_dist(1):.2f}")
            break
    else:
        print(f"Timeout: d0={world.get_food_dist(0):.2f} d1={world.get_food_dist(1):.2f}")
