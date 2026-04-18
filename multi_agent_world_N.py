"""multi_agent_world_N.py - N-agent cooperative FlyWorld with ToM sensors.

Generalization of coop_world.py to N agents (target: 5-20).

Each agent has:
  - Own food item (N foods total)
  - 13 sensor inputs including:
      0-2: own food direction (left/right/olfactory)
      3:   nearest other agent proximity
      4-6: nearest other agent's voice/nav/speed (Theory of Mind)
      7-9: 2nd nearest other agent's voice/nav/speed
      10-12: 3rd nearest
  - 3 motor outputs: nav (node 5), speed (node 11), voice (node 1)

Cooperative scoring (per agent):
  own_approach * 40
  + avg_partner_approach * 30
  + (all_reached ? 30 : 0)
  + (own_reached ? 20 : 0)

So agents benefit from BOTH own success AND partners' success. Cooperation
incentive + individual incentive combined.
"""
import numpy as np


class MultiAgentCoopWorldN:
    """N-agent cooperative world with voice + ToM observations."""

    def __init__(self, n_agents=5, world_size=14.0, seed=None,
                 n_tom_tracked=3):
        self.n_agents = n_agents
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        self.n_tom_tracked = n_tom_tracked  # number of nearest neighbors observed

        self.agent_pos = [None] * n_agents
        self.agent_angle = [0.0] * n_agents
        self.food_pos = [None] * n_agents
        self.reached = [False] * n_agents
        self.last_actions = [None] * n_agents  # (nav, speed, voice) per agent
        self.step_count = 0

    def reset(self):
        # Spawn agents on a circle for fair start
        for i in range(self.n_agents):
            theta = 2 * np.pi * i / self.n_agents
            r = self.world_size * 0.35
            self.agent_pos[i] = np.array([
                self.world_size / 2 + r * np.cos(theta),
                self.world_size / 2 + r * np.sin(theta),
            ])
            self.agent_angle[i] = self.rng.uniform(0, 2 * np.pi)

        # Spawn foods randomly
        margin = self.world_size * 0.15
        for i in range(self.n_agents):
            self.food_pos[i] = self.rng.uniform(
                margin, self.world_size - margin, size=2
            )

        self.reached = [False] * self.n_agents
        self.last_actions = [(0.5, 0.5, 0.0)] * self.n_agents  # neutral
        self.step_count = 0
        return self._get_all_sensors()

    def _get_all_sensors(self):
        """Returns list of 16-dim sensor vectors (one per agent)."""
        return [self._get_sensors(i) for i in range(self.n_agents)]

    def _get_sensors(self, agent_idx):
        """16-dim sensor for agent_idx. Last 3 dims reserved/zero."""
        inputs = np.zeros(16)
        me = self.agent_pos[agent_idx]
        angle = self.agent_angle[agent_idx]
        my_food = self.food_pos[agent_idx]

        # 0-2: own food direction
        diff = my_food - me
        dist = np.linalg.norm(diff) + 1e-6
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel = (angle_to_food - angle + np.pi) % (2 * np.pi) - np.pi
        # 0: left sensor, 1: placeholder, 2: right sensor (matches 16N layout)
        inputs[0] = max(0.0, np.sin(rel)) / (dist * 0.3 + 1)
        inputs[2] = max(0.0, -np.sin(rel)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)  # olfactory (pushed to channel 10)

        # Identify nearest N other agents by distance
        distances = []
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            d = float(np.linalg.norm(self.agent_pos[j] - me))
            distances.append((d, j))
        distances.sort()
        nearest = distances[:self.n_tom_tracked]

        # 3: nearest proximity
        if nearest:
            inputs[3] = 1.0 / (nearest[0][0] * 0.5 + 0.5)

        # ToM observations: (voice, nav, speed) for 3 nearest
        # channels 4-6 (1st), 7-9 (2nd), 10-12 (3rd)
        # But channel 10 is used for olfactory above - repurpose nearby channels
        # Let's place ToM at 4-6, 7-9, 11-13
        tom_channels = [(4, 5, 6), (7, 8, 9), (11, 12, 13)]
        for k, (d, j) in enumerate(nearest):
            if k >= len(tom_channels):
                break
            c_voice, c_nav, c_speed = tom_channels[k]
            if j < len(self.last_actions) and self.last_actions[j]:
                nav_j, speed_j, voice_j = self.last_actions[j]
                inputs[c_voice] = voice_j
                inputs[c_nav] = nav_j
                inputs[c_speed] = speed_j

        # Node 8 is inhibitory (Dale's law) - keep slightly lower baseline
        # inputs[14, 15] reserved
        return inputs

    def get_food_dist(self, agent_idx):
        return float(np.linalg.norm(
            self.food_pos[agent_idx] - self.agent_pos[agent_idx]))

    def get_own_food_direction(self, agent_idx):
        """Categorical direction of own food (for MI analysis)."""
        me = self.agent_pos[agent_idx]
        angle = self.agent_angle[agent_idx]
        diff = self.food_pos[agent_idx] - me
        dist = np.linalg.norm(diff)
        if dist < 1.0:
            return "near"
        at = np.arctan2(diff[1], diff[0])
        rel = (at - angle + np.pi) % (2 * np.pi) - np.pi
        return "left" if rel > 0.1 else ("right" if rel < -0.1 else "front")

    def step(self, actions):
        """Advance all agents simultaneously.

        actions: list of (nav, speed, voice) per agent
        Returns: (new_sensors_list, all_reached_flag, per_agent_reached_flags)
        """
        self.step_count += 1
        assert len(actions) == self.n_agents

        # Apply actions
        for i in range(self.n_agents):
            nav, speed, voice = actions[i]
            turn = (nav - 0.5) * 0.6
            self.agent_angle[i] += turn
            velocity = speed * 0.4
            self.agent_pos[i][0] += np.cos(self.agent_angle[i]) * velocity
            self.agent_pos[i][1] += np.sin(self.agent_angle[i]) * velocity
            self._handle_walls(i)
            self.last_actions[i] = (nav, speed, voice)

        # Check reaches
        reached_this_step = [False] * self.n_agents
        for i in range(self.n_agents):
            if not self.reached[i]:
                d = self.get_food_dist(i)
                if d < 0.5:
                    self.reached[i] = True
                    reached_this_step[i] = True

        all_reached = all(self.reached)
        return self._get_all_sensors(), all_reached, self.reached.copy()

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
    # Smoke test: 5-agent world, random actions, check reaches
    import time

    print("=" * 60)
    print("  MultiAgentCoopWorldN smoke test (N=5)")
    print("=" * 60)

    world = MultiAgentCoopWorldN(n_agents=5, seed=42)
    sensors_list = world.reset()
    print(f"\n  reset: {len(sensors_list)} agents, "
          f"sensor shape = {sensors_list[0].shape}")
    for i in range(5):
        print(f"    agent {i}: pos={world.agent_pos[i].round(1)} "
              f"food={world.food_pos[i].round(1)} "
              f"food_dist={world.get_food_dist(i):.2f}")

    # Run 50 steps with random actions
    rng = np.random.RandomState(0)
    t0 = time.time()
    for step in range(50):
        actions = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1))
                   for _ in range(5)]
        sensors_list, all_reached, reached = world.step(actions)
        if all_reached:
            print(f"\n  All reached at step {step}!")
            break
    else:
        print(f"\n  Timeout after 50 steps")
    print(f"  final reaches: {reached}")
    print(f"  elapsed: {(time.time() - t0) * 1000:.0f}ms")

    # Test ToM sensors are populated
    s = world._get_sensors(0)
    print(f"\n  Agent 0 sensor:")
    print(f"    food dir (0, 2, 10): {s[0]:.2f}, {s[2]:.2f}, {s[10]:.2f}")
    print(f"    nearest other (3):   {s[3]:.2f}")
    print(f"    ToM ch 4-6:          {s[4]:.2f}, {s[5]:.2f}, {s[6]:.2f}")
    print(f"    ToM ch 7-9:          {s[7]:.2f}, {s[8]:.2f}, {s[9]:.2f}")
    print(f"    ToM ch 11-13:        {s[11]:.2f}, {s[12]:.2f}, {s[13]:.2f}")

    # Larger N test
    print(f"\n  N=10 test:")
    world = MultiAgentCoopWorldN(n_agents=10, seed=42)
    world.reset()
    t0 = time.time()
    for _ in range(30):
        actions = [(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1))
                   for _ in range(10)]
        world.step(actions)
    print(f"    30 steps, 10 agents: {(time.time()-t0)*1000:.0f}ms")

    print("\n  Smoke test passed.")
