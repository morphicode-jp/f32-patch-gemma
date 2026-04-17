"""coop_world.py - Cooperative multi-agent FlyWorld.

Each agent has its own food. Scoring rewards BOTH reaching:
  agent_score = own_approach * 60 + partner_approach * 40 + (50 bonus if both reached)

Agent A can SEE its own food + hear agent B's voice.
Agent B can SEE its own food + hear agent A's voice.
Agent A's food is NOT visible to B directly (and vice versa).

-> B's food direction is only accessible to A via voice communication.
-> Selection pressure for honest signaling.
"""
import numpy as np


class CoopFlyWorld:
    def __init__(self, world_size=10.0, seed=None):
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        self.agent_pos = [None, None]
        self.agent_angle = [0.0, 0.0]
        self.food_pos = [None, None]  # one food per agent
        self.reached = [False, False]
        self.step_count = 0
        self.last_signal = [0.0, 0.0]

    def reset(self):
        self.agent_pos[0] = np.array([self.world_size * 0.25, self.world_size * 0.5])
        self.agent_pos[1] = np.array([self.world_size * 0.75, self.world_size * 0.5])
        self.agent_angle[0] = self.rng.uniform(0, 2 * np.pi)
        self.agent_angle[1] = self.rng.uniform(0, 2 * np.pi)
        margin = self.world_size * 0.2
        # Foods placed randomly. A's food might be near B, or vice versa.
        self.food_pos[0] = self.rng.uniform(margin, self.world_size - margin, size=2)
        self.food_pos[1] = self.rng.uniform(margin, self.world_size - margin, size=2)
        self.reached = [False, False]
        self.step_count = 0
        self.last_signal = [0.0, 0.0]
        return self._get_sensors(0), self._get_sensors(1)

    def _get_sensors(self, agent_idx):
        inputs = np.zeros(12)
        me = self.agent_pos[agent_idx]
        opp = self.agent_pos[1 - agent_idx]
        angle = self.agent_angle[agent_idx]
        my_food = self.food_pos[agent_idx]

        # Own food signal (7, 9, 10)
        diff = my_food - me
        dist = np.linalg.norm(diff) + 1e-6
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel = (angle_to_food - angle + np.pi) % (2 * np.pi) - np.pi
        inputs[7] = max(0.0, np.sin(rel)) / (dist * 0.3 + 1)
        inputs[9] = max(0.0, -np.sin(rel)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)

        # Opponent proximity (8)
        opp_dist = np.linalg.norm(opp - me) + 1e-6
        inputs[8] = 1.0 / (opp_dist * 0.5 + 0.5)

        # Heard signal (0)
        inputs[0] = self.last_signal[1 - agent_idx]

        return inputs

    def get_food_direction_categorical(self, agent_idx, for_food_of=None):
        """Direction of food (belonging to for_food_of) relative to agent_idx's heading."""
        if for_food_of is None:
            for_food_of = agent_idx
        me = self.agent_pos[agent_idx]
        angle = self.agent_angle[agent_idx]
        target_food = self.food_pos[for_food_of]
        diff = target_food - me
        dist = np.linalg.norm(diff)
        if dist < 1.0: return "near"
        at = np.arctan2(diff[1], diff[0])
        rel = (at - angle + np.pi) % (2 * np.pi) - np.pi
        return "left" if rel > 0.1 else ("right" if rel < -0.1 else "front")

    def step(self, nav0, cen0, voice0, nav1, cen1, voice1):
        self.step_count += 1
        self.last_signal = [float(voice0), float(voice1)]
        actions = [(nav0, cen0), (nav1, cen1)]
        for i in range(2):
            nav, cen = actions[i]
            turn = (nav - 0.5) * 0.6
            self.agent_angle[i] += turn
            speed = cen * 0.4
            self.agent_pos[i][0] += np.cos(self.agent_angle[i]) * speed
            self.agent_pos[i][1] += np.sin(self.agent_angle[i]) * speed
            self._handle_walls(i)

        for i in range(2):
            if not self.reached[i]:
                d = float(np.linalg.norm(self.food_pos[i] - self.agent_pos[i]))
                if d < 0.5:
                    self.reached[i] = True

        done = all(self.reached)
        return self._get_sensors(0), self._get_sensors(1), done

    def get_food_dist(self, agent_idx):
        return float(np.linalg.norm(self.food_pos[agent_idx] - self.agent_pos[agent_idx]))

    def _handle_walls(self, i):
        for dim in range(2):
            if self.agent_pos[i][dim] < 0:
                self.agent_pos[i][dim] = 0.0
                if dim == 0: self.agent_angle[i] = np.pi - self.agent_angle[i]
                else: self.agent_angle[i] = -self.agent_angle[i]
            elif self.agent_pos[i][dim] > self.world_size:
                self.agent_pos[i][dim] = self.world_size
                if dim == 0: self.agent_angle[i] = np.pi - self.agent_angle[i]
                else: self.agent_angle[i] = -self.agent_angle[i]
