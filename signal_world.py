"""signal_world.py - Multi-agent FlyWorld with inter-agent signaling.

Phase 3 MVP (language seed):
  Each agent has a "voice" output and can "hear" the other.
    Sensor node 0: heard signal from opponent (previous step)
    Motor node 1: own voice output

  After co-evolution, test if signals carry information:
    Mutual Information( agent_A_signal_t, food_direction_A_t ) > chance?

If yes -> proto-language (signal correlates with perceived world).
"""
import numpy as np


class SignalFlyWorld:
    """Same as MultiAgentFlyWorld but with voice signals."""

    def __init__(self, world_size=10.0, seed=None):
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        self.agent_pos = [None, None]
        self.agent_angle = [0.0, 0.0]
        self.food_pos = None
        self.winner = None
        self.step_count = 0
        self.last_signal = [0.0, 0.0]  # last voice output of each agent

    def reset(self):
        self.agent_pos[0] = np.array([self.world_size * 0.25, self.world_size * 0.5])
        self.agent_pos[1] = np.array([self.world_size * 0.75, self.world_size * 0.5])
        self.agent_angle[0] = self.rng.uniform(0, 2 * np.pi)
        self.agent_angle[1] = self.rng.uniform(0, 2 * np.pi)
        margin = self.world_size * 0.2
        self.food_pos = self.rng.uniform(margin, self.world_size - margin, size=2)
        self.winner = None
        self.step_count = 0
        self.last_signal = [0.0, 0.0]
        return self._get_sensors(0), self._get_sensors(1)

    def _get_sensors(self, agent_idx):
        inputs = np.zeros(12)
        me = self.agent_pos[agent_idx]
        opp = self.agent_pos[1 - agent_idx]
        angle = self.agent_angle[agent_idx]

        diff = self.food_pos - me
        dist = np.linalg.norm(diff) + 1e-6
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel_angle = angle_to_food - angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        inputs[7] = max(0.0, np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[9] = max(0.0, -np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)

        opp_diff = opp - me
        opp_dist = np.linalg.norm(opp_diff) + 1e-6
        inputs[8] = 1.0 / (opp_dist * 0.5 + 0.5)

        # NEW: heard signal from opponent
        inputs[0] = self.last_signal[1 - agent_idx]

        return inputs

    def get_food_dist(self, agent_idx):
        return float(np.linalg.norm(self.food_pos - self.agent_pos[agent_idx]))

    def get_food_direction(self, agent_idx):
        """Returns 'left', 'right', or 'near' relative to agent heading."""
        me = self.agent_pos[agent_idx]
        angle = self.agent_angle[agent_idx]
        diff = self.food_pos - me
        dist = np.linalg.norm(diff)
        if dist < 1.0:
            return "near"
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel = (angle_to_food - angle + np.pi) % (2 * np.pi) - np.pi
        return "left" if rel > 0.1 else ("right" if rel < -0.1 else "front")

    def step(self, nav0, cen0, voice0, nav1, cen1, voice1):
        """Advance with voice signals. voice in [0,1]."""
        self.step_count += 1
        # Store new signals for NEXT step's sensors
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
            reached_by = 0; done = True
        elif reached1:
            reached_by = 1; done = True
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
