"""Kathara Brain v8 - Biological Structure Principles

v7: FlyWorld Evolution + Mode B+ + Fetal development
v8: Biological structure principles from paper_brain_structure_intelligence.md
  (1) Dale's law: 20% inhibitory nodes (GABA interneurons)
  (2) Asymmetric connections: FF:FB = 2:1
  (3) Subsystem specialization: sensor(26%inhib), memory(9%), motor(23%)
  (4) Inhibition test suite: 6 experiments validating each principle

Key insight (v7 failure analysis):
  12N beats 48N on V3p because 48N lacks inhibitory "don't go" signals.
  Dale's law (20% GABA) is an evolutionary constant across ALL species.
  Node 8 in Kathara = inhibitory interneuron (wiring_is_intelligence.py).
  FF:FB = 2.13:1 enables directional information flow.

Usage:
    python kathara_brain_sim_v8.py --mode inhib_test_12n --time 300
    python kathara_brain_sim_v8.py --mode inhib_test_48n --time 600
    python kathara_brain_sim_v8.py --mode v3p --time 120
"""

import numpy as np
import time
import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Kathara 12/30 + simulate (v6 reuse)
# ============================================================
KATHARA_EDGES = []
for i in range(12):
    for offset in [1, 4, 6]:
        j = (i + offset) % 12
        if (i, j) not in KATHARA_EDGES and (j, i) not in KATHARA_EDGES:
            KATHARA_EDGES.append((i, j))
assert len(KATHARA_EDGES) == 30

NEIGHBORS = [[] for _ in range(12)]
for idx, (a, b) in enumerate(KATHARA_EDGES):
    NEIGHBORS[a].append((b, idx))
    NEIGHBORS[b].append((a, idx))

SENSORY_NODES = [3, 7, 8, 9, 10]  # obstacle, left_vis, danger, right_vis, olfactory
MOTOR_NODES = [5, 11]              # navigation, central

PARAMS_PER_NODE = 5
TOTAL_PARAMS = 30 + 12 * PARAMS_PER_NODE + 1  # 91

NODE_NAMES = {
    0: "node_0", 1: "node_1", 2: "node_2", 3: "obstacle",
    4: "node_4", 5: "navigation", 6: "node_6",
    7: "left_vis", 8: "danger", 9: "right_vis",
    10: "olfactory", 11: "central"
}

PARAM_RANGES = (
    [(-3.0, 3.0)] * 30 +
    [(0.1, 5.0), (-3.0, 3.0), (0.1, 5.0), (0.01, 0.9), (0.01, 1.0)] * 12 +
    [(10, 80)]
)

PARAM_NAMES = []
for i in range(30):
    a, b = KATHARA_EDGES[i]
    PARAM_NAMES.append(f"e{a}_{b}")
for i in range(12):
    name = NODE_NAMES[i]
    for pname in ["gain", "bias", "iw", "leak", "thr"]:
        PARAM_NAMES.append(f"{name}_{pname}")
PARAM_NAMES.append("n_steps")


def simulate_step(params, inputs, states, inhibit_sign=None):
    """1-step brain simulation with optional Dale's law. Returns (new_states, firing)."""
    edge_weights = params[:30]
    node_params = params[30:90].reshape(12, PARAMS_PER_NODE)
    gains = node_params[:, 0]
    biases = node_params[:, 1]
    input_ws = node_params[:, 2]
    leaks = node_params[:, 3]
    thresholds = node_params[:, 4]

    # Dale's law: inhibitory nodes flip output sign
    effective_states = states * inhibit_sign if inhibit_sign is not None else states

    new_states = np.zeros(12, dtype=np.float64)
    for i in range(12):
        syn_input = 0.0
        for j, eidx in NEIGHBORS[i]:
            syn_input += effective_states[j] * edge_weights[eidx]
        total = syn_input + input_ws[i] * inputs[i] + biases[i]
        x = gains[i] * total
        activation = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        if activation < thresholds[i]:
            activation *= 0.1
        new_states[i] = activation

    states_out = states * (1 - leaks) + new_states * leaks
    states_out = np.clip(states_out, 0, 1)
    return states_out, states_out.copy()


def compute_step_reward(prev_dist, curr_dist, reached, moved):
    """Step reward."""
    reward = (prev_dist - curr_dist) * 2.0
    reward += 10.0 * float(reached)
    reward += 0.1 if moved > 0.05 else -0.2
    return reward


# ============================================================
# FlyWorld V1/V2/V3
# ============================================================
class FlyWorldV1:
    """Single food, no obstacles. Same as v6."""

    def __init__(self, world_size=10.0, seed=None, **kwargs):
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        self.fly_pos = None
        self.fly_angle = 0.0
        self.food_pos = None

    def reset(self):
        self.fly_pos = np.array([self.world_size / 2, self.world_size / 2])
        self.fly_angle = self.rng.uniform(0, 2 * np.pi)
        margin = self.world_size * 0.1
        self.food_pos = self.rng.uniform(margin, self.world_size - margin, size=2)
        return self._get_sensors()

    def _get_sensors(self):
        inputs = np.zeros(12)
        diff = self.food_pos - self.fly_pos
        dist = np.linalg.norm(diff) + 1e-6
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel_angle = angle_to_food - self.fly_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        inputs[7] = max(0.0, np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[9] = max(0.0, -np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)
        return inputs

    def get_food_dist(self):
        return np.linalg.norm(self.food_pos - self.fly_pos)

    def step(self, nav, cen):
        turn = (nav - 0.5) * 0.6
        self.fly_angle += turn
        speed = cen * 0.4
        self.fly_pos[0] += np.cos(self.fly_angle) * speed
        self.fly_pos[1] += np.sin(self.fly_angle) * speed
        self._handle_walls()
        dist = self.get_food_dist()
        reached = dist < 0.5
        return self._get_sensors(), dist, reached, False  # 4-tuple unified

    def _handle_walls(self):
        for dim in range(2):
            if self.fly_pos[dim] < 0:
                self.fly_pos[dim] = 0.0
                self.fly_angle = np.pi - self.fly_angle if dim == 0 else -self.fly_angle
            elif self.fly_pos[dim] > self.world_size:
                self.fly_pos[dim] = self.world_size
                self.fly_angle = np.pi - self.fly_angle if dim == 0 else -self.fly_angle


class FlyWorldV2(FlyWorldV1):
    """Multiple foods (3). Reach any = success."""

    def __init__(self, n_foods=3, **kwargs):
        super().__init__(**kwargs)
        self.n_foods = n_foods
        self.food_positions = []
        self.foods_reached = []

    def reset(self):
        self.fly_pos = np.array([self.world_size / 2, self.world_size / 2])
        self.fly_angle = self.rng.uniform(0, 2 * np.pi)
        margin = self.world_size * 0.1
        self.food_positions = [
            self.rng.uniform(margin, self.world_size - margin, size=2)
            for _ in range(self.n_foods)
        ]
        self.foods_reached = [False] * self.n_foods
        return self._get_sensors()

    def _get_sensors(self):
        inputs = np.zeros(12)
        min_dist = float('inf')
        best_diff = None
        for i, fp in enumerate(self.food_positions):
            if self.foods_reached[i]:
                continue
            d = np.linalg.norm(fp - self.fly_pos)
            if d < min_dist:
                min_dist = d
                best_diff = fp - self.fly_pos
        if best_diff is None:
            return inputs
        dist = min_dist + 1e-6
        angle_to_food = np.arctan2(best_diff[1], best_diff[0])
        rel_angle = angle_to_food - self.fly_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        inputs[7] = max(0.0, np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[9] = max(0.0, -np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)
        return inputs

    def get_food_dist(self):
        dists = [np.linalg.norm(fp - self.fly_pos)
                 for i, fp in enumerate(self.food_positions)
                 if not self.foods_reached[i]]
        return min(dists) if dists else 0.0

    def step(self, nav, cen):
        turn = (nav - 0.5) * 0.6
        self.fly_angle += turn
        speed = cen * 0.4
        self.fly_pos[0] += np.cos(self.fly_angle) * speed
        self.fly_pos[1] += np.sin(self.fly_angle) * speed
        self._handle_walls()
        # Check reach for each food
        reached = False
        for i, fp in enumerate(self.food_positions):
            if not self.foods_reached[i] and np.linalg.norm(fp - self.fly_pos) < 0.5:
                self.foods_reached[i] = True
                reached = True
                break
        dist = self.get_food_dist()
        return self._get_sensors(), dist, reached, False


class FlyWorldV3(FlyWorldV1):
    """Obstacles + optional predator.

    New sensors:
        Node 3 (obstacle): proximity to nearest obstacle
        Node 8 (danger):   proximity to predator
    """

    def __init__(self, n_obstacles=3, has_predator=False, predator_speed=0.15, **kwargs):
        super().__init__(**kwargs)
        self.n_obstacles = n_obstacles
        self.has_predator = has_predator
        self.predator_speed = predator_speed
        self.obstacles = []
        self.predator_pos = None

    def reset(self):
        self.fly_pos = np.array([self.world_size / 2, self.world_size / 2])
        self.fly_angle = self.rng.uniform(0, 2 * np.pi)
        margin = self.world_size * 0.1
        self.food_pos = self.rng.uniform(margin, self.world_size - margin, size=2)

        # Generate obstacles between fly and food
        self.obstacles = []
        for _ in range(self.n_obstacles):
            w = self.rng.uniform(0.8, 2.0)
            h = self.rng.uniform(0.8, 2.0)
            t = self.rng.uniform(0.2, 0.8)
            center = self.fly_pos * (1 - t) + self.food_pos * t
            center += self.rng.uniform(-1.5, 1.5, size=2)
            center = np.clip(center, 0, self.world_size)
            self.obstacles.append((center[0] - w / 2, center[1] - h / 2, w, h))

        # Predator from map edge
        if self.has_predator:
            side = self.rng.randint(4)
            if side == 0:
                self.predator_pos = np.array([self.rng.uniform(0, self.world_size), 0.0])
            elif side == 1:
                self.predator_pos = np.array([self.world_size, self.rng.uniform(0, self.world_size)])
            elif side == 2:
                self.predator_pos = np.array([self.rng.uniform(0, self.world_size), self.world_size])
            else:
                self.predator_pos = np.array([0.0, self.rng.uniform(0, self.world_size)])
        else:
            self.predator_pos = None

        return self._get_sensors()

    def _point_in_obstacle(self, pos):
        for ox, oy, ow, oh in self.obstacles:
            if ox <= pos[0] <= ox + ow and oy <= pos[1] <= oy + oh:
                return True
        return False

    def _obstacle_proximity(self, pos):
        min_dist = float('inf')
        for ox, oy, ow, oh in self.obstacles:
            cx = np.clip(pos[0], ox, ox + ow)
            cy = np.clip(pos[1], oy, oy + oh)
            d = np.linalg.norm(pos - np.array([cx, cy]))
            min_dist = min(min_dist, d)
        return 1.0 / (min_dist + 0.1) if self.obstacles else 0.0

    def _get_sensors(self):
        inputs = np.zeros(12)
        # Food sensors (same as V1)
        diff = self.food_pos - self.fly_pos
        dist = np.linalg.norm(diff) + 1e-6
        angle_to_food = np.arctan2(diff[1], diff[0])
        rel_angle = angle_to_food - self.fly_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
        inputs[7] = max(0.0, np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[9] = max(0.0, -np.sin(rel_angle)) / (dist * 0.3 + 1)
        inputs[10] = 1.0 / (dist * 0.5 + 0.1)
        # Obstacle proximity (node 3)
        inputs[3] = self._obstacle_proximity(self.fly_pos)
        # Danger/predator (node 8)
        if self.predator_pos is not None:
            pred_dist = np.linalg.norm(self.predator_pos - self.fly_pos) + 1e-6
            inputs[8] = 1.0 / (pred_dist * 0.3 + 0.1)
        return inputs

    def step(self, nav, cen):
        turn = (nav - 0.5) * 0.6
        self.fly_angle += turn
        speed = cen * 0.4
        new_pos = self.fly_pos.copy()
        new_pos[0] += np.cos(self.fly_angle) * speed
        new_pos[1] += np.sin(self.fly_angle) * speed

        # Obstacle collision: cancel move + rotate 90deg
        if self._point_in_obstacle(new_pos):
            self.fly_angle += np.pi * 0.5
        else:
            self.fly_pos = new_pos

        self._handle_walls()

        # Predator chases fly
        caught = False
        if self.predator_pos is not None:
            pred_dir = self.fly_pos - self.predator_pos
            pred_dist = np.linalg.norm(pred_dir) + 1e-6
            self.predator_pos += (pred_dir / pred_dist) * self.predator_speed
            if pred_dist < 0.5:
                caught = True

        dist = self.get_food_dist()
        reached = dist < 0.5
        return self._get_sensors(), dist, reached, caught


# ============================================================
# Mode B+ (tuned Mode B)
# ============================================================
OPTIMAL_RL = {
    "lr": 0.05,
    "momentum": 0.8,
    "reward_threshold": 0.01,  # update when |reward| > this
    "n_episodes": 8,
}


def load_v5_warm_start():
    """Load v5 best_params for warm-start."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "kathara_brain_sim_v5_result.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("best_params"):
            return np.array(data["best_params"], dtype=np.float64)
    return None


def run_mode_b_plus(world_factory, time_budget=120, warm_start=None,
                    n_episodes=8, verbose=True, label="B+", inhibit_ratio=0.0):
    """Mode B+ = v6 Mode B Hebbian + warm-start + 8 episodes.

    v8: inhibit_ratio designates fraction of nodes as inhibitory (Dale's law).
    """
    lr = OPTIMAL_RL["lr"]
    momentum_coeff = OPTIMAL_RL["momentum"]
    reward_thr = OPTIMAL_RL["reward_threshold"]

    # v8: Build inhibit_sign vector
    inhibit_sign = None
    if inhibit_ratio > 0:
        inhibit_sign = np.ones(12)
        n_inhib = max(1, int(12 * inhibit_ratio))
        rng_inh = np.random.RandomState(42)
        # Node 8 = danger/inhibitory (biological role)
        inhibit_sign[8] = -1.0
        n_inhib -= 1
        if n_inhib > 0:
            # Exclude motor nodes (5, 11) and already-assigned node 8
            candidates = [i for i in range(12) if i not in {5, 8, 11}]
            chosen = rng_inh.choice(candidates, min(n_inhib, len(candidates)), replace=False)
            inhibit_sign[chosen] = -1.0
        n_actual = int((inhibit_sign < 0).sum())
        inhib_str = f"inhib={n_actual}/12 ({n_actual/12*100:.0f}%)"
    else:
        inhib_str = "no inhibition"

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Mode {label}")
        print(f"  RL: lr={lr}, mom={momentum_coeff}, "
              f"reward_thr={reward_thr}, eps={n_episodes}")
        ws_info = "warm-start" if warm_start is not None else "random"
        print(f"  Init: {ws_info}, {inhib_str}")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)

    if warm_start is not None:
        # Full warm-start when params come from Hebbian (progression chain)
        base_params = warm_start.copy()
    else:
        base_params = np.array([rng.uniform(lo, hi) for lo, hi in PARAM_RANGES])

    best_params = base_params.copy()
    best_score = -float('inf')
    n_evals = 0
    scores_log = []
    reach_counts = []

    while time.time() - t0 < time_budget:
        params = base_params.copy()
        # Light perturbation on node params only
        for i in range(30, 91):
            lo, hi = PARAM_RANGES[i]
            params[i] += rng.normal(0, 0.1 * (hi - lo))
            params[i] = np.clip(params[i], lo, hi)

        momentum = np.zeros(30, dtype=np.float64)
        ep_scores = []
        ep_reached = 0

        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors = world.reset()
            initial_dist = world.get_food_dist()
            states = np.zeros(12, dtype=np.float64)
            min_dist = initial_dist
            reached = False
            caught = False
            move_sum = 0.0

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                states, firing = simulate_step(params, sensors, states,
                                               inhibit_sign=inhibit_sign)
                nav = firing[5]
                cen = firing[11]

                sensors, dist, reached_now, caught_now = world.step(nav, cen)
                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                # Hebbian edge update (v6 proven params)
                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))
                if abs(reward) > reward_thr:
                    for eidx, (a, b) in enumerate(KATHARA_EDGES):
                        hebbian = firing[a] * firing[b] * reward * lr
                        momentum[eidx] = momentum_coeff * momentum[eidx] + hebbian
                        params[eidx] += momentum[eidx]
                        params[eidx] = np.clip(params[eidx], -3.0, 3.0)

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        avg_score = np.mean(ep_scores)
        n_evals += 1

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()
            base_params = params.copy()

        scores_log.append(best_score)
        reach_counts.append(ep_reached)

        if verbose and n_evals % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] eval={n_evals}, best={best_score:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    elapsed = time.time() - t0
    avg_reach = np.mean(reach_counts[-20:]) if reach_counts else 0

    if verbose:
        print(f"  {label} done: best={best_score:.2f}, evals={n_evals}, "
              f"reach_rate={avg_reach / n_episodes:.0%}, {elapsed:.0f}s")

    return {
        "mode": label,
        "best_score": best_score,
        "best_params": list(best_params),
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "avg_reach_rate": avg_reach / n_episodes,
        "scores_log": scores_log[-20:],
        "rl_params": OPTIMAL_RL,
        "inhibit_ratio": inhibit_ratio,
    }


# ============================================================
# Multi-layer Hebbian
# ============================================================
def simulate_step_multilayer(node_params_flat, layer_edge_weights, inputs, layer_states,
                              feedback=None, inhibit_sign=None):
    """Multi-layer 1-step with skip connections + feedback + Dale's law.

    v8: inhibit_sign (12D) flips inhibitory node outputs before synaptic input.

    Args:
        node_params_flat: 60D node params (shared/tied across layers)
        layer_edge_weights: list of 30D arrays, one per layer
        inputs: 12D external sensor input
        layer_states: list of 12D state arrays
        feedback: 12D feedback from previous timestep's last layer (or None)
        inhibit_sign: 12D array of +1/-1 (Dale's law). None = all excitatory.
    Returns:
        new_layer_states, layer_firings, motor_output (2D: nav, cen)
    """
    n_layers = len(layer_edge_weights)
    node_params = node_params_flat.reshape(12, PARAMS_PER_NODE)
    gains = node_params[:, 0]
    biases = node_params[:, 1]
    input_ws = node_params[:, 2]
    leaks = node_params[:, 3]
    thresholds = node_params[:, 4]

    new_layer_states = []
    layer_firings = []

    for layer_idx in range(n_layers):
        states = layer_states[layer_idx]
        edge_weights = layer_edge_weights[layer_idx]

        # Dale's law: inhibitory nodes flip their output sign
        effective_states = states * inhibit_sign if inhibit_sign is not None else states

        # Layer input: sensor + previous layer + feedback
        if layer_idx == 0:
            layer_input = inputs.copy()
            if feedback is not None:
                layer_input += 0.3 * feedback
        else:
            attenuation = 0.3 ** layer_idx
            layer_input = layer_firings[layer_idx - 1] + attenuation * inputs

        new_states = np.zeros(12, dtype=np.float64)
        for i in range(12):
            syn_input = 0.0
            for j, eidx in NEIGHBORS[i]:
                syn_input += effective_states[j] * edge_weights[eidx]
            total = syn_input + input_ws[i] * layer_input[i] + biases[i]
            x = gains[i] * total
            activation = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
            if activation < thresholds[i]:
                activation *= 0.1
            new_states[i] = activation

        states_out = states * (1 - leaks) + new_states * leaks
        states_out = np.clip(states_out, 0, 1)
        new_layer_states.append(states_out)
        layer_firings.append(states_out.copy())

    motor_nav = 0.0
    motor_cen = 0.0
    for li in range(n_layers):
        motor_nav += layer_firings[li][5]
        motor_cen += layer_firings[li][11]
    motor_nav /= n_layers
    motor_cen /= n_layers

    return new_layer_states, layer_firings, (motor_nav, motor_cen)


def run_hebbian_multilayer(world_factory, n_layers=3, time_budget=120,
                           n_episodes=4, verbose=True, label="ML"):
    """Multi-layer Hebbian with skip connections + feedback + layer-wise lr.

    Fixes over naive multi-layer:
    1. Skip connections: motor output from ALL layers (weighted)
    2. Feedback: last layer output recurs to L0 (context memory)
    3. Layer-wise lr: output layer learns faster (better credit assignment)
    4. Direct sensor to all layers (attenuated)
    """
    base_lr = 0.05
    momentum_coeff = 0.8
    reward_thr = 0.01

    # Layer-wise learning rates: deeper = faster
    layer_lrs = [base_lr * (i + 1) / n_layers for i in range(n_layers)]
    # e.g., 3L: [0.017, 0.033, 0.05]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {label}: {n_layers}L Hebbian+Skip+FB ({n_layers * 12} nodes, "
              f"{n_layers * 30} edges)")
        print(f"  RL: lr={[f'{x:.3f}' for x in layer_lrs]}, "
              f"mom={momentum_coeff}, eps={n_episodes}")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)

    # Node params: 60D (shared across layers)
    node_param_ranges = PARAM_RANGES[30:90]  # 60 params
    base_node_params = np.array([rng.uniform(lo, hi) for lo, hi in node_param_ranges])

    best_score = -float('inf')
    best_node_params = base_node_params.copy()
    n_evals = 0
    scores_log = []
    reach_counts = []

    while time.time() - t0 < time_budget:
        # Perturb node params
        node_params = base_node_params.copy()
        for i in range(60):
            lo, hi = node_param_ranges[i]
            node_params[i] += rng.normal(0, 0.1 * (hi - lo))
            node_params[i] = np.clip(node_params[i], lo, hi)

        # Random edge weights per layer (Hebbian will adapt these)
        layer_edge_weights = [rng.uniform(-3, 3, 30).astype(np.float64)
                              for _ in range(n_layers)]
        layer_momentum = [np.zeros(30, dtype=np.float64)
                          for _ in range(n_layers)]

        ep_scores = []
        ep_reached = 0

        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors = world.reset()
            initial_dist = world.get_food_dist()
            layer_states = [np.zeros(12, dtype=np.float64)
                            for _ in range(n_layers)]
            min_dist = initial_dist
            reached = False
            caught = False
            move_sum = 0.0
            feedback = None  # No feedback on first step

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                # Multi-layer simulation with skip + feedback
                layer_states, layer_firings, (nav, cen) = simulate_step_multilayer(
                    node_params, layer_edge_weights, sensors, layer_states,
                    feedback=feedback)

                # Update feedback for next step
                feedback = layer_firings[-1]

                sensors, dist, reached_now, caught_now = world.step(nav, cen)
                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                # Hebbian update with LAYER-WISE learning rates
                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))
                if abs(reward) > reward_thr:
                    for li in range(n_layers):
                        firing = layer_firings[li]
                        llr = layer_lrs[li]
                        for eidx, (a, b) in enumerate(KATHARA_EDGES):
                            hebbian = firing[a] * firing[b] * reward * llr
                            layer_momentum[li][eidx] = (
                                momentum_coeff * layer_momentum[li][eidx] + hebbian)
                            layer_edge_weights[li][eidx] += layer_momentum[li][eidx]
                            layer_edge_weights[li][eidx] = np.clip(
                                layer_edge_weights[li][eidx], -3.0, 3.0)

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        avg_score = np.mean(ep_scores)
        n_evals += 1

        if avg_score > best_score:
            best_score = avg_score
            best_node_params = node_params.copy()
            base_node_params = node_params.copy()

        scores_log.append(best_score)
        reach_counts.append(ep_reached)

        if verbose and n_evals % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] eval={n_evals}, best={best_score:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    elapsed = time.time() - t0
    avg_reach = np.mean(reach_counts[-20:]) if reach_counts else 0

    if verbose:
        print(f"  {label} done: best={best_score:.2f}, evals={n_evals}, "
              f"reach_rate={avg_reach / n_episodes:.0%}, {elapsed:.0f}s")

    return {
        "mode": label,
        "n_layers": n_layers,
        "best_score": best_score,
        "n_evals": n_evals,
        "elapsed_s": elapsed,
        "avg_reach_rate": avg_reach / n_episodes,
        "scores_log": scores_log[-20:],
    }


def run_layered_development(world_factory, n_layers=3, time_budget=300,
                             n_episodes=4, verbose=True, label="Dev"):
    """Layered development: train layers sequentially like biological brain.

    Phase 1: Train 1L to convergence → freeze L0 edges
    Phase 2: Add L1, train only L1 edges (L0 frozen)
    Phase 3: Add L2, train only L2 edges (L0, L1 frozen)

    L0 = reflex (brainstem). L1 = modulation (limbic). L2 = planning (cortex).
    Motor output = skip connection weighted sum from all developed layers.
    """
    lr = 0.05
    momentum_coeff = 0.8
    reward_thr = 0.01

    # Phase budget: biological development — foundation gets the most time
    # 3L: [60%, 25%, 15%]  2L: [65%, 35%]  1L: [100%]
    if n_layers == 1:
        phase_budgets = [time_budget]
    elif n_layers == 2:
        phase_budgets = [time_budget * 0.65, time_budget * 0.35]
    else:
        # General: exponential decay. Phase 0 gets most.
        raw = [0.6 ** i for i in range(n_layers)]  # [1, 0.6, 0.36, ...]
        total_raw = sum(raw)
        phase_budgets = [time_budget * r / total_raw for r in raw]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {label}: Layered Development ({n_layers}L)")
        budget_str = " + ".join(f"{b:.0f}s" for b in phase_budgets)
        print(f"  Phase budgets: {budget_str} = {time_budget}s")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)

    # Node params: 60D (shared, fine-tuned throughout)
    node_param_ranges = PARAM_RANGES[30:90]
    base_node_params = np.array([rng.uniform(lo, hi) for lo, hi in node_param_ranges])

    # Edge weights per layer, built up progressively
    frozen_edges = []  # list of frozen 30D arrays
    best_node_params = base_node_params.copy()
    best_score = -float('inf')
    total_evals = 0

    for phase in range(n_layers):
        phase_t0 = time.time()
        this_phase_budget = phase_budgets[phase]
        n_frozen = len(frozen_edges)
        n_total_layers = n_frozen + 1  # frozen layers + 1 training layer

        if verbose:
            print(f"\n  === Phase {phase + 1}/{n_layers}: "
                  f"{n_frozen} frozen + 1 training = {n_total_layers}L "
                  f"({this_phase_budget:.0f}s) ===")

        phase_best_score = -float('inf')
        phase_best_node_params = best_node_params.copy()
        phase_best_train_edges = None
        base_np = best_node_params.copy()
        base_train_edges = rng.uniform(-3, 3, 30).astype(np.float64)
        phase_evals = 0

        while time.time() - phase_t0 < this_phase_budget:
            # Perturb node params
            node_params = base_np.copy()
            for i in range(60):
                lo, hi = node_param_ranges[i]
                node_params[i] += rng.normal(0, 0.1 * (hi - lo))
                node_params[i] = np.clip(node_params[i], lo, hi)

            # Training layer: start from best edges so far (not random!)
            # Hebbian will adapt from this starting point
            train_edges = base_train_edges.copy()
            train_momentum = np.zeros(30, dtype=np.float64)

            ep_scores = []
            ep_reached = 0

            for ep in range(n_episodes):
                world = world_factory(seed=ep * 7 + 13)
                sensors = world.reset()
                initial_dist = world.get_food_dist()

                # States for all layers
                all_states = [np.zeros(12, dtype=np.float64)
                              for _ in range(n_total_layers)]
                min_dist = initial_dist
                reached = False
                caught = False
                move_sum = 0.0

                for step in range(60):
                    prev_pos = world.fly_pos.copy()
                    prev_dist = world.get_food_dist()

                    # Build full edge list: frozen + training
                    all_edges = [fe.copy() for fe in frozen_edges] + [train_edges]
                    np_flat = node_params.reshape(-1)

                    # Simulate all layers
                    all_states, all_firings, (nav, cen) = simulate_step_multilayer(
                        np_flat, all_edges, sensors, all_states)

                    sensors, dist, reached_now, caught_now = world.step(nav, cen)
                    move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                    min_dist = min(min_dist, dist)

                    if caught_now:
                        caught = True
                        break

                    # Hebbian: ONLY update training layer (last one)
                    reward = compute_step_reward(
                        prev_dist, dist, reached_now,
                        np.linalg.norm(world.fly_pos - prev_pos))
                    if abs(reward) > reward_thr:
                        firing = all_firings[-1]  # training layer only
                        for eidx, (a, b) in enumerate(KATHARA_EDGES):
                            hebbian = firing[a] * firing[b] * reward * lr
                            train_momentum[eidx] = (
                                momentum_coeff * train_momentum[eidx] + hebbian)
                            train_edges[eidx] += train_momentum[eidx]
                            train_edges[eidx] = np.clip(train_edges[eidx], -3.0, 3.0)

                    if reached_now:
                        reached = True
                        break

                approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
                ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
                if caught:
                    ep_score *= 0.3
                ep_scores.append(ep_score)
                if reached:
                    ep_reached += 1

            avg_score = np.mean(ep_scores)
            phase_evals += 1
            total_evals += 1

            if avg_score > phase_best_score:
                phase_best_score = avg_score
                phase_best_node_params = node_params.copy()
                phase_best_train_edges = train_edges.copy()
                base_np = node_params.copy()
                base_train_edges = train_edges.copy()  # carry forward best edges

            if verbose and phase_evals % 20 == 0:
                elapsed = time.time() - phase_t0
                print(f"  [{elapsed:.0f}s] eval={phase_evals}, "
                      f"best={phase_best_score:.2f}, reach={ep_reached}/{n_episodes}")

        # Phase complete: freeze this layer's best edges
        if phase_best_train_edges is not None:
            frozen_edges.append(phase_best_train_edges)
            best_node_params = phase_best_node_params.copy()
            if phase_best_score > best_score:
                best_score = phase_best_score
        else:
            # Fallback: random edges
            frozen_edges.append(rng.uniform(-3, 3, 30).astype(np.float64))

        if verbose:
            print(f"  Phase {phase + 1} done: {n_total_layers}L best={phase_best_score:.2f}, "
                  f"evals={phase_evals}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  {label} FINAL: {n_layers}L, best={best_score:.2f}, "
              f"total_evals={total_evals}, {elapsed:.0f}s")

    return {
        "mode": label,
        "n_layers": n_layers,
        "best_score": best_score,
        "n_evals": total_evals,
        "elapsed_s": elapsed,
        "avg_reach_rate": 0,  # tracked per phase
        "scores_log": [],
    }


def run_layer_comparison(time_budget=600, verbose=True):
    """1L vs 3L Hebbian (with skip+feedback) on V1 and V3+predator."""
    if verbose:
        print("\n" + "=" * 60)
        print("  Layer Comparison v2: 1L vs 3L (Skip+FB+LayerLR)")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    per_run = time_budget // 4
    results = {}

    # 3 configs per environment: 1L, 3L(simultaneous), 3L(developmental)
    per_run = time_budget // 6
    configs = [
        ("1L_V1", "1l", 1, lambda **kw: FlyWorldV1(**kw)),
        ("3L_V1", "3l", 3, lambda **kw: FlyWorldV1(**kw)),
        ("Dev3_V1", "dev", 3, lambda **kw: FlyWorldV1(**kw)),
        ("1L_V3p", "1l", 1, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
        ("3L_V3p", "3l", 3, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
        ("Dev3_V3p", "dev", 3, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
    ]

    for name, mode, n_layers, factory in configs:
        if verbose:
            print(f"\n  --- {name} ({per_run}s) ---")

        if mode == "1l":
            r = run_mode_b_plus(
                world_factory=factory,
                time_budget=per_run,
                warm_start=None,
                n_episodes=4,
                label=name,
                verbose=verbose,
            )
        elif mode == "dev":
            r = run_layered_development(
                world_factory=factory,
                n_layers=n_layers,
                time_budget=per_run,
                n_episodes=4,
                label=name,
                verbose=verbose,
            )
        else:
            r = run_hebbian_multilayer(
                world_factory=factory,
                n_layers=n_layers,
                time_budget=per_run,
                n_episodes=4,
                label=name,
                verbose=verbose,
            )

        results[name] = {
            "best_score": r["best_score"],
            "n_evals": r["n_evals"],
            "reach_rate": r["avg_reach_rate"],
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  LAYER COMPARISON v2 RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<15} {'Score':>8} {'Reach%':>8} {'Evals':>8}")
        print(f"  {'-' * 39}")
        for name, r in results.items():
            print(f"  {name:<15} {r['best_score']:>8.2f} "
                  f"{r['reach_rate']:>7.0%} {r['n_evals']:>8}")

        # Key comparisons
        for env in ["V1", "V3p"]:
            k1 = f"1L_{env}"
            k3 = f"3L_{env}"
            kd = f"Dev3_{env}"
            scores = {}
            for k in [k1, k3, kd]:
                if k in results:
                    scores[k] = results[k]["best_score"]
            if scores:
                best_k = max(scores, key=scores.get)
                print(f"\n  {env}: " + ", ".join(f"{k}={v:.1f}" for k, v in scores.items()))
                print(f"  -> Winner: {best_k}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__),
                            "kathara_brain_sim_v7_layer_v2_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out_path)}")

    return results


# ============================================================
# Difficulty Progression
# ============================================================
def run_difficulty_progression(time_budget=600, verbose=True):
    """V1 -> V2 -> V3 -> V3+predator.

    V1: random init (v5 warm-start is harmful for Hebbian).
    V2+: full warm-start from previous world's best (Hebbian-found params are compatible).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  Difficulty Progression Experiment")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    per_world = time_budget // 4

    worlds = [
        ("V1_single", lambda **kw: FlyWorldV1(**kw)),
        ("V2_multi3", lambda **kw: FlyWorldV2(n_foods=3, **kw)),
        ("V3_obstacles", lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=False, **kw)),
        ("V3_predator", lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
    ]

    warm_start = None  # V1 starts fresh (random)
    results = {}
    for name, factory in worlds:
        if verbose:
            print(f"\n  --- {name} ({per_world}s) ---")

        r = run_mode_b_plus(
            world_factory=factory,
            time_budget=per_world,
            warm_start=warm_start,
            n_episodes=4,  # 4 episodes for speed (more evals)
            label=f"B+_{name}",
            verbose=verbose,
        )
        results[name] = {
            "best_score": r["best_score"],
            "n_evals": r["n_evals"],
            "reach_rate": r["avg_reach_rate"],
        }
        # Full warm-start chain (Hebbian-found params are compatible)
        warm_start = np.array(r["best_params"])

    # Summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("  DIFFICULTY PROGRESSION RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'World':<20} {'Score':>8} {'Reach%':>8} {'Evals':>8}")
        print(f"  {'-' * 44}")
        for name, r in results.items():
            print(f"  {name:<20} {r['best_score']:>8.2f} "
                  f"{r['reach_rate']:>7.0%} {r['n_evals']:>8}")

        scores = [r["best_score"] for r in results.values()]
        if scores[0] > 0:
            for name, r in results.items():
                pct = r["best_score"] / scores[0] * 100
                print(f"  {name}: {pct:.0f}% of V1")

    return results


# ============================================================
# Main
# ============================================================
# ============================================================
# Scalable Brain: 48N × 6L (vectorized)
# ============================================================

class ScalableBrain:
    """Configurable brain with vectorized numpy simulation.

    v8 additions:
    - inhibit_ratio: fraction of nodes that are inhibitory (Dale's law)
    - asymmetric: if True, edge_weights has 2*n_edges (forward + backward)
    """

    def __init__(self, n_nodes, offsets, inhibit_ratio=0.0, asymmetric=False, rng=None):
        self.n_nodes = n_nodes
        self.asymmetric = asymmetric
        # Build circulant graph
        seen = set()
        self.edges = []
        for i in range(n_nodes):
            for off in offsets:
                j = (i + off) % n_nodes
                key = (min(i, j), max(i, j))
                if i != j and key not in seen:
                    self.edges.append((i, j))
                    seen.add(key)
        self.n_edges = len(self.edges)
        self.adj_i = np.array([e[0] for e in self.edges], dtype=np.int32)
        self.adj_j = np.array([e[1] for e in self.edges], dtype=np.int32)

        # Dale's law: inhibit_ratio of nodes are inhibitory
        self.inhibit_sign = np.ones(n_nodes)
        if inhibit_ratio > 0:
            n_inhib = max(1, int(n_nodes * inhibit_ratio))
            if rng is None:
                rng = np.random.RandomState(42)
            # Designate node 8 as inhibitory (biological: danger/inhibit role)
            # Then fill remaining from candidates (exclude motor nodes 5, 11)
            protected = {5, 11}  # motor nodes must stay excitatory
            if n_nodes >= 12 and 8 < n_nodes:
                self.inhibit_sign[8] = -1.0
                n_inhib -= 1
                protected.add(8)
            if n_inhib > 0:
                candidates = [i for i in range(n_nodes) if i not in protected
                              and self.inhibit_sign[i] > 0]
                n_pick = min(n_inhib, len(candidates))
                if n_pick > 0:
                    chosen = rng.choice(candidates, n_pick, replace=False)
                    self.inhibit_sign[chosen] = -1.0
            n_actual = int((self.inhibit_sign < 0).sum())
            self.inhibit_ratio_actual = n_actual / n_nodes

    def n_edge_params(self):
        """Number of edge weight parameters (2x if asymmetric)."""
        return self.n_edges * 2 if self.asymmetric else self.n_edges

    def weights_to_matrix(self, edge_weights):
        """Edge weight vector → adjacency matrix.

        If asymmetric: edge_weights[:n_edges] = forward (i→j),
                       edge_weights[n_edges:] = backward (j→i)
        """
        M = np.zeros((self.n_nodes, self.n_nodes))
        if self.asymmetric:
            ne = self.n_edges
            M[self.adj_i, self.adj_j] = edge_weights[:ne]
            M[self.adj_j, self.adj_i] = edge_weights[ne:2*ne]
        else:
            M[self.adj_i, self.adj_j] = edge_weights
            M[self.adj_j, self.adj_i] = edge_weights
        return M

    def simulate_layer(self, node_params_2d, edge_matrix, inputs, states):
        """Vectorized 1-layer step with Dale's law inhibition.

        node_params_2d: (n_nodes, 5) - gain, bias, iw, leak, threshold
        edge_matrix: (n_nodes, n_nodes)
        inputs, states: (n_nodes,)
        """
        # Dale's law: inhibitory nodes flip their output sign
        effective_states = states * self.inhibit_sign
        syn = edge_matrix @ effective_states
        total = syn + node_params_2d[:, 2] * inputs + node_params_2d[:, 1]
        x = node_params_2d[:, 0] * total
        act = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        act = np.where(act < node_params_2d[:, 4], act * 0.1, act)
        out = states * (1 - node_params_2d[:, 3]) + act * node_params_2d[:, 3]
        return np.clip(out, 0, 1)

    def hebbian_update_vec(self, edge_weights, firing, reward, lr, momentum, mom_coeff):
        """Vectorized Hebbian update. Asymmetric: backward learns at 0.47x."""
        pre = firing[self.adj_i]
        post = firing[self.adj_j]
        hebb_fw = pre * post * reward * lr
        if self.asymmetric:
            hebb_bw = pre * post * reward * lr * 0.47  # 1/2.13 = FF:FB ratio
            hebb = np.concatenate([hebb_fw, hebb_bw])
        else:
            hebb = hebb_fw
        momentum = mom_coeff * momentum + hebb
        edge_weights = np.clip(edge_weights + momentum, -3.0, 3.0)
        return edge_weights, momentum


# Node param ranges: gain, bias, input_weight, leak, threshold
NODE_PARAM_RANGES_SINGLE = [
    (0.1, 5.0), (-3.0, 3.0), (0.1, 5.0), (0.01, 0.9), (0.01, 1.0)
]


def run_scalable_hebbian(brain, world_factory, n_layers, sensor_nodes, motor_nodes,
                          time_budget=300, n_episodes=4, verbose=True, label="Scale"):
    """Scalable Hebbian brain with developmental training.

    Architecture:
    - Layer 0: receives sensor input (mapped from FlyWorld's 12D)
    - Layers 1-4: processing (attenuated direct sensor input)
    - Layer 5: motor output via skip connections
    - All layers: equal-weight skip connections to motor
    - Hebbian: vectorized per-layer update
    """
    n = brain.n_nodes
    lr_base = 0.05
    mom_coeff = 0.8
    reward_thr = 0.01

    # Phase budgets: foundation gets most time
    raw = [0.6 ** i for i in range(n_layers)]
    total_raw = sum(raw)
    phase_budgets = [time_budget * r / total_raw for r in raw]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {label}: {n}N × {n_layers}L Dev ({n * n_layers} neurons, "
              f"{brain.n_edges}/layer edges)")
        budget_str = " + ".join(f"{b:.0f}s" for b in phase_budgets)
        print(f"  Phases: {budget_str}")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)

    # Node params: (n_nodes, 5) shared across layers
    node_ranges = NODE_PARAM_RANGES_SINGLE * n
    base_node_flat = np.array([rng.uniform(lo, hi) for lo, hi in node_ranges])

    frozen_edge_matrices = []  # precomputed adjacency matrices for frozen layers
    best_node_flat = base_node_flat.copy()
    best_score = -float('inf')
    total_evals = 0
    phase_scores = []

    for phase in range(n_layers):
        phase_t0 = time.time()
        budget = phase_budgets[phase]
        n_frozen = len(frozen_edge_matrices)
        n_total = n_frozen + 1

        # Layer-wise lr: output layer faster
        layer_lrs = [lr_base * (i + 1) / n_total for i in range(n_total)]

        if verbose:
            print(f"\n  === Phase {phase + 1}/{n_layers}: "
                  f"{n_frozen}F + 1T = {n_total}L ({budget:.0f}s) ===")

        phase_best = -float('inf')
        phase_best_np = best_node_flat.copy()
        base_np = best_node_flat.copy()
        base_train_ew = rng.uniform(-3, 3, brain.n_edges).astype(np.float64)
        phase_best_ew = base_train_ew.copy()
        phase_evals = 0

        while time.time() - phase_t0 < budget:
            # Perturb node params
            np_flat = base_np.copy()
            for i in range(n * 5):
                lo, hi = node_ranges[i]
                np_flat[i] += rng.normal(0, 0.1 * (hi - lo))
                np_flat[i] = np.clip(np_flat[i], lo, hi)
            np_2d = np_flat.reshape(n, 5)

            # Training layer edges
            train_ew = base_train_ew.copy()
            train_mom = np.zeros(brain.n_edges)
            layer_moms = [np.zeros(brain.n_edges) for _ in range(n_frozen)]

            ep_scores = []
            ep_reached = 0

            for ep in range(n_episodes):
                world = world_factory(seed=ep * 7 + 13)
                sensors_12 = world.reset()
                initial_dist = world.get_food_dist()

                # Map 12D sensors to nD
                sensors = np.zeros(n)
                sensors[:12] = sensors_12

                # Layer states
                layer_states = [np.zeros(n) for _ in range(n_total)]

                # Build training edge matrix
                train_matrix = brain.weights_to_matrix(train_ew)

                min_dist = initial_dist
                reached = False
                caught = False
                move_sum = 0.0

                for step in range(60):
                    prev_pos = world.fly_pos.copy()
                    prev_dist = world.get_food_dist()

                    # Forward pass through all layers
                    firings = []
                    for li in range(n_total):
                        if li == 0:
                            layer_in = sensors
                        else:
                            layer_in = firings[-1] + (0.3 ** li) * sensors

                        if li < n_frozen:
                            em = frozen_edge_matrices[li]
                        else:
                            em = train_matrix

                        layer_states[li] = brain.simulate_layer(
                            np_2d, em, layer_in, layer_states[li])
                        firings.append(layer_states[li].copy())

                    # Skip-connection motor output (equal weights)
                    nav = sum(f[motor_nodes[0]] for f in firings) / n_total
                    cen = sum(f[motor_nodes[1]] for f in firings) / n_total

                    sensors_12, dist, reached_now, caught_now = world.step(nav, cen)
                    sensors = np.zeros(n)
                    sensors[:12] = sensors_12

                    move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                    min_dist = min(min_dist, dist)

                    if caught_now:
                        caught = True
                        break

                    # Hebbian update
                    reward = compute_step_reward(
                        prev_dist, dist, reached_now,
                        np.linalg.norm(world.fly_pos - prev_pos))
                    if abs(reward) > reward_thr:
                        # Only train the active training layer
                        train_ew, train_mom = brain.hebbian_update_vec(
                            train_ew, firings[-1], reward,
                            layer_lrs[-1], train_mom, mom_coeff)
                        train_matrix = brain.weights_to_matrix(train_ew)

                    if reached_now:
                        reached = True
                        break

                approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
                ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
                if caught:
                    ep_score *= 0.3
                ep_scores.append(ep_score)
                if reached:
                    ep_reached += 1

            avg_score = np.mean(ep_scores)
            phase_evals += 1
            total_evals += 1

            if avg_score > phase_best:
                phase_best = avg_score
                phase_best_np = np_flat.copy()
                phase_best_ew = train_ew.copy()
                base_np = np_flat.copy()
                base_train_ew = train_ew.copy()

            if verbose and phase_evals % 10 == 0:
                el = time.time() - phase_t0
                print(f"  [{el:.0f}s] eval={phase_evals}, best={phase_best:.2f}, "
                      f"reach={ep_reached}/{n_episodes}")

        # Freeze this layer
        frozen_edge_matrices.append(brain.weights_to_matrix(phase_best_ew))
        best_node_flat = phase_best_np.copy()
        if phase_best > best_score:
            best_score = phase_best
        phase_scores.append(phase_best)

        if verbose:
            print(f"  Phase {phase + 1}: {n_total}L best={phase_best:.2f}, "
                  f"evals={phase_evals}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  {label} FINAL: {n_layers}L, best={best_score:.2f}, "
              f"evals={total_evals}, {elapsed:.0f}s")
        print(f"  Phase progression: {' -> '.join(f'{s:.1f}' for s in phase_scores)}")

    return {
        "mode": label,
        "n_nodes": brain.n_nodes,
        "n_layers": n_layers,
        "n_edges_per_layer": brain.n_edges,
        "best_score": best_score,
        "phase_scores": phase_scores,
        "n_evals": total_evals,
        "elapsed_s": elapsed,
    }


def run_scale_comparison(time_budget=900, verbose=True):
    """Compare: 12N×1L (baseline) vs 48N×1L vs 48N×6L.

    Tests whether more nodes and more layers improve performance.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  Scale Comparison: 12N×1L vs 48N×1L vs 48N×6L")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    per_run = time_budget // 6  # 6 configs
    results = {}

    # Build brains
    brain_48 = ScalableBrain(48, [1, 4, 12, 24])

    configs = [
        # (name, use_scalable, n_layers, world_factory)
        ("12N_1L_V1", False, 1, lambda **kw: FlyWorldV1(**kw)),
        ("48N_1L_V1", True, 1, lambda **kw: FlyWorldV1(**kw)),
        ("48N_6L_V1", True, 6, lambda **kw: FlyWorldV1(**kw)),
        ("12N_1L_V3p", False, 1, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
        ("48N_1L_V3p", True, 1, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
        ("48N_6L_V3p", True, 6, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
    ]

    for name, use_scalable, n_layers, factory in configs:
        if verbose:
            print(f"\n  --- {name} ({per_run}s) ---")

        if not use_scalable:
            r = run_mode_b_plus(
                world_factory=factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose)
            results[name] = {"best_score": r["best_score"], "n_evals": r["n_evals"]}
        else:
            r = run_scalable_hebbian(
                brain=brain_48, world_factory=factory, n_layers=n_layers,
                sensor_nodes=[3, 7, 8, 9, 10], motor_nodes=[5, 11],
                time_budget=per_run, n_episodes=4, label=name, verbose=verbose)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "phase_scores": r.get("phase_scores", []),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  SCALE COMPARISON RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<18} {'Score':>8} {'Evals':>8}")
        print(f"  {'-' * 36}")
        for name, r in results.items():
            ps = r.get("phase_scores", [])
            extra = f"  phases: {[f'{s:.0f}' for s in ps]}" if ps else ""
            print(f"  {name:<18} {r['best_score']:>8.2f} {r['n_evals']:>8}{extra}")

        for env in ["V1", "V3p"]:
            k12 = f"12N_1L_{env}"
            k48_1 = f"48N_1L_{env}"
            k48_6 = f"48N_6L_{env}"
            if all(k in results for k in [k12, k48_1, k48_6]):
                s12 = results[k12]["best_score"]
                s48_1 = results[k48_1]["best_score"]
                s48_6 = results[k48_6]["best_score"]
                print(f"\n  {env}:")
                print(f"    48N_1L vs 12N_1L: {s48_1 - s12:+.2f} (more nodes effect)")
                print(f"    48N_6L vs 48N_1L: {s48_6 - s48_1:+.2f} (more layers effect)")
                print(f"    48N_6L vs 12N_1L: {s48_6 - s12:+.2f} (total improvement)")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v7_scale_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")

    return results


# ============================================================
# Parallel Brain: biological development model
# ============================================================
# All layers receive SAME input in parallel.
# Each layer = different "dimension" of processing.
# Motor output = sum of all layers.
# Development: synaptogenesis → Hebbian → pruning
#
#   L0(input) ─┐
#   L1(input) ─┤
#   L2(input) ─┼→ sum → motor
#   L3(input) ─┤
#   L4(input) ─┤
#   L5(input) ─┘

def run_parallel_brain(brain, world_factory, n_layers, sensor_nodes, motor_nodes,
                       time_budget=300, n_episodes=4, verbose=True, label="Parallel"):
    """Parallel multi-layer brain with biological development.

    Key differences from serial model:
    1. ALL layers receive the SAME sensor input (no attenuation)
    2. Motor output = average of ALL layer outputs (true parallel)
    3. Development phases:
       Phase 1 (60%): Synaptogenesis — all layers active, random edges, Hebbian learning
       Phase 2 (25%): Pruning — weak edges removed (below threshold)
       Phase 3 (15%): Refinement — fine-tune surviving edges
    """
    n = brain.n_nodes
    lr_base = 0.05
    mom_coeff = 0.8
    reward_thr = 0.01

    # Development phase budgets (synaptogenesis → pruning → refinement)
    phase_ratios = [0.60, 0.25, 0.15]
    phase_budgets = [time_budget * r for r in phase_ratios]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {label}: {n}N × {n_layers}L PARALLEL ({n * n_layers} neurons, "
              f"{brain.n_edges}/layer edges)")
        print(f"  Dev phases: Synaptogenesis({phase_budgets[0]:.0f}s) → "
              f"Pruning({phase_budgets[1]:.0f}s) → Refinement({phase_budgets[2]:.0f}s)")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)

    # Node params: (n_nodes, 5) shared across all layers
    node_ranges = NODE_PARAM_RANGES_SINGLE * n
    base_node_flat = np.array([rng.uniform(lo, hi) for lo, hi in node_ranges])

    # Each layer gets its own edge weights (independent processing dimensions)
    layer_edge_weights = [rng.uniform(-3, 3, brain.n_edges).astype(np.float64)
                          for _ in range(n_layers)]
    layer_momentums = [np.zeros(brain.n_edges) for _ in range(n_layers)]

    best_node_flat = base_node_flat.copy()
    best_edge_weights = [ew.copy() for ew in layer_edge_weights]
    best_score = -float('inf')
    total_evals = 0
    phase_scores = []
    prune_masks = [np.ones(brain.n_edges, dtype=bool) for _ in range(n_layers)]

    def evaluate(np_flat, edge_ws, masks):
        """Run episodes with parallel brain architecture."""
        np_2d = np_flat.reshape(n, 5)
        # Build edge matrices for all layers (apply masks)
        edge_matrices = []
        for li in range(n_layers):
            ew = edge_ws[li].copy()
            ew[~masks[li]] = 0.0  # pruned edges = 0
            edge_matrices.append(brain.weights_to_matrix(ew))

        ep_scores = []
        ep_reached = 0

        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors_12 = world.reset()
            initial_dist = world.get_food_dist()

            # Map 12D sensors to nD
            sensors = np.zeros(n)
            sensors[:12] = sensors_12

            # Each layer has its own state (independent dimensions)
            layer_states = [np.zeros(n) for _ in range(n_layers)]
            min_dist = initial_dist
            reached = False
            caught = False
            move_sum = 0.0

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                # PARALLEL: all layers receive SAME input
                firings = []
                for li in range(n_layers):
                    layer_states[li] = brain.simulate_layer(
                        np_2d, edge_matrices[li], sensors, layer_states[li])
                    firings.append(layer_states[li].copy())

                # Motor = AVERAGE of all layers (parallel sum)
                nav = sum(f[motor_nodes[0]] for f in firings) / n_layers
                cen = sum(f[motor_nodes[1]] for f in firings) / n_layers

                sensors_12, dist, reached_now, caught_now = world.step(nav, cen)
                sensors = np.zeros(n)
                sensors[:12] = sensors_12

                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                # Hebbian: ALL layers learn in parallel (independent)
                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))
                if abs(reward) > reward_thr:
                    for li in range(n_layers):
                        edge_ws[li], layer_momentums[li] = brain.hebbian_update_vec(
                            edge_ws[li], firings[li], reward,
                            lr_base, layer_momentums[li], mom_coeff)
                        # Re-apply mask
                        ew_masked = edge_ws[li].copy()
                        ew_masked[~masks[li]] = 0.0
                        edge_matrices[li] = brain.weights_to_matrix(ew_masked)

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        return np.mean(ep_scores), ep_reached

    # =============================================
    # Phase 1: Synaptogenesis (all layers, all edges, learn)
    # =============================================
    phase_t0 = time.time()
    phase1_best = -float('inf')
    phase1_evals = 0
    base_np = base_node_flat.copy()

    if verbose:
        print(f"\n  === Phase 1: Synaptogenesis ({phase_budgets[0]:.0f}s) ===")
        print(f"    All {n_layers} layers active, {brain.n_edges} edges each, Hebbian ON")

    while time.time() - phase_t0 < phase_budgets[0]:
        # Perturb node params
        np_trial = base_np.copy()
        for i in range(n * 5):
            lo, hi = node_ranges[i]
            np_trial[i] += rng.normal(0, 0.1 * (hi - lo))
            np_trial[i] = np.clip(np_trial[i], lo, hi)

        # Reset momentums for trial
        for li in range(n_layers):
            layer_momentums[li] = np.zeros(brain.n_edges)

        trial_ews = [ew.copy() for ew in layer_edge_weights]
        score, ep_reached = evaluate(np_trial, trial_ews, prune_masks)
        phase1_evals += 1
        total_evals += 1

        if score > phase1_best:
            phase1_best = score
            best_node_flat = np_trial.copy()
            best_edge_weights = [ew.copy() for ew in trial_ews]
            layer_edge_weights = [ew.copy() for ew in trial_ews]
            base_np = np_trial.copy()

        if verbose and phase1_evals % 10 == 0:
            el = time.time() - phase_t0
            print(f"    [{el:.0f}s] eval={phase1_evals}, best={phase1_best:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    phase_scores.append(phase1_best)
    if verbose:
        print(f"    Phase 1 done: best={phase1_best:.2f}, evals={phase1_evals}")

    # =============================================
    # Phase 2: Pruning (remove weak edges, re-learn)
    # =============================================
    phase_t0 = time.time()

    # Prune: remove edges where |weight| < threshold in ALL layers
    prune_threshold = 0.3
    active_edges_before = [m.sum() for m in prune_masks]
    for li in range(n_layers):
        prune_masks[li] = np.abs(best_edge_weights[li]) >= prune_threshold

    active_edges_after = [m.sum() for m in prune_masks]

    if verbose:
        print(f"\n  === Phase 2: Pruning ({phase_budgets[1]:.0f}s) ===")
        for li in range(n_layers):
            print(f"    L{li}: {active_edges_before[li]} → {active_edges_after[li]} edges "
                  f"({active_edges_after[li] / active_edges_before[li] * 100:.0f}% survive)")

    phase2_best = phase1_best
    phase2_evals = 0
    base_np = best_node_flat.copy()
    layer_edge_weights = [ew.copy() for ew in best_edge_weights]

    while time.time() - phase_t0 < phase_budgets[1]:
        np_trial = base_np.copy()
        for i in range(n * 5):
            lo, hi = node_ranges[i]
            np_trial[i] += rng.normal(0, 0.05 * (hi - lo))  # smaller perturbation
            np_trial[i] = np.clip(np_trial[i], lo, hi)

        for li in range(n_layers):
            layer_momentums[li] = np.zeros(brain.n_edges)

        trial_ews = [ew.copy() for ew in layer_edge_weights]
        score, ep_reached = evaluate(np_trial, trial_ews, prune_masks)
        phase2_evals += 1
        total_evals += 1

        if score > phase2_best:
            phase2_best = score
            best_node_flat = np_trial.copy()
            best_edge_weights = [ew.copy() for ew in trial_ews]
            layer_edge_weights = [ew.copy() for ew in trial_ews]
            base_np = np_trial.copy()

        if verbose and phase2_evals % 10 == 0:
            el = time.time() - phase_t0
            print(f"    [{el:.0f}s] eval={phase2_evals}, best={phase2_best:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    phase_scores.append(phase2_best)
    if verbose:
        print(f"    Phase 2 done: best={phase2_best:.2f}, evals={phase2_evals}")

    # =============================================
    # Phase 3: Refinement (fine-tune surviving edges)
    # =============================================
    phase_t0 = time.time()
    phase3_best = phase2_best
    phase3_evals = 0

    if verbose:
        print(f"\n  === Phase 3: Refinement ({phase_budgets[2]:.0f}s) ===")

    while time.time() - phase_t0 < phase_budgets[2]:
        np_trial = base_np.copy()
        for i in range(n * 5):
            lo, hi = node_ranges[i]
            np_trial[i] += rng.normal(0, 0.02 * (hi - lo))  # even smaller
            np_trial[i] = np.clip(np_trial[i], lo, hi)

        for li in range(n_layers):
            layer_momentums[li] = np.zeros(brain.n_edges)

        trial_ews = [ew.copy() for ew in layer_edge_weights]
        score, ep_reached = evaluate(np_trial, trial_ews, prune_masks)
        phase3_evals += 1
        total_evals += 1

        if score > phase3_best:
            phase3_best = score
            best_node_flat = np_trial.copy()
            best_edge_weights = [ew.copy() for ew in trial_ews]
            layer_edge_weights = [ew.copy() for ew in trial_ews]
            base_np = np_trial.copy()

        if verbose and phase3_evals % 10 == 0:
            el = time.time() - phase_t0
            print(f"    [{el:.0f}s] eval={phase3_evals}, best={phase3_best:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    phase_scores.append(phase3_best)
    best_score = max(phase_scores)

    # Edge survival stats
    total_possible = brain.n_edges * n_layers
    total_surviving = sum(m.sum() for m in prune_masks)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n    Phase 3 done: best={phase3_best:.2f}, evals={phase3_evals}")
        print(f"\n  {label} FINAL: {n_layers}L parallel, best={best_score:.2f}, "
              f"evals={total_evals}, {elapsed:.0f}s")
        print(f"  Dev phases: {' → '.join(f'{s:.1f}' for s in phase_scores)}")
        print(f"  Edge survival: {total_surviving}/{total_possible} "
              f"({total_surviving / total_possible * 100:.0f}%)")

    return {
        "mode": label,
        "n_nodes": brain.n_nodes,
        "n_layers": n_layers,
        "architecture": "parallel",
        "n_edges_per_layer": brain.n_edges,
        "best_score": best_score,
        "phase_scores": phase_scores,
        "phase_names": ["synaptogenesis", "pruning", "refinement"],
        "edge_survival": [int(m.sum()) for m in prune_masks],
        "n_evals": total_evals,
        "elapsed_s": elapsed,
    }


# ============================================================
# Fetal Brain: 4-stage biological development
# ============================================================
# Phase 0 (Fetal): Over-generate 2x → spontaneous firing → apoptosis
# Phase 1 (Infant): Synaptogenesis — Hebbian on surviving structure
# Phase 2 (Child): Pruning — remove weak edges
# Phase 3 (Adult): Refinement — fine-tune

def fetal_phase(brain_2x, target_n, n_layers, rng, time_budget=30, verbose=True,
                inhibit_ratio=0.0, inhib_placement="hub"):
    """Phase 0: Fetal development.

    1. Start with 2x nodes (over-generation)
    2. Random node params
    3. Spontaneous firing WITHOUT input (circuit self-test)
    4. Measure node activity (how much each node fires)
    5. Kill bottom 50% nodes (apoptosis)
    6. Assign inhibit_sign to survivors (Dale's law)
    7. Return surviving node indices + their edge matrices + inhibit_sign

    inhib_placement: "hub" = highest-degree nodes (biological),
                     "low_activity" = lowest-activity nodes (v8 original)
    """
    n_2x = brain_2x.n_nodes

    if verbose:
        print(f"\n  === Phase 0: Fetal Development ({time_budget:.0f}s) ===")
        print(f"    {n_2x} nodes generated (2x over-production)")

    # Random node params for the oversized brain
    node_ranges = NODE_PARAM_RANGES_SINGLE * n_2x
    node_params = np.array([rng.uniform(lo, hi) for lo, hi in node_ranges])
    np_2d = node_params.reshape(n_2x, 5)

    # Per-layer random edges
    layer_ews = [rng.uniform(-3, 3, brain_2x.n_edges).astype(np.float64)
                 for _ in range(n_layers)]
    layer_matrices = [brain_2x.weights_to_matrix(ew) for ew in layer_ews]

    # Spontaneous firing test: NO external input
    # Run many trials with different random initial states
    node_activity = np.zeros(n_2x)  # cumulative firing per node
    n_trials = 50
    n_steps = 30

    t0 = time.time()
    trial = 0
    while time.time() - t0 < time_budget and trial < n_trials:
        # Random initial state (spontaneous activity)
        states = [rng.uniform(0, 1, n_2x) for _ in range(n_layers)]
        zero_input = np.zeros(n_2x)

        for step in range(n_steps):
            for li in range(n_layers):
                states[li] = brain_2x.simulate_layer(
                    np_2d, layer_matrices[li], zero_input, states[li])
                # Track sustained activity
                node_activity += states[li]

        trial += 1

    # Normalize
    node_activity /= (trial * n_steps * n_layers)

    # Apoptosis: kill bottom 50% (nodes that couldn't sustain activity)
    sorted_indices = np.argsort(node_activity)[::-1]  # highest activity first
    survivors = sorted(sorted_indices[:target_n])  # keep top N
    dead = sorted(sorted_indices[target_n:])

    if verbose:
        print(f"    Spontaneous firing: {trial} trials × {n_steps} steps")
        print(f"    Activity range: [{node_activity.min():.3f}, {node_activity.max():.3f}]")
        print(f"    Survivor threshold: {node_activity[sorted_indices[target_n - 1]]:.3f}")
        print(f"    Apoptosis: {len(dead)} nodes killed, {len(survivors)} survive")
        # Show top 5 most active
        top5 = sorted_indices[:5]
        top5_act = [f"N{i}={node_activity[i]:.3f}" for i in top5]
        print(f"    Most active: {', '.join(top5_act)}")

    # Extract surviving sub-network
    # Map old indices → new indices
    old_to_new = {old: new for new, old in enumerate(survivors)}

    # Build new smaller brain
    surviving_edges = []
    for e_idx, (i, j) in enumerate(brain_2x.edges):
        if i in old_to_new and j in old_to_new:
            surviving_edges.append((old_to_new[i], old_to_new[j], e_idx))

    # Extract surviving node params
    surviving_np = node_params.reshape(n_2x, 5)[survivors].flatten()

    # Extract surviving edge weights for each layer
    surviving_ews = []
    for li in range(n_layers):
        ew_new = np.array([layer_ews[li][e_idx] for _, _, e_idx in surviving_edges])
        surviving_ews.append(ew_new)

    # v8: Assign inhibit_sign to survivors (Dale's law)
    survivor_inhibit = np.ones(target_n)
    survivor_edges_list = [(i, j) for i, j, _ in surviving_edges]
    if inhibit_ratio > 0:
        n_inhib = max(1, int(target_n * inhibit_ratio))
        # Protected nodes: motor positions
        motor_a = min(target_n - 1, max(0, int(target_n * 5 / 48)))
        motor_b = min(target_n - 1, max(0, int(target_n * 11 / 48)))
        motor_positions = {motor_a, motor_b}
        # Also protect first 12 sensor nodes
        sensor_positions = set(range(min(12, target_n)))

        degree = np.zeros(target_n)
        for (i, j) in survivor_edges_list:
            degree[i] += 1
            degree[j] += 1

        if inhib_placement == "hub":
            # Hub-based: highest-degree nodes
            sorted_by_degree = np.argsort(degree)[::-1]
            picked = 0
            for idx in sorted_by_degree:
                if picked >= n_inhib:
                    break
                if idx not in motor_positions and idx not in sensor_positions:
                    survivor_inhibit[idx] = -1.0
                    picked += 1
            if picked < n_inhib:
                for idx in sorted_by_degree:
                    if picked >= n_inhib:
                        break
                    if idx not in motor_positions and survivor_inhibit[idx] > 0:
                        survivor_inhibit[idx] = -1.0
                        picked += 1
        elif inhib_placement == "mid":
            # Mid-degree: interneurons (not hub, not leaf, not sensor/motor)
            # Sort by |degree - median_degree| → closest to median first
            median_deg = np.median(degree)
            dist_to_median = np.abs(degree - median_deg)
            sorted_by_mid = np.argsort(dist_to_median)  # closest to median first
            picked = 0
            for idx in sorted_by_mid:
                if picked >= n_inhib:
                    break
                if idx not in motor_positions and idx not in sensor_positions:
                    survivor_inhibit[idx] = -1.0
                    picked += 1
            if picked < n_inhib:
                for idx in sorted_by_mid:
                    if picked >= n_inhib:
                        break
                    if idx not in motor_positions and survivor_inhibit[idx] > 0:
                        survivor_inhibit[idx] = -1.0
                        picked += 1
        else:
            # Low-activity: original v8 method
            survivor_activities = node_activity[survivors]
            sorted_by_act = np.argsort(survivor_activities)
            picked = 0
            for idx in sorted_by_act:
                if picked >= n_inhib:
                    break
                if idx not in motor_positions:
                    survivor_inhibit[idx] = -1.0
                    picked += 1

        n_actual = int((survivor_inhibit < 0).sum())
        if verbose:
            inhib_nodes = np.where(survivor_inhibit < 0)[0]
            if inhib_placement == "hub":
                degree = np.zeros(target_n)
                for (i, j) in survivor_edges_list:
                    degree[i] += 1
                    degree[j] += 1
                avg_deg_inhib = np.mean(degree[inhib_nodes]) if len(inhib_nodes) > 0 else 0
                avg_deg_all = np.mean(degree)
                print(f"    Inhibitory: {n_actual}/{target_n} ({n_actual/target_n*100:.0f}%) "
                      f"[{inhib_placement}] avg_degree={avg_deg_inhib:.1f} vs all={avg_deg_all:.1f}")
            else:
                print(f"    Inhibitory: {n_actual}/{target_n} ({n_actual/target_n*100:.0f}%) "
                      f"[{inhib_placement}]")

    if verbose:
        print(f"    Surviving edges: {len(surviving_edges)} "
              f"(from {brain_2x.n_edges})")
        el = time.time() - t0
        print(f"    Fetal phase: {el:.1f}s")

    return {
        "survivors": survivors,
        "n_nodes": target_n,
        "edges": [(i, j) for i, j, _ in surviving_edges],
        "node_params": surviving_np,
        "edge_weights": surviving_ews,
        "node_activity": node_activity,
        "inhibit_sign": survivor_inhibit,
    }


class SurvivorBrain:
    """Brain built from fetal phase survivors. Arbitrary topology.

    v8: inhibit_sign and asymmetric support.
    """

    def __init__(self, n_nodes, edges, inhibit_sign=None, asymmetric=False):
        self.n_nodes = n_nodes
        self.edges = edges
        self.n_edges = len(edges)
        self.asymmetric = asymmetric
        self.adj_i = np.array([e[0] for e in edges], dtype=np.int32)
        self.adj_j = np.array([e[1] for e in edges], dtype=np.int32)
        # Dale's law: inherit inhibit_sign from fetal phase
        if inhibit_sign is not None:
            self.inhibit_sign = inhibit_sign
        else:
            self.inhibit_sign = np.ones(n_nodes)

    def n_edge_params(self):
        return self.n_edges * 2 if self.asymmetric else self.n_edges

    def weights_to_matrix(self, edge_weights):
        M = np.zeros((self.n_nodes, self.n_nodes))
        if len(edge_weights) > 0:
            if self.asymmetric:
                ne = self.n_edges
                M[self.adj_i, self.adj_j] = edge_weights[:ne]
                M[self.adj_j, self.adj_i] = edge_weights[ne:2*ne]
            else:
                M[self.adj_i, self.adj_j] = edge_weights
                M[self.adj_j, self.adj_i] = edge_weights
        return M

    def simulate_layer(self, node_params_2d, edge_matrix, inputs, states):
        effective_states = states * self.inhibit_sign
        syn = edge_matrix @ effective_states
        total = syn + node_params_2d[:, 2] * inputs + node_params_2d[:, 1]
        x = node_params_2d[:, 0] * total
        act = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        act = np.where(act < node_params_2d[:, 4], act * 0.1, act)
        out = states * (1 - node_params_2d[:, 3]) + act * node_params_2d[:, 3]
        return np.clip(out, 0, 1)

    def hebbian_update_vec(self, edge_weights, firing, reward, lr, momentum, mom_coeff):
        if len(edge_weights) == 0:
            return edge_weights, momentum
        pre = firing[self.adj_i]
        post = firing[self.adj_j]
        hebb_fw = pre * post * reward * lr
        if self.asymmetric:
            hebb_bw = pre * post * reward * lr * 0.47
            hebb = np.concatenate([hebb_fw, hebb_bw])
        else:
            hebb = hebb_fw
        momentum = mom_coeff * momentum + hebb
        edge_weights = np.clip(edge_weights + momentum, -3.0, 3.0)
        return edge_weights, momentum


def run_fetal_brain(world_factory, n_target=48, n_layers=6,
                    time_budget=300, n_episodes=4, verbose=True, label="Fetal"):
    """Full 4-stage biological brain development.

    Phase 0: Fetal (over-generate 2x → spontaneous fire → apoptosis)
    Phase 1: Infant (Hebbian learning on survivor brain)
    Phase 2: Child (pruning weak edges)
    Phase 3: Adult (refinement)
    """
    n_2x = n_target * 2  # over-generate
    offsets_2x = [1, 4, 12, 24, 48]  # wider connectivity for larger brain
    brain_2x = ScalableBrain(n_2x, offsets_2x)

    # Budget allocation
    fetal_budget = time_budget * 0.10  # 10% for fetal
    infant_budget = time_budget * 0.55  # 55% for Hebbian
    child_budget = time_budget * 0.20  # 20% for pruning
    adult_budget = time_budget * 0.15  # 15% for refinement

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  {label}: 4-Stage Biological Development")
        print(f"  Start: {n_2x}N → Apoptosis → {n_target}N × {n_layers}L")
        print(f"  Budget: Fetal({fetal_budget:.0f}s) → Infant({infant_budget:.0f}s) "
              f"→ Child({child_budget:.0f}s) → Adult({adult_budget:.0f}s)")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)
    lr_base = 0.05
    mom_coeff = 0.8
    reward_thr = 0.01

    # ===================
    # Phase 0: Fetal
    # ===================
    fetal = fetal_phase(brain_2x, n_target, n_layers, rng,
                        time_budget=fetal_budget, verbose=verbose)

    # Build survivor brain
    brain = SurvivorBrain(fetal["n_nodes"], fetal["edges"])
    node_flat = fetal["node_params"].copy()
    node_ranges = NODE_PARAM_RANGES_SINGLE * fetal["n_nodes"]
    layer_ews = [ew.copy() for ew in fetal["edge_weights"]]
    layer_moms = [np.zeros(brain.n_edges) for _ in range(n_layers)]

    # Sensor/motor mapping: first 12 surviving nodes get sensor input
    # Use motor from nodes closest to original positions 5 and 11
    n = fetal["n_nodes"]

    best_node_flat = node_flat.copy()
    best_ews = [ew.copy() for ew in layer_ews]
    best_score = -float('inf')
    total_evals = 0
    phase_scores = []
    prune_masks = [np.ones(brain.n_edges, dtype=bool) for _ in range(n_layers)]

    # Motor node selection: use node indices proportional to original 5/48 and 11/48
    motor_a = min(n - 1, max(0, int(n * 5 / 48)))   # navigation
    motor_b = min(n - 1, max(0, int(n * 11 / 48)))   # centering
    if motor_a == motor_b:
        motor_b = min(n - 1, motor_a + 1)

    if verbose:
        print(f"    Motor nodes: nav=N{motor_a}, cen=N{motor_b}")
        print(f"    Sensor input: first 12 of {n} nodes")

    def run_eval(np_flat, ews, masks):
        np_2d = np_flat.reshape(n, 5)
        edge_matrices = []
        for li in range(n_layers):
            ew = ews[li].copy()
            if len(ew) > 0:
                ew[~masks[li]] = 0.0
            edge_matrices.append(brain.weights_to_matrix(ew))

        ep_scores = []
        ep_reached = 0

        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors_12 = world.reset()
            initial_dist = world.get_food_dist()

            sensors = np.zeros(n)
            sensors[:min(12, n)] = sensors_12[:min(12, n)]

            layer_states = [np.zeros(n) for _ in range(n_layers)]
            min_dist = initial_dist
            reached = False
            caught = False
            move_sum = 0.0

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                # Parallel: all layers get same input
                firings = []
                for li in range(n_layers):
                    layer_states[li] = brain.simulate_layer(
                        np_2d, edge_matrices[li], sensors, layer_states[li])
                    firings.append(layer_states[li].copy())

                nav = sum(f[motor_a] for f in firings) / n_layers
                cen = sum(f[motor_b] for f in firings) / n_layers

                sensors_12, dist, reached_now, caught_now = world.step(nav, cen)
                sensors = np.zeros(n)
                sensors[:min(12, n)] = sensors_12[:min(12, n)]

                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))
                if abs(reward) > reward_thr:
                    for li in range(n_layers):
                        ews[li], layer_moms[li] = brain.hebbian_update_vec(
                            ews[li], firings[li], reward,
                            lr_base, layer_moms[li], mom_coeff)
                        ew_m = ews[li].copy()
                        if len(ew_m) > 0:
                            ew_m[~masks[li]] = 0.0
                        edge_matrices[li] = brain.weights_to_matrix(ew_m)

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        return np.mean(ep_scores), ep_reached

    # ===================
    # Phase 1: Infant (Hebbian)
    # ===================
    if verbose:
        print(f"\n  === Phase 1: Infant - Hebbian Learning ({infant_budget:.0f}s) ===")

    phase_t0 = time.time()
    base_np = node_flat.copy()
    p1_best = -float('inf')
    p1_evals = 0

    while time.time() - phase_t0 < infant_budget:
        np_trial = base_np.copy()
        for i in range(n * 5):
            lo, hi = node_ranges[i]
            np_trial[i] += rng.normal(0, 0.1 * (hi - lo))
            np_trial[i] = np.clip(np_trial[i], lo, hi)

        for li in range(n_layers):
            layer_moms[li] = np.zeros(brain.n_edges)

        trial_ews = [ew.copy() for ew in layer_ews]
        score, ep_reached = run_eval(np_trial, trial_ews, prune_masks)
        p1_evals += 1
        total_evals += 1

        if score > p1_best:
            p1_best = score
            best_node_flat = np_trial.copy()
            best_ews = [ew.copy() for ew in trial_ews]
            layer_ews = [ew.copy() for ew in trial_ews]
            base_np = np_trial.copy()

        if verbose and p1_evals % 10 == 0:
            el = time.time() - phase_t0
            print(f"    [{el:.0f}s] eval={p1_evals}, best={p1_best:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    phase_scores.append(p1_best)
    if verbose:
        print(f"    Phase 1 done: best={p1_best:.2f}, evals={p1_evals}")

    # ===================
    # Phase 2: Child (Pruning)
    # ===================
    if verbose:
        print(f"\n  === Phase 2: Child - Pruning ({child_budget:.0f}s) ===")

    prune_threshold = 0.3
    for li in range(n_layers):
        if len(best_ews[li]) > 0:
            before = prune_masks[li].sum()
            prune_masks[li] = np.abs(best_ews[li]) >= prune_threshold
            after = prune_masks[li].sum()
            if verbose:
                print(f"    L{li}: {before} → {after} edges "
                      f"({after / max(before, 1) * 100:.0f}% survive)")

    phase_t0 = time.time()
    p2_best = p1_best
    p2_evals = 0
    base_np = best_node_flat.copy()
    layer_ews = [ew.copy() for ew in best_ews]

    while time.time() - phase_t0 < child_budget:
        np_trial = base_np.copy()
        for i in range(n * 5):
            lo, hi = node_ranges[i]
            np_trial[i] += rng.normal(0, 0.05 * (hi - lo))
            np_trial[i] = np.clip(np_trial[i], lo, hi)

        for li in range(n_layers):
            layer_moms[li] = np.zeros(brain.n_edges)

        trial_ews = [ew.copy() for ew in layer_ews]
        score, ep_reached = run_eval(np_trial, trial_ews, prune_masks)
        p2_evals += 1
        total_evals += 1

        if score > p2_best:
            p2_best = score
            best_node_flat = np_trial.copy()
            best_ews = [ew.copy() for ew in trial_ews]
            layer_ews = [ew.copy() for ew in trial_ews]
            base_np = np_trial.copy()

        if verbose and p2_evals % 10 == 0:
            el = time.time() - phase_t0
            print(f"    [{el:.0f}s] eval={p2_evals}, best={p2_best:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    phase_scores.append(p2_best)
    if verbose:
        print(f"    Phase 2 done: best={p2_best:.2f}, evals={p2_evals}")

    # ===================
    # Phase 3: Adult (Refinement)
    # ===================
    if verbose:
        print(f"\n  === Phase 3: Adult - Refinement ({adult_budget:.0f}s) ===")

    phase_t0 = time.time()
    p3_best = p2_best
    p3_evals = 0

    while time.time() - phase_t0 < adult_budget:
        np_trial = base_np.copy()
        for i in range(n * 5):
            lo, hi = node_ranges[i]
            np_trial[i] += rng.normal(0, 0.02 * (hi - lo))
            np_trial[i] = np.clip(np_trial[i], lo, hi)

        for li in range(n_layers):
            layer_moms[li] = np.zeros(brain.n_edges)

        trial_ews = [ew.copy() for ew in layer_ews]
        score, ep_reached = run_eval(np_trial, trial_ews, prune_masks)
        p3_evals += 1
        total_evals += 1

        if score > p3_best:
            p3_best = score
            best_node_flat = np_trial.copy()
            best_ews = [ew.copy() for ew in trial_ews]
            layer_ews = [ew.copy() for ew in trial_ews]
            base_np = np_trial.copy()

        if verbose and p3_evals % 10 == 0:
            el = time.time() - phase_t0
            print(f"    [{el:.0f}s] eval={p3_evals}, best={p3_best:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    phase_scores.append(p3_best)
    best_score = max(phase_scores)

    total_possible = brain.n_edges * n_layers
    total_surviving = sum(m.sum() for m in prune_masks)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n    Phase 3 done: best={p3_best:.2f}, evals={p3_evals}")
        print(f"\n  {label} FINAL: {n}N(from {n_2x}) × {n_layers}L, "
              f"best={best_score:.2f}, evals={total_evals}, {elapsed:.0f}s")
        print(f"  Development: {' → '.join(f'{s:.1f}' for s in phase_scores)}")
        if total_possible > 0:
            print(f"  Edge survival: {total_surviving}/{total_possible} "
                  f"({total_surviving / total_possible * 100:.0f}%)")

    return {
        "mode": label,
        "n_nodes_initial": n_2x,
        "n_nodes_final": n,
        "n_layers": n_layers,
        "architecture": "fetal_parallel",
        "n_edges": brain.n_edges,
        "best_score": best_score,
        "phase_scores": phase_scores,
        "phase_names": ["infant_hebbian", "child_pruning", "adult_refinement"],
        "edge_survival": [int(m.sum()) for m in prune_masks],
        "n_evals": total_evals,
        "elapsed_s": elapsed,
    }


def run_fetal_comparison(time_budget=600, verbose=True):
    """Compare: 12N×1L vs 48N×6L(fetal) on V1 and V3p."""
    if verbose:
        print("\n" + "=" * 60)
        print("  Fetal Brain vs Baseline Comparison")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    per_run = time_budget // 4
    results = {}

    configs = [
        ("12N_1L_V1", False, lambda **kw: FlyWorldV1(**kw)),
        ("Fetal_48N_6L_V1", True, lambda **kw: FlyWorldV1(**kw)),
        ("12N_1L_V3p", False, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
        ("Fetal_48N_6L_V3p", True, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
    ]

    for name, use_fetal, factory in configs:
        if verbose:
            print(f"\n  --- {name} ({per_run}s) ---")

        if not use_fetal:
            r = run_mode_b_plus(
                world_factory=factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose)
            results[name] = {"best_score": r["best_score"], "n_evals": r["n_evals"]}
        else:
            r = run_fetal_brain(
                world_factory=factory, n_target=48, n_layers=6,
                time_budget=per_run, n_episodes=4, label=name, verbose=verbose)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "phase_scores": r.get("phase_scores", []),
                "n_nodes_final": r.get("n_nodes_final"),
                "n_edges": r.get("n_edges"),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  FETAL BRAIN COMPARISON RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<22} {'Score':>8} {'Evals':>8}")
        print(f"  {'-' * 40}")
        for name, r in results.items():
            print(f"  {name:<22} {r['best_score']:>8.2f} {r['n_evals']:>8}")

        print(f"\n  Reference (previous results):")
        print(f"    12N_1L_V1:           94.97")
        print(f"    48N_6L_serial_V1:    97.08")
        print(f"    48N_6L_parallel_V1:  84.81")
        print(f"    12N_1L_V3p:          94.10")
        print(f"    48N_6L_serial_V3p:   84.78")
        print(f"    48N_6L_parallel_V3p: 58.50")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v7_fetal_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")

    return results


# ============================================================
# Deep dive: 3 hypotheses
# ============================================================

def run_fetal_brain_v2(world_factory, n_target=48, n_layers=6,
                       time_budget=300, n_episodes=4, verbose=True, label="FetalV2",
                       sensor_broadcast=False, guided_apoptosis=False):
    """Fetal brain with optional improvements.

    sensor_broadcast: ALL nodes receive sensor input (repeated across nodes)
    guided_apoptosis: Bias survival toward sensor/motor-connected nodes
    """
    n_2x = n_target * 2
    offsets_2x = [1, 4, 12, 24, 48]
    brain_2x = ScalableBrain(n_2x, offsets_2x)

    fetal_budget = time_budget * 0.10
    infant_budget = time_budget * 0.55
    child_budget = time_budget * 0.20
    adult_budget = time_budget * 0.15

    if verbose:
        flags = []
        if sensor_broadcast:
            flags.append("SensorBroadcast")
        if guided_apoptosis:
            flags.append("GuidedApoptosis")
        flag_str = " + ".join(flags) if flags else "Baseline"
        print(f"\n{'=' * 60}")
        print(f"  {label}: Fetal Brain [{flag_str}]")
        print(f"  {n_2x}N → {n_target}N × {n_layers}L")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)
    lr_base = 0.05
    mom_coeff = 0.8
    reward_thr = 0.01

    # === Phase 0: Fetal ===
    node_ranges_2x = NODE_PARAM_RANGES_SINGLE * n_2x
    node_params_2x = np.array([rng.uniform(lo, hi) for lo, hi in node_ranges_2x])
    np_2d_2x = node_params_2x.reshape(n_2x, 5)

    layer_ews_2x = [rng.uniform(-3, 3, brain_2x.n_edges).astype(np.float64)
                    for _ in range(n_layers)]
    layer_matrices_2x = [brain_2x.weights_to_matrix(ew) for ew in layer_ews_2x]

    # Spontaneous firing
    node_activity = np.zeros(n_2x)
    n_trials = 50
    n_steps = 30
    ft0 = time.time()
    for trial in range(n_trials):
        if time.time() - ft0 > fetal_budget:
            break
        states = [rng.uniform(0, 1, n_2x) for _ in range(n_layers)]
        for step in range(n_steps):
            for li in range(n_layers):
                states[li] = brain_2x.simulate_layer(
                    np_2d_2x, layer_matrices_2x[li], np.zeros(n_2x), states[li])
                node_activity += states[li]

    node_activity /= max(1, (min(n_trials, int((time.time()-ft0)/fetal_budget*n_trials+1)))
                         * n_steps * n_layers)

    # Survival scoring
    survival_score = node_activity.copy()

    if guided_apoptosis:
        # Boost nodes near sensor positions (0-11) and motor positions (5, 11)
        # Use graph distance from sensor/motor nodes
        sensor_motor_nodes = list(range(12)) + [5, 11]  # 0-11 sensor, 5/11 motor
        # For each node, compute min graph distance to any sensor/motor node
        # Use BFS on the 2x brain
        adj_list = {i: set() for i in range(n_2x)}
        for (i, j) in brain_2x.edges:
            adj_list[i].add(j)
            adj_list[j].add(i)

        proximity_bonus = np.zeros(n_2x)
        for target in sensor_motor_nodes:
            if target >= n_2x:
                continue
            # BFS from target
            dist = np.full(n_2x, n_2x)
            dist[target] = 0
            queue = [target]
            qi = 0
            while qi < len(queue):
                curr = queue[qi]
                qi += 1
                for nb in adj_list[curr]:
                    if dist[nb] > dist[curr] + 1:
                        dist[nb] = dist[curr] + 1
                        queue.append(nb)
            # Closer = higher bonus (exponential decay)
            proximity_bonus += np.exp(-dist / 3.0)

        # Normalize and mix: 60% activity + 40% proximity
        proximity_bonus /= (proximity_bonus.max() + 1e-9)
        node_activity_norm = node_activity / (node_activity.max() + 1e-9)
        survival_score = 0.6 * node_activity_norm + 0.4 * proximity_bonus

    sorted_indices = np.argsort(survival_score)[::-1]
    survivors = sorted(sorted_indices[:n_target])
    dead = sorted(sorted_indices[n_target:])

    if verbose:
        print(f"  Apoptosis: {len(dead)} killed, {len(survivors)} survive")
        top5 = sorted_indices[:5]
        print(f"  Top survivors: {[f'N{i}={survival_score[i]:.3f}' for i in top5]}")
        if guided_apoptosis:
            # Count how many sensor nodes (0-11) survived
            sensor_survived = sum(1 for s in survivors if s < 12)
            print(f"  Sensor nodes surviving: {sensor_survived}/12")

    # Build survivor brain
    old_to_new = {old: new for new, old in enumerate(survivors)}
    surviving_edges = []
    for e_idx, (i, j) in enumerate(brain_2x.edges):
        if i in old_to_new and j in old_to_new:
            surviving_edges.append((old_to_new[i], old_to_new[j], e_idx))

    surviving_np = node_params_2x.reshape(n_2x, 5)[survivors].flatten()
    surviving_ews = []
    for li in range(n_layers):
        surviving_ews.append(np.array([layer_ews_2x[li][e_idx]
                                       for _, _, e_idx in surviving_edges]))

    n = n_target
    brain = SurvivorBrain(n, [(i, j) for i, j, _ in surviving_edges])
    node_flat = surviving_np.copy()
    node_ranges = NODE_PARAM_RANGES_SINGLE * n
    layer_ews = [ew.copy() for ew in surviving_ews]
    layer_moms = [np.zeros(brain.n_edges) for _ in range(n_layers)]

    # Sensor mapping
    if sensor_broadcast:
        # ALL nodes receive sensor input (tiled/repeated)
        def map_sensors(sensors_12, n):
            s = np.zeros(n)
            for i in range(n):
                s[i] = sensors_12[i % 12]  # tile: 0,1,...,11,0,1,...,11,...
            return s
    else:
        def map_sensors(sensors_12, n):
            s = np.zeros(n)
            s[:min(12, n)] = sensors_12[:min(12, n)]
            return s

    # Motor nodes
    motor_a = min(n - 1, max(0, int(n * 5 / 48)))
    motor_b = min(n - 1, max(0, int(n * 11 / 48)))
    if motor_a == motor_b:
        motor_b = min(n - 1, motor_a + 1)

    if verbose:
        print(f"  Edges surviving: {brain.n_edges} (from {brain_2x.n_edges})")
        print(f"  Motor: nav=N{motor_a}, cen=N{motor_b}")
        sens_mode = "broadcast (all nodes)" if sensor_broadcast else "first 12 only"
        print(f"  Sensors: {sens_mode}")

    best_node_flat = node_flat.copy()
    best_ews = [ew.copy() for ew in layer_ews]
    best_score = -float('inf')
    total_evals = 0
    phase_scores = []
    prune_masks = [np.ones(brain.n_edges, dtype=bool) for _ in range(n_layers)]

    def run_eval(np_flat, ews, masks):
        np_2d = np_flat.reshape(n, 5)
        edge_matrices = []
        for li in range(n_layers):
            ew = ews[li].copy()
            if len(ew) > 0:
                ew[~masks[li]] = 0.0
            edge_matrices.append(brain.weights_to_matrix(ew))

        ep_scores = []
        ep_reached = 0
        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors_12 = world.reset()
            initial_dist = world.get_food_dist()
            sensors = map_sensors(sensors_12, n)

            layer_states = [np.zeros(n) for _ in range(n_layers)]
            min_dist = initial_dist
            reached = caught = False
            move_sum = 0.0

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                firings = []
                for li in range(n_layers):
                    layer_states[li] = brain.simulate_layer(
                        np_2d, edge_matrices[li], sensors, layer_states[li])
                    firings.append(layer_states[li].copy())

                nav = sum(f[motor_a] for f in firings) / n_layers
                cen = sum(f[motor_b] for f in firings) / n_layers

                sensors_12, dist, reached_now, caught_now = world.step(nav, cen)
                sensors = map_sensors(sensors_12, n)
                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))
                if abs(reward) > reward_thr:
                    for li in range(n_layers):
                        ews[li], layer_moms[li] = brain.hebbian_update_vec(
                            ews[li], firings[li], reward,
                            lr_base, layer_moms[li], mom_coeff)
                        ew_m = ews[li].copy()
                        if len(ew_m) > 0:
                            ew_m[~masks[li]] = 0.0
                        edge_matrices[li] = brain.weights_to_matrix(ew_m)

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        return np.mean(ep_scores), ep_reached

    # Phase 1: Infant
    phase_configs = [
        ("Infant", infant_budget, 0.10),
        ("Child", child_budget, 0.05),
        ("Adult", adult_budget, 0.02),
    ]

    base_np = node_flat.copy()
    for phase_name, budget, perturb_scale in phase_configs:
        if verbose:
            print(f"\n  === {phase_name} ({budget:.0f}s) ===")

        # Pruning at start of Child phase
        if phase_name == "Child":
            for li in range(n_layers):
                if len(best_ews[li]) > 0:
                    before = prune_masks[li].sum()
                    prune_masks[li] = np.abs(best_ews[li]) >= 0.3
                    after = prune_masks[li].sum()
                    if verbose:
                        print(f"    L{li}: {before}→{after} edges "
                              f"({after/max(before,1)*100:.0f}%)")

        phase_t0 = time.time()
        p_best = best_score if best_score > -float('inf') else -float('inf')
        p_evals = 0
        layer_ews = [ew.copy() for ew in best_ews]

        while time.time() - phase_t0 < budget:
            np_trial = base_np.copy()
            for i in range(n * 5):
                lo, hi = node_ranges[i]
                np_trial[i] += rng.normal(0, perturb_scale * (hi - lo))
                np_trial[i] = np.clip(np_trial[i], lo, hi)

            for li in range(n_layers):
                layer_moms[li] = np.zeros(brain.n_edges)

            trial_ews = [ew.copy() for ew in layer_ews]
            score, ep_reached = run_eval(np_trial, trial_ews, prune_masks)
            p_evals += 1
            total_evals += 1

            if score > p_best:
                p_best = score
                best_score = score
                best_node_flat = np_trial.copy()
                best_ews = [ew.copy() for ew in trial_ews]
                layer_ews = [ew.copy() for ew in trial_ews]
                base_np = np_trial.copy()

            if verbose and p_evals % 10 == 0:
                el = time.time() - phase_t0
                print(f"    [{el:.0f}s] eval={p_evals}, best={p_best:.2f}, "
                      f"reach={ep_reached}/{n_episodes}")

        phase_scores.append(p_best)
        if verbose:
            print(f"    {phase_name} done: best={p_best:.2f}, evals={p_evals}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  {label} FINAL: best={best_score:.2f}, evals={total_evals}, {elapsed:.0f}s")
        print(f"  Dev: {' → '.join(f'{s:.1f}' for s in phase_scores)}")

    return {
        "mode": label,
        "best_score": best_score,
        "phase_scores": phase_scores,
        "n_evals": total_evals,
        "n_edges": brain.n_edges,
        "elapsed_s": elapsed,
    }


def run_deep_dive(time_budget=900, verbose=True):
    """3 hypotheses tested on V3p (the hard environment).

    H1: Sensor broadcast (all nodes see all sensors)
    H2: Guided apoptosis (bias survival toward sensor/motor nodes)
    H3: H1 + H2 combined
    + Baseline: 12N×1L for comparison
    + Time control: fetal baseline with same time
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  DEEP DIVE: Why does 48N lose to 12N on V3p?")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    # All on V3p (the problem environment)
    v3p_factory = lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)
    v1_factory = lambda **kw: FlyWorldV1(**kw)

    per_run = time_budget // 6  # 6 experiments
    results = {}

    configs = [
        # name, factory, use_12n, sensor_broadcast, guided_apoptosis
        ("12N_1L_V3p", v3p_factory, True, False, False),
        ("Fetal_base_V3p", v3p_factory, False, False, False),
        ("H1_SensorBC_V3p", v3p_factory, False, True, False),
        ("H2_GuidedApo_V3p", v3p_factory, False, False, True),
        ("H3_BC+Guided_V3p", v3p_factory, False, True, True),
        ("H3_BC+Guided_V1", v1_factory, False, True, True),
    ]

    for name, factory, use_12n, sbc, gapo in configs:
        if verbose:
            print(f"\n  {'=' * 50}")
            print(f"  {name} ({per_run}s)")
            print(f"  {'=' * 50}")

        if use_12n:
            r = run_mode_b_plus(
                world_factory=factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose)
            results[name] = {"best_score": r["best_score"], "n_evals": r["n_evals"]}
        else:
            r = run_fetal_brain_v2(
                world_factory=factory, n_target=48, n_layers=6,
                time_budget=per_run, n_episodes=4, label=name, verbose=verbose,
                sensor_broadcast=sbc, guided_apoptosis=gapo)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "phase_scores": r.get("phase_scores", []),
                "n_edges": r.get("n_edges"),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  DEEP DIVE RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<22} {'Score':>8} {'Evals':>8}")
        print(f"  {'-' * 40}")
        for name, r in results.items():
            print(f"  {name:<22} {r['best_score']:>8.2f} {r['n_evals']:>8}")

        print(f"\n  Analysis:")
        base_12n = results.get("12N_1L_V3p", {}).get("best_score", 0)
        fetal_base = results.get("Fetal_base_V3p", {}).get("best_score", 0)
        h1 = results.get("H1_SensorBC_V3p", {}).get("best_score", 0)
        h2 = results.get("H2_GuidedApo_V3p", {}).get("best_score", 0)
        h3 = results.get("H3_BC+Guided_V3p", {}).get("best_score", 0)

        print(f"    Baseline 12N:        {base_12n:.2f}")
        print(f"    Fetal baseline:      {fetal_base:.2f} ({fetal_base - base_12n:+.2f})")
        print(f"    H1 SensorBroadcast:  {h1:.2f} ({h1 - fetal_base:+.2f} vs fetal)")
        print(f"    H2 GuidedApoptosis:  {h2:.2f} ({h2 - fetal_base:+.2f} vs fetal)")
        print(f"    H3 Both:             {h3:.2f} ({h3 - fetal_base:+.2f} vs fetal)")

        if h3 > base_12n:
            print(f"\n    ★ 48N BEATS 12N! ({h3:.2f} > {base_12n:.2f})")
        else:
            gap = base_12n - h3
            print(f"\n    Gap remaining: {gap:.2f}")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v7_deepdive_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")
    return results


def run_parallel_comparison(time_budget=600, verbose=True):
    """Compare: 12N×1L vs 48N×1L(parallel) vs 48N×6L(parallel) on V1 and V3p."""
    if verbose:
        print("\n" + "=" * 60)
        print("  Parallel vs Serial Brain Comparison")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    per_run = time_budget // 4  # 4 configs
    results = {}

    brain_48 = ScalableBrain(48, [1, 4, 12, 24])

    configs = [
        ("12N_1L_V1", None, 1, lambda **kw: FlyWorldV1(**kw)),
        ("48N_6L_par_V1", brain_48, 6, lambda **kw: FlyWorldV1(**kw)),
        ("12N_1L_V3p", None, 1, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
        ("48N_6L_par_V3p", brain_48, 6, lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)),
    ]

    for name, brain, n_layers, factory in configs:
        if verbose:
            print(f"\n  --- {name} ({per_run}s) ---")

        if brain is None:
            r = run_mode_b_plus(
                world_factory=factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose)
            results[name] = {"best_score": r["best_score"], "n_evals": r["n_evals"],
                             "architecture": "serial_12N"}
        else:
            r = run_parallel_brain(
                brain=brain, world_factory=factory, n_layers=n_layers,
                sensor_nodes=[3, 7, 8, 9, 10], motor_nodes=[5, 11],
                time_budget=per_run, n_episodes=4, label=name, verbose=verbose)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "architecture": "parallel",
                "phase_scores": r.get("phase_scores", []),
                "phase_names": r.get("phase_names", []),
                "edge_survival": r.get("edge_survival", []),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  PARALLEL COMPARISON RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<20} {'Score':>8} {'Evals':>8} {'Arch':<10}")
        print(f"  {'-' * 48}")
        for name, r in results.items():
            arch = r.get("architecture", "")
            print(f"  {name:<20} {r['best_score']:>8.2f} {r['n_evals']:>8} {arch:<10}")

        # Comparison
        for env in ["V1", "V3p"]:
            k12 = f"12N_1L_{env}"
            k_par = f"48N_6L_par_{env}"
            if k12 in results and k_par in results:
                s12 = results[k12]["best_score"]
                s_par = results[k_par]["best_score"]
                print(f"\n  {env}: 48N×6L(parallel) vs 12N×1L: {s_par - s12:+.2f}")

        # Previous serial results for comparison
        print(f"\n  Reference (serial model):")
        print(f"    48N_6L_serial_V1:  97.08")
        print(f"    48N_6L_serial_V3p: 84.78")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v7_parallel_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")

    return results


# ============================================================
# v8: Biological structure experiments
# ============================================================

def run_fetal_brain_v3(world_factory, n_target=48, n_layers=6,
                       time_budget=300, n_episodes=4, verbose=True, label="FetalV3",
                       inhibit_ratio=0.0, asymmetric=False, inhib_placement="hub"):
    """Fetal brain with Dale's law inhibition + optional asymmetric connections.

    v8 core: tests whether biological structural principles improve 48N.
    inhib_placement: "hub" (high-degree nodes) or "low_activity" (lowest firing).
    """
    n_2x = n_target * 2
    offsets_2x = [1, 4, 12, 24, 48]
    # Clean fetal: brain_2x has NO inhibition during natural selection.
    # Inhibition is assigned ONLY to survivors via fetal_phase().
    brain_2x = ScalableBrain(n_2x, offsets_2x, inhibit_ratio=0.0,
                              rng=np.random.RandomState(42))

    fetal_budget = time_budget * 0.10
    infant_budget = time_budget * 0.55
    child_budget = time_budget * 0.20
    adult_budget = time_budget * 0.15

    if verbose:
        flags = []
        if inhibit_ratio > 0:
            flags.append(f"GABA={inhibit_ratio*100:.0f}%")
        if asymmetric:
            flags.append("Asymmetric(FF:FB=2:1)")
        flag_str = " + ".join(flags) if flags else "NoInhib"
        print(f"\n{'=' * 60}")
        print(f"  {label}: Fetal Brain [{flag_str}]")
        print(f"  {n_2x}N → {n_target}N × {n_layers}L")
        print(f"{'=' * 60}")

    t0 = time.time()
    rng = np.random.RandomState(42)
    lr_base = 0.05
    mom_coeff = 0.8
    reward_thr = 0.01

    # === Phase 0: Fetal (with inhibition) ===
    fetal = fetal_phase(brain_2x, n_target, n_layers, rng,
                        time_budget=fetal_budget, verbose=verbose,
                        inhibit_ratio=inhibit_ratio,
                        inhib_placement=inhib_placement)

    # Build survivor brain with Dale's law
    brain = SurvivorBrain(fetal["n_nodes"], fetal["edges"],
                           inhibit_sign=fetal.get("inhibit_sign"),
                           asymmetric=asymmetric)
    node_flat = fetal["node_params"].copy()
    node_ranges = NODE_PARAM_RANGES_SINGLE * fetal["n_nodes"]
    n = fetal["n_nodes"]

    # Edge weights: double if asymmetric
    if asymmetric:
        layer_ews = []
        for ew in fetal["edge_weights"]:
            # Forward = original, backward = 0.47x (FF:FB = 2.13:1)
            bw = ew * 0.47
            layer_ews.append(np.concatenate([ew.copy(), bw]))
        ne_per_layer = brain.n_edges * 2
    else:
        layer_ews = [ew.copy() for ew in fetal["edge_weights"]]
        ne_per_layer = brain.n_edges

    layer_moms = [np.zeros(ne_per_layer) for _ in range(n_layers)]

    # Motor/sensor mapping
    motor_a = min(n - 1, max(0, int(n * 5 / 48)))
    motor_b = min(n - 1, max(0, int(n * 11 / 48)))
    if motor_a == motor_b:
        motor_b = min(n - 1, motor_a + 1)

    if verbose:
        n_inhib = int((brain.inhibit_sign < 0).sum())
        print(f"  Edges: {brain.n_edges} (params/layer: {ne_per_layer})")
        print(f"  Motor: nav=N{motor_a}, cen=N{motor_b}")
        print(f"  Inhibitory: {n_inhib}/{n} ({n_inhib/n*100:.0f}%)")

    best_node_flat = node_flat.copy()
    best_ews = [ew.copy() for ew in layer_ews]
    best_score = -float('inf')
    total_evals = 0
    phase_scores = []
    prune_masks = [np.ones(ne_per_layer, dtype=bool) for _ in range(n_layers)]

    def run_eval(np_flat, ews, masks):
        np_2d = np_flat.reshape(n, 5)
        edge_matrices = []
        for li in range(n_layers):
            ew = ews[li].copy()
            if len(ew) > 0:
                ew[~masks[li]] = 0.0
            edge_matrices.append(brain.weights_to_matrix(ew))

        ep_scores = []
        ep_reached = 0
        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors_12 = world.reset()
            initial_dist = world.get_food_dist()
            sensors = np.zeros(n)
            sensors[:min(12, n)] = sensors_12[:min(12, n)]

            layer_states = [np.zeros(n) for _ in range(n_layers)]
            min_dist = initial_dist
            reached = caught = False
            move_sum = 0.0

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                firings = []
                for li in range(n_layers):
                    layer_states[li] = brain.simulate_layer(
                        np_2d, edge_matrices[li], sensors, layer_states[li])
                    firings.append(layer_states[li].copy())

                nav = sum(f[motor_a] for f in firings) / n_layers
                cen = sum(f[motor_b] for f in firings) / n_layers

                sensors_12, dist, reached_now, caught_now = world.step(nav, cen)
                sensors = np.zeros(n)
                sensors[:min(12, n)] = sensors_12[:min(12, n)]
                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))
                if abs(reward) > reward_thr:
                    for li in range(n_layers):
                        ews[li], layer_moms[li] = brain.hebbian_update_vec(
                            ews[li], firings[li], reward,
                            lr_base, layer_moms[li], mom_coeff)
                        ew_m = ews[li].copy()
                        if len(ew_m) > 0:
                            ew_m[~masks[li]] = 0.0
                        edge_matrices[li] = brain.weights_to_matrix(ew_m)

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        return np.mean(ep_scores), ep_reached

    # 3-phase development: Infant → Child → Adult
    phase_configs = [
        ("Infant", infant_budget, 0.10),
        ("Child", child_budget, 0.05),
        ("Adult", adult_budget, 0.02),
    ]

    base_np = node_flat.copy()
    for phase_name, budget, perturb_scale in phase_configs:
        if verbose:
            print(f"\n  === {phase_name} ({budget:.0f}s) ===")

        if phase_name == "Child":
            for li in range(n_layers):
                if len(best_ews[li]) > 0:
                    before = prune_masks[li].sum()
                    prune_masks[li] = np.abs(best_ews[li]) >= 0.3
                    after = prune_masks[li].sum()
                    if verbose:
                        print(f"    L{li}: {before}→{after} edges "
                              f"({after/max(before,1)*100:.0f}%)")

        phase_t0 = time.time()
        p_best = best_score if best_score > -float('inf') else -float('inf')
        p_evals = 0
        layer_ews_work = [ew.copy() for ew in best_ews]

        while time.time() - phase_t0 < budget:
            np_trial = base_np.copy()
            for i in range(n * 5):
                lo, hi = node_ranges[i]
                np_trial[i] += rng.normal(0, perturb_scale * (hi - lo))
                np_trial[i] = np.clip(np_trial[i], lo, hi)

            for li in range(n_layers):
                layer_moms[li] = np.zeros(ne_per_layer)

            trial_ews = [ew.copy() for ew in layer_ews_work]
            score, ep_reached = run_eval(np_trial, trial_ews, prune_masks)
            p_evals += 1
            total_evals += 1

            if score > p_best:
                p_best = score
                best_score = score
                best_node_flat = np_trial.copy()
                best_ews = [ew.copy() for ew in trial_ews]
                layer_ews_work = [ew.copy() for ew in trial_ews]
                base_np = np_trial.copy()

            if verbose and p_evals % 10 == 0:
                el = time.time() - phase_t0
                print(f"    [{el:.0f}s] eval={p_evals}, best={p_best:.2f}, "
                      f"reach={ep_reached}/{n_episodes}")

        phase_scores.append(p_best)
        if verbose:
            print(f"    {phase_name} done: best={p_best:.2f}, evals={p_evals}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  {label} FINAL: best={best_score:.2f}, evals={total_evals}, {elapsed:.0f}s")
        print(f"  Dev: {' → '.join(f'{s:.1f}' for s in phase_scores)}")

    return {
        "mode": label,
        "best_score": best_score,
        "phase_scores": phase_scores,
        "n_evals": total_evals,
        "n_edges": brain.n_edges,
        "elapsed_s": elapsed,
        "inhibit_ratio": inhibit_ratio,
        "asymmetric": asymmetric,
    }


def run_inhibition_test_12n(time_budget=300, verbose=True):
    """Test: 12N with/without inhibition on V3p.

    Key question: Does Dale's law (20% GABA) improve predator avoidance?
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  v8 INHIBITION TEST: 12N × V3p")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    v3p_factory = lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)
    per_run = time_budget // 2
    results = {}

    configs = [
        ("12N_noInhib_V3p", 0.0),
        ("12N_20pct_V3p", 0.20),
    ]

    for name, inhib in configs:
        if verbose:
            print(f"\n  {'=' * 50}")
            print(f"  {name} ({per_run}s)")
            print(f"  {'=' * 50}")

        r = run_mode_b_plus(
            world_factory=v3p_factory, time_budget=per_run,
            warm_start=None, n_episodes=4, label=name, verbose=verbose,
            inhibit_ratio=inhib)
        results[name] = {
            "best_score": r["best_score"],
            "n_evals": r["n_evals"],
            "inhibit_ratio": inhib,
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  12N INHIBITION TEST RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<22} {'Score':>8} {'Evals':>8} {'Inhib':>8}")
        print(f"  {'-' * 48}")
        for name, r in results.items():
            print(f"  {name:<22} {r['best_score']:>8.2f} {r['n_evals']:>8} "
                  f"{r['inhibit_ratio']:>7.0%}")

        no_inhib = results.get("12N_noInhib_V3p", {}).get("best_score", 0)
        with_inhib = results.get("12N_20pct_V3p", {}).get("best_score", 0)
        delta = with_inhib - no_inhib
        print(f"\n  Effect: {delta:+.2f}")
        print(f"  Reference v7: 12N_V3p = 94.10")
        if delta > 0:
            print(f"  ★ INHIBITION IMPROVES V3p!")
        else:
            print(f"  Inhibition did not help on 12N")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v8_inhib12n_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")
    return results


def run_inhibition_test_48n(time_budget=600, verbose=True):
    """Test: 48N with inhibition variants on V3p.

    4 conditions:
    1. 48N no inhibition (v7 baseline)
    2. 48N + 20% inhibition (Dale's law)
    3. 48N + 20% + asymmetric (FF:FB = 2:1)
    4. 48N + 20% + asymmetric + 12N baseline for reference
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  v8 INHIBITION TEST: 48N × V3p")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    v3p_factory = lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)
    per_run = time_budget // 4
    results = {}

    configs = [
        # (name, use_12n, inhibit_ratio, asymmetric)
        ("12N_ref_V3p", True, 0.20, False),  # 12N with inhib (reference)
        ("48N_noInhib_V3p", False, 0.0, False),
        ("48N_20pct_V3p", False, 0.20, False),
        ("48N_asym_V3p", False, 0.20, True),
    ]

    for name, use_12n, inhib, asym in configs:
        if verbose:
            print(f"\n  {'=' * 50}")
            print(f"  {name} ({per_run}s)")
            print(f"  {'=' * 50}")

        if use_12n:
            r = run_mode_b_plus(
                world_factory=v3p_factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose,
                inhibit_ratio=inhib)
            results[name] = {
                "best_score": r["best_score"],
                "n_evals": r["n_evals"],
                "inhibit_ratio": inhib,
                "asymmetric": False,
            }
        else:
            r = run_fetal_brain_v3(
                world_factory=v3p_factory, n_target=48, n_layers=6,
                time_budget=per_run, n_episodes=4, label=name, verbose=verbose,
                inhibit_ratio=inhib, asymmetric=asym)
            results[name] = {
                "best_score": r["best_score"],
                "n_evals": r["n_evals"],
                "inhibit_ratio": inhib,
                "asymmetric": asym,
                "phase_scores": r.get("phase_scores", []),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  48N INHIBITION TEST RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<22} {'Score':>8} {'Evals':>8} {'Inhib':>6} {'Asym':>6}")
        print(f"  {'-' * 52}")
        for name, r in results.items():
            asym_str = "Y" if r.get("asymmetric") else "N"
            print(f"  {name:<22} {r['best_score']:>8.2f} {r['n_evals']:>8} "
                  f"{r['inhibit_ratio']:>5.0%} {asym_str:>6}")

        ref = results.get("12N_ref_V3p", {}).get("best_score", 0)
        no_inhib = results.get("48N_noInhib_V3p", {}).get("best_score", 0)
        with_inhib = results.get("48N_20pct_V3p", {}).get("best_score", 0)
        asym_score = results.get("48N_asym_V3p", {}).get("best_score", 0)

        print(f"\n  Analysis:")
        print(f"    12N+inhib reference:  {ref:.2f}")
        print(f"    48N baseline:         {no_inhib:.2f}")
        print(f"    48N+inhib:            {with_inhib:.2f} ({with_inhib - no_inhib:+.2f})")
        print(f"    48N+inhib+asym:       {asym_score:.2f} ({asym_score - no_inhib:+.2f})")
        print(f"    v7 reference:         12N=94.10, 48N_fetal=83.06")

        if asym_score > ref:
            print(f"\n    ★★ 48N BEATS 12N! ({asym_score:.2f} > {ref:.2f})")
        elif with_inhib > no_inhib:
            print(f"\n    ★ Inhibition improves 48N ({with_inhib - no_inhib:+.2f})")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v8_inhib48n_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")
    return results


def run_hub_inhibition_test(time_budget=600, verbose=True):
    """Test: clean fetal + post-survival inhibition assignment.

    Key fix: NO inhibition during fetal phase (pure natural selection).
    Inhibition assigned ONLY to survivors, at lower rates.

    Paper re-read:
    - Backbone = ACh 1.09x, GABA 0.83x → backbone is EXCITATORY
    - Inhibition is NOT on hub nodes, it's on mid-degree "interneurons"
    - Memory system GABA = 9% (lowest) → 10% is biologically valid
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  v8 CLEAN FETAL + POST-SURVIVAL INHIBITION TEST")
        print(f"  Budget: {time_budget}s total")
        print("  Fix: NO inhibition during fetal phase")
        print("=" * 60)

    v3p_factory = lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)
    per_run = time_budget // 5
    results = {}

    configs = [
        # (name, use_12n, fetal_inhib, survivor_inhib, asym, placement)
        # fetal_inhib = 0 means clean fetal, survivor_inhib applied after survival
        ("12N_ref_V3p", True, 0.0, 0.20, False, "hub"),
        ("48N_noInhib_V3p", False, 0.0, 0.0, False, "mid"),
        ("48N_clean10_mid", False, 0.0, 0.10, False, "mid"),
        ("48N_clean20_mid", False, 0.0, 0.20, False, "mid"),
        ("48N_clean10_asym", False, 0.0, 0.10, True, "mid"),
    ]

    for name, use_12n, f_inh, s_inh, asym, placement in configs:
        if verbose:
            print(f"\n  {'=' * 50}")
            print(f"  {name} ({per_run}s)")
            print(f"  {'=' * 50}")

        if use_12n:
            r = run_mode_b_plus(
                world_factory=v3p_factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose,
                inhibit_ratio=s_inh)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "fetal_inhib": f_inh, "survivor_inhib": s_inh,
            }
        else:
            # Clean fetal: brain_2x has NO inhibition (f_inh=0)
            # Survivor gets s_inh inhibition
            r = run_fetal_brain_v3(
                world_factory=v3p_factory, n_target=48, n_layers=6,
                time_budget=per_run, n_episodes=4, label=name, verbose=verbose,
                inhibit_ratio=s_inh, asymmetric=asym,
                inhib_placement=placement)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "fetal_inhib": f_inh, "survivor_inhib": s_inh,
                "asymmetric": asym, "placement": placement,
                "phase_scores": r.get("phase_scores", []),
            }

    if verbose:
        print(f"\n{'=' * 60}")
        print("  CLEAN FETAL + POST-SURVIVAL INHIBITION RESULTS")
        print(f"{'=' * 60}")
        print(f"  {'Config':<22} {'Score':>8} {'F_inh':>6} {'S_inh':>6} {'Asym':>5}")
        print(f"  {'-' * 50}")
        for name, r in results.items():
            asym_str = "Y" if r.get("asymmetric") else "N"
            print(f"  {name:<22} {r['best_score']:>8.2f} "
                  f"{r['fetal_inhib']:>5.0%} {r['survivor_inhib']:>5.0%} {asym_str:>5}")

        ref = results.get("12N_ref_V3p", {}).get("best_score", 0)
        base = results.get("48N_noInhib_V3p", {}).get("best_score", 0)
        c10 = results.get("48N_clean10_mid", {}).get("best_score", 0)
        c20 = results.get("48N_clean20_mid", {}).get("best_score", 0)
        c10a = results.get("48N_clean10_asym", {}).get("best_score", 0)

        print(f"\n  Analysis:")
        print(f"    12N reference:           {ref:.2f}")
        print(f"    48N no inhib:            {base:.2f}")
        print(f"    48N clean fetal + 10%:   {c10:.2f} ({c10 - base:+.2f})")
        print(f"    48N clean fetal + 20%:   {c20:.2f} ({c20 - base:+.2f})")
        print(f"    48N clean + 10% + asym:  {c10a:.2f} ({c10a - base:+.2f})")
        print(f"\n    Previous dirty fetal: 48N_20%=60.40, hub_20%=47.49")

        best_48 = max(c10, c20, c10a)
        if best_48 > base:
            print(f"\n    ★ INHIBITION IMPROVES 48N! ({best_48:.2f} > {base:.2f})")
        if best_48 > ref:
            print(f"    ★★★ 48N BEATS 12N! ({best_48:.2f} > {ref:.2f})")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v8_hub_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")
    return results


# ============================================================
# Hierarchical Kathara: n × Kathara(12) clusters
# ============================================================
def run_hierarchical_kathara(world_factory, time_budget=300, n_episodes=4,
                              verbose=True, label="HierK", inhibit_ratio=0.0,
                              n_clusters=4, skip_connections=False, feedback=False):
    """Hierarchical Kathara: n_clusters × Kathara(12,{1,4,6}).

    Structure: cluster0(sensors) → proj → cluster1 → ... → clusterN → motor
    Each cluster preserves the PROVEN 12N Kathara topology (λ₂=4.0, diameter 2).

    Options:
      skip_connections: cluster 0 → c (c>=2) direct skip projections
      feedback: cluster[-1] → cluster[0] recurrent connection
    """
    offsets = [1, 4, 6]
    n_per = 12
    n_total = n_clusters * n_per

    rng = np.random.RandomState(42)

    # Create cluster brains — each is a proven 12N Kathara
    clusters = [ScalableBrain(n_per, offsets, inhibit_ratio=inhibit_ratio,
                               rng=np.random.RandomState(42 + c))
                for c in range(n_clusters)]
    n_edges = clusters[0].n_edges  # 30

    # Per-cluster parameters
    node_ranges = NODE_PARAM_RANGES_SINGLE * n_per
    cluster_ews = [rng.uniform(-1, 1, n_edges).astype(np.float64) for _ in range(n_clusters)]
    cluster_nps = []
    for c in range(n_clusters):
        np_flat = np.array([rng.uniform(lo, hi) for lo, hi in node_ranges])
        cluster_nps.append(np_flat)

    # Projection weights layout:
    #   [0 .. n_proj_fwd-1]                    = forward (c-1 → c)
    #   [n_proj_fwd .. +n_proj_skip-1]         = skip (0 → c, c>=2)
    #   [n_proj_fwd+n_proj_skip .. +n_proj_fb] = feedback (last → 0)
    n_proj_per = n_per  # 12
    n_proj_fwd = (n_clusters - 1) * n_proj_per
    n_proj_skip = (n_clusters - 2) * n_proj_per if (skip_connections and n_clusters >= 3) else 0
    n_proj_fb = n_proj_per if feedback else 0
    n_proj_total = n_proj_fwd + n_proj_skip + n_proj_fb
    skip_offset = n_proj_fwd
    fb_offset = n_proj_fwd + n_proj_skip

    proj_weights = rng.uniform(0.3, 1.0, n_proj_total).astype(np.float64)

    # Motor: nodes 5, 11 in LAST cluster
    motor_a, motor_b = 5, 11

    lr = OPTIMAL_RL["lr"]
    momentum_coeff = OPTIMAL_RL["momentum"]
    reward_thr = OPTIMAL_RL["reward_threshold"]

    best_score = -float('inf')
    best_ews = [ew.copy() for ew in cluster_ews]
    best_nps = [np.copy() for np in cluster_nps]
    best_proj = proj_weights.copy()
    base_ews = [ew.copy() for ew in cluster_ews]
    base_nps = [np.copy() for np in cluster_nps]
    base_proj = proj_weights.copy()
    n_evals = 0
    scores_log = []

    n_params = n_clusters * (n_edges + n_per * 5) + n_proj_total
    if verbose:
        flags = []
        if inhibit_ratio > 0:
            flags.append(f"GABA={inhibit_ratio*100:.0f}%")
        if skip_connections:
            flags.append("Skip")
        if feedback:
            flags.append("Feedback")
        flag_str = " + ".join(flags) if flags else "baseline"
        print(f"\n{'='*60}")
        print(f"  {label}: Hierarchical Kathara [{flag_str}]")
        print(f"  {n_clusters} clusters × {n_per}N = {n_total}N total")
        print(f"  Params: {n_params} (fwd:{n_proj_fwd} skip:{n_proj_skip} fb:{n_proj_fb})")
        print(f"  Motor: cluster{n_clusters-1} N{motor_a}/N{motor_b}")
        print(f"{'='*60}")

    t0 = time.time()

    while time.time() - t0 < time_budget:
        # Perturb from base
        trial_ews = []
        trial_nps = []
        for c in range(n_clusters):
            ew = base_ews[c].copy()
            for i in range(n_edges):
                ew[i] += rng.normal(0, 0.1)
                ew[i] = np.clip(ew[i], -3.0, 3.0)
            trial_ews.append(ew)

            np_flat = base_nps[c].copy()
            for i, (lo, hi) in enumerate(node_ranges):
                np_flat[i] += rng.normal(0, 0.1 * (hi - lo))
                np_flat[i] = np.clip(np_flat[i], lo, hi)
            trial_nps.append(np_flat)

        trial_proj = base_proj.copy()
        for i in range(n_proj_total):
            trial_proj[i] += rng.normal(0, 0.1)
            trial_proj[i] = np.clip(trial_proj[i], -3.0, 3.0)

        # Build edge matrices
        edge_matrices = [clusters[c].weights_to_matrix(trial_ews[c])
                         for c in range(n_clusters)]

        # Reset momentums for this trial
        trial_moms = [np.zeros(n_edges) for _ in range(n_clusters)]
        trial_proj_mom = np.zeros(n_proj_total)

        ep_scores = []
        ep_reached = 0

        for ep in range(n_episodes):
            world = world_factory(seed=ep * 7 + 13)
            sensors = world.reset()
            initial_dist = world.get_food_dist()

            cluster_states = [np.zeros(n_per) for _ in range(n_clusters)]
            min_dist = initial_dist
            reached = caught = False
            move_sum = 0.0

            for step in range(60):
                prev_pos = world.fly_pos.copy()
                prev_dist = world.get_food_dist()

                # === Hierarchical processing ===
                cluster_firings = []

                # Cluster 0: sensors + optional feedback from last cluster
                input_0 = sensors[:n_per].copy()
                if feedback and step > 0:
                    fb_w = trial_proj[fb_offset:fb_offset + n_proj_per]
                    input_0 = input_0 + cluster_states[-1] * fb_w * 0.3

                np_2d = trial_nps[0].reshape(n_per, 5)
                cluster_states[0] = clusters[0].simulate_layer(
                    np_2d, edge_matrices[0], input_0, cluster_states[0])
                cluster_firings.append(cluster_states[0].copy())

                # Clusters 1+: forward projection + optional skip from cluster 0
                for c in range(1, n_clusters):
                    p_start = (c - 1) * n_proj_per
                    p_w = trial_proj[p_start:p_start + n_proj_per]
                    input_c = cluster_states[c - 1] * p_w

                    if skip_connections and c >= 2:
                        skip_idx = c - 2
                        s_start = skip_offset + skip_idx * n_proj_per
                        skip_w = trial_proj[s_start:s_start + n_proj_per]
                        input_c = input_c + cluster_states[0] * skip_w * (0.5 ** (c - 1))

                    np_2d_c = trial_nps[c].reshape(n_per, 5)
                    cluster_states[c] = clusters[c].simulate_layer(
                        np_2d_c, edge_matrices[c], input_c, cluster_states[c])
                    cluster_firings.append(cluster_states[c].copy())

                # Motor from last cluster
                nav = cluster_states[-1][motor_a]
                cen = cluster_states[-1][motor_b]

                sensors, dist, reached_now, caught_now = world.step(nav, cen)
                move_sum += np.linalg.norm(world.fly_pos - prev_pos)
                min_dist = min(min_dist, dist)

                if caught_now:
                    caught = True
                    break

                # Hebbian update per cluster
                reward = compute_step_reward(
                    prev_dist, dist, reached_now,
                    np.linalg.norm(world.fly_pos - prev_pos))

                if abs(reward) > reward_thr:
                    for c in range(n_clusters):
                        trial_ews[c], trial_moms[c] = clusters[c].hebbian_update_vec(
                            trial_ews[c], cluster_firings[c], reward, lr,
                            trial_moms[c], momentum_coeff)

                    # Forward projection Hebbian: pre × post × reward
                    for c in range(1, n_clusters):
                        p_start = (c - 1) * n_proj_per
                        pre = cluster_firings[c - 1]
                        post = cluster_firings[c]
                        hebb_proj = pre * post * reward * lr * 0.5
                        trial_proj_mom[p_start:p_start + n_proj_per] = (
                            momentum_coeff * trial_proj_mom[p_start:p_start + n_proj_per]
                            + hebb_proj)
                        trial_proj[p_start:p_start + n_proj_per] = np.clip(
                            trial_proj[p_start:p_start + n_proj_per]
                            + trial_proj_mom[p_start:p_start + n_proj_per],
                            -3.0, 3.0)

                    # Skip projection Hebbian (cluster 0 → c, c>=2)
                    if skip_connections:
                        for c in range(2, n_clusters):
                            skip_idx = c - 2
                            s_start = skip_offset + skip_idx * n_proj_per
                            pre = cluster_firings[0]
                            post = cluster_firings[c]
                            hebb_skip = pre * post * reward * lr * 0.3
                            trial_proj_mom[s_start:s_start + n_proj_per] = (
                                momentum_coeff * trial_proj_mom[s_start:s_start + n_proj_per]
                                + hebb_skip)
                            trial_proj[s_start:s_start + n_proj_per] = np.clip(
                                trial_proj[s_start:s_start + n_proj_per]
                                + trial_proj_mom[s_start:s_start + n_proj_per],
                                -3.0, 3.0)

                    # Feedback projection Hebbian (last → 0)
                    if feedback:
                        pre = cluster_firings[-1]
                        post = cluster_firings[0]
                        hebb_fb = pre * post * reward * lr * 0.3
                        trial_proj_mom[fb_offset:fb_offset + n_proj_per] = (
                            momentum_coeff * trial_proj_mom[fb_offset:fb_offset + n_proj_per]
                            + hebb_fb)
                        trial_proj[fb_offset:fb_offset + n_proj_per] = np.clip(
                            trial_proj[fb_offset:fb_offset + n_proj_per]
                            + trial_proj_mom[fb_offset:fb_offset + n_proj_per],
                            -3.0, 3.0)

                    # Rebuild edge matrices after Hebbian update
                    edge_matrices = [clusters[c].weights_to_matrix(trial_ews[c])
                                     for c in range(n_clusters)]

                if reached_now:
                    reached = True
                    break

            approach = max(0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60 + (30 if reached else 0) + min(10, move_sum * 2)
            if caught:
                ep_score *= 0.3
            ep_scores.append(ep_score)
            if reached:
                ep_reached += 1

        avg_score = np.mean(ep_scores)
        n_evals += 1

        if avg_score > best_score:
            best_score = avg_score
            best_ews = [ew.copy() for ew in trial_ews]
            best_nps = [np.copy() for np in trial_nps]
            best_proj = trial_proj.copy()
            base_ews = [ew.copy() for ew in trial_ews]
            base_nps = [np.copy() for np in trial_nps]
            base_proj = trial_proj.copy()

        scores_log.append(best_score)

        if verbose and n_evals % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] eval={n_evals}, best={best_score:.2f}, "
                  f"reach={ep_reached}/{n_episodes}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  {label} FINAL: best={best_score:.2f}, evals={n_evals}, "
              f"reach_rate={ep_reached}/{n_episodes}, {elapsed:.0f}s")

    return {
        "mode": label,
        "best_score": best_score,
        "n_evals": n_evals,
        "n_clusters": n_clusters,
        "n_total": n_total,
        "n_params": n_params,
        "elapsed_s": elapsed,
        "scores_log": scores_log[-20:],
        "skip_connections": skip_connections,
        "feedback": feedback,
    }


def run_hierarchical_test(time_budget=600, verbose=True):
    """Compare: 12N Kathara vs 48N fetal vs 48N Hierarchical Kathara (4×12)."""
    if verbose:
        print("\n" + "=" * 60)
        print("  v8 HIERARCHICAL KATHARA TEST")
        print(f"  Budget: {time_budget}s total")
        print("  Core hypothesis: 4×Kathara(12) > random fetal 48N")
        print("=" * 60)

    v3p_factory = lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)
    per_run = time_budget // 3
    results = {}

    # 1. 12N reference
    if verbose:
        print(f"\n  {'='*50}\n  12N_ref_V3p ({per_run}s)\n  {'='*50}")
    r = run_mode_b_plus(world_factory=v3p_factory, time_budget=per_run,
                        warm_start=None, n_episodes=4, label="12N_ref", verbose=verbose,
                        inhibit_ratio=0.20)
    results["12N_ref_V3p"] = {"best_score": r["best_score"], "n_evals": r["n_evals"]}

    # 2. 48N fetal (no inhib, clean)
    if verbose:
        print(f"\n  {'='*50}\n  48N_fetal_V3p ({per_run}s)\n  {'='*50}")
    r = run_fetal_brain_v3(world_factory=v3p_factory, n_target=48, n_layers=6,
                           time_budget=per_run, n_episodes=4, label="48N_fetal",
                           verbose=verbose, inhibit_ratio=0.0, asymmetric=False)
    results["48N_fetal_V3p"] = {"best_score": r["best_score"], "n_evals": r["n_evals"]}

    # 3. 48N Hierarchical Kathara (4×12)
    if verbose:
        print(f"\n  {'='*50}\n  48N_hier4_V3p ({per_run}s)\n  {'='*50}")
    r = run_hierarchical_kathara(world_factory=v3p_factory, time_budget=per_run,
                                  n_episodes=4, label="48N_hier4", verbose=verbose,
                                  inhibit_ratio=0.0, n_clusters=4)
    results["48N_hier4_V3p"] = {"best_score": r["best_score"], "n_evals": r["n_evals"],
                                 "n_params": r["n_params"]}

    if verbose:
        print(f"\n{'='*60}")
        print("  HIERARCHICAL KATHARA RESULTS")
        print(f"{'='*60}")
        ref = results["12N_ref_V3p"]["best_score"]
        fetal = results["48N_fetal_V3p"]["best_score"]
        hier = results["48N_hier4_V3p"]["best_score"]

        print(f"  12N Kathara (ref):        {ref:.2f}")
        print(f"  48N fetal (baseline):     {fetal:.2f} ({fetal - ref:+.2f} vs 12N)")
        print(f"  48N hier 4×12:            {hier:.2f} ({hier - ref:+.2f} vs 12N, "
              f"{hier - fetal:+.2f} vs fetal)")

        if hier > fetal:
            print(f"\n  ★ HIERARCHICAL BEATS FETAL! ({hier:.2f} > {fetal:.2f})")
        if hier > ref:
            print(f"  ★★★ HIERARCHICAL BEATS 12N! ({hier:.2f} > {ref:.2f})")
        else:
            gap = ref - hier
            print(f"\n  Gap to 12N: {gap:.2f} points")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v8_hier_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")
    return results


def run_hier_full_test(time_budget=900, verbose=True):
    """Full hierarchical Kathara test: 4 experiments.

    1. hier + 20% inhibition
    2. Cluster scaling (2×12)
    3. Skip connections
    4. Feedback
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  v8 HIERARCHICAL KATHARA FULL TEST (4 experiments)")
        print(f"  Budget: {time_budget}s total")
        print("=" * 60)

    v3p_factory = lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw)
    per_run = time_budget // 6
    results = {}

    configs = [
        # (name, n_clusters, inhibit, skip, feedback)
        ("12N_ref",          None, 0.20, False, False),
        ("48N_hier4",        4,    0.0,  False, False),
        ("48N_hier4_inhib",  4,    0.20, False, False),
        ("24N_hier2",        2,    0.0,  False, False),
        ("48N_hier4_skip",   4,    0.0,  True,  False),
        ("48N_hier4_fb",     4,    0.0,  False, True),
    ]

    for name, nc, inh, skip, fb in configs:
        if verbose:
            print(f"\n  {'='*50}\n  {name} ({per_run}s)\n  {'='*50}")

        if nc is None:
            # 12N reference
            r = run_mode_b_plus(
                world_factory=v3p_factory, time_budget=per_run,
                warm_start=None, n_episodes=4, label=name, verbose=verbose,
                inhibit_ratio=inh)
            results[name] = {"best_score": r["best_score"], "n_evals": r["n_evals"]}
        else:
            r = run_hierarchical_kathara(
                world_factory=v3p_factory, time_budget=per_run,
                n_episodes=4, label=name, verbose=verbose,
                inhibit_ratio=inh, n_clusters=nc,
                skip_connections=skip, feedback=fb)
            results[name] = {
                "best_score": r["best_score"], "n_evals": r["n_evals"],
                "n_params": r["n_params"], "n_clusters": nc,
            }

    if verbose:
        print(f"\n{'='*60}")
        print("  HIERARCHICAL KATHARA FULL RESULTS")
        print(f"{'='*60}")
        print(f"  {'Config':<22} {'Score':>8} {'Params':>7} {'vs 12N':>8}")
        print(f"  {'-'*50}")
        ref = results["12N_ref"]["best_score"]
        for name, r in results.items():
            params = r.get("n_params", 91)
            delta = r["best_score"] - ref
            print(f"  {name:<22} {r['best_score']:>8.2f} {params:>7} {delta:>+8.2f}")

        # Find best 48N+
        best_name = max(
            [(k, v["best_score"]) for k, v in results.items() if k != "12N_ref"],
            key=lambda x: x[1])
        print(f"\n  Best non-12N: {best_name[0]} = {best_name[1]:.2f}")
        if best_name[1] > ref:
            print(f"  ★★★ {best_name[0]} BEATS 12N! ({best_name[1]:.2f} > {ref:.2f})")

    out = os.path.join(os.path.dirname(__file__),
                       "kathara_brain_sim_v8_hier_full_result.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n  Saved: {os.path.basename(out)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kathara Brain v8 - Biological Structure")
    parser.add_argument("--mode", default="inhib_test_12n",
                        choices=["b_plus", "b_baseline", "v1", "v2", "v3", "v3p",
                                 "progression", "layer_compare", "scale", "parallel",
                                 "fetal", "deepdive",
                                 "inhib_test_12n", "inhib_test_48n", "hub_test",
                                 "hier", "hier_full"])
    parser.add_argument("--time", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("  Kathara Brain v8 - Biological Structure Principles")
    print(f"  Mode: {args.mode}, Budget: {args.time}s")
    print("=" * 60)

    warm = load_v5_warm_start()
    ws_str = "v5 warm-start loaded" if warm is not None else "random init"
    print(f"  Init: {ws_str}")

    result = None

    if args.mode == "b_plus" or args.mode == "v1":
        # No warm-start: v5 static params are harmful for Hebbian
        result = run_mode_b_plus(
            world_factory=lambda **kw: FlyWorldV1(**kw),
            time_budget=args.time, warm_start=None, label="B+_V1")

    elif args.mode == "b_baseline":
        # v6 Mode B reproduction test (no warm-start, 4 episodes)
        result = run_mode_b_plus(
            world_factory=lambda **kw: FlyWorldV1(**kw),
            time_budget=args.time, warm_start=None,
            n_episodes=4, label="B_baseline")

    elif args.mode == "v2":
        result = run_mode_b_plus(
            world_factory=lambda **kw: FlyWorldV2(n_foods=3, **kw),
            time_budget=args.time, warm_start=warm, label="B+_V2")

    elif args.mode == "v3":
        result = run_mode_b_plus(
            world_factory=lambda **kw: FlyWorldV3(n_obstacles=3, **kw),
            time_budget=args.time, warm_start=warm, label="B+_V3")

    elif args.mode == "v3p":
        result = run_mode_b_plus(
            world_factory=lambda **kw: FlyWorldV3(n_obstacles=3, has_predator=True, **kw),
            time_budget=args.time, warm_start=warm, label="B+_V3p")

    elif args.mode == "progression":
        result = run_difficulty_progression(time_budget=args.time)

    elif args.mode == "layer_compare":
        result = run_layer_comparison(time_budget=args.time)

    elif args.mode == "scale":
        result = run_scale_comparison(time_budget=args.time)

    elif args.mode == "parallel":
        result = run_parallel_comparison(time_budget=args.time)

    elif args.mode == "fetal":
        result = run_fetal_comparison(time_budget=args.time)

    elif args.mode == "deepdive":
        result = run_deep_dive(time_budget=args.time)

    elif args.mode == "inhib_test_12n":
        result = run_inhibition_test_12n(time_budget=args.time)

    elif args.mode == "inhib_test_48n":
        result = run_inhibition_test_48n(time_budget=args.time)

    elif args.mode == "hub_test":
        result = run_hub_inhibition_test(time_budget=args.time)

    elif args.mode == "hier":
        result = run_hierarchical_test(time_budget=args.time)

    elif args.mode == "hier_full":
        result = run_hier_full_test(time_budget=args.time)

    if result:
        fname = f"kathara_brain_sim_v8_{args.mode}_result.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved: {fname}")
