"""cortical_brain.py — Cortical-style layered Kathara brain

Based on deep brain comparison analysis (脳比較分析.md):
- 5 layers with differentiated roles (input/sensory/association/motor_prep/output)
- Dale's law DEFAULT ON (20% inhibitory per layer)
- Asymmetric FF:FB = 2.13:1 (biological ratio)
- Each layer is Kathara(12, {1,4,6}) = standard cortical column analog

Layer roles:
  L0 (input):       receives 12D sensor vector directly
  L1 (sensory):     processes input, strong internal Kathara
  L2 (association): high connectivity, integrates L1 with feedback from L3/L4
  L3 (motor_prep):  filters and consolidates toward action
  L4 (motor_out):   drives nav/cen outputs (nodes 5 and 11)

Parameter layout (184D):
  [  0: 30]  L0 intra-layer edge weights
  [ 30: 60]  L1 edge weights
  [ 60: 90]  L2 edge weights
  [ 90:120]  L3 edge weights
  [120:150]  L4 edge weights
  [150:154]  FF projection scales (L0->L1, L1->L2, L2->L3, L3->L4)
  [154:158]  FB projection scales (L4->L3, L3->L2, L2->L1, L1->L0) * 0.47 asymmetry
  [158:183]  Per-layer node params: 5 layers * 5 params (gain, bias, iw, leak, thr)
  [    183]  n_steps

Dale's law:
  Each layer has fixed inhibitory pattern:
    Node 8 (danger) always inhibitory
    Nodes 0, 4 randomly designated inhibitory (with seed=42)
    3/12 = 25% per layer (close to biological 20%)

Usage:
    from cortical_brain import CorticalBrain, PARAM_RANGES, run_cortical_flyworld
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse Kathara 12N topology from v8
from kathara_brain_sim_v8 import KATHARA_EDGES, NEIGHBORS

N_LAYERS = 5
N_PER_LAYER = 12
N_EDGES_PER_LAYER = 30
TOTAL_INTRA_EDGES = N_LAYERS * N_EDGES_PER_LAYER  # 150
TOTAL_FF = N_LAYERS - 1  # 4
TOTAL_FB = N_LAYERS - 1  # 4
NODE_PARAMS_PER_LAYER = 5  # gain, bias, iw, leak, thr
TOTAL_NODE_PARAMS = N_LAYERS * NODE_PARAMS_PER_LAYER  # 25
TOTAL_PARAMS = TOTAL_INTRA_EDGES + TOTAL_FF + TOTAL_FB + TOTAL_NODE_PARAMS + 1  # 184

LAYER_ROLES = {
    0: "input",
    1: "sensory",
    2: "association",
    3: "motor_prep",
    4: "output",
}

# Dale's law: fixed inhibitory pattern per layer (same across layers for simplicity)
# Node 8 (biological danger signal) + node 0 + node 4 = 3/12 = 25%
_INHIBIT_NODES = [0, 4, 8]


def make_inhibit_sign():
    """12D array of +1 (excitatory) / -1 (inhibitory) for each layer."""
    sign = np.ones(N_PER_LAYER, dtype=np.float64)
    for n in _INHIBIT_NODES:
        sign[n] = -1.0
    return sign


DEFAULT_INHIBIT = make_inhibit_sign()


# Parameter ranges (184D)
PARAM_RANGES = (
    [(-3.0, 3.0)] * TOTAL_INTRA_EDGES +          # 150 edges
    [(0.0, 2.0)] * TOTAL_FF +                     # 4 FF scales (positive only)
    [(0.0, 1.0)] * TOTAL_FB +                     # 4 FB scales (positive, ~0.47 nominal)
    ([(0.1, 5.0), (-3.0, 3.0), (0.1, 5.0), (0.01, 0.9), (0.01, 1.0)] * N_LAYERS) +
    [(10, 80)]                                    # n_steps
)
assert len(PARAM_RANGES) == TOTAL_PARAMS, f"expected {TOTAL_PARAMS}, got {len(PARAM_RANGES)}"


PARAM_NAMES = []
for layer in range(N_LAYERS):
    for ei in range(N_EDGES_PER_LAYER):
        a, b = KATHARA_EDGES[ei]
        PARAM_NAMES.append(f"L{layer}_e{a}_{b}")
for i in range(TOTAL_FF):
    PARAM_NAMES.append(f"FF_L{i}_to_L{i+1}")
for i in range(TOTAL_FB):
    PARAM_NAMES.append(f"FB_L{i+1}_to_L{i}")
for layer in range(N_LAYERS):
    role = LAYER_ROLES[layer]
    for pname in ["gain", "bias", "iw", "leak", "thr"]:
        PARAM_NAMES.append(f"{role}_{pname}")
PARAM_NAMES.append("n_steps")


def unpack_params(params):
    """Split the 184D param vector into structured arrays."""
    p = np.asarray(params, dtype=np.float64)
    offset = 0
    # Intra-layer edges
    intra_edges = p[offset:offset + TOTAL_INTRA_EDGES].reshape(N_LAYERS, N_EDGES_PER_LAYER)
    offset += TOTAL_INTRA_EDGES
    # FF scales
    ff_scale = p[offset:offset + TOTAL_FF]
    offset += TOTAL_FF
    # FB scales (will be multiplied by 0.47 for asymmetry)
    fb_scale_raw = p[offset:offset + TOTAL_FB]
    fb_scale = fb_scale_raw * 0.47  # Biological FF:FB = 2.13:1
    offset += TOTAL_FB
    # Node params per layer
    node_params = p[offset:offset + TOTAL_NODE_PARAMS].reshape(N_LAYERS, NODE_PARAMS_PER_LAYER)
    offset += TOTAL_NODE_PARAMS
    # n_steps (unused in step-based simulation, kept for dim consistency)
    return intra_edges, ff_scale, fb_scale, node_params


def _simulate_layer(edges, node_p, input_signal, states, inhibit_sign):
    """One layer's Kathara dynamics. Returns new state vector (12D).

    edges: (30,) intra-layer edge weights
    node_p: (5,) gain, bias, iw, leak, thr
    input_signal: (12,) combined input (FF + FB + sensor if L0)
    states: (12,) previous state
    inhibit_sign: (12,) Dale's law sign
    """
    gain, bias, iw, leak, thr = node_p
    effective_states = states * inhibit_sign  # Dale's law
    new_states = np.zeros(N_PER_LAYER, dtype=np.float64)
    for i in range(N_PER_LAYER):
        syn = 0.0
        for j, eidx in NEIGHBORS[i]:
            syn += effective_states[j] * edges[eidx]
        total = syn + iw * input_signal[i] + bias
        x = gain * total
        activation = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        if activation < thr:
            activation *= 0.1
        new_states[i] = activation
    # Leaky integration
    out = states * (1 - leak) + new_states * leak
    return np.clip(out, 0, 1)


class CorticalBrain:
    """5-layer cortical brain wrapping 5 Kathara columns.

    Processes input_vec (12D) → returns motor output (nav, cen) via layer 4.
    Maintains internal state across steps.
    """

    def __init__(self, params):
        (self.intra_edges, self.ff_scale, self.fb_scale,
         self.node_params) = unpack_params(params)
        self.inhibit_sign = DEFAULT_INHIBIT
        self.states = [np.zeros(N_PER_LAYER, dtype=np.float64)
                       for _ in range(N_LAYERS)]

    def step(self, sensor_input):
        """One forward step through all 5 layers.

        sensor_input: (12,) sensor vector (only used at L0)
        Returns: nav, cen (floats from L4 nodes 5 and 11)
        """
        # Snapshot previous states for FB computation (predictive coding-like)
        prev_states = [s.copy() for s in self.states]

        new_states = [None] * N_LAYERS
        for layer in range(N_LAYERS):
            # Compute input signal to this layer
            # Sensor input only at L0
            sensor = sensor_input if layer == 0 else np.zeros(N_PER_LAYER)
            # FF input from layer-1
            ff_in = (self.ff_scale[layer - 1] * prev_states[layer - 1]
                     if layer > 0 else np.zeros(N_PER_LAYER))
            # FB input from layer+1 (using previous step's state = predictive coding)
            fb_in = (self.fb_scale[layer] * prev_states[layer + 1]
                     if layer < N_LAYERS - 1 else np.zeros(N_PER_LAYER))

            combined_input = sensor + ff_in + fb_in
            new_states[layer] = _simulate_layer(
                self.intra_edges[layer],
                self.node_params[layer],
                combined_input,
                prev_states[layer],
                self.inhibit_sign,
            )

        self.states = new_states
        # Motor output from L4
        nav = float(self.states[N_LAYERS - 1][5])
        cen = float(self.states[N_LAYERS - 1][11])
        return nav, cen

    def reset(self):
        self.states = [np.zeros(N_PER_LAYER, dtype=np.float64)
                       for _ in range(N_LAYERS)]

    def firing_accum(self, n_steps, sensor_fn):
        """Accumulate firing across layers for ISS-style analysis.

        sensor_fn: callable(step_idx) -> (12,) sensor vector
        Returns: accumulated firing (5, 12) averaged
        """
        accum = np.zeros((N_LAYERS, N_PER_LAYER), dtype=np.float64)
        for step in range(n_steps):
            sensors = sensor_fn(step)
            self.step(sensors)
            for layer in range(N_LAYERS):
                accum[layer] += self.states[layer]
        return accum / max(1, n_steps)


# ============================================================
# FlyWorld evaluation
# ============================================================

def run_cortical_flyworld(params, world_cls, n_episodes=4, n_steps=60,
                          has_predator=False, seed_offset=13):
    """Behavioral score: run CorticalBrain through FlyWorld episodes."""
    from kathara_brain_sim_v8 import FlyWorldV3
    total_score = 0.0
    for ep in range(n_episodes):
        kwargs = {"seed": ep * 7 + seed_offset}
        if world_cls is FlyWorldV3:
            kwargs["has_predator"] = has_predator
        world = world_cls(**kwargs)
        sensors = world.reset()
        initial_dist = world.get_food_dist()
        min_dist = initial_dist
        reached = False
        caught = False
        move_sum = 0.0

        brain = CorticalBrain(params)

        for _ in range(n_steps):
            prev_pos = world.fly_pos.copy()
            nav, cen = brain.step(sensors)
            sensors, dist, reached_now, caught_now = world.step(nav, cen)
            move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
            if dist < min_dist:
                min_dist = dist
            if reached_now:
                reached = True
                break
            if caught_now:
                caught = True
                break

        approach = max(0.0, initial_dist - min_dist) / (initial_dist + 1e-6)
        ep_score = approach * 60.0
        if reached:
            ep_score += 30.0
        ep_score += min(10.0, move_sum * 2.0)
        if caught:
            ep_score *= 0.3
        total_score += ep_score

    return total_score / n_episodes


def make_cortical_eval(world_cls=None, n_episodes=4, n_steps=60, has_predator=False):
    """Build eval_fn for Sentinel using cortical brain."""
    from kathara_brain_sim_v8 import FlyWorldV3
    wc = world_cls if world_cls is not None else FlyWorldV3

    def eval_fn(params):
        return run_cortical_flyworld(params, wc, n_episodes=n_episodes,
                                     n_steps=n_steps, has_predator=has_predator)
    return eval_fn


def make_cortical_structural_guard(n_episodes=2, n_steps=30):
    """Structural guard for cortical brain: ISS-from-firing at L4 (output layer)."""
    from kathara_brain_sim_v8 import FlyWorldV1
    from kathara_brain_sim_v6 import compute_iss_from_firing

    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=ep * 11 + 7)
            sensors = world.reset()
            brain = CorticalBrain(params)
            firing_accum = np.zeros((N_LAYERS, N_PER_LAYER), dtype=np.float64)
            steps_done = 0
            for _ in range(n_steps):
                nav, cen = brain.step(sensors)
                for layer in range(N_LAYERS):
                    firing_accum[layer] += brain.states[layer]
                sensors, _, reached, _ = world.step(nav, cen)
                steps_done += 1
                if reached:
                    break
            avg = firing_accum / max(1, steps_done)
            # Score L4 (output layer) — most behaviorally relevant
            l4_score = compute_iss_from_firing(avg[N_LAYERS - 1], None)
            # Also score the average across all layers for multi-layer health
            layer_scores = [compute_iss_from_firing(avg[l], None) for l in range(N_LAYERS)]
            mean_layer = float(np.mean(layer_scores))
            # Combined: motor layer + average structural health
            total += 0.5 * float(l4_score) + 0.5 * mean_layer
        return total / n_episodes
    return guard_fn


if __name__ == "__main__":
    # Quick smoke test
    baseline = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    print(f"CorticalBrain dims: {TOTAL_PARAMS}")
    print(f"Layer roles: {LAYER_ROLES}")
    print(f"Dale's law inhibitory nodes: {_INHIBIT_NODES} per layer")
    print(f"FF:FB ratio: 1:{0.47:.2f}")

    from kathara_brain_sim_v8 import FlyWorldV1, FlyWorldV3
    score_v1 = run_cortical_flyworld(baseline, FlyWorldV1, n_episodes=2, n_steps=30)
    score_v3 = run_cortical_flyworld(baseline, FlyWorldV3, n_episodes=2, n_steps=30)
    print(f"\nBaseline scores (midpoint params):")
    print(f"  FlyWorldV1: {score_v1:.2f}")
    print(f"  FlyWorldV3: {score_v3:.2f}")

    guard_fn = make_cortical_structural_guard(n_episodes=2, n_steps=20)
    g = guard_fn(baseline)
    print(f"  Structural guard: {g:.2f}")
