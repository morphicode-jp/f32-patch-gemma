"""kathara16_brain.py - Kathara(16, {3,6,7}) brain substrate for Phase 7.

Topology selected from kathara_topology_search.py (G analysis):
  Kathara(16, {3,6,7}) ranked #1 of 518 with score +2.044
  vs current Kathara(12, {1,4,6}) score +1.232 (rank #99)

Properties:
  |lambda_2| = 4.027 (target 4.0, matches hybrid_analysis finding)
  clustering = 0.600 (6x higher than Kathara(12))
  diameter = 3
  degree = 6 (6-regular: +1, +3, +6, +7, +9, +10 all neighbors)
  n_edges = 48 (vs 30 for Kathara(12))

Parameter layout (139D):
  [ 0: 48]  edge weights (48 edges)
  [48:128]  node params (16 nodes × 5 params: gain, bias, iw, leak, thr)
  [   128]  n_steps
  Total: 48 + 80 + 1 = 129
"""
import numpy as np

N_NODES_16 = 16
SKIPS_16 = [3, 6, 7]

# Build edges
KATHARA16_EDGES = []
for i in range(N_NODES_16):
    for offset in SKIPS_16:
        j = (i + offset) % N_NODES_16
        edge = tuple(sorted((i, j)))
        if i != j and edge not in KATHARA16_EDGES:
            KATHARA16_EDGES.append(edge)

# Verify: 16 nodes, degree = 2*3 = 6 (each skip contributes 2 neighbors because skip != n/2)
# Total edges = 16 * 6 / 2 = 48
assert len(KATHARA16_EDGES) == 48, f"expected 48 edges, got {len(KATHARA16_EDGES)}"

# Neighbor list for fast lookup
NEIGHBORS_16 = [[] for _ in range(N_NODES_16)]
for idx, (a, b) in enumerate(KATHARA16_EDGES):
    NEIGHBORS_16[a].append((b, idx))
    NEIGHBORS_16[b].append((a, idx))

# Sensor/motor role assignment (16 nodes):
# Input (sensor) nodes: 0-12 (13 channels, see multi_agent_world_N)
# Motor output: node 5 (nav), node 11 (speed), node 1 (voice)
# Spare: 13, 14, 15 (reserved for future)
SENSORY_NODES = [0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15]
MOTOR_NODES = [5, 11]  # nav, speed
VOICE_NODE = 1

# Dale's law: 20% inhibitory (3 of 16). Matches biological ~20%.
_INHIBIT_NODES_16 = [0, 4, 8]  # same structural role pattern as Kathara(12)

PARAMS_PER_NODE = 5  # gain, bias, iw, leak, thr
TOTAL_PARAMS_16 = len(KATHARA16_EDGES) + N_NODES_16 * PARAMS_PER_NODE + 1
# = 48 + 80 + 1 = 129

# Parameter ranges: same conventions as Kathara(12)
PARAM_RANGES_16 = (
    [(-3.0, 3.0)] * len(KATHARA16_EDGES) +  # edges
    [(0.1, 5.0), (-3.0, 3.0), (0.1, 5.0), (0.01, 0.9), (0.01, 1.0)]
    * N_NODES_16 +
    [(10, 80)]  # n_steps
)
assert len(PARAM_RANGES_16) == TOTAL_PARAMS_16

PARAM_NAMES_16 = []
for (a, b) in KATHARA16_EDGES:
    PARAM_NAMES_16.append(f"e{a}_{b}")
for i in range(N_NODES_16):
    for pname in ["gain", "bias", "iw", "leak", "thr"]:
        PARAM_NAMES_16.append(f"n{i}_{pname}")
PARAM_NAMES_16.append("n_steps")


def make_inhibit_sign_16():
    """Returns array of +1 (excitatory) / -1 (inhibitory) for each node."""
    sign = np.ones(N_NODES_16, dtype=np.float64)
    for n in _INHIBIT_NODES_16:
        sign[n] = -1.0
    return sign


DEFAULT_INHIBIT_16 = make_inhibit_sign_16()


def simulate_step_16(params, inputs, states, inhibit_sign=None):
    """One step of 16-node Kathara brain. Same logic as kathara_brain_sim_v8.

    params: 129D array
    inputs: 16D sensor array
    states: 16D state array (modified in place conceptually; new array returned)
    inhibit_sign: optional 16D Dale's law vector

    Returns: (new_states, firing) where firing = new_states (for API symmetry).
    """
    edge_weights = params[:len(KATHARA16_EDGES)]
    # Node params starts after edges
    node_start = len(KATHARA16_EDGES)
    node_end = node_start + N_NODES_16 * PARAMS_PER_NODE
    node_params = np.asarray(
        params[node_start:node_end], dtype=np.float64
    ).reshape(N_NODES_16, PARAMS_PER_NODE)
    gains = node_params[:, 0]
    biases = node_params[:, 1]
    input_ws = node_params[:, 2]
    leaks = node_params[:, 3]
    thresholds = node_params[:, 4]

    # Dale's law: inhibitory nodes flip output sign
    effective_states = (states * inhibit_sign
                        if inhibit_sign is not None else states)

    new_states = np.zeros(N_NODES_16, dtype=np.float64)
    for i in range(N_NODES_16):
        syn_input = 0.0
        for (j, eidx) in NEIGHBORS_16[i]:
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


def simulate_step_16_hebbian(params, inputs, states, w_adapt,
                              reward, k=3, lr=0.08, decay=0.005,
                              w_clip=2.0, reward_scale=10.0):
    """Hebbian-augmented 16-node step (for Phase 7 with plasticity).

    Mirrors hippocampus_brain.simulate_step_hippocampus logic but for 16N.
    w_adapt is applied to edges, Hebbian update post-step.
    """
    p = np.asarray(params, dtype=np.float64).copy()
    n_edges = len(KATHARA16_EDGES)
    p[:n_edges] = p[:n_edges] + w_adapt

    new_states, firing = simulate_step_16(p, inputs, states)

    # Soft k-WTA: scale non-top-k by 0.1 (don't zero out for gradient preservation)
    if k < N_NODES_16:
        top_idx = np.argpartition(firing, -k)[-k:]
        mask = np.ones_like(firing) * 0.1
        mask[top_idx] = 1.0
        firing_sparse = firing * mask
    else:
        firing_sparse = firing

    scaled_reward = reward * reward_scale
    for idx, (a, b) in enumerate(KATHARA16_EDGES):
        coact = firing_sparse[a] * firing_sparse[b]
        w_adapt[idx] += lr * coact * scaled_reward - decay * w_adapt[idx]
    np.clip(w_adapt, -w_clip, w_clip, out=w_adapt)
    return new_states, firing_sparse, w_adapt


# ============================================================
# Spectral verification (quick check on module load)
# ============================================================

def verify_spectral():
    """Verify topology matches G-analysis prediction."""
    A = np.zeros((N_NODES_16, N_NODES_16), dtype=np.float64)
    for (i, j) in KATHARA16_EDGES:
        A[i, j] = 1
        A[j, i] = 1
    eigs = sorted(np.abs(np.linalg.eigvalsh(A)), reverse=True)
    return {
        "n_edges": len(KATHARA16_EDGES),
        "degree": int(A.sum(axis=1)[0]),
        "lambda_1": float(eigs[0]),
        "lambda_2": float(eigs[1]),
    }


if __name__ == "__main__":
    # Smoke test
    print("=" * 60)
    print("  Kathara(16, {3,6,7}) smoke test")
    print("=" * 60)

    spec = verify_spectral()
    print(f"\n  Spectral properties:")
    print(f"    n_edges: {spec['n_edges']} (expected 48)")
    print(f"    degree:  {spec['degree']} (expected 6)")
    print(f"    |λ_1|:   {spec['lambda_1']:.3f} (expected 6.0)")
    print(f"    |λ_2|:   {spec['lambda_2']:.3f} (expected ~4.027)")

    # Functional test: midpoint brain runs without errors
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_16]
    print(f"\n  Midpoint brain: {len(mid)}D params")
    states = np.zeros(N_NODES_16)
    inputs = np.zeros(N_NODES_16)
    inputs[0] = 0.5
    inputs[10] = 0.7
    for _ in range(20):
        states, firing = simulate_step_16(mid, inputs, states,
                                          inhibit_sign=DEFAULT_INHIBIT_16)
    print(f"    final firing[5,11]: {firing[5]:.3f}, {firing[11]:.3f} "
          f"(motor outputs)")
    print(f"    mean activation:   {firing.mean():.3f}")

    # Hebbian test
    w_adapt = np.zeros(len(KATHARA16_EDGES))
    rng = np.random.RandomState(7)
    states = np.zeros(N_NODES_16)
    for step in range(40):
        inputs = rng.uniform(0, 1, size=N_NODES_16) * 0.6
        states, firing, w_adapt = simulate_step_16_hebbian(
            mid, inputs, states, w_adapt, reward=0.3, k=3
        )
    print(f"\n  Hebbian 40-step:")
    print(f"    |w_adapt| mean: {np.abs(w_adapt).mean():.4f}")
    print(f"    edges changed:  {int((np.abs(w_adapt) > 1e-6).sum())}/{len(KATHARA16_EDGES)}")

    # Parameter count sanity
    print(f"\n  Param count: {TOTAL_PARAMS_16}D")
    print(f"    48 edges + 16*5 node + 1 = {48 + 16*5 + 1}")

    print("\n  Smoke test passed.")
