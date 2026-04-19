"""kathara_dense_brain.py - 16-node brain with ALL C(16,2)=120 edges.

Biological motivation:
  Infant brain has ~1000T synapses; adult ~500T (after pruning).
  Start with overcapacity → select what works via activity/reward.
  Our current Kathara(16, {3,6,7}) has only 48 edges — may be too
  sparse to find navigation-capable configurations.

Implementation:
  Every pair (i, j) with i < j is an edge. 120 edges total.
  Parameter count: 120 + 16*5 + 1 = 201D (vs 129D for sparse).
  Structural/sensor role assignment identical to Kathara(16).

Pruning workflow (in separate script):
  1. Train full dense brain with Sentinel
  2. Identify edges where |weight| < prune_threshold
  3. Mask those edges (set to 0, freeze)
  4. Retrain remaining edges
  → functional subset emerges, like synaptic pruning
"""
import numpy as np


N_NODES_DENSE = 16
DENSE_EDGES = [(i, j) for i in range(N_NODES_DENSE)
               for j in range(i + 1, N_NODES_DENSE)]
assert len(DENSE_EDGES) == 120

NEIGHBORS_DENSE = [[] for _ in range(N_NODES_DENSE)]
for idx, (a, b) in enumerate(DENSE_EDGES):
    NEIGHBORS_DENSE[a].append((b, idx))
    NEIGHBORS_DENSE[b].append((a, idx))

# Same sensor/motor role as Kathara(16)
# Input: 0, 2, 10 (food), 3 (proximity), 4-9, 11-13 (ToM)
# Motor: 5 (nav), 11 (speed), 1 (voice)
# Dale's law: nodes 0, 4, 8 inhibitory (default matches Kathara(16))
_INHIBIT_NODES_DENSE = [0, 4, 8]

PARAMS_PER_NODE = 5
TOTAL_PARAMS_DENSE = len(DENSE_EDGES) + N_NODES_DENSE * PARAMS_PER_NODE + 1
# = 120 + 80 + 1 = 201

PARAM_RANGES_DENSE = (
    [(-3.0, 3.0)] * len(DENSE_EDGES) +
    [(0.1, 5.0), (-3.0, 3.0), (0.1, 5.0), (0.01, 0.9), (0.01, 1.0)]
    * N_NODES_DENSE +
    [(10, 80)]
)
assert len(PARAM_RANGES_DENSE) == TOTAL_PARAMS_DENSE

PARAM_NAMES_DENSE = []
for (a, b) in DENSE_EDGES:
    PARAM_NAMES_DENSE.append(f"e{a}_{b}")
for i in range(N_NODES_DENSE):
    for pname in ["gain", "bias", "iw", "leak", "thr"]:
        PARAM_NAMES_DENSE.append(f"n{i}_{pname}")
PARAM_NAMES_DENSE.append("n_steps")


def make_inhibit_sign_dense():
    sign = np.ones(N_NODES_DENSE, dtype=np.float64)
    for n in _INHIBIT_NODES_DENSE:
        sign[n] = -1.0
    return sign


DEFAULT_INHIBIT_DENSE = make_inhibit_sign_dense()


def simulate_step_dense(params, inputs, states, inhibit_sign=None,
                        edge_mask=None):
    """Forward pass on dense 16-node brain.

    edge_mask: optional 120-element array of 0/1 to disable specific edges
               (post-pruning stage 2 uses this).
    """
    n_edges = len(DENSE_EDGES)
    edge_weights = np.asarray(params[:n_edges], dtype=np.float64)
    if edge_mask is not None:
        edge_weights = edge_weights * edge_mask

    node_start = n_edges
    node_end = node_start + N_NODES_DENSE * PARAMS_PER_NODE
    node_params = np.asarray(
        params[node_start:node_end], dtype=np.float64
    ).reshape(N_NODES_DENSE, PARAMS_PER_NODE)
    gains = node_params[:, 0]
    biases = node_params[:, 1]
    input_ws = node_params[:, 2]
    leaks = node_params[:, 3]
    thresholds = node_params[:, 4]

    effective_states = (states * inhibit_sign
                        if inhibit_sign is not None else states)

    new_states = np.zeros(N_NODES_DENSE, dtype=np.float64)
    for i in range(N_NODES_DENSE):
        syn_input = 0.0
        for (j, eidx) in NEIGHBORS_DENSE[i]:
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


def compute_edge_mask_from_magnitudes(params, prune_fraction=0.5):
    """Identify bottom `prune_fraction` of edges by |weight|. Return mask array
    where 1 = keep, 0 = prune.

    This is our "synaptic pruning" operator — biologically analogous to
    activity-based pruning but applied via weight magnitude.
    """
    n_edges = len(DENSE_EDGES)
    edge_weights = np.abs(np.asarray(params[:n_edges], dtype=np.float64))
    threshold = np.percentile(edge_weights, prune_fraction * 100)
    mask = (edge_weights >= threshold).astype(np.float64)
    return mask


if __name__ == "__main__":
    print("=" * 60)
    print("  Dense Kathara (16-node, all edges)")
    print("=" * 60)
    print(f"  n_edges: {len(DENSE_EDGES)} (vs 48 for sparse Kathara(16))")
    print(f"  total params: {TOTAL_PARAMS_DENSE} (vs 129 sparse)")
    print(f"  each node has degree: {N_NODES_DENSE - 1} (vs 6 sparse)")

    # Spectral
    A = np.zeros((N_NODES_DENSE, N_NODES_DENSE))
    for i, j in DENSE_EDGES:
        A[i, j] = 1; A[j, i] = 1
    eigs = sorted(np.abs(np.linalg.eigvalsh(A)), reverse=True)
    print(f"  |lambda_1|: {eigs[0]:.1f} (expected 15 for K_16)")
    print(f"  |lambda_2|: {eigs[1]:.3f}")

    # Functional test
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_DENSE]
    states = np.zeros(N_NODES_DENSE)
    inputs = np.zeros(N_NODES_DENSE)
    inputs[0] = 0.5
    inputs[10] = 0.7
    import time
    t0 = time.time()
    for _ in range(20):
        states, firing = simulate_step_dense(mid, inputs, states,
                                              inhibit_sign=DEFAULT_INHIBIT_DENSE)
    print(f"\n  20 steps elapsed: {(time.time()-t0)*1000:.1f}ms "
          f"(vs ~2ms for sparse)")
    print(f"  Final firing[5, 11, 1]: {firing[5]:.3f} {firing[11]:.3f} {firing[1]:.3f}")

    # Pruning demo
    rng = np.random.RandomState(7)
    test_params = list(mid)
    for i in range(len(DENSE_EDGES)):
        test_params[i] = rng.uniform(-3, 3)
    mask = compute_edge_mask_from_magnitudes(test_params, prune_fraction=0.6)
    print(f"\n  Pruning demo: keep top 40% (48 edges out of 120)")
    print(f"    edges kept: {int(mask.sum())}")
    print(f"    edges pruned: {120 - int(mask.sum())}")

    print("\n  Smoke test passed.")
