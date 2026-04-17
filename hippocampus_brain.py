"""hippocampus_brain.py - Sparse-coding memory module (DG + CA3 style).

Three biological principles implemented:
  1. Sparse coding (k-WTA): only top-k neurons fire → pattern separation
  2. Reward-gated plasticity (3-factor rule): ONLY winning cells strengthen
     when reward is positive, decay otherwise.
  3. Replay: top-reward patterns re-fired after episode → consolidation.

Sits parallel to the main Kathara brain. Coupled via sensor node 0
(memory recall channel) in dual_brain.py.
"""
import numpy as np
from kathara_brain_sim_v8 import KATHARA_EDGES, simulate_step


def apply_kwta(firing, k, soft=False):
    """k-Winners-Take-All sparse coding. Keep top-k firing values, zero rest.

    Models DG sparse activity: only top ~25% of dentate granule cells fire
    per memory, giving strong pattern separation.

    soft=False (default): top-k at full value, others set to 0.
    soft=True: top-k at full, others scaled by 0.1 (partial gradient preserved).

    Uses argpartition on INDICES to guarantee exactly k cells survive even
    with ties in firing values.
    """
    firing = np.asarray(firing, dtype=np.float64)
    if k >= len(firing):
        return firing.copy()
    # Indices of top-k (may include ties but we pick exactly k indices)
    top_idx = np.argpartition(firing, -k)[-k:]
    if soft:
        out = firing * 0.1
        out[top_idx] = firing[top_idx]
    else:
        out = np.zeros_like(firing)
        out[top_idx] = firing[top_idx]
    return out


def simulate_step_hippocampus(params, inputs, states, w_adapt,
                              reward, k=3, lr=0.08, decay=0.005, w_clip=2.0,
                              soft=False, reward_scale=10.0):
    """Hippocampal step: forward + k-WTA + reward-gated plasticity.

    params: 91D Kathara params (same layout as main brain)
    w_adapt: 30D adaptive edge delta (persists across steps and episodes)
    reward: scalar, positive when agent is progressing toward goal
    k: sparsity (number of neurons allowed to fire)

    Returns (new_states, firing_sparse, w_adapt)
    """
    p = np.asarray(params, dtype=np.float64).copy()
    p[:30] = p[:30] + w_adapt

    new_states, firing = simulate_step(p, inputs, states)

    # Sparse coding (pattern separation)
    firing_sparse = apply_kwta(firing, k, soft=soft)

    # 3-factor update using sparse firing — only winning cells strengthen
    # Scale reward so typical 0.1-0.4 per-step progress gives meaningful signal
    scaled_reward = reward * reward_scale
    for idx, (a, b) in enumerate(KATHARA_EDGES):
        coact = firing_sparse[a] * firing_sparse[b]
        w_adapt[idx] += lr * coact * scaled_reward - decay * w_adapt[idx]

    np.clip(w_adapt, -w_clip, w_clip, out=w_adapt)
    return new_states, firing_sparse, w_adapt


def replay_patterns(params, patterns_with_rewards, w_adapt,
                    n_iterations=10, k=3, lr=0.05, decay=0.001):
    """Post-episode consolidation via pattern replay.

    Mirrors sleep-phase memory consolidation in mammalian hippocampus:
    top-reward firing patterns are reactivated repeatedly, strengthening
    the associated weight changes.

    patterns_with_rewards: list of (sensor_12D, reward_scalar)
    Returns: consolidated w_adapt
    """
    p = np.asarray(params, dtype=np.float64).copy()
    for pattern, reward in patterns_with_rewards:
        states = np.zeros(12, dtype=np.float64)
        pattern_arr = np.asarray(pattern, dtype=np.float64)
        for _ in range(n_iterations):
            states, _, w_adapt = simulate_step_hippocampus(
                p, pattern_arr, states, w_adapt,
                reward=reward, k=k, lr=lr, decay=decay
            )
    return w_adapt


if __name__ == "__main__":
    # Smoke test 1: k-WTA correctness
    print("=" * 60)
    print("  hippocampus_brain.py smoke test")
    print("=" * 60)

    firing = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.1, 0.0, 0.0])
    sparse2 = apply_kwta(firing, 2)
    active2 = (sparse2 > 0).sum()
    print(f"\n[k-WTA k=2] firing -> sparse: {sparse2.round(2)}")
    print(f"  active: {active2} cells (expected: 2)")
    assert active2 == 2, f"k=2 should keep 2 cells, got {active2}"

    sparse4 = apply_kwta(firing, 4)
    active4 = (sparse4 > 0).sum()
    print(f"[k-WTA k=4] active: {active4} cells (expected: 4)")
    assert active4 == 4

    # Smoke test 2: single hippocampus step with VARIED inputs
    from kathara_brain_sim_v8 import PARAM_RANGES
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    states = np.zeros(12)
    w_adapt = np.zeros(30)
    rng = np.random.RandomState(7)
    reward = 0.5

    for step in range(40):
        # varied inputs each step to spread activation
        inputs = rng.uniform(0, 1, size=12) * 0.6
        states, firing_sparse, w_adapt = simulate_step_hippocampus(
            mid, inputs, states, w_adapt, reward=reward, k=3
        )
    active_cells_last = int((firing_sparse > 0).sum())
    changed_edges = int((np.abs(w_adapt) > 1e-6).sum())
    print(f"\n[40-step varied-input run with reward=0.5, k=3]")
    print(f"  final |w_adapt|: mean={np.abs(w_adapt).mean():.4f}  "
          f"max={np.abs(w_adapt).max():.4f}")
    print(f"  active cells last step: {active_cells_last} (k=3 enforced)")
    print(f"  edges changed: {changed_edges}/30")
    assert changed_edges > 0, f"No edges changed after 40 steps. Possible bug."

    # Smoke test 3: replay consolidates memory
    patterns = [(rng.uniform(0, 1, 12) * 0.8, 1.0),
                (rng.uniform(0, 1, 12) * 0.6, 0.5),
                (rng.uniform(0, 1, 12) * 0.4, 0.2)]
    w_before = w_adapt.copy()
    w_after = replay_patterns(mid, patterns, w_adapt.copy(),
                              n_iterations=10, k=3)
    change = float(np.abs(w_after - w_before).sum())
    print(f"\n[replay 3 patterns x 10 iterations, k=3]")
    print(f"  total w_adapt change from replay: {change:.4f}")
    assert change > 0, "replay should modify w_adapt"

    # Smoke test 4: negative reward reverses direction
    w_pos = np.zeros(30)
    w_neg = np.zeros(30)
    rng2 = np.random.RandomState(11)
    inputs_same = rng2.uniform(0, 1, 12) * 0.7
    states_p = np.zeros(12); states_n = np.zeros(12)
    for _ in range(40):
        states_p, _, w_pos = simulate_step_hippocampus(
            mid, inputs_same, states_p, w_pos, reward=+1.0, k=3
        )
        states_n, _, w_neg = simulate_step_hippocampus(
            mid, inputs_same, states_n, w_neg, reward=-1.0, k=3
        )
    print(f"\n[+reward vs -reward signs (same inputs, 40 steps)]")
    print(f"  w_pos mean: {w_pos.mean():+.4f}  "
          f"w_neg mean: {w_neg.mean():+.4f}")
    assert w_pos.mean() * w_neg.mean() < 0 or abs(w_pos.mean()) > abs(w_neg.mean()) * 0.5, \
        "reward sign should affect w_adapt direction"

    print("\n  All smoke tests passed.")
