"""hebbian_brain.py - Kathara brain with online Hebbian plasticity.

Extension of kathara_brain_sim_v8:
  Each simulation step also UPDATES edge weights based on co-firing.
    Δw_edge = lr * (firing_a * firing_b - decay * w_edge)

Result: neurons that fire together strengthen their connection.
Enables within-episode and across-episode memory traces.

API:
  simulate_step_hebbian(params, inputs, states, w_adapt, lr=0.02, decay=0.01)
    Returns (new_states, firing, new_w_adapt)

  w_adapt starts at zeros (np.zeros(30))
  Effective edge weight at step t: params[:30] + w_adapt
"""
import numpy as np
from kathara_brain_sim_v8 import (
    KATHARA_EDGES, NEIGHBORS, PARAMS_PER_NODE, simulate_step
)


def simulate_step_hebbian(params, inputs, states, w_adapt,
                          lr=0.02, decay=0.01, inhibit_sign=None,
                          w_clip=3.0):
    """Forward pass + Hebbian update on edge weights.

    params: full 91-param array (edges [0:30] are base weights)
    w_adapt: 30-element array of adaptive weight deltas
    Effective edge weight = params[0:30] + w_adapt
    """
    p = np.asarray(params, dtype=np.float64).copy()
    # Apply adaptive weights for this step's forward pass
    p[:30] = p[:30] + w_adapt

    new_states, firing = simulate_step(p, inputs, states, inhibit_sign=inhibit_sign)

    # Hebbian update based on co-firing (post-step firing values)
    for idx, (a, b) in enumerate(KATHARA_EDGES):
        coact = firing[a] * firing[b]
        # dw = lr * coact - decay * current_adapt
        # Centered around zero so it only strengthens when active
        w_adapt[idx] += lr * (coact - 0.3) - decay * w_adapt[idx]

    # Clip to keep stability
    np.clip(w_adapt, -w_clip, w_clip, out=w_adapt)
    return new_states, firing, w_adapt


if __name__ == "__main__":
    # Smoke test: does Hebbian update produce weight changes?
    from kathara_brain_sim_v8 import PARAM_RANGES
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    mid = np.array(mid, dtype=np.float64)

    states = np.zeros(12)
    w_adapt = np.zeros(30)
    inputs = np.zeros(12)
    inputs[7] = 0.5
    inputs[10] = 0.7
    for step in range(20):
        states, firing, w_adapt = simulate_step_hebbian(mid, inputs, states, w_adapt)

    print(f"After 20 steps:")
    print(f"  w_adapt: min={w_adapt.min():.3f}  max={w_adapt.max():.3f}  "
          f"mean={w_adapt.mean():.3f}  std={w_adapt.std():.3f}")
    print(f"  active edges (|w_adapt| > 0.01): {(np.abs(w_adapt) > 0.01).sum()}/30")
    print(f"  firing final: {firing[:5].round(2)}")
