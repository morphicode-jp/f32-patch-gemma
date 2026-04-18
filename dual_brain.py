"""dual_brain.py - Main Kathara + Hippocampus coupled system.

Architecture:

    sensors (12D) -> Main brain (12N Kathara, fixed weights) -> nav, cen
                                   ^
                          sensor[0] = hippo recall
                                   |
                 -> Hippocampus (12N + k-WTA + 3-factor plasticity)
                           |
                           w_adapt persists across episodes with decay

Main and hippocampus are SEPARATE Kathara brains with identical topology
but independent parameters. They communicate via:
  - Forward: both receive same sensor input
  - Backward: hippo firing summary projects to main sensor[0]
"""
import numpy as np
from kathara_brain_sim_v8 import simulate_step, PARAM_RANGES as KATHARA_PR
from hippocampus_brain import simulate_step_hippocampus

N_MAIN = 91
N_HIPPO = 91
N_COUPLING = 4  # m_to_h, h_to_m, k_wta, decay
DUAL_TOTAL = N_MAIN + N_HIPPO + N_COUPLING  # = 186

# DUAL PARAM_RANGES:
#   [0:91]    main brain (Kathara 91D)
#   [91:182]  hippocampus (Kathara 91D)
#   [182]     main→hippo gain
#   [183]     hippo→main gain
#   [184]     k-WTA k (continuous, mapped to int [2, 5])
#   [185]     persistence decay [0.85, 0.99]
DUAL_PARAM_RANGES = (
    list(KATHARA_PR) +
    list(KATHARA_PR) +
    [(0.3, 2.0),    # main→hippo gain (floor 0.3 ensures hippo receives signal)
     (0.3, 3.0),    # hippo→main gain (floor 0.3 forces main to use recall)
     (2.0, 5.0),    # k-WTA k (rounds to int)
     (0.85, 0.99)]  # persistence decay
)

DUAL_PARAM_NAMES = (
    [f"main_{i}" for i in range(N_MAIN)] +
    [f"hippo_{i}" for i in range(N_HIPPO)] +
    ["m_to_h_gain", "h_to_m_gain", "k_wta", "persist_decay"]
)


def split_params(dual_params):
    """Split 186D dual params into components."""
    p = np.asarray(dual_params, dtype=np.float64)
    assert len(p) == DUAL_TOTAL, f"expected {DUAL_TOTAL} params, got {len(p)}"
    main_p = p[0:N_MAIN]
    hippo_p = p[N_MAIN:N_MAIN + N_HIPPO]
    m_to_h = float(p[N_MAIN + N_HIPPO])
    h_to_m = float(p[N_MAIN + N_HIPPO + 1])
    k_raw = float(p[N_MAIN + N_HIPPO + 2])
    k = max(2, min(5, int(round(k_raw))))
    decay = float(p[N_MAIN + N_HIPPO + 3])
    return main_p, hippo_p, m_to_h, h_to_m, k, decay


def dual_step(main_p, hippo_p, sensors, main_states, hippo_states,
              w_adapt_hippo, reward, m_to_h, h_to_m, k):
    """One step of dual brain system.

    Architectural detail (v2): hippo reinstates sensory patterns.
      Real hippocampus projects back to entorhinal cortex, which then
      re-excites the same cortical patterns that originally encoded
      the memory. We mirror this: hippo_firing is MIXED element-wise
      into main brain's sensor input using h_to_m as gate strength.

      main_input[i] = sensors[i] * (1 - gate) + hippo_firing[i] * gate
      where gate = sigmoid(h_to_m - 1.5)  (0..1, centered at h_to_m=1.5)

    When gate=0 (h_to_m small): main ignores hippo (Phase 5b behavior)
    When gate=1 (h_to_m large): main sees hippo pattern instead of sensors

    This way hippo "fills in" missing sensor information when memory recall
    is strong, rather than corrupting the existing sensor channel.
    """
    sensors_arr = np.asarray(sensors, dtype=np.float64)

    # Hippocampus pass: takes sensors scaled by m_to_h
    hippo_input = sensors_arr * m_to_h
    hippo_states_new, hippo_firing_sparse, w_adapt_hippo = simulate_step_hippocampus(
        hippo_p, hippo_input, hippo_states, w_adapt_hippo,
        reward=reward, k=k
    )

    # Main brain pass: sensor-hippo MIX via sigmoid gate
    # h_to_m in [0.3, 3.0] -> gate in ~[0.23, 0.82]
    gate = 1.0 / (1.0 + np.exp(-(h_to_m - 1.5)))
    main_input = sensors_arr * (1.0 - gate) + hippo_firing_sparse * gate
    main_states_new, main_firing = simulate_step(main_p, main_input, main_states)

    return (main_states_new, hippo_states_new, w_adapt_hippo,
            main_firing, hippo_firing_sparse)


if __name__ == "__main__":
    # Smoke test: run dual_step 20 times with midpoint params
    print("=" * 60)
    print("  dual_brain.py smoke test")
    print("=" * 60)

    mid = [(lo + hi) / 2 for lo, hi in DUAL_PARAM_RANGES]
    main_p, hippo_p, m_to_h, h_to_m, k, decay = split_params(mid)
    print(f"\n  Total params: {len(mid)} (expected {DUAL_TOTAL})")
    print(f"  Coupling: m_to_h={m_to_h:.2f}  h_to_m={h_to_m:.2f}  "
          f"k={k}  decay={decay:.2f}")

    sensors = np.zeros(12); sensors[7] = 0.5; sensors[10] = 0.7
    main_states = np.zeros(12); hippo_states = np.zeros(12)
    w_adapt = np.zeros(30)

    for step in range(20):
        reward = 0.1 + 0.02 * step
        main_states, hippo_states, w_adapt, main_firing, hippo_firing = dual_step(
            main_p, hippo_p, sensors, main_states, hippo_states,
            w_adapt, reward, m_to_h, h_to_m, k
        )

    print(f"\n  After 20 steps:")
    print(f"    main firing[5,11]: {main_firing[5]:.3f}, {main_firing[11]:.3f} (nav, cen)")
    print(f"    hippo active cells: {int((hippo_firing > 0).sum())} (k={k})")
    print(f"    w_adapt |mean|: {np.abs(w_adapt).mean():.4f}  "
          f"edges changed: {int((np.abs(w_adapt) > 1e-6).sum())}/30")

    # Test: persistence decay across "episode boundary"
    w_adapt_decayed = w_adapt * decay
    preserved = float(np.abs(w_adapt_decayed).sum() / max(np.abs(w_adapt).sum(), 1e-6))
    print(f"\n  After decay ({decay:.2f}): {preserved*100:.1f}% of w_adapt preserved")

    assert len(mid) == DUAL_TOTAL
    assert main_firing.shape == (12,)
    assert hippo_firing.shape == (12,)
    assert w_adapt.shape == (30,)
    print("\n  Smoke test passed.")
