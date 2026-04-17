"""criticality.py — Brain criticality analysis and guard_fn

Biological brains operate at the **edge of chaos** (self-organized criticality):
  - Too ordered -> unable to process information
  - Too chaotic -> unable to maintain information
  - Edge = maximum computational power

Measured via neural avalanches: cluster sizes follow power-law P(s) ~ s^(-alpha)
with alpha ~ 1.5 at critical point (Beggs & Plenz 2003, many replications).

This module provides:
  - compute_criticality_score(firing_history) -> 0-100
  - make_criticality_guard(brain_factory) -> Sentinel-compatible guard_fn
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fit_powerlaw_alpha(cluster_sizes):
    """Estimate power-law exponent alpha from cluster size distribution.

    MLE estimator (Clauset et al. 2009, simple version):
      alpha = 1 + n / sum(log(x_i / x_min))
    """
    sizes = np.array([s for s in cluster_sizes if s >= 2], dtype=np.float64)
    if len(sizes) < 3:
        return None
    x_min = sizes.min()
    n = len(sizes)
    ratios = sizes / x_min
    log_sum = np.sum(np.log(ratios + 1e-12))
    if log_sum <= 0:
        return None
    alpha = 1.0 + n / log_sum
    return float(alpha)


def _avalanche_clusters(firing_sequence, threshold=0.3):
    """Extract avalanche cluster sizes from firing time-series.

    An avalanche = consecutive timesteps where >= 1 neuron fires above threshold.
    Cluster size = total spikes in that avalanche.
    """
    # firing_sequence: (T, N) array of activations
    spikes = firing_sequence > threshold  # (T, N) bool
    total_per_step = spikes.sum(axis=1)  # (T,)

    clusters = []
    current_size = 0
    in_avalanche = False
    for s in total_per_step:
        if s > 0:
            current_size += int(s)
            in_avalanche = True
        else:
            if in_avalanche and current_size > 0:
                clusters.append(current_size)
            current_size = 0
            in_avalanche = False
    if in_avalanche and current_size > 0:
        clusters.append(current_size)
    return clusters


def compute_criticality_score(firing_history, target_alpha=1.5, sigma=0.3):
    """Score a firing history by its distance from critical point.

    firing_history: (T, N) array of per-neuron activations over T timesteps
    Returns: 0-100 (higher = closer to alpha=1.5)
    """
    clusters = _avalanche_clusters(firing_history)
    if len(clusters) < 5:
        return 0.0  # Too few avalanches (brain too quiet)
    alpha = _fit_powerlaw_alpha(clusters)
    if alpha is None:
        return 0.0
    # Gaussian peak at target_alpha
    score = 100.0 * float(np.exp(-((alpha - target_alpha) ** 2) / (2.0 * sigma ** 2)))
    return score


# ============================================================
# Sentinel-compatible guard_fn builders
# ============================================================

def make_criticality_guard_12n(n_episodes=2, n_steps=60):
    """Criticality guard for 12N single-layer brain."""
    from kathara_brain_sim_v8 import FlyWorldV1, simulate_step

    def guard_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=ep * 11 + 7)
            sensors = world.reset()
            states = np.zeros(12)
            history = []
            for _ in range(n_steps):
                states, firing = simulate_step(p, sensors, states)
                history.append(firing.copy())
                nav, cen = float(firing[5]), float(firing[11])
                sensors, _, reached, _ = world.step(nav, cen)
                if reached:
                    break
            if len(history) >= 5:
                firing_arr = np.array(history)
                total += compute_criticality_score(firing_arr)
        return total / n_episodes
    return guard_fn


def make_criticality_guard_cortical(n_episodes=2, n_steps=60):
    """Criticality guard for 5-layer cortical brain. Measures L2 (association)."""
    from cortical_brain import CorticalBrain, N_LAYERS
    from kathara_brain_sim_v8 import FlyWorldV1

    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=ep * 11 + 7)
            sensors = world.reset()
            brain = CorticalBrain(params)
            history = []
            for _ in range(n_steps):
                nav, cen = brain.step(sensors)
                # Record L2 (association) firing
                history.append(brain.states[2].copy())
                sensors, _, reached, _ = world.step(nav, cen)
                if reached:
                    break
            if len(history) >= 5:
                firing_arr = np.array(history)
                total += compute_criticality_score(firing_arr)
        return total / n_episodes
    return guard_fn


if __name__ == "__main__":
    # Smoke test
    print("Criticality module smoke test\n")

    # Test 1: 12N baseline
    from kathara_brain_sim_v8 import PARAM_RANGES as RANGES_12N
    baseline_12n = [(lo + hi) / 2 for lo, hi in RANGES_12N]
    g12 = make_criticality_guard_12n(n_episodes=2, n_steps=30)
    print(f"12N baseline criticality: {g12(baseline_12n):.2f}")

    # Test 2: Cortical baseline
    from cortical_brain import PARAM_RANGES as RANGES_CORT
    baseline_cort = [(lo + hi) / 2 for lo, hi in RANGES_CORT]
    g5 = make_criticality_guard_cortical(n_episodes=2, n_steps=30)
    print(f"Cortical baseline criticality: {g5(baseline_cort):.2f}")

    # Test 3: Simulated power-law history
    rng = np.random.RandomState(42)
    # Synthetic critical activity
    T, N = 200, 12
    history = rng.exponential(0.3, size=(T, N))
    history = np.clip(history, 0, 1)
    score = compute_criticality_score(history)
    alpha = _fit_powerlaw_alpha(_avalanche_clusters(history))
    print(f"\nSynthetic random firing:")
    print(f"  alpha ~= {alpha}")
    print(f"  criticality score: {score:.2f}")
