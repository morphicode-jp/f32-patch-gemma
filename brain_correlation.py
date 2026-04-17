"""brain_correlation.py -- ISS vs FlyWorld behavioral score correlation

Tests the hypothesis: "higher ISS score = better behavior"
by sampling random 12N brain configs and measuring BOTH:
  - ISS-from-firing (structural intelligence)
  - FlyWorldV1 food-reach score (behavioral)

Then computes Pearson correlation. High correlation validates that
compute_iss_from_firing is a meaningful proxy for brain quality.

Usage:
    python brain_correlation.py --n-samples 100
"""
import os
import sys
import time
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v6 import compute_iss_from_firing
from kathara_brain_sim_v8 import (
    FlyWorldV1,
    simulate_step,
    PARAM_RANGES,
)


def measure_iss(params, n_steps=30, seed=7):
    """ISS-from-firing score on a probe episode (no movement)."""
    p = np.array(params, dtype=np.float64)
    world = FlyWorldV1(seed=seed)
    sensors = world.reset()
    firing_accum = np.zeros(12)
    states = np.zeros(12)
    for _ in range(n_steps):
        states, firing = simulate_step(p, sensors, states)
        firing_accum += firing
        nav, cen = float(firing[5]), float(firing[11])
        sensors, _, reached, _ = world.step(nav, cen)
        if reached:
            break
    return float(compute_iss_from_firing(firing_accum / n_steps, p))


def measure_behavioral(params, n_episodes=4, n_steps=60):
    """FlyWorldV1 food-reach score (0-100). Same formula as v6 eval_fly."""
    p = np.array(params, dtype=np.float64)
    total = 0.0
    for ep in range(n_episodes):
        world = FlyWorldV1(seed=ep * 7 + 13)
        sensors = world.reset()
        initial_dist = world.get_food_dist()
        min_dist = initial_dist
        reached = False
        move_sum = 0.0
        states = np.zeros(12)
        for _ in range(n_steps):
            prev_pos = world.fly_pos.copy()
            states, firing = simulate_step(p, sensors, states)
            nav, cen = float(firing[5]), float(firing[11])
            sensors, dist, reached_now, _ = world.step(nav, cen)
            move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
            if dist < min_dist:
                min_dist = dist
            if reached_now:
                reached = True
                break
        approach = max(0.0, initial_dist - min_dist) / (initial_dist + 1e-6)
        ep_score = approach * 60.0 + (30.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
        total += ep_score
    return total / n_episodes


def pearson(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    mx, my = xs.mean(), ys.mean()
    num = ((xs - mx) * (ys - my)).sum()
    den = np.sqrt(((xs - mx) ** 2).sum() * ((ys - my) ** 2).sum())
    return num / den if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    print(f"\n{'=' * 60}")
    print(f"  ISS vs Behavioral Correlation Test ({args.n_samples} random brains)")
    print(f"{'=' * 60}\n")

    iss_scores = []
    bhv_scores = []

    t0 = time.time()
    for i in range(args.n_samples):
        params = [rng.uniform(lo, hi) for lo, hi in PARAM_RANGES]
        iss = measure_iss(params)
        bhv = measure_behavioral(params)
        iss_scores.append(iss)
        bhv_scores.append(bhv)
        if (i + 1) % 20 == 0:
            r_so_far = pearson(iss_scores, bhv_scores)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{args.n_samples}] elapsed={elapsed:.1f}s, "
                  f"r_so_far={r_so_far:.3f}")

    elapsed = time.time() - t0
    r = pearson(iss_scores, bhv_scores)

    iss_arr = np.array(iss_scores)
    bhv_arr = np.array(bhv_scores)

    print(f"\n{'=' * 60}")
    print(f"  Results ({args.n_samples} samples, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Pearson r(ISS, behavior) = {r:.3f}")
    print(f"  ISS  range: {iss_arr.min():.1f} - {iss_arr.max():.1f}, "
          f"mean={iss_arr.mean():.1f}, std={iss_arr.std():.1f}")
    print(f"  Bhv  range: {bhv_arr.min():.1f} - {bhv_arr.max():.1f}, "
          f"mean={bhv_arr.mean():.1f}, std={bhv_arr.std():.1f}")

    if abs(r) >= 0.7:
        verdict = "STRONG correlation -- ISS is a reliable behavioral proxy"
    elif abs(r) >= 0.3:
        verdict = "MODERATE correlation -- ISS predicts behavior somewhat"
    else:
        verdict = "WEAK/NONE -- ISS does not capture behavioral quality"
    print(f"\n  Verdict: {verdict}")

    # Top-10 by ISS: do they also have top behavior?
    idx_by_iss = np.argsort(iss_arr)[::-1]
    top10_iss_behav = [bhv_scores[i] for i in idx_by_iss[:10]]
    bot10_iss_behav = [bhv_scores[i] for i in idx_by_iss[-10:]]
    print(f"\n  Top-10 ISS: behavior mean = {np.mean(top10_iss_behav):.1f}")
    print(f"  Bot-10 ISS: behavior mean = {np.mean(bot10_iss_behav):.1f}")
    print(f"  Gap:        {np.mean(top10_iss_behav) - np.mean(bot10_iss_behav):+.1f}")

    # Save
    with open("brain_correlation_result.json", "w") as f:
        json.dump({
            "n_samples": args.n_samples,
            "pearson_r": round(r, 4),
            "elapsed_s": round(elapsed, 1),
            "iss_range": [round(float(iss_arr.min()), 2), round(float(iss_arr.max()), 2)],
            "bhv_range": [round(float(bhv_arr.min()), 2), round(float(bhv_arr.max()), 2)],
            "top10_iss_behavior_mean": round(float(np.mean(top10_iss_behav)), 2),
            "bot10_iss_behavior_mean": round(float(np.mean(bot10_iss_behav)), 2),
            "verdict": verdict,
            "samples": [
                {"iss": round(iss, 2), "bhv": round(bhv, 2)}
                for iss, bhv in zip(iss_scores, bhv_scores)
            ],
        }, f, indent=2)
    print("\n  Saved: brain_correlation_result.json")


if __name__ == "__main__":
    main()
