"""cortical_generalization.py — Overfit check for cortical brain.

Same methodology as brain_generalization.py but for 5-layer cortical brain.
Compares training seeds vs unseen seeds.

Usage:
    python cortical_generalization.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical_brain import CorticalBrain
from kathara_brain_sim_v8 import FlyWorldV3


def run_single_episode(params, seed, n_steps=60, has_predator=False):
    """Run one episode with given seed. Returns behavioral score."""
    world = FlyWorldV3(seed=seed, has_predator=has_predator)
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
    ep_score = approach * 60.0 + (30.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
    if caught:
        ep_score *= 0.3
    return ep_score


def main():
    result_file = "cortical_sentinel_result.json"
    if not os.path.exists(result_file):
        print(f"ERROR: {result_file} not found. Run cortical_sentinel_run.py first.")
        return

    with open(result_file) as f:
        data = json.load(f)

    # Prefer best_ever_params if available
    params = data.get("best_ever_params") or data["best_params"]
    saved_eval = data.get("best_ever_score", data["eval_score"])
    print(f"Loaded from {result_file}")
    print(f"  Saved eval (training):  {saved_eval:.2f}")
    print(f"  Architecture: {data.get('architecture')}")
    print(f"  Params: {len(params)}D")

    # Training seeds
    train_seeds = [ep * 7 + 13 for ep in range(4)]
    rng = np.random.RandomState(999)
    unseen_seeds = [int(s) for s in rng.randint(1000, 10000, size=20)]
    unseen_seeds = [s for s in unseen_seeds if s not in train_seeds][:20]

    # Training seeds eval
    print("\n--- TRAINING seeds (4 ep) ---")
    train_scores = [run_single_episode(params, s) for s in train_seeds]
    train_mean = float(np.mean(train_scores))
    print(f"  Scores: {[round(s, 1) for s in train_scores]}")
    print(f"  Mean:   {train_mean:.2f}")

    # Unseen seeds eval
    print("\n--- UNSEEN seeds (20 ep) ---")
    unseen_scores = [run_single_episode(params, s) for s in unseen_seeds]
    unseen_mean = float(np.mean(unseen_scores))
    unseen_std = float(np.std(unseen_scores))
    reach_rate = sum(1 for s in unseen_scores if s >= 30) / len(unseen_scores)
    print(f"  Mean: {unseen_mean:.2f}  std={unseen_std:.2f}")
    print(f"  Min:  {min(unseen_scores):.2f}")
    print(f"  Max:  {max(unseen_scores):.2f}")
    print(f"  Reach rate: {reach_rate * 100:.0f}%")

    gap = train_mean - unseen_mean
    gap_pct = gap / max(train_mean, 1) * 100
    print(f"\n{'='*60}")
    print(f"  Cortical generalization verdict")
    print(f"{'='*60}")
    print(f"  Training mean: {train_mean:.2f}")
    print(f"  Unseen mean:   {unseen_mean:.2f}")
    print(f"  Gap:           {gap:+.2f}  ({gap_pct:+.1f}%)")

    if abs(gap_pct) < 10:
        verdict = "STRONG generalization -- cortical structure robust"
    elif gap_pct < 30:
        verdict = "MODERATE generalization"
    elif gap_pct < 50:
        verdict = "WEAK generalization"
    else:
        verdict = "OVERFIT"
    print(f"  Verdict: {verdict}")

    # Compare to 12N baseline
    baseline_file = "brain_generalization_result.json"
    if os.path.exists(baseline_file):
        with open(baseline_file) as f:
            baseline = json.load(f)
        print(f"\n{'='*60}")
        print(f"  VS 12N single-layer baseline")
        print(f"{'='*60}")
        print(f"  {'':20s}  {'12N':>8s}  {'Cortical':>8s}")
        print(f"  {'Training mean':20s}  {baseline['train_mean']:>8.2f}  {train_mean:>8.2f}")
        print(f"  {'Unseen mean':20s}  {baseline['unseen_mean']:>8.2f}  {unseen_mean:>8.2f}")
        print(f"  {'Gap %':20s}  {baseline['gap_pct']:>7.1f}%  {gap_pct:>7.1f}%")
        print(f"  {'Unseen reach rate':20s}  {baseline['unseen_reach_rate']*100:>7.0f}%  {reach_rate*100:>7.0f}%")

    # Save
    out = {
        "architecture": data.get("architecture", "cortical"),
        "params_source": result_file,
        "source_eval_score": saved_eval,
        "train_seeds": train_seeds,
        "train_scores": [round(s, 2) for s in train_scores],
        "train_mean": round(train_mean, 2),
        "unseen_seeds": unseen_seeds,
        "unseen_scores": [round(s, 2) for s in unseen_scores],
        "unseen_mean": round(unseen_mean, 2),
        "unseen_std": round(unseen_std, 2),
        "unseen_reach_rate": round(reach_rate, 3),
        "gap": round(gap, 2),
        "gap_pct": round(gap_pct, 1),
        "verdict": verdict,
    }
    with open("cortical_generalization_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: cortical_generalization_result.json")


if __name__ == "__main__":
    main()
