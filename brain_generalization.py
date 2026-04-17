"""brain_generalization.py — Overfit check for Sentinel-optimized brain.

Compares behavioral score on:
  - TRAINING seeds (used during Sentinel run: offset=13, episodes 0-3)
  - UNSEEN seeds (random seeds not in training)

If training_score >> unseen_score: overfit (memorized specific worlds)
If training_score ≈ unseen_score: true generalization

Uses best_params from brain_sentinel_biological_result.json.

Usage:
    python brain_generalization.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import FlyWorldV3, simulate_step


def run_episodes(params, seeds, n_steps=60, has_predator=False):
    """Run FlyWorldV3 episodes with specific seeds. Returns list of episode scores."""
    p = np.array(params, dtype=np.float64)
    scores = []
    for seed in seeds:
        world = FlyWorldV3(seed=seed, has_predator=has_predator)
        sensors = world.reset()
        initial_dist = world.get_food_dist()
        min_dist = initial_dist
        reached = False
        caught = False
        move_sum = 0.0
        states = np.zeros(12, dtype=np.float64)

        for _ in range(n_steps):
            prev_pos = world.fly_pos.copy()
            states, firing = simulate_step(p, sensors, states)
            nav = float(firing[5])
            cen = float(firing[11])
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
        scores.append(ep_score)
    return scores


def main():
    # Load the best params from Sentinel
    result_file = "brain_sentinel_biological_result.json"
    if not os.path.exists(result_file):
        print(f"ERROR: {result_file} not found")
        return

    with open(result_file) as f:
        data = json.load(f)
    params = data["best_params"]
    saved_eval = data["eval_score"]
    print(f"Loaded params from {result_file}")
    print(f"  Saved eval_score (training): {saved_eval:.2f}")
    print(f"  Params: {params[:5]}...  (len={len(params)})")

    # Training seeds (from make_flyworld_eval with seed_offset=13, n_episodes=4)
    # Formula: ep * 7 + 13  for ep in range(n_episodes)
    train_seeds = [ep * 7 + 13 for ep in range(4)]  # [13, 20, 27, 34]

    # Unseen seeds: deliberately avoid the training pattern
    rng = np.random.RandomState(999)
    unseen_seeds = [int(s) for s in rng.randint(1000, 10000, size=20)]
    # Ensure no overlap
    unseen_seeds = [s for s in unseen_seeds if s not in train_seeds][:20]

    print(f"\nTraining seeds ({len(train_seeds)}): {train_seeds}")
    print(f"Unseen seeds  ({len(unseen_seeds)}): {unseen_seeds[:5]}...\n")

    # Training seeds evaluation
    print("--- TRAINING seeds (FlyWorldV3, 4 episodes) ---")
    train_scores = run_episodes(params, train_seeds)
    train_mean = float(np.mean(train_scores))
    print(f"  Scores: {[round(s, 1) for s in train_scores]}")
    print(f"  Mean:   {train_mean:.2f}")

    # Unseen seeds evaluation
    print("\n--- UNSEEN seeds (FlyWorldV3, 20 episodes) ---")
    unseen_scores = run_episodes(params, unseen_seeds)
    unseen_mean = float(np.mean(unseen_scores))
    unseen_std = float(np.std(unseen_scores))
    print(f"  Mean: {unseen_mean:.2f}  std={unseen_std:.2f}")
    print(f"  Min:  {min(unseen_scores):.2f}")
    print(f"  Max:  {max(unseen_scores):.2f}")
    print(f"  Reach rate: {sum(1 for s in unseen_scores if s >= 30) / len(unseen_scores) * 100:.0f}%"
          f" (score >= 30 means reached food)")

    # Comparison
    print(f"\n{'='*60}")
    print(f"  Generalization verdict")
    print(f"{'='*60}")
    gap = train_mean - unseen_mean
    gap_pct = gap / max(train_mean, 1) * 100
    print(f"  Training mean: {train_mean:.2f}")
    print(f"  Unseen mean:   {unseen_mean:.2f}")
    print(f"  Gap:           {gap:+.2f}  ({gap_pct:+.1f}%)")

    if abs(gap_pct) < 10:
        verdict = "STRONG generalization -- brain truly learned the task"
    elif gap_pct < 30:
        verdict = "MODERATE generalization -- some overfit but still functional"
    elif gap_pct < 50:
        verdict = "WEAK generalization -- significant overfit"
    else:
        verdict = "OVERFIT -- brain memorized training seeds"
    print(f"  Verdict: {verdict}")

    # Save
    with open("brain_generalization_result.json", "w") as f:
        json.dump({
            "source_result": result_file,
            "source_eval_score": saved_eval,
            "train_seeds": train_seeds,
            "train_scores": [round(s, 2) for s in train_scores],
            "train_mean": round(train_mean, 2),
            "unseen_seeds": unseen_seeds,
            "unseen_scores": [round(s, 2) for s in unseen_scores],
            "unseen_mean": round(unseen_mean, 2),
            "unseen_std": round(unseen_std, 2),
            "unseen_reach_rate": round(sum(1 for s in unseen_scores if s >= 30) / len(unseen_scores), 3),
            "gap": round(gap, 2),
            "gap_pct": round(gap_pct, 1),
            "verdict": verdict,
        }, f, indent=2)
    print("\n  Saved: brain_generalization_result.json")


if __name__ == "__main__":
    main()
