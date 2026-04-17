"""lambda_floor_sentinel.py - Sentinel with |λ₂| floor as structural guard.

Setup:
  eval_fn  = FlyWorldV3 behavioral performance (same as biological mode)
  guard_fn = |λ₂| (force spectral richness; baseline Kathara=3.73, target >= 6)

Hypothesis: Behavior-first optimization bounded by spectral floor should
  outperform biological mode (guard=ISS structural, unseen=61.86).

Validation caveat: |λ₂| -> unseen Pearson r = +0.31 at n=25
  (weak predictor, but Q4/Q1 lift = 4.4x suggests floor is still useful).

5 min budget.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import (
    KATHARA_EDGES, PARAM_RANGES, PARAM_NAMES,
    FlyWorldV3, simulate_step,
)
from twelve.agent.sentinel import Sentinel
from hybrid_eval_sentinel import run_unseen


def lambda2(params):
    p = np.array(params, dtype=np.float64)
    W = np.zeros((12, 12))
    for i, (a, b) in enumerate(KATHARA_EDGES):
        W[a, b] = p[i]; W[b, a] = p[i]
    eigs = sorted(np.abs(np.linalg.eigvalsh(W)), reverse=True)
    return eigs[1] if len(eigs) > 1 else 0.0


def make_flyworld_eval(n_ep=4, n_steps=60):
    def eval_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_ep):
            world = FlyWorldV3(seed=ep * 7 + 13)
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
            total += ep_score
        return total / n_ep
    return eval_fn


def lambda_guard_fn(params):
    """Guard = |λ₂| directly. Baseline Kathara uniform = 3.73."""
    return lambda2(params)


def main():
    print("=" * 70)
    print("  Lambda-floor Sentinel: eval=FlyWorld, guard=|λ₂| floor")
    print("=" * 70)

    t0 = time.time()
    result = Sentinel(
        eval_fn=make_flyworld_eval(n_ep=4, n_steps=60),
        guard_fn=lambda_guard_fn,
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id="lambda_floor_12n",
        learn=True,
    ).run(time_budget=300, verbose=True)

    elapsed = time.time() - t0
    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")
    verdict = result.get("verdict")

    print(f"\n  [training done] elapsed={elapsed:.0f}s")
    print(f"  eval (FlyWorld):  {eval_s:.2f}")
    print(f"  guard (|λ₂|):     {guard_s:.2f}")
    print(f"  verdict: {verdict}")

    print("\n--- UNSEEN seeds (20 episodes, FlyWorldV3) ---")
    seeds, scores = run_unseen(best)
    unseen_mean = float(np.mean(scores))
    unseen_std = float(np.std(scores))
    reach = sum(1 for s in scores if s >= 30) / len(scores)
    print(f"  Mean: {unseen_mean:.2f}  std={unseen_std:.2f}")
    print(f"  Reach rate: {reach * 100:.0f}%")

    # Compare with ISS-opt baseline
    iss_file = "brain_generalization_result.json"
    if os.path.exists(iss_file):
        iss = json.load(open(iss_file))
        print("\n" + "=" * 70)
        print("  VS ISS-guard biological Sentinel")
        print("=" * 70)
        print(f"                     ISS-guard    λ-floor guard")
        print(f"  Training eval    {iss.get('source_eval_score', 0):>8.2f}    {eval_s:>8.2f}")
        print(f"  |λ₂|             {7.63:>8.2f}    {guard_s:>8.2f}")
        print(f"  Unseen mean      {iss['unseen_mean']:>8.2f}    {unseen_mean:>8.2f}")
        print(f"  Reach rate         {iss['unseen_reach_rate']*100:>6.0f}%      {reach*100:>6.0f}%")

        if unseen_mean > iss["unseen_mean"]:
            verdict_cmp = f"LAMBDA-FLOOR WINS (+{unseen_mean - iss['unseen_mean']:.1f})"
        else:
            verdict_cmp = f"ISS-GUARD WINS ({unseen_mean - iss['unseen_mean']:+.1f})"
        print(f"\n  Verdict: {verdict_cmp}")

    with open("lambda_floor_result.json", "w") as f:
        json.dump({
            "elapsed_s": round(elapsed, 1),
            "eval_score": round(float(eval_s), 3),
            "guard_score_lambda2": round(float(guard_s), 3),
            "sentinel_verdict": verdict,
            "best_params": [float(x) for x in best],
            "unseen_seeds": seeds,
            "unseen_scores": [round(s, 2) for s in scores],
            "unseen_mean": round(unseen_mean, 2),
            "unseen_std": round(unseen_std, 2),
            "unseen_reach_rate": round(reach, 3),
        }, f, indent=2)
    print("\n  Saved: lambda_floor_result.json")


if __name__ == "__main__":
    main()
