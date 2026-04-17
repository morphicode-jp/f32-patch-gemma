"""lambda_eval_sentinel.py - Universe-aligned eval: spectral gap lambda_2 = 4

Hypothesis (user):
  The universe favors grid/circulant topology with spectral gap lambda_2 ~ 4.
  A brain whose connectivity matrix has eigenvalue spectrum matching this
  universal pattern should generalize BETTER than one optimized for ISS
  (which is our constructed metric).

Test:
  1. Optimize 12N Kathara brain with lambda_eval_fn (target lambda_2 = 4)
  2. Guard: FlyWorld reach (unchanged from biological run)
  3. Evaluate best_params on 20 unseen FlyWorld seeds
  4. Compare gap/reach vs ISS-optimized baseline

If lambda wins: universal physics > constructed heuristic.
If ISS wins: task-specific metric > abstract principle.
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


TARGET_LAMBDA = 4.0
SIGMA = 1.0


def lambda_eval_fn(params):
    """Score = closeness to universe-preferred spectral gap (lambda_2 = 4)."""
    p = np.array(params, dtype=np.float64)
    edge_weights = p[:30]

    W = np.zeros((12, 12), dtype=np.float64)
    for i, (a, b) in enumerate(KATHARA_EDGES):
        W[a, b] = edge_weights[i]
        W[b, a] = edge_weights[i]

    try:
        eigs = np.linalg.eigvalsh(W)
    except np.linalg.LinAlgError:
        return 0.0

    eigs_abs = sorted(np.abs(eigs), reverse=True)
    lambda_2 = eigs_abs[1] if len(eigs_abs) > 1 else 0.0
    score = 100.0 * float(np.exp(-((lambda_2 - TARGET_LAMBDA) ** 2) / (2.0 * SIGMA ** 2)))
    return score


def make_flyworld_guard(n_ep=3, n_steps=40):
    """Guard = FlyWorld behavioral score (same formula as brain_generalization)."""
    def guard_fn(params):
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
    return guard_fn


def run_unseen(params, n_seeds=20, n_steps=60):
    rng = np.random.RandomState(999)
    seeds = [int(s) for s in rng.randint(1000, 10000, size=n_seeds)]
    train_seeds = [ep * 7 + 13 for ep in range(4)]
    seeds = [s for s in seeds if s not in train_seeds][:n_seeds]

    p = np.array(params, dtype=np.float64)
    scores = []
    for seed in seeds:
        world = FlyWorldV3(seed=seed)
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
    return seeds, scores


def main():
    print("=" * 70)
    print("  Lambda-eval Sentinel: universe-aligned spectral gap target = 4.0")
    print("=" * 70)

    t0 = time.time()
    guard_fn = make_flyworld_guard(n_ep=3, n_steps=40)

    result = Sentinel(
        eval_fn=lambda_eval_fn,
        guard_fn=guard_fn,
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id="lambda_eval_12n",
        learn=True,
    ).run(time_budget=300, verbose=True)

    elapsed = time.time() - t0
    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")

    print(f"\n  [training done] elapsed={elapsed:.0f}s")
    print(f"  eval_score (lambda): {eval_s:.2f}")
    print(f"  guard_score (FlyWorld): {guard_s:.2f}")

    # Generalization test on unseen seeds
    print("\n--- UNSEEN seeds (20 episodes, FlyWorldV3) ---")
    seeds, scores = run_unseen(best)
    unseen_mean = float(np.mean(scores))
    unseen_std = float(np.std(scores))
    reach = sum(1 for s in scores if s >= 30) / len(scores)
    print(f"  Mean: {unseen_mean:.2f}  std={unseen_std:.2f}")
    print(f"  Reach rate: {reach * 100:.0f}%")

    # Compare with ISS baseline (brain_generalization_result.json)
    iss_file = "brain_generalization_result.json"
    if os.path.exists(iss_file):
        with open(iss_file) as f:
            iss = json.load(f)
        print("\n" + "=" * 70)
        print("  VS ISS-optimized 12N baseline")
        print("=" * 70)
        print(f"                         ISS         Lambda")
        print(f"  Training guard      {iss['train_mean']:>7.2f}   {guard_s:>7.2f}")
        print(f"  Unseen mean         {iss['unseen_mean']:>7.2f}   {unseen_mean:>7.2f}")
        print(f"  Reach rate unseen   {iss['unseen_reach_rate']*100:>6.0f}%    {reach*100:>6.0f}%")
        print(f"  Gap %               {iss['gap_pct']:>+6.1f}%   ", end="")
        gap_pct = (guard_s - unseen_mean) / max(guard_s, 1) * 100
        print(f"{gap_pct:>+6.1f}%")

        # Verdict
        print("\n  HYPOTHESIS: lambda >> ISS for generalization")
        print("-" * 70)
        iss_unseen = iss["unseen_mean"]
        iss_reach = iss["unseen_reach_rate"]
        if unseen_mean > iss_unseen:
            print(f"  [unseen] lambda {unseen_mean:.2f} > ISS {iss_unseen:.2f}  --> SUPPORTED")
        else:
            print(f"  [unseen] lambda {unseen_mean:.2f} <= ISS {iss_unseen:.2f}  --> NOT supported")
        if reach > iss_reach:
            print(f"  [reach] lambda {reach*100:.0f}% > ISS {iss_reach*100:.0f}%  --> SUPPORTED")
        else:
            print(f"  [reach] lambda {reach*100:.0f}% <= ISS {iss_reach*100:.0f}%  --> NOT supported")

    # Save
    with open("lambda_eval_result.json", "w") as f:
        json.dump({
            "target_lambda": TARGET_LAMBDA,
            "sigma": SIGMA,
            "elapsed_s": round(elapsed, 1),
            "best_eval_score": round(float(eval_s), 3),
            "best_guard_score": round(float(guard_s), 3),
            "best_params": [float(x) for x in best],
            "unseen_seeds": seeds,
            "unseen_scores": [round(s, 2) for s in scores],
            "unseen_mean": round(unseen_mean, 2),
            "unseen_std": round(unseen_std, 2),
            "unseen_reach_rate": round(reach, 3),
            "verdict": result.get("verdict"),
        }, f, indent=2)
    print("\n  Saved: lambda_eval_result.json")


if __name__ == "__main__":
    main()
