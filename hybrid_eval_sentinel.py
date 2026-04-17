"""hybrid_eval_sentinel.py - Combine lambda (universe) x ISS (task-structure).

Prior experiments:
  ISS-optimized:    guard 87.00 / unseen 61.86 / reach 75%
  lambda-optimized: guard 10.00 / unseen 23.88 / reach 25%  (collapsed)

Prior insight:
  lambda=4 is NECESSARY (structural backbone) but NOT SUFFICIENT.
  Pure lambda-max destroys micro-parameter tuning.

Hypothesis (this run):
  Geometric mean: eval = sqrt(lambda_score * iss_score) forces BOTH
  universe alignment AND functional structure. If either collapses
  the score goes to 0. Expect: unseen >= ISS-only baseline (61.86).
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import (
    KATHARA_EDGES, PARAM_RANGES, PARAM_NAMES,
    FlyWorldV1, FlyWorldV3, simulate_step,
)
from kathara_brain_sim_v6 import compute_iss_from_firing
from twelve.agent.sentinel import Sentinel


TARGET_LAMBDA = 4.0
SIGMA = 1.0


def lambda_score(params):
    p = np.array(params, dtype=np.float64)
    W = np.zeros((12, 12), dtype=np.float64)
    for i, (a, b) in enumerate(KATHARA_EDGES):
        W[a, b] = p[i]
        W[b, a] = p[i]
    try:
        eigs = np.linalg.eigvalsh(W)
    except np.linalg.LinAlgError:
        return 0.0
    eigs_abs = sorted(np.abs(eigs), reverse=True)
    l2 = eigs_abs[1] if len(eigs_abs) > 1 else 0.0
    return 100.0 * float(np.exp(-((l2 - TARGET_LAMBDA) ** 2) / (2.0 * SIGMA ** 2)))


def iss_score_fn(params, n_ep=2, n_steps=25):
    p = np.array(params, dtype=np.float64)
    total = 0.0
    for ep in range(n_ep):
        world = FlyWorldV1(seed=ep * 11 + 23)
        sensors = world.reset()
        firing_accum = np.zeros(12, dtype=np.float64)
        states = np.zeros(12, dtype=np.float64)
        steps_done = 0
        for _ in range(n_steps):
            states, firing = simulate_step(p, sensors, states)
            firing_accum += firing
            nav = float(firing[5])
            cen = float(firing[11])
            sensors, _, reached, _ = world.step(nav, cen)
            steps_done += 1
            if reached:
                break
        avg_firing = firing_accum / max(1, steps_done)
        total += float(compute_iss_from_firing(avg_firing, p))
    return total / n_ep


def hybrid_eval_fn(params):
    """Geometric mean of lambda and ISS. Both must be high."""
    l = lambda_score(params)
    i = iss_score_fn(params)
    return float(np.sqrt(max(l, 0.0) * max(i, 0.0)))


def make_flyworld_guard(n_ep=3, n_steps=40):
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
    print("  Hybrid-eval Sentinel: sqrt(lambda * ISS)")
    print("=" * 70)

    t0 = time.time()
    result = Sentinel(
        eval_fn=hybrid_eval_fn,
        guard_fn=make_flyworld_guard(n_ep=3, n_steps=40),
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id="hybrid_eval_12n",
        learn=True,
    ).run(time_budget=300, verbose=True)

    elapsed = time.time() - t0
    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")

    # Break down components at best_ever
    l_s = lambda_score(best)
    i_s = iss_score_fn(best, n_ep=4, n_steps=30)

    print(f"\n  [training done] elapsed={elapsed:.0f}s")
    print(f"  hybrid_eval:        {eval_s:.2f}")
    print(f"    lambda component: {l_s:.2f}")
    print(f"    ISS component:    {i_s:.2f}")
    print(f"  guard (FlyWorld):   {guard_s:.2f}")

    print("\n--- UNSEEN seeds (20 episodes, FlyWorldV3) ---")
    seeds, scores = run_unseen(best)
    unseen_mean = float(np.mean(scores))
    unseen_std = float(np.std(scores))
    reach = sum(1 for s in scores if s >= 30) / len(scores)
    print(f"  Mean: {unseen_mean:.2f}  std={unseen_std:.2f}")
    print(f"  Reach rate: {reach * 100:.0f}%")

    # 3-way comparison
    iss_file = "brain_generalization_result.json"
    lam_file = "lambda_eval_result.json"
    iss = json.load(open(iss_file)) if os.path.exists(iss_file) else None
    lam = json.load(open(lam_file)) if os.path.exists(lam_file) else None

    print("\n" + "=" * 70)
    print("  3-way comparison")
    print("=" * 70)
    print(f"                         ISS        Lambda       Hybrid")
    iss_u = iss["unseen_mean"] if iss else 0.0
    lam_u = lam["unseen_mean"] if lam else 0.0
    iss_r = iss["unseen_reach_rate"] if iss else 0.0
    lam_r = lam["unseen_reach_rate"] if lam else 0.0
    print(f"  Unseen mean      {iss_u:>8.2f}    {lam_u:>8.2f}    {unseen_mean:>8.2f}")
    print(f"  Reach rate         {iss_r*100:>6.0f}%      {lam_r*100:>6.0f}%      {reach*100:>6.0f}%")

    if unseen_mean > iss_u and reach >= iss_r:
        verdict = "HYBRID WINS -- lambda + ISS > either alone"
    elif unseen_mean > iss_u or reach > iss_r:
        verdict = "HYBRID PARTIAL WIN -- one metric improved"
    else:
        verdict = "HYBRID FAILS -- combining did not help"
    print(f"\n  Verdict: {verdict}")

    with open("hybrid_eval_result.json", "w") as f:
        json.dump({
            "target_lambda": TARGET_LAMBDA,
            "elapsed_s": round(elapsed, 1),
            "hybrid_eval_score": round(float(eval_s), 3),
            "lambda_component": round(l_s, 3),
            "iss_component": round(i_s, 3),
            "guard_score": round(float(guard_s), 3),
            "best_params": [float(x) for x in best],
            "unseen_seeds": seeds,
            "unseen_scores": [round(s, 2) for s in scores],
            "unseen_mean": round(unseen_mean, 2),
            "unseen_std": round(unseen_std, 2),
            "unseen_reach_rate": round(reach, 3),
            "verdict_str": verdict,
            "sentinel_verdict": result.get("verdict"),
        }, f, indent=2)
    print("\n  Saved: hybrid_eval_result.json")


if __name__ == "__main__":
    main()
