"""kathara16_solo_train.py - Solo pre-training for Kathara(16) brain.

Stage 1 of Phase 7 curriculum: each Kathara(16) brain learns
FlyWorldV1 reach-food individually BEFORE joining cooperative world.

Target: >80% reach on 20 unseen seeds (matches Kathara(12) solo achievement).

Uses Kathara(16) sensor layout:
  - Input channels: [0, 2, 10] = left_vis, right_vis, olfactory
  - Motor outputs: node 5 (nav), node 11 (speed)
  - Voice (node 1) unused in solo task
  - ToM channels (3-9, 11-13) all zero (no other agents)

Training: Sentinel with 300s budget (extends ~15-30 min actual).
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from kathara16_brain import (
    PARAM_RANGES_16, PARAM_NAMES_16, TOTAL_PARAMS_16,
    N_NODES_16, simulate_step_16, DEFAULT_INHIBIT_16,
)
from kathara_brain_sim_v8 import FlyWorldV1


def sensor_vector_from_flyworld(world):
    """Convert FlyWorldV1 12-dim sensor to Kathara(16) 16-dim layout.

    FlyWorld sensors (12-dim):
      [7]  = left_vis
      [9]  = right_vis
      [10] = olfactory

    Kathara(16) layout:
      [0]  = left_vis
      [2]  = right_vis
      [10] = olfactory
    """
    raw = world._get_sensors()  # 12-dim
    inputs = np.zeros(16)
    inputs[0] = raw[7]    # left_vis -> channel 0
    inputs[2] = raw[9]    # right_vis -> channel 2
    inputs[10] = raw[10]  # olfactory -> channel 10
    return inputs


def run_solo_episode(params, seed, n_steps=40, return_detail=False):
    """Run single FlyWorldV1 episode with a Kathara(16) brain.

    Returns (score, reached, final_dist).
    """
    p = np.array(params, dtype=np.float64)
    world = FlyWorldV1(seed=seed)
    world.reset()
    sensors = sensor_vector_from_flyworld(world)
    states = np.zeros(N_NODES_16)
    initial_d = world.get_food_dist()
    min_d = initial_d
    reached = False
    move_sum = 0.0

    for step in range(n_steps):
        prev_pos = world.fly_pos.copy()
        states, firing = simulate_step_16(p, sensors, states,
                                          inhibit_sign=DEFAULT_INHIBIT_16)
        nav = float(firing[5])
        cen = float(firing[11])
        _, d, reached_now, _ = world.step(nav, cen)
        sensors = sensor_vector_from_flyworld(world)
        move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
        min_d = min(min_d, d)
        if reached_now:
            reached = True
            break

    approach = max(0.0, initial_d - min_d) / (initial_d + 1e-6)
    score = approach * 60 + (30 if reached else 0) + min(10.0, move_sum * 2.0)
    if return_detail:
        return score, reached, min_d, move_sum
    return score, reached, min_d


def make_eval_fn(n_episodes=4, seed_base=13):
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            s, _, _ = run_solo_episode(params, seed=ep * 7 + seed_base)
            total += s
        return total / n_episodes
    return eval_fn


def make_guard_fn(n_episodes=2, seed_base=5):
    """Guard = movement activity (prevent stuck brains)."""
    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            s, _, _, move = run_solo_episode(
                params, seed=ep * 11 + seed_base,
                return_detail=True
            )
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def evaluate_solo(params, n_seeds=20):
    scores, reaches, dists = [], [], []
    for seed in range(2000, 2000 + n_seeds):
        s, r, d = run_solo_episode(params, seed, n_steps=50)
        scores.append(s); reaches.append(r); dists.append(d)
    return {
        "n_seeds": n_seeds,
        "reach_rate": sum(reaches) / len(reaches),
        "mean_score": float(np.mean(scores)),
        "mean_final_dist": float(np.mean(dists)),
    }


def main():
    print("=" * 70)
    print("  Kathara(16) SOLO PRE-TRAINING (Phase 7 Stage 1)")
    print("=" * 70)

    # Baseline at midpoint
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_16]
    print("\n[baseline midpoint]")
    base = evaluate_solo(mid, n_seeds=10)
    print(f"  reach={base['reach_rate']*100:.0f}%  "
          f"score={base['mean_score']:.1f}  "
          f"final_dist={base['mean_final_dist']:.2f}")

    # Train
    print("\n[training] Sentinel 300s budget...")
    t0 = time.time()
    result = Sentinel(
        eval_fn=make_eval_fn(n_episodes=4),
        guard_fn=make_guard_fn(n_episodes=2),
        param_ranges=PARAM_RANGES_16,
        param_names=PARAM_NAMES_16,
        experience_id="kathara16_solo_pretrain",
        learn=True,
    ).run(time_budget=300, verbose=False)
    t_train = time.time() - t0

    best = result.get("best_ever_params") or result.get("best_params")
    best_eval = result.get("best_ever_score", 0.0)
    verdict = result.get("verdict")
    print(f"  trained: eval={best_eval:.2f}  elapsed={t_train:.0f}s "
          f"verdict={verdict}")

    # Evaluate on unseen
    print("\n[evaluation] 20 unseen seeds")
    res = evaluate_solo(best, n_seeds=20)
    print(f"  reach_rate:       {res['reach_rate']*100:.0f}%")
    print(f"  mean_score:       {res['mean_score']:.1f}")
    print(f"  mean_final_dist:  {res['mean_final_dist']:.2f}")

    # Decision
    if res["reach_rate"] >= 0.80:
        print(f"\n  PASS: >=80% reach achieved. Ready for Stage 2 warm-start.")
    elif res["reach_rate"] >= 0.50:
        print(f"\n  PARTIAL: {res['reach_rate']*100:.0f}% reach. "
              f"Proceeding with Stage 2 (reduced ambition).")
    else:
        print(f"\n  LOW: {res['reach_rate']*100:.0f}% reach. "
              f"Retry with longer budget recommended.")

    # Save
    out = {
        "baseline": base,
        "trained": res,
        "train_elapsed_s": round(t_train, 1),
        "train_verdict": verdict,
        "best_eval_score": round(float(best_eval), 2),
        "best_params": [float(x) for x in best],
    }
    with open("kathara16_solo_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: kathara16_solo_result.json")


if __name__ == "__main__":
    main()
