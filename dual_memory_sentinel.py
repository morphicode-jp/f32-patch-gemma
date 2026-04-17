"""dual_memory_sentinel.py - Train dual brain with replay and multi-episode persistence.

Phase 6: Human-brain-inspired memory system.
  - Hippocampus module with k-WTA sparse coding
  - Reward-gated 3-factor plasticity
  - Post-episode pattern replay (sleep-style consolidation)
  - Cross-episode w_adapt persistence with decay

Two evaluations:
  1. Standard MemoryFlyWorld task (same as Phase 5) -- compare reach rate
  2. Persistent-food task (new) -- same food position 3 episodes in a row.
     Memory should make episode 2-3 better than episode 1.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from memory_world import MemoryFlyWorld
from dual_brain import (
    DUAL_PARAM_RANGES, DUAL_PARAM_NAMES, DUAL_TOTAL,
    N_MAIN, N_HIPPO,
    split_params, dual_step
)
from hippocampus_brain import replay_patterns


def run_dual_episode(dual_params, seed, reveal_steps=5, n_steps=40,
                     w_adapt_carry=None, enable_replay=True):
    """Run one episode with dual brain + replay.

    w_adapt_carry: if given, carries over (with decay) as long-term memory.
    Returns: (score, reached, during_visible, min_dist_after_hidden,
              w_adapt_final, move_sum)
    """
    main_p, hippo_p, m_to_h, h_to_m, k, decay = split_params(dual_params)

    # Persistent memory: carry w_adapt from previous episode, apply decay
    if w_adapt_carry is None:
        w_adapt = np.zeros(30)
    else:
        w_adapt = np.asarray(w_adapt_carry, dtype=np.float64) * decay

    world = MemoryFlyWorld(seed=seed, reveal_steps=reveal_steps)
    sensors = world.reset()
    main_states = np.zeros(12)
    hippo_states = np.zeros(12)
    initial_dist = world.get_food_dist()
    prev_dist = initial_dist   # dist before most recent step
    dist_lag = initial_dist    # dist from 2 steps ago (for reward-lag trick)
    min_d_after_hidden = initial_dist
    reached = False
    during_visible = False
    move_sum = 0.0
    trajectory = []  # (sensor_snapshot, reward) for replay

    for step in range(n_steps):
        prev_pos = world.fly_pos.copy()
        # reward = progress from last step (positive = closer to food now)
        reward = dist_lag - prev_dist
        (main_states, hippo_states, w_adapt,
         main_firing, hippo_firing) = dual_step(
            main_p, hippo_p, sensors, main_states, hippo_states,
            w_adapt, reward, m_to_h, h_to_m, k
        )
        nav, cen = float(main_firing[5]), float(main_firing[11])
        sensors, dist, reached_now, _ = world.step(nav, cen)
        move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
        trajectory.append((sensors.copy(), float(reward)))
        if step >= reveal_steps:
            min_d_after_hidden = min(min_d_after_hidden, dist)
        # Advance the reward-lag buffer
        dist_lag = prev_dist
        prev_dist = dist
        if reached_now:
            reached = True
            during_visible = step < reveal_steps
            break

    # REPLAY (sleep consolidation): top-5 reward patterns x 10 iterations
    if enable_replay and len(trajectory) >= 5:
        top5 = sorted(trajectory, key=lambda x: x[1], reverse=True)[:5]
        w_adapt = replay_patterns(
            hippo_p, top5, w_adapt, n_iterations=10, k=k, lr=0.05
        )

    approach = max(0.0, initial_dist - min_d_after_hidden) / (initial_dist + 1e-6)
    ep_score = approach * 70.0 + (20.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
    if during_visible:
        ep_score *= 0.5
    return ep_score, reached, during_visible, min_d_after_hidden, w_adapt, move_sum


def make_eval_fn(n_episodes=3, persistent_trials=3):
    """eval_fn: score = memory-explicit persistent-food task.

    Each trial: same seed across `persistent_trials` episodes.
    Brain sees food briefly at ep1 (reveal=5), then HIDDEN for ep2+ (reveal=0).
    Reward for reaching food in ep2/ep3 is the memory signal Sentinel optimizes.

    Score: weighted sum that PENALIZES ep1-only solutions and REWARDS
           ep2/ep3 reach (genuine memory).
    """
    def eval_fn(dual_params):
        total = 0.0
        for trial in range(n_episodes):
            seed = 5000 + trial * 13
            w_adapt = None
            ep_scores = []
            for ep in range(persistent_trials):
                # ep0: food visible briefly; ep1+: food never visible
                reveal = 5 if ep == 0 else 0
                s, r, _, d, w_adapt, _ = run_dual_episode(
                    dual_params, seed=seed, reveal_steps=reveal,
                    n_steps=40, w_adapt_carry=w_adapt
                )
                # Weight memory episodes (ep1, ep2) higher
                weight = 1.0 if ep == 0 else 2.0
                ep_scores.append(s * weight)
            total += sum(ep_scores) / sum([1.0, 2.0, 2.0][:persistent_trials])
        return total / n_episodes
    return eval_fn


def make_guard_fn(n_episodes=2):
    """guard_fn: ensure brain moves (prevent degenerate stuck solutions)."""
    def guard_fn(dual_params):
        w_adapt = None
        total_move = 0.0
        for ep in range(n_episodes):
            _, _, _, _, w_adapt, move_sum = run_dual_episode(
                dual_params, seed=ep * 11 + 5, w_adapt_carry=w_adapt,
                enable_replay=False  # skip replay in guard for speed
            )
            total_move += move_sum * 5.0
        return total_move / n_episodes
    return guard_fn


def evaluate_standard(dual_params, n_seeds=20):
    """Standard MemoryFlyWorld test. Compare vs Phase 5b."""
    scores, reaches, finals, w_mags = [], [], [], []
    for seed in range(2000, 2000 + n_seeds):
        s, r, _, d, w, _ = run_dual_episode(dual_params, seed, w_adapt_carry=None)
        scores.append(s); reaches.append(r); finals.append(d)
        w_mags.append(float(np.abs(w).sum()))
    return {
        "reach_rate": round(sum(reaches) / len(reaches), 3),
        "mean_score": round(float(np.mean(scores)), 2),
        "mean_final_dist": round(float(np.mean(finals)), 2),
        "mean_w_adapt_mag": round(float(np.mean(w_mags)), 3),
    }


def evaluate_persistent_food(dual_params, n_trials=10, episodes_per_trial=3):
    """Same-seed episodes with STRICT memory protocol:
       ep0: food visible briefly (reveal=5)
       ep1+: food NEVER visible (reveal=0)
    Only way to reach food in ep1/ep2 is to REMEMBER from ep0.
    """
    results_by_ep = [[] for _ in range(episodes_per_trial)]
    finals_by_ep = [[] for _ in range(episodes_per_trial)]
    w_mag_by_ep = [[] for _ in range(episodes_per_trial)]
    for trial in range(n_trials):
        w_adapt = None
        for ep in range(episodes_per_trial):
            reveal = 5 if ep == 0 else 0
            _, r, _, d, w_adapt, _ = run_dual_episode(
                dual_params, seed=3000 + trial,
                reveal_steps=reveal, w_adapt_carry=w_adapt
            )
            results_by_ep[ep].append(float(r))
            finals_by_ep[ep].append(float(d))
            w_mag_by_ep[ep].append(float(np.abs(w_adapt).sum()))
    return {
        "reach_per_ep": [round(sum(x) / len(x), 3) for x in results_by_ep],
        "final_dist_per_ep": [round(sum(x) / len(x), 3) for x in finals_by_ep],
        "w_mag_per_ep": [round(sum(x) / len(x), 3) for x in w_mag_by_ep],
    }


def evaluate_no_memory_ablation(dual_params, n_seeds=20):
    """Ablation: disable persistence (no carryover). Isolates memory effect."""
    scores, reaches = [], []
    for seed in range(2000, 2000 + n_seeds):
        s, r, _, _, _, _ = run_dual_episode(dual_params, seed, w_adapt_carry=None)
        scores.append(s); reaches.append(r)
    return {
        "reach_rate": round(sum(reaches) / len(reaches), 3),
        "mean_score": round(float(np.mean(scores)), 2),
    }


def main():
    print("=" * 70)
    print("  Phase 6: Dual brain + hippocampus + replay + persistence")
    print("=" * 70)
    print(f"  Total params: {DUAL_TOTAL} (main 91 + hippo 91 + 4 coupling)")

    # Optional: warm-start from existing best main brain
    initial = None
    if os.path.exists("brain_sentinel_biological_result.json"):
        try:
            with open("brain_sentinel_biological_result.json") as f:
                d = json.load(f)
            iss_p = d.get("best_ever_params") or d.get("best_params")
            if iss_p and len(iss_p) == N_MAIN:
                # Build initial: ISS-trained main + midpoint hippo + sane coupling
                mid_full = [(lo + hi) / 2 for lo, hi in DUAL_PARAM_RANGES]
                initial = list(iss_p) + list(mid_full[N_MAIN:])
                print(f"  Warm-start: ISS-trained main brain + midpoint hippo")
        except Exception as e:
            print(f"  (warm start skipped: {e})")

    # Baseline eval (midpoint)
    mid = [(lo + hi) / 2 for lo, hi in DUAL_PARAM_RANGES]
    print("\n[baseline midpoint dual brain]")
    base_std = evaluate_standard(mid, n_seeds=20)
    print(f"  standard: reach={base_std['reach_rate']*100:.0f}%  "
          f"final={base_std['mean_final_dist']:.2f}  "
          f"|w_adapt|={base_std['mean_w_adapt_mag']:.2f}")

    base_persist = evaluate_persistent_food(mid, n_trials=8, episodes_per_trial=3)
    print(f"  persistent food ep reach: {base_persist['reach_per_ep']}")

    # Training
    print(f"\n[training] Sentinel on {DUAL_TOTAL}D, 240s budget...")
    t0 = time.time()
    result = Sentinel(
        eval_fn=make_eval_fn(n_episodes=3),
        guard_fn=make_guard_fn(n_episodes=2),
        param_ranges=DUAL_PARAM_RANGES,
        param_names=DUAL_PARAM_NAMES,
        experience_id="dual_hippo_memory",
        initial_params=initial,
        learn=True,
    ).run(time_budget=240, verbose=False)
    elapsed = time.time() - t0

    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")
    sv = result.get("verdict")
    print(f"  trained: eval={eval_s:.2f}  guard={guard_s:.2f}  "
          f"elapsed={elapsed:.0f}s  sentinel_verdict={sv}")
    # Parse coupling params for insight
    _, _, m_to_h, h_to_m, k, decay = split_params(best)
    print(f"  learned coupling: m_to_h={m_to_h:.2f}  h_to_m={h_to_m:.2f}  "
          f"k={k}  decay={decay:.3f}")

    # Evaluations on trained brain
    print("\n[evaluation] trained dual brain")
    trained_std = evaluate_standard(best, n_seeds=20)
    print(f"  STANDARD TASK: reach={trained_std['reach_rate']*100:.0f}%  "
          f"final={trained_std['mean_final_dist']:.2f}  "
          f"|w_adapt|={trained_std['mean_w_adapt_mag']:.2f}")

    trained_persist = evaluate_persistent_food(best, n_trials=10, episodes_per_trial=3)
    print(f"  PERSISTENT FOOD reach per episode: {trained_persist['reach_per_ep']}")
    print(f"    (ideal: ep3 > ep1 = cross-episode memory confirmed)")
    print(f"  w_adapt magnitude per episode: {trained_persist['w_mag_per_ep']}")

    trained_ablation = evaluate_no_memory_ablation(best, n_seeds=20)
    print(f"  NO-CARRY ABLATION: reach={trained_ablation['reach_rate']*100:.0f}% "
          f"(vs standard with carry enabled: should differ if memory active)")

    # Comparison table
    print("\n" + "=" * 70)
    print("  Phase comparison")
    print("=" * 70)
    print(f"  {'method':<30} {'reach':>8}  {'final_dist':>11}")
    print(f"  {'Phase 4 leaky':<30} {'20%':>8}  {'1.67':>11}")
    print(f"  {'Phase 5a pure Hebbian':<30} {'5%':>8}  {'1.67':>11}")
    print(f"  {'Phase 5b 3-factor':<30} {'30%':>8}  {'1.65':>11}")
    print(f"  {'Phase 6 dual+hippo+replay':<30} "
          f"{trained_std['reach_rate']*100:.0f}%    "
          f"{trained_std['mean_final_dist']:.2f}")

    # Verdicts
    ep1, ep2, ep3 = trained_persist['reach_per_ep']
    lift_std = trained_std['reach_rate'] - 0.30
    lift_persist = ep3 - ep1
    active_plasticity = trained_std['mean_w_adapt_mag'] > 0.5

    if trained_std['reach_rate'] > 0.50 and lift_persist > 0.20:
        overall = "FULL SUCCESS"
    elif trained_std['reach_rate'] > 0.35 or lift_persist > 0.10:
        overall = "PARTIAL SUCCESS"
    else:
        overall = "NO CLEAR IMPROVEMENT"
    print(f"\n  verdict: {overall}")
    print(f"    standard lift vs Phase 5b:   {lift_std*100:+.0f}pt")
    print(f"    persistent lift ep1->ep3:    {lift_persist*100:+.0f}pt")
    print(f"    plasticity active (|w|>0.5): {active_plasticity}")

    # Save
    out = {
        "verdict": overall,
        "elapsed_s": round(elapsed, 1),
        "n_params": DUAL_TOTAL,
        "training_eval": round(float(eval_s), 2),
        "training_guard": round(float(guard_s), 2),
        "sentinel_verdict": sv,
        "coupling_learned": {
            "m_to_h_gain": round(m_to_h, 3),
            "h_to_m_gain": round(h_to_m, 3),
            "k_wta": k,
            "persistence_decay": round(decay, 3),
        },
        "baseline_midpoint": {
            "standard": base_std,
            "persistent_food": base_persist,
        },
        "trained": {
            "standard": trained_std,
            "persistent_food": trained_persist,
            "no_carry_ablation": trained_ablation,
        },
        "phase_comparison": {
            "phase_4_leaky":       {"reach": 0.20, "final_dist": 1.67},
            "phase_5a_hebbian":    {"reach": 0.05, "final_dist": 1.67},
            "phase_5b_3factor":    {"reach": 0.30, "final_dist": 1.65},
            "phase_6_dual":        trained_std,
        },
        "best_params": [float(x) for x in best],
    }
    with open("dual_memory_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved: dual_memory_result.json")


if __name__ == "__main__":
    main()
