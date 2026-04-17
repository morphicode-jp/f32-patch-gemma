"""rgated_memory_sentinel.py - Phase 5 retry: Reward-gated 3-factor plasticity.

Correct Hebbian (from Frémaux & Gerstner 2016):
  Δw = lr * pre * post * reward_signal - decay * w

Reward signal in memory task = progress toward food:
  reward = (prev_dist - curr_dist)  # positive when approaching

Key difference from pure Hebbian:
  Unsupervised Hebbian strengthens ALL co-firing (can drift to noise).
  3-factor strengthens ONLY co-firing that COINCIDED with progress.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import (
    KATHARA_EDGES, PARAM_RANGES, PARAM_NAMES, simulate_step,
)
from twelve.agent.sentinel import Sentinel
from memory_world import MemoryFlyWorld


def simulate_step_3factor(params, inputs, states, w_adapt, reward,
                          lr=0.03, decay=0.005, w_clip=1.5):
    """Forward pass + reward-gated Hebbian."""
    p = np.asarray(params, dtype=np.float64).copy()
    p[:30] = p[:30] + w_adapt
    new_states, firing = simulate_step(p, inputs, states)

    # 3-factor update: only strengthen co-firing when reward > 0
    # and weaken when reward < 0
    for idx, (a, b) in enumerate(KATHARA_EDGES):
        coact = firing[a] * firing[b]
        w_adapt[idx] += lr * coact * reward - decay * w_adapt[idx]

    np.clip(w_adapt, -w_clip, w_clip, out=w_adapt)
    return new_states, firing, w_adapt


def run_rgated_episode(params, seed, reveal_steps=5, n_steps=40, lr=0.03):
    p = np.asarray(params, dtype=np.float64)
    world = MemoryFlyWorld(seed=seed, reveal_steps=reveal_steps)
    sensors = world.reset()
    initial_dist = world.get_food_dist()
    prev_dist = initial_dist
    min_dist_after_hidden = initial_dist
    states = np.zeros(12)
    w_adapt = np.zeros(30)
    reached = False
    during_visible = False
    move_sum = 0.0

    for step in range(n_steps):
        prev_pos = world.fly_pos.copy()
        # Reward signal = progress from last step (positive = approaching)
        reward = prev_dist - world.get_food_dist()
        states, firing, w_adapt = simulate_step_3factor(
            p, sensors, states, w_adapt, reward=reward, lr=lr
        )
        nav, cen = float(firing[5]), float(firing[11])
        sensors, dist, reached_now, _ = world.step(nav, cen)
        move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
        if step >= reveal_steps:
            min_dist_after_hidden = min(min_dist_after_hidden, dist)
        prev_dist = dist
        if reached_now:
            reached = True
            if step < reveal_steps:
                during_visible = True
            break

    approach_after = max(0.0, initial_dist - min_dist_after_hidden) / (initial_dist + 1e-6)
    ep_score = approach_after * 70.0 + (20.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
    if during_visible:
        ep_score *= 0.5
    return ep_score, reached, during_visible, min_dist_after_hidden, w_adapt


def make_eval_fn(n_episodes=3, lr=0.03):
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            s, _, _, _, _ = run_rgated_episode(params, seed=ep * 7 + 13, lr=lr)
            total += s
        return total / n_episodes
    return eval_fn


def make_guard_fn(n_episodes=2, lr=0.03):
    def guard_fn(params):
        p = np.asarray(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = MemoryFlyWorld(seed=ep * 11 + 5, reveal_steps=5)
            sensors = world.reset()
            states = np.zeros(12); w_adapt = np.zeros(30)
            prev_dist = world.get_food_dist()
            move = 0.0
            for _ in range(30):
                prev = world.fly_pos.copy()
                reward = prev_dist - world.get_food_dist()
                states, firing, w_adapt = simulate_step_3factor(
                    p, sensors, states, w_adapt, reward=reward, lr=lr
                )
                sensors, d, reached, _ = world.step(float(firing[5]), float(firing[11]))
                move += float(np.linalg.norm(world.fly_pos - prev))
                prev_dist = d
                if reached: break
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def evaluate(params, n_seeds=20, label="", lr=0.03):
    scores, reaches, finals, w_mags = [], [], [], []
    during_count = 0
    for seed in range(2000, 2000 + n_seeds):
        s, r, v, d, w = run_rgated_episode(params, seed, lr=lr)
        scores.append(s); reaches.append(r); finals.append(d)
        w_mags.append(float(np.abs(w).sum()))
        if v: during_count += 1
    print(f"  [{label}] score={np.mean(scores):.2f}  "
          f"reach={sum(reaches)/len(reaches)*100:.0f}%  "
          f"final={np.mean(finals):.2f}  "
          f"|w_adapt|={np.mean(w_mags):.2f}  trivial={during_count}")
    return {
        "label": label,
        "mean_score": round(float(np.mean(scores)), 2),
        "reach_rate": round(sum(reaches)/len(reaches), 3),
        "mean_final_dist": round(float(np.mean(finals)), 2),
        "mean_w_mag": round(float(np.mean(w_mags)), 2),
    }


def main():
    print("=" * 70)
    print("  Phase 5 retry: Reward-gated 3-factor plasticity")
    print("=" * 70)

    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]

    print("\n[baseline] midpoint with 3-factor:")
    base = evaluate(mid, n_seeds=20, label="midpoint")

    if os.path.exists("brain_sentinel_biological_result.json"):
        with open("brain_sentinel_biological_result.json") as f:
            d = json.load(f)
        iss = d.get("best_ever_params") or d.get("best_params")
        print("\n[ISS-trained + 3-factor]:")
        evaluate(iss, n_seeds=20, label="iss_rgated")

    print("\n[training] Sentinel with 3-factor eval_fn (180s)...")
    t0 = time.time()
    result = Sentinel(
        eval_fn=make_eval_fn(n_episodes=3, lr=0.03),
        guard_fn=make_guard_fn(n_episodes=2, lr=0.03),
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id="rgated_memory_brain",
        learn=True,
    ).run(time_budget=180, verbose=False)
    elapsed = time.time() - t0

    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    print(f"\n  training done: eval={eval_s:.2f} elapsed={elapsed:.0f}s "
          f"verdict={result.get('verdict')}")

    print("\n[evaluation] 3-factor-trained brain on 20 unseen seeds:")
    trained = evaluate(best, n_seeds=20, label="rgated_trained")

    print("\n" + "=" * 70)
    print("  3-factor plasticity verdict")
    print("=" * 70)
    print(f"  Phase 4 leaky only:       reach 20%  final 1.67")
    print(f"  Phase 5a pure Hebbian:    reach  5%  final 1.67")
    print(f"  Phase 5b midpoint 3-fac:  reach {base['reach_rate']*100:.0f}%  "
          f"final {base['mean_final_dist']:.2f}")
    print(f"  Phase 5b trained 3-fac:   reach {trained['reach_rate']*100:.0f}%  "
          f"final {trained['mean_final_dist']:.2f}")

    lift_vs_p4 = trained['reach_rate'] - 0.20
    print(f"\n  Lift vs Phase 4: {lift_vs_p4*100:+.0f}pt")
    if trained['reach_rate'] > 0.50:
        verdict = "STRONG 3-FACTOR MEMORY"
    elif trained['reach_rate'] > 0.35:
        verdict = "3-FACTOR > ISS-momentum-memory"
    elif trained['reach_rate'] > 0.25:
        verdict = "3-FACTOR > Phase 4 leaky"
    else:
        verdict = "NO IMPROVEMENT - need different approach"
    print(f"  Verdict: {verdict}")

    with open("rgated_memory_result.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "elapsed_s": round(elapsed, 1),
            "training_eval": round(float(eval_s), 2),
            "sentinel_verdict": result.get("verdict"),
            "midpoint_3factor": base,
            "rgated_trained": trained,
            "best_params": [float(x) for x in best],
        }, f, indent=2)
    print("\n  Saved: rgated_memory_result.json")


if __name__ == "__main__":
    main()
