"""hebbian_memory_sentinel.py - Long-term memory via online Hebbian plasticity.

Same memory task as Phase 4 (food visible 0-4, hidden after),
but now the brain uses simulate_step_hebbian which strengthens
co-active edges during visible phase.

Expectation: After visible phase, the Hebbian-strengthened edges
maintain the food-direction activity pattern, allowing navigation
to food target even without sensors.

Comparison:
  Phase 4 leaky-only memory:  reach 20%,  final_dist = 1.67
  Phase 5 Hebbian memory:     reach ???,  final_dist ???

If Hebbian > 20% reach -> long-term memory confirmed.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, PARAM_NAMES
from twelve.agent.sentinel import Sentinel
from memory_world import MemoryFlyWorld
from hebbian_brain import simulate_step_hebbian


def run_hebbian_episode(params, seed, reveal_steps=5, n_steps=40, lr=0.05):
    p = np.asarray(params, dtype=np.float64)
    world = MemoryFlyWorld(seed=seed, reveal_steps=reveal_steps)
    sensors = world.reset()
    initial_dist = world.get_food_dist()
    min_dist_after_hidden = initial_dist
    states = np.zeros(12, dtype=np.float64)
    w_adapt = np.zeros(30, dtype=np.float64)
    reached = False
    move_sum = 0.0
    reached_during_visible = False

    for step in range(n_steps):
        prev_pos = world.fly_pos.copy()
        states, firing, w_adapt = simulate_step_hebbian(
            p, sensors, states, w_adapt, lr=lr
        )
        nav, cen = float(firing[5]), float(firing[11])
        sensors, dist, reached_now, _ = world.step(nav, cen)
        move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
        if step >= reveal_steps:
            min_dist_after_hidden = min(min_dist_after_hidden, dist)
        if reached_now:
            reached = True
            if step < reveal_steps:
                reached_during_visible = True
            break

    approach_after = max(0.0, initial_dist - min_dist_after_hidden) / (initial_dist + 1e-6)
    ep_score = approach_after * 70.0 + (20.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
    if reached_during_visible:
        ep_score *= 0.5
    return ep_score, reached, reached_during_visible, min_dist_after_hidden, w_adapt


def make_eval_fn(n_episodes=3, n_steps=40, lr=0.05):
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            s, _, _, _, _ = run_hebbian_episode(params, seed=ep * 7 + 13,
                                                n_steps=n_steps, lr=lr)
            total += s
        return total / n_episodes
    return eval_fn


def make_guard_fn(n_episodes=2, n_steps=30, lr=0.05):
    def guard_fn(params):
        p = np.asarray(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = MemoryFlyWorld(seed=ep * 11 + 5, reveal_steps=5)
            sensors = world.reset()
            states = np.zeros(12); w_adapt = np.zeros(30)
            move = 0.0
            for _ in range(n_steps):
                prev = world.fly_pos.copy()
                states, firing, w_adapt = simulate_step_hebbian(
                    p, sensors, states, w_adapt, lr=lr
                )
                sensors, _, reached, _ = world.step(float(firing[5]), float(firing[11]))
                move += float(np.linalg.norm(world.fly_pos - prev))
                if reached: break
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def evaluate_brain(params, n_seeds=20, label="", lr=0.05):
    scores = []; reaches = []; finals = []; during = 0
    w_changes = []
    for seed in range(2000, 2000 + n_seeds):
        s, r, v, d, w = run_hebbian_episode(params, seed, lr=lr)
        scores.append(s); reaches.append(r); finals.append(d)
        w_changes.append(float(np.abs(w).sum()))
        if v: during += 1
    print(f"  [{label}] score={np.mean(scores):.2f}  "
          f"reach={sum(reaches)/len(reaches)*100:.0f}%  "
          f"final_dist={np.mean(finals):.2f}  "
          f"Hebbian_delta={np.mean(w_changes):.2f}  "
          f"trivial={during}/{n_seeds}")
    return {
        "label": label,
        "mean_score": round(float(np.mean(scores)), 2),
        "reach_rate": round(sum(reaches)/len(reaches), 3),
        "mean_final_dist": round(float(np.mean(finals)), 2),
        "mean_hebbian_delta": round(float(np.mean(w_changes)), 2),
    }


def main():
    print("=" * 70)
    print("  Phase 5 MVP: Hebbian long-term memory")
    print("  Food visible 0-4, hidden 5+. Brain uses online Hebbian plasticity.")
    print("=" * 70)

    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]

    print("\n[baseline] midpoint brain (Hebbian on):")
    base = evaluate_brain(mid, n_seeds=20, label="midpoint_hebbian")

    # Phase 4 memory-trained brain (non-Hebbian) on this same Hebbian eval
    if os.path.exists("memory_result.json"):
        with open("memory_result.json") as f:
            d = json.load(f)
        p4 = d.get("best_params") or d.get("best_ever_params")
        print("\n[Phase 4 brain + Hebbian]:")
        p4_heb = evaluate_brain(p4, n_seeds=20, label="phase4_brain")

    print("\n[training] Sentinel with Hebbian eval_fn (180s)...")
    t0 = time.time()
    result = Sentinel(
        eval_fn=make_eval_fn(n_episodes=3, n_steps=40, lr=0.05),
        guard_fn=make_guard_fn(n_episodes=2, n_steps=30, lr=0.05),
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id="hebbian_memory_brain",
        learn=True,
    ).run(time_budget=180, verbose=False)
    elapsed = time.time() - t0

    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")
    print(f"\n  training done: eval={eval_s:.2f} guard={guard_s:.2f} "
          f"elapsed={elapsed:.0f}s verdict={result.get('verdict')}")

    print("\n[evaluation] Hebbian-trained brain on 20 unseen seeds:")
    mem = evaluate_brain(best, n_seeds=20, label="hebbian_trained")

    print("\n" + "=" * 70)
    print("  Hebbian memory verdict")
    print("=" * 70)
    print(f"  Phase 4 leaky-only:      reach 20%  final 1.67")
    print(f"  midpoint + Hebbian:      reach {base['reach_rate']*100:.0f}%  "
          f"final {base['mean_final_dist']:.2f}")
    print(f"  Hebbian-trained:         reach {mem['reach_rate']*100:.0f}%  "
          f"final {mem['mean_final_dist']:.2f}")

    lift_vs_p4 = mem['reach_rate'] - 0.20
    print(f"\n  Lift vs Phase 4: {lift_vs_p4*100:+.0f}pt")
    if mem['reach_rate'] > 0.5:
        verdict = "STRONG LONG-TERM MEMORY (> 50% reach with Hebbian)"
    elif mem['reach_rate'] > 0.35:
        verdict = "GOOD MEMORY (> Phase 4's 20% and ISS-trained's 35%)"
    elif mem['reach_rate'] > 0.25:
        verdict = "WEAK IMPROVEMENT over Phase 4 leaky"
    else:
        verdict = "NO CLEAR IMPROVEMENT from Hebbian"
    print(f"  Verdict: {verdict}")

    with open("hebbian_memory_result.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "elapsed_s": round(elapsed, 1),
            "training_eval": round(float(eval_s), 2),
            "training_guard": round(float(guard_s), 2),
            "sentinel_verdict": result.get("verdict"),
            "midpoint_hebbian": base,
            "hebbian_trained": mem,
            "best_params": [float(x) for x in best],
        }, f, indent=2)
    print("\n  Saved: hebbian_memory_result.json")


if __name__ == "__main__":
    main()
