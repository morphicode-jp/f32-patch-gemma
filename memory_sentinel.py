"""memory_sentinel.py - Phase 4 MVP: evolve brain to navigate from memory.

Task: Food visible steps 0-4, then HIDDEN. Brain must reach food using
only internal state (memory) from the brief sighting.

eval_fn: final approach ratio after 40 steps (food hidden after step 5).
guard_fn: movement activity (prevent stuck behavior).

Key comparison:
  - Midpoint brain (no training): expected random wandering
  - Sentinel-optimized: can the leaky integration + weights encode
    food direction in sustained activity?

If optimized brain reaches food >50% after hidden phase -> memory works.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, PARAM_NAMES, simulate_step
from twelve.agent.sentinel import Sentinel
from memory_world import MemoryFlyWorld


def run_memory_episode(params, seed, reveal_steps=5, n_steps=40):
    p = np.array(params, dtype=np.float64)
    world = MemoryFlyWorld(seed=seed, reveal_steps=reveal_steps)
    sensors = world.reset()
    initial_dist = world.get_food_dist()
    min_dist_after_hidden = initial_dist  # track only after hidden phase
    states = np.zeros(12, dtype=np.float64)
    reached = False
    move_sum = 0.0
    reached_during_visible = False

    for step in range(n_steps):
        prev_pos = world.fly_pos.copy()
        states, firing = simulate_step(p, sensors, states)
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

    # Score: reward for approach AFTER food hidden (the memory portion)
    # Approach relative to dist AT reveal end
    approach_after = max(0.0, initial_dist - min_dist_after_hidden) / (initial_dist + 1e-6)
    ep_score = approach_after * 70.0 + (20.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
    # Penalty if reached in visible phase (task was trivial)
    if reached_during_visible:
        ep_score *= 0.5
    return ep_score, reached, reached_during_visible, min_dist_after_hidden


def make_eval_fn(n_episodes=3, n_steps=40):
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            score, _, _, _ = run_memory_episode(params, seed=ep * 7 + 13, n_steps=n_steps)
            total += score
        return total / n_episodes
    return eval_fn


def make_guard_fn(n_episodes=2, n_steps=30):
    """Guard = movement activity (prevent degenerate stuck brains)."""
    def guard_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = MemoryFlyWorld(seed=ep * 11 + 5, reveal_steps=5)
            sensors = world.reset()
            states = np.zeros(12, dtype=np.float64)
            move = 0.0
            for _ in range(n_steps):
                prev = world.fly_pos.copy()
                states, firing = simulate_step(p, sensors, states)
                sensors, _, reached, _ = world.step(float(firing[5]), float(firing[11]))
                move += float(np.linalg.norm(world.fly_pos - prev))
                if reached: break
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def evaluate_brain(params, n_seeds=20, label=""):
    """Test on 20 unseen seeds. Report reach rate and mean final distance."""
    scores = []
    reaches = []
    final_dists = []
    reached_during_visible_count = 0
    for seed in range(2000, 2000 + n_seeds):
        score, reached, during_vis, final_d = run_memory_episode(params, seed, n_steps=40)
        scores.append(score)
        reaches.append(reached)
        final_dists.append(final_d)
        if during_vis:
            reached_during_visible_count += 1
    mean_score = float(np.mean(scores))
    reach_rate = sum(reaches) / len(reaches)
    mean_final_dist = float(np.mean(final_dists))
    print(f"  [{label}] score={mean_score:.2f}  reach={reach_rate*100:.0f}%  "
          f"mean_final_dist={mean_final_dist:.2f}  "
          f"trivially_reached={reached_during_visible_count}/{n_seeds}")
    return {
        "label": label,
        "mean_score": round(mean_score, 2),
        "reach_rate": round(reach_rate, 3),
        "mean_final_dist": round(mean_final_dist, 2),
        "trivially_reached": reached_during_visible_count,
    }


def main():
    print("=" * 70)
    print("  Phase 4 MVP: Memory-driven navigation")
    print("  Food visible steps 0-4, then HIDDEN. Brain navigates from memory.")
    print("=" * 70)

    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]

    print("\n[baseline] midpoint brain:")
    base = evaluate_brain(mid, n_seeds=20, label="midpoint")

    # Also test biological-trained (no memory task, from brain_sentinel_biological)
    if os.path.exists("brain_sentinel_biological_result.json"):
        with open("brain_sentinel_biological_result.json") as f:
            d = json.load(f)
        iss_params = d.get("best_ever_params") or d.get("best_params")
        print("\n[no-memory-training baseline] ISS-trained brain:")
        no_mem = evaluate_brain(iss_params, n_seeds=20, label="iss_trained")

    print("\n[training] Sentinel with memory eval_fn (180s)...")
    t0 = time.time()
    result = Sentinel(
        eval_fn=make_eval_fn(n_episodes=3, n_steps=40),
        guard_fn=make_guard_fn(n_episodes=2, n_steps=30),
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id="memory_brain",
        learn=True,
    ).run(time_budget=180, verbose=False)
    elapsed = time.time() - t0

    best = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")
    print(f"  training done: eval={eval_s:.2f}  guard={guard_s:.2f}  elapsed={elapsed:.0f}s")
    print(f"  verdict: {result.get('verdict')}")

    print("\n[evaluation] memory-trained brain on 20 unseen seeds:")
    mem = evaluate_brain(best, n_seeds=20, label="memory_trained")

    # Verdict
    print("\n" + "=" * 70)
    print("  Memory emergence verdict")
    print("=" * 70)
    print(f"  Midpoint reach:       {base['reach_rate']*100:.0f}%")
    if os.path.exists("brain_sentinel_biological_result.json"):
        print(f"  ISS-trained reach:    {no_mem['reach_rate']*100:.0f}%")
    print(f"  Memory-trained reach: {mem['reach_rate']*100:.0f}%")
    improvement = mem['reach_rate'] - base['reach_rate']
    print(f"\n  Memory training lift: {improvement*100:+.0f}pt vs midpoint")

    if mem['reach_rate'] > 0.5:
        verdict = "STRONG MEMORY (reach > 50% despite hidden food)"
    elif mem['reach_rate'] > 0.3:
        verdict = "WEAK MEMORY (above chance, below half)"
    elif mem['reach_rate'] > base['reach_rate'] + 0.1:
        verdict = "INCREMENTAL (training improves but memory unclear)"
    else:
        verdict = "NO MEMORY (no clear gain)"
    print(f"  Verdict: {verdict}")

    with open("memory_result.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "elapsed_s": round(elapsed, 1),
            "training_eval": round(float(eval_s), 2),
            "training_guard": round(float(guard_s), 2),
            "sentinel_verdict": result.get("verdict"),
            "baseline_midpoint": base,
            "baseline_iss_trained": no_mem if os.path.exists("brain_sentinel_biological_result.json") else None,
            "memory_trained": mem,
            "best_params": [float(x) for x in best],
        }, f, indent=2)
    print("\n  Saved: memory_result.json")


if __name__ == "__main__":
    main()
