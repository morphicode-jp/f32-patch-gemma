"""Phase 7 — Autobiographical memory test (criterion 5).

Hippocampus persistent_memory=true: slots survive reset_episode().
Test: over N episodes, does memory BUILD UP and get REUSED?

Success criteria:
  1. n_stored grows across episodes (building autobiography)
  2. recall_events > 0 in later episodes (reusing earlier memories)
  3. recall rate increases over time (familiar world → more reactivation)
  4. Agent still functions (food eaten doesn't collapse)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.runner_3d import build_fluctlight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--same_world", action="store_true", default=True,
                    help="Use same seed every episode (memorize same env)")
    ap.add_argument("--no_same_world", dest="same_world", action="store_false")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_7_autobio.json")
    args = ap.parse_args()

    from world_3d import VoxelWorld3D

    shells = ["core_brain", "brainstem", "cerebellum", "salience",
              "hippocampus", "prefrontal", "dmn"]
    configs_dir = os.path.join(THIS_DIR, "configs")

    # Build ONE agent, reuse across episodes so hippocampus persists
    agent = build_fluctlight(
        shells, configs_dir, trained_dir="tamashii/configs",
        use_hebbian_core=False, use_3d_brain=True,
    )
    # Find the hippocampus shell
    hip_shell = None
    for s in agent.shells:
        if s.name == "hippocampus":
            hip_shell = s
            break
    assert hip_shell is not None

    print("=" * 70, flush=True)
    print(f"  PHASE 7 - Autobiographical memory (criterion 5)", flush=True)
    print(f"  episodes={args.episodes} steps={args.n_steps} "
          f"same_world={args.same_world}", flush=True)
    print(f"  persistent_memory: {getattr(hip_shell, 'persistent_memory', False)}",
          flush=True)
    print("=" * 70, flush=True)

    per_episode = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed_base if args.same_world else args.seed_base + ep * 7
        world = VoxelWorld3D(
            size=16, n_food=5, n_walls=20, seed=seed, n_agents=1)
        sensors = world.reset()
        agent.reset_episode()

        stats_before = hip_shell.memory_stats()

        for step in range(args.n_steps):
            with agent._lock:
                agent.S[0:16] = np.asarray(sensors, dtype=np.float64)
            for _ in range(3):
                agent.tick_once()
            S = agent.read_state()
            nav = float(np.clip(S[16], 0.0, 1.0))
            speed = float(np.clip(S[17], 0.0, 1.0))
            voice = float(np.clip(S[18], 0.0, 1.0))
            sensors, ate, done = world.step(nav, speed, voice)
            if done:
                break

        stats_after = hip_shell.memory_stats()
        food = world.food_eaten

        growth = stats_after["n_stored"] - stats_before["n_stored"]
        recalls = stats_after["recall_events"]

        print(f"\n[ep {ep} seed={seed}] food={food} "
              f"stored: {stats_before['n_stored']}→{stats_after['n_stored']} "
              f"(+{growth}), recall_events={recalls}", flush=True)

        per_episode.append({
            "episode": ep,
            "seed": seed,
            "food_eaten": food,
            "n_stored_before": stats_before["n_stored"],
            "n_stored_after": stats_after["n_stored"],
            "n_stored_growth": growth,
            "recall_events": recalls,
            "recall_rate": recalls / max(1, args.n_steps),
        })

    elapsed = time.time() - t0

    # Analysis
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 7 SUMMARY ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  {'ep':>3s} {'food':>5s} {'stored':>8s} {'+new':>5s} "
          f"{'recalls':>8s} {'recall_rate':>12s}", flush=True)
    for p in per_episode:
        print(f"  {p['episode']:>3d} {p['food_eaten']:>5d} "
              f"{p['n_stored_after']:>8d} {p['n_stored_growth']:>+5d} "
              f"{p['recall_events']:>8d} {p['recall_rate']:>11.3f}",
              flush=True)

    # Success check
    final_stored = per_episode[-1]["n_stored_after"]
    total_recalls = sum(p["recall_events"] for p in per_episode)
    # Recall-per-episode trend: late eps recall more than early ones?
    mid = len(per_episode) // 2
    early_recalls = sum(p["recall_events"] for p in per_episode[:mid])
    late_recalls = sum(p["recall_events"] for p in per_episode[mid:])
    total_food = sum(p["food_eaten"] for p in per_episode)

    print(f"\n  Final memory size:    {final_stored}/8 slots", flush=True)
    print(f"  Total recall events:  {total_recalls}", flush=True)
    print(f"  Early half recalls:   {early_recalls}", flush=True)
    print(f"  Late half recalls:    {late_recalls}", flush=True)
    print(f"  Total food eaten:     {total_food}", flush=True)

    pass_stored = final_stored >= 4
    pass_recall = total_recalls > 0
    pass_trend = late_recalls >= early_recalls
    pass_function = total_food > 0
    all_pass = pass_stored and pass_recall and pass_trend and pass_function
    print(f"\n  PASS: stored>=4: {pass_stored}  recall>0: {pass_recall}  "
          f"late>=early: {pass_trend}  functional: {pass_function}",
          flush=True)
    if all_pass:
        print(f"\n  ✓ CRITERION 5 PASS: autobiographical memory WORKS",
              flush=True)
    else:
        print(f"\n  × partial criterion 5", flush=True)

    out = {
        "per_episode": per_episode,
        "final_stored": final_stored,
        "total_recalls": total_recalls,
        "early_recalls": early_recalls,
        "late_recalls": late_recalls,
        "total_food": total_food,
        "criterion_5_pass": all_pass,
        "elapsed_s": round(elapsed, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
