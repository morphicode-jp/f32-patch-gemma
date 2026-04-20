"""Phase 6 — Language / proto-signal emergence measurement.

Does the Fluctlight population's voice channel carry information?

Tests for:
  1. Self-information: MI(voice_i(t), own_food_direction_i(t))
     -> "I vocalize about my food situation"
  2. Cross-prediction: MI(voice_i(t-1), action_j(t)) for i ≠ j
     -> "Agent i's voice predicts agent j's next move" (= ToM signal)
  3. Coordination: MI(voice_i, voice_j) pairwise
     -> "Agents synchronize voice patterns"

Setup: 3 3D-trained Fluctlights in same 16×16 voxel world, no taboo.
Collect ~2000+ tuples, compute discrete MI via binning.

PASS (pre-registered):
  - At least one non-trivial MI > 0.05 bits (above random chance ~0.01)
  - Demonstrates voice channel carries information (not just noise)
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


def categorize(x: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Discretize continuous array into n_bins equal-count bins."""
    if x.size == 0:
        return x.astype(int)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    boundaries = np.quantile(x, quantiles)
    return np.digitize(x, boundaries)


def mutual_information(X: np.ndarray, Y: np.ndarray, n_bins: int = 4) -> float:
    """Discrete MI(X; Y) in bits. Uses equal-count binning."""
    if len(X) != len(Y) or len(X) < 10:
        return 0.0
    x_bin = categorize(np.asarray(X, dtype=float), n_bins)
    y_bin = categorize(np.asarray(Y, dtype=float), n_bins)
    # Joint histogram
    joint = np.zeros((n_bins, n_bins), dtype=np.float64)
    for xi, yi in zip(x_bin, y_bin):
        xi = min(max(xi, 0), n_bins - 1)
        yi = min(max(yi, 0), n_bins - 1)
        joint[xi, yi] += 1.0
    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return float(mi)


def run_language_episode(agents, world, n_steps: int = 400,
                         ticks_per_step: int = 3) -> dict:
    """Collect per-step signals for MI analysis."""
    N = world.n_agents
    sensors_list = world.reset()
    for a in agents:
        a.reset_episode()

    # Collectors per agent
    voice_series = [[] for _ in range(N)]
    nav_series = [[] for _ in range(N)]
    speed_series = [[] for _ in range(N)]
    food_left_series = [[] for _ in range(N)]
    food_right_series = [[] for _ in range(N)]
    olfactory_series = [[] for _ in range(N)]

    for step in range(n_steps):
        for i, agent in enumerate(agents):
            with agent._lock:
                agent.S[0:16] = np.asarray(sensors_list[i], dtype=np.float64)
        for _ in range(ticks_per_step):
            for agent in agents:
                agent.tick_once()
        actions = []
        for i, agent in enumerate(agents):
            S = agent.read_state()
            nav = float(np.clip(S[16], 0.0, 1.0))
            speed = float(np.clip(S[17], 0.0, 1.0))
            voice = float(np.clip(S[18], 0.0, 1.0))
            actions.append((nav, speed, voice))
            voice_series[i].append(voice)
            nav_series[i].append(nav)
            speed_series[i].append(speed)
            food_left_series[i].append(sensors_list[i][0])
            food_right_series[i].append(sensors_list[i][2])
            olfactory_series[i].append(sensors_list[i][10])
        sensors_list, ate, done = world.step(actions)
    return {
        "voice": [np.array(s) for s in voice_series],
        "nav": [np.array(s) for s in nav_series],
        "speed": [np.array(s) for s in speed_series],
        "food_left": [np.array(s) for s in food_left_series],
        "food_right": [np.array(s) for s in food_right_series],
        "olfactory": [np.array(s) for s in olfactory_series],
        "n_steps_run": step + 1,
        "food_eaten": world.food_eaten,
        "per_agent_food": list(world.agent_food_eaten),
    }


def compute_mi_matrix(data: dict, N: int) -> dict:
    """Compute suite of MI metrics."""
    out = {"self_mi": [], "cross_mi_prev": [[0.0] * N for _ in range(N)],
           "voice_voice_mi": [[0.0] * N for _ in range(N)],
           "voice_food_mi_max": 0.0}

    # 1. Self-MI: voice_i vs own food signals (concurrent)
    for i in range(N):
        mi_left = mutual_information(data["voice"][i], data["food_left"][i])
        mi_right = mutual_information(data["voice"][i], data["food_right"][i])
        mi_olf = mutual_information(data["voice"][i], data["olfactory"][i])
        best = max(mi_left, mi_right, mi_olf)
        out["self_mi"].append({
            "agent": i,
            "mi_left": mi_left, "mi_right": mi_right, "mi_olf": mi_olf,
            "max": best,
        })
        out["voice_food_mi_max"] = max(out["voice_food_mi_max"], best)

    # 2. Cross-prediction: voice_i(t-1) vs nav_j(t)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v_prev = data["voice"][i][:-1]
            nav_next = data["nav"][j][1:]
            mi = mutual_information(v_prev, nav_next)
            out["cross_mi_prev"][i][j] = mi

    # 3. Voice-voice
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            mi = mutual_information(data["voice"][i], data["voice"][j])
            out["voice_voice_mi"][i][j] = mi

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=400)
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--output", type=str, default="tamashii_phase_6_language.json")
    args = ap.parse_args()

    from world_3d import VoxelWorld3D

    print("=" * 70, flush=True)
    print("  PHASE 6 - Language / proto-signal MI measurement", flush=True)
    print(f"  N={args.n_agents} Fluctlights (3D-trained), "
          f"{args.episodes} eps × {args.n_steps} steps", flush=True)
    print("=" * 70, flush=True)

    shells = ["core_brain", "brainstem", "cerebellum", "salience",
              "hippocampus", "prefrontal", "dmn"]
    configs_dir = os.path.join(THIS_DIR, "configs")

    t0 = time.time()
    all_eps = []
    for ep in range(args.episodes):
        seed = args.seed_base + ep * 7
        world = VoxelWorld3D(
            size=16, n_food=5, n_walls=20, seed=seed, n_agents=args.n_agents)
        agents = [
            build_fluctlight(
                shells, configs_dir, trained_dir="tamashii/configs",
                use_hebbian_core=False, use_3d_brain=True,
            )
            for _ in range(args.n_agents)
        ]
        data = run_language_episode(agents, world, n_steps=args.n_steps)
        mi = compute_mi_matrix(data, args.n_agents)

        print(f"\n[ep {ep} seed={seed}] food={data['food_eaten']} "
              f"per-agent={data['per_agent_food']}", flush=True)
        print(f"  SELF-MI (voice → own food signals):", flush=True)
        for s in mi["self_mi"]:
            print(f"    agent{s['agent']}: max={s['max']:.3f} bits "
                  f"(left={s['mi_left']:.3f} right={s['mi_right']:.3f} "
                  f"olf={s['mi_olf']:.3f})", flush=True)
        print(f"  CROSS-MI matrix (voice_i(t-1) → nav_j(t)):", flush=True)
        for i in range(args.n_agents):
            row = " ".join(f"{mi['cross_mi_prev'][i][j]:.3f}"
                           for j in range(args.n_agents))
            print(f"    from{i}: {row}", flush=True)
        print(f"  VOICE-VOICE MI matrix:", flush=True)
        for i in range(args.n_agents):
            row = " ".join(f"{mi['voice_voice_mi'][i][j]:.3f}"
                           for j in range(args.n_agents))
            print(f"    {i}: {row}", flush=True)

        all_eps.append({"seed": seed, "food_eaten": data["food_eaten"],
                        "per_agent_food": data["per_agent_food"], "mi": mi})

    elapsed = time.time() - t0

    # Aggregate best MI across episodes
    best_self = max(
        (s["max"] for ep in all_eps for s in ep["mi"]["self_mi"]),
        default=0.0)
    best_cross = max(
        (row for ep in all_eps
         for r in ep["mi"]["cross_mi_prev"] for row in r),
        default=0.0)
    best_vv = max(
        (row for ep in all_eps
         for r in ep["mi"]["voice_voice_mi"] for row in r),
        default=0.0)

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 6 SUMMARY ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Best SELF-MI (voice ↔ own food):      {best_self:.4f} bits",
          flush=True)
    print(f"  Best CROSS-MI (voice → peer action):  {best_cross:.4f} bits",
          flush=True)
    print(f"  Best VOICE-VOICE sync:                {best_vv:.4f} bits",
          flush=True)

    pass_threshold = 0.05
    proto_language = (best_self > pass_threshold or
                      best_cross > pass_threshold or
                      best_vv > pass_threshold)
    if proto_language:
        print(f"\n  ✓ PROTO-LANGUAGE DETECTED: MI > {pass_threshold} bits",
              flush=True)
    else:
        print(f"\n  × No proto-language: all MI < {pass_threshold} bits "
              f"(voice channel = noise)", flush=True)

    out = {
        "n_agents": args.n_agents,
        "episodes": args.episodes,
        "n_steps": args.n_steps,
        "best_self_mi": best_self,
        "best_cross_mi": best_cross,
        "best_voice_voice_mi": best_vv,
        "proto_language_detected": proto_language,
        "all_episodes": all_eps,
        "elapsed_s": round(elapsed, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
