"""train_cortical_3d_mimir.py — Train 60N cortical on 3D gravity world via mimir.

The 2D-pretrained cortical (cortical_sentinel_result.json, eval 87.14) doesn't
transfer cleanly to 3D because:
  - Tamashii sensor layout differs from FlyWorld
  - Sensor[15] (vertical food direction) is new in 3D
  - Motor output nodes need different weighting for 3D task

This script uses mimir (primary entry per CLAUDE.md Rule 9) to train 184D
cortical_params directly on a 3D food-gathering task.

Eval_fn: Build Tamashii agents with cortical core, run in GravityVoxelWorld3D
for N steps, score = food_eaten_total - 2.0 * deaths.

Budget: ~10 min. cheap_cascade owl+Reigen expected.

Output: tamashii/configs/cortical_core_brain_trained.json (overwrite 2D version).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "tamashii")

from cortical_brain import PARAM_RANGES, TOTAL_PARAMS
from tamashii.runner_3d import build_fluctlight
from tamashii.phase_9_ecology import SHELLS_DEFAULT
from world_3d_gravity import GravityVoxelWorld3D


def make_3d_eval_fn(n_episodes: int = 3, n_steps: int = 400,
                    n_agents: int = 3, seed_base: int = 42):
    """Build an eval_fn: cortical params (184D) → score (higher=better)."""
    configs_dir = "tamashii/configs"
    shells = SHELLS_DEFAULT

    def eval_fn(params):
        cp = list(params)
        total_score = 0.0
        for ep in range(n_episodes):
            # Fresh world each episode, fixed seeds
            ep_seed = seed_base + ep * 113

            # Build N agents with the trial cortical params
            # Write params to a temp overlay: set via build_fluctlight then override
            agents = []
            for a_idx in range(n_agents):
                a = build_fluctlight(shells, configs_dir, configs_dir,
                                      use_3d_brain=False,
                                      use_cortical_core=True)
                # Override cortical params to evaluate
                for s in a.shells:
                    if s.name == "cortical_core_brain":
                        s.kathara_params = np.asarray(cp, dtype=np.float64)
                        break
                agents.append(a)

            # Build 3D world
            world = GravityVoxelWorld3D(
                size=12, n_food=7, n_agents=n_agents,
                seed=ep_seed,
                gravity=-0.04, jump_impulse=0.5, max_height=4.0,
                vertical_food_frac=0.25, food_z_bimodal=True,
            )
            world.register_agents(agents)
            world.reset()
            for a in agents:
                a.reset_episode()

            # Run
            total_food = 0
            total_deaths = 0
            sensors_list = [world.get_sensors(i) for i in range(n_agents)]
            for step in range(n_steps):
                # Inject sensors
                for i, ag in enumerate(agents[:world.n_agents]):
                    if world.agent_alive[i]:
                        with ag._lock:
                            ag.S[0:16] = np.asarray(
                                sensors_list[i] if isinstance(sensors_list, list)
                                else sensors_list, dtype=np.float64)
                # Tick × 3
                for _ in range(3):
                    for i, ag in enumerate(agents[:world.n_agents]):
                        if world.agent_alive[i]:
                            ag.tick_once()
                # Extract 4 actions
                actions = []
                for i, ag in enumerate(agents[:world.n_agents]):
                    if world.agent_alive[i]:
                        S = ag.read_state()
                        actions.append((
                            float(np.clip(S[16], 0, 1)),
                            float(np.clip(S[17], 0, 1)),
                            float(np.clip(S[18], 0, 1)),
                            float(np.clip(S[19], 0, 1)) if len(S) > 19 else 0.0,
                        ))
                    else:
                        actions.append((0.5, 0.0, 0.0, 0.0))
                sensors_list, ate, done = world.step(actions)
                if world.stats()["n_alive"] == 0:
                    break

            stats = world.stats()
            food_total = sum(world.agent_food_eaten[i]
                              for i in range(world.n_agents))
            deaths = stats["n_deaths"]
            births = stats["n_births"]
            # Score aligned with Cardinal quality:
            #   rewards food (energy), survival (alive), reproduction (births)
            #   penalizes deaths
            score = (food_total
                      + 3.0 * births           # strong reproduction reward
                      + 0.5 * stats["n_alive"] # survival
                      - 1.5 * deaths           # mild death penalty
                      + 0.3 * stats["max_generation"])  # multi-gen bonus
            total_score += score

        # Average across episodes
        return total_score / n_episodes

    return eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time_budget", type=int, default=600,
                     help="seconds for mimir")
    ap.add_argument("--n_episodes",  type=int, default=3)
    ap.add_argument("--n_steps",     type=int, default=400)
    ap.add_argument("--n_agents",    type=int, default=3)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="tamashii/configs/cortical_core_brain_trained.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  TRAIN 60N CORTICAL on 3D via mimir", flush=True)
    print(f"  184D param × {args.time_budget}s budget", flush=True)
    print(f"  eval_fn: {args.n_episodes} episodes × {args.n_steps} steps × "
          f"{args.n_agents} agents in GravityVoxelWorld3D", flush=True)
    print("=" * 72, flush=True)

    eval_fn = make_3d_eval_fn(
        n_episodes=args.n_episodes, n_steps=args.n_steps,
        n_agents=args.n_agents, seed_base=args.seed)

    # Measure eval cost once
    print("\n  Timing baseline eval...", flush=True)
    t0 = time.time()
    baseline = eval_fn([(lo + hi) / 2 for lo, hi in PARAM_RANGES])
    eval_cost = time.time() - t0
    print(f"  baseline eval: {baseline:.3f} ({eval_cost:.2f}s)", flush=True)

    # Call mimir
    print(f"\n  Starting mimir with {args.time_budget}s budget...", flush=True)
    from twelve.agent.mimir import mimir
    t0 = time.time()
    result = mimir(
        eval_fn=eval_fn,
        param_ranges=PARAM_RANGES,
        time_budget=args.time_budget,
        experience_id="cortical_3d_mimir",
        thread_safe_eval=False,  # Tamashii state is not thread-safe
        n_seed_samples=30,  # Override owl's default max(5, n_dims+2)=186 for 184D
    )
    elapsed = time.time() - t0
    print(f"\n  mimir done in {elapsed:.0f}s", flush=True)

    print(f"  tool_used: {result.get('tool_used', 'unknown')}", flush=True)
    print(f"  route: {result.get('route', 'unknown')}", flush=True)
    print(f"  best_score: {result['best_score']:.3f}", flush=True)
    print(f"  confidence: {result.get('confidence', 'n/a')}", flush=True)
    print(f"  proxy_r2: {result.get('proxy_r2', 'n/a')}", flush=True)
    print(f"  Improvement vs baseline: {result['best_score'] - baseline:+.3f}",
          flush=True)

    # Save trained params
    trained = {
        "shell_name": "cortical_core_brain",
        "tick_ms": 50,
        "gain": 1.0,
        "params": {
            "sensor_start": 0, "sensor_end": 16,
            "motor_nav": 16, "motor_speed": 17,
            "motor_voice": 18, "motor_jump": 19,
            "firing_start": 20,
            "cortical_params": [float(p) for p in result["best_params"]],
            "enable_jump": True,  # now trained for jump
        },
        "source": "train_cortical_3d_mimir.py (3D gravity world)",
        "source_eval_score": float(result["best_score"]),
        "source_baseline": float(baseline),
        "source_elapsed_s": round(elapsed, 1),
        "source_tool_used": result.get("tool_used", "unknown"),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(trained, f, indent=2)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
