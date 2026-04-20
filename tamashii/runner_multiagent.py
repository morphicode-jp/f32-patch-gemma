"""Multi-agent runner — Phase A smoke + later demos.

Each agent has its own Tamashii instance. Shells tick synchronously for
determinism. World provides 16D sensors; agents write motor to S[16:19]
which is read as (nav, speed, voice) tuple.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.core import Tamashii
from tamashii.shell_base import load_shell_from_config
from tamashii.shells.brainstem import Brainstem
from tamashii.shells.cerebellum import Cerebellum
from tamashii.shells.core_brain import CoreBrain
from tamashii.shells.hippocampus import Hippocampus
from tamashii.shells.salience import Salience


SHELL_REGISTRY = {
    "core_brain": CoreBrain,
    "brainstem": Brainstem,
    "cerebellum": Cerebellum,
    "salience": Salience,
    "hippocampus": Hippocampus,
}


def build_agent(shell_names: list[str], configs_dir: str,
                trained_dir: str | None = None, D: int = 192) -> Tamashii:
    """Build a Tamashii agent with the named shells."""
    shells = []
    for name in shell_names:
        cls = SHELL_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown shell: {name}")
        base_json = os.path.join(configs_dir, f"{name}.json")
        trained_json = None
        if trained_dir:
            cand = os.path.join(trained_dir, f"{name}_trained.json")
            if os.path.exists(cand):
                trained_json = cand
        shell = load_shell_from_config(cls, base_json, trained_json)
        shells.append(shell)
    return Tamashii(shells=shells, D=D, async_mode=False)


def extract_action(agent: Tamashii) -> tuple[float, float, float]:
    """Read (nav, speed, voice) from S[16:19]."""
    S = agent.read_state()
    nav = float(np.clip(S[16], 0.0, 1.0))
    speed = float(np.clip(S[17], 0.0, 1.0))
    voice = float(np.clip(S[18], 0.0, 1.0))
    return nav, speed, voice


def write_sensors(agent: Tamashii, sensors: np.ndarray):
    """Overwrite S[0:16] with fresh sensor vector (world → brain).

    This is a direct write (not via shell) because sensors come from outside.
    """
    with agent._lock:
        agent.S[0:16] = np.asarray(sensors, dtype=np.float64)


def run_episode(agents: list[Tamashii], world_cls, N_AGENTS: int = 3,
                n_steps: int = 50, seed: int = 0, shell_ticks_per_step: int = 3,
                verbose: bool = False) -> dict:
    """Run one episode. Each world step, shells tick N times internally."""
    world = world_cls(n_agents=N_AGENTS, seed=seed)
    sensors_list = world.reset()

    # Reset agents
    for a in agents:
        a.reset_episode()

    initial_dists = [world.get_food_dist(i) for i in range(N_AGENTS)]
    min_dists = list(initial_dists)
    reached_steps = [-1] * N_AGENTS
    action_log = []
    delta_norms_log = []

    for step in range(n_steps):
        # 1. Write sensors to each agent's S
        for i, agent in enumerate(agents):
            write_sensors(agent, sensors_list[i])

        # 2. Tick all shells K times (lets core → motor → brainstem cycle stabilize)
        for _ in range(shell_ticks_per_step):
            for agent in agents:
                agent.tick_once()

        # 3. Extract actions
        actions = [extract_action(agent) for agent in agents]
        action_log.append(actions)
        delta_norms_log.append([a.shell_delta_norms() for a in agents])

        # 4. Step world
        sensors_list, all_reached, reached_flags = world.step(actions)

        # 5. Track minima
        for i in range(N_AGENTS):
            d = world.get_food_dist(i)
            if d < min_dists[i]:
                min_dists[i] = d
            if reached_flags[i] and reached_steps[i] == -1:
                reached_steps[i] = step

        if verbose and step % 10 == 0:
            act_str = " ".join(
                f"a{i}=({n:.2f},{s:.2f},{v:.2f})"
                for i, (n, s, v) in enumerate(actions)
            )
            print(f"  step {step}: {act_str}", flush=True)

        if all_reached:
            break

    reached = [bool(world.reached[i]) for i in range(N_AGENTS)]
    reach_rate = float(np.mean(reached))
    final_dists = [world.get_food_dist(i) for i in range(N_AGENTS)]
    approaches = [
        max(0.0, initial_dists[i] - min_dists[i]) / (initial_dists[i] + 1e-6)
        for i in range(N_AGENTS)
    ]

    # S divergence check (did brainstem keep it bounded?)
    final_S_norms = [float(np.linalg.norm(a.read_state())) for a in agents]
    S_ranges = [
        (float(a.S.min()), float(a.S.max())) for a in agents
    ]

    # Motor diversity: did actions vary?
    nav_series = np.array([a[0] for _act in action_log for a in _act])
    motor_std = float(np.std(nav_series))

    return {
        "reached": reached,
        "reach_rate": reach_rate,
        "approaches": approaches,
        "reached_steps": reached_steps,
        "final_dists": final_dists,
        "n_steps_run": len(action_log),
        "final_S_norms": final_S_norms,
        "S_ranges": S_ranges,
        "motor_std": motor_std,
        "delta_norms_final": delta_norms_log[-1] if delta_norms_log else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", type=str, default="core_brain,brainstem",
                    help="comma-separated shell names")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--configs_dir", type=str,
                    default=os.path.join(THIS_DIR, "configs"))
    ap.add_argument("--trained_dir", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from multi_agent_world_N import MultiAgentCoopWorldN

    shell_names = [s.strip() for s in args.shells.split(",") if s.strip()]
    print(f"Shells: {shell_names}", flush=True)

    # Build N_AGENTS independent Tamashii instances
    agents = [
        build_agent(shell_names, args.configs_dir, args.trained_dir)
        for _ in range(args.n_agents)
    ]
    for i, a in enumerate(agents):
        print(f"  agent{i}: {a}", flush=True)

    episodes = []
    t0 = time.time()
    for ep in range(args.episodes):
        seed = args.seed_base + ep * 7
        result = run_episode(
            agents, MultiAgentCoopWorldN,
            N_AGENTS=args.n_agents, n_steps=args.n_steps,
            seed=seed, verbose=args.verbose,
        )
        episodes.append(result)
        print(f"[ep {ep} seed={seed}] reach_rate={result['reach_rate']*100:.0f}% "
              f"approaches={[f'{a*100:.0f}%' for a in result['approaches']]} "
              f"motor_std={result['motor_std']:.3f} "
              f"S_norm={[f'{n:.1f}' for n in result['final_S_norms']]} "
              f"S_range={result['S_ranges']}", flush=True)

    elapsed = time.time() - t0
    mean_reach = float(np.mean([e["reach_rate"] for e in episodes]))
    mean_motor_std = float(np.mean([e["motor_std"] for e in episodes]))
    print(f"\nSummary: {args.episodes} episodes in {elapsed:.1f}s", flush=True)
    print(f"  mean reach_rate: {mean_reach*100:.1f}%", flush=True)
    print(f"  mean motor_std:  {mean_motor_std:.3f}", flush=True)
    print(f"  final_S_norms last ep: {episodes[-1]['final_S_norms']}", flush=True)


if __name__ == "__main__":
    main()
