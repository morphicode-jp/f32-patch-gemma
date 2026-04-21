"""Benchmark: CPU (phase_12_emergence) vs GPU (phase_12_emergence_gpu).

Run identical small workload on both paths, report effective rate.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.phase_11_cardinal_3d import (
    GravityUniverse, make_3d_universe_params,
)
from tamashii.phase_9_ecology import SHELLS_DEFAULT
from tamashii.gpu_runner import GPUBatchRunner


def bench_cpu(n_universes, n_agents, n_steps, seed):
    """CPU path: per-universe, per-agent sequential."""
    shells = SHELLS_DEFAULT
    cfg = "tamashii/configs"
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = GravityUniverse(u_id, wp, n_agents, seed + u_id * 101,
                             shells, cfg, cfg)
        universes.append(u)
    t0 = time.time()
    for u in universes:
        u.run_epoch(n_steps, log_every=n_steps)
    elapsed = time.time() - t0
    return elapsed, universes


def bench_gpu(n_universes, n_agents, n_steps, seed, device="cuda"):
    """GPU path: cross-universe batched via GPUBatchRunner."""
    shells = SHELLS_DEFAULT
    cfg = "tamashii/configs"
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = GravityUniverse(u_id, wp, n_agents, seed + u_id * 101,
                             shells, cfg, cfg)
        universes.append(u)

    all_agents = []
    for u in universes:
        for a in u.agents:
            all_agents.append(a)
    runner = GPUBatchRunner(all_agents, device=device)

    t0 = time.time()
    for step in range(n_steps):
        # Gather sensors
        agent_map = []
        for u in universes:
            for i in range(u.world.n_agents):
                agent_map.append((u, i))
        current_N = len(runner.agents)
        sensors_mega = np.zeros((current_N, 16), dtype=np.float64)
        for mega_idx in range(min(current_N, len(agent_map))):
            u_ref, local_i = agent_map[mega_idx]
            if u_ref.world.agent_alive[local_i]:
                sensors_mega[mega_idx] = u_ref.world.get_sensors(local_i)
        # Batched tick
        runner.tick_all(sensors_mega)
        # Per-universe world step (4-action)
        mega_idx = 0
        for u in universes:
            N_u = u.world.n_agents
            actions = []
            for i in range(N_u):
                if mega_idx < current_N:
                    ag = runner.agents[mega_idx]
                    S = ag.read_state()
                    actions.append((
                        float(np.clip(S[16], 0.0, 1.0)),
                        float(np.clip(S[17], 0.0, 1.0)),
                        float(np.clip(S[18], 0.0, 1.0)),
                        float(np.clip(S[19], 0.0, 1.0)) if len(S) > 19 else 0.0,
                    ))
                    mega_idx += 1
                else:
                    actions.append((0.5, 0.0, 0.0, 0.0))
            u.world.step(actions)
            while len(u.world.agents_external) > len(u.agents):
                new_child = u.world.agents_external[len(u.agents)]
                new_child.reset_episode()
                u.agents.append(new_child)
                runner.add_agent(new_child)
    elapsed = time.time() - t0
    return elapsed, universes


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("  CPU vs GPU BENCHMARK - 3D Cardinal emergence", flush=True)
    print("=" * 72, flush=True)

    configs = [
        ("4u × 5 agents = 20 total", 4, 5),
        ("4u × 15 agents = 60 total", 4, 15),
        ("4u × 30 agents = 120 total", 4, 30),
        ("8u × 20 agents = 160 total", 8, 20),
    ]

    print(f"  {'config':30s} {'CPU (s)':>10s} {'GPU (s)':>10s} "
          f"{'speedup':>10s} {'CPU/sec':>10s} {'GPU/sec':>10s}", flush=True)
    print("  " + "-" * 88, flush=True)

    for label, n_u, n_a in configs:
        try:
            cpu_s, _ = bench_cpu(n_u, n_a, args.steps, args.seed)
        except Exception as e:
            cpu_s = float('nan')
            print(f"  CPU {label} failed: {e}", flush=True)
        try:
            gpu_s, _ = bench_gpu(n_u, n_a, args.steps, args.seed, device="cuda")
        except Exception as e:
            gpu_s = float('nan')
            print(f"  GPU {label} failed: {e}", flush=True)
        total_agent_steps = n_u * n_a * args.steps
        cpu_rate = total_agent_steps / cpu_s if cpu_s > 0 else 0
        gpu_rate = total_agent_steps / gpu_s if gpu_s > 0 else 0
        speedup = cpu_s / gpu_s if gpu_s > 0 else 0
        print(f"  {label:30s} {cpu_s:>10.1f} {gpu_s:>10.1f} "
              f"{speedup:>10.2f}x {cpu_rate:>10.0f} {gpu_rate:>10.0f}",
              flush=True)


if __name__ == "__main__":
    main()
