"""Phase 10.0 end-to-end benchmark: full tamashii stack CPU vs GPU.

Measures realistic speedup: core_brain GPU batched + other shells CPU
(which is the actual production path).

Compares at matching N_agents:
  CPU path: sequential Tamashii.tick_once() for each agent, each step
  GPU path: GPUBatchRunner.tick_all() (batched core_brain + seq other shells)
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
from tamashii.gpu_runner import GPUBatchRunner


SHELLS = ["core_brain", "brainstem", "cerebellum", "salience",
          "hippocampus", "prefrontal", "dmn"]


def bench_cpu_path(agents, n_steps: int, sensors_matrix_np: np.ndarray):
    """Sequential CPU path: each agent ticks individually."""
    N = len(agents)
    for ag in agents:
        ag.reset_episode()
    t0 = time.time()
    for step in range(n_steps):
        for i, ag in enumerate(agents):
            with ag._lock:
                ag.S[0:16] = sensors_matrix_np[i]
            ag.tick_once()
    return time.time() - t0


def bench_gpu_path(agents, n_steps: int, sensors_matrix_np: np.ndarray,
                    device: str):
    """Batched GPU path via GPUBatchRunner."""
    runner = GPUBatchRunner(agents, device=device)
    for ag in agents:
        ag.reset_episode()
    runner.reset()
    t0 = time.time()
    for step in range(n_steps):
        runner.tick_all(sensors_matrix_np)
    return time.time() - t0


def build_agent_pool(N: int, configs_dir: str):
    print(f"Building {N} agents...", flush=True)
    agents = []
    for _ in range(N):
        a = build_fluctlight(SHELLS, configs_dir, trained_dir="tamashii/configs",
                              use_hebbian_core=False, use_3d_brain=True)
        agents.append(a)
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=500)
    ap.add_argument("--batches", type=str, default="3,10,50,100,300")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_10_0_end2end.json")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("PyTorch not available", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    configs_dir = os.path.join(THIS_DIR, "configs")
    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    rng = np.random.default_rng(42)
    results = []

    print(f"\n  N     CPU (s)      GPU (s)    Speedup  CPU/agent_step   GPU/agent_step",
          flush=True)
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*16} {'-'*16}", flush=True)

    for N in batches:
        agents = build_agent_pool(N, configs_dir)
        sensors_np = rng.uniform(0, 1, size=(N, 16)).astype(np.float64)

        # CPU path
        t_cpu = bench_cpu_path(agents, args.n_steps, sensors_np)

        # GPU path (rebuild agents to reset state)
        agents2 = build_agent_pool(N, configs_dir)
        t_gpu = bench_gpu_path(agents2, args.n_steps, sensors_np, device)

        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        cpu_per = (t_cpu / (args.n_steps * N)) * 1000  # ms
        gpu_per = (t_gpu / (args.n_steps * N)) * 1000
        print(f"  {N:4d}   {t_cpu:8.3f}  {t_gpu:8.3f}  {speedup:6.1f}x  "
              f"{cpu_per:10.4f} ms      {gpu_per:10.4f} ms", flush=True)
        results.append({
            "N": N, "n_steps": args.n_steps,
            "cpu_s": round(t_cpu, 3),
            "gpu_s": round(t_gpu, 3),
            "speedup": round(speedup, 2),
            "cpu_per_agent_ms": round(cpu_per, 4),
            "gpu_per_agent_ms": round(gpu_per, 4),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"device": device, "results": results}, f, indent=2)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
