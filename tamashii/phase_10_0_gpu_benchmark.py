"""Phase 10.0 GPU benchmark: CPU numpy vs GPU batched simulate_step_16.

Measures per-step latency of running N agents' brain.
- CPU: loop over agents, call numpy simulate_step_16
- GPU: single batched call on CUDA
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

from kathara16_brain import (
    simulate_step_16, PARAM_RANGES_16, make_inhibit_sign_16,
)


def make_random_batch(B: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    params = np.stack([
        np.array([rng.uniform(lo, hi) for lo, hi in PARAM_RANGES_16],
                 dtype=np.float32)
        for _ in range(B)
    ])
    inputs = rng.uniform(0, 1, size=(B, 16)).astype(np.float32)
    states = np.zeros((B, 16), dtype=np.float32)
    return params, inputs, states


def bench_cpu(params_np, inputs_np, states_np, n_steps: int):
    """Loop over agents for n_steps, timed."""
    B = params_np.shape[0]
    inh = make_inhibit_sign_16().astype(np.float32)
    # Warmup
    for a in range(min(B, 3)):
        _ = simulate_step_16(params_np[a], inputs_np[a], states_np[a], inh)

    states = states_np.copy().astype(np.float64)
    params = params_np.astype(np.float64)
    inputs = inputs_np.astype(np.float64)
    inh64 = inh.astype(np.float64)

    t0 = time.time()
    for step in range(n_steps):
        for a in range(B):
            ns, _ = simulate_step_16(params[a], inputs[a], states[a], inh64)
            states[a] = ns
    elapsed = time.time() - t0
    return elapsed


def bench_gpu(params_np, inputs_np, states_np, n_steps: int, device: str):
    """Batched GPU calls for n_steps."""
    import torch
    from kathara16_brain_gpu import simulate_step_16_batched

    params = torch.from_numpy(params_np).to(device)
    inputs = torch.from_numpy(inputs_np).to(device)
    states = torch.from_numpy(states_np).to(device)
    inh = torch.from_numpy(make_inhibit_sign_16().astype(np.float32)).to(device)

    # Warmup
    for _ in range(3):
        states, _ = simulate_step_16_batched(params, inputs, states, inh)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.time()
    for step in range(n_steps):
        states, _ = simulate_step_16_batched(params, inputs, states, inh)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=1000)
    ap.add_argument("--batches", type=str, default="1,10,100,500,1000,2000",
                    help="comma-separated B sizes")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_10_0_benchmark.json")
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

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    results = []

    print(f"\n  Config        CPU (s)      GPU (s)    Speedup  CPU/agent(ms/step)  GPU/agent(ms/step)",
          flush=True)
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*20} {'-'*20}", flush=True)

    for B in batches:
        params_np, inputs_np, states_np = make_random_batch(B)
        # CPU: skip if prohibitively slow
        if B <= 500 or args.n_steps <= 200:
            t_cpu = bench_cpu(params_np, inputs_np, states_np, args.n_steps)
        else:
            # Sample small portion, extrapolate
            sample_steps = 50
            t_sample = bench_cpu(params_np, inputs_np, states_np, sample_steps)
            t_cpu = t_sample * args.n_steps / sample_steps
            print(f"  (CPU extrapolated from {sample_steps}-step sample)",
                  flush=True)

        t_gpu = bench_gpu(params_np, inputs_np, states_np, args.n_steps, device)
        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        cpu_per = (t_cpu / (args.n_steps * B)) * 1000  # ms per agent step
        gpu_per = (t_gpu / (args.n_steps * B)) * 1000
        print(f"  N={B:4d} step={args.n_steps:4d}  "
              f"{t_cpu:8.3f}s  {t_gpu:8.3f}s  {speedup:6.1f}x  "
              f"{cpu_per:10.4f}ms       {gpu_per:10.4f}ms", flush=True)
        results.append({
            "B": B, "n_steps": args.n_steps,
            "cpu_s": round(t_cpu, 4),
            "gpu_s": round(t_gpu, 4),
            "speedup": round(speedup, 2),
            "cpu_per_agent_step_ms": round(cpu_per, 4),
            "gpu_per_agent_step_ms": round(gpu_per, 4),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"device": device, "results": results}, f, indent=2)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
