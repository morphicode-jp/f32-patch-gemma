"""Phase 13 - ISS-quality Cardinal: evolve shell topology toward human brain.

paper §4.2 + §4.0e: ISS as quality function for AT/Cardinal-style meta-evolution.
When applied to Transformer architecture, AT autonomously converged on human
brain structural constants (C=0.284, I=20%, hub=12.4%) without being told.
This reproduces that finding on Tamashii shell architecture.

Key differences from prior Cardinal (phase_12_emergence_gpu):
  - Quality function = ISS v2 of the shell dependency graph (not food/births)
  - "Genome" = shell layer assignments + inhibition flags + connectivity hints
  - Population = multiple shell-architecture variants
  - Selection: worst ISS → variant of best ISS

Universe params = shell topology variation:
  - layer_shifts: perturbation to each shell's layer (±1)
  - inhibition_flips: toggle shell_sign for candidate shells
  - gain_scales: adjust inter-layer gain (proxy for connection strength)

Expected outcome (per paper §4.0e):
  ISS v2 converges upward; specific metrics (H, hub, I) stabilize near
  paper's human reference values.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Callable

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.runner_3d import build_fluctlight
from tamashii.phase_9_ecology import SHELLS_DEFAULT
from iss_compute import measure_tamashii_graph, compute_iss, HUMAN_REF


# Shell roster for ISS Cardinal experiments
SHELL_ROSTER = [
    "core_brain", "brainstem", "cerebellum", "salience",
    "hippocampus", "prefrontal", "dmn", "taboo",
    "mimir_shell", "inhibition_L1", "inhibition_L3", "inhibition_L4",
]

# Layer bounds for mutation
LAYER_MIN = 0
LAYER_MAX = 5


def make_baseline_genome() -> dict:
    """Default topology: current committed M3 layer assignments."""
    return {
        "layer_assignments": {
            "brainstem": 0, "dmn": 0,
            "core_brain": 1, "inhibition_L1": 1,
            "salience": 2, "taboo": 2,
            "cerebellum": 3, "inhibition_L3": 3,
            "hippocampus": 4, "prefrontal": 4, "inhibition_L4": 4,
            "mimir_shell": 5,
        },
        "shell_signs": {
            "core_brain": +1, "brainstem": +1, "cerebellum": +1,
            "salience": +1, "hippocampus": +1, "prefrontal": +1,
            "dmn": +1, "mimir_shell": +1,
            "taboo": -1, "inhibition_L1": -1,
            "inhibition_L3": -1, "inhibition_L4": -1,
        },
        "gain_scales": {s: 1.0 for s in SHELL_ROSTER},
    }


def mutate_genome(parent: dict, sigma: float = 0.3, seed: int = 0) -> dict:
    """Perturb a topology genome.

    Mutations:
      - 30% chance: shift one shell's layer by ±1 (clamped)
      - 20% chance: flip one non-core shell's sign
      - 50%: perturb gain_scales (log-normal noise)
    """
    rng = np.random.default_rng(seed)
    child = copy.deepcopy(parent)

    # Layer shift mutations
    for shell_name in list(child["layer_assignments"].keys()):
        if rng.random() < 0.3 * sigma:
            direction = int(rng.choice([-1, +1]))
            new_layer = child["layer_assignments"][shell_name] + direction
            new_layer = max(LAYER_MIN, min(LAYER_MAX, new_layer))
            child["layer_assignments"][shell_name] = new_layer

    # Sign flip (only for non-core shells to preserve function)
    CORE_EXCITATORY = {"core_brain"}  # don't flip core brain
    for shell_name in list(child["shell_signs"].keys()):
        if shell_name in CORE_EXCITATORY:
            continue
        if rng.random() < 0.2 * sigma:
            child["shell_signs"][shell_name] = (
                -child["shell_signs"][shell_name])

    # Gain scale perturbation
    for shell_name in child["gain_scales"]:
        if rng.random() < 0.5 * sigma:
            child["gain_scales"][shell_name] *= float(
                np.exp(rng.normal(0, 0.3)))
            child["gain_scales"][shell_name] = float(np.clip(
                child["gain_scales"][shell_name], 0.1, 5.0))

    return child


def build_agent_from_genome(genome: dict):
    """Instantiate a Tamashii agent with topology from genome."""
    agent = build_fluctlight(
        SHELL_ROSTER, "tamashii/configs", "tamashii/configs",
        use_3d_brain=True)
    for shell in agent.shells:
        if shell.name in genome["layer_assignments"]:
            shell.layer = int(genome["layer_assignments"][shell.name])
        if shell.name in genome["shell_signs"]:
            shell.shell_sign = int(genome["shell_signs"][shell.name])
        if shell.name in genome["gain_scales"]:
            shell.gain = float(shell.gain *
                                genome["gain_scales"][shell.name])
    return agent


def iss_quality(genome: dict, warmup: int = 60) -> tuple[float, dict]:
    """Build agent from genome, measure ISS v2. Higher = better."""
    try:
        agent = build_agent_from_genome(genome)
        graph = measure_tamashii_graph(agent, n_warmup_steps=warmup)
        metrics = compute_iss(graph)
        return float(metrics["iss_v2"]), metrics
    except Exception as e:
        return 0.0, {"error": str(e)}


def run_iss_cardinal(n_universes: int = 6, n_epochs: int = 20,
                       sigma: float = 0.4, seed: int = 42,
                       warmup: int = 60,
                       output: str = "phase_13_iss_cardinal.json") -> dict:
    print("=" * 72, flush=True)
    print(f"  PHASE 13 - ISS-quality Cardinal (paper §4.0e reproduction)",
          flush=True)
    print(f"  {n_universes} universes × {n_epochs} epochs", flush=True)
    print(f"  Quality fn: ISS v2 (uncapped)", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)

    # Initialize population: 1 baseline + rest mutated
    population = [make_baseline_genome()]
    for _ in range(n_universes - 1):
        population.append(mutate_genome(population[0], sigma=sigma,
                                          seed=int(rng.integers(1e9))))

    # Evaluate initial
    qualities = []
    metrics_list = []
    for i, g in enumerate(population):
        q, m = iss_quality(g, warmup=warmup)
        qualities.append(q)
        metrics_list.append(m)
    print(f"\n  Initial qualities (ISS v2):", flush=True)
    for i, (q, m) in enumerate(zip(qualities, metrics_list)):
        if "error" in m:
            print(f"    u{i}: ERROR {m['error'][:60]}", flush=True)
        else:
            print(f"    u{i}: ISS v2={q:.2f}  H={m['H']:.1f} L={m['L']:.2f} "
                  f"C={m['C']:.3f} I={m['I']*100:.1f}% hub={m['hub']:.1f}%",
                  flush=True)

    history = []
    t_start = time.time()

    for epoch in range(n_epochs):
        # Snapshot
        best_idx = int(np.argmax(qualities))
        worst_idx = int(np.argmin(qualities))
        history.append({
            "epoch": epoch,
            "qualities": list(qualities),
            "best_idx": best_idx,
            "best_iss_v2": float(qualities[best_idx]),
            "best_metrics": metrics_list[best_idx],
            "mean_iss_v2": float(np.mean(qualities)),
        })

        # Meta-evolve: worst gets variant of best
        if epoch < n_epochs - 1:
            new_genome = mutate_genome(
                population[best_idx], sigma=sigma,
                seed=seed + epoch * 73)
            population[worst_idx] = new_genome
            new_q, new_m = iss_quality(new_genome, warmup=warmup)
            qualities[worst_idx] = new_q
            metrics_list[worst_idx] = new_m

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            bm = metrics_list[best_idx]
            print(f"\n  ep{epoch}: best u{best_idx} ISS v2={qualities[best_idx]:.2f} "
                  f"(mean={np.mean(qualities):.2f})", flush=True)
            if "error" not in bm:
                print(f"    best: H={bm['H']:.1f} L={bm['L']:.2f} "
                      f"C={bm['C']:.3f} I={bm['I']*100:.1f}% hub={bm['hub']:.1f}%",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"\n  DONE in {elapsed:.0f}s", flush=True)

    # Final analysis
    final_best_idx = int(np.argmax(qualities))
    final_m = metrics_list[final_best_idx]
    print(f"\n  FINAL BEST (u{final_best_idx}):", flush=True)
    if "error" not in final_m:
        print(f"    ISS v2     : {qualities[final_best_idx]:.2f}",  flush=True)
        print(f"    H (paper 6): {final_m['H']:.2f}", flush=True)
        print(f"    L (paper 2.14): {final_m['L']:.3f}", flush=True)
        print(f"    C (paper 0.283): {final_m['C']:.3f}", flush=True)
        print(f"    I (paper 0.20): {final_m['I']*100:.1f}%", flush=True)
        print(f"    hub (paper 12.4%): {final_m['hub']:.1f}%", flush=True)

        # Convergence check (paper §4.0e)
        dH = abs(final_m['H'] - 6.0)
        dhub = abs(final_m['hub'] - 12.4)
        dI = abs(final_m['I']*100 - 20.0)
        print(f"\n  Paper §4.0e convergence check:", flush=True)
        def mark(delta, pass_t, warn_t):
            return "[HIT]" if delta <= pass_t else ("[near]" if delta <= warn_t else "[MISS]")
        print(f"    |H - 6|     = {dH:.2f}  {mark(dH, 0.5, 1.5)}", flush=True)
        print(f"    |hub-12.4%| = {dhub:.2f}  {mark(dhub, 2, 5)}", flush=True)
        print(f"    |I - 20%|   = {dI:.2f}  {mark(dI, 3, 8)}", flush=True)

    out = {
        "config": {
            "n_universes": n_universes, "n_epochs": n_epochs,
            "sigma": sigma, "seed": seed, "warmup": warmup,
        },
        "elapsed_s": round(elapsed, 1),
        "baseline_genome": make_baseline_genome(),
        "final_best_genome": population[final_best_idx],
        "final_best_metrics": final_m,
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=6)
    ap.add_argument("--n_epochs",    type=int, default=20)
    ap.add_argument("--sigma",       type=float, default=0.4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--warmup",      type=int, default=60)
    ap.add_argument("--output",      type=str,
                     default="phase_13_iss_cardinal.json")
    args = ap.parse_args()
    run_iss_cardinal(
        n_universes=args.n_universes,
        n_epochs=args.n_epochs,
        sigma=args.sigma,
        seed=args.seed,
        warmup=args.warmup,
        output=args.output,
    )


if __name__ == "__main__":
    main()
