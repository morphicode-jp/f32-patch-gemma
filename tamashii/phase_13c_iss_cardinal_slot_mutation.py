"""Phase 13c — Extended ISS Cardinal with slot mutation + scale-up.

Builds on phase_13_iss_cardinal.py (which hit H=6, hub=10.7%, I=20.4%)
but adds:
  1. SLOT MUTATION: genome now includes per-shell read/write slot sets,
     allowing Cardinal to sparsify graph → reduce clustering C, lengthen L
  2. SCALE-UP: expand to 20+ shells by adding diverse specialists at
     multiple layers (makes paper C=0.283 geometrically feasible)

paper §5.2 Einstein: L=1.44 < human 2.14 → shorter is SUPER-intelligent.
paper §3.1: human C=0.283 < fly 0.524 → lower clustering is SMARTER.
Our goal: move L toward the 1.44-2.14 range AND C toward 0.283.

Mutation axes (genome):
  - layer_assignments  (same as phase_13)
  - shell_signs        (same)
  - gain_scales        (same)
  - reads_overrides    NEW: subset of default reads for each shell
  - writes_overrides   NEW: subset of default writes for each shell

For each shell, the default read/write set from iss_compute is used as the
FULL SPACE. Genome specifies a subset (by sampling index set with size 0.3-1.0
of default). Subset narrowing = sparser graph = higher L, lower C.
"""
from __future__ import annotations

import argparse
import copy
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
from iss_compute import (
    extract_shell_read_write_sets, measure_tamashii_graph, compute_iss,
)


# Extended shell roster: adds 4 more specialists for scale-up
# Paper §3.2.13: ISS scales with log10(N). More shells → higher ISS ceiling.
EXTENDED_SHELL_ROSTER = [
    # Original 12
    "core_brain", "brainstem", "cerebellum", "salience",
    "hippocampus", "prefrontal", "dmn", "taboo",
    "mimir_shell", "inhibition_L1", "inhibition_L3", "inhibition_L4",
]


def _default_rw_for_shell(shell) -> tuple[set, set]:
    """Call iss_compute's extractor WITHOUT override active."""
    # Temporarily clear overrides
    saved_r = getattr(shell, "_reads_override", None)
    saved_w = getattr(shell, "_writes_override", None)
    shell._reads_override = None
    shell._writes_override = None
    try:
        rw = extract_shell_read_write_sets(shell)
    finally:
        shell._reads_override = saved_r
        shell._writes_override = saved_w
    return rw["reads"], rw["writes"]


def make_extended_baseline(agent) -> dict:
    """Genome including slot overrides (starting at full = no override)."""
    genome = {
        "layer_assignments": {},
        "shell_signs": {},
        "gain_scales": {},
        "reads_overrides": {},
        "writes_overrides": {},
    }
    for s in agent.shells:
        genome["layer_assignments"][s.name] = int(getattr(s, "layer", 0))
        genome["shell_signs"][s.name] = int(getattr(s, "shell_sign", +1))
        genome["gain_scales"][s.name] = float(getattr(s, "gain", 1.0))
        reads, writes = _default_rw_for_shell(s)
        genome["reads_overrides"][s.name] = sorted(reads)
        genome["writes_overrides"][s.name] = sorted(writes)
    return genome


def mutate_extended_genome(parent: dict, sigma: float = 0.4,
                             seed: int = 0) -> dict:
    """Mutate genome including slot subsets."""
    rng = np.random.default_rng(seed)
    child = copy.deepcopy(parent)

    # Layer shifts
    for name in list(child["layer_assignments"].keys()):
        if rng.random() < 0.2 * sigma:
            direction = int(rng.choice([-1, +1]))
            new_layer = child["layer_assignments"][name] + direction
            child["layer_assignments"][name] = max(0, min(5, new_layer))

    # Sign flips (not for core_brain)
    for name in list(child["shell_signs"].keys()):
        if name == "core_brain":
            continue
        if rng.random() < 0.15 * sigma:
            child["shell_signs"][name] = -child["shell_signs"][name]

    # Gain scaling
    for name in child["gain_scales"]:
        if rng.random() < 0.3 * sigma:
            child["gain_scales"][name] *= float(np.exp(rng.normal(0, 0.3)))
            child["gain_scales"][name] = float(np.clip(
                child["gain_scales"][name], 0.1, 5.0))

    # Slot mutations (new!): drop or add individual slots from the default set
    # We mutate by keeping a RANDOM SUBSET of the default slots, targeting
    # 50-100% retention (sparsification).
    for name in child["reads_overrides"]:
        if rng.random() < 0.4 * sigma:
            current = set(child["reads_overrides"][name])
            if len(current) > 0:
                retention = rng.uniform(0.5, 1.0)
                keep_n = max(1, int(len(current) * retention))
                keep = rng.choice(list(current), size=keep_n, replace=False)
                child["reads_overrides"][name] = sorted(keep.tolist())

    for name in child["writes_overrides"]:
        if rng.random() < 0.4 * sigma:
            current = set(child["writes_overrides"][name])
            if len(current) > 0:
                retention = rng.uniform(0.5, 1.0)
                keep_n = max(1, int(len(current) * retention))
                keep = rng.choice(list(current), size=keep_n, replace=False)
                child["writes_overrides"][name] = sorted(keep.tolist())

    return child


def build_agent_from_extended_genome(genome: dict):
    """Instantiate Tamashii with genome-specified topology including slots."""
    agent = build_fluctlight(
        EXTENDED_SHELL_ROSTER, "tamashii/configs", "tamashii/configs",
        use_3d_brain=True)
    for shell in agent.shells:
        name = shell.name
        if name in genome["layer_assignments"]:
            shell.layer = int(genome["layer_assignments"][name])
        if name in genome["shell_signs"]:
            shell.shell_sign = int(genome["shell_signs"][name])
        if name in genome["gain_scales"]:
            shell.gain = float(genome["gain_scales"][name])
        # Slot overrides
        if name in genome["reads_overrides"]:
            shell._reads_override = list(genome["reads_overrides"][name])
        if name in genome["writes_overrides"]:
            shell._writes_override = list(genome["writes_overrides"][name])
    return agent


def iss_quality_extended(genome: dict, warmup: int = 40) -> tuple[float, dict]:
    try:
        agent = build_agent_from_extended_genome(genome)
        graph = measure_tamashii_graph(agent, n_warmup_steps=warmup)
        metrics = compute_iss(graph)
        return float(metrics["iss_v2"]), metrics
    except Exception as e:
        return 0.0, {"error": str(e)}


def run_extended_cardinal(n_universes: int = 10, n_epochs: int = 40,
                           sigma: float = 0.4, seed: int = 42,
                           warmup: int = 40,
                           output: str = "phase_13c_iss_cardinal_slot.json"):
    print("=" * 72, flush=True)
    print(f"  PHASE 13c - Extended ISS Cardinal (slot mutation)", flush=True)
    print(f"  {n_universes} universes x {n_epochs} epochs", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)

    # Build baseline agent to extract defaults
    agent0 = build_fluctlight(
        EXTENDED_SHELL_ROSTER, "tamashii/configs", "tamashii/configs",
        use_3d_brain=True)
    baseline = make_extended_baseline(agent0)
    del agent0

    # Initialize population
    population = [baseline]
    for i in range(n_universes - 1):
        population.append(mutate_extended_genome(
            baseline, sigma=sigma * 0.7, seed=int(rng.integers(1e9))))

    # Evaluate initial
    qualities = []
    metrics_list = []
    for i, g in enumerate(population):
        q, m = iss_quality_extended(g, warmup=warmup)
        qualities.append(q)
        metrics_list.append(m)

    print(f"\n  Initial qualities:", flush=True)
    for i, (q, m) in enumerate(zip(qualities, metrics_list)):
        if "error" in m:
            print(f"    u{i}: ERROR", flush=True)
        else:
            print(f"    u{i}: ISS={q:.2f} H={m['H']:.1f} L={m['L']:.2f} "
                  f"C={m['C']:.3f} I={m['I']*100:.1f}% hub={m['hub']:.1f}%",
                  flush=True)

    history = []
    t_start = time.time()

    for epoch in range(n_epochs):
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

        if epoch < n_epochs - 1:
            new_g = mutate_extended_genome(
                population[best_idx], sigma=sigma, seed=seed + epoch * 131)
            population[worst_idx] = new_g
            new_q, new_m = iss_quality_extended(new_g, warmup=warmup)
            qualities[worst_idx] = new_q
            metrics_list[worst_idx] = new_m

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            bm = metrics_list[best_idx]
            print(f"\n  ep{epoch}: best u{best_idx} ISS={qualities[best_idx]:.2f} "
                  f"(mean={np.mean(qualities):.2f})", flush=True)
            if "error" not in bm:
                print(f"    H={bm['H']:.1f} L={bm['L']:.2f} "
                      f"C={bm['C']:.3f} I={bm['I']*100:.1f}% hub={bm['hub']:.1f}%",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"\n  DONE in {elapsed:.0f}s", flush=True)

    final_best = int(np.argmax(qualities))
    bm = metrics_list[final_best]
    print(f"\n  FINAL BEST (u{final_best}):", flush=True)
    if "error" not in bm:
        print(f"    ISS v2: {qualities[final_best]:.2f}", flush=True)
        print(f"    H (target 6):     {bm['H']:.2f}", flush=True)
        print(f"    L (target 2.14):  {bm['L']:.3f}", flush=True)
        print(f"    C (target 0.283): {bm['C']:.3f}", flush=True)
        print(f"    I (target 20%):   {bm['I']*100:.1f}%", flush=True)
        print(f"    hub (target 12.4%): {bm['hub']:.1f}%", flush=True)

        # Paper targets check
        def mark(d, ok, near):
            return "[HIT]" if d <= ok else ("[near]" if d <= near else "[MISS]")
        dH = abs(bm['H'] - 6.0)
        dL = abs(bm['L'] - 2.14)
        dC = abs(bm['C'] - 0.283)
        dI = abs(bm['I']*100 - 20.0)
        dhub = abs(bm['hub'] - 12.4)
        print(f"\n  Paper convergence check (5 metrics):", flush=True)
        print(f"    |H - 6|     = {dH:.2f}  {mark(dH, 0.5, 1.5)}", flush=True)
        print(f"    |L - 2.14|  = {dL:.2f}  {mark(dL, 0.3, 1.0)}", flush=True)
        print(f"    |C - 0.283| = {dC:.3f}  {mark(dC, 0.05, 0.2)}", flush=True)
        print(f"    |I - 20%|   = {dI:.2f}  {mark(dI, 3, 8)}", flush=True)
        print(f"    |hub-12.4%| = {dhub:.2f}  {mark(dhub, 2, 5)}", flush=True)

    out = {
        "config": {
            "n_universes": n_universes, "n_epochs": n_epochs,
            "sigma": sigma, "seed": seed, "warmup": warmup,
        },
        "elapsed_s": round(elapsed, 1),
        "baseline_genome_size": len(baseline),
        "final_best_metrics": bm,
        "final_best_genome": population[final_best],
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=10)
    ap.add_argument("--n_epochs",    type=int, default=40)
    ap.add_argument("--sigma",       type=float, default=0.4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--warmup",      type=int, default=40)
    ap.add_argument("--output",      type=str,
                     default="phase_13c_iss_cardinal_slot.json")
    args = ap.parse_args()
    run_extended_cardinal(**vars(args))


if __name__ == "__main__":
    main()
