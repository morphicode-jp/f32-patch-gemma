"""Phase 13f — Adaptive ISS Cardinal with 50 shells.

Breaks through 30-shell trade-off (L HIT but I/hub drifted) by:
  1. Scaling to 50 shells (paper §3.2.13 scaling law: more N = higher ISS ceiling)
  2. ADAPTIVE per-metric mutation bias: focus mutations on the metric
     furthest from paper target, while preserving already-HIT metrics.

Adaptive mutation policy:
  - Compute per-metric normalized gap: d_k = |m_k - target_k| / tolerance_k
  - Metric with largest gap = "dominant direction"
  - Increase mutation rate on dimensions that affect that metric
  - Decrease mutation rate on dimensions that would worsen HIT metrics
  - Effectively: aim mutations at the specific bottleneck

This prevents random walk wandering; Cardinal becomes metric-aware.

Target: ISS v2 > 90 (within Einstein-class 107).
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
from iss_compute import measure_tamashii_graph, compute_iss
from tamashii.phase_13c_iss_cardinal_slot_mutation import (
    build_agent_from_extended_genome, make_extended_baseline,
)


# 50-shell roster: 30 from 13e + 20 new specialists
ROSTER_50 = [
    # Core 12
    "core_brain", "brainstem", "cerebellum", "salience",
    "hippocampus", "prefrontal", "dmn", "taboo",
    "mimir_shell", "inhibition_L1", "inhibition_L3", "inhibition_L4",
    # 13d (+8)
    "inhibition_L0_a", "inhibition_L0_b",
    "inhibition_L2_a", "inhibition_L2_b",
    "inhibition_L3_b",
    "inhibition_L4_a", "inhibition_L4_b",
    "inhibition_L5",
    # 13e (+10)
    "inhibition_L1_b", "inhibition_L1_c",
    "inhibition_L2_c", "inhibition_L2_d",
    "inhibition_L3_c", "inhibition_L3_d",
    "inhibition_L4_c", "inhibition_L4_d",
    "inhibition_L5_b", "inhibition_L5_c",
    # 13f (+20 specialists)
    *[f"inhibition_L{L}_spec{i}" for L in [0, 1, 3, 4, 5] for i in range(4)],
]


# Paper targets with per-metric tolerance (for gap calculation)
PAPER_TARGETS = {
    "H":   {"target": 6.0,   "tol": 1.0},     # ±1 acceptable
    "L":   {"target": 2.14,  "tol": 0.5},
    "C":   {"target": 0.283, "tol": 0.10},
    "I":   {"target": 0.20,  "tol": 0.05},
    "hub": {"target": 12.4,  "tol": 5.0},
}


def metric_gap(metrics: dict) -> dict:
    """Compute normalized gap for each paper metric."""
    gaps = {}
    for k, info in PAPER_TARGETS.items():
        v = metrics.get(k, 0)
        if k == "I":
            v = v  # already fractional
        elif k == "hub":
            v = v  # percent
        gap = abs(v - info["target"])
        gaps[k] = gap / info["tol"]  # normalized
    return gaps


def dominant_gap_metric(metrics: dict) -> str:
    """Which metric has the LARGEST normalized gap? Focus mutation there."""
    gaps = metric_gap(metrics)
    return max(gaps.items(), key=lambda x: x[1])[0]


def mutate_adaptive(parent: dict, target_metric: str,
                     sigma: float = 0.4, seed: int = 0) -> dict:
    """Adaptive mutation biased toward fixing `target_metric`.

    Mutation dimensions → metric impact map:
      layer_assignments: mostly H
      shell_signs:       mostly I (and indirectly hub via signed edges)
      gain_scales:       mostly L (indirect via activity weighting)
      reads_overrides:   mostly C, L (graph sparsity)
      writes_overrides:  mostly C, L (graph sparsity)
    """
    rng = np.random.default_rng(seed)
    child = copy.deepcopy(parent)

    # Bias per dimension based on target_metric
    bias = {
        "H":   {"layer": 2.0, "sign": 0.5, "gain": 0.5, "reads": 0.3, "writes": 0.3},
        "L":   {"layer": 0.5, "sign": 0.5, "gain": 1.2, "reads": 2.0, "writes": 2.0},
        "C":   {"layer": 0.3, "sign": 0.5, "gain": 0.3, "reads": 2.5, "writes": 2.5},
        "I":   {"layer": 0.3, "sign": 3.0, "gain": 0.5, "reads": 0.3, "writes": 0.3},
        "hub": {"layer": 0.5, "sign": 1.0, "gain": 2.0, "reads": 1.5, "writes": 1.5},
    }.get(target_metric, {"layer": 1, "sign": 1, "gain": 1, "reads": 1, "writes": 1})

    # Layer shifts
    for name in list(child["layer_assignments"].keys()):
        if rng.random() < 0.2 * sigma * bias["layer"]:
            direction = int(rng.choice([-1, +1]))
            child["layer_assignments"][name] = max(0, min(5,
                child["layer_assignments"][name] + direction))

    # Sign flips (except core_brain)
    for name in list(child["shell_signs"].keys()):
        if name == "core_brain":
            continue
        if rng.random() < 0.15 * sigma * bias["sign"]:
            child["shell_signs"][name] = -child["shell_signs"][name]

    # Gain
    for name in child["gain_scales"]:
        if rng.random() < 0.3 * sigma * bias["gain"]:
            child["gain_scales"][name] *= float(np.exp(rng.normal(0, 0.3)))
            child["gain_scales"][name] = float(np.clip(
                child["gain_scales"][name], 0.1, 5.0))

    # Slot subsetting (key for C reduction)
    for name in child["reads_overrides"]:
        if rng.random() < 0.4 * sigma * bias["reads"]:
            current = set(child["reads_overrides"][name])
            if len(current) > 0:
                # Biased toward sparser subsets if targeting C
                if target_metric == "C":
                    retention = rng.uniform(0.3, 0.8)  # more aggressive cuts
                else:
                    retention = rng.uniform(0.5, 1.0)
                keep_n = max(1, int(len(current) * retention))
                keep = rng.choice(list(current), size=keep_n, replace=False)
                child["reads_overrides"][name] = sorted(keep.tolist())

    for name in child["writes_overrides"]:
        if rng.random() < 0.4 * sigma * bias["writes"]:
            current = set(child["writes_overrides"][name])
            if len(current) > 0:
                if target_metric == "C":
                    retention = rng.uniform(0.3, 0.8)
                else:
                    retention = rng.uniform(0.5, 1.0)
                keep_n = max(1, int(len(current) * retention))
                keep = rng.choice(list(current), size=keep_n, replace=False)
                child["writes_overrides"][name] = sorted(keep.tolist())

    return child


def evaluate_genome(genome: dict, roster, warmup: int = 40):
    try:
        # Build with 50-shell roster (re-scoped each call)
        import tamashii.phase_13c_iss_cardinal_slot_mutation as p13c
        p13c.EXTENDED_SHELL_ROSTER = roster
        agent = build_agent_from_extended_genome(genome)
        graph = measure_tamashii_graph(agent, n_warmup_steps=warmup)
        metrics = compute_iss(graph)
        return float(metrics["iss_v2"]), metrics
    except Exception as e:
        return 0.0, {"error": str(e)}


def run_adaptive_cardinal(n_universes: int = 12, n_epochs: int = 80,
                            sigma: float = 0.4, seed: int = 42,
                            warmup: int = 40,
                            output: str = "phase_13f_adaptive_50shell.json"):
    print("=" * 72, flush=True)
    print(f"  PHASE 13f - Adaptive ISS Cardinal (50 shells, target-biased)",
          flush=True)
    print(f"  {n_universes} universes x {n_epochs} epochs", flush=True)
    print(f"  Roster: {len(ROSTER_50)} shells", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)

    # Baseline from 50-shell roster
    import tamashii.phase_13c_iss_cardinal_slot_mutation as p13c
    p13c.EXTENDED_SHELL_ROSTER = ROSTER_50
    agent0 = build_fluctlight(
        ROSTER_50, "tamashii/configs", "tamashii/configs",
        use_3d_brain=True)
    baseline = make_extended_baseline(agent0)
    del agent0

    # Initial population: 1 baseline + mutated variants
    population = [baseline]
    for _ in range(n_universes - 1):
        # random initial target metric
        init_target = str(rng.choice(list(PAPER_TARGETS.keys())))
        population.append(mutate_adaptive(
            baseline, init_target, sigma=sigma*0.7,
            seed=int(rng.integers(1e9))))

    # Evaluate initial
    qualities = []
    metrics_list = []
    for g in population:
        q, m = evaluate_genome(g, ROSTER_50, warmup=warmup)
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
        best_m = metrics_list[best_idx]
        # Adaptive target: what's furthest from paper on the BEST universe?
        if "error" not in best_m:
            target = dominant_gap_metric(best_m)
        else:
            target = "H"

        history.append({
            "epoch": epoch,
            "qualities": list(qualities),
            "best_idx": best_idx,
            "best_iss_v2": float(qualities[best_idx]),
            "best_metrics": best_m,
            "mean_iss_v2": float(np.mean(qualities)),
            "mutation_target": target,
        })

        if epoch < n_epochs - 1:
            new_g = mutate_adaptive(
                population[best_idx], target, sigma=sigma,
                seed=seed + epoch * 131)
            population[worst_idx] = new_g
            new_q, new_m = evaluate_genome(new_g, ROSTER_50, warmup=warmup)
            qualities[worst_idx] = new_q
            metrics_list[worst_idx] = new_m

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            bm = best_m
            print(f"\n  ep{epoch}: best u{best_idx} ISS={qualities[best_idx]:.2f} "
                  f"target={target}", flush=True)
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
        for k, info in PAPER_TARGETS.items():
            v = bm[k] if k != "I" else bm["I"]
            d = abs(v - info["target"])
            ok = d <= info["tol"] * 0.5
            near = d <= info["tol"]
            status = "[HIT]" if ok else ("[near]" if near else "[MISS]")
            v_disp = v * 100 if k == "I" else v
            t_disp = info["target"] * 100 if k == "I" else info["target"]
            print(f"    {k:5s}={v_disp:6.3f} target={t_disp} diff={d:6.3f} {status}",
                  flush=True)

    out = {
        "config": {"n_universes": n_universes, "n_epochs": n_epochs,
                    "sigma": sigma, "seed": seed, "warmup": warmup,
                    "roster_size": len(ROSTER_50)},
        "elapsed_s": round(elapsed, 1),
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
    ap.add_argument("--n_universes", type=int, default=12)
    ap.add_argument("--n_epochs",    type=int, default=80)
    ap.add_argument("--sigma",       type=float, default=0.4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--warmup",      type=int, default=40)
    ap.add_argument("--output",      type=str,
                     default="phase_13f_adaptive_50shell.json")
    args = ap.parse_args()
    run_adaptive_cardinal(**vars(args))


if __name__ == "__main__":
    main()
