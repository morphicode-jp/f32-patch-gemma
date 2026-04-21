"""Phase 11.4 — Level 2 Recursion test (THE key missing experiment).

Question: does Zenron^∞'s INFINITE recursion claim actually hold?
Specifically: if we let meta-mutation strength τ ALSO evolve (instead of
fixed), does Cardinal find better σ faster?

  Level 0 (agent): DNA evolves via σ=universe.mutation_rate
  Level 1 (universe): σ itself evolves via meta-mutation τ
  Level 2 (THIS): τ also evolves via outer-level sigma (fixed here)

Cardinal v1 used τ=0.25 hardcoded. This experiment:
  A. Level 1 only: τ fixed = 0.25 (baseline, standard Cardinal)
  B. Level 2: τ evolves (each universe has own τ, meta-mutates in 2-level selection)

Hypothesis if Zenron^∞ holds:
  - B converges faster or to better σ
  - B's final τ is non-random (shows selection acts on τ)
  - Recursion deepens optimization

If B is no better than A → "infinite" is just poetry; recursion diminishing returns.
If B clearly wins → Zenron^∞ real, more levels should help further.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import copy

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import tamashii.phase_10_3_cardinal as card
from tamashii.phase_10_3_cardinal import Universe
from tamashii.phase_10_3_cardinal_gpu import run_cardinal_gpu


# Extend meta-ranges with τ if not present. IMPORTANT: don't put τ into
# DEFAULT_WORLD_PARAMS (VoxelWorld3D doesn't know about it). Keep as a
# "virtual" meta-parameter.
if "meta_mutation_sigma" not in card.PARAM_META_RANGES:
    card.PARAM_META_RANGES["meta_mutation_sigma"] = (0.05, 0.40, "float")

# Patch Universe.initialize_agents to STRIP meta_mutation_sigma before
# passing to VoxelWorld3D.
_original_init_agents = Universe.initialize_agents
def _patched_init_agents(self):
    # Strip meta-params before world construction
    meta_keys = {"meta_mutation_sigma"}
    saved = {}
    for k in list(self.world_params.keys()):
        if k in meta_keys:
            saved[k] = self.world_params.pop(k)
    _original_init_agents(self)
    # Restore meta-params onto the Universe so mutate_fn can read
    for k, v in saved.items():
        self.world_params[k] = v
Universe.initialize_agents = _patched_init_agents

# Save original mutate function
_original_mutate = card.mutate_universe_params


def mutate_universe_params_level2(parent: dict, sigma: float = 0.2,
                                    seed: int = 0) -> dict:
    """Level 2 version: use parent's OWN meta_mutation_sigma as sigma for
    all non-τ params. τ itself mutates with a higher-level fixed outer σ.
    """
    rng = np.random.default_rng(seed)
    child = dict(parent)
    # τ = parent's own meta_mutation_sigma (Level 2 parameter)
    parent_tau = float(parent.get("meta_mutation_sigma", 0.25))
    # Outer σ for τ itself (Level 3 placeholder, fixed small)
    outer_sigma_for_tau = 0.12

    for key, (lo, hi, typ) in card.PARAM_META_RANGES.items():
        if rng.random() < 0.5:
            continue  # mutate ~50% of meta-params
        if key == "meta_mutation_sigma":
            # τ mutates with its own outer sigma
            delta = rng.normal(0, outer_sigma_for_tau * (hi - lo))
        else:
            # Other params mutate with parent's τ (Level 2 effect)
            delta = rng.normal(0, parent_tau * (hi - lo))
        val = parent[key] + delta
        val = np.clip(val, lo, hi)
        if typ == "int":
            val = int(round(val))
        child[key] = val
    return child


def mutate_universe_params_level1(parent: dict, sigma: float = 0.25,
                                    seed: int = 0) -> dict:
    """Level 1 baseline: fixed σ=0.25 (original Cardinal behavior).
    Ignore parent.meta_mutation_sigma, use fixed outer sigma.
    """
    rng = np.random.default_rng(seed)
    child = dict(parent)
    for key, (lo, hi, typ) in card.PARAM_META_RANGES.items():
        if rng.random() < 0.5:
            continue
        if key == "meta_mutation_sigma":
            continue  # don't mutate τ in Level 1 (treated as fixed)
        delta = rng.normal(0, sigma * (hi - lo))
        val = parent[key] + delta
        val = np.clip(val, lo, hi)
        if typ == "int":
            val = int(round(val))
        child[key] = val
    return child


def run_condition(label: str, mutate_fn, n_universes: int, n_epochs: int,
                   epoch_steps: int, agents_per_universe: int, seed: int):
    """Run Cardinal with specified mutate function (Level 1 or Level 2)."""
    print(f"\n{'='*72}", flush=True)
    print(f"  CONDITION: {label}", flush=True)
    print(f"{'='*72}", flush=True)
    # Critical: monkey-patch in BOTH modules (cardinal_gpu imports by name)
    import tamashii.phase_10_3_cardinal_gpu as card_gpu
    card.mutate_universe_params = mutate_fn
    card_gpu.mutate_universe_params = mutate_fn  # <- this was missing
    out_path = f"tamashii_phase_11_4_{label}.json"
    run_cardinal_gpu(
        n_universes=n_universes,
        agents_per_universe=agents_per_universe,
        epoch_steps=epoch_steps,
        n_epochs=n_epochs,
        base_seed=seed,
        output=out_path,
        device="cuda",
    )
    # Restore
    card.mutate_universe_params = _original_mutate
    card_gpu.mutate_universe_params = _original_mutate
    # Load results
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def analyze(label: str, data: dict):
    """Extract σ and τ trajectory metrics."""
    print(f"\n  {label} analysis:", flush=True)
    # Across epochs, winner's σ and τ
    for h in data["history"]:
        best = h["universes"][0]
        wp = best["world_params"]
        sigma = wp.get("mutation_rate", 0)
        tau = wp.get("meta_mutation_sigma", 0.25)
        print(f"    ep{h['epoch']}: winner u{best['id']} "
              f"σ={sigma:.4f} τ={tau:.4f} q={best['quality']:.2f}", flush=True)
    # Final population stats
    final_sigmas = [u["world_params"].get("mutation_rate", 0) for u in data["final_ranking"]]
    final_taus = [u["world_params"].get("meta_mutation_sigma", 0.25) for u in data["final_ranking"]]
    final_qs = [u["quality"] for u in data["final_ranking"]]
    return {
        "label": label,
        "final_mean_sigma": float(np.mean(final_sigmas)),
        "final_max_sigma": float(np.max(final_sigmas)),
        "final_mean_tau": float(np.mean(final_taus)),
        "final_max_tau": float(np.max(final_taus)),
        "final_mean_quality": float(np.mean(final_qs)),
        "final_max_quality": float(np.max(final_qs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=6)
    ap.add_argument("--n_epochs", type=int, default=5)
    ap.add_argument("--epoch_steps", type=int, default=2000)
    ap.add_argument("--agents_per_universe", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_11_4_level2_summary.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("  PHASE 11.4 — Level 2 Recursion (THE key experiment)", flush=True)
    print(f"  {args.n_universes} universes × {args.agents_per_universe} agents × "
          f"{args.n_epochs} epochs × {args.epoch_steps} steps", flush=True)
    print("=" * 72, flush=True)

    # --- Condition A: Level 1 only (fixed τ=0.25) ---
    t0 = time.time()
    data_A = run_condition("A_level1", mutate_universe_params_level1,
                            args.n_universes, args.n_epochs,
                            args.epoch_steps, args.agents_per_universe, args.seed)
    analysis_A = analyze("A_level1", data_A)
    t_A = time.time() - t0

    # --- Condition B: Level 2 (τ evolves) ---
    t0 = time.time()
    data_B = run_condition("B_level2", mutate_universe_params_level2,
                            args.n_universes, args.n_epochs,
                            args.epoch_steps, args.agents_per_universe, args.seed)
    analysis_B = analyze("B_level2", data_B)
    t_B = time.time() - t0

    # --- Compare ---
    print(f"\n{'='*72}", flush=True)
    print(f"  LEVEL 1 vs LEVEL 2 COMPARISON", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'metric':30s} {'A (Level 1)':>15s} {'B (Level 2)':>15s}", flush=True)
    for key in ["final_mean_sigma", "final_max_sigma", "final_mean_tau",
                "final_max_tau", "final_mean_quality", "final_max_quality"]:
        a_val = analysis_A[key]
        b_val = analysis_B[key]
        print(f"  {key:30s} {a_val:>15.4f} {b_val:>15.4f}", flush=True)
    print(f"  {'elapsed_s':30s} {t_A:>15.1f} {t_B:>15.1f}", flush=True)

    # Key indicators
    print(f"\n  KEY CHECKS:", flush=True)
    tau_range_A = "N/A (fixed 0.25)"
    tau_mean_B = analysis_B['final_mean_tau']
    tau_spread_B = abs(tau_mean_B - 0.25)
    print(f"    Level 2 τ diverged from fixed 0.25?  "
          f"mean τ = {tau_mean_B:.4f}, delta = {tau_spread_B:+.4f}",
          flush=True)
    quality_boost = analysis_B['final_max_quality'] - analysis_A['final_max_quality']
    print(f"    Level 2 quality vs Level 1?           "
          f"Δ = {quality_boost:+.2f} ({'Level 2 WINS' if quality_boost > 0.5 else 'marginal'})",
          flush=True)
    sigma_reached_A = analysis_A['final_max_sigma']
    sigma_reached_B = analysis_B['final_max_sigma']
    print(f"    Higher σ reached?  "
          f"A={sigma_reached_A:.4f}, B={sigma_reached_B:.4f}, "
          f"{'B wins' if sigma_reached_B > sigma_reached_A else 'A wins'}",
          flush=True)

    out = {
        "A_level1": analysis_A,
        "B_level2": analysis_B,
        "comparison": {
            "tau_diverged": tau_spread_B > 0.03,
            "quality_boost": quality_boost,
            "level2_wins_on_sigma": sigma_reached_B > sigma_reached_A,
        },
        "elapsed_A_s": round(t_A, 1),
        "elapsed_B_s": round(t_B, 1),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
