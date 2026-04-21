"""Phase 12 — Cardinal v2 with Tournament Selection.

Root cause of Priority 4 null: original Cardinal replaces ONLY worst universe
with best's variant. Winner's params (σ, τ) never mutate. Level 2 effect
restricted to losers → overall quality unchanged.

Fix: Tournament selection.
  - All universes probabilistically replaced (softmax-weighted by quality)
  - Strong universes MORE LIKELY preserved, but not guaranteed
  - Weak universes MORE LIKELY replaced, but not guaranteed
  - Winner's σ, τ CAN evolve over time

Test (re-run Priority 4 with new mechanism):
  A. Cardinal v1 (original, winner never mutates), Level 2 τ = 0.25 fixed
  B. Cardinal v2 (tournament), Level 2 τ evolves

Hypothesis:
  With tournament, winner's σ can drift toward optimum via selection pressure.
  Level 2 τ evolution might now show measurable effect.

If A ≈ B: Level 2 truly has no benefit (strong refutation of Zenron^∞)
If B > A: Our Priority 4 null was an implementation artifact, not theoretical
"""
from __future__ import annotations

import argparse
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

from tamashii.phase_10_3_cardinal import (
    Universe, make_universe_params, PARAM_META_RANGES, DEFAULT_WORLD_PARAMS,
)
from tamashii.phase_9_ecology import SHELLS_DEFAULT, make_child_factory


def mutate_universe_v2(parent: dict, sigma: float, seed: int) -> dict:
    """Mutate universe params with given sigma (Level 1 or Level 2 configurable)."""
    rng = np.random.default_rng(seed)
    child = dict(parent)
    for key, (lo, hi, typ) in PARAM_META_RANGES.items():
        if rng.random() < 0.5:
            continue
        delta = rng.normal(0, sigma * (hi - lo))
        val = parent[key] + delta
        val = np.clip(val, lo, hi)
        if typ == "int":
            val = int(round(val))
        child[key] = val
    return child


def tournament_replace(universes: list[Universe], temperature: float,
                        seed: int, mutate_fn: Callable,
                        n_replace: int) -> list[Universe]:
    """Tournament-based meta-evolution. Returns new universe list.

    All universes sorted by quality. Softmax probabilities decide survival.
    Strong more likely preserved, but not guaranteed. Winner CAN be replaced.
    """
    rng = np.random.default_rng(seed)
    qualities = np.array([u.quality() for u in universes])
    N = len(universes)

    # Softmax selection probabilities for survival
    # Higher temperature = more random; lower = more deterministic
    q_norm = (qualities - qualities.mean()) / (qualities.std() + 1e-6)
    survival_probs = np.exp(q_norm / temperature)
    survival_probs /= survival_probs.sum()

    # Sample N-n_replace survivors (with replacement = stronger ones can clone)
    survivor_idx = rng.choice(N, size=N - n_replace, p=survival_probs, replace=True)

    # Survivors keep their universe state
    survivors = [universes[i] for i in survivor_idx]

    # Pick parents for new universes (proportional to quality)
    parent_idx = rng.choice(N, size=n_replace, p=survival_probs)
    return survivors, parent_idx


def run_cardinal_tournament(n_universes: int, n_epochs: int, epoch_steps: int,
                             agents_per_universe: int, tournament_temp: float,
                             meta_sigma: float, seed: int, use_level2: bool,
                             label: str):
    """Run Cardinal with tournament selection.

    Args:
      tournament_temp: softmax temperature (1.0=moderate, 0.3=sharp, 3.0=flat)
      meta_sigma: Level 1 fixed sigma if not using level2
      use_level2: if True, each universe has own meta_mutation_sigma evolving
    """
    shells = SHELLS_DEFAULT
    configs_dir = os.path.join(THIS_DIR, "configs")
    trained_dir = "tamashii/configs"

    print(f"\n{'='*72}", flush=True)
    print(f"  {label} (tournament_temp={tournament_temp}, use_level2={use_level2})",
          flush=True)
    print(f"{'='*72}", flush=True)

    # Initialize universes
    universes = []
    for u_id in range(n_universes):
        wp = make_universe_params(seed + u_id * 17)
        if use_level2:
            # Each universe has own meta_mutation_sigma
            wp_meta_sigma = float(np.random.default_rng(seed + u_id * 23).uniform(0.05, 0.35))
        else:
            wp_meta_sigma = meta_sigma
        # Keep meta_sigma inside universe as attribute, not in world_params
        u = Universe(u_id, dict(wp), agents_per_universe,
                      seed + u_id * 101,
                      shells, configs_dir, trained_dir)
        u.meta_sigma = wp_meta_sigma
        universes.append(u)

    history = []
    t_total = time.time()

    for epoch in range(n_epochs):
        t_ep = time.time()
        # Run each universe
        for u in universes:
            u.run_epoch(epoch_steps, log_every=epoch_steps // 2)

        qualities = [(u.id, u.quality()) for u in universes]
        qualities.sort(key=lambda x: -x[1])
        print(f"\n  EPOCH {epoch}: qualities = {[(i, round(q,2)) for i,q in qualities]}",
              flush=True)
        history.append({
            "epoch": epoch,
            "universes": [{
                "id": u.id,
                "world_params": dict(u.world_params),
                "meta_sigma": float(u.meta_sigma),
                "quality": float(u.quality()),
                "stats": u.world.stats(),
            } for u in universes],
        })

        # Tournament replacement (except last epoch)
        if epoch < n_epochs - 1:
            n_replace = max(1, n_universes // 3)  # replace 1/3 each epoch
            qualities_arr = np.array([u.quality() for u in universes])
            q_norm = (qualities_arr - qualities_arr.mean()) / (qualities_arr.std() + 1e-6)
            probs = np.exp(q_norm / tournament_temp)
            probs /= probs.sum()

            # Decide which to replace (lower prob = higher replace chance)
            rng = np.random.default_rng(seed + epoch * 7)
            # Replace indices: sample WITHOUT replacement from low-quality side
            replace_probs = 1.0 - probs
            replace_probs /= replace_probs.sum()
            replace_idx = rng.choice(n_universes, size=n_replace,
                                      replace=False, p=replace_probs)
            # Parent indices: sample WITH replacement from high-quality side
            parent_idx = rng.choice(n_universes, size=n_replace, p=probs)

            for r_idx, p_idx in zip(replace_idx, parent_idx):
                parent = universes[p_idx]
                if use_level2:
                    # Level 2: use parent's OWN meta_sigma
                    effective_sigma = parent.meta_sigma
                    # Also evolve meta_sigma itself (fixed outer sigma)
                    new_meta_sigma = parent.meta_sigma + rng.normal(0, 0.1 * 0.3)
                    new_meta_sigma = float(np.clip(new_meta_sigma, 0.03, 0.4))
                else:
                    effective_sigma = meta_sigma
                    new_meta_sigma = meta_sigma

                new_params = mutate_universe_v2(
                    parent.world_params, effective_sigma,
                    seed + epoch * 31 + int(r_idx))

                # Create new universe
                new_u = Universe(
                    int(r_idx), new_params, agents_per_universe,
                    seed + int(r_idx) * 101 + epoch * 3,
                    shells, configs_dir, trained_dir)
                new_u.meta_sigma = new_meta_sigma
                universes[int(r_idx)] = new_u
                print(f"  TOURNAMENT: u{r_idx} ← variant of u{p_idx} "
                      f"(parent q={parent.quality():.2f}, "
                      f"effective_σ={effective_sigma:.3f})", flush=True)

        print(f"  Epoch elapsed: {time.time() - t_ep:.0f}s", flush=True)

    total = time.time() - t_total
    print(f"\n  {label} TOTAL: {total:.0f}s", flush=True)

    # Summary
    final_qualities = [u.quality() for u in universes]
    final_sigmas = [u.world_params.get("mutation_rate", 0) for u in universes]
    final_meta_sigmas = [u.meta_sigma for u in universes]

    summary = {
        "label": label,
        "total_elapsed_s": round(total, 1),
        "tournament_temp": tournament_temp,
        "use_level2": use_level2,
        "final_max_quality": float(max(final_qualities)),
        "final_mean_quality": float(np.mean(final_qualities)),
        "final_std_quality": float(np.std(final_qualities)),
        "final_mean_sigma": float(np.mean(final_sigmas)),
        "final_max_sigma": float(max(final_sigmas)),
        "final_mean_meta_sigma": float(np.mean(final_meta_sigmas)),
        "final_std_meta_sigma": float(np.std(final_meta_sigmas)),
        "history": history,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=6)
    ap.add_argument("--n_epochs", type=int, default=5)
    ap.add_argument("--epoch_steps", type=int, default=1500)
    ap.add_argument("--agents_per_universe", type=int, default=12)
    ap.add_argument("--tournament_temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_12_tournament.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("  PHASE 12: Cardinal v2 Tournament Selection — Level 2 re-test",
          flush=True)
    print("=" * 72, flush=True)

    # Condition A: Tournament v2 + Level 1 (fixed meta_sigma)
    result_A = run_cardinal_tournament(
        args.n_universes, args.n_epochs, args.epoch_steps,
        args.agents_per_universe, args.tournament_temp,
        meta_sigma=0.25, seed=args.seed,
        use_level2=False, label="A_v2_level1")

    # Condition B: Tournament v2 + Level 2 (meta_sigma evolves)
    result_B = run_cardinal_tournament(
        args.n_universes, args.n_epochs, args.epoch_steps,
        args.agents_per_universe, args.tournament_temp,
        meta_sigma=0.25, seed=args.seed,
        use_level2=True, label="B_v2_level2")

    # Compare
    print(f"\n{'='*72}", flush=True)
    print(f"  A (v2 Level 1) vs B (v2 Level 2)", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'metric':30s} {'A Level1':>12s} {'B Level2':>12s}", flush=True)
    for k in ["final_max_quality", "final_mean_quality", "final_std_quality",
              "final_mean_sigma", "final_max_sigma",
              "final_mean_meta_sigma", "final_std_meta_sigma"]:
        a_v = result_A[k]
        b_v = result_B[k]
        print(f"  {k:30s} {a_v:>12.4f} {b_v:>12.4f}", flush=True)

    delta = result_B["final_max_quality"] - result_A["final_max_quality"]
    print(f"\n  Quality improvement B vs A: {delta:+.3f}", flush=True)
    if delta > 0.3:
        print(f"  ✓ Level 2 WINS over Level 1 under tournament selection",
              flush=True)
    elif delta < -0.3:
        print(f"  ✗ Level 2 LOSES (worse than Level 1 fixed)", flush=True)
    else:
        print(f"  ≈ Level 2 ≈ Level 1 (within noise)", flush=True)

    out = {
        "A_v2_level1": result_A,
        "B_v2_level2": result_B,
        "improvement": delta,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
