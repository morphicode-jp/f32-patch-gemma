"""Phase 11.3 — Break through gen_max=1 wall.

Priority 3 of Zenron^∞ validation: does evolution go multi-generational
when food pressure forces real turnover?

Current default: max_pop=25, n_food=6 → everyone survives, gen=1 stalls.
This: max_pop=15, n_food=3 (tight), epoch 10000 step (longer) = deaths happen,
reproduction opportunities created.

Goal: observe gen_max 5+ with DNA accumulated change across generations.
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

from tamashii.phase_10_3_cardinal import Universe
from tamashii.phase_10_3_cardinal_gpu import run_cardinal_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=3)
    ap.add_argument("--agents_per_universe", type=int, default=12)
    ap.add_argument("--epoch_steps", type=int, default=8000)
    ap.add_argument("--n_epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_11_3_multigen.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("  PHASE 11.3 — Multi-generation breakthrough", flush=True)
    print("  Tight food pressure → death → turnover → G2+ reproduction", flush=True)
    print("=" * 72, flush=True)

    # Override default world params in Universe init (harsh conditions)
    # We do this via monkey-patch
    import tamashii.phase_10_3_cardinal as card
    original_defaults = dict(card.DEFAULT_WORLD_PARAMS)
    card.DEFAULT_WORLD_PARAMS.update({
        "n_food": 3,            # tight (was 6)
        "max_population": 15,   # low cap (was 25)
        "energy_decay_per_step": 0.20,  # harsher (was 0.15)
        "food_energy_gain": 22.0,       # lower (was 25)
    })

    # Also adjust the PARAM_META_RANGES lo-hi so random init doesn't undo
    card.PARAM_META_RANGES["n_food"] = (2, 5, "int")
    card.PARAM_META_RANGES["energy_decay_per_step"] = (0.15, 0.28, "float")
    card.PARAM_META_RANGES["food_energy_gain"] = (15.0, 28.0, "float")

    t0 = time.time()
    run_cardinal_gpu(
        n_universes=args.n_universes,
        agents_per_universe=args.agents_per_universe,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
        output=args.output,
        device="cuda",
    )
    elapsed = time.time() - t0

    # Restore defaults
    card.DEFAULT_WORLD_PARAMS = original_defaults

    # Analyze: did gen_max > 1 occur?
    with open(args.output, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n{'='*72}", flush=True)
    print(f"  MULTI-GEN RESULTS ({elapsed:.0f}s)", flush=True)
    print(f"{'='*72}", flush=True)
    max_gens_per_universe = []
    for u in data["final_ranking"]:
        s = u["stats"]
        max_gens_per_universe.append(s["max_generation"])
        print(f"  u{u['id']}: alive={s['n_alive']} births={s['n_births']} "
              f"deaths={s['n_deaths']} gen_max={s['max_generation']} "
              f"DNA={s['dna_diversity']:.4f}", flush=True)

    best_gen = max(max_gens_per_universe)
    print(f"\n  Best gen_max achieved: {best_gen}", flush=True)
    if best_gen >= 3:
        print(f"  ✓ PASS: G{best_gen} reached — multi-generation evolution active",
              flush=True)
    else:
        print(f"  × Still stalled at G{best_gen} — may need even harsher conditions",
              flush=True)


if __name__ == "__main__":
    main()
