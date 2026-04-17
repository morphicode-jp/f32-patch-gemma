"""cortical_sentinel_run.py — Run Sentinel on 5-layer cortical brain

Compares to 12N single-layer baseline (brain_sentinel biological).

Hypothesis:
  - Cortical peak eval will be LOWER than 12N (more params, harder optimization)
  - Cortical generalization gap will be SMALLER (proper layer roles = robustness)
  - Cortical guard stays higher (multi-layer health redundancy)

Usage:
    python cortical_sentinel_run.py --time-budget 1800
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from cortical_brain import (
    PARAM_RANGES,
    PARAM_NAMES,
    TOTAL_PARAMS,
    make_cortical_eval,
    make_cortical_structural_guard,
    run_cortical_flyworld,
    LAYER_ROLES,
    _INHIBIT_NODES,
)
from kathara_brain_sim_v8 import FlyWorldV3


def run(time_budget=1800, experience_id="cortical_biological", learn=True):
    print(f"\n{'=' * 70}")
    print(f"  Cortical Sentinel run")
    print(f"{'=' * 70}")
    print(f"  Architecture: 5 layers × 12N = 60 neurons, {TOTAL_PARAMS}D params")
    print(f"  Layer roles: {LAYER_ROLES}")
    print(f"  Dale's law: {_INHIBIT_NODES} inhibitory per layer (25%)")
    print(f"  FF:FB asymmetry: 1:0.47")
    print(f"  Budget: {time_budget}s")
    print(f"  experience_id: {experience_id}, learn={learn}\n")

    eval_fn = make_cortical_eval(FlyWorldV3, n_episodes=4, n_steps=60)
    guard_fn = make_cortical_structural_guard(n_episodes=2, n_steps=30)

    result = Sentinel(
        eval_fn=eval_fn,
        guard_fn=guard_fn,
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id=experience_id,
        learn=learn,
    ).run(time_budget=time_budget, verbose=True)

    # Save
    out = {
        "architecture": "5-layer cortical",
        "n_params": TOTAL_PARAMS,
        "verdict": result["verdict"],
        "eval_score": round(float(result["eval_score"]), 2),
        "best_ever_score": round(float(result.get("best_ever_score", result["eval_score"])), 2),
        "guard_score": round(float(result["guard_score"]), 2),
        "baseline_guard": round(float(result["baseline_guard"]), 2),
        "optimization_mode": result.get("optimization_mode", "owl"),
        "proxy_r2": round(float(result["proxy_r2"]), 3),
        "elapsed_s": round(float(result["elapsed_s"]), 1),
        "best_params": [round(float(p), 4) for p in result["best_params"]],
        "best_ever_params": (
            [round(float(p), 4) for p in result["best_ever_params"]]
            if result.get("best_ever_params") else None
        ),
    }
    with open("cortical_sentinel_result.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Saved: cortical_sentinel_result.json")
    print(f"  eval_score:      {result['eval_score']:.2f}")
    print(f"  best_ever_score: {result.get('best_ever_score', 'N/A')}")
    print(f"  guard_score:     {result['guard_score']:.2f}  (baseline {result['baseline_guard']:.2f})")

    return result, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-budget", type=int, default=1800)
    ap.add_argument("--experience-id", default="cortical_biological")
    args = ap.parse_args()
    run(time_budget=args.time_budget, experience_id=args.experience_id)
