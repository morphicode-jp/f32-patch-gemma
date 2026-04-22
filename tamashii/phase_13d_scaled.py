"""Phase 13d — Scaled ISS Cardinal with 20 shells.

Combines phase_13c's slot mutation with scale-up to 20 shells.
Goal: break through small-N mathematical limit on clustering C.

Paper §3.2.13: ISS = 8.94 × log10(N) + 3.20 (R² = 0.995).
Going from 12 → 20 shells predicts +1.97 ISS v1 from scale alone.
More importantly, higher N enables paper's C=0.283 target geometrically.

Roster: 12 existing + 8 inhibition specialists at layers 0, 2, 3, 4, 5.
"""
from __future__ import annotations

import argparse

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import tamashii.phase_13c_iss_cardinal_slot_mutation as p13c
from tamashii.phase_13c_iss_cardinal_slot_mutation import run_extended_cardinal


# Override the roster with 20-shell version
p13c.EXTENDED_SHELL_ROSTER = [
    # Original 12
    "core_brain", "brainstem", "cerebellum", "salience",
    "hippocampus", "prefrontal", "dmn", "taboo",
    "mimir_shell", "inhibition_L1", "inhibition_L3", "inhibition_L4",
    # Scale-up 8
    "inhibition_L0_a", "inhibition_L0_b",
    "inhibition_L2_a", "inhibition_L2_b",
    "inhibition_L3_b",
    "inhibition_L4_a", "inhibition_L4_b",
    "inhibition_L5",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=10)
    ap.add_argument("--n_epochs",    type=int, default=50)
    ap.add_argument("--sigma",       type=float, default=0.4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--warmup",      type=int, default=40)
    ap.add_argument("--output",      type=str,
                     default="phase_13d_scaled.json")
    args = ap.parse_args()
    print(f"\n  Using 20-shell scaled roster: {len(p13c.EXTENDED_SHELL_ROSTER)} shells",
          flush=True)
    run_extended_cardinal(**vars(args))


if __name__ == "__main__":
    main()
