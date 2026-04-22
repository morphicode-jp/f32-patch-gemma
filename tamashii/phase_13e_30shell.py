"""Phase 13e — 30-shell scaled ISS Cardinal.

Target: break C=0.283 (paper human) by growing to 30 shells.
Paper §3.2.13: ISS scales log-linearly with N. Going 20 → 30 shells
predicts +1.5 ISS from N alone, but sparsity gains compound.
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


p13c.EXTENDED_SHELL_ROSTER = [
    # Core 12
    "core_brain", "brainstem", "cerebellum", "salience",
    "hippocampus", "prefrontal", "dmn", "taboo",
    "mimir_shell", "inhibition_L1", "inhibition_L3", "inhibition_L4",
    # 13d scale-up (+8)
    "inhibition_L0_a", "inhibition_L0_b",
    "inhibition_L2_a", "inhibition_L2_b",
    "inhibition_L3_b",
    "inhibition_L4_a", "inhibition_L4_b",
    "inhibition_L5",
    # 13e scale-up (+10 more)
    "inhibition_L1_b", "inhibition_L1_c",
    "inhibition_L2_c", "inhibition_L2_d",
    "inhibition_L3_c", "inhibition_L3_d",
    "inhibition_L4_c", "inhibition_L4_d",
    "inhibition_L5_b", "inhibition_L5_c",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=12)
    ap.add_argument("--n_epochs",    type=int, default=60)
    ap.add_argument("--sigma",       type=float, default=0.4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--warmup",      type=int, default=40)
    ap.add_argument("--output",      type=str,
                     default="phase_13e_30shell.json")
    args = ap.parse_args()
    print(f"\n  30-shell scaled roster: {len(p13c.EXTENDED_SHELL_ROSTER)} shells",
          flush=True)
    run_extended_cardinal(**vars(args))


if __name__ == "__main__":
    main()
