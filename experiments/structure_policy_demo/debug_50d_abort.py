"""50d 純連続で structure_policy が 5 秒 abort した原因を特定."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from twelve.agent.mimir_odin_structure_policy import mimir_odin_structure_policy


DIM = 50
BOUNDS = [(-1.0, 1.5)] * DIM


def f_hard_50d(p):
    p = np.asarray(p)
    sharp = math.exp(-30 * np.linalg.norm(p - 0.9) ** 2)
    broad = 0.55 * math.exp(-1.0 * np.linalg.norm(p - 0.3) ** 2)
    return float(sharp + broad)


def identity_decode(p):
    return {"values": list(map(float, p))}


def identity_detail(p):
    return {"score": float(f_hard_50d(p)), "policy": identity_decode(p)}


def main():
    rng = np.random.default_rng(42)
    seeds = [list(map(float, rng.uniform(-1.0, 1.5, size=DIM))) for _ in range(8)]

    print("=" * 78)
    print("50d 純連続 + structure_policy: abort 原因調査")
    print("=" * 78)
    print(f"DIM={DIM}, 8 random seeds, identity decode")
    for i, s in enumerate(seeds):
        print(f"  seed[{i}]: score={f_hard_50d(s):+.4f}")

    r = mimir_odin_structure_policy(
        f_hard_50d, BOUNDS,
        policy_points=seeds,
        param_decoder=identity_decode,
        detail_fn=identity_detail,
        time_budget=60,
        executor="thread",
        verbose=True,
    )

    print(f"\n--- result keys: {list(r.keys())[:20]} ---")
    print(f"\naborted: {r.get('aborted')}")
    print(f"abort_reason: {r.get('abort_reason')}")
    print(f"\ncheck_eval: {json.dumps(r.get('check_eval', {}), indent=2, default=str, ensure_ascii=False)[:1500]}")
    print(f"\npolicy_response_probe: {json.dumps(r.get('policy_response_probe', {}), indent=2, default=str, ensure_ascii=False)[:1500]}")
    print(f"\nstructure_policy_mode: {r.get('structure_policy_mode')}")


if __name__ == "__main__":
    main()
