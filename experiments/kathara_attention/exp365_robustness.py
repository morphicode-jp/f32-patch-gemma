"""第365期: 162 number robustness — 異なる random seed で 結果 stable か.

approach:
  異なる 5 seed で 162 family enumerate
  結果が 162 ± few で stable か確認
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import time
import sys


def main():
    print("=" * 80)
    print("第365期: 162 family size の robustness")
    print("=" * 80)
    sys.stdout.flush()

    target_deg = (2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4)
    seeds = [42, 123, 7, 2024, 31415, 271828]
    N_max = 15_000_000

    results = []
    for seed in seeds:
        print(f"\n  --- seed {seed} ---")
        rng = np.random.default_rng(seed)
        siblings = {}
        last_count = 0
        sat = 0
        t0 = time.time()

        for trial in range(N_max):
            if trial % 1_000_000 == 0 and trial > 0:
                cur = len(siblings)
                if cur == last_count:
                    sat += 1
                else:
                    sat = 0
                last_count = cur
                if sat >= 6:
                    break

            stubs = []
            for v, d in enumerate(target_deg):
                stubs.extend([v] * d)
            rng.shuffle(stubs)
            A = np.zeros((12, 12), dtype=np.int64)
            valid = True
            for i in range(0, len(stubs), 2):
                u, w = stubs[i], stubs[i+1]
                if u == w or A[u, w] == 1:
                    valid = False
                    break
                A[u, w] = A[w, u] = 1
            if not valid:
                continue
            A2 = A @ A
            if int(np.trace(A2 @ A)) != 0:
                continue
            A4 = A2 @ A2
            if int(np.trace(A4)) != 270:
                continue
            evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
            if abs((19 + abs(evs[0])) - 22) > 0.01:
                continue
            if (4 + int(np.trace(A2))) != 42:
                continue
            spec = tuple(round(e, 4) for e in evs)
            if spec not in siblings:
                siblings[spec] = True

        elapsed = time.time() - t0
        print(f"    seed {seed}: family size = {len(siblings)}, time {elapsed:.0f}s")
        results.append((seed, len(siblings)))
        sys.stdout.flush()

    # ============================================================
    # Stats
    # ============================================================
    sizes = [r[1] for r in results]
    print(f"\n{'='*80}")
    print(f"★ 全 seed の family size: {sizes}")
    print(f"  mean: {np.mean(sizes):.1f}")
    print(f"  median: {int(np.median(sizes))}")
    print(f"  min/max: {min(sizes)} / {max(sizes)}")
    print(f"  std: {np.std(sizes):.2f}")
    print(f"")
    if max(sizes) - min(sizes) < 5:
        print(f"  ★ tight cluster ({max(sizes)-min(sizes)} range)、 → 真の family size 確定的")
    else:
        print(f"  spread {max(sizes)-min(sizes)} — saturation 未完全")
    print(f"")
    print(f"  Most likely 真 family size: ~{int(np.median(sizes))}")
    print(f"  cf. 6 × 27 = 162 (= K¹ degree × E_6 short rep)")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "seeds": [s for s, _ in results],
        "family_sizes": sizes,
        "mean": float(np.mean(sizes)),
        "median": int(np.median(sizes)),
        "min": min(sizes), "max": max(sizes),
        "std": float(np.std(sizes)),
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round365_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
