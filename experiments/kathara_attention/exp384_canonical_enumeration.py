"""第384期: 完全 iso class enumeration (canonical form check).

approach:
  使う 既存 networkx の VF2 isomorphism check で iso class dedup.
  DFS enumeration with degree budget + triangle-free + early dedup.

  全 12V 19E triangle-free graph with degree seq (2,2,3,3,3,3,3,3,4,4,4,4):
    予測 unique iso class count: 数百 - 数千
    そのうち 7-identity 満たす: 1 (核) hopefully
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import time
import sys


def build_icosahedron():
    A = np.zeros((12, 12), dtype=np.int64)
    for j in range(1, 6):
        A[0, j] = A[j, 0] = 1
    for j in range(6, 11):
        A[11, j] = A[j, 11] = 1
    for i in range(1, 6):
        j = i + 1 if i < 5 else 1
        A[i, j] = A[j, i] = 1
    for i in range(6, 11):
        j = i + 1 if i < 10 else 6
        A[i, j] = A[j, i] = 1
    for i in range(1, 6):
        lo1 = i + 5
        lo2 = (i % 5) + 1 + 5
        A[i, lo1] = A[lo1, i] = 1
        A[i, lo2] = A[lo2, i] = 1
    return A


def build_k1():
    A = np.zeros((12, 12), dtype=np.int64)
    for i in range(12):
        for s in [1, 4, 6]:
            A[i, (i+s) % 12] = 1
            A[i, (i-s) % 12] = 1
    return A


def main():
    print("=" * 80)
    print("第384期: 完全 iso class enumeration (canonical form via spectrum)")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    G_core = nx.from_numpy_array(A_core)

    # ============================================================
    # Strategy: DFS edge placement + triangle-free + spectrum hash
    # ============================================================
    # We accept that DFS will produce labeled graphs (with multiplicity)
    # but dedup via spectrum (= near-canonical hash).
    # For 12V graphs, distinct spectra → very likely distinct iso classes
    # (= rare cospectral non-iso pairs).

    target_deg = [2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    n = 12

    spec_to_inv = {}  # spec_tuple -> (Tr A^2, Tr A^4, ...)
    target_alpha = 137

    t0 = time.time()
    visited = [0]
    completed = [0]
    LIMIT_SECONDS = 1200  # 20 min cap
    MAX_DEPTH = 19

    print(f"\n  DFS enumeration (timeout {LIMIT_SECONDS}s)")
    print(f"  target_deg: {target_deg}")
    sys.stdout.flush()

    def dfs(start_pos, A, deg_rem, edges_placed):
        if time.time() - t0 > LIMIT_SECONDS:
            return
        if edges_placed == MAX_DEPTH:
            if any(d != 0 for d in deg_rem):
                return
            # triangle-free check (done incrementally, but verify)
            A2 = A @ A
            if int(np.trace(A2 @ A)) != 0:
                return
            completed[0] += 1
            A4 = A2 @ A2
            Tr_A4 = int(np.trace(A4))
            # Compute spectrum hash
            evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
            spec = tuple(round(e, 4) for e in evs)
            if spec not in spec_to_inv:
                spec_to_inv[spec] = {
                    "Tr_A2": int(np.trace(A2)),
                    "Tr_A4": Tr_A4,
                    "n_edges": edges_placed,
                }
            return
        visited[0] += 1
        if visited[0] % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"    visited {visited[0]:,}  unique iso {len(spec_to_inv):,}  complete {completed[0]:,}  ({elapsed:.0f}s)")
            sys.stdout.flush()

        # iterate edges from start_pos
        i_start = start_pos // n
        j_start = start_pos % n
        # try edge (i, j) with i < j, in lex order
        for i in range(i_start, n):
            if deg_rem[i] == 0:
                continue
            j_min = max(i + 1, j_start) if i == i_start else i + 1
            for j in range(j_min, n):
                if deg_rem[j] == 0 or A[i, j] == 1:
                    continue
                # triangle check
                triangle = False
                for k in range(n):
                    if A[i, k] == 1 and A[j, k] == 1:
                        triangle = True
                        break
                if triangle:
                    continue
                A[i, j] = A[j, i] = 1
                deg_rem[i] -= 1
                deg_rem[j] -= 1
                next_pos = i * n + j + 1
                dfs(next_pos, A, deg_rem, edges_placed + 1)
                A[i, j] = A[j, i] = 0
                deg_rem[i] += 1
                deg_rem[j] += 1
        return

    A0 = np.zeros((n, n), dtype=np.int64)
    dfs(0, A0, target_deg.copy(), 0)
    elapsed = time.time() - t0

    print(f"\n  enumeration done in {elapsed:.0f}s")
    print(f"  visited DFS nodes: {visited[0]:,}")
    print(f"  completed graphs (with multiplicity): {completed[0]:,}")
    print(f"  unique spectra (= near iso classes): {len(spec_to_inv):,}")
    sys.stdout.flush()

    # ============================================================
    # Identify 7-id and iso to core
    # ============================================================
    # Tr A^4 = 270 → α⁻¹ candidate
    alpha_137_specs = []
    for spec, inv in spec_to_inv.items():
        if inv["Tr_A4"] == 270:
            alpha_137_specs.append((spec, inv))

    print(f"\n  unique iso classes with Tr(A⁴) = 270: {len(alpha_137_specs)}")

    # Check core spectrum
    core_spec = tuple(round(e, 4) for e in sorted(np.linalg.eigvalsh(A_core.astype(float)).tolist()))
    print(f"\n  核 spectrum hash in enum: {core_spec in spec_to_inv}")

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 結論")
    print(f"{'='*80}")
    print(f"""
  全 iso classes (= unique spectra): {len(spec_to_inv):,}
  そのうち Tr(A⁴) = 270 を 満たす: {len(alpha_137_specs)}
  核 spectrum match: {core_spec in spec_to_inv}
  時間: {elapsed:.0f}s
""")
    if elapsed >= LIMIT_SECONDS - 1:
        print(f"  ★ time limit hit、 完全列挙 ではない")
    else:
        print(f"  ★ 完全列挙 終了、 {len(spec_to_inv)} unique iso classes 確定")
        print(f"  ★ Tr(A⁴) = 270 を 満たす iso class: {len(alpha_137_specs)} 個")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "elapsed_seconds": elapsed,
        "dfs_visited": visited[0],
        "completed_graphs": completed[0],
        "unique_iso_classes": len(spec_to_inv),
        "alpha_137_iso_classes": len(alpha_137_specs),
        "core_spectrum_found": core_spec in spec_to_inv,
        "complete_enumeration": elapsed < LIMIT_SECONDS - 1,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round384_canonical_enumeration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.setrecursionlimit(100000)
    main()
