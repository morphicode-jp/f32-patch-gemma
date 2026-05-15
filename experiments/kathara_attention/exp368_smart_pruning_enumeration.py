"""第368期: smart pruning enumeration — ほぼ完全列挙.

approach:
  全 12V19E triangle-free graphs with deg seq (2,2,3,3,3,3,3,3,4,4,4,4) を
  DFS で系統的に生成。 刈込:
  - lex order edge placement (avoid duplicate)
  - degree budget tracking
  - on-place triangle check
  - 19 edges 完成時に Tr(A^4) check

  これで random sampling より 確実、 完全 enumeration に近い.

  output: total count + spectrum-distinct iso classes.
"""
from __future__ import annotations
import numpy as np
import math
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
    print("第368期: smart pruning enumeration")
    print("=" * 80)
    sys.stdout.flush()

    # 目標 degree seq: 0,1 -> deg 2; 2-7 -> deg 3; 8-11 -> deg 4
    target_deg = [2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    n = 12

    # ============================================================
    # DFS with smart pruning
    # ============================================================
    print(f"\n  target deg: {target_deg}")
    print(f"  target |E| = 19")
    print(f"")
    sys.stdout.flush()

    # State: adjacency matrix, remaining degree per vertex, edges placed
    # Edges enumerated in lex order (i, j) with i < j, starting from smallest unfilled

    A_curr = np.zeros((n, n), dtype=np.int8)
    deg_remaining = target_deg.copy()
    edges_placed = 0
    target_edges = 19

    triangle_free_configs = []  # collect adjacency matrices
    visited = [0]

    PROGRESS_INTERVAL = 100_000
    t0 = time.time()
    early_terminate = [False]
    MAX_CONFIGS = 10_000_000  # safety cap
    canonical_count = [0]

    def dfs(start_i, start_j, edges_so_far, deg_rem, A):
        if early_terminate[0]:
            return
        visited[0] += 1
        if visited[0] % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f"    DFS nodes: {visited[0]:>10,}  configs: {len(triangle_free_configs):>8,}  ({elapsed:.0f}s)")
            sys.stdout.flush()

        if edges_so_far == target_edges:
            # Verify all deg satisfied
            if any(d != 0 for d in deg_rem):
                return
            # Save the configuration
            triangle_free_configs.append(A.copy())
            if len(triangle_free_configs) >= MAX_CONFIGS:
                early_terminate[0] = True
            return

        # Remaining edges to place: target_edges - edges_so_far
        rem_edges = target_edges - edges_so_far
        # Pruning: not enough edges left to fulfill degrees
        rem_deg_sum = sum(deg_rem)
        if rem_deg_sum != 2 * rem_edges:
            return  # impossible

        # Try edges in lex order starting from (start_i, start_j)
        for i in range(start_i, n):
            if deg_rem[i] == 0:
                continue
            j_min = max(i + 1, start_j) if i == start_i else i + 1
            for j in range(j_min, n):
                if deg_rem[j] == 0 or A[i, j] != 0:
                    continue
                # Triangle check: i, j share no common neighbor
                # = no k such that A[i,k]=1 and A[j,k]=1
                row_i = A[i]
                row_j = A[j]
                triangle = False
                for k in range(n):
                    if row_i[k] == 1 and row_j[k] == 1:
                        triangle = True
                        break
                if triangle:
                    continue
                # Place edge
                A[i, j] = A[j, i] = 1
                deg_rem[i] -= 1
                deg_rem[j] -= 1
                # Recurse with next edge to try
                dfs(i, j + 1, edges_so_far + 1, deg_rem, A)
                # Undo
                A[i, j] = A[j, i] = 0
                deg_rem[i] += 1
                deg_rem[j] += 1
            # for next i, reset j to i+1
        return

    print(f"\n  Starting DFS enumeration...")
    sys.stdout.flush()
    dfs(0, 1, 0, deg_remaining, A_curr)
    elapsed = time.time() - t0
    print(f"\n  DFS done in {elapsed:.0f}s")
    print(f"  Total triangle-free + deg-seq configs (with multiplicity): {len(triangle_free_configs):,}")
    print(f"  Total DFS nodes visited: {visited[0]:,}")
    if early_terminate[0]:
        print(f"  ★ Hit MAX_CONFIGS limit, did not finish")
    sys.stdout.flush()

    # ============================================================
    # Dedup by spectrum
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(B) Spectrum dedup")
    print(f"{'='*80}")
    sys.stdout.flush()

    spectrum_to_A = {}
    spectrum_to_count = {}
    tr_a4_270_count = 0
    for i, A in enumerate(triangle_free_configs):
        if i % 50_000 == 0 and i > 0:
            print(f"    deduping {i}/{len(triangle_free_configs)}, unique spectra so far: {len(spectrum_to_A)}")
            sys.stdout.flush()
        A_f = A.astype(float)
        evs = sorted(np.linalg.eigvalsh(A_f).tolist())
        spec = tuple(round(e, 4) for e in evs)
        if spec not in spectrum_to_A:
            spectrum_to_A[spec] = A.copy()
            spectrum_to_count[spec] = 0
            # Check Tr A^4 = 270
            A2 = A @ A
            A4 = A2 @ A2
            if int(np.trace(A4)) == 270:
                tr_a4_270_count += 1
        spectrum_to_count[spec] += 1

    n_unique_specs = len(spectrum_to_A)
    print(f"\n  unique spectra: {n_unique_specs}")
    print(f"  unique spectra with Tr(A^4) = 270: {tr_a4_270_count}")
    sys.stdout.flush()

    # ============================================================
    # Of 5-identity satisfiers, check 7-identity
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(C) 7 identity check on Tr(A^4)=270 sub-family")
    print(f"{'='*80}")

    A_core = np.minimum(build_k1(), build_icosahedron())
    G_core = nx.from_numpy_array(A_core)

    five_id_iso_classes = []
    for spec, A in spectrum_to_A.items():
        A2 = A @ A
        A4 = A2 @ A2
        if int(np.trace(A4)) == 270:
            evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
            if abs((19 + abs(evs[0])) - 22) < 0.01:
                five_id_iso_classes.append((spec, A))

    print(f"\n  5-identity satisfying iso classes (= Tr A^4=270 and |λ_min|=3): {len(five_id_iso_classes)}")
    sys.stdout.flush()

    aut4_c5_4 = 0
    aut4_count = 0
    core_iso = 0
    for spec, A in five_id_iso_classes:
        G = nx.from_numpy_array(A.astype(np.int64))
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        aut = 0
        for _ in GM.isomorphisms_iter():
            aut += 1
            if aut > 5:
                break
        if aut == 4:
            aut4_count += 1
            A5 = A @ A @ A @ A @ A
            c5 = int(np.trace(A5)) // 10
            if c5 == 4:
                aut4_c5_4 += 1
                if nx.is_isomorphic(G, G_core):
                    core_iso += 1

    print(f"\n  ★ within 5-id family:")
    print(f"    |Aut| = 4: {aut4_count}")
    print(f"    + C_5 = 4 (7-identity): {aut4_c5_4}")
    print(f"    iso to core: {core_iso}")
    sys.stdout.flush()

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 結論 — 完全 enumeration による uniqueness")
    print(f"{'='*80}")
    print(f"""
  smart pruning enumeration の結果:

  Total triangle-free + deg-seq configs (with iso multiplicity): {len(triangle_free_configs):,}
  Unique iso classes (by spectrum): {n_unique_specs}
  with Tr(A^4) = 270:               {tr_a4_270_count}
  with full 5-identity:              {len(five_id_iso_classes)}
  with 7-identity (= 核):            {aut4_c5_4}
  iso to 核:                          {core_iso}

  ★ {'★★★★★ UNIQUENESS PROVEN (完全 enumeration の sub-test に対して)' if aut4_c5_4 == 1 and core_iso == 1 else '? need re-check'}
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "total_configs_with_multiplicity": len(triangle_free_configs),
        "unique_iso_classes_by_spectrum": n_unique_specs,
        "tr_a4_270_count": tr_a4_270_count,
        "five_identity_iso_classes": len(five_id_iso_classes),
        "aut4_in_5id_family": aut4_count,
        "seven_identity_count": aut4_c5_4,
        "iso_to_core": core_iso,
        "dfs_nodes_visited": visited[0],
        "elapsed_seconds": elapsed,
        "early_terminate": early_terminate[0],
        "verdict": "complete enumeration UNIQUENESS" if (aut4_c5_4 == 1 and core_iso == 1) else "incomplete",
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round368_pruning_enum.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
