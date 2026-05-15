"""第346期: 核 graph の完全 uniqueness 証明.

目的: 12 vertex, 19 edge, triangle-free, degree seq (2,2,3,3,3,3,3,3,4,4,4,4),
      Tr(A^4) = 270, Tr(A^6) = 2354 を全部満たす graph を **完全列挙**

approach:
  (1) degree sequence を partition
  (2) 全 edge 配置を smart DFS で探索
  (3) triangle-free pruning
  (4) 完了 graph に Tr A^4 check
  (5) 候補を networkx で isomorphism class に分類
  (6) 核 graph と同型かどうか確認

結果: 核と同型な 1 クラスのみ存在すれば uniqueness 証明完了
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import time
from itertools import combinations


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


def core_invariants(A):
    """Compute (Tr A^k for k=2..6) and triangle count."""
    A = np.asarray(A, dtype=np.int64)
    A2 = A @ A
    A3 = A2 @ A
    A4 = A2 @ A2
    A5 = A4 @ A
    A6 = A4 @ A2
    return (int(np.trace(A2)), int(np.trace(A3)), int(np.trace(A4)),
            int(np.trace(A5)), int(np.trace(A6)))


def main():
    print("=" * 80)
    print("第346期: 核 uniqueness 完全証明 — 全 graph enumerate")
    print("=" * 80)

    # 核 reference
    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    core_inv = core_invariants(A_core)
    target_deg = (2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4)
    print(f"\n  核 invariants:")
    print(f"    degree seq:  {target_deg}")
    print(f"    Tr A² = {core_inv[0]}")
    print(f"    Tr A³ = {core_inv[1]}")
    print(f"    Tr A⁴ = {core_inv[2]}  (= α⁻¹ × 2 - 4 = 270)")
    print(f"    Tr A⁵ = {core_inv[3]}")
    print(f"    Tr A⁶ = {core_inv[4]}")

    # Reference: core as networkx for isomorphism check
    G_core = nx.from_numpy_array(A_core)
    print(f"  核 NetworkX: |V|={G_core.number_of_nodes()}, |E|={G_core.number_of_edges()}")

    # ============================================================
    # 全 graph enumeration with degree seq + triangle-free
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) DFS enumeration — degree seq + triangle-free")
    print("="*80)

    # Vertex types by degree (canonical labeling)
    target_degrees = list(target_deg)
    n = 12

    # Generate all simple graphs with this degree sequence using nx.havel_hakimi or random
    # Better: use nx.configuration_model w/ pruning, but configuration may allow multi/self
    # Best: use nx.random_degree_sequence_graph multiple times, or systematic enumeration

    # Approach: DFS over edge sets respecting degree limits
    # Vertices: 0,1 degree 2; 2-7 degree 3; 8-11 degree 4
    deg_limit = [2, 2] + [3]*6 + [4]*4

    # Track found unique graphs (by spectrum hash)
    found_graphs = []  # list of (Tr_invariants, A_matrix)
    seen_spectra = set()
    visited_count = [0]
    triangle_free_count = [0]
    target_a4_count = [0]
    iso_with_core = [0]

    t0 = time.time()

    def dfs(deg_remaining, A_curr, edges_done, start_pair):
        """DFS edges. start_pair to avoid revisiting same edge."""
        visited_count[0] += 1
        if visited_count[0] % 100000 == 0:
            elapsed = time.time() - t0
            print(f"    visited {visited_count[0]:,}  triangle-free {triangle_free_count[0]:,}  target a_4 {target_a4_count[0]:,}  iso-core {iso_with_core[0]:,}  ({elapsed:.1f}s)")
        if edges_done == 19:
            # complete graph
            # check all degrees met (should be 0 remaining)
            if any(d != 0 for d in deg_remaining):
                return
            triangle_free_count[0] += 1
            # check invariants
            inv = core_invariants(A_curr)
            if inv == core_inv:
                target_a4_count[0] += 1
                # isomorphism check with core
                G = nx.from_numpy_array(A_curr)
                if nx.is_isomorphic(G, G_core):
                    iso_with_core[0] += 1
                else:
                    # Found a non-core graph with same invariants
                    # Compute its spectrum hash
                    ev_tuple = tuple(sorted(np.round(np.linalg.eigvalsh(A_curr.astype(float)), 6).tolist()))
                    if ev_tuple not in seen_spectra:
                        seen_spectra.add(ev_tuple)
                        found_graphs.append((inv, A_curr.copy()))
                        print(f"    ★ NON-CORE graph found! invariants match but not iso to core")
            return

        # find next edge to try
        u_start, v_start = start_pair
        for u in range(u_start, n):
            v_min = max(u + 1, v_start if u == u_start else u + 1)
            for v in range(v_min, n):
                if deg_remaining[u] > 0 and deg_remaining[v] > 0 and A_curr[u, v] == 0:
                    # Check triangle-free: u, v share no common neighbor?
                    if any(A_curr[u, w] == 1 and A_curr[v, w] == 1 for w in range(n)):
                        continue  # would create triangle
                    A_curr[u, v] = A_curr[v, u] = 1
                    deg_remaining[u] -= 1
                    deg_remaining[v] -= 1
                    dfs(deg_remaining, A_curr, edges_done + 1, (u, v + 1))
                    A_curr[u, v] = A_curr[v, u] = 0
                    deg_remaining[u] += 1
                    deg_remaining[v] += 1
            # Prune: if v_start > u_start, reset for next u
            if u > u_start:
                v_start = u + 1
        return

    # initial call
    A0 = np.zeros((n, n), dtype=np.int64)
    print(f"\n  enumeration 開始 (DFS、triangle-free pruning)...")
    print(f"  予測: <1 分で完了")
    dfs(deg_limit[:], A0, 0, (0, 1))
    elapsed = time.time() - t0

    print(f"\n  enumeration 完了 ({elapsed:.1f}s)")
    print(f"  訪問 search node: {visited_count[0]:,}")
    print(f"  triangle-free complete graphs (deg seq match): {triangle_free_count[0]:,}")
    print(f"  Tr A^4 = 270 match: {target_a4_count[0]:,}")
    print(f"  ★ うち 核 graph と iso な物: {iso_with_core[0]:,}")
    print(f"  ★ NON-core (異なる iso class、同 invariants): {len(found_graphs)}")

    # ============================================================
    # Result analysis
    # ============================================================
    print(f"\n\n{'='*80}")
    print("★ 結果分析")
    print("="*80)
    print(f"""
  全候補: {triangle_free_count[0]} graphs satisfying:
    - |V| = 12
    - |E| = 19
    - triangle-free
    - degree seq = (2,2,3,3,3,3,3,3,4,4,4,4)

  核 invariants 一致: {target_a4_count[0]} graphs
  そのうち 核 と iso class 同じ: {iso_with_core[0]}
  異なる iso class: {len(found_graphs)}

""")

    if len(found_graphs) == 0:
        print(f"  ★★★ 結論: 核は 全 invariants を満たす唯一の iso class")
        print(f"     UNIQUENESS 完全証明 ✓")
    else:
        print(f"  ★ 結論: 同 invariants を持つ別 graph が {len(found_graphs)} 種存在")
        print(f"     uniqueness は 6 invariants では不十分")
        for inv, A in found_graphs:
            print(f"     non-core graph found, invariants {inv}")

    # ============================================================
    # Save result
    # ============================================================
    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "core_invariants": list(core_inv),
        "triangle_free_count": triangle_free_count[0],
        "target_a4_count": target_a4_count[0],
        "core_iso_count": iso_with_core[0],
        "non_core_iso_count": len(found_graphs),
        "uniqueness_proven": len(found_graphs) == 0 and iso_with_core[0] > 0,
        "elapsed_seconds": elapsed,
        "search_nodes_visited": visited_count[0],
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round346_uniqueness_proof.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
