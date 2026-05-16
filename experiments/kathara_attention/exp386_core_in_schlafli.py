"""第386期: 核 graph が Schläfli graph の 12-vertex induced subgraph か?

approach:
  Schläfli は 27 vertex.
  C(27, 12) = 17M sub-vertex sets.
  random sampling 1M で 12-vertex induced subgraph を 取り、
  核 (12V19E triangle-free) と iso か check.

期待:
  もし 核 ⊂ Schläfli (induced subgraph relation) → 厳密 構造 link 確立
  もし なし → 核 と Schläfli は 数値的 関係 のみ
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import random
import sys
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


def build_schlafli():
    A = np.zeros((27, 27), dtype=np.int64)
    c_pairs = [(i, j) for i in range(1, 7) for j in range(i+1, 7)]
    c_idx = {pair: 12 + k for k, pair in enumerate(c_pairs)}
    # a_i (0-5) and b_j (6-11) meet iff i != j
    for i in range(6):
        for j in range(6):
            if i != j:
                A[i, 6+j] = A[6+j, i] = 1
    for i in range(6):
        for (k, l), idx in c_idx.items():
            if (i+1) in [k, l]:
                A[i, idx] = A[idx, i] = 1
                A[6+i, idx] = A[idx, 6+i] = 1
    for pair1 in c_pairs:
        for pair2 in c_pairs:
            if pair1 >= pair2:
                continue
            if set(pair1).isdisjoint(set(pair2)):
                A[c_idx[pair1], c_idx[pair2]] = A[c_idx[pair2], c_idx[pair1]] = 1
    return A


def core_invariants(A):
    """compute (Tr A^2, Tr A^3, Tr A^4) for quick filter"""
    A2 = A @ A
    A4 = A2 @ A2
    return (int(np.trace(A2)), int(np.trace(A2 @ A)), int(np.trace(A4)))


def main():
    print("=" * 80)
    print("第386期: 核 ⊂ Schläfli (induced subgraph relation) 検証")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    G_core = nx.from_numpy_array(A_core)
    A_S = build_schlafli()
    n_S = A_S.shape[0]
    core_inv = core_invariants(A_core)
    print(f"\n  核 invariants: Tr A² = {core_inv[0]}, Tr A³ = {core_inv[1]}, Tr A⁴ = {core_inv[2]}")
    print(f"  核 |V|=12, |E|={int(A_core.sum()/2)}, triangle-free: {core_inv[1] == 0}")

    # ============================================================
    # Random sample 12-vertex subsets
    # ============================================================
    print(f"\n  Schläfli から 12-vertex 部分集合 を sampling:")
    print(f"  total combinations: C(27,12) = {math.comb(27, 12):,}")

    rng = random.Random(42)
    N_SAMPLES = 1_000_000
    iso_count = 0
    inv_match_count = 0
    triangle_free_count = 0
    target_e = 19  # 核の edge 数
    target_e_match = 0

    t0 = time.time()
    for trial in range(N_SAMPLES):
        if trial % 100_000 == 0 and trial > 0:
            elapsed = time.time() - t0
            print(f"    trial {trial:,}  iso 核 {iso_count}  inv-match {inv_match_count}  Δ-free {triangle_free_count}  E=19 {target_e_match}  ({elapsed:.0f}s)")
            sys.stdout.flush()

        verts = rng.sample(range(n_S), 12)
        # Build induced subgraph adjacency
        A_sub = A_S[np.ix_(verts, verts)]
        # quick check: edges
        n_e = int(A_sub.sum() / 2)
        if n_e != target_e:
            continue
        target_e_match += 1
        # triangle check
        A2 = A_sub @ A_sub
        if int(np.trace(A2 @ A_sub)) != 0:
            continue
        triangle_free_count += 1
        # full invariants
        inv = (int(np.trace(A2)), 0, int(np.trace(A2 @ A2)))
        if inv != core_inv:
            continue
        inv_match_count += 1
        # full iso check
        G_sub = nx.from_numpy_array(A_sub)
        if nx.is_isomorphic(G_sub, G_core):
            iso_count += 1
            if iso_count <= 3:
                print(f"    ★ FOUND! trial {trial}, vertices = {sorted(verts)}")

    elapsed = time.time() - t0
    print(f"\n  {N_SAMPLES:,} trials in {elapsed:.0f}s")
    print(f"  E=19 graphs: {target_e_match:,}")
    print(f"  triangle-free + E=19: {triangle_free_count:,}")
    print(f"  full invariants match (= 核 spectrum): {inv_match_count}")
    print(f"  ★ iso to 核: {iso_count}")

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 結論")
    print(f"{'='*80}")
    print(f"""
  Schläfli (27V) から 12V induced subgraph を {N_SAMPLES:,} 個 sample:
    E=19 を 持つ subgraph: {target_e_match:,}
    そのうち triangle-free: {triangle_free_count:,}
    そのうち 核 invariants 完全一致: {inv_match_count}
    そのうち 核 と iso: {iso_count}
""")
    if iso_count > 0:
        print(f"  ★★★★★ 核 ⊂ Schläfli 厳密 確認!")
        print(f"     → 核 = Schläfli graph の 特定 12-vertex induced subgraph")
        print(f"     → 「核 = E_6 cubic surface 27 線 の 部分構造」 EXPLICIT 証明")
    elif inv_match_count > 0:
        print(f"  ★★ inv match あり ({inv_match_count}) だが iso 不一致")
        print(f"     → cospectral mate、 厳密 同定 部分的")
    elif triangle_free_count > 0:
        print(f"  ★ triangle-free + E=19 あり ({triangle_free_count}) だが invariants 違う")
        print(f"     → 核 spectrum 別物")
    else:
        print(f"  ✗ Schläfli induced subgraph で 核 候補 なし")
        print(f"     → 核 ⊄ Schläfli (induced) — 数値関係 のみ")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "n_samples": N_SAMPLES,
        "elapsed_seconds": elapsed,
        "E_19_count": target_e_match,
        "triangle_free_count": triangle_free_count,
        "invariant_match_count": inv_match_count,
        "iso_to_core_count": iso_count,
        "core_subseteq_schlafli_induced": iso_count > 0,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round386_core_in_schlafli.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
