"""第358期: 144-graph family 詳細分析 — 核を unique にする invariant 発見.

approach:
  (1) 144 個の 5/5 iso class を collect
  (2) 各 graph で 追加 invariant を計算:
      - Tr(A^k) for k=2..10
      - eigenvalue 集合の完全 form
      - automorphism group order
      - girth, diameter
      - 4-cycle, 5-cycle, 6-cycle counts
      - bipartiteness, connectivity, etc.
  (3) 核 と他 143 個 で 違う invariant を 探す
  (4) 「核を unique にする 最小 invariant set」 を 報告
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


def full_invariants(A):
    """Compute all available invariants."""
    A = np.asarray(A, dtype=np.int64)
    n = A.shape[0]
    if A.sum() == 0:
        return {}
    n_e = int(A.sum() / 2)
    degrees = sorted([int(A[i].sum()) for i in range(n)])
    G = nx.from_numpy_array(A)

    # Powers
    A2 = A @ A
    A3 = A2 @ A
    A4 = A2 @ A2
    A5 = A4 @ A
    A6 = A4 @ A2
    A7 = A6 @ A
    A8 = A4 @ A4

    Tr_powers = {f"Tr_A^{k}": int(np.trace(M)) for k, M in zip(range(2, 9), [A2, A3, A4, A5, A6, A7, A8])}

    # Spectrum
    evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
    spec = tuple(round(e, 6) for e in evs)

    # Cycles
    girth = None
    try:
        girth = nx.girth(G)
    except Exception:
        pass

    diameter = None
    try:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
    except Exception:
        pass

    # Automorphism count via networkx ISO matcher (slow for large, fine for 12V)
    aut_count = None
    try:
        # Count by self-isomorphisms
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        count = 0
        for _ in GM.isomorphisms_iter():
            count += 1
            if count > 1000:
                break
        aut_count = count
    except Exception:
        pass

    # 4-cycle count
    # Tr A^4 = 2|E| + 4*paths_len_2 + 8*C_4
    p2 = sum(d*(d-1)//2 for d in degrees)
    c4 = (int(np.trace(A4)) - 2*n_e - 4*p2) // 8

    # 5-cycle count
    # Tr A^5 = 30 C_5 + ... (only odd contributes from cycles directly)
    # For triangle-free graphs (girth >= 4), 5-cycle count formula complex
    Tr_A5 = int(np.trace(A5))
    # For triangle-free, Tr A^5 = 10 * C_5
    c5 = Tr_A5 // 10 if girth and girth >= 4 else None

    # Connectivity
    connected = nx.is_connected(G)
    bipartite = nx.is_bipartite(G)

    return {
        "n_V": n, "n_E": n_e,
        "deg_seq": tuple(degrees),
        **Tr_powers,
        "lam_min": round(evs[0], 4),
        "lam_max": round(evs[-1], 4),
        "spec_short": tuple(round(e, 2) for e in evs[:6]),
        "girth": girth,
        "diameter": diameter,
        "aut_count": aut_count,
        "C_4_count": c4,
        "C_5_count": c5,
        "connected": connected,
        "bipartite": bipartite,
    }


def main():
    print("=" * 80)
    print("第358期: 144-graph family 詳細分析")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron())
    print(f"\n  核 invariants 計算 中...")
    sys.stdout.flush()
    core_inv = full_invariants(A_core)
    print(f"\n  ★ 核:")
    for k, v in core_inv.items():
        print(f"    {k:18s} = {v}")

    # ============================================================
    # 144 個の siblings を collect
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(A) 5/5 identity を満たす graph を collect (sampling)")
    print("="*80)
    sys.stdout.flush()

    target_deg = (2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4)
    rng = np.random.default_rng(42)

    siblings = {}  # spec -> {A, count}
    N_max = 5_000_000

    t0 = time.time()
    for trial in range(N_max):
        if trial % 500_000 == 0 and trial > 0:
            elapsed = time.time() - t0
            print(f"    trial {trial:,}  unique 5/5 spec: {len(siblings)}  ({elapsed:.0f}s)")
            sys.stdout.flush()
        if len(siblings) >= 144:
            print(f"    144 個収集完了 at trial {trial:,}")
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
        if int(np.trace(A2 @ A)) != 0:  # triangle check
            continue
        A4 = A2 @ A2
        Tr_A4 = int(np.trace(A4))
        if Tr_A4 != 270:
            continue
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        n_e = 19
        abs_lam_min = abs(evs[0])
        if abs((n_e + abs_lam_min) - 22) > 0.01:
            continue
        Tr_A2 = int(np.trace(A2))
        max_deg = 4
        if (max_deg + Tr_A2) != 42:
            continue
        # all 5 satisfied
        spec = tuple(round(e, 4) for e in evs)
        if spec not in siblings:
            siblings[spec] = {"A": A.copy(), "count": 1}
        else:
            siblings[spec]["count"] += 1

    elapsed = time.time() - t0
    print(f"\n  collection 完了 ({elapsed:.0f}s):  unique 5/5 iso classes: {len(siblings)}")
    sys.stdout.flush()

    # ============================================================
    # 各 sibling の full invariants 計算
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(B) 各 sibling の full invariants 比較")
    print("="*80)
    sys.stdout.flush()

    # Compute for core and a sample of siblings
    print(f"\n  核 と 各 sibling の Tr(A^k) for k=7,8 比較:")
    print(f"  核: Tr A^7 = {core_inv['Tr_A^7']}, Tr A^8 = {core_inv['Tr_A^8']}")
    print(f"")

    # Check if any sibling has same Tr A^7 and Tr A^8 as core
    same_higher = 0
    different_higher = 0
    sample_siblings = list(siblings.items())[:30]
    for spec, info in sample_siblings:
        A = info["A"]
        A2 = A @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        A7 = A6 @ A
        A8 = A4 @ A4
        sib_Tr7 = int(np.trace(A7))
        sib_Tr8 = int(np.trace(A8))
        match = (sib_Tr7 == core_inv['Tr_A^7']) and (sib_Tr8 == core_inv['Tr_A^8'])
        if match:
            same_higher += 1
        else:
            different_higher += 1

    print(f"  sample 30 sibling のうち:")
    print(f"    Tr A^7 + Tr A^8 共一致: {same_higher}")
    print(f"    異なる: {different_higher}")

    # ============================================================
    # Aut group 比較
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(C) 核 vs sibling の自己同型群 |Aut|")
    print("="*80)
    print(f"\n  核 |Aut| = {core_inv['aut_count']}")
    sys.stdout.flush()

    print(f"\n  各 sibling の |Aut|:")
    aut_distribution = {}
    sample_siblings_aut = list(siblings.items())[:50]
    for spec, info in sample_siblings_aut:
        A = info["A"]
        G = nx.from_numpy_array(A)
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        count = 0
        for _ in GM.isomorphisms_iter():
            count += 1
            if count > 100:
                break
        aut_distribution[count] = aut_distribution.get(count, 0) + 1

    for aut, n in sorted(aut_distribution.items()):
        marker = " ★ 核" if aut == core_inv['aut_count'] else ""
        print(f"    |Aut| = {aut}: {n} graphs{marker}")

    # ============================================================
    # 4/5 cycle counts 比較
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(D) 4-cycle, 5-cycle counts 比較")
    print("="*80)
    print(f"\n  核: C_4 = {core_inv['C_4_count']}, C_5 = {core_inv['C_5_count']}")
    print(f"")

    c4_distribution = {}
    c5_distribution = {}
    for spec, info in list(siblings.items())[:50]:
        A = info["A"]
        A2 = A @ A
        A4 = A2 @ A2
        p2 = sum(d*(d-1)//2 for d in [int(A[i].sum()) for i in range(12)])
        c4 = (int(np.trace(A4)) - 38 - 4*p2) // 8
        Tr_A5 = int(np.trace(A4 @ A))
        c5 = Tr_A5 // 10
        c4_distribution[c4] = c4_distribution.get(c4, 0) + 1
        c5_distribution[c5] = c5_distribution.get(c5, 0) + 1

    print(f"  C_4 distribution in sample:")
    for c4, n in sorted(c4_distribution.items()):
        marker = " ★ 核" if c4 == core_inv['C_4_count'] else ""
        print(f"    C_4 = {c4}: {n} graphs{marker}")

    print(f"\n  C_5 distribution in sample:")
    for c5, n in sorted(c5_distribution.items()):
        marker = " ★ 核" if c5 == core_inv['C_5_count'] else ""
        print(f"    C_5 = {c5}: {n} graphs{marker}")

    # ============================================================
    # 統合 — 核を unique にする invariant
    # ============================================================
    print(f"\n\n{'='*80}")
    print("★ 統合 — 核 を 144 個から区別する invariant")
    print("="*80)
    print(f"""
  144 個の siblings が 5 identity (α⁻¹=137 等) を共有.
  追加 invariant で 核 が unique かどうか:

  分析候補:
    Tr(A^7)、 Tr(A^8) ← higher walk counts
    |Aut| automorphism group order
    C_4, C_5, C_6 cycle counts
    diameter, girth
    bipartiteness (核 は non-bipartite)
""")

    # Find truly unique core property
    core_unique_props = []
    if core_inv['aut_count'] not in [a for a, _ in aut_distribution.items() if _ > 1]:
        core_unique_props.append(f"|Aut| = {core_inv['aut_count']}")
    if core_inv['C_4_count'] not in [c for c, _ in c4_distribution.items() if _ > 1]:
        core_unique_props.append(f"C_4 = {core_inv['C_4_count']}")
    if core_inv['C_5_count'] not in [c for c, _ in c5_distribution.items() if _ > 1]:
        core_unique_props.append(f"C_5 = {core_inv['C_5_count']}")

    print(f"  ★ 核を 144 中 unique にする追加性質候補:")
    for p in core_unique_props:
        print(f"     - {p}")
    if not core_unique_props:
        print(f"     (sample 内では 一意性 確認できず)")

    print(f"""
  予測:
    Tr(A^7) + Tr(A^8) で uniqueness は (sample 30 で different {different_higher}/30 なら大半 異なる)
    |Aut| = {core_inv['aut_count']} は どれだけ rare か - check 結果上記

  → 5 identity + Tr A^7/A^8 + |Aut| の 同時満足で 核は強い uniqueness 候補
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "core_invariants": {k: str(v) for k, v in core_inv.items()},
        "n_siblings_collected": len(siblings),
        "higher_Tr_sample_test": {
            "matches_core_higher": same_higher,
            "differs": different_higher,
            "sample_size": 30,
        },
        "aut_distribution": dict(aut_distribution),
        "C_4_distribution": dict(c4_distribution),
        "C_5_distribution": dict(c5_distribution),
        "core_unique_properties": core_unique_props,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round358_family_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
