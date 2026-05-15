"""第354期: 核 vs 有名 graph 全 list で 強 baseline test.

「random 12V graph で比較」 では弱い。
有名な構造 graph (Petersen, Heawood, Möbius-Kantor 等) と比較すべき.

approach:
  (1) NetworkX で 多種の small named graph を生成
  (2) 各 graph に対し:
      α⁻¹ = ½Tr(A^4) + 2 を計算
      他の integer identity も計算
      物理 constants との match 数
  (3) 核 graph が unique か, あるいは他の famous graph も似た性質を持つか
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx


def build_icosahedron_adj():
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


def build_k1_adj():
    A = np.zeros((12, 12), dtype=np.int64)
    for i in range(12):
        for s in [1, 4, 6]:
            A[i, (i+s) % 12] = 1
            A[i, (i-s) % 12] = 1
    return A


def compute_invariants(A):
    A = A.astype(float)
    n = A.shape[0]
    n_edges = int(A.sum() / 2)
    degrees = sorted([int(A[i].sum()) for i in range(n)])
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    Tr_A2 = int(np.trace(A2))
    Tr_A3 = int(np.trace(A2 @ A))
    Tr_A4 = int(np.trace(A4))
    Tr_A6 = int(np.trace(A6))
    evs = sorted(np.linalg.eigvalsh(A).tolist())
    lam_min = evs[0]
    lam_max = evs[-1]
    alpha_inv_pred = Tr_A4 / 2 + 2
    K3_pred = n_edges + lam_min  # = |E| + λ_min
    catalan_pred = max(degrees) + Tr_A2
    return {
        "n_V": n, "n_E": n_edges,
        "deg_seq": tuple(degrees),
        "Tr_A2": Tr_A2, "Tr_A3": Tr_A3, "Tr_A4": Tr_A4, "Tr_A6": Tr_A6,
        "lam_min": round(lam_min, 4), "lam_max": round(lam_max, 4),
        "alpha_inv_pred": alpha_inv_pred,
        "K3_pred": round(K3_pred, 2),
        "catalan_pred": catalan_pred,
        "triangle_free": Tr_A3 == 0,
    }


def main():
    print("=" * 80)
    print("第354期: 核 vs 有名 graph 全 list で 強 baseline 比較")
    print("=" * 80)

    # ============================================================
    # (1) 核 reference
    # ============================================================
    A_core = np.minimum(build_k1_adj(), build_icosahedron_adj())
    core_inv = compute_invariants(A_core)
    print(f"\n  ★ 核 (K¹ ∩ Ico):")
    for k, v in core_inv.items():
        print(f"    {k:20s} = {v}")

    # ============================================================
    # (2) 有名 graph 集 (NetworkX built-in)
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(A) 有名 small graph list")
    print("="*80)

    famous_graphs = {}

    # NetworkX built-in named graphs
    try:
        famous_graphs["Petersen (10V)"] = nx.petersen_graph()
        famous_graphs["Heawood (14V)"] = nx.heawood_graph()
        famous_graphs["Möbius-Kantor (16V)"] = nx.moebius_kantor_graph()
        famous_graphs["Pappus (18V)"] = nx.pappus_graph()
        famous_graphs["Desargues (20V)"] = nx.desargues_graph()
        famous_graphs["Coxeter (28V)"] = nx.LCF_graph(28, [-10, -7, -2, 6, 4, 6, 2, 9], 4)
        famous_graphs["Tutte–Coxeter (30V)"] = nx.tutte_graph()
        famous_graphs["Tetrahedral (4V)"] = nx.tetrahedral_graph()
        famous_graphs["Octahedral (6V)"] = nx.octahedral_graph()
        famous_graphs["Cube (8V)"] = nx.cubical_graph()
        famous_graphs["Dodecahedral (20V)"] = nx.dodecahedral_graph()
        famous_graphs["Icosahedral (12V)"] = nx.icosahedral_graph()
        famous_graphs["K_5"] = nx.complete_graph(5)
        famous_graphs["K_6"] = nx.complete_graph(6)
        famous_graphs["K_{3,3}"] = nx.complete_bipartite_graph(3, 3)
        famous_graphs["K_{4,4}"] = nx.complete_bipartite_graph(4, 4)
        famous_graphs["K_{6,6}"] = nx.complete_bipartite_graph(6, 6)
        famous_graphs["C_12 (cycle)"] = nx.cycle_graph(12)
        famous_graphs["P_12 (path)"] = nx.path_graph(12)
        famous_graphs["Frucht (12V, 3-reg)"] = nx.frucht_graph()
        famous_graphs["Truncated tetrahedron (12V)"] = nx.truncated_tetrahedron_graph()
        famous_graphs["Cuboctahedral (12V)"] = nx.LCF_graph(12, [3, 2, 4, -3, -2, -4], 2)  # Cuboctahedron-like
        # Add cube of 12 vertices
        # Add some Cayley graphs
    except Exception as e:
        print(f"  warning: {e}")

    print(f"\n  total named graphs to test: {len(famous_graphs)}")

    # Add ALL 5-regular 12-vertex Cayley graphs via Z/12
    from itertools import combinations
    print(f"\n  Adding 5-regular Z/12 Cayley graphs:")
    for gen in combinations([1,2,3,4,5], 3):
        # Check if 0 not in gen (Cayley needs non-zero)
        if 6 in gen:
            continue
        # Try to make 5-regular by S = {±a, ±b, ±c, 6}
        try:
            S = list(gen) + [12-g for g in gen]
            if 6 not in S:
                S = S + [6]  # 6 is self-inverse, gives 5-regular
            S = sorted(set(S))
            A = np.zeros((12, 12), dtype=np.int64)
            for i in range(12):
                for s in S:
                    A[i, (i+s) % 12] = 1
            # Verify degree
            degs = A.sum(axis=1)
            if degs[0] == 5 and all(d == 5 for d in degs):
                G = nx.from_numpy_array(A)
                famous_graphs[f"Cay(Z/12, S={S})"] = G
        except Exception:
            pass

    print(f"  total graphs to test: {len(famous_graphs)}")

    # ============================================================
    # (3) Test each for α⁻¹ identity
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(B) Test each graph for α⁻¹ = ½Tr(A⁴)+2 = 137")
    print("="*80)

    matches_alpha_inv = []
    matches_K3 = []
    matches_catalan = []
    results = []

    for name, G in famous_graphs.items():
        n = G.number_of_nodes()
        if n < 4 or n > 30:
            continue
        A_g = nx.to_numpy_array(G).astype(int)
        try:
            inv = compute_invariants(A_g)
        except Exception:
            continue
        results.append((name, inv))
        # check α⁻¹
        if 130 <= inv["alpha_inv_pred"] <= 145:
            matches_alpha_inv.append((name, inv))
        # check K3 = 22
        if 21.5 <= inv["K3_pred"] <= 22.5:
            matches_K3.append((name, inv))
        # check Catalan = 42
        if inv["catalan_pred"] == 42:
            matches_catalan.append((name, inv))

    print(f"\n  graphs near α⁻¹ ∈ [130, 145]:")
    for name, inv in matches_alpha_inv:
        diff = abs(inv["alpha_inv_pred"] - 137) / 137 * 100
        flag = "★" if diff < 1 else " "
        print(f"    {flag} {name:35s}  n={inv['n_V']:2d} |E|={inv['n_E']:3d}  α⁻¹_pred={inv['alpha_inv_pred']}  diff {diff:.2f}%")

    print(f"\n  graphs near K3 rank ∈ [21.5, 22.5]:")
    for name, inv in matches_K3:
        print(f"    {name:35s}  K3_pred = {inv['K3_pred']}")

    print(f"\n  graphs with Catalan = 42 EXACT:")
    for name, inv in matches_catalan:
        print(f"    {name:35s}  catalan = {inv['catalan_pred']}")

    # ============================================================
    # (4) 同じ # of vertex の graph で 詳細比較
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(C) 12-vertex graph に絞った詳細比較")
    print("="*80)
    twelve_v = [(n, i) for n, i in results if i["n_V"] == 12]
    print(f"\n  12V graphs found: {len(twelve_v)}")
    for name, inv in twelve_v:
        is_core = name == "Cay(Z/12, S=[1, 4, 6, 8, 11])"  # K¹
        triangle_free = "triangle-free" if inv["triangle_free"] else "has triangles"
        flag = "★" if abs(inv["alpha_inv_pred"] - 137) < 1 else " "
        print(f"  {flag} {name:35s}  |E|={inv['n_E']:3d}  Tr A⁴={inv['Tr_A4']:5d}  α⁻¹_pred={inv['alpha_inv_pred']}  ({triangle_free})")

    # ============================================================
    # (5) The strict test: core's full invariant signature
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(D) 核 の 全 invariants 一致 graph search")
    print("="*80)
    core_signature = (
        core_inv["deg_seq"],
        core_inv["Tr_A2"], core_inv["Tr_A3"], core_inv["Tr_A4"],
        core_inv["Tr_A6"],
    )
    print(f"\n  核 signature: deg={core_inv['deg_seq']}, Tr(A²-A⁶) = ({core_inv['Tr_A2']}, {core_inv['Tr_A3']}, {core_inv['Tr_A4']}, {core_inv['Tr_A6']})")
    matches_full = []
    for name, inv in results:
        if inv["n_V"] != 12:
            continue
        sig = (inv["deg_seq"], inv["Tr_A2"], inv["Tr_A3"], inv["Tr_A4"], inv["Tr_A6"])
        if sig == core_signature:
            matches_full.append((name, inv))

    print(f"\n  全 invariants 一致 graph: {len(matches_full)}")
    for name, inv in matches_full:
        print(f"    {name}")

    # ============================================================
    # 統合
    # ============================================================
    print(f"\n\n{'='*80}")
    print("★ 統合 — 強 baseline test 結果")
    print("="*80)

    n_total = len(results)
    n_alpha_near = len([1 for n, i in results if 130 <= i["alpha_inv_pred"] <= 145])
    n_alpha_close = len([1 for n, i in results if abs(i["alpha_inv_pred"] - 137) < 1])
    n_alpha_exact = len([1 for n, i in results if i["alpha_inv_pred"] == 137])

    print(f"""
  テスト した graph 数:              {n_total}
  α⁻¹ ∈ [130, 145] な graph:        {n_alpha_near}
  α⁻¹ - 137 < 1 な graph:           {n_alpha_close}
  α⁻¹ = 137 EXACT な graph:         {n_alpha_exact}

  → 核 は α⁻¹ = 137 EXACT を 達成する **特殊な 1 個** か?
  → 全 12V graph 中 invariants 一致: {len(matches_full)} 個
""")

    if n_alpha_exact == 1:
        verdict = "★★★★★ 核は 有名 graph 中 唯一 α⁻¹ = 137 EXACT を達成 → 強い uniqueness"
    elif n_alpha_exact <= 3:
        verdict = f"★★★ 核を含む {n_alpha_exact} 個 graph が α⁻¹ = 137 達成 → 中庸"
    else:
        verdict = f"★★ {n_alpha_exact} 個 graph が α⁻¹ = 137 → 核以外にも同性質"
    print(f"  {verdict}")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "n_graphs_tested": n_total,
        "core_invariants": {k: str(v) for k, v in core_inv.items()},
        "alpha_inv_137_exact_count": n_alpha_exact,
        "alpha_inv_near_count": n_alpha_close,
        "K3_22_matches": len(matches_K3),
        "Catalan_42_matches": len(matches_catalan),
        "full_invariant_matches": len(matches_full),
        "verdict": verdict,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round354_famous_graphs.json"
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
