"""第382期: K¹ ∩ 他 12V graph intersection で 物理 invariant 探索.

approach:
  Cay(Z/12, S) for multiple S, ∩ Ico
  または K¹ ∩ X for X ∈ {12V 正多面体 candidates}
  各 intersection の Tr(A⁴)/2 + 2 を 物理定数 と照合
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import sys
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


def build_cay(n, generators):
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for s in generators:
            A[i, (i+s) % n] = 1
            A[i, (i-s) % n] = 1
    return A


def build_cuboctahedron():
    """Cuboctahedron 12V 24E 4-regular."""
    return nx.to_numpy_array(nx.LCF_graph(12, [3, 2, -1, -2, -3, 1], 2)).astype(np.int64)


def build_truncated_tetra():
    """Truncated tetrahedron 12V 18E 3-regular."""
    return nx.to_numpy_array(nx.truncated_tetrahedron_graph()).astype(np.int64)


def build_anti_prism_12():
    """Antiprism Z/6 × Z/2, 12V."""
    return nx.to_numpy_array(nx.LCF_graph(12, [-3, 3], 6)).astype(np.int64)


def build_hexagonal_prism():
    """Hexagonal prism 12V 18E 3-regular."""
    G = nx.cartesian_product(nx.cycle_graph(6), nx.path_graph(2))
    return nx.to_numpy_array(G).astype(np.int64)


def compute_inv(A):
    A = np.asarray(A, dtype=np.int64)
    n = A.shape[0]
    n_e = int(A.sum() / 2)
    if n_e == 0:
        return None
    A2 = A @ A
    A4 = A2 @ A2
    Tr_A4 = int(np.trace(A4))
    Tr_A3 = int(np.trace(A2 @ A))
    degrees = [int(A[i].sum()) for i in range(n)]
    alpha_pred = Tr_A4 // 2 + 2
    return {
        "n_V": n, "n_E": n_e,
        "max_deg": max(degrees), "min_deg": min(degrees),
        "Tr_A2": int(np.trace(A2)),
        "Tr_A3": Tr_A3,
        "Tr_A4": Tr_A4,
        "alpha_pred": alpha_pred,
        "triangle_free": Tr_A3 == 0,
    }


def main():
    print("=" * 80)
    print("第382期: K¹ ∩ 他 12V graph 系統探索")
    print("=" * 80)
    sys.stdout.flush()

    # K¹ base
    K1 = build_cay(12, [1, 4, 6])
    Ico = build_icosahedron()

    # Various 12V partners
    partners = {
        "Icosahedron (= 既知 核 母)": Ico,
        "Cuboctahedron": build_cuboctahedron(),
        "Truncated tetra": build_truncated_tetra(),
        "Antiprism": build_anti_prism_12(),
        "Hexagonal prism": build_hexagonal_prism(),
        "K_{6,6}": nx.to_numpy_array(nx.complete_bipartite_graph(6, 6)).astype(np.int64),
        "C_12 (cycle)": nx.to_numpy_array(nx.cycle_graph(12)).astype(np.int64),
        "Frucht (12V cubic)": nx.to_numpy_array(nx.frucht_graph()).astype(np.int64),
        "Cay(Z/12,{1,5,6})": build_cay(12, [1, 5, 6]),
        "Cay(Z/12,{2,3,6})": build_cay(12, [2, 3, 6]),
        "Cay(Z/12,{1,3,6})": build_cay(12, [1, 3, 6]),
    }

    PHYSICAL = {
        137: "α⁻¹ (微細構造)",
        1836: "m_p/m_e",
        206: "m_μ/m_e",
        125: "m_H GeV (5³)",
        91: "m_Z GeV",
        173: "m_top GeV",
        240: "E_8 root",
        248: "E_8 adjoint",
        78: "E_6 adjoint",
        65: "δ_CP_quark °",
        22: "K3 rank",
        42: "Catalan C_5",
        26: "bosonic D",
        24: "Niemeier",
    }

    print(f"\n  K¹ alone:")
    inv_k1 = compute_inv(K1)
    print(f"    {inv_k1}")
    print(f"\n  各 partner と の intersection:")
    print(f"  {'partner':30s}  {'|V|':>4s} {'|E|':>4s} {'Tr A⁴':>8s} {'α⁻¹_pred':>10s} {'physical hit':>20s}")
    sys.stdout.flush()

    results = []
    for name, partner in partners.items():
        # ensure same shape
        if partner.shape != (12, 12):
            print(f"  {name}: shape {partner.shape}, skip")
            continue
        intersection = np.minimum(K1, partner)
        inv = compute_inv(intersection)
        if inv is None:
            continue
        # match against physical
        hits = []
        for target, p_name in PHYSICAL.items():
            if abs(inv["alpha_pred"] - target) <= 2:
                hits.append(f"{target} ({p_name})")
        hit_str = ", ".join(hits) if hits else "-"
        is_tri_free = "Δ-free" if inv["triangle_free"] else ""
        flag = "★" if hits else " "
        print(f"  {flag} {name:28s}  {inv['n_V']:>4d} {inv['n_E']:>4d} {inv['Tr_A4']:>8d} {inv['alpha_pred']:>10d}  {hit_str:>20s} {is_tri_free}")
        results.append({"partner": name, "inv": inv, "hits": hits})

    sys.stdout.flush()

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 結論 — K¹ ∩ 他 graph で 物理 invariant encode 探索")
    print(f"{'='*80}")
    n_hits = sum(1 for r in results if r["hits"])
    print(f"\n  total partners tested: {len(results)}")
    print(f"  K¹ ∩ X で 物理定数 hit する partner 数: {n_hits}")
    print(f"")
    if n_hits > 1:
        print(f"  hits リスト:")
        for r in results:
            if r["hits"]:
                print(f"    {r['partner']}: pred = {r['inv']['alpha_pred']}, hits = {r['hits']}")
    print(f"""
  honest 解釈:
    Icosahedron 以外 の partner で 物理 hit が ある か.
    多数 hit → 「K¹ ∩ X」 一般 メカニズム → 核 unique 性 弱化
    1 つ (Ico) のみ → 核 は special intersection (= 既知 主張 維持)
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "n_partners": len(results),
        "n_physical_hits": n_hits,
        "results": [
            {"partner": r["partner"], "alpha_pred": r["inv"]["alpha_pred"],
             "Tr_A4": r["inv"]["Tr_A4"], "n_E": r["inv"]["n_E"],
             "triangle_free": r["inv"]["triangle_free"], "hits": r["hits"]}
            for r in results
        ],
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round382_other_intersections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
