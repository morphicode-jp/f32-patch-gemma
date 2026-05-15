"""第376期: Kathara 階層 12 → 24 → 36 vertex の 核 探し.

全論.md Ch18: Kathara hierarchy 12/24/36 が 兄弟.
12 で α⁻¹=137 出るなら、 24 で 何 出る? 36 で 何 出る?

approach:
  (1) 24-vertex Cayley graph (Z/24) + icosahedron 系 から 候補生成
  (2) 各 size で 「物理 invariant 同時満足 graph」 探索
  (3) 24 vertex 核 から 出る integer を 物理定数 と照合
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import sympy as sp
import random
import sys


def build_k1_n(n=12, generators=[1, 4, 6]):
    """Cay(Z/n, generators) の adjacency."""
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for s in generators:
            A[i, (i+s) % n] = 1
            A[i, (i-s) % n] = 1
    return A


def compute_invariants(A):
    A = np.asarray(A, dtype=np.int64)
    n = A.shape[0]
    n_e = int(A.sum() / 2)
    degrees = sorted([int(A[i].sum()) for i in range(n)])
    A2 = A @ A
    A3 = A2 @ A
    A4 = A2 @ A2
    A5 = A4 @ A
    A6 = A4 @ A2
    try:
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
    except Exception:
        return None
    return {
        "n_V": n, "n_E": n_e,
        "max_deg": max(degrees),
        "Tr_A2": int(np.trace(A2)),
        "Tr_A3": int(np.trace(A3)),
        "Tr_A4": int(np.trace(A4)),
        "Tr_A5": int(np.trace(A5)),
        "Tr_A6": int(np.trace(A6)),
        "lam_min": round(evs[0], 4),
        "lam_max": round(evs[-1], 4),
        "alpha_inv_pred": int(np.trace(A4)) // 2 + 2,
        "triangle_free": int(np.trace(A3)) == 0,
    }


def main():
    print("=" * 80)
    print("第376期: Kathara 階層 12→24→36 vertex 探索")
    print("=" * 80)
    sys.stdout.flush()

    # ============================================================
    # 既知 12V K¹ (control)
    # ============================================================
    print(f"\n--- 12 vertex K¹ (Cay(Z/12, {{1,4,6}})) — control ---")
    A_k1_12 = build_k1_n(12, [1, 4, 6])
    inv12 = compute_invariants(A_k1_12)
    print(f"  K¹_12: Tr A^4 = {inv12['Tr_A4']}, α⁻¹_pred = {inv12['alpha_inv_pred']}")
    sys.stdout.flush()

    # ============================================================
    # 24-vertex Cayley graph family
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(A) 24-vertex Cayley graph 探索 (Z/24, 多種 generators)")
    print(f"{'='*80}")
    sys.stdout.flush()

    # All 3-element subsets of {1,...,11} (since 12=-12 in Z/24)
    from itertools import combinations
    n24 = 24
    candidates = []
    for gens in combinations(range(1, 12), 3):
        gens_list = list(gens)
        A = build_k1_n(24, gens_list)
        inv = compute_invariants(A)
        if inv:
            inv["gens"] = gens_list
            candidates.append(inv)

    # Look for "physical" Tr A^4 values
    print(f"\n  Total Cay(Z/24) generators tried: {len(candidates)}")
    # Find any with α⁻¹_pred = 137
    alpha_137_candidates = [c for c in candidates if c["alpha_inv_pred"] == 137]
    print(f"  α⁻¹ = 137 matches at 24V: {len(alpha_137_candidates)}")

    # Find ones with interesting physical-looking integers
    print(f"\n  Interesting 24V Cayley invariants:")
    print(f"  {'gens':>20s} {'TrA²':>8s} {'TrA³':>8s} {'TrA⁴':>10s} {'TrA⁶':>10s} {'α⁻¹_pr':>8s} {'Δ-free':>8s}")
    for c in sorted(candidates, key=lambda c: c["Tr_A4"])[:20]:
        print(f"  {str(c['gens']):>20s} {c['Tr_A2']:>8d} {c['Tr_A3']:>8d} {c['Tr_A4']:>10d} {c['Tr_A6']:>10d} {c['alpha_inv_pred']:>8d} {str(c['triangle_free']):>8s}")
    sys.stdout.flush()

    # Find triangle-free ones with notable values
    tri_free_24 = [c for c in candidates if c["triangle_free"]]
    print(f"\n  Triangle-free Cay(Z/24): {len(tri_free_24)}")
    for c in tri_free_24[:10]:
        print(f"    {c['gens']}: Tr A⁴ = {c['Tr_A4']}, α⁻¹_pred = {c['alpha_inv_pred']}")

    # Look for physical numbers
    physical_targets = {
        1836: "m_p/m_e (proton/electron mass ratio)",
        137: "α⁻¹ (fine structure)",
        125: "Higgs mass GeV (5³)",
        240: "E_8 root",
        91: "Z mass GeV",
        80: "W mass GeV",
        173: "top mass GeV",
        65: "δ_CP_quark degrees",
    }
    print(f"\n  Hunt for physical numbers in Cay(Z/24) Tr A^k:")
    for c in candidates:
        all_traces = [c["Tr_A2"], c["Tr_A4"], c["Tr_A6"]]
        for trace_val in all_traces:
            for target, name in physical_targets.items():
                # exact or 2x match
                if trace_val == target or trace_val == 2*target or trace_val // 2 + 2 == target:
                    print(f"    gens {c['gens']}: trace {trace_val} matches {name} ({target})")

    # ============================================================
    # 24V icosahedron-like graph 候補
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(B) 24-vertex Snub Cube / 24-cell vertices などの 候補")
    print(f"{'='*80}")
    # 24-cell: F_4 Coxeter, 24 vertices, 96 edges, 6-regular
    # Snub cube: 24 vertices, 60 edges
    # K_{4,4,4,4,4,4}: complete 6-partite
    # K_{6,6,6,6}: complete 4-partite

    try:
        # 24-cell adjacency: each vertex connected to 8 others (8-regular)
        # Use known LCF code or explicit construction
        # 24-cell as Cay(W(F_4)/H) — complex
        pass
    except Exception:
        pass

    # K_{4,4,4,4,4,4} — 6-partite, 6 groups of 4
    G_multi = nx.complete_multipartite_graph(*[4]*6)
    A_multi = nx.to_numpy_array(G_multi).astype(np.int64)
    inv_multi = compute_invariants(A_multi)
    print(f"\n  K_{{4,4,4,4,4,4}} (24V, 6-partite):")
    print(f"    Tr A⁴ = {inv_multi['Tr_A4']}, α⁻¹_pred = {inv_multi['alpha_inv_pred']}")

    # K_{6,6,6,6}
    G_4part = nx.complete_multipartite_graph(*[6]*4)
    A_4part = nx.to_numpy_array(G_4part).astype(np.int64)
    inv_4 = compute_invariants(A_4part)
    print(f"\n  K_{{6,6,6,6}} (24V, 4-partite):")
    print(f"    Tr A⁴ = {inv_4['Tr_A4']}, α⁻¹_pred = {inv_4['alpha_inv_pred']}")

    # ============================================================
    # 36-vertex search (limited)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(C) 36-vertex limited candidates")
    print(f"{'='*80}")
    # 36 = 6², dim SO(9) = 36 = |Aut(extended Mathieu)|? no
    # Cay(Z/36) with various generators
    cay36_candidates = []
    for gens in [[1, 5, 9], [1, 7, 11], [1, 4, 6], [1, 8, 17], [1, 6, 10]]:
        A = build_k1_n(36, gens)
        inv = compute_invariants(A)
        if inv:
            inv["gens"] = gens
            cay36_candidates.append(inv)
    print(f"\n  Cay(Z/36) sample candidates:")
    print(f"  {'gens':>15s} {'TrA²':>8s} {'TrA⁴':>10s} {'α⁻¹_pr':>10s} {'Δ-free':>8s}")
    for c in cay36_candidates:
        print(f"  {str(c['gens']):>15s} {c['Tr_A2']:>8d} {c['Tr_A4']:>10d} {c['alpha_inv_pred']:>10d} {str(c['triangle_free']):>8s}")

    # ============================================================
    # 統合
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 統合 — Kathara 階層 探索 結果")
    print(f"{'='*80}")
    print(f"""
  12V: K¹ ∩ Ico = 核、 α⁻¹ = 137 (既知)

  24V Cay(Z/24) {len(candidates)} 種類 試行:
    triangle-free: {len(tri_free_24)}
    α⁻¹_pred = 137 完全一致: {len(alpha_137_candidates)}
    その他 物理 number 一致: 上記 list

  36V 限定 sample:
    上記 各 candidate

  → Kathara 階層 12/24/36 で 各 size に "core-like" graph
    candidate が 存在するか は initial sweep で 部分的 evidence

  honest:
    12V の 核 のような decisive uniqueness を 24V / 36V で 確認するには
    数百万 graph の enumeration 必要 (まだ 実施せず)
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "12V_K1": {"Tr_A4": inv12['Tr_A4'], "alpha_inv": inv12['alpha_inv_pred']},
        "24V_Cay_total": len(candidates),
        "24V_triangle_free": len(tri_free_24),
        "24V_alpha_137_match": len(alpha_137_candidates),
        "24V_multipartite": {
            "K_{4,4,4,4,4,4}": {"Tr_A4": inv_multi['Tr_A4'], "alpha_inv": inv_multi['alpha_inv_pred']},
            "K_{6,6,6,6}": {"Tr_A4": inv_4['Tr_A4'], "alpha_inv": inv_4['alpha_inv_pred']},
        },
        "36V_samples": [{"gens": c["gens"], "alpha_inv": c["alpha_inv_pred"]} for c in cay36_candidates],
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round376_kathara_hierarchy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
