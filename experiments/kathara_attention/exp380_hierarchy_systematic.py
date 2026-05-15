"""第380期: Kathara 階層 系統探索 — 12-60 vertex の Cayley graph で 物理定数 検出.

F587 で 36V Cay(Z/36,{1,7,11}) が 1838 ≈ m_p/m_e を encode 発見.
他 size でも 同様の encoding が あるか 系統探索:

各 N = 12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 60:
  Cay(Z/N, all 3-element generator subsets) を generate
  Tr(A^4)/2 + 2 = X を 計算
  X が 既知物理定数 と 一致するか check
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import sys
from itertools import combinations


def build_cay(n, generators):
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for s in generators:
            A[i, (i+s) % n] = 1
            A[i, (i-s) % n] = 1
    return A


def Tr_A4(A):
    A2 = A @ A
    A4 = A2 @ A2
    return int(np.trace(A4))


# 物理定数 と target value
PHYSICAL = {
    137: "α⁻¹ (微細構造定数)",
    1836: "m_p/m_e (陽子/電子)",
    206: "m_μ/m_e (μ/電子) [≈ 206.77]",
    3477: "m_τ/m_e (τ/電子)",
    125: "m_H GeV (Higgs 質量)",
    91: "m_Z GeV (Z 質量)",
    80: "m_W GeV (W 質量)",
    173: "m_top GeV",
    240: "E_8 root",
    248: "E_8 adjoint",
    78: "E_6 adjoint",
    133: "E_7 adjoint",
    26: "bosonic D",
    24: "Niemeier / K3 χ",
    42: "Catalan C_5",
    22: "K3 lattice rank",
    19: "|E_core|",
    65: "δ_CP_quark deg",
    195: "δ_CP_lep deg (abs)",
    1377: "α⁻¹ × 10 (?)",
    1057: "Lamb shift (?)",
    1027: "?",
    1839: "near m_p/m_e",
    1838: "near m_p/m_e",
    1837: "near m_p/m_e",
}


def main():
    print("=" * 80)
    print("第380期: Kathara 階層 系統探索 — 各 size で 物理定数 encode 検証")
    print("=" * 80)
    sys.stdout.flush()

    sizes = [12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 60]
    print(f"\n  探索 sizes: {sizes}")
    print(f"  各 size: 3-element Cayley generator subset 全探索")
    print(f"  物理 hit window: ±2 (= +2 offset pattern と整合)")
    sys.stdout.flush()

    all_hits = []
    for n in sizes:
        # 3-element subsets of {1, ..., (n-1)//2}
        max_gen = n // 2
        gens_pool = list(range(1, max_gen + 1))
        # If n even, n/2 is self-inverse (single edge per vertex), counts once
        # Take all 3-element subsets
        n_subsets = math.comb(len(gens_pool), 3)
        if n_subsets > 5000:
            # Limit to 5000 for compute time
            from itertools import islice
            subsets = list(islice(combinations(gens_pool, 3), 5000))
        else:
            subsets = list(combinations(gens_pool, 3))
        print(f"\n  --- N = {n}, trying {len(subsets)} generator sets ---")
        sys.stdout.flush()

        size_hits = []
        all_alpha_pred = []
        for gens in subsets:
            A = build_cay(n, list(gens))
            t4 = Tr_A4(A)
            alpha_pred = t4 // 2 + 2
            all_alpha_pred.append((alpha_pred, gens))
            # check hit
            for target, name in PHYSICAL.items():
                if abs(alpha_pred - target) <= 2:
                    size_hits.append((gens, alpha_pred, target, name))

        # Filter to most striking hits
        unique_hits = {}
        for gens, ap, t, n_name in size_hits:
            key = (t, n_name)
            if key not in unique_hits:
                unique_hits[key] = (gens, ap)
        print(f"    hits within ±2 of physical:")
        for (t, name), (gens, ap) in sorted(unique_hits.items(), key=lambda x: abs(x[1][1] - x[0][0])):
            diff = ap - t
            print(f"      {name:35s} target={t:>5d}  pred={ap:>5d} (diff {diff:+3d})  gens={list(gens)}")
            all_hits.append({"size": n, "gens": list(gens), "alpha_pred": ap, "target": t, "name": name, "diff": diff})
        sys.stdout.flush()

    # ============================================================
    # まとめ
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 統合 — Kathara 階層 物理定数 encode map")
    print(f"{'='*80}")
    print(f"\n  全 size 全 hit (±2 以内):")
    # Sort by size, then by target
    for hit in sorted(all_hits, key=lambda h: (h["size"], h["target"])):
        print(f"    N={hit['size']:>3d}, gens={hit['gens']}, pred={hit['alpha_pred']:>5d} ≈ {hit['target']} ({hit['name']})")

    # Most striking
    print(f"\n  ★ size ごと の 主要 encoding hint:")
    size_summary = {}
    for hit in all_hits:
        size = hit["size"]
        if size not in size_summary:
            size_summary[size] = []
        size_summary[size].append(hit)
    for size in sorted(size_summary.keys()):
        targets = set(h["target"] for h in size_summary[size])
        print(f"    N={size:>3d}: {len(targets)} unique physical target hits — {list(targets)[:5]}")

    print(f"""
  honest 解釈:
    各 size で 「+2 offset で 物理定数」 hit が ある か 確認.
    多数 hit があれば 階層 hypothesis 強化.
    少なければ 12V は 偶然強、 階層化 弱い.
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "sizes_tested": sizes,
        "n_total_hits": len(all_hits),
        "hits": all_hits[:200],  # cap
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round380_hierarchy_systematic.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
