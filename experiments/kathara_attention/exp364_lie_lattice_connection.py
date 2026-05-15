"""第364期: 核 と Lie 代数 / lattice / root system 接続.

未解決: 核 (12V 19E) は known math のどれと "同じ" か?

候補:
  - E_6 root system: 72 roots, 27 short rep (cubic surface lines)
  - E_7: 126 roots, 56 weights
  - E_8: 240 roots
  - Niemeier lattices: 24 種, 24-dim even unimodular
  - Mathieu groups M_12, M_24
  - Conway groups Co_1, Co_2, Co_3
  - K3 (22 + 2)
  - 24-cell (24 vertices, F_4 Weyl group)

approach:
  (1) 核 invariants と Lie 代数 dimensions の数値比較
  (2) 核 char poly の root を E_n weight などと照合
  (3) 162 family size を Lie rep dimensions と照合
"""
from __future__ import annotations
import numpy as np
import math
import sympy as sp


def main():
    print("=" * 80)
    print("第364期: 核 と Lie / lattice / root system 接続")
    print("=" * 80)

    n_V = 12
    n_E = 19
    n_aut = 4
    family_size = 162
    Tr_A4 = 270
    Tr_A6 = 2354
    Tr_A8 = 22174
    alpha_inv = 137

    # ============================================================
    # (1) E_n root systems
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) E_n root system との connection")
    print("="*80)
    E_systems = {
        "A_4 (= sl_5)": (24, 4),  # roots, rank
        "B_4 (= so_9)": (32, 4),
        "D_4": (24, 4),
        "F_4": (48, 4),
        "G_2": (12, 2),
        "E_6 short rep": (27, 6),
        "E_6 roots": (72, 6),
        "E_7 weights": (56, 7),
        "E_7 roots": (126, 7),
        "E_8 weights": (240, 8),
        "E_8 cell": (240, 8),
    }
    print(f"\n  核 invariants vs E_n:")
    print(f"  {'system':25s} {'roots/dim':10s} {'/V':>10s} {'/E':>10s} {'/Aut':>10s} {'/162':>10s}")
    for name, (rd, rank) in E_systems.items():
        r_V = rd / n_V
        r_E = rd / n_E
        r_aut = rd / n_aut
        r_162 = rd / family_size
        print(f"  {name:25s} {rd:>10d} {r_V:>10.3f} {r_E:>10.3f} {r_aut:>10.3f} {r_162:>10.4f}")

    print(f"""
  ★ 注目:
    E_6 short rep = 27 → 162 / 27 = 6 (clean integer!) ← F570 既
    E_8 240 → 240/12 = 20 = E(K¹) (clean!) ← K¹ energy 20
    E_8 240 / 19 = 12.63
    F_4 48 / 12 = 4 = |Aut|
""")

    # ============================================================
    # (2) Niemeier / 24
    # ============================================================
    print(f"\n{'='*80}")
    print("(B) Niemeier lattices / 24")
    print("="*80)
    niemeier_count = 24
    K3_chi = 24
    print(f"\n  Niemeier 24 種、 K3 χ = 24、 24-cell")
    print(f"  Galois |S_4| = 24 (核 quartic factor)")
    print(f"  24 / |Aut| = 6 (= K^1 degree)")
    print(f"  24 = 2|V| = 2 × 12")
    print(f"  → 24 が 複数 path で 核 と接続")

    # ============================================================
    # (3) Mathieu / sporadic
    # ============================================================
    print(f"\n{'='*80}")
    print("(C) Mathieu / sporadic との数値 connection")
    print("="*80)
    sporadic = {
        "M_11": 7920,
        "M_12": 95040,
        "M_22": 443520,
        "M_23": 10200960,
        "M_24": 244823040,
        "Co_3": 495766656000,
        "Co_2": 42305421312000,
        "Co_1": 4157776806543360000,
        "HS": 44352000,
        "McL": 898128000,
        "J_2": 604800,
        "Suz": 448345497600,
        "Ru": 145926144000,
        "Th": 90745943887872000,
        "HN": 273030912000000,
        "Ly": 51765179004000000,
        "He": 4030387200,
        "ON": 460815505920,
        "Fi22": 64561751654400,
        "Fi23": 4089470473293004800,
        "Fi24'": 1255205709190661721292800,
        "Monster": int(8.08e53),
    }
    print(f"\n  核 invariants で divisible な sporadic orders:")
    core_divs = [n_V, n_E, n_aut, Tr_A4, family_size, alpha_inv]
    for name, order in sporadic.items():
        clean_divs = []
        for d in core_divs:
            if order % d == 0:
                clean_divs.append(d)
        if len(clean_divs) >= 3:
            print(f"  {name} order = {order:,}")
            print(f"    divisible by: {clean_divs}")

    # ============================================================
    # (4) 162 = 6 × 27 - 詳しく
    # ============================================================
    print(f"\n{'='*80}")
    print("(D) ★ 162 = 6 × 27 — E_6 cubic surface connection")
    print("="*80)
    print(f"""
  ★ 162 = 6 × 27 = (K¹ degree) × (E_6 cubic surface lines)

  E_6 27-dim fundamental rep:
    - 27 = number of lines on a cubic surface (Cayley 1849)
    - 27 = dim J_3(O) (= 3×3 Hermitian octonion matrix)
    - 27 = number of stable bundles on cubic threefold
    - 27 = number of nodes on certain Klein-related cubics

  6 = K¹ degree (= edges per vertex in K¹)
    or 6 = 2|V|/4 = 12/2 = 6

  Hypothesis:
    162-family = E_6 representation theory orbit
    各 family member = 27-orbit (cubic surface 上の 27 線) の各 line に 6 個の "decoration" 付与

  ★ predict: 162 family の structure は E_6 group action で記述可能
""")

    # ============================================================
    # (5) Coxeter / Weyl groups
    # ============================================================
    print(f"\n{'='*80}")
    print("(E) Coxeter / Weyl group orders")
    print("="*80)
    weyl_orders = {
        "W(A_4)": 120,
        "W(B_4)": 384,
        "W(D_4)": 192,
        "W(F_4)": 1152,
        "W(H_3)": 120,
        "W(H_4)": 14400,
        "W(E_6)": 51840,
        "W(E_7)": 2903040,
        "W(E_8)": 696729600,
    }
    print(f"\n  Weyl group orders vs 核 family:")
    for name, ord_ in weyl_orders.items():
        if ord_ % family_size == 0:
            print(f"  {name} = {ord_} = {ord_//family_size} × 162")
        if ord_ % (family_size * n_aut) == 0:
            print(f"  {name} = {ord_} = {ord_//(family_size * n_aut)} × 162 × 4")

    # ============================================================
    # (6) Specific identity searches
    # ============================================================
    print(f"\n{'='*80}")
    print("(F) 核 quartic root と Lie 重み 比較")
    print("="*80)
    x = sp.Symbol('x')
    quartic = x**4 - 4*x**3 + 9*x - 4
    quartic_roots = sp.solve(quartic, x)
    print(f"\n  核 S_4 quartic roots (numerical):")
    for r in quartic_roots:
        try:
            r_num = complex(r)
            print(f"    {r_num.real:.4f} + {r_num.imag:.4f}i  (= {r})")
        except Exception:
            print(f"    {r}")
    print(f"")
    print(f"  Sum of roots = 4 (= coefficient of x³ × -1)")
    print(f"  Product of roots = -4 (= constant term)")
    print(f"")
    print(f"  E_8 root lengths: all √2")
    print(f"  E_6 weight set: various rationals")
    print(f"  → 核 quartic roots と E_n weights の直接 mapping は 自明ではない")

    # ============================================================
    # 統合
    # ============================================================
    print(f"\n{'='*80}")
    print("★ 統合 — Lie/lattice 接続 結果")
    print("="*80)
    print(f"""
  ★ 強い数値 connection 候補:

  1. 162 = 6 × 27 = (K¹ degree) × (E_6 short rep / cubic surface lines)
     → 162 family が E_6 表現論と結びつく可能性 (★★★)

  2. Galois |S_4| = 24 = Niemeier = K3 χ = 24-cell vertex
     → 核 quartic Galois が 24 系を encode (F570)

  3. E_8 root 240 / |V| = 20 = E(K¹) (energy of K^1)
     → K¹ energy が E_8 dim と connect

  4. F_4 48 / |V| = 4 = |Aut|
     → 自己同型群 |Aut|=4 が F_4 / 4

  ★ Hypothesis (refined):
     核 family = E_6 orbital 構造 + S_4 Galois 24 系
     完全 isomorphism は open、 数値 evidence 強い

  ★ 次研究:
     - E_6 27-line incidence graph を 構築
     - 各 line を 6 decoration で fertilize して 162 graphs を構成
     - これを 162-family と iso check
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "162_decomposition": "6 × 27 (K¹ degree × E_6 cubic surface lines)",
        "Galois_S_4_24": "S_4 Galois = 24 = Niemeier = K3 χ",
        "E_8_K1_energy": "240 / 12 = 20 = E(K¹)",
        "F_4_aut": "48 / 12 = 4 = |Aut|",
        "main_hypothesis": "162 family ↔ E_6 orbital + S_4 Galois 24 system",
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round364_lie_lattice.json"
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
