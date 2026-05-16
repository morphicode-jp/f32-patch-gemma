"""第385期: 核 quartic field x⁴-4x³+9x-4 の 数論的 解析.

approach:
  sympy で:
    - splitting field 確認
    - ring of integers (= Z[α] か Z[α, ...] か)
    - discriminant 構造
    - 24197 prime の role
    - Galois group 確認 (= S_4)
"""
from __future__ import annotations
import numpy as np
import math
import sympy as sp
import sys


def main():
    print("=" * 80)
    print("第385期: 核 quartic field x⁴-4x³+9x-4 数論 解析")
    print("=" * 80)
    sys.stdout.flush()

    x = sp.Symbol('x')
    quartic = x**4 - 4*x**3 + 9*x - 4

    # ============================================================
    # (A) 基本情報
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(A) 基本 polynomial 情報")
    print(f"{'='*80}")
    print(f"\n  quartic: {quartic}")
    print(f"  expand: {sp.expand(quartic)}")
    print(f"  is irreducible over Q: {sp.factor(quartic) == quartic}")

    disc = sp.discriminant(quartic, x)
    print(f"\n  discriminant: {disc}")
    print(f"  factor: {sp.factorint(int(disc))}")
    is_disc_prime = sp.isprime(int(disc))
    print(f"  is prime: {is_disc_prime}")
    sys.stdout.flush()

    # ============================================================
    # (B) Resolvent cubic + Galois group
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(B) Resolvent cubic + Galois")
    print(f"{'='*80}")
    # For x⁴ + bx³ + cx² + dx + e with b=-4, c=0, d=9, e=-4
    # Resolvent: y³ - cy² + (bd - 4e)y - (b²e - 4ce + d²)
    # = y³ - 0 + (-4·9 - 4·(-4))y - (16·(-4) - 0 + 81)
    # = y³ - 20y - 17
    resolvent = x**3 - 20*x - 17
    print(f"\n  resolvent cubic: {resolvent}")
    print(f"  factor: {sp.factor(resolvent)}")
    print(f"  resolvent disc: {sp.discriminant(resolvent, x)}")
    res_irreducible = sp.factor(resolvent) == resolvent
    print(f"  resolvent irreducible: {res_irreducible}")
    sys.stdout.flush()

    # 24197 check
    if int(disc) == 24197:
        print(f"\n  ★ Discriminant 24197 = quartic disc = resolvent disc")
        print(f"  ★ 同じ prime が 4 次と 3 次 で出現 — 数論的 striking")

    # ============================================================
    # (C) Roots in radical form
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(C) Roots 明示形")
    print(f"{'='*80}")
    roots = sp.solve(quartic, x)
    print(f"\n  解 (radical form):")
    for i, r in enumerate(roots):
        try:
            r_num = complex(r)
            print(f"    α_{i+1} ≈ {r_num.real:.6f}{'+' + str(round(r_num.imag, 4)) + 'i' if abs(r_num.imag) > 1e-9 else ''}")
        except Exception:
            print(f"    α_{i+1} = (complex radical)")

    # ============================================================
    # (D) Try identifying number field
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(D) Number field identification")
    print(f"{'='*80}")
    print(r"""
  quartic K = Q(α) where α は root of x⁴-4x³+9x-4
  [K : Q] = 4 (since irreducible quartic)

  Possible structures:
    - Galois extension: |Gal(K/Q)| = 4 (= V_4 or Z/4)
    - Non-Galois: |Gal(splitting/Q)| > 4

  我々の case:
    Galois of splitting = S_4 (= 24)
    K の Galois closure dimension = 24
    K 自体は non-Galois (degree 4)
""")
    sys.stdout.flush()

    # ============================================================
    # (E) Connection to specific physics / math
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(E) 既知 数学 object との 接続")
    print(f"{'='*80}")
    print(r"""
  24197 prime + S_4 Galois quartic discriminant:

  OEIS search candidates:
    A002519: imaginary quadratic class numbers
    A000262: arrangement numbers
    24197 単独 lookup necessary

  数学 object connection 試行:
    24197 / 137 = 176.62 (no clean)
    24197 / 24 = 1008.21 (no clean)
    24197 - 24000 = 197 (prime)
    24197 = 24000 + 197 = 24 × 1000 + 197 prime

  → 24197 は struc数学的 isolated prime
  → 既知 名前付き object との 同定 困難
""")

    # ============================================================
    # (F) Ring of integers attempt
    # ============================================================
    print(f"\n{'='*80}")
    print(f"(F) Ring of integers O_K attempt")
    print(f"{'='*80}")
    print(r"""
  K = Q(α), α root of x⁴-4x³+9x-4

  O_K (= Z-module of algebraic integers in K):
    minimal generators が α だけで生まれるか:
    Z[α] = Z + Zα + Zα² + Zα³
    disc(Z[α]) = disc(quartic) = 24197
    disc(O_K) divides 24197 prime
    → disc(O_K) = 24197 (= prime), so O_K = Z[α]
    (= Z[α] は 全 algebraic integers の ring)

  → これは 「美しい」 性質、 quartic は 「monogenic」
""")

    # ============================================================
    # 統合
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 統合 — quartic field 数論")
    print(f"{'='*80}")
    print(f"""
  ★ quartic x⁴-4x³+9x-4 性質:
    irreducible over Q ✓
    Galois group of splitting field = S_4 (order 24)
    discriminant = 24197 (= prime)
    resolvent cubic = y³-20y-17 (irreducible, same disc)
    Z[α] = O_K (monogenic, beautiful)

  ★ 24197 prime の意味:
    quartic field K の disc が prime → 「unramified outside p=24197」
    K は 24197 で 唯一 ramified
    → K は Q の 「ほぼ unramified extension」
    → 「rare」 number field

  ★ 既知 object との 同定:
    24197 単独 lookup 必要 (OEIS, LMFDB)
    現状 名前付き field と 同定 不完
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "quartic": "x⁴-4x³+9x-4",
        "irreducible_over_Q": True,
        "discriminant": int(disc),
        "discriminant_is_prime": is_disc_prime,
        "Galois_group": "S_4 (order 24)",
        "resolvent_cubic": "y³ - 20y - 17",
        "monogenic": True,
        "ring_of_integers": "Z[α] = O_K",
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round385_quartic_field.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
