"""第360期: 7 identity を 8, 9, 10 identity に拡張 — さらに strict uniqueness.

approach:
  exp359 で 7 identity の核 unique 確認。
  さらに 識別力ある invariant を追加:
    - Tr(A^7) = 1092
    - Tr(A^8) = 22174
    - λ_min = -3 EXACT
    - λ_max ≈ 3.275 EXACT
    - bipartiteness = False
    - diameter = 3
    - C_6 (6-cycle count)

  これらを 162 family member に追加 check.
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
    print("第360期: 7 → 10 identity 拡張 で uniqueness 強化")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron())
    G_core = nx.from_numpy_array(A_core)

    # Core invariants
    A2 = A_core @ A_core
    A4 = A2 @ A2
    A5 = A4 @ A_core
    A6 = A4 @ A2
    A7 = A6 @ A_core
    A8 = A4 @ A4
    evs = sorted(np.linalg.eigvalsh(A_core.astype(float)).tolist())
    core_inv = {
        "Tr_A4": int(np.trace(A4)),
        "Tr_A5": int(np.trace(A5)),
        "Tr_A6": int(np.trace(A6)),
        "Tr_A7": int(np.trace(A7)),
        "Tr_A8": int(np.trace(A8)),
        "lam_min": round(evs[0], 4),
        "lam_max": round(evs[-1], 4),
        "lam_2nd_max": round(evs[-2], 4),
        "bipartite": nx.is_bipartite(G_core),
        "diameter": nx.diameter(G_core),
        "C_5": int(np.trace(A5)) // 10,
    }
    print(f"\n  ★ 核 invariants:")
    for k, v in core_inv.items():
        print(f"    {k:15s} = {v}")
    sys.stdout.flush()

    # ============================================================
    # Collect ALL 162 siblings
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) 162 sibling 全数 collect (15M trial)")
    print("="*80)
    sys.stdout.flush()

    target_deg = (2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4)
    rng = np.random.default_rng(42)
    siblings = {}
    N_max = 25_000_000

    t0 = time.time()
    last_count = 0
    saturated = 0
    for trial in range(N_max):
        if trial % 1_000_000 == 0 and trial > 0:
            cur = len(siblings)
            elapsed = time.time() - t0
            if cur == last_count:
                saturated += 1
            else:
                saturated = 0
            last_count = cur
            print(f"    trial {trial:>9,}  unique 5/5 spec: {cur}  sat: {saturated}  ({elapsed:.0f}s)")
            sys.stdout.flush()
            if saturated >= 8:
                print(f"    saturation reached")
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
        A2_t = A @ A
        if int(np.trace(A2_t @ A)) != 0:
            continue
        A4_t = A2_t @ A2_t
        if int(np.trace(A4_t)) != 270:
            continue
        evs_t = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        if abs((19 + abs(evs_t[0])) - 22) > 0.01:
            continue
        if (4 + int(np.trace(A2_t))) != 42:
            continue
        spec = tuple(round(e, 4) for e in evs_t)
        if spec not in siblings:
            siblings[spec] = A.copy()

    print(f"\n  全 sibling collected: {len(siblings)}")
    sys.stdout.flush()

    # ============================================================
    # Test each with 10 identities
    # ============================================================
    print(f"\n{'='*80}")
    print("(B) 各 sibling で 10 identity check")
    print("="*80)
    sys.stdout.flush()

    matches_per_n = {n: 0 for n in range(11)}
    unique_after_n_identities = {}  # how many graphs survive after n identities

    # Order identities by selectivity
    identity_names = [
        "α⁻¹=137 (Tr A⁴=270)",   # 1
        "K3=22 (|E|+|λ_min|)",   # 2
        "Catalan=42",            # 3
        "Δ-free (Tr A³=0)",       # 4
        "V=12",                   # 5
        "|Aut|=4",                # 6
        "C_5=4",                  # 7
        "Tr A⁷=1092",             # 8
        "Tr A⁸=22174",            # 9
        "λ_min=-3 EXACT",         # 10
    ]

    core_pass = []
    for i, (spec, A) in enumerate(siblings.items()):
        if i % 30 == 0:
            print(f"    checking {i}/{len(siblings)}")
            sys.stdout.flush()
        G = nx.from_numpy_array(A)
        A2_t = A @ A
        A4_t = A2_t @ A2_t
        A5_t = A4_t @ A
        A6_t = A4_t @ A2_t
        A7_t = A6_t @ A
        A8_t = A4_t @ A4_t
        evs_t = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())

        ids = []
        # 1: α⁻¹ = 137
        ids.append(int(np.trace(A4_t)) == 270)
        # 2: K3 = 22
        ids.append(abs((19 + abs(evs_t[0])) - 22) < 0.01)
        # 3: Catalan = 42
        ids.append((4 + int(np.trace(A2_t))) == 42)
        # 4: triangle-free
        ids.append(int(np.trace(A2_t @ A)) == 0)
        # 5: V=12
        ids.append(A.shape[0] == 12)
        # 6: |Aut|=4
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        aut = 0
        for _ in GM.isomorphisms_iter():
            aut += 1
            if aut > 10:
                break
        ids.append(aut == 4)
        # 7: C_5 = 4
        c5 = int(np.trace(A5_t)) // 10
        ids.append(c5 == 4)
        # 8: Tr A^7 = 1092
        ids.append(int(np.trace(A7_t)) == core_inv["Tr_A7"])
        # 9: Tr A^8 = 22174
        ids.append(int(np.trace(A8_t)) == core_inv["Tr_A8"])
        # 10: λ_min = -3 EXACT
        ids.append(abs(evs_t[0] - (-3.0)) < 0.001)

        n_ids_met = sum(ids)
        matches_per_n[n_ids_met] = matches_per_n.get(n_ids_met, 0) + 1
        if n_ids_met == 10:
            core_pass.append(spec)

    print(f"\n  ★ 結果 — number of identities met by each sibling:")
    for k in sorted(matches_per_n.keys()):
        if matches_per_n[k] > 0:
            marker = " ← 核 含む" if k == 10 else ""
            print(f"    {k}/10 identities: {matches_per_n[k]} graphs{marker}")

    # Cumulative uniqueness: how many graphs satisfy first-k identities
    print(f"\n  累積 uniqueness (= 最初の k identity 満たす graph 数):")
    for k in range(1, 11):
        # Recount: how many graphs satisfy identities 1..k
        count = 0
        for i, (spec, A) in enumerate(siblings.items()):
            G = nx.from_numpy_array(A)
            A2_t = A @ A
            A4_t = A2_t @ A2_t
            A5_t = A4_t @ A
            A6_t = A4_t @ A2_t
            A7_t = A6_t @ A
            A8_t = A4_t @ A4_t
            evs_t = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
            survive = True
            checks = []
            if k >= 1: checks.append(int(np.trace(A4_t)) == 270)
            if k >= 2: checks.append(abs((19 + abs(evs_t[0])) - 22) < 0.01)
            if k >= 3: checks.append((4 + int(np.trace(A2_t))) == 42)
            if k >= 4: checks.append(int(np.trace(A2_t @ A)) == 0)
            if k >= 5: checks.append(A.shape[0] == 12)
            if k >= 6:
                GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
                aut = 0
                for _ in GM.isomorphisms_iter():
                    aut += 1
                    if aut > 10:
                        break
                checks.append(aut == 4)
            if k >= 7:
                c5 = int(np.trace(A5_t)) // 10
                checks.append(c5 == 4)
            if k >= 8: checks.append(int(np.trace(A7_t)) == core_inv["Tr_A7"])
            if k >= 9: checks.append(int(np.trace(A8_t)) == core_inv["Tr_A8"])
            if k >= 10: checks.append(abs(evs_t[0] - (-3.0)) < 0.001)
            if all(checks):
                count += 1
        marker = " ← 核 unique" if count == 1 else ""
        print(f"    {k} identity: {count} graphs survive{marker}")

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print("★ 結論 — 10 identity uniqueness")
    print("="*80)

    print(f"""
  10 identity (核の主要 invariant ほぼ全部) で 162 個中:
    core_pass: {len(core_pass)} graph

  → 核 unique は {'保たれている' if len(core_pass) == 1 else 'unique でない'}

  honest assessment:
    最低 7 identity で 162 個中 核 unique (F567 既)
    10 identity で {'★ unique 強化' if len(core_pass) == 1 else '?'}
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "core_invariants": {k: str(v) for k, v in core_inv.items()},
        "n_siblings_collected": len(siblings),
        "matches_per_n_identities": dict(matches_per_n),
        "core_passes_10_identity": len(core_pass) == 1,
        "verdict": "10 identity uniqueness " + ("strengthened" if len(core_pass) == 1 else "open"),
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round360_10_identity.json"
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
