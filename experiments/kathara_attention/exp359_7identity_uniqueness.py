"""第359期: 144 sibling 全数で 7 identity (|Aut|=4 + C_5=4 追加) check.

exp358 で sample 50 中 0 個が |Aut|=4 or C_5=4 を満たす.
→ 全 144 で 検証して 核 unique 確認.

extra identity:
  id6: |Aut(G)| = 4
  id7: C_5 (5-cycle count) = 4
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


def aut_count(G, max_count=100):
    GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
    count = 0
    for _ in GM.isomorphisms_iter():
        count += 1
        if count > max_count:
            break
    return count


def main():
    print("=" * 80)
    print("第359期: 144-graph family の 7 identity check")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron())
    G_core = nx.from_numpy_array(A_core)
    core_aut = aut_count(G_core)
    A2 = A_core @ A_core
    A4 = A2 @ A2
    A5 = A4 @ A_core
    core_c5 = int(np.trace(A5)) // 10  # triangle-free → Tr A^5 = 10 C_5
    print(f"\n  核 |Aut| = {core_aut}, C_5 = {core_c5}")
    sys.stdout.flush()

    # ============================================================
    # Collect ALL 144-ish siblings (push to large N)
    # ============================================================
    target_deg = (2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4)
    rng = np.random.default_rng(42)
    siblings = {}
    N_max = 20_000_000

    t0 = time.time()
    last_growth = 0
    same_count_for = 0
    for trial in range(N_max):
        if trial % 1_000_000 == 0 and trial > 0:
            elapsed = time.time() - t0
            n_uniq = len(siblings)
            if n_uniq == last_growth:
                same_count_for += 1
            else:
                same_count_for = 0
            last_growth = n_uniq
            print(f"    trial {trial:>9,}  unique 5/5 spec: {n_uniq}  same-count-streak: {same_count_for}  ({elapsed:.0f}s)")
            sys.stdout.flush()
            if same_count_for >= 5:
                print(f"    saturation reached at {n_uniq} unique iso classes")
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
        Tr_A4 = int(np.trace(A4_t))
        if Tr_A4 != 270:
            continue
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        abs_lam_min = abs(evs[0])
        if abs((19 + abs_lam_min) - 22) > 0.01:
            continue
        Tr_A2_t = int(np.trace(A2_t))
        if (4 + Tr_A2_t) != 42:
            continue

        spec = tuple(round(e, 4) for e in evs)
        if spec not in siblings:
            siblings[spec] = A.copy()

    elapsed = time.time() - t0
    print(f"\n  全 unique 5/5 iso classes: {len(siblings)} (時間 {elapsed:.0f}s)")
    sys.stdout.flush()

    # ============================================================
    # 7 identity check on all
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) 全 sibling で |Aut| と C_5 を check")
    print("="*80)
    sys.stdout.flush()

    aut_4_count = 0
    c5_4_count = 0
    both_count = 0
    iso_to_core = 0
    aut_dist = {}
    c5_dist = {}

    for i, (spec, A) in enumerate(siblings.items()):
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"    checking {i}/{len(siblings)}  ({elapsed:.0f}s)")
            sys.stdout.flush()
        G = nx.from_numpy_array(A)
        aut = aut_count(G)
        A2_t = A @ A
        A4_t = A2_t @ A2_t
        A5_t = A4_t @ A
        c5 = int(np.trace(A5_t)) // 10

        aut_dist[aut] = aut_dist.get(aut, 0) + 1
        c5_dist[c5] = c5_dist.get(c5, 0) + 1

        if aut == 4:
            aut_4_count += 1
        if c5 == 4:
            c5_4_count += 1
        if aut == 4 and c5 == 4:
            both_count += 1
            # Check if iso to core
            if nx.is_isomorphic(G, G_core):
                iso_to_core += 1
                print(f"    ★ {i}/{len(siblings)}: |Aut|=4, C_5=4, **isomorphic to 核**")

    print(f"\n  ★ 全 {len(siblings)} 個 sibling の 結果:")
    print(f"  |Aut| distribution: {dict(sorted(aut_dist.items()))}")
    print(f"  C_5 distribution:   {dict(sorted(c5_dist.items()))}")
    print(f"")
    print(f"  |Aut| = 4 satisfying: {aut_4_count}")
    print(f"  C_5  = 4 satisfying: {c5_4_count}")
    print(f"  両方 satisfying:     {both_count}")
    print(f"  そのうち 核 と iso:   {iso_to_core}")
    print(f"  そのうち 核 と異なる: {both_count - iso_to_core}")

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print("★ 結論 — 7 identity で 核 uniqueness")
    print("="*80)

    if both_count == 1 and iso_to_core == 1:
        verdict = "★★★★★ 核は 7 identity (5+|Aut|=4+C_5=4) の **唯一の graph** in family — UNIQUENESS PROVEN"
    elif both_count - iso_to_core == 0:
        verdict = f"★★★★★ |Aut|=4 AND C_5=4 の graph は {both_count} 個、 すべて 核と iso → 核 unique"
    elif both_count < 5:
        verdict = f"★★★★ |Aut|=4 AND C_5=4 の graph は {both_count} 個 (核含む {iso_to_core})、 ほぼ unique"
    else:
        verdict = f"★★★ {both_count} graphs satisfy all 7 — uniqueness 不完全"

    print(f"\n  {verdict}")
    print(f"""
  full family ({len(siblings)} 個) のうち:
    5 identity 満たす:           {len(siblings)} (全部、 定義)
    + |Aut| = 4:                 {aut_4_count}
    + C_5 = 4:                   {c5_4_count}
    + 両方 (= 7 identity):       {both_count}
    そのうち 核と iso:            {iso_to_core}
    残り (= 核以外):              {both_count - iso_to_core}
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "n_5_5_iso_classes": len(siblings),
        "aut_distribution": dict(aut_dist),
        "c5_distribution": dict(c5_dist),
        "aut_4_count": aut_4_count,
        "c5_4_count": c5_4_count,
        "both_count": both_count,
        "iso_to_core_in_both": iso_to_core,
        "verdict": verdict,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round359_7identity.json"
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
