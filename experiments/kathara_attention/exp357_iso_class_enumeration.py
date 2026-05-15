"""第357期: iso class enumeration で 核 uniqueness を 数学的に詰める.

approach:
  (1) 大量 random graph 生成 (10M)
  (2) deg seq + triangle-free filter
  (3) spectrum で iso class dedupe
  (4) 各 unique iso class で 5 identity check
  (5) 5/5 一致する iso class 数 を測定

これで「核は 5 identity 同時満足する唯一の iso class」 か empirical 判定.
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
    print("第357期: iso class enumeration — 核 uniqueness 数学的詰め")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    n = 12
    target_deg = tuple(sorted([int(A_core[i].sum()) for i in range(n)]))
    print(f"\n  target degree seq: {target_deg}")
    sys.stdout.flush()

    # 核 invariants
    def compute_invs(A):
        A = np.asarray(A, dtype=np.int64)
        A2 = A @ A
        A3 = A2 @ A
        A4 = A2 @ A2
        A6 = A4 @ A2
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        spec = tuple(round(e, 6) for e in evs)
        n_e = int(A.sum() / 2)
        max_deg = max(int(A[i].sum()) for i in range(A.shape[0]))
        Tr_A2 = int(np.trace(A2))
        Tr_A3 = int(np.trace(A3))
        Tr_A4 = int(np.trace(A4))
        Tr_A6 = int(np.trace(A6))
        abs_lam_min = abs(evs[0])
        # 5 identity check
        id_alpha = (Tr_A4 // 2 + 2) == 137 and Tr_A4 == 270
        id_K3 = abs((n_e + abs_lam_min) - 22) < 0.001
        id_cat = (max_deg + Tr_A2) == 42
        id_tri_free = Tr_A3 == 0
        id_V12 = A.shape[0] == 12
        return spec, (id_alpha, id_K3, id_cat, id_tri_free, id_V12)

    core_spec, core_ids = compute_invs(A_core)
    print(f"\n  核 spectrum: {core_spec[:6]}...")
    print(f"  核 5 identity: {core_ids}, sum = {sum(core_ids)}/5")
    sys.stdout.flush()

    # ============================================================
    # Random sampling with iso class collection
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) random graph 生成 (10M trial)")
    print("="*80)
    sys.stdout.flush()

    rng = np.random.default_rng(42)
    N_TRIALS = 10_000_000

    unique_specs = {}  # spec -> (A, ids, count)
    iso_5_5_count = 0
    iso_with_core_spec = 0
    triangle_free_count = 0

    t0 = time.time()

    for trial in range(N_TRIALS):
        if trial % 500_000 == 0 and trial > 0:
            elapsed = time.time() - t0
            n_uniq = len(unique_specs)
            n_5_5 = sum(1 for _, ids, _ in unique_specs.values() if sum(ids) == 5)
            print(f"    trial {trial:>10,}  Δ-free {triangle_free_count:>8,}  uniq spec {n_uniq:>6,}  5/5 iso class {n_5_5}  iso-core spec {iso_with_core_spec}  ({elapsed:.0f}s)")
            sys.stdout.flush()

        # Configuration model
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

        # Quick triangle check (avoid expensive linalg for non-triangle-free)
        A2_small = A @ A
        A3_small = A2_small @ A
        if int(np.trace(A3_small)) != 0:
            continue
        triangle_free_count += 1

        # Compute full invariants only for triangle-free
        A4 = A2_small @ A2_small
        A6 = A4 @ A2_small
        n_e = 19
        Tr_A2 = int(np.trace(A2_small))
        Tr_A3 = 0
        Tr_A4 = int(np.trace(A4))
        Tr_A6 = int(np.trace(A6))
        max_deg = 4
        try:
            evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        except Exception:
            continue
        spec = tuple(round(e, 4) for e in evs)
        abs_lam_min = abs(evs[0])
        # 5 identity check
        id_alpha = Tr_A4 == 270
        id_K3 = abs((n_e + abs_lam_min) - 22) < 0.01
        id_cat = (max_deg + Tr_A2) == 42
        id_tri_free = True  # filtered
        id_V12 = True
        ids = (id_alpha, id_K3, id_cat, id_tri_free, id_V12)

        if spec == core_spec:
            iso_with_core_spec += 1

        if spec not in unique_specs:
            unique_specs[spec] = (A.copy(), ids, 1)
            if sum(ids) == 5:
                iso_5_5_count += 1
        else:
            old = unique_specs[spec]
            unique_specs[spec] = (old[0], old[1], old[2] + 1)

    elapsed = time.time() - t0
    print(f"\n  完了 ({elapsed:.0f}s)")
    print(f"  triangle-free graphs found: {triangle_free_count:,}")
    print(f"  unique spectra: {len(unique_specs):,}")
    print(f"  ★ 5/5 identity iso class: {iso_5_5_count}")
    print(f"  iso to 核 spectrum: {iso_with_core_spec}")
    sys.stdout.flush()

    # ============================================================
    # Analysis: what are the 5/5 iso classes?
    # ============================================================
    print(f"\n\n{'='*80}")
    print("(B) 5/5 identity 達成 iso class の中身")
    print("="*80)
    five_classes = [(s, info[0], info[1], info[2]) for s, info in unique_specs.items() if sum(info[1]) == 5]
    print(f"\n  5/5 iso class 数: {len(five_classes)}")
    for s, A, ids, count in five_classes[:10]:
        is_core = s == core_spec
        marker = " ★ 核 spectrum" if is_core else ""
        print(f"    spectrum {s[:4]}... count={count} ids={ids}{marker}")

    # Check isomorphism with core for 5/5 classes
    G_core = nx.from_numpy_array(A_core)
    n_iso_with_core = 0
    n_distinct = 0
    for s, A, ids, count in five_classes:
        G = nx.from_numpy_array(A)
        if nx.is_isomorphic(G, G_core):
            n_iso_with_core += 1
        else:
            n_distinct += 1

    print(f"\n  → 5/5 iso class が核と同型: {n_iso_with_core}")
    print(f"  → 5/5 iso class で 核と異なる: {n_distinct}")
    sys.stdout.flush()

    # ============================================================
    # Final verdict
    # ============================================================
    print(f"\n\n{'='*80}")
    print("★ 結論 — 数学的 uniqueness 評価")
    print("="*80)

    if n_distinct == 0:
        verdict = f"★★★★★ 全 5/5 iso class が **核と同型** — 核は唯一 (empirically)"
    elif n_distinct < 5:
        verdict = f"★★★★ 5/5 達成 iso class が {n_distinct + n_iso_with_core} 個 (うち核以外 {n_distinct})"
    else:
        verdict = f"★★★ 多くの iso class が 5/5 達成 ({n_distinct} 個) — uniqueness 弱"
    print(f"\n  {verdict}")

    print(f"""
  数値結果:
    10M trial で:
      triangle-free graphs: {triangle_free_count:,}
      distinct iso classes (by spectrum): {len(unique_specs):,}
      5/5 identity を満たす iso classes: {len(five_classes)}
        うち 核と同型: {n_iso_with_core}
        うち 核と異なる: {n_distinct}

    iso to 核: {iso_with_core_spec} times (out of {triangle_free_count} triangle-free)
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "N_trials": N_TRIALS,
        "triangle_free_count": triangle_free_count,
        "unique_spectra": len(unique_specs),
        "five_identity_iso_classes": len(five_classes),
        "iso_with_core_count": n_iso_with_core,
        "distinct_iso_classes_5_of_5": n_distinct,
        "iso_to_core_in_random": iso_with_core_spec,
        "elapsed_seconds": elapsed,
        "verdict": verdict,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round357_iso_enumeration.json"
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
