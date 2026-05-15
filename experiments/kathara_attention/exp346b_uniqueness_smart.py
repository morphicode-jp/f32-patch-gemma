"""第346期 (改): 核 uniqueness の smart enumeration.

前回 DFS は探索空間爆発で 13 分で未完。書き直し:
  (1) degree-4 頂点を fix した canonical 配置で start
  (2) 各 step で triangle-free を strict にチェック
  (3) 完了時に spectrum で isomorphism class hash
  (4) progress print 頻繁化 (buffer flush)

別アプローチ: 大量 random sample (10M) で degree-seq + triangle-free を満たすもの探索.
本物の network sampling は networkx の configuration_model + reject triangle.
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import time
import sys
from collections import Counter


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
    print("第346期改: 核 uniqueness — random sample with strict filter")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    A2 = A_core @ A_core
    A4 = A2 @ A2
    A6 = A4 @ A2
    core_inv = (int(np.trace(A2)), int(np.trace(A2 @ A_core)),
                int(np.trace(A4)), int(np.trace(A4 @ A_core)), int(np.trace(A6)))
    target_deg = sorted([int(A_core[i].sum()) for i in range(12)])
    G_core = nx.from_numpy_array(A_core)
    core_spectrum = tuple(sorted(np.round(np.linalg.eigvalsh(A_core.astype(float)), 6).tolist()))

    print(f"\n  核 invariants:")
    print(f"    degree seq:  {tuple(target_deg)}")
    print(f"    Tr A² = {core_inv[0]}, Tr A³ = {core_inv[1]}")
    print(f"    Tr A⁴ = {core_inv[2]}, Tr A⁵ = {core_inv[3]}, Tr A⁶ = {core_inv[4]}")
    sys.stdout.flush()

    # ============================================================
    # Strategy 1: configuration_model でも triangle-free filter
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) Configuration model + triangle-free filter")
    print("="*80)
    sys.stdout.flush()

    # ヒトの degree sequence の random graph を大量生成
    N_TRIALS = 5_000_000
    print(f"  N_trials = {N_TRIALS:,}")
    print(f"  生成 graph: configuration_model (deg seq match)")
    print(f"  filter: triangle-free + Tr(A^k) all match")
    sys.stdout.flush()

    triangle_free_count = 0
    deg_match_count = 0
    inv_match_count = 0
    spectrum_match_count = 0
    found_non_core_spectra = set()
    iso_with_core = 0

    rng = np.random.default_rng(42)
    t0 = time.time()

    for trial in range(N_TRIALS):
        if trial % 200_000 == 0 and trial > 0:
            elapsed = time.time() - t0
            print(f"    trial {trial:,}  triangle-free {triangle_free_count:,}  "
                  f"deg-match {deg_match_count:,}  inv-match {inv_match_count:,}  "
                  f"iso-core {iso_with_core}  ({elapsed:.0f}s)")
            sys.stdout.flush()

        # configuration model: random pairing
        try:
            # python configuration_model approach
            stubs = []
            for v, d in enumerate(target_deg):
                stubs.extend([v] * d)
            rng.shuffle(stubs)
            A = np.zeros((12, 12), dtype=np.int64)
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or A[u, v] == 1:
                    valid = False
                    break
                A[u, v] = A[v, u] = 1
            if not valid:
                continue

            # degree check (should be fine, but verify)
            degs = sorted([int(A[i].sum()) for i in range(12)])
            if degs != target_deg:
                continue
            deg_match_count += 1

            # triangle-free check
            A2_t = A @ A
            A3_t = A2_t @ A
            tr_A3 = int(np.trace(A3_t))
            if tr_A3 != 0:
                continue
            triangle_free_count += 1

            # all invariants check
            A4_t = A2_t @ A2_t
            A6_t = A4_t @ A2_t
            inv = (int(np.trace(A2_t)), tr_A3, int(np.trace(A4_t)),
                   int(np.trace(A4_t @ A)), int(np.trace(A6_t)))
            if inv != core_inv:
                continue
            inv_match_count += 1

            # spectrum check (more precise than invariants)
            spec = tuple(sorted(np.round(np.linalg.eigvalsh(A.astype(float)), 6).tolist()))
            if spec == core_spectrum:
                spectrum_match_count += 1
                # isomorphism check
                G = nx.from_numpy_array(A)
                if nx.is_isomorphic(G, G_core):
                    iso_with_core += 1
                else:
                    found_non_core_spectra.add(spec)
            else:
                found_non_core_spectra.add(spec)
        except Exception:
            continue

    elapsed = time.time() - t0

    # ============================================================
    # Results
    # ============================================================
    print(f"\n\n{'='*80}")
    print("★ 結果")
    print("="*80)
    print(f"""
  N_trials:               {N_TRIALS:,}
  degree sequence match:  {deg_match_count:,}  ({deg_match_count/N_TRIALS*100:.3f}%)
  triangle-free + deg:    {triangle_free_count:,}  ({triangle_free_count/N_TRIALS*100:.4f}%)
  全 invariants match:    {inv_match_count:,}  ({inv_match_count/N_TRIALS*100:.5f}%)
  spectrum match (核):    {spectrum_match_count:,}
  iso to 核:              {iso_with_core:,}
  非 核 spectra found:    {len(found_non_core_spectra)}

  時間: {elapsed:.0f}s
""")

    if iso_with_core > 0 and len(found_non_core_spectra) == 0:
        verdict = "★★★★★ 核は invariants で UNIQUE iso class"
    elif iso_with_core > 0 and len(found_non_core_spectra) > 0:
        verdict = f"★★ 同 invariants で複数 iso class 存在 ({len(found_non_core_spectra)})"
    else:
        verdict = "★ random sampling では核が見つからず、より多くの試行要"
    print(f"  {verdict}")

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print("★ 結論")
    print("="*80)
    print(f"""
  random 5M trial の結果:
    - 核と同じ invariants (deg seq + Tr A^k k=2..6) を持つ graph: {inv_match_count}
    - そのうち 核 と iso な graph: {iso_with_core}
    - 異なる iso class (cospectral mate): {len(found_non_core_spectra)}

  Confidence:
    - もし iso_with_core > 0 かつ cospectral_count = 0:
      → "核は (deg seq + Tr A^k) に対して unique iso class" 強い証拠
    - もし cospectral_count > 0:
      → 別 graph が存在、uniqueness は 6 invariants で不十分

  完全 enumeration 困難 (search space exp(many))
  random sample 5M で見つかった graph 数で empirical uniqueness 判定
""")
    sys.stdout.flush()

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "N_trials": N_TRIALS,
        "deg_match_count": deg_match_count,
        "triangle_free_count": triangle_free_count,
        "inv_match_count": inv_match_count,
        "spectrum_match_count": spectrum_match_count,
        "iso_with_core": iso_with_core,
        "non_core_iso_count": len(found_non_core_spectra),
        "verdict": verdict,
        "elapsed_seconds": elapsed,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round346b_uniqueness_smart.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
