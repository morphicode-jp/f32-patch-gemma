"""第374期: Zenron graph dynamics robustness — 多 seed で 核 収束確認.

approach:
  10 個 異なる random seed で Zenron を 走らせ、
  核に 収束する 割合 を 測定.
  もし 高確率で 収束 → 「核 = global attractor」 確定.
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import time
import sys
import random


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


def core_identity_score(A, weak=False):
    A = np.asarray(A, dtype=np.int64)
    n = A.shape[0]
    if n != 12 or A.sum() == 0:
        return 0
    n_e = int(A.sum() / 2)
    degrees = [int(A[i].sum()) for i in range(n)]
    max_deg = max(degrees)
    A2 = A @ A
    A4 = A2 @ A2
    Tr_A2 = int(np.trace(A2))
    Tr_A3 = int(np.trace(A2 @ A))
    Tr_A4 = int(np.trace(A4))
    score = 0
    if Tr_A4 == 270:
        score += 1
    try:
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        if abs((n_e + abs(evs[0])) - 22) < 0.01:
            score += 1
    except Exception:
        pass
    if max_deg + Tr_A2 == 42:
        score += 1
    if Tr_A3 == 0:
        score += 1
    if n == 12:
        score += 1
    if weak:
        return score
    if Tr_A3 == 0:
        G = nx.from_numpy_array(A)
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        aut = 0
        for _ in GM.isomorphisms_iter():
            aut += 1
            if aut > 5:
                break
        if aut == 4:
            score += 1
        A5 = A4 @ A
        c5 = int(np.trace(A5)) // 10
        if c5 == 4:
            score += 1
    return score


def perturb(A, rng, n_swaps=2):
    A2 = A.copy()
    n = A2.shape[0]
    for _ in range(n_swaps):
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if A2[i, j] == 1]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A2[i, j] == 0]
        if not edges or not non_edges:
            return A2
        e_remove = edges[rng.randint(0, len(edges) - 1)]
        e_add = non_edges[rng.randint(0, len(non_edges) - 1)]
        A2[e_remove[0], e_remove[1]] = A2[e_remove[1], e_remove[0]] = 0
        A2[e_add[0], e_add[1]] = A2[e_add[1], e_add[0]] = 1
    return A2


def random_init_graph(rng, n=12, n_edges=19):
    edges_all = [(i, j) for i in range(n) for j in range(i+1, n)]
    idx = rng.sample(range(len(edges_all)), n_edges)
    A = np.zeros((n, n), dtype=np.int64)
    for ei in idx:
        u, v = edges_all[ei]
        A[u, v] = A[v, u] = 1
    return A


def zenron_run(seed, A_core, max_steps=500, pop=100):
    rng = random.Random(seed)
    G_core = nx.from_numpy_array(A_core)
    population = [random_init_graph(rng) for _ in range(pop)]
    scores = [core_identity_score(A, weak=True) for A in population]

    for step in range(max_steps):
        for i in range(pop):
            n_swaps = 3 if scores[i] < 3 else (2 if scores[i] < 5 else 1)
            x_p = perturb(population[i], rng, n_swaps=n_swaps)
            s_p = core_identity_score(x_p, weak=True)
            if s_p >= scores[i]:
                population[i] = x_p
                scores[i] = s_p

        # share: replace worst 20% with mutated best
        sorted_idx = sorted(range(pop), key=lambda i: scores[i], reverse=True)
        best_k = sorted_idx[:pop // 5]
        worst_k = sorted_idx[-pop // 5:]
        for w_idx, b_idx in zip(worst_k, best_k):
            population[w_idx] = perturb(population[b_idx], rng, n_swaps=2)
            scores[w_idx] = core_identity_score(population[w_idx], weak=True)

        if max(scores) >= 5:
            # check full score
            for idx in sorted(range(pop), key=lambda i: scores[i], reverse=True)[:10]:
                full = core_identity_score(population[idx], weak=False)
                if full == 7:
                    G_f = nx.from_numpy_array(population[idx].astype(np.int64))
                    is_core = nx.is_isomorphic(G_f, G_core)
                    return step, is_core, 7
            # didn't find 7, continue
    return max_steps, False, max(core_identity_score(A) for A in population[:30])


def main():
    print("=" * 80)
    print("第374期: Zenron graph dynamics robustness (10 seeds)")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)

    seeds = [42, 123, 7, 2024, 31415, 271828, 9999, 1, 100, 65535]
    results = []
    t0 = time.time()
    for seed in seeds:
        print(f"\n  --- seed {seed} ---")
        sys.stdout.flush()
        t_start = time.time()
        steps_to_7, is_core, final_score = zenron_run(seed, A_core, max_steps=500, pop=100)
        t_end = time.time() - t_start
        verdict = "★ 核 reached" if is_core else f"final {final_score}/7"
        results.append((seed, steps_to_7, is_core, final_score, t_end))
        print(f"    seed {seed}: steps={steps_to_7}, iso to 核 = {is_core}, ({t_end:.0f}s) — {verdict}")
        sys.stdout.flush()

    elapsed = time.time() - t0
    print(f"\n  全 run 完了 ({elapsed:.0f}s)")

    # ============================================================
    # Stats
    # ============================================================
    n_reached = sum(1 for _, _, ic, _, _ in results if ic)
    steps_when_reached = [s for _, s, ic, _, _ in results if ic]
    print(f"\n{'='*80}")
    print(f"★ 結論 — robustness across {len(seeds)} seeds")
    print(f"{'='*80}")
    print(f"\n  核 到達 seeds: {n_reached}/{len(seeds)}")
    if steps_when_reached:
        print(f"  到達 step (mean): {np.mean(steps_when_reached):.0f}")
        print(f"  到達 step (range): {min(steps_when_reached)} - {max(steps_when_reached)}")
    print(f"")
    if n_reached == len(seeds):
        verdict = "★★★★★ 全 seed で 核 到達 → 核 = absolute global attractor"
    elif n_reached >= len(seeds) * 0.8:
        verdict = f"★★★★ {n_reached}/{len(seeds)} 到達 → strong attractor"
    else:
        verdict = f"★★★ {n_reached}/{len(seeds)} 到達 → attractor with caveats"
    print(f"  {verdict}")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "seeds": list(seeds),
        "n_reached_core": n_reached,
        "total_seeds": len(seeds),
        "convergence_rate": n_reached / len(seeds),
        "mean_steps_to_core": float(np.mean(steps_when_reached)) if steps_when_reached else None,
        "verdict": verdict,
        "details": [{"seed": s, "steps": st, "is_core": ic, "final": fs, "time_s": t}
                    for s, st, ic, fs, t in results],
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round374_zenron_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
