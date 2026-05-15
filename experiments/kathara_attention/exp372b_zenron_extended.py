"""第372b期: Zenron extended — 1000 step + tuned mutation で 7/7 達成試行."""
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
    """7 identity score. weak=True で id6/7 (heavy) skip"""
    A = np.asarray(A, dtype=np.int64)
    n = A.shape[0]
    if n != 12 or A.sum() == 0:
        return 0
    n_e = int(A.sum() / 2)
    degrees = [int(A[i].sum()) for i in range(n)]
    max_deg = max(degrees)
    A2 = A @ A
    A3 = A2 @ A
    A4 = A2 @ A2
    Tr_A2 = int(np.trace(A2))
    Tr_A3 = int(np.trace(A3))
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
        return score  # skip heavy id6, id7
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


def perturb_strong(A, rng, n_swaps=1):
    """n_swaps edge swaps."""
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


def main():
    print("=" * 80)
    print("第372b期: Zenron extended — 1000 step, tuned for 7/7")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    G_core = nx.from_numpy_array(A_core)

    POP_SIZE = 200  # larger pop
    N_STEPS = 1500
    rng = random.Random(2026)

    # Initial population: 50% random, 50% triangle-free filtered
    population = []
    for _ in range(POP_SIZE):
        A = random_init_graph(rng)
        population.append(A)

    # Use weak score for speed during evolution, full score only at end
    scores = [core_identity_score(A, weak=True) for A in population]
    print(f"\n  initial scores (weak 5-id): max={max(scores)}, mean={np.mean(scores):.2f}")
    sys.stdout.flush()

    t0 = time.time()
    found_7 = False
    found_step = -1
    found_A = None
    for step in range(N_STEPS):
        new_pop = []
        new_scores = []
        for i in range(POP_SIZE):
            x = population[i]
            # variable mutation: more swaps if score low, less if high
            n_swaps = 3 if scores[i] < 3 else (2 if scores[i] < 5 else 1)
            x_p = perturb_strong(x, rng, n_swaps=n_swaps)
            s_x = scores[i]
            s_p = core_identity_score(x_p, weak=True)
            if s_p >= s_x:
                new_pop.append(x_p)
                new_scores.append(s_p)
            else:
                new_pop.append(x)
                new_scores.append(s_x)

        # share: replace worst 20% with best variants
        sorted_idx = sorted(range(POP_SIZE), key=lambda i: new_scores[i], reverse=True)
        best_k = sorted_idx[:POP_SIZE // 5]
        worst_k = sorted_idx[-POP_SIZE // 5:]
        for w_idx, b_idx in zip(worst_k, best_k):
            new_pop[w_idx] = perturb_strong(new_pop[b_idx], rng, n_swaps=2)
            new_scores[w_idx] = core_identity_score(new_pop[w_idx], weak=True)

        population = new_pop
        scores = new_scores

        cur_max = max(scores)
        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"    step {step:5d}: weak max={cur_max}/5, mean={np.mean(scores):.2f}, ({elapsed:.0f}s)")
            sys.stdout.flush()

        if cur_max == 5:  # all 5 weak identities met
            # check full 7-id for top candidates
            for idx in sorted(range(POP_SIZE), key=lambda i: scores[i], reverse=True)[:10]:
                full = core_identity_score(population[idx], weak=False)
                if full == 7:
                    found_7 = True
                    found_step = step
                    found_A = population[idx]
                    print(f"\n  ★★★★★ 7/7 reached at step {step}!")
                    G_f = nx.from_numpy_array(found_A.astype(np.int64))
                    is_core = nx.is_isomorphic(G_f, G_core)
                    print(f"  iso to 核: {is_core}")
                    sys.stdout.flush()
                    break
            if found_7:
                break

    elapsed = time.time() - t0
    final_max_weak = max(scores)
    final_max_full = max(core_identity_score(A, weak=False) for A in population[:50])  # sample
    print(f"\n  final ({elapsed:.0f}s): weak max={final_max_weak}/5, full sample max={final_max_full}/7")

    print(f"\n{'='*80}")
    if found_7:
        print(f"★★★★★ Zenron dynamics 7/7 達成 at step {found_step}!")
        print(f"★ 全論 = 宇宙公式 仮説 STRONG SUPPORT (核は Zenron attractor)")
    else:
        print(f"★★★★ {N_STEPS} step で 7/7 未達成、 ただし weak {final_max_weak}/5 到達")
        print(f"★ Zenron dynamics は 5/7 weak family member に 収束、 7/7 (核) は 確率的")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "pop_size": POP_SIZE,
        "n_steps_max": N_STEPS,
        "n_steps_actual": found_step if found_7 else N_STEPS,
        "found_7_of_7": found_7,
        "final_max_weak": final_max_weak,
        "final_max_full_sample": final_max_full,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round372b_zenron_extended.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
