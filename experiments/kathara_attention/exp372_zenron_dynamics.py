"""第372期: 全論 (Zenron) dynamics を graph space で走らせ、 核 に収束するか検証.

全論公式: x_i ← best(perturb(x_i), share(neighbors_i))

graph space での 解釈:
  x_i:        12-vertex graph (population of agents)
  perturb:    edge swap (1 edge remove + 1 edge add)
  share:      neighbor agent と adjacency XOR
  best:       7-identity score 高いものを keep

target: 核 graph (= 7-identity 完全達成)

approach:
  (1) 100 random 12V19E triangle-free graphs = initial population
  (2) 各 step:
      - perturb each (1 edge swap)
      - share with random neighbor (avg of 2 graphs, then prune to 19 edges)
      - best (= 7-id score 上位 100 keep)
  (3) 何 step で 7-id score = 7 達成?
  (4) 達成 graph が 核 と iso か?
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


def core_identity_score(A):
    """7 identity score (0-7)。 高いほど 核に近い"""
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
    A5 = A4 @ A
    Tr_A2 = int(np.trace(A2))
    Tr_A3 = int(np.trace(A3))
    Tr_A4 = int(np.trace(A4))
    score = 0
    if Tr_A4 == 270:
        score += 1  # id1
    try:
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        if abs((n_e + abs(evs[0])) - 22) < 0.01:
            score += 1  # id2
    except Exception:
        pass
    if max_deg + Tr_A2 == 42:
        score += 1  # id3
    if Tr_A3 == 0:
        score += 1  # id4
    if n == 12:
        score += 1  # id5
    # id6, id7 (heavy: aut count, C_5)
    if Tr_A3 == 0:  # only if triangle-free
        G = nx.from_numpy_array(A)
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        aut = 0
        for _ in GM.isomorphisms_iter():
            aut += 1
            if aut > 5:
                break
        if aut == 4:
            score += 1  # id6
        c5 = int(np.trace(A5)) // 10
        if c5 == 4:
            score += 1  # id7
    return score


def perturb(A, rng):
    """1 edge swap: remove a random edge, add a random non-edge."""
    A2 = A.copy()
    n = A2.shape[0]
    # collect existing edges and non-edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if A2[i, j] == 1]
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A2[i, j] == 0]
    if not edges or not non_edges:
        return A2
    e_remove = edges[rng.randint(0, len(edges) - 1)]
    e_add = non_edges[rng.randint(0, len(non_edges) - 1)]
    A2[e_remove[0], e_remove[1]] = A2[e_remove[1], e_remove[0]] = 0
    A2[e_add[0], e_add[1]] = A2[e_add[1], e_add[0]] = 1
    return A2


def share(A1, A2, rng):
    """Take graph "between" A1 and A2 - random subset of union edges keeping |E|=19."""
    n = A1.shape[0]
    union = np.maximum(A1, A2)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if union[i, j] == 1]
    if len(edges) < 19:
        return A1.copy()  # too few edges; fall back
    rng.shuffle(edges)
    chosen = edges[:19]
    A_new = np.zeros((n, n), dtype=np.int64)
    for (i, j) in chosen:
        A_new[i, j] = A_new[j, i] = 1
    return A_new


def random_init_graph(rng, n=12, n_edges=19):
    """Random 12V19E (any degree)."""
    edges_all = [(i, j) for i in range(n) for j in range(i+1, n)]
    idx = rng.sample(range(len(edges_all)), n_edges)
    A = np.zeros((n, n), dtype=np.int64)
    for ei in idx:
        u, v = edges_all[ei]
        A[u, v] = A[v, u] = 1
    return A


def main():
    print("=" * 80)
    print("第372期: Zenron dynamics — graph space 上で 核 に 収束 するか")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)
    G_core = nx.from_numpy_array(A_core)
    core_score = core_identity_score(A_core)
    print(f"\n  核 7-identity score: {core_score}/7 (should be 7)")
    sys.stdout.flush()

    # ============================================================
    # Zenron dynamics
    # ============================================================
    print(f"\n{'='*80}")
    print("(A) Zenron dynamics 設定")
    print("="*80)
    POP_SIZE = 80
    N_STEPS = 100
    rng = random.Random(42)

    print(f"\n  population size: {POP_SIZE}")
    print(f"  N steps: {N_STEPS}")
    print(f"  rule per agent:")
    print(f"    perturb_x = 1 edge swap")
    print(f"    share_neighbors = random union 19-edge subset")
    print(f"    best = keep higher 7-id score")
    sys.stdout.flush()

    # initial population
    population = [random_init_graph(rng) for _ in range(POP_SIZE)]
    scores = [core_identity_score(A) for A in population]

    initial_max = max(scores)
    initial_mean = np.mean(scores)
    print(f"\n  initial scores: max={initial_max}/7, mean={initial_mean:.2f}/7")

    t0 = time.time()
    history = []
    for step in range(N_STEPS):
        # apply Zenron: each agent's new state = best of perturb / share
        new_pop = []
        new_scores = []
        for i in range(POP_SIZE):
            x = population[i]
            x_p = perturb(x, rng)
            # pick random neighbor for share
            j = rng.randint(0, POP_SIZE - 1)
            while j == i:
                j = rng.randint(0, POP_SIZE - 1)
            x_s = share(x, population[j], rng)
            # best
            s_x = scores[i]
            s_p = core_identity_score(x_p)
            s_s = core_identity_score(x_s)
            options = [(s_x, x), (s_p, x_p), (s_s, x_s)]
            best = max(options, key=lambda t: t[0])
            new_pop.append(best[1])
            new_scores.append(best[0])
        population = new_pop
        scores = new_scores

        cur_max = max(scores)
        cur_mean = np.mean(scores)
        history.append((step, cur_max, cur_mean))
        if step % 10 == 0 or cur_max >= 7:
            elapsed = time.time() - t0
            print(f"    step {step:4d}: max={cur_max}/7, mean={cur_mean:.2f}/7  ({elapsed:.0f}s)")
            sys.stdout.flush()
        if cur_max >= 7:
            # find the agent
            idx = scores.index(7)
            A_found = population[idx]
            G_found = nx.from_numpy_array(A_found.astype(np.int64))
            is_core = nx.is_isomorphic(G_found, G_core)
            print(f"\n  ★ 7/7 reached at step {step}!")
            print(f"  Agent {idx} is iso to 核: {is_core}")
            sys.stdout.flush()
            break

    elapsed = time.time() - t0
    final_max = max(scores)
    final_mean = np.mean(scores)
    print(f"\n  final ({elapsed:.0f}s): max={final_max}/7, mean={final_mean:.2f}/7")

    # Distribution
    from collections import Counter
    score_dist = Counter(scores)
    print(f"  score distribution: {dict(sorted(score_dist.items()))}")

    # ============================================================
    # Verdict
    # ============================================================
    print(f"\n{'='*80}")
    print("★ 結論 — Zenron dynamics は 核 に converge?")
    print("="*80)
    if final_max == 7:
        verdict = "★★★★★ Zenron dynamics が 核 (7/7) に達成 → 仮説 STRONG SUPPORT"
    elif final_max >= 6:
        verdict = "★★★★ Zenron が 6/7 に到達 → 弱い support、 もっと step 必要"
    elif final_max > initial_max:
        verdict = f"★★★ improvement (init max {initial_max} → final {final_max}) → optimization 効いてる"
    else:
        verdict = f"★★ no improvement → Zenron rule design 見直し要"
    print(f"\n  {verdict}")
    print(f"  initial max: {initial_max}/7 → final max: {final_max}/7")
    print(f"  initial mean: {initial_mean:.2f} → final mean: {final_mean:.2f}")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "population_size": POP_SIZE,
        "n_steps": N_STEPS,
        "initial_max": initial_max,
        "initial_mean": initial_mean,
        "final_max": final_max,
        "final_mean": final_mean,
        "core_reached": final_max == 7,
        "final_score_distribution": dict(score_dist),
        "history_last_10": history[-10:],
        "verdict": verdict,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round372_zenron_dynamics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
