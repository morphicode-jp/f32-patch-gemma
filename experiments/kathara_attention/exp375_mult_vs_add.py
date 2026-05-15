"""第375期: 全論の本質「掛け算 (annihilation 公理)」 vs 「足し算」 を 検証.

全論主張:
  importance = √(truth × connectivity)  ← 掛け算
  足し算 では 0 × x = 0 を表現できない (annihilation 公理 違反)

検証: 同じ Zenron dynamics で
  - 掛け算 importance: score_mult = √(完成度 × 安定度)
  - 足し算 importance: score_add  = (完成度 + 安定度) / 2
  → どちら が 核 に 早く 高確率で 到達 するか

graph space:
  truth      = 7-identity 達成度 (= 7 つのうち 満たす数)
  connectivity = graph の "良さ" (= adjacency が core 似ているか)
"""
from __future__ import annotations
import numpy as np
import math
import networkx as nx
import random
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


def compute_truth_connectivity(A):
    """全論 importance を 計算.
       truth = 7-identity 完成度 (0-7)
       connectivity = stability = (1 - 三角形ペナルティ) × triangle-free 達成度
    """
    A = np.asarray(A, dtype=np.int64)
    n = A.shape[0]
    if n != 12 or A.sum() == 0:
        return 0, 0
    n_e = int(A.sum() / 2)
    degrees = [int(A[i].sum()) for i in range(n)]
    max_deg = max(degrees)
    A2 = A @ A
    A4 = A2 @ A2
    A5 = A4 @ A
    Tr_A2 = int(np.trace(A2))
    Tr_A3 = int(np.trace(A2 @ A))
    Tr_A4 = int(np.trace(A4))
    Tr_A5 = int(np.trace(A5))
    score = 0
    if Tr_A4 == 270: score += 1
    try:
        evs = sorted(np.linalg.eigvalsh(A.astype(float)).tolist())
        if abs((n_e + abs(evs[0])) - 22) < 0.01: score += 1
    except Exception:
        pass
    if max_deg + Tr_A2 == 42: score += 1
    if Tr_A3 == 0: score += 1
    if n == 12: score += 1
    if Tr_A3 == 0:
        G = nx.from_numpy_array(A)
        GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
        aut = 0
        for _ in GM.isomorphisms_iter():
            aut += 1
            if aut > 5: break
        if aut == 4: score += 1
        c5 = Tr_A5 // 10
        if c5 == 4: score += 1
    # truth = 完成度 (0-7)
    truth = score
    # connectivity = stability = 三角形なさ + degree 均等性
    # connectivity range 0-1
    tri_free_score = 1.0 if Tr_A3 == 0 else max(0, 1 - Tr_A3/100)
    deg_balance = 1 - np.std(degrees) / 4  # 標準偏差 が 小さい ほど 高
    deg_balance = max(0, min(1, deg_balance))
    connectivity = (tri_free_score + deg_balance) / 2
    return truth, connectivity


def importance_mult(truth, conn):
    """全論 公式: √(truth × connectivity)"""
    return math.sqrt(truth * conn)


def importance_add(truth, conn):
    """対照: (truth + conn × 7) / 2 (= 同じ scale)"""
    return (truth + conn * 7) / 2


def perturb(A, rng, n_swaps=2):
    A2 = A.copy()
    n = A2.shape[0]
    for _ in range(n_swaps):
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if A2[i, j] == 1]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A2[i, j] == 0]
        if not edges or not non_edges: return A2
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


def zenron_run_with_importance(seed, A_core, importance_fn, max_steps=500, pop=100):
    rng = random.Random(seed)
    G_core = nx.from_numpy_array(A_core)
    population = [random_init_graph(rng) for _ in range(pop)]
    truths_conns = [compute_truth_connectivity(A) for A in population]
    importances = [importance_fn(t, c) for t, c in truths_conns]

    for step in range(max_steps):
        new_pop = []
        new_tc = []
        new_imp = []
        for i in range(pop):
            x = population[i]
            t_i, c_i = truths_conns[i]
            imp_i = importances[i]
            n_swaps = 3 if t_i < 3 else (2 if t_i < 5 else 1)
            x_p = perturb(x, rng, n_swaps=n_swaps)
            t_p, c_p = compute_truth_connectivity(x_p)
            imp_p = importance_fn(t_p, c_p)
            if imp_p >= imp_i:
                new_pop.append(x_p)
                new_tc.append((t_p, c_p))
                new_imp.append(imp_p)
            else:
                new_pop.append(x)
                new_tc.append((t_i, c_i))
                new_imp.append(imp_i)

        # share
        sorted_idx = sorted(range(pop), key=lambda i: new_imp[i], reverse=True)
        best_k = sorted_idx[:pop // 5]
        worst_k = sorted_idx[-pop // 5:]
        for w_idx, b_idx in zip(worst_k, best_k):
            new_pop[w_idx] = perturb(new_pop[b_idx], rng, n_swaps=2)
            t, c = compute_truth_connectivity(new_pop[w_idx])
            new_tc[w_idx] = (t, c)
            new_imp[w_idx] = importance_fn(t, c)

        population = new_pop
        truths_conns = new_tc
        importances = new_imp

        max_truth = max(t for t, c in truths_conns)
        if max_truth == 7:
            for idx in sorted(range(pop), key=lambda i: truths_conns[i][0], reverse=True)[:10]:
                if truths_conns[idx][0] == 7:
                    G_f = nx.from_numpy_array(population[idx].astype(np.int64))
                    if nx.is_isomorphic(G_f, G_core):
                        return step, True
    return max_steps, False


def main():
    print("=" * 80)
    print("第375期: 掛け算 (全論) vs 足し算 importance — 核到達率 比較")
    print("=" * 80)
    sys.stdout.flush()

    A_core = np.minimum(build_k1(), build_icosahedron()).astype(np.int64)

    seeds = [42, 123, 7, 2024, 31415, 271828, 9999, 1, 100, 65535]

    print(f"\n--- 掛け算 importance: √(truth × connectivity) ---")
    sys.stdout.flush()
    mult_reach = []
    t0 = time.time()
    for seed in seeds:
        t_start = time.time()
        steps, reached = zenron_run_with_importance(seed, A_core, importance_mult, max_steps=500, pop=100)
        t_end = time.time() - t_start
        mult_reach.append(reached)
        print(f"  seed {seed}: steps={steps}, reach={reached}, ({t_end:.0f}s)")
        sys.stdout.flush()

    print(f"\n--- 足し算 importance: (truth + connectivity × 7) / 2 ---")
    sys.stdout.flush()
    add_reach = []
    for seed in seeds:
        t_start = time.time()
        steps, reached = zenron_run_with_importance(seed, A_core, importance_add, max_steps=500, pop=100)
        t_end = time.time() - t_start
        add_reach.append(reached)
        print(f"  seed {seed}: steps={steps}, reach={reached}, ({t_end:.0f}s)")
        sys.stdout.flush()

    # ============================================================
    # 結論
    # ============================================================
    print(f"\n{'='*80}")
    print(f"★ 結論")
    print(f"{'='*80}")
    n_mult = sum(mult_reach)
    n_add = sum(add_reach)
    print(f"\n  掛け算 importance: 核到達 {n_mult}/{len(seeds)}")
    print(f"  足し算 importance: 核到達 {n_add}/{len(seeds)}")

    if n_mult > n_add:
        verdict = f"★★★★★ 掛け算 ({n_mult}) > 足し算 ({n_add})、 全論主張支持"
    elif n_mult == n_add:
        verdict = f"★★★ 同等 ({n_mult} = {n_add})、 差なし"
    else:
        verdict = f"★★ 足し算 ({n_add}) > 掛け算 ({n_mult})、 全論主張支持されず"
    print(f"\n  {verdict}")
    print(f"""
  honest 解釈:
    全論主張 「掛け算 が 唯一無二 (annihilation 公理)」 を
    この test で 検証.
    結果: {n_mult} / {n_add} = mult/add 比
""")

    import json, os
    THIS = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(os.path.dirname(THIS))
    out = {
        "n_seeds": len(seeds),
        "multiplicative_reach": n_mult,
        "additive_reach": n_add,
        "mult_results": mult_reach,
        "add_results": add_reach,
        "verdict": verdict,
    }
    out_path = f"{REPO}/experiments/kathara_attention/results_round375_mult_vs_add.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
