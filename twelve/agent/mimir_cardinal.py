"""mimir_cardinal — Council × Cardinal Hierarchy (Pattern 3).

背景:
  mimir_brunnir は 4 specialist 並列で弱点を相互補完するが、高次元多峰問題では
  単体 mimir と同様 proxy に嵌まることがある。Cardinal の進化 (集団 + 変異 +
  淘汰) は局所解を飛び越える力を持つが、高次元で世代数が足りない弱点がある。

Hierarchy: Council で次元圧縮 → Cardinal (汎用 GA) で進化探索
  Step 1 (20% budget): mimir_brunnir で structure_only、active_dims を抽出
    - 30 dim → 5 dim に圧縮 (2^25 = 33M× 探索空間縮小)
    - dead dims は中点固定
  Step 2 (80% budget): 汎用 GA で active_dims 空間を進化探索
    - population 集団選択で局所解脱出
    - tournament + crossover + Gaussian mutation
    - noise 集団平均で robust

既存 Cardinal (tamashii/phase_10_3_cardinal.py) は 3D voxel 専用、汎用最適化に
使えないので本 module で simple GA を実装。Cardinal の「仕組み」(populaiton /
tournament / mutation) だけ流用した形。

使い方:
    from twelve.agent.mimir_cardinal import mimir_cardinal_hierarchy
    r = mimir_cardinal_hierarchy(eval_fn, ranges, time_budget=600)
    print(r["best_params"], r["best_score"])
    print(r["active_dims"])  # council が絞った次元
    print(r["ga_generations"])  # 実行した世代数
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional, Sequence


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _make_individual(ranges: Sequence[tuple], rng: random.Random) -> list:
    return [rng.uniform(lo, hi) for lo, hi in ranges]


def _mutate(ind: Sequence[float], ranges: Sequence[tuple], sigma: float,
            rng: random.Random) -> list:
    """Gaussian mutation with per-dim sigma relative to range."""
    child = []
    for v, (lo, hi) in zip(ind, ranges):
        span = hi - lo
        nv = v + rng.gauss(0, sigma * span)
        child.append(_clip(nv, lo, hi))
    return child


def _crossover(a: Sequence[float], b: Sequence[float],
               rng: random.Random) -> list:
    """Uniform crossover: each gene ~50% from parent a or b."""
    return [rng.choice([va, vb]) for va, vb in zip(a, b)]


def _tournament_select(pop: list, fitnesses: list, k: int,
                       rng: random.Random) -> int:
    """Tournament selection: pick k random, return index of best among them."""
    candidates = rng.sample(range(len(pop)), min(k, len(pop)))
    best_idx = candidates[0]
    for i in candidates[1:]:
        if fitnesses[i] > fitnesses[best_idx]:
            best_idx = i
    return best_idx


def _ga_optimize(
    eval_fn: Callable,
    ranges: Sequence[tuple],
    *,
    time_budget: float,
    population_size: int = 16,
    generations: int = 30,
    mutation_sigma: float = 0.1,
    tournament_k: int = 3,
    elitism: int = 2,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Simple tournament-based GA. eval_fn: higher = better (matches mimir convention)."""
    rng = random.Random(seed)
    t0 = time.time()

    pop = [_make_individual(ranges, rng) for _ in range(population_size)]
    fitnesses = []
    for ind in pop:
        try:
            fitnesses.append(float(eval_fn(ind)))
        except Exception:
            fitnesses.append(float("-inf"))

    best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best = {
        "params": list(pop[best_idx]),
        "score": fitnesses[best_idx],
        "gen_found": 0,
    }

    gen_history = []
    for gen in range(1, generations + 1):
        if time.time() - t0 > time_budget:
            if verbose:
                print(f"  [GA] budget exhausted at gen {gen}")
            break

        # Sort by fitness (descending)
        order = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)

        # Elitism: top N survive
        next_pop = [list(pop[i]) for i in order[:elitism]]
        next_fitnesses = [fitnesses[i] for i in order[:elitism]]

        # Fill via tournament + crossover + mutation
        while len(next_pop) < population_size:
            p1 = pop[_tournament_select(pop, fitnesses, tournament_k, rng)]
            p2 = pop[_tournament_select(pop, fitnesses, tournament_k, rng)]
            child = _crossover(p1, p2, rng)
            child = _mutate(child, ranges, mutation_sigma, rng)
            try:
                score = float(eval_fn(child))
            except Exception:
                score = float("-inf")
            next_pop.append(child)
            next_fitnesses.append(score)

        pop = next_pop
        fitnesses = next_fitnesses

        gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best_score = fitnesses[gen_best_idx]
        if gen_best_score > best["score"]:
            best = {
                "params": list(pop[gen_best_idx]),
                "score": gen_best_score,
                "gen_found": gen,
            }

        gen_history.append({
            "gen": gen,
            "best_score": gen_best_score,
            "mean_score": sum(f for f in fitnesses if f != float("-inf"))
                          / max(1, sum(1 for f in fitnesses if f != float("-inf"))),
        })

        if verbose and gen % max(1, generations // 5) == 0:
            print(f"  [GA] gen {gen}: best={gen_best_score:.4f}")

    return {
        "best_params": best["params"],
        "best_score": best["score"],
        "best_found_at_gen": best["gen_found"],
        "n_generations_run": len(gen_history),
        "gen_history": gen_history,
        "population_final": pop,
        "fitnesses_final": fitnesses,
        "elapsed_s": time.time() - t0,
    }


def mimir_cardinal_hierarchy(
    eval_fn: Callable,
    param_ranges: Sequence[tuple],
    *,
    time_budget: float = 600.0,
    council_budget_share: float = 0.2,
    ga_population: int = 16,
    ga_generations: int = 30,
    ga_mutation_sigma: float = 0.1,
    ga_seed: int = 0,
    dead_dim_fill: str = "midpoint",
    experience_id: str = "hierarchy",
    verbose: bool = False,
) -> dict:
    """Council で次元圧縮 → GA で進化探索 (Pattern 3 Hierarchy).

    Step 1: mimir_brunnir(mode="structure_only") で active_dims 抽出
            (budget: council_budget_share × time_budget)
    Step 2: GA で active 次元のみ進化
            (budget: (1 - council_budget_share) × time_budget)
    Step 3: best_params を full 次元に復元 (dead は midpoint で埋める)

    Args:
      eval_fn: f(full_params) -> float
      param_ranges: [(lo, hi), ...] 全次元
      time_budget: 総時間予算
      council_budget_share: Council に回す予算比 (0.2 = 20%)
      ga_population: GA 集団サイズ
      ga_generations: GA 世代数上限 (時間切れで途中終了可)
      ga_mutation_sigma: 変異 sigma (range 相対)
      ga_seed: GA RNG seed
      dead_dim_fill: "midpoint" (default) / "council_best" (council の best_params)
      experience_id: namespace
      verbose: 進捗出力

    Returns:
      best_params, best_score (full dim)
      active_dims, dead_dims (council 由来)
      council_result (Step 1 full dict)
      ga_result (Step 2 full dict)
      ga_generations (実際に走った世代数)
      elapsed_s, council_elapsed_s, ga_elapsed_s
    """
    from twelve.agent.mimir_brunnir import mimir_brunnir

    t0 = time.time()
    n_dims = len(list(param_ranges))

    # -------- Step 1: Council structure scan --------
    council_budget = time_budget * council_budget_share
    if verbose:
        print(f"[hierarchy] Step 1: council structure_only ({council_budget:.1f}s budget)")
    council_specialists = [
        {"name": "default",   "kwargs": {},                          "role": ""},
        {"name": "expensive", "kwargs": {"eval_cost_hint": 2.0},     "role": ""},
    ]  # lightweight 2-specialist scan to save budget for GA
    r_council = mimir_brunnir(
        eval_fn, list(param_ranges),
        time_budget=council_budget,
        specialists=council_specialists,
        mode="structure_only",
        executor="thread",
        experience_id=f"{experience_id}_scan",
        verbose=verbose,
    )
    council_elapsed = time.time() - t0

    active_dims = list(r_council.get("active_dims") or [])
    dead_dims = list(r_council.get("dead_dims") or [])

    if not active_dims:
        # Pathological: all dims dead; fall back to full-space GA
        if verbose:
            print("[hierarchy] no active_dims; falling back to full-space GA")
        active_dims = list(range(n_dims))

    # -------- Step 2: GA on active dims only --------
    reduced_ranges = [param_ranges[i] for i in active_dims]

    # Dead dim filler values
    council_best = r_council.get("best_params")
    if dead_dim_fill == "council_best" and council_best is not None:
        dead_vals = list(council_best)
    else:
        dead_vals = [(lo + hi) / 2.0 for lo, hi in param_ranges]

    def _reduced_eval(reduced_p: Sequence[float]) -> float:
        full = list(dead_vals)  # start from dead-filled full vector
        for i, idx in enumerate(active_dims):
            full[idx] = float(reduced_p[i])
        return float(eval_fn(full))

    ga_budget = time_budget - council_elapsed
    if verbose:
        print(f"[hierarchy] Step 2: GA on {len(active_dims)}d active "
              f"({ga_budget:.1f}s budget)")
    t_ga = time.time()
    r_ga = _ga_optimize(
        _reduced_eval,
        reduced_ranges,
        time_budget=ga_budget,
        population_size=ga_population,
        generations=ga_generations,
        mutation_sigma=ga_mutation_sigma,
        seed=ga_seed,
        verbose=verbose,
    )
    ga_elapsed = time.time() - t_ga

    # -------- Step 3: reconstruct full best_params --------
    best_full = list(dead_vals)
    for i, idx in enumerate(active_dims):
        best_full[idx] = r_ga["best_params"][i]

    return {
        "best_params": best_full,
        "best_score": r_ga["best_score"],
        "active_dims": active_dims,
        "dead_dims": dead_dims,
        "n_active": len(active_dims),
        "n_dead": len(dead_dims),
        "ga_generations_run": r_ga["n_generations_run"],
        "ga_best_found_at_gen": r_ga["best_found_at_gen"],
        "council_result": r_council,
        "ga_result": r_ga,
        "council_elapsed_s": council_elapsed,
        "ga_elapsed_s": ga_elapsed,
        "elapsed_s": time.time() - t0,
        "dead_dim_fill": dead_dim_fill,
    }


__all__ = ["mimir_cardinal_hierarchy", "_ga_optimize"]
