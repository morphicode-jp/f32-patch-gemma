"""mimir_coevolution — Param × Weight Co-evolution (Pattern 4).

背景:
  multi-metric eval_fn (LLM キャリブの {"hellaswag": ..., "mmlu": ..., "ppl": ...})
  で「どう aggregate するか (=何の weight をかけるか)」を人間が決めるのは難しい。
  Pattern 4: params と weights を同時進化させ、**全 metric で頑張る params** と
  **そういう params を露出させる weights の組合せ** を発見する。

Emergence 狙い:
  - 単一 metric に特化した params (easy specialist) は淘汰される
  - 複数 metric で同時に高得点を取る balanced params が生存
  - どの metric が互いにトレードオフか、どの metric が共働するかが weight の
    進化過程から可視化される (observer correlation と類似)

仕組み:
  個体 = (param_vector, weight_vector) のペア
  Fitness = aggregator(multi_metric_eval(param_vector))
    - "min": 最悪 metric の score (保守、balanced params が勝つ)
    - "harmonic": harmonic mean (0 付近で急落 → 全 metric 高い param が勝つ)
    - "weighted": 個体自身の weight で加重和 (weights も選択圧受ける)
  Selection: tournament
  Mutation: param と weight 両方に Gaussian 摂動
  Crossover: 50/50 uniform

使い方:
    from twelve.agent.mimir_coevolution import mimir_cardinal_coevolution

    def multi_eval(p):
        return {"hs": hellaswag_score(p),
                "mmlu": mmlu_score(p),
                "ppl_neg": -measure_ppl(p)}

    r = mimir_cardinal_coevolution(
        multi_eval,
        param_ranges=[(0.5, 1.5)] * 61,
        metric_names=["hs", "mmlu", "ppl_neg"],
        time_budget=3600,
        population=16,
        generations=30,
    )
    print(r["best_params"])        # balanced param (全 metric 高い)
    print(r["best_weights"])       # その param を押し上げた weight 組合せ
    print(r["best_metrics"])       # 各 metric の実測値
    print(r["weight_evolution"])   # weight が世代で収束した先 (= 重要な metric)
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Optional, Sequence


def _harmonic_mean(vals: Sequence[float]) -> float:
    """Harmonic mean — heavily penalizes any near-0 or negative value.
    For mixed-sign metrics, we min-shift before harmonic, then shift back.
    """
    vs = [float(v) for v in vals]
    if not vs:
        return 0.0
    m = min(vs)
    if m <= 0:
        shift = abs(m) + 1e-6
        vs_shifted = [v + shift for v in vs]
        h = len(vs_shifted) / sum(1.0 / v for v in vs_shifted)
        return h - shift
    return len(vs) / sum(1.0 / v for v in vs)


def _aggregate(metrics: dict, weights: Optional[Sequence[float]],
               names: Sequence[str], mode: str) -> float:
    vals = [float(metrics.get(n, 0.0)) for n in names]
    if mode == "min":
        return min(vals) if vals else 0.0
    if mode == "harmonic":
        return _harmonic_mean(vals)
    if mode == "weighted":
        if weights is None:
            return sum(vals) / max(1, len(vals))
        ws = [max(0.0, float(w)) for w in weights]
        s = sum(ws)
        if s <= 0:
            return sum(vals) / max(1, len(vals))
        return sum(w * v for w, v in zip(ws, vals)) / s
    # fallback: mean
    return sum(vals) / max(1, len(vals))


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mutate_vec(vec: Sequence[float], ranges: Sequence[tuple],
                sigma: float, rng: random.Random) -> list:
    child = []
    for v, (lo, hi) in zip(vec, ranges):
        span = hi - lo
        nv = v + rng.gauss(0, sigma * span)
        child.append(_clip(nv, lo, hi))
    return child


def _crossover_vec(a: Sequence[float], b: Sequence[float],
                   rng: random.Random) -> list:
    return [rng.choice([va, vb]) for va, vb in zip(a, b)]


def _tournament(pop_fits: Sequence[float], k: int,
                rng: random.Random) -> int:
    n = len(pop_fits)
    idxs = rng.sample(range(n), min(k, n))
    return max(idxs, key=lambda i: pop_fits[i])


def mimir_cardinal_coevolution(
    multi_metric_eval: Callable,
    param_ranges: Sequence[tuple],
    metric_names: Sequence[str],
    *,
    weight_ranges: Optional[Sequence[tuple]] = None,
    time_budget: float = 3600.0,
    population: int = 16,
    generations: int = 30,
    mutation_sigma_params: float = 0.1,
    mutation_sigma_weights: float = 0.15,
    fitness_mode: str = "harmonic",
    tournament_k: int = 3,
    elitism: int = 2,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Co-evolve (params, weights) pairs, report balanced winner.

    Args:
      multi_metric_eval: f(params) -> dict[metric_name, float]
      param_ranges: [(lo, hi), ...] for params
      metric_names: ordered list of metric keys to aggregate
      weight_ranges: per-weight [(lo, hi), ...], default [(0, 1)] * len(metric_names)
      time_budget: total wall budget (s)
      population: individuals per gen
      generations: max gens (time-bounded)
      mutation_sigma_params: Gaussian sigma for param mutation (range-relative)
      mutation_sigma_weights: Gaussian sigma for weight mutation
      fitness_mode: "min" / "harmonic" / "weighted"
        - "min": worst metric (conservative, balanced params win)
        - "harmonic": harmonic mean (strongly balanced, near-0 metric crashes fitness)
        - "weighted": use individual's own weights (co-evolution active)
      seed: RNG seed
      verbose: progress print

    Returns:
      best_params, best_score (aggregated), best_weights
      best_metrics: full dict of metrics at best_params
      weight_evolution: per-gen mean weights (shows which metrics stabilize)
      gen_history: per-gen best/mean fitness
      n_metrics, fitness_mode
      elapsed_s, n_generations_run
    """
    rng = random.Random(seed)
    n_metrics = len(metric_names)
    if weight_ranges is None:
        weight_ranges = [(0.0, 1.0)] * n_metrics
    if len(weight_ranges) != n_metrics:
        raise ValueError(
            f"weight_ranges length ({len(weight_ranges)}) != "
            f"metric_names length ({n_metrics})"
        )

    t0 = time.time()

    # -------- Init population --------
    def _make_param():
        return [rng.uniform(lo, hi) for lo, hi in param_ranges]

    def _make_weight():
        return [rng.uniform(lo, hi) for lo, hi in weight_ranges]

    pop_params = [_make_param() for _ in range(population)]
    pop_weights = [_make_weight() for _ in range(population)]

    # Evaluate
    pop_metrics: list[dict] = []
    pop_fitness: list[float] = []
    for p, w in zip(pop_params, pop_weights):
        try:
            m = multi_metric_eval(p)
            if not isinstance(m, dict):
                m = {metric_names[0]: float(m)}
            f = _aggregate(m, w, metric_names, fitness_mode)
        except Exception:
            m = {n: 0.0 for n in metric_names}
            f = float("-inf")
        pop_metrics.append(m)
        pop_fitness.append(f)

    best_idx = max(range(len(pop_fitness)), key=lambda i: pop_fitness[i])
    best = {
        "params": list(pop_params[best_idx]),
        "weights": list(pop_weights[best_idx]),
        "metrics": dict(pop_metrics[best_idx]),
        "score": pop_fitness[best_idx],
        "gen_found": 0,
    }

    gen_history: list[dict] = []
    weight_evolution: list[list[float]] = []

    for gen in range(1, generations + 1):
        if time.time() - t0 > time_budget:
            if verbose:
                print(f"  [coevo] budget exhausted at gen {gen}")
            break

        # Rank
        order = sorted(range(len(pop_fitness)),
                       key=lambda i: pop_fitness[i], reverse=True)

        # Elitism
        next_p = [list(pop_params[i]) for i in order[:elitism]]
        next_w = [list(pop_weights[i]) for i in order[:elitism]]
        next_m = [dict(pop_metrics[i]) for i in order[:elitism]]
        next_f = [pop_fitness[i] for i in order[:elitism]]

        # Reproduce
        while len(next_p) < population:
            i1 = _tournament(pop_fitness, tournament_k, rng)
            i2 = _tournament(pop_fitness, tournament_k, rng)
            cp = _crossover_vec(pop_params[i1], pop_params[i2], rng)
            cw = _crossover_vec(pop_weights[i1], pop_weights[i2], rng)
            cp = _mutate_vec(cp, param_ranges, mutation_sigma_params, rng)
            cw = _mutate_vec(cw, weight_ranges, mutation_sigma_weights, rng)
            try:
                cm = multi_metric_eval(cp)
                if not isinstance(cm, dict):
                    cm = {metric_names[0]: float(cm)}
                cf = _aggregate(cm, cw, metric_names, fitness_mode)
            except Exception:
                cm = {n: 0.0 for n in metric_names}
                cf = float("-inf")
            next_p.append(cp)
            next_w.append(cw)
            next_m.append(cm)
            next_f.append(cf)

        pop_params = next_p
        pop_weights = next_w
        pop_metrics = next_m
        pop_fitness = next_f

        # Track gen best
        g_best_idx = max(range(len(pop_fitness)),
                         key=lambda i: pop_fitness[i])
        if pop_fitness[g_best_idx] > best["score"]:
            best = {
                "params": list(pop_params[g_best_idx]),
                "weights": list(pop_weights[g_best_idx]),
                "metrics": dict(pop_metrics[g_best_idx]),
                "score": pop_fitness[g_best_idx],
                "gen_found": gen,
            }

        # Weight evolution: mean of top-half weights (where selection pressure points)
        top_half = order[:len(order) // 2] if len(order) >= 2 else order
        if top_half:
            mean_w = [sum(pop_weights[i][j] for i in top_half) / len(top_half)
                      for j in range(n_metrics)]
        else:
            mean_w = [0.0] * n_metrics
        weight_evolution.append(mean_w)

        valid_fits = [f for f in pop_fitness if f != float("-inf")]
        gen_history.append({
            "gen": gen,
            "best_score": pop_fitness[g_best_idx],
            "mean_score": sum(valid_fits) / len(valid_fits) if valid_fits else 0.0,
            "mean_weights": mean_w,
        })

        if verbose and (gen % max(1, generations // 5) == 0):
            print(f"  [coevo] gen {gen}: best={pop_fitness[g_best_idx]:.4f} "
                  f"weights(mean)={['%.2f' % x for x in mean_w]}")

    # Weight stability: final-gen weight std across top half
    final_weights = weight_evolution[-1] if weight_evolution else [0.0] * n_metrics
    weight_std = []
    if weight_evolution:
        last = weight_evolution[-1]
        prev = weight_evolution[max(0, len(weight_evolution) - 5):-1] if len(weight_evolution) > 1 else [last]
        for j in range(n_metrics):
            vals = [w[j] for w in prev]
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                weight_std.append(math.sqrt(var))
            else:
                weight_std.append(0.0)
    else:
        weight_std = [0.0] * n_metrics

    # Which metrics emerged as most important (highest mean weight in top-half)
    ranked_metrics = sorted(
        zip(metric_names, final_weights),
        key=lambda x: x[1], reverse=True,
    )

    return {
        "best_params": best["params"],
        "best_weights": dict(zip(metric_names, best["weights"])),
        "best_metrics": best["metrics"],
        "best_score": best["score"],
        "best_found_at_gen": best["gen_found"],
        "metric_ranking_by_weight": ranked_metrics,
        "weight_evolution_mean": weight_evolution,
        "weight_stability_std_last5": dict(zip(metric_names, weight_std)),
        "gen_history": gen_history,
        "n_generations_run": len(gen_history),
        "n_metrics": n_metrics,
        "fitness_mode": fitness_mode,
        "elapsed_s": time.time() - t0,
    }


__all__ = ["mimir_cardinal_coevolution"]
