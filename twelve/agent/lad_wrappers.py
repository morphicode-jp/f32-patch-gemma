"""LaD wrappers — stochastic eval_fn を structured 測定に昇格するヘルパー群.

問題: 現 mimir/owl の auto-seed は 1 call = 1 scalar を前提にしてる。stochastic
eval_fn (LLM 生成等) では各 sample が noisy 1-shot になり、seed 段階で proxy が
noise を fit する。

解: eval_fn を包んで「1 call → N 実測の LaD 化」に変換する wrapper を提供:

  1. wrap_stochastic:  N 回集約、scalar 1 個返す (最小、backward compat)
  2. wrap_multi_obs:   N 個の個別結果を dict で返す → multi-observer path 活性
  3. wrap_llm_judge:   LLM 生成 + LLM 判定、dimension 別 dict 返す (LaD 完全版)

使い方:

    from twelve.agent.lad_wrappers import wrap_stochastic, wrap_multi_obs
    from twelve.agent.mimir import mimir

    # 最小 — seed でもう 1 回ずつ 20 call 集約
    r = mimir(wrap_stochastic(my_llm_eval, n=20), ranges, time_budget=1800)

    # LaD — 20 個の個別 generation を multi-observer で見る
    r = mimir(wrap_multi_obs(my_llm_eval, n=20), ranges, time_budget=1800)
    print(r["stable_active"])  # 20 gen 全てで意味ある dim
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence


def _aggregate(vals: Sequence[float], method: str = "median") -> float:
    """Aggregate a list of floats into a single scalar. median is noise-robust."""
    vs = [float(v) for v in vals]
    if not vs:
        return 0.0
    if method == "mean":
        return sum(vs) / len(vs)
    if method == "min":
        return min(vs)
    if method == "max":
        return max(vs)
    # median
    vs_s = sorted(vs)
    n = len(vs_s)
    return vs_s[n // 2] if n % 2 else (vs_s[n // 2 - 1] + vs_s[n // 2]) / 2


def wrap_stochastic(
    eval_fn: Callable,
    n: int = 20,
    agg: str = "median",
) -> Callable:
    """Wrap a stochastic eval_fn to call N times and aggregate to scalar.

    Use this when:
      - eval_fn returns a single float
      - the float is noisy across calls with same params (e.g., LLM sampling)
      - you want a robust scalar measurement for mimir's scalar path

    Args:
      eval_fn: f(params) -> float  (stochastic)
      n:       number of calls per wrapped invocation (default 20)
      agg:     "median" (noise-robust, default) / "mean" / "min" / "max"

    Returns:
      wrapped(params) -> float  (aggregated across n calls)

    Example:
      noisy = lambda p: base_fn(p) + np.random.randn() * 0.3
      robust = wrap_stochastic(noisy, n=20)
      r = mimir(robust, ranges, time_budget=300)
    """
    if n < 1:
        raise ValueError(f"wrap_stochastic: n must be >= 1, got {n}")

    def wrapped(params):
        vals = [float(eval_fn(params)) for _ in range(n)]
        return _aggregate(vals, agg)
    wrapped.__name__ = f"wrap_stochastic_{getattr(eval_fn, '__name__', 'fn')}_n{n}"
    return wrapped


def wrap_multi_obs(
    eval_fn: Callable,
    n: int = 20,
    key_prefix: str = "sample",
) -> Callable:
    """Wrap a stochastic eval_fn to return N observations as a LaD dict.

    Activates mimir/owl's multi-observer analysis path. Each call produces
    a dict of N independent measurements, which mimir treats as distinct
    observers. stable_active (dims consistent across observers) / observer_dependent
    (dims varying with observer = noise) become available.

    Use this when:
      - eval_fn returns a single float
      - you want LaD structured measurement (not just aggregated scalar)
      - you want mimir to SEPARATE signal (structure consistent across gens)
        from noise (structure varying with gen)

    Args:
      eval_fn:    f(params) -> float  (stochastic)
      n:          number of independent observations (default 20)
      key_prefix: observer name prefix (default "sample" → "sample_0", "sample_1", ...)

    Returns:
      wrapped(params) -> dict  {f"{prefix}_{i}": float, ...}

    Example:
      noisy = lambda p: base_fn(p) + np.random.randn() * 0.3
      lad = wrap_multi_obs(noisy, n=20)
      r = mimir(lad, ranges, time_budget=300)
      print(r["stable_active"])   # robust across all 20 samples
      print(r["observer_dependent"])  # gen-dependent = noise-suspect dims
    """
    if n < 1:
        raise ValueError(f"wrap_multi_obs: n must be >= 1, got {n}")

    def wrapped(params):
        return {f"{key_prefix}_{i}": float(eval_fn(params)) for i in range(n)}
    wrapped.__name__ = f"wrap_multi_obs_{getattr(eval_fn, '__name__', 'fn')}_n{n}"
    return wrapped


def wrap_llm_judge(
    generator_fn: Callable,
    judge_fn: Callable,
    dimensions: Sequence[str] = ("fluency", "accuracy", "coherence"),
    n_generations: int = 5,
    aggregator: str = "mean",
) -> Callable:
    """LLM-assisted LaD wrapper: generate N times + judge each dimension.

    Each `wrapped(params)` call:
      1. Calls generator_fn(params) N times to produce N text samples
      2. For each (sample, dimension) pair, calls judge_fn(text, dim) → float
      3. Aggregates within-dimension across samples (mean default)
      4. Returns dict {dim: aggregated_score} for multi-observer analysis

    Args:
      generator_fn: f(params) -> str  (LLM generation, stochastic)
      judge_fn:     f(text, dimension) -> float  (LLM judge, returns score 0-1)
      dimensions:   tuple of str dimension names
      n_generations: how many times to call generator_fn per params
      aggregator:   "mean" (default) / "median" / "min" / "max"

    Returns:
      wrapped(params) -> dict  {dim_name: float, ...}
      mimir treats these as multi-observer → stable_active / observer_dependent.

    Example:
      def gen(p):
          return llm.generate(prompt_template.format(**p_to_kwargs(p)),
                              temperature=p[0], top_p=p[1])
      def judge(text, dim):
          j = llm.generate(f"Rate this {dim} 0-1: {text}")
          return float(j.strip())

      lad = wrap_llm_judge(gen, judge,
                            dimensions=["fluency", "accuracy", "safety"],
                            n_generations=5)
      r = mimir(lad, param_ranges, time_budget=3600)
      # r["stable_active"] = dims robustly affecting all 3 dimensions
      # r["observer_dependent"] = dims only affecting one dimension (trade-off)
    """
    if n_generations < 1:
        raise ValueError(f"wrap_llm_judge: n_generations must be >= 1")
    if not dimensions:
        raise ValueError("wrap_llm_judge: dimensions must be non-empty")

    def wrapped(params):
        texts = [generator_fn(params) for _ in range(n_generations)]
        out = {}
        for dim in dimensions:
            scores = []
            for t in texts:
                try:
                    scores.append(float(judge_fn(t, dim)))
                except Exception:
                    # Judge failure → skip this sample, don't fail whole wrap
                    continue
            if not scores:
                # All judges failed — assign 0 so mimir doesn't propagate error
                out[str(dim)] = 0.0
            else:
                out[str(dim)] = _aggregate(scores, aggregator)
        return out
    wrapped.__name__ = f"wrap_llm_judge_{'_'.join(dimensions)}_n{n_generations}"
    return wrapped


__all__ = ["wrap_stochastic", "wrap_multi_obs", "wrap_llm_judge"]
