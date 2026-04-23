"""Tests for twelve.agent.lad_wrappers.

Covers:
  - wrap_stochastic: aggregates N calls, reduces variance vs single-call
  - wrap_multi_obs: returns dict with N observers
  - wrap_llm_judge: multi-dimensional score dict from mock generator+judge
  - validation: n < 1 rejected, empty dimensions rejected
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.lad_wrappers import (
    _aggregate,
    wrap_llm_judge,
    wrap_multi_obs,
    wrap_stochastic,
)


# -----------------------------------------------------------------
# _aggregate
# -----------------------------------------------------------------

def test_aggregate_median_noise_robust():
    """Median ignores outliers better than mean."""
    vals = [1.0, 1.0, 1.0, 1.0, 100.0]
    assert _aggregate(vals, "median") == 1.0
    assert _aggregate(vals, "mean") > 20.0  # dragged up by outlier


def test_aggregate_mean_min_max():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _aggregate(vals, "mean") == 3.0
    assert _aggregate(vals, "min") == 1.0
    assert _aggregate(vals, "max") == 5.0
    assert _aggregate(vals, "median") == 3.0


def test_aggregate_empty_returns_zero():
    assert _aggregate([], "median") == 0.0


# -----------------------------------------------------------------
# wrap_stochastic
# -----------------------------------------------------------------

def test_wrap_stochastic_reduces_variance():
    """Aggregating N noisy calls should give lower variance than single calls."""
    rng = random.Random(42)

    def noisy(p):
        return p[0] + rng.gauss(0, 1.0)

    wrapped = wrap_stochastic(noisy, n=30, agg="median")

    single_samples = [noisy([5.0]) for _ in range(20)]
    wrapped_samples = [wrapped([5.0]) for _ in range(20)]

    def var(vs):
        m = sum(vs) / len(vs)
        return sum((v - m) ** 2 for v in vs) / len(vs)

    assert var(wrapped_samples) < var(single_samples) * 0.5, (
        f"wrap_stochastic failed to reduce variance: "
        f"single={var(single_samples):.3f}, wrapped={var(wrapped_samples):.3f}"
    )


def test_wrap_stochastic_returns_scalar():
    def fn(p):
        return sum(p)

    wrapped = wrap_stochastic(fn, n=5)
    out = wrapped([1.0, 2.0])
    assert isinstance(out, float)


def test_wrap_stochastic_n_validation():
    def fn(p):
        return 0.0

    with pytest.raises(ValueError):
        wrap_stochastic(fn, n=0)


# -----------------------------------------------------------------
# wrap_multi_obs
# -----------------------------------------------------------------

def test_wrap_multi_obs_returns_dict_of_n():
    def fn(p):
        return p[0] * 2

    wrapped = wrap_multi_obs(fn, n=10)
    out = wrapped([3.0])
    assert isinstance(out, dict)
    assert len(out) == 10
    assert all(v == 6.0 for v in out.values())
    assert "sample_0" in out and "sample_9" in out


def test_wrap_multi_obs_custom_prefix():
    def fn(p):
        return 1.0

    wrapped = wrap_multi_obs(fn, n=3, key_prefix="gen")
    out = wrapped([0.0])
    assert set(out.keys()) == {"gen_0", "gen_1", "gen_2"}


def test_wrap_multi_obs_stochastic_preserves_variation():
    """Each key should get an independent draw — not all identical."""
    rng = random.Random(7)

    def noisy(p):
        return rng.random()

    wrapped = wrap_multi_obs(noisy, n=10)
    out = wrapped([0.0])
    assert len(set(out.values())) > 5  # most are distinct


def test_wrap_multi_obs_n_validation():
    def fn(p):
        return 0.0

    with pytest.raises(ValueError):
        wrap_multi_obs(fn, n=0)


# -----------------------------------------------------------------
# wrap_llm_judge
# -----------------------------------------------------------------

def test_wrap_llm_judge_mock_returns_dim_dict():
    """Mock generator+judge should produce one aggregated score per dimension."""
    def gen(p):
        return f"text_for_{p[0]}"

    # Judge returns different scores per dim for determinism check
    judge_map = {"fluency": 0.8, "accuracy": 0.6, "safety": 0.9}

    def judge(text, dim):
        return judge_map[dim]

    wrapped = wrap_llm_judge(
        gen, judge,
        dimensions=["fluency", "accuracy", "safety"],
        n_generations=3,
    )
    out = wrapped([0.5])
    assert isinstance(out, dict)
    assert set(out.keys()) == {"fluency", "accuracy", "safety"}
    assert out["fluency"] == pytest.approx(0.8)
    assert out["accuracy"] == pytest.approx(0.6)
    assert out["safety"] == pytest.approx(0.9)


def test_wrap_llm_judge_judge_exception_skipped():
    """If judge_fn raises for some samples, those are dropped, not whole wrap."""
    def gen(p):
        return "text"

    call_count = {"n": 0}

    def judge(text, dim):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            raise RuntimeError("judge transient error")
        return 0.7

    wrapped = wrap_llm_judge(gen, judge, dimensions=["q"], n_generations=4)
    out = wrapped([0.0])
    # At least one judge succeeded per dim → not 0.0 fallback
    assert out["q"] == pytest.approx(0.7)


def test_wrap_llm_judge_all_judge_fail_gives_zero():
    def gen(p):
        return "text"

    def judge(text, dim):
        raise RuntimeError("always fails")

    wrapped = wrap_llm_judge(gen, judge, dimensions=["x"], n_generations=2)
    out = wrapped([0.0])
    assert out["x"] == 0.0


def test_wrap_llm_judge_validation():
    def gen(p):
        return ""

    def judge(t, d):
        return 0.0

    with pytest.raises(ValueError):
        wrap_llm_judge(gen, judge, dimensions=["x"], n_generations=0)
    with pytest.raises(ValueError):
        wrap_llm_judge(gen, judge, dimensions=(), n_generations=3)
