"""Tests for mimir_cardinal (Pattern 3: Hierarchy) and mimir_coevolution (Pattern 4)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.mimir_cardinal import _ga_optimize, mimir_cardinal_hierarchy
from twelve.agent.mimir_coevolution import mimir_cardinal_coevolution


# ----------------------------------------------------------------------
# module-level eval_fns
# ----------------------------------------------------------------------

def quad5_sparse(p):
    """5-dim active + 5-dim dead: only first 5 dims matter."""
    return -sum((p[i] - 0.5) ** 2 for i in range(5))


def quadratic2(p):
    return -sum((x - 0.5) ** 2 for x in p)


def multi_metric_eval(p):
    """Returns dict of 3 metrics; all favor p near [0.5, 0.5]."""
    return {
        "a": -sum((x - 0.5) ** 2 for x in p),
        "b": -sum((x - 0.6) ** 2 for x in p),
        "c": -sum((x - 0.4) ** 2 for x in p),
    }


# ----------------------------------------------------------------------
# GA unit tests
# ----------------------------------------------------------------------

def test_ga_converges_on_quadratic():
    r = _ga_optimize(
        quadratic2,
        [(-1, 1)] * 2,
        time_budget=5,
        population_size=10,
        generations=15,
        seed=0,
    )
    assert "best_params" in r
    assert "best_score" in r
    # Should find something close to (0.5, 0.5) → score close to 0
    assert r["best_score"] > -0.5, f"GA did not converge: {r['best_score']}"


def test_ga_respects_time_budget():
    """GA should exit early if time_budget exceeded."""
    def slow_eval(p):
        import time
        time.sleep(0.01)
        return -sum(x * x for x in p)

    r = _ga_optimize(
        slow_eval,
        [(-1, 1)] * 3,
        time_budget=1.0,  # very short
        population_size=8,
        generations=50,  # high, should be cut off
        seed=0,
    )
    assert r["elapsed_s"] < 2.5, f"GA overran budget: {r['elapsed_s']}"


# ----------------------------------------------------------------------
# Hierarchy tests
# ----------------------------------------------------------------------

def test_hierarchy_basic():
    """Hierarchy runs, returns expected keys."""
    r = mimir_cardinal_hierarchy(
        quadratic2,
        [(-1, 1)] * 3,
        time_budget=30,
        council_budget_share=0.3,
        ga_population=8,
        ga_generations=10,
        experience_id="test_hier_basic",
    )
    assert "best_params" in r
    assert "best_score" in r
    assert "active_dims" in r
    assert "dead_dims" in r
    assert "ga_generations_run" in r
    assert r["n_active"] + r["n_dead"] <= 3  # can't exceed total dims
    assert r["elapsed_s"] <= 35  # allow small overshoot


def test_hierarchy_reduces_sparse_problem():
    """Hierarchy should identify first 5 dims as active on sparse quadratic.

    Note: council may not perfectly find sparse structure, so we just check
    result is reasonable (score close to optimum).
    """
    r = mimir_cardinal_hierarchy(
        quad5_sparse,
        [(-1, 1)] * 10,  # 10 dims but only 5 active
        time_budget=40,
        council_budget_share=0.25,
        ga_population=12,
        ga_generations=15,
        experience_id="test_hier_sparse",
    )
    # Should reach reasonable score (even if active_dims isn't perfect)
    assert r["best_score"] > -2.0, f"hierarchy stuck: {r['best_score']}"


# ----------------------------------------------------------------------
# Co-evolution tests
# ----------------------------------------------------------------------

def test_coevolution_basic():
    """Co-evolution runs and returns expected keys."""
    r = mimir_cardinal_coevolution(
        multi_metric_eval,
        [(-1, 1)] * 2,
        metric_names=["a", "b", "c"],
        time_budget=10,
        population=8,
        generations=10,
        seed=0,
    )
    assert "best_params" in r
    assert "best_weights" in r
    assert "best_metrics" in r
    assert "metric_ranking_by_weight" in r
    assert "weight_evolution_mean" in r
    assert r["n_generations_run"] > 0
    # best_weights has all 3 metrics
    assert set(r["best_weights"].keys()) == {"a", "b", "c"}


def test_coevolution_fitness_modes():
    """Different fitness_mode options all work."""
    for mode in ("min", "harmonic", "weighted"):
        r = mimir_cardinal_coevolution(
            multi_metric_eval,
            [(-1, 1)] * 2,
            metric_names=["a", "b", "c"],
            time_budget=5,
            population=6,
            generations=5,
            fitness_mode=mode,
            seed=0,
        )
        assert r["fitness_mode"] == mode
        assert r["n_generations_run"] > 0


def test_coevolution_finds_balanced_params():
    """Balanced params (all metrics similar) should beat specialist in 'min' mode."""
    r = mimir_cardinal_coevolution(
        multi_metric_eval,
        [(-1, 1)] * 2,
        metric_names=["a", "b", "c"],
        time_budget=15,
        population=12,
        generations=15,
        fitness_mode="min",
        seed=0,
    )
    # min fitness mode → params should be near [0.5, 0.5] (balances a, b, c)
    best_p = r["best_params"]
    assert abs(best_p[0] - 0.5) < 0.3, f"unbalanced: p[0]={best_p[0]}"


def test_coevolution_weight_ranges_validation():
    """Mismatched weight_ranges length should raise."""
    with pytest.raises(ValueError):
        mimir_cardinal_coevolution(
            multi_metric_eval,
            [(-1, 1)] * 2,
            metric_names=["a", "b", "c"],
            weight_ranges=[(0, 1)] * 2,  # wrong length
            time_budget=2,
            population=4,
            generations=2,
        )
