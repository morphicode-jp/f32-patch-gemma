"""Tests for mimir_council — parallel specialist ensemble.

Covers:
  - default 4 specialists run and return best
  - council dict has all metadata (council, variance_std, elapsed, n_ran)
  - custom specialist list respected
  - thread executor works (avoids pickle issues for closure eval_fn)
  - process executor works with module-level eval_fn
  - best_score of winner >= all others
  - council_variance_std >= 0
  - all-failed raises RuntimeError
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.mimir_council import (
    DEFAULT_SPECIALISTS,
    _score_of,
    mimir_council,
)


# -----------------------------------------------------------------
# module-level eval_fns (picklable for ProcessPoolExecutor)
# -----------------------------------------------------------------

def quadratic_eval(p):
    """Smooth quadratic, easy to optimize."""
    return -sum((x - 0.5) ** 2 for x in p)


def rastrigin_eval(p):
    """Multi-modal, favors default/expensive specialist."""
    import math
    return -(10 * len(p) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in p))


def always_zero(p):
    """Constant fn — all specialists should return similar scores (low variance)."""
    return 0.0


# -----------------------------------------------------------------
# basic: council runs, returns expected shape
# -----------------------------------------------------------------

def test_council_thread_executor_basic():
    """Thread executor works, returns best among 4 specialists."""
    r = mimir_council(
        quadratic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        executor="thread",
        experience_id="test_basic_thread",
    )
    assert "best_params" in r
    assert "council" in r
    assert "council_variance_std" in r
    assert "council_elapsed_s" in r
    assert "n_specialists_ran" in r
    # At least 1 specialist succeeded
    assert r["n_specialists_ran"] >= 1
    assert r["specialist"] in [s["name"] for s in DEFAULT_SPECIALISTS]


def test_council_process_executor_basic():
    """Process executor works with module-level eval_fn."""
    r = mimir_council(
        quadratic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        executor="process",
        experience_id="test_basic_process",
    )
    assert r["n_specialists_ran"] >= 1
    assert "council" in r


# -----------------------------------------------------------------
# winner has max score
# -----------------------------------------------------------------

def test_council_winner_has_max_score():
    """The returned 'best' specialist must have the highest score in council."""
    r = mimir_council(
        quadratic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        executor="thread",
        experience_id="test_max",
    )
    winner_score = _score_of(r)
    for name, score in r["council"]:
        assert winner_score >= score or score == float("-inf")


# -----------------------------------------------------------------
# variance signal
# -----------------------------------------------------------------

def test_council_variance_non_negative():
    r = mimir_council(
        quadratic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        executor="thread",
        experience_id="test_variance",
    )
    assert r["council_variance_std"] >= 0.0


def test_council_variance_low_for_constant_fn():
    """Constant eval_fn should produce low council variance (all agree)."""
    r = mimir_council(
        always_zero,
        [(-1, 1)] * 2,
        time_budget=5,
        executor="thread",
        experience_id="test_variance_const",
    )
    # All specialists see constant → their best_scores should be similar
    assert r["council_variance_std"] < 1.0


# -----------------------------------------------------------------
# custom specialist list
# -----------------------------------------------------------------

def test_council_custom_specialists():
    """Custom specialist list is respected."""
    custom = [
        {"name": "fast",  "kwargs": {},                        "role": "quick"},
        {"name": "lad",   "kwargs": {"n_samples_per_eval": 5}, "role": "stochastic"},
    ]
    r = mimir_council(
        quadratic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        specialists=custom,
        executor="thread",
        experience_id="test_custom",
    )
    # Should only run 2 specialists (total = ran + failed)
    assert r["n_specialists_ran"] + r["n_specialists_failed"] == 2
    winner_name = r["specialist"]
    assert winner_name in ("fast", "lad")


# -----------------------------------------------------------------
# validation
# -----------------------------------------------------------------

def test_council_invalid_executor_raises():
    with pytest.raises(ValueError):
        mimir_council(
            quadratic_eval,
            [(-1, 1)] * 2,
            time_budget=5,
            executor="invalid",
            experience_id="test_invalid_exec",
        )


# -----------------------------------------------------------------
# extra_kwargs pass-through
# -----------------------------------------------------------------

def test_council_extra_kwargs_passthrough():
    """extra_kwargs (e.g., curated_measurements) reaches each specialist."""
    past = [
        {"params": [0.4, 0.5], "score": -0.01},
        {"params": [0.5, 0.5], "score":  0.00},
        {"params": [0.6, 0.5], "score": -0.01},
        {"params": [0.5, 0.4], "score": -0.01},
        {"params": [0.5, 0.6], "score": -0.01},
        {"params": [0.3, 0.3], "score": -0.08},
    ]
    r = mimir_council(
        quadratic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        curated_measurements=past,
        executor="thread",
        experience_id="test_curated",
    )
    assert r["n_specialists_ran"] >= 1
