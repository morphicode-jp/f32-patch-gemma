"""Tests for hagen (覇玄) — the owl/Reigen meta-dispatcher.

hagen routes problems to owl or cascades owl→Reigen. These tests verify:
  - Basic smoke (returns sensible result)
  - Expensive eval routes to owl-only (no Reigen escalation)
  - Curated measurements provided → owl single-shot path
  - guard_fn is passed through
  - Return dict includes structure-discovery info from owl
  - Total wall time stays within a reasonable multiple of time_budget
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.hagen import hagen


# -----------------------------------------------------------------
# smoke test
# -----------------------------------------------------------------

def test_hagen_basic_smoke():
    """hagen returns best_params + best_score on a trivial quadratic."""
    def quad(p):
        return -sum((x - 0.5) ** 2 for x in p)
    r = hagen(
        eval_fn=quad,
        param_ranges=[(-1, 1)] * 3,
        time_budget=10,
        experience_id="test_hagen_smoke",
    )
    assert r["best_params"] is not None
    assert r["best_score"] is not None
    assert r["best_score"] > -1.0  # should reach quadratic minimum easily
    assert r["tool_used"] in (
        "owl", "owl+reigen", "owl(reigen_tried)", "expensive_single")


# -----------------------------------------------------------------
# expensive-eval routing: skip Reigen escalation
# -----------------------------------------------------------------

def test_hagen_expensive_eval_skips_reigen():
    """eval_cost_hint > 0.5s routes to owl single-shot, no Reigen call."""
    def quad(p):
        return -sum(x * x for x in p)
    r = hagen(
        eval_fn=quad,
        param_ranges=[(-1, 1)] * 2,
        time_budget=10,
        eval_cost_hint=1.0,  # explicitly mark as expensive
        experience_id="test_hagen_expensive",
    )
    assert r["tool_used"] == "owl"
    assert r["route"] == "expensive_single"
    # No reigen_result key when not escalated
    assert "reigen_result" not in r


# -----------------------------------------------------------------
# curated measurements path
# -----------------------------------------------------------------

def test_hagen_curated_measurements_single_shot():
    """curated_measurements provided triggers expensive route (owl only)."""
    def quad(p):
        return -sum(x * x for x in p)
    curated = [
        {"params": [0.1, -0.1], "score": quad([0.1, -0.1])},
        {"params": [0.2, 0.3], "score": quad([0.2, 0.3])},
        {"params": [-0.1, 0.2], "score": quad([-0.1, 0.2])},
        {"params": [0.5, 0.5], "score": quad([0.5, 0.5])},
        {"params": [-0.5, -0.5], "score": quad([-0.5, -0.5])},
    ]
    r = hagen(
        eval_fn=quad,
        param_ranges=[(-1, 1)] * 2,
        curated_measurements=curated,
        time_budget=10,
        experience_id="test_hagen_curated",
    )
    assert r["tool_used"] == "owl"
    assert r["route"] == "expensive_single"


# -----------------------------------------------------------------
# guard_fn passthrough
# -----------------------------------------------------------------

def test_hagen_guard_fn_passthrough():
    """guard_fn is forwarded to owl; result contains best_params."""
    def eval_fn(p):
        return -sum((x - 0.3) ** 2 for x in p)
    def guard_fn(p):
        # penalize params outside small cube
        return -max(0.0, max(abs(x) for x in p) - 0.8)
    r = hagen(
        eval_fn=eval_fn,
        param_ranges=[(-1, 1)] * 2,
        guard_fn=guard_fn,
        time_budget=8,
        eval_cost_hint=1.0,  # keep it single-shot for deterministic test
        experience_id="test_hagen_guard",
    )
    assert r["best_params"] is not None


# -----------------------------------------------------------------
# return schema: structure info passed through
# -----------------------------------------------------------------

def test_hagen_returns_structure_info():
    """hagen's return dict includes owl's structure-discovery outputs."""
    def eval_fn(p):
        # Clear dead dim: p[2] has no effect
        return -(p[0] ** 2 + p[1] ** 2)
    r = hagen(
        eval_fn=eval_fn,
        param_ranges=[(-1, 1)] * 3,
        time_budget=8,
        eval_cost_hint=1.0,  # single-shot so owl runs fully
        experience_id="test_hagen_structure",
    )
    # Structure keys present (may be None if owl couldn't compute, but key must exist)
    for k in ("dead_dims", "active_dims", "fragility", "proxy_type", "proxy_r2"):
        assert k in r, f"expected key {k} in hagen result"
    # owl_result always present
    assert "owl_result" in r


# -----------------------------------------------------------------
# tool_used reflects actual cascade decision
# -----------------------------------------------------------------

def test_hagen_cascade_fires_for_cheap_eval():
    """Cheap eval without curated data triggers owl→Reigen cascade."""
    def eval_fn(p):
        return -sum((x - 0.5) ** 2 for x in p)
    r = hagen(
        eval_fn=eval_fn,
        param_ranges=[(-1, 1)] * 2,
        time_budget=15,
        # eval_cost auto-measured, will be <<0.5s for this cheap fn
        experience_id="test_hagen_cascade",
    )
    # Either full cascade OR reigen-tried; definitely not "owl" with route=expensive
    assert r["route"] == "cheap_cascade"
    assert r["tool_used"] in ("owl+reigen", "owl(reigen_tried)")


# -----------------------------------------------------------------
# wall time discipline
# -----------------------------------------------------------------

def test_hagen_wall_time_reasonable():
    """Total wall time stays within 3× time_budget (soft guarantee)."""
    def quad(p):
        return -sum(x * x for x in p)
    t0 = time.time()
    r = hagen(
        eval_fn=quad,
        param_ranges=[(-1, 1)] * 2,
        time_budget=6,
        experience_id="test_hagen_walltime",
    )
    wall = time.time() - t0
    assert wall < 6 * 3, (
        f"hagen wall {wall:.1f}s exceeded 3× budget 6s")
    assert r["elapsed_s"] <= wall + 0.1
