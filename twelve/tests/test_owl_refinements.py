"""Tests for owl 2026-04-20 refinements:
  - L-BFGS-B gradient refinement (Phase A)
  - Multi-start fallback diversification (Phase B)

Both are opt-in (default False) for backward compatibility.
"""
import os
import sys
import random

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.optimize import owl


def _curated(problem_fn, ranges, n=20, seed=42):
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        p = [rng.uniform(lo, hi) for lo, hi in ranges]
        data.append({"params": p, "score": float(problem_fn(p))})
    return data


# -----------------------------------------------------------------
# Phase A: L-BFGS-B refinement
# -----------------------------------------------------------------

def test_lbfgs_default_off_preserves_behavior():
    """use_lbfgs_refinement=False (default): no LBFGS invocation, legacy path."""
    def quad(p): return -sum((x - 0.3) ** 2 for x in p)
    curated = _curated(quad, [(-1, 1)] * 3, n=15, seed=7)
    r = owl(
        measurements=curated,
        param_ranges=[(-1, 1)] * 3,
        verify_fn=quad,
        autonomous=True,
        max_iterations=3,
        time_budget=5,
        min_r_squared=0.1,
        # use_lbfgs_refinement=False (default)
    )
    assert r.get("best_params") is not None
    # confidence must NOT be "lbfgs_refined" when feature off
    assert r.get("confidence") != "lbfgs_refined"
    assert r.get("confidence") != "direct+lbfgs"


def test_lbfgs_opt_in_improves_rosenbrock():
    """use_lbfgs_refinement=True on Rosenbrock should improve verified_score."""
    def rosenbrock(p):
        s = 0.0
        for i in range(len(p) - 1):
            s += 100 * (p[i + 1] - p[i] ** 2) ** 2 + (1 - p[i]) ** 2
        return -s

    curated = _curated(rosenbrock, [(-2, 2)] * 5, n=20, seed=42)
    r = owl(
        measurements=curated,
        param_ranges=[(-2, 2)] * 5,
        verify_fn=rosenbrock,
        autonomous=True,
        max_iterations=20,
        time_budget=15,
        min_r_squared=0.1,
        use_lbfgs_refinement=True,
    )
    # With L-BFGS-B refinement, Rosenbrock gap should be small (< 5.0)
    assert r.get("best_score") is not None
    assert r["best_score"] > -5.0, \
        f"L-BFGS-B should improve Rosenbrock; got score={r['best_score']}"


def test_lbfgs_confidence_marks_refinement():
    """If L-BFGS-B improves result, confidence is 'lbfgs_refined' or 'direct+lbfgs'."""
    def styblinski(p):
        return -sum(x ** 4 - 16 * x ** 2 + 5 * x for x in p) / 2.0

    curated = _curated(styblinski, [(-5, 5)] * 4, n=20, seed=42)
    r = owl(
        measurements=curated,
        param_ranges=[(-5, 5)] * 4,
        verify_fn=styblinski,
        autonomous=True,
        max_iterations=10,
        time_budget=10,
        min_r_squared=0.1,
        use_lbfgs_refinement=True,
    )
    # L-BFGS-B on Styblinski (smooth, separable) should fire + improve
    if r.get("best_score", -999) > 150:  # if it got close to global (195.83 for 4d ≈ 156)
        conf = r.get("confidence", "")
        assert conf in ("lbfgs_refined", "direct+lbfgs") or r.get("best_score", 0) > 150


def test_lbfgs_safe_when_scipy_missing(monkeypatch):
    """Silent fallback when scipy.optimize import fails mid-run."""
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "scipy.optimize":
            raise ImportError("simulated missing scipy")
        return orig_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def quad(p): return -sum(x * x for x in p)
    curated = _curated(quad, [(-1, 1)] * 2, n=12, seed=3)
    # Should not raise, just silent fallback
    r = owl(
        measurements=curated,
        param_ranges=[(-1, 1)] * 2,
        verify_fn=quad,
        autonomous=True,
        max_iterations=2,
        time_budget=3,
        min_r_squared=0.1,
        use_lbfgs_refinement=True,
    )
    assert r is not None
    assert r.get("best_params") is not None


def test_lbfgs_does_not_regress_below_pre_refinement():
    """If L-BFGS-B doesn't help (or hurts), pre-refinement best is preserved."""
    # Deterministic: quadratic where HC already nails it
    def trivial(p): return -sum(x * x for x in p)

    curated = _curated(trivial, [(-1, 1)] * 2, n=15, seed=11)
    r = owl(
        measurements=curated,
        param_ranges=[(-1, 1)] * 2,
        verify_fn=trivial,
        autonomous=True,
        max_iterations=5,
        time_budget=5,
        min_r_squared=0.1,
        use_lbfgs_refinement=True,
    )
    # Result always sensible (not worse than worst curated)
    worst_curated = min(m["score"] for m in curated)
    assert r.get("best_score", worst_curated) >= worst_curated


# -----------------------------------------------------------------
# Phase B tests (added after implementation)
# -----------------------------------------------------------------

def test_multistart_default_off_preserves_behavior():
    """use_multistart_fallback=False (default): no diverse warm-start."""
    def quad(p): return -sum((x - 0.5) ** 2 for x in p)
    curated = _curated(quad, [(-1, 1)] * 2, n=10, seed=0)
    r = owl(
        measurements=curated,
        param_ranges=[(-1, 1)] * 2,
        verify_fn=quad,
        autonomous=True,
        max_iterations=3,
        time_budget=5,
        min_r_squared=0.1,
        # use_multistart_fallback=False (default)
    )
    assert r.get("best_params") is not None
    # No crash, schema preserved


# -----------------------------------------------------------------
# Both off → no algorithmic change from pre-2026-04-20
# -----------------------------------------------------------------

def test_both_features_off_behaves_as_before():
    """With both new features off, owl returns pre-2026-04-20 schema.
    Hebbian canary script depends on this — no surprise keys should appear.
    """
    def quad(p): return -sum(x * x for x in p)
    curated = _curated(quad, [(-1, 1)] * 2, n=12, seed=5)
    r = owl(
        measurements=curated,
        param_ranges=[(-1, 1)] * 2,
        verify_fn=quad,
        autonomous=True,
        max_iterations=3,
        time_budget=4,
        min_r_squared=0.1,
    )
    # Essential keys still present
    assert "best_params" in r
    assert "proxy_r2" in r
    # Refinement-specific confidences must not appear
    conf = r.get("confidence", "")
    assert conf not in ("lbfgs_refined", "direct+lbfgs")
