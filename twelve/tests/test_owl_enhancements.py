"""Tests for owl() enhancements: empty-data start, curated_measurements alias,
guard_fn / safe_dim_analysis. These extract the value from Sentinel back into
owl directly (2026-04-19 architectural refactor).
"""
import os
import sys
import random
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.optimize import owl


# -----------------------------------------------------------------
# Empty-data autonomous start
# -----------------------------------------------------------------

def test_empty_measurements_with_verify_and_ranges_seeds():
    """len(measurements)==0 + verify_fn + param_ranges → auto-seed via verify_fn."""
    calls = {"n": 0}
    def vfn(p):
        calls["n"] += 1
        return -sum((x - 0.5) ** 2 for x in p)
    r = owl(
        measurements=[],
        param_ranges=[(-1.0, 1.0)] * 2,
        verify_fn=vfn,
        n_seed_samples=6,
        n_rounds=1,
        min_r_squared=0.01,
        verbose=False,
    )
    # verify_fn was called 6 times for seeding + possibly once for verification
    assert calls["n"] >= 6
    # n_measurements in result should reflect seeds (and any verify calls)
    assert r.get("n_measurements", 0) >= 6


def test_empty_measurements_raises_without_verify_fn():
    """Empty data without verify_fn or ranges must raise clearly."""
    with pytest.raises(ValueError, match="empty measurements requires"):
        owl(measurements=[], param_ranges=[(-1, 1)])
    with pytest.raises(ValueError, match="empty measurements requires"):
        owl(measurements=[], verify_fn=lambda p: 0.0)


def test_empty_measurements_autonomous_grows():
    """Empty start + autonomous mode should grow the dataset materially."""
    r = owl(
        measurements=[],
        param_ranges=[(-1, 1)] * 2,
        verify_fn=lambda p: -sum((x - 0.5) ** 2 for x in p),
        n_seed_samples=10,
        autonomous=True,
        max_iterations=3,
        verbose=False,
    )
    # Seed 10 + autonomous growth (3 iterations × up to 3 neighbors + 3 verifies)
    assert r.get("n_measurements", 0) >= 12


def test_seed_rng_state_reproducible():
    """Same seed_rng_state → same seed points (bit-identical)."""
    vfn = lambda p: -sum(x * x for x in p)
    r1 = owl([], param_ranges=[(-1, 1)] * 2, verify_fn=vfn,
             n_seed_samples=5, n_rounds=1, min_r_squared=0.01,
             seed_rng_state=7, verbose=False)
    r2 = owl([], param_ranges=[(-1, 1)] * 2, verify_fn=vfn,
             n_seed_samples=5, n_rounds=1, min_r_squared=0.01,
             seed_rng_state=7, verbose=False)
    # same best_score (deterministic seed)
    assert r1.get("best_score") == r2.get("best_score") or \
           (r1.get("best_score") is None and r2.get("best_score") is None)


# -----------------------------------------------------------------
# curated_measurements explicit alias
# -----------------------------------------------------------------

def test_curated_measurements_alias_used_when_measurements_none():
    data = [{"params": [i * 0.1, i * 0.1], "score": float(i)} for i in range(10)]
    r = owl(
        measurements=None,
        curated_measurements=data,
        param_ranges=[(0.0, 1.0)] * 2,
        n_rounds=1,
        min_r_squared=0.01,
        verbose=False,
    )
    assert r.get("n_measurements") == 10


def test_explicit_measurements_win_over_curated():
    """If both given, measurements wins (curated is just alias)."""
    explicit = [{"params": [i * 0.1, i * 0.1], "score": float(i)} for i in range(8)]
    curated = [{"params": [0, 0], "score": 0.0}] * 25  # different size & shape
    r = owl(
        measurements=explicit,
        curated_measurements=curated,
        param_ranges=[(0.0, 1.0)] * 2,
        n_rounds=1,
        min_r_squared=0.01,
        verbose=False,
    )
    assert r.get("n_measurements") == 8  # not 25


# -----------------------------------------------------------------
# guard_fn + safe_dim_analysis
# -----------------------------------------------------------------

def _make_data(n=25, seed=42):
    random.seed(seed)
    return [
        {"params": [random.uniform(0, 1), random.uniform(0, 1)],
         "score": -sum((x - 0.5) ** 2 for x in [random.uniform(0, 1), random.uniform(0, 1)])}
        for _ in range(n)
    ]


def test_guard_fn_adds_guard_keys():
    """guard_fn present → result has guard_score, baseline_guard, guard_verdict."""
    data = _make_data()
    r = owl(
        measurements=data,
        param_ranges=[(0.0, 1.0)] * 2,
        guard_fn=lambda p: -abs(p[0] - p[1]),
        n_rounds=1,
        min_r_squared=0.01,
    )
    if r.get("best_params") is not None:
        assert "guard_score" in r
        assert "baseline_guard" in r
        assert "guard_verdict" in r
        assert r["guard_verdict"] in ("approved", "failed")


def test_no_guard_fn_means_no_guard_keys():
    """Backward compat: result has NO guard_* keys if guard_fn=None."""
    data = _make_data()
    r = owl(
        measurements=data,
        param_ranges=[(0.0, 1.0)] * 2,
        n_rounds=1,
        min_r_squared=0.01,
    )
    assert "guard_score" not in r
    assert "baseline_guard" not in r
    assert "guard_verdict" not in r


def test_guard_threshold_overrides_midpoint_baseline():
    """Explicit guard_threshold bypasses midpoint evaluation."""
    data = _make_data()
    r = owl(
        measurements=data,
        param_ranges=[(0.0, 1.0)] * 2,
        guard_fn=lambda p: -abs(p[0] - p[1]),
        guard_threshold=-999.0,  # impossibly low → always approved
        n_rounds=1,
        min_r_squared=0.01,
    )
    if r.get("best_params") is not None:
        assert r["baseline_guard"] == -999.0
        assert r["guard_verdict"] == "approved"


def test_safe_dim_analysis_surfaces_safe_dims_on_guard_failure():
    """guard_fn fails + safe_dim_analysis=True → safe_dims in result."""
    data = _make_data()
    r = owl(
        measurements=data,
        param_ranges=[(0.0, 1.0)] * 2,
        guard_fn=lambda p: -p[0] * 10.0,  # big penalty on p0
        safe_dim_analysis=True,
        n_rounds=1,
        min_r_squared=0.01,
    )
    if r.get("best_params") is not None and r.get("guard_verdict") == "failed":
        assert "safe_dims" in r


def test_guard_fn_error_swallowed():
    """guard_fn that raises must not crash owl — logged in verbose only."""
    data = _make_data()

    def flaky_guard(p):
        raise RuntimeError("boom")

    r = owl(
        measurements=data,
        param_ranges=[(0.0, 1.0)] * 2,
        guard_fn=flaky_guard,
        n_rounds=1,
        min_r_squared=0.01,
        verbose=False,
    )
    # Should still return a result (guard_* keys may or may not appear)
    assert r is not None
    assert "n_measurements" in r


# -----------------------------------------------------------------
# Signature compatibility — no breaking changes
# -----------------------------------------------------------------

def test_existing_owl_call_pattern_still_works():
    """Plain owl(data) call (no new kwargs) works like before."""
    data = _make_data(n=15)
    r = owl(measurements=data, param_ranges=[(0.0, 1.0)] * 2,
            n_rounds=1, min_r_squared=0.01, verbose=False)
    # Existing schema preserved
    assert "n_measurements" in r
    assert "active_dims" in r
    assert "dead_dims" in r
    assert "proxy_r2" in r


def test_positional_measurements_still_works():
    """Positional call `owl(data)` must still work (backward compat)."""
    data = _make_data(n=15)
    r = owl(data, param_ranges=[(0.0, 1.0)] * 2, n_rounds=1, min_r_squared=0.01)
    assert r.get("n_measurements") == 15
