"""Smoke tests for brain_sentinel.py — Sentinel x Brain Evolution.

Fast tests (no Sentinel run): verify eval_fn/guard_fn builders return valid
scores. Slow integration test is gated by @pytest.mark.slow.
"""

import os
import sys
import pytest
import numpy as np

# Add project root to path so `import brain_sentinel` works
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from brain_sentinel import (  # noqa: E402
    make_flyworld_eval,
    make_inhibition_guard,
    make_structural_guard,
    make_iss_eval,
    _get_mode_fns,
    _interpret_dim,
)
from kathara_brain_sim_v8 import FlyWorldV1, FlyWorldV3, PARAM_RANGES  # noqa: E402


@pytest.fixture
def baseline_params():
    """Midpoint of all 91 parameter ranges."""
    return [(lo + hi) / 2.0 for lo, hi in PARAM_RANGES]


@pytest.fixture
def random_params():
    """Random point in parameter space."""
    rng = np.random.RandomState(42)
    return [rng.uniform(lo, hi) for lo, hi in PARAM_RANGES]


class TestEvalFnBuilders:
    """Each builder must produce a callable returning float in [0, 100]."""

    def test_flyworld_v1_eval(self, baseline_params):
        fn = make_flyworld_eval(FlyWorldV1, n_episodes=2, n_steps=20)
        s = fn(baseline_params)
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0, f"score {s} out of [0, 100]"

    def test_flyworld_v3_eval(self, baseline_params):
        fn = make_flyworld_eval(FlyWorldV3, n_episodes=2, n_steps=20)
        s = fn(baseline_params)
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0

    def test_flyworld_v3_predator_eval(self, baseline_params):
        fn = make_flyworld_eval(FlyWorldV3, has_predator=True,
                                n_episodes=2, n_steps=20)
        s = fn(baseline_params)
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0

    def test_inhibition_guard(self, baseline_params):
        fn = make_inhibition_guard(n_episodes=2, n_steps=15)
        s = fn(baseline_params)
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0

    def test_structural_guard(self, baseline_params):
        """Structural guard (compute_iss_from_firing) returns non-zero baseline."""
        fn = make_structural_guard(n_episodes=2, n_steps=15)
        s = fn(baseline_params)
        assert isinstance(s, float)
        assert 0.0 < s <= 100.0, f"structural guard should be >0 at baseline, got {s}"

    def test_iss_eval(self, baseline_params):
        fn = make_iss_eval(n_episodes=2, n_steps=15)
        s = fn(baseline_params)
        assert isinstance(s, float)
        assert 0.0 <= s <= 100.0

    def test_eval_accepts_list_not_array(self, baseline_params):
        """Sentinel calls with list[float]. eval_fn must accept it."""
        assert isinstance(baseline_params, list)
        fn = make_flyworld_eval(FlyWorldV1, n_episodes=1, n_steps=10)
        s = fn(baseline_params)  # list input
        assert isinstance(s, float)

    def test_eval_is_deterministic(self, baseline_params):
        """Same params + same seeds -> same score."""
        fn = make_flyworld_eval(FlyWorldV1, n_episodes=2, n_steps=15, seed_offset=99)
        s1 = fn(baseline_params)
        s2 = fn(baseline_params)
        assert abs(s1 - s2) < 1e-9, f"non-deterministic: {s1} != {s2}"

    def test_eval_varies_with_params(self, baseline_params, random_params):
        """Different params should produce different scores (usually)."""
        fn = make_flyworld_eval(FlyWorldV1, n_episodes=2, n_steps=20)
        s1 = fn(baseline_params)
        s2 = fn(random_params)
        assert s1 != s2, "scores should differ for different params"


class TestModeDispatch:
    """Mode dispatcher produces valid (eval_fn, guard_fn, description)."""

    @pytest.mark.parametrize("mode", ["biological", "progressive", "intelligent"])
    def test_mode_builds(self, mode):
        eval_fn, guard_fn, desc = _get_mode_fns(mode)
        assert callable(eval_fn)
        assert callable(guard_fn)
        assert isinstance(desc, str) and len(desc) > 10

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            _get_mode_fns("nonsense")

    def test_biological_mode_callable(self, baseline_params):
        eval_fn, guard_fn, _ = _get_mode_fns("biological")
        e = eval_fn(baseline_params)
        g = guard_fn(baseline_params)
        assert isinstance(e, float) and 0.0 <= e <= 100.0
        assert isinstance(g, float) and 0.0 <= g <= 100.0


class TestDimInterpretation:
    """_interpret_dim maps dim index to human-readable brain part."""

    def test_edge_dims(self):
        assert "Edge 0" in _interpret_dim(0)
        assert "Edge 29" in _interpret_dim(29)

    def test_node_dims(self):
        # dim 30 = node 0, param 0 = gain
        assert "Node 0" in _interpret_dim(30) and "gain" in _interpret_dim(30)
        # dim 35 = node 1, param 0 = gain
        assert "Node 1" in _interpret_dim(35) and "gain" in _interpret_dim(35)
        # dim 89 = node 11, param 4 = threshold
        assert "Node 11" in _interpret_dim(89)
        assert "threshold" in _interpret_dim(89)

    def test_n_steps_dim(self):
        assert _interpret_dim(90) == "n_steps"

    def test_named_nodes(self):
        # dim 30 + 5*5 = 55 = node 5 = "navigation"
        assert "navigation" in _interpret_dim(55)
        # dim 30 + 7*5 = 65 = node 7 = "left_vis"
        assert "left_vis" in _interpret_dim(65)


class TestSentinelIntegration:
    """End-to-end Sentinel run with a short budget."""

    @pytest.mark.slow
    def test_biological_mode_smoke(self):
        """~90s: verify Sentinel returns valid verdict on brain sim."""
        from brain_sentinel import run_mode
        result = run_mode("biological", time_budget=90)
        assert result["verdict"] in ("approved", "pivoted", "failed")
        assert len(result["best_params"]) == 91
        assert isinstance(result["eval_score"], float)
        assert isinstance(result["guard_score"], float)
