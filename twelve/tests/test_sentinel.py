"""Sentinel テストスイート。

pytest -v twelve/tests/test_sentinel.py で実行。
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "c:/新しいフォルダー (2)/パイプラインオートメーション")

from twelve.agent.sentinel import Sentinel


class TestSentinelBasic:
    def test_smoke(self):
        """全フィールドが返る。"""
        def fn(p):
            return -sum((x - 2) ** 2 for x in p)

        r = Sentinel(eval_fn=fn, guard_fn=fn,
                     param_ranges=[(-5, 5)] * 3).run(time_budget=10, verbose=False)

        for key in ["verdict", "best_params", "eval_score", "guard_score",
                    "baseline_guard", "proxy_r2", "elapsed_s",
                    "safe_dims", "conflict_dims", "multi_observer"]:
            assert key in r, f"Missing: {key}"
        assert r["verdict"] in ("approved", "pivoted", "failed")

    def test_approved(self):
        """eval_fn=guard_fn、明確なシグナル → approved。"""
        def fn(p):
            return -((p[0] - 3) ** 2 + (p[1] - 3) ** 2)

        r = Sentinel(eval_fn=fn, guard_fn=fn,
                     param_ranges=[(-5, 5)] * 2).run(time_budget=15, verbose=False)

        assert r["verdict"] == "approved"
        assert r["guard_score"] > r["baseline_guard"]


class TestSentinelConflict:
    def test_conflict_completes(self):
        """eval_fnとguard_fnが矛盾 → 完走。"""
        def eval_fn(p):
            return -sum((x - 3) ** 2 for x in p)

        def guard_fn(p):
            safe = -((p[0] - 3) ** 2 + (p[1] - 3) ** 2)
            danger = -200 * sum((x - 0) ** 2 for x in p[2:])
            return safe + danger

        r = Sentinel(eval_fn=eval_fn, guard_fn=guard_fn,
                     param_ranges=[(-5, 5)] * 5).run(time_budget=15, verbose=False)
        assert r["verdict"] in ("approved", "pivoted", "failed")

    def test_ppl_vs_accuracy(self):
        """PPL的 vs 精度的のシミュレーション。"""
        target = [2.0] * 10

        def ppl_fn(p):
            return -sum((p[i] - target[i]) ** 2 for i in range(10))

        def acc_fn(p):
            safe = -sum((p[i] - target[i]) ** 2 for i in range(5))
            danger = -100 * sum(p[i] ** 2 for i in range(5, 10))
            return safe + danger

        r = Sentinel(eval_fn=ppl_fn, guard_fn=acc_fn,
                     param_ranges=[(-5, 5)] * 10).run(time_budget=20, verbose=False)
        assert r["verdict"] in ("approved", "pivoted", "failed")
        assert len(r["best_params"]) == 10

    def test_forced_pivot_has_multi_observer(self):
        """guard_fnが常にベースラインより悪い → pivot、multi_observerあり。"""
        def eval_fn(p):
            return -sum((x - 3) ** 2 for x in p)

        def guard_fn(p):
            return -sum((x - 3) ** 2 for x in p) - 500 * sum(x ** 2 for x in p)

        r = Sentinel(eval_fn=eval_fn, guard_fn=guard_fn,
                     param_ranges=[(-5, 5)] * 3).run(time_budget=15, verbose=False)

        assert r["verdict"] in ("pivoted", "failed")
        if r["multi_observer"] is not None:
            mo = r["multi_observer"]
            assert "stable_active" in mo
            assert "observer_dependent" in mo
            assert "eval" in mo["observers"]
            assert "guard" in mo["observers"]


class TestSentinelStructure:
    def test_baseline_guard_value(self):
        """baseline_guardはレンジ中央でのguard_fnスコア。"""
        def fn(p):
            return -sum((x - 2) ** 2 for x in p)

        r = Sentinel(eval_fn=fn, guard_fn=fn,
                     param_ranges=[(-5, 5)] * 3).run(time_budget=5, verbose=False)
        assert abs(r["baseline_guard"] - (-12.0)) < 0.01


class TestSentinelDiscrete:
    """Discrete param space must fall back to optimize() successfully."""

    def test_discrete_fallback_improves(self):
        """Integer-rounded eval_fn: fallback finds better than midpoint."""
        def eval_fn(params):
            i, j = int(round(params[0])), int(round(params[1]))
            return -((i - 3) ** 2 + (j - 7) ** 2)  # peak at (3,7)

        def guard_fn(params):
            return 0.0  # constant - always approved

        r = Sentinel(
            eval_fn=eval_fn, guard_fn=guard_fn,
            param_ranges=[(0.0, 10.0), (0.0, 10.0)],
            initial_params=[2.0, 6.0],  # near optimum
        ).run(time_budget=30, verbose=False)

        assert r["verdict"] == "approved"
        # baseline midpoint: (5,5) -> eval = -4 -4 = -8
        # optimum: (3,7) -> eval = 0
        assert r["eval_score"] > -8, \
            f"should improve over midpoint, got {r['eval_score']}"

    def test_initial_params_in_first_measurement(self):
        """Verify initial_params is used as seed in _collect."""
        seen = []

        def eval_fn(params):
            seen.append(list(params))
            return 0.0

        def guard_fn(params):
            return 0.0

        initial = [1.5, 2.5, 3.5]
        s = Sentinel(
            eval_fn=eval_fn, guard_fn=guard_fn,
            param_ranges=[(0.0, 5.0)] * 3,
            initial_params=initial,
        )
        s._collect(5, verbose=False)
        assert seen[0] == initial, "first measurement should use initial_params"

    def test_optimization_mode_key(self):
        """Result dict includes optimization_mode."""
        def fn(p):
            return -sum((x - 2) ** 2 for x in p)

        r = Sentinel(eval_fn=fn, guard_fn=fn,
                     param_ranges=[(-5, 5)] * 3).run(time_budget=5, verbose=False)
        assert "optimization_mode" in r
        assert r["optimization_mode"] in ("owl", "direct", "insufficient", "high", "low")

    def test_learn_flag_accepted(self):
        """learn=True does not break Sentinel."""
        def fn(p):
            return -sum((x - 2) ** 2 for x in p)

        r = Sentinel(
            eval_fn=fn, guard_fn=fn,
            param_ranges=[(-5, 5)] * 3,
            learn=True,
            experience_id="test_learn",
        ).run(time_budget=5, verbose=False)
        assert r["verdict"] in ("approved", "pivoted", "failed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
