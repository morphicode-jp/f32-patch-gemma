"""Reigen (零玄) テストスイート。

pytest -v twelve/tests/test_reigen.py で実行。

戦略:
- TestReigenUnit: Sentinel.run を stub して高速に Reigen
  本体の config 構築・schema 組み立てを検証 (各 <5 秒)
- TestTunableSentinelFidelity: _TunableSentinel が default で Sentinel 相当
  (1 回だけ実 Sentinel 実行、<30 秒)
- TestReigenIntegration: end-to-end 1 ケースのみ、実 run (<3 分)
"""
import os
import sys
import time

import pytest

sys.path.insert(0, "c:/新しいフォルダー (2)/パイプラインオートメーション")

from twelve.agent.reigen import (
    Reigen,
    reigen,
    _TunableSentinel,
)
from twelve.agent.sentinel import Sentinel


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _quadratic(target=None):
    """Cheap eval_fn. Higher = better."""
    def fn(p):
        if target is None:
            return -sum((x - 2.0) ** 2 for x in p)
        return -sum((x - t) ** 2 for x, t in zip(p, target))
    return fn


def _stub_sentinel_run(canned_result):
    """Decorator/util that patches Sentinel.run to return canned_result.
    Caller must restore Sentinel.run after use.
    """
    original = Sentinel.run

    def fake_run(self, time_budget=600, verbose=True):
        return dict(canned_result)

    Sentinel.run = fake_run
    return original


# -------------------------------------------------------------
# Validation tests (constructor only, <1s each)
# -------------------------------------------------------------

class TestReigenValidation:
    def test_convenience_fn_is_callable(self):
        """reigen() module-level function exists and is callable."""
        assert callable(reigen)
        # Signature should accept positional eval_fn, guard_fn, user_param_ranges
        import inspect
        sig = inspect.signature(reigen)
        params = list(sig.parameters.keys())
        assert params[:3] == ["eval_fn", "guard_fn", "user_param_ranges"]

    def test_empty_user_ranges_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[],
            )

    def test_self_range_zero_lo_raises(self):
        """Rule 7: lo<=0 禁止 (scale=0 含む範囲)."""
        with pytest.raises(ValueError, match="Rule 7"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[(0.1, 1.0)],
                self_param_ranges=[(0.0, 1.0)],
                self_param_names=["s"],
                self_param_defaults=[0.5],
            )

    def test_self_range_inverted_raises(self):
        """hi <= lo は不正."""
        with pytest.raises(ValueError, match="hi"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[(0.1, 1.0)],
                self_param_ranges=[(1.0, 0.5)],
                self_param_names=["s"],
                self_param_defaults=[0.7],
            )

    def test_self_default_out_of_range_raises(self):
        with pytest.raises(ValueError, match="not in range"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[(0.1, 1.0)],
                self_param_ranges=[(0.1, 0.5)],
                self_param_names=["s"],
                self_param_defaults=[0.9],
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[(0.1, 1.0)],
                self_param_ranges=[(0.1, 1.0), (0.2, 2.0)],
                self_param_names=["only_one"],
                self_param_defaults=[0.5],
            )

    def test_inner_budget_zero_raises(self):
        with pytest.raises(ValueError, match="inner_time_budget"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[(0.1, 1.0)],
                inner_time_budget=0,
            )

    def test_construction_defaults(self):
        """デフォルト構築は kathara_12 (n_self=12) になった."""
        rec = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(0.1, 1.0)] * 5,
        )
        assert rec.n_user == 5
        assert rec.n_self == 17   # NEW default: kathara_17_adaptive (2026-04-19 A/B)
        assert len(rec.self_param_names) == 17
        assert rec.self_param_names == Reigen.PRESETS["kathara_17_adaptive"]["names"]
        assert rec.self_dim_preset == "kathara_17_adaptive"


# -------------------------------------------------------------
# Stubbed unit tests (Sentinel.run mocked, <5s each)
# -------------------------------------------------------------

class TestReigenStubbedRun:

    def _canned_outer_result(self, n_user=3, n_self=4):
        # Canned "approved" Sentinel result over (n_user + n_self)-dim joint
        joint = [1.5] * (n_user + n_self)
        return {
            "verdict": "approved",
            "best_params": joint,
            "eval_score": -0.5,
            "guard_score": -0.5,
            "baseline_guard": -1.0,
            "proxy_r2": 0.8,
            "optimization_mode": "owl",
            "elapsed_s": 1.0,
            "best_ever_score": -0.5,
            "best_ever_params": joint,
            "safe_dims": None,
            "conflict_dims": None,
            "multi_observer": None,
        }

    def test_return_schema_complete(self):
        """Sentinel.run を stub して、Reigen が全 schema キーを足すか."""
        canned = self._canned_outer_result(n_user=3, n_self=4)
        original = Sentinel.run
        try:
            Sentinel.run = lambda self, time_budget=600, verbose=True: dict(canned)
            r = Reigen(
                eval_fn=_quadratic(), guard_fn=_quadratic(),
                user_param_ranges=[(-5, 5)] * 3,
                inner_time_budget=2,
            ).run(time_budget=60, verbose=False)
        finally:
            Sentinel.run = original

        required_keys = [
            "verdict", "best_params", "eval_score", "guard_score",
            "baseline_guard", "proxy_r2", "elapsed_s",
            "user_best_params", "user_best_score", "self_best_params",
            "self_best_names", "combined_best_params",
            "n_user_dims", "n_self_dims",
            "self_dim_importance", "self_dead_dims", "user_dead_dims",
            "self_diagnostic",
        ]
        for k in required_keys:
            assert k in r, f"Missing key: {k}"
        assert r["verdict"] == "approved"

    def test_user_best_length(self):
        """user_best_params は n_user 長、self_best_params は n_self 項目 dict.

        Uses explicit minimal_4 preset (default became kathara_12 after B-unify).
        """
        canned = self._canned_outer_result(n_user=3, n_self=4)
        original = Sentinel.run
        try:
            Sentinel.run = lambda self, time_budget=600, verbose=True: dict(canned)
            r = Reigen(
                eval_fn=_quadratic(), guard_fn=_quadratic(),
                user_param_ranges=[(-5, 5)] * 3,
                self_dim_preset="minimal_4",   # pin 4 dims for this test
                inner_time_budget=2,
            ).run(time_budget=60, verbose=False)
        finally:
            Sentinel.run = original
        assert len(r["user_best_params"]) == 3
        assert isinstance(r["self_best_params"], dict)
        assert len(r["self_best_params"]) == 4
        assert len(r["best_params"]) == 3
        assert len(r["combined_best_params"]) == 7
        assert r["n_user_dims"] == 3
        assert r["n_self_dims"] == 4

    def test_self_diagnostic_always_present(self):
        """multi_observer=None でも self_diagnostic は付く."""
        canned = self._canned_outer_result(n_user=3, n_self=4)
        original = Sentinel.run
        try:
            Sentinel.run = lambda self, time_budget=600, verbose=True: dict(canned)
            r = Reigen(
                eval_fn=_quadratic(), guard_fn=_quadratic(),
                user_param_ranges=[(-5, 5)] * 3,
                inner_time_budget=2,
            ).run(time_budget=60, verbose=False)
        finally:
            Sentinel.run = original
        diag = r["self_diagnostic"]
        assert "self_informative" in diag
        assert "recommend_freeze" in diag
        assert "self_dead_ratio" in diag
        assert "drift_from_defaults" in diag
        assert 0.0 <= diag["self_dead_ratio"] <= 1.0

    def test_self_best_dict_has_all_names(self):
        """self_best_params dict が DEFAULT_SELF_PARAM_NAMES 全部を持つ."""
        canned = self._canned_outer_result(n_user=3, n_self=4)
        original = Sentinel.run
        try:
            Sentinel.run = lambda self, time_budget=600, verbose=True: dict(canned)
            r = Reigen(
                eval_fn=_quadratic(), guard_fn=_quadratic(),
                user_param_ranges=[(-5, 5)] * 3,
                inner_time_budget=2,
            ).run(time_budget=60, verbose=False)
        finally:
            Sentinel.run = original
        for name in Reigen.DEFAULT_SELF_PARAM_NAMES:
            assert name in r["self_best_params"]


# -------------------------------------------------------------
# Combined eval/guard behavior tests (inspect without running outer)
# -------------------------------------------------------------

class TestReigenCombinedEval:
    def test_combined_guard_uses_user_slice_only(self):
        """combined_guard は user slice だけで guard_fn を呼ぶ."""
        received = []

        def guard_fn(p):
            received.append(list(p))
            return -sum(x ** 2 for x in p)

        rec = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=guard_fn,
            user_param_ranges=[(0.1, 1.0)] * 3,
            inner_time_budget=2,
        )
        combined_guard = rec._make_combined_guard()
        # 7-dim joint input
        joint = [0.5, 0.6, 0.7, 0.3, 0.6, 0.5, 1.0]
        combined_guard(joint)
        # guard_fn received 3 values (user slice only)
        assert len(received) == 1
        assert len(received[0]) == 3
        assert received[0] == [0.5, 0.6, 0.7]

    def test_combined_guard_clips_out_of_range(self):
        """outer が範囲外の joint を投げても user_p は clipped."""
        received = []

        def guard_fn(p):
            received.append(list(p))
            return 0.0

        rec = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=guard_fn,
            user_param_ranges=[(0.1, 1.0), (0.1, 1.0)],
            inner_time_budget=2,
        )
        combined_guard = rec._make_combined_guard()
        # 範囲外
        joint = [-5.0, 100.0, 0.5, 0.6, 0.5, 1.0]
        combined_guard(joint)
        assert received[0][0] == 0.1  # clipped to lo
        assert received[0][1] == 1.0  # clipped to hi


# -------------------------------------------------------------
# Additive cost test (proves linear scaling, not multiplicative)
# -------------------------------------------------------------

class TestReigenAdditive:
    def test_single_outer_sentinel_not_nested(self):
        """Reigen creates ONE outer Sentinel (not nested loops).

        This is the architectural claim of dimension-additive design:
        1 outer Sentinel over (N+M) dims, rather than outer Sentinel (M dims)
        × inner Sentinel (N dims) which would be the multiplicative design.
        """
        init_count = {"n": 0}
        original_init = Sentinel.__init__
        original_run = Sentinel.run

        canned_result = {
            "verdict": "approved",
            "best_params": [0.5] * 6,
            "eval_score": -0.5,
            "guard_score": -0.5,
            "baseline_guard": -1.0,
            "proxy_r2": 0.8,
            "optimization_mode": "owl",
            "elapsed_s": 0.1,
            "best_ever_score": -0.5,
            "best_ever_params": [0.5] * 6,
            "safe_dims": None,
            "conflict_dims": None,
            "multi_observer": None,
        }

        def counting_init(self, *args, **kwargs):
            init_count["n"] += 1
            original_init(self, *args, **kwargs)

        try:
            Sentinel.__init__ = counting_init
            Sentinel.run = lambda self, time_budget=600, verbose=True: dict(canned_result)
            Reigen(
                eval_fn=_quadratic(), guard_fn=_quadratic(),
                user_param_ranges=[(-5, 5)] * 2,
                inner_time_budget=2,
            ).run(time_budget=10, verbose=False)
        finally:
            Sentinel.__init__ = original_init
            Sentinel.run = original_run

        # Only ONE Sentinel is constructed at the outer level.
        # (inner _TunableSentinel constructions don't fire because
        # combined_eval is never called — Sentinel.run is stubbed.)
        assert init_count["n"] == 1, \
            f"Expected 1 outer Sentinel, got {init_count['n']} (nested design?)"


# -------------------------------------------------------------
# _TunableSentinel fidelity (one real Sentinel run)
# -------------------------------------------------------------

class TestTunableSentinelFidelity:
    def test_defaults_give_reasonable_result(self):
        """_TunableSentinel at Sentinel's internal defaults (0.5, 0.5, 1.0) runs
        and produces near-optimum score on quadratic task (target=2).
        """
        fn = _quadratic()
        r = _TunableSentinel(
            eval_fn=fn, guard_fn=fn,
            param_ranges=[(-3, 3)] * 2,
            _owl_budget_ratio=0.5,
            _pivot_budget_ratio=0.5,
            _collect_min_ratio=1.0,
        ).run(time_budget=10, verbose=False)
        assert r["verdict"] in ("approved", "pivoted", "failed")
        score = r.get("best_ever_score") or r.get("eval_score", 0.0)
        # target=2, range=[-3,3]. Midpoint=0 gives score=-8. Optimum gives ~0.
        assert score > -10, f"Score {score} is very far from optimum"

    def test_tunable_scale_ratio_affects_collect(self):
        """_collect_min_ratio=2.0 → actually collects 2x samples."""
        collected_sizes = []

        fn = _quadratic()
        orig_collect = Sentinel._collect

        def spy_collect(self, n_samples, verbose=False):
            collected_sizes.append(n_samples)
            return orig_collect(self, n_samples, verbose)

        try:
            Sentinel._collect = spy_collect
            _TunableSentinel(
                eval_fn=fn, guard_fn=fn,
                param_ranges=[(-3, 3)] * 2,
                _collect_min_ratio=2.0,
            ).run(time_budget=6, verbose=False)
        finally:
            Sentinel._collect = orig_collect

        # With ratio=2.0, super()._collect receives int(round(20 * 2.0)) = 40
        assert any(s >= 40 for s in collected_sizes), \
            f"No collect call with >=40 samples: {collected_sizes}"


# -------------------------------------------------------------
# Phase 1: PRESETS (minimal_4 / kathara_12)
# -------------------------------------------------------------

class TestReigenPresets:
    def test_minimal_4_preset_explicit(self):
        """明示的 self_dim_preset='minimal_4' で n_self=4, 名前正確."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(-1, 1)] * 3,
            self_dim_preset="minimal_4",
        )
        assert r.n_self == 4
        assert r.self_param_names == Reigen.PRESETS["minimal_4"]["names"]

    def test_kathara_12_preset(self):
        """self_dim_preset='kathara_12' で n_self=12, 12 名前存在."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(-1, 1)] * 3,
            self_dim_preset="kathara_12",
        )
        assert r.n_self == 12
        expected = Reigen.PRESETS["kathara_12"]["names"]
        assert r.self_param_names == expected
        # 必須ノブ全部存在
        for key in ("self_min_r_squared", "self_owl_max_iterations",
                    "self_ms_exp", "self_dead_threshold_floor"):
            assert key in r.self_param_names

    def test_default_is_kathara_17_adaptive(self):
        """preset 未指定時は kathara_17_adaptive (2026-04-19 A/B 3/3 勝利で昇格)."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(-1, 1)] * 3,
        )
        assert r.n_self == 17
        assert r.self_dim_preset == "kathara_17_adaptive"

    def test_explicit_ranges_bypass_preset_default(self):
        """self_param_ranges 明示指定なら preset 自動デフォルトは適用されない."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(-1, 1)],
            self_param_ranges=[(0.1, 1.0)] * 4,
            self_param_names=["a", "b", "c", "d"],
            self_param_defaults=[0.5] * 4,
        )
        assert r.n_self == 4
        assert r.self_dim_preset is None  # preset auto-default skipped

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="unknown self_dim_preset"):
            Reigen(
                eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
                user_param_ranges=[(-1, 1)],
                self_dim_preset="bogus",
            )

    def test_build_inner_kwargs_maps_all_12(self):
        """_build_inner_kwargs が kathara_12 の 12 値を全て kwargs に変換."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(-1, 1)] * 2,
            self_dim_preset="kathara_12",
        )
        self_p = list(r.self_param_defaults)  # 12 values
        kwargs = r._build_inner_kwargs(self_p)
        assert len(kwargs) == 12
        assert "min_r_squared" in kwargs
        assert "_owl_max_iterations" in kwargs
        assert "_ms_exp" in kwargs
        assert "_dead_floor" in kwargs

    def test_explicit_ranges_override_preset(self):
        """self_param_ranges 明示指定は preset より優先."""
        custom_ranges = [(0.2, 0.8)] * 4
        custom_names = ["a", "b", "c", "d"]
        custom_defaults = [0.5] * 4
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(-1, 1)],
            self_dim_preset="kathara_12",  # requests 12 but explicit overrides
            self_param_ranges=custom_ranges,
            self_param_names=custom_names,
            self_param_defaults=custom_defaults,
        )
        assert r.n_self == 4
        assert r.self_param_ranges == custom_ranges


# -------------------------------------------------------------
# Phase 1: _TunableSentinel new kwargs
# -------------------------------------------------------------

class TestTunableSentinelPhase1:
    def test_accepts_new_kwargs(self):
        """_TunableSentinel が全 8 新 kwargs を受け入れる (例外なし)."""
        ts = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)] * 2,
            _owl_max_iterations=7,
            _owl_n_rounds=3,
            _stagnation_threshold=5,
            _proxy_pref=0.5,
            _ms_exp=0.4,
            _ms_floor=0.1,
            _dead_ratio=0.2,
            _dead_floor=0.05,
        )
        assert ts._owl_max_iterations == 7
        assert ts._owl_n_rounds == 3
        assert ts._stagnation_threshold == 5
        assert ts._ms_exp == 0.4
        assert ts._dead_ratio == 0.2

    def test_continuous_to_int_rounding(self):
        """整数 knob は int(round(x)) で丸められる."""
        ts = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)],
            _owl_max_iterations=7.7,
            _stagnation_threshold=2.9,
        )
        assert ts._owl_max_iterations == 8
        assert ts._stagnation_threshold == 3

    def test_resolve_proxy_pref(self):
        """_proxy_pref の [0,1] → カテゴリ変換."""
        ts0 = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)], _proxy_pref=0.1)
        assert ts0._resolve_proxy_pref() == "zenron"
        ts1 = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)], _proxy_pref=0.5)
        assert ts1._resolve_proxy_pref() == "zenron_interact"
        ts2 = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)], _proxy_pref=0.9)
        assert ts2._resolve_proxy_pref() == "linear"
        ts3 = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)], _proxy_pref=None)
        assert ts3._resolve_proxy_pref() is None

    def test_ms_mp_override_patches_and_restores(self):
        """_ms_mp_override コンテキスト内で _mp が上書きされ、抜けると元に戻る."""
        from twelve.agent import mirror_agent as _ma
        original_mp = _ma._mp
        ts = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)],
            _ms_exp=0.9, _ms_floor=0.2,
        )
        with ts._ms_mp_override():
            # patched
            assert _ma._mp != original_mp
            assert _ma._mp("mirror_scan", "importance_exponent", 0.3064) == 0.9
            assert _ma._mp("mirror_scan", "connectivity_floor", 0.1411) == 0.2
            # non-overridden keys pass through to original
            other = _ma._mp("mirror_scan", "gap_multiplier", 2.0)
            assert other is not None
        # restored
        assert _ma._mp is original_mp

    def test_ms_mp_override_noop_when_all_none(self):
        """override 値が全 None の場合、monkey-patch しない (no-op)."""
        from twelve.agent import mirror_agent as _ma
        original_mp = _ma._mp
        ts = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(0.1, 1.0)],
        )
        with ts._ms_mp_override():
            assert _ma._mp is original_mp  # no patch


# -------------------------------------------------------------
# Phase 1: owl() kwargs integration (Type C)
# -------------------------------------------------------------

class TestOwlNewKwargs:
    def test_owl_stagnation_threshold_accepted(self):
        """owl() が stagnation_threshold kwarg を受け入れて crash しない."""
        from twelve.optimize import owl
        data = [{"params": [0.5, 0.3], "score": 1.0 + i * 0.1}
                for i in range(20)]
        # Must not raise
        result = owl(data, time_budget=2, stagnation_threshold=5)
        assert "best_params" in result

    def test_owl_force_proxy_type_zenron(self):
        """force_proxy_type='zenron' で結果が得られる."""
        from twelve.optimize import owl
        data = [{"params": [0.5 + i * 0.02, 0.3 + i * 0.01], "score": float(i)}
                for i in range(20)]
        result = owl(data, time_budget=2, force_proxy_type="zenron")
        assert "best_params" in result

    def test_build_proxy_force_fallback(self):
        """force_proxy_type が未マッチなら R²-best フォールバック."""
        from twelve.agent.mirror_agent import MirrorScan
        # Build MS then force a non-existent type; should not crash
        data = [{"params": [0.5 + i * 0.01], "score": float(i)}
                for i in range(15)]
        ms = MirrorScan.from_measurements(data, param_names=["x"])
        fn, r2, name = ms.build_proxy(force_proxy_type="nonexistent_type")
        # Fallback gives any valid proxy (R²-best)
        assert name is not None


# -------------------------------------------------------------
# Phase 2: Kathara Hebbian propagation
# -------------------------------------------------------------

class TestKatharaTopology:
    def test_adj_12_is_5_regular(self):
        """KATHARA_ADJ_12 が 12 ノード 5 正則 30 辺."""
        from twelve.agent.reigen import KATHARA_ADJ_12
        assert len(KATHARA_ADJ_12) == 12
        degrees = [len(v) for v in KATHARA_ADJ_12.values()]
        assert all(d == 5 for d in degrees), f"degrees={degrees}"
        total_edges = sum(degrees) // 2
        assert total_edges == 30, f"edges={total_edges}"

    def test_adj_12_is_symmetric(self):
        """i∈adj(j) ⇔ j∈adj(i)."""
        from twelve.agent.reigen import KATHARA_ADJ_12
        for i, nbrs in KATHARA_ADJ_12.items():
            for j in nbrs:
                assert i in KATHARA_ADJ_12[j], f"asymmetric edge {i}↔{j}"


class TestReigenHebbian:
    def _make_r12(self, enable_hebbian=True, lr=0.05):
        return Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(0.1, 1.0)] * 2,
            self_dim_preset="kathara_12",
            enable_hebbian=enable_hebbian,
            hebbian_lr=lr,
        )

    def test_hebbian_noop_when_disabled(self):
        """enable_hebbian=False で self_p 不変."""
        r = self._make_r12(enable_hebbian=False)
        self_p = list(r.self_param_defaults)
        r._self_history = [(tuple([0.5] * 12), 100.0)] * 10
        out = r._hebbian_propagate(self_p)
        assert out == self_p

    def test_hebbian_noop_minimal_4(self):
        """minimal_4 preset (n_self=4) では Hebbian skip."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(0.1, 1.0)],
            self_dim_preset="minimal_4",
            enable_hebbian=True,
        )
        self_p = [0.3, 0.6, 0.5, 1.0]
        r._self_history = [(tuple(self_p), 10.0)] * 10
        out = r._hebbian_propagate(self_p)
        assert out == self_p  # n_self != 12 → no-op

    def test_hebbian_noop_short_history(self):
        """履歴 < 5 件では no-op."""
        r = self._make_r12()
        self_p = list(r.self_param_defaults)
        r._self_history = [(tuple([0.5] * 12), 100.0)] * 3
        out = r._hebbian_propagate(self_p)
        assert out == self_p

    def test_hebbian_changes_self_p_on_reward(self):
        """履歴に大きく違う best がある + 現 score が下位なら近傍が動く."""
        r = self._make_r12(lr=0.5)  # 大きめの LR で効果確認
        # history: 10 samples with varying scores; best is at a different self_p
        defaults = list(r.self_param_defaults)
        best_p = [defaults[i] * 0.5 if i < 6 else defaults[i] * 1.1 for i in range(12)]
        # clamp to valid ranges
        for i, (lo, hi) in enumerate(r.self_param_ranges):
            best_p[i] = max(lo, min(hi, best_p[i]))
        # history: many low scores + one high score at best_p
        history = [(tuple([0.3] * 12), 1.0)] * 10
        history.append((tuple(best_p), 1000.0))   # clear winner
        r._self_history = history
        # Current self_p is close to defaults
        current = list(defaults)
        updated = r._hebbian_propagate(current)
        # At least one dim should have moved
        changed = any(updated[i] != current[i] for i in range(12))
        assert changed, f"no change: current={current}, updated={updated}"

    def test_record_self_history_caps_at_50(self):
        """_record_self_history が履歴 50 件で打ち切り."""
        r = self._make_r12()
        r._self_history = []
        for i in range(60):
            r._record_self_history([0.5] * 12, float(i))
        assert len(r._self_history) == 50
        # Newest is i=59, oldest remaining should be i=10
        assert r._self_history[-1][1] == 59.0
        assert r._self_history[0][1] == 10.0

    def test_run_initializes_history(self):
        """run() は _self_history を初期化 (前回分のリーク防止)."""
        from twelve.agent.sentinel import Sentinel
        original = Sentinel.run
        canned = {
            "verdict": "approved", "best_params": [0.5] * 14,
            "eval_score": -0.5, "guard_score": -0.5, "baseline_guard": -1.0,
            "proxy_r2": 0.8, "optimization_mode": "owl", "elapsed_s": 0.1,
            "best_ever_score": -0.5, "best_ever_params": [0.5] * 14,
            "safe_dims": None, "conflict_dims": None, "multi_observer": None,
        }
        try:
            Sentinel.run = lambda self, time_budget=600, verbose=True: dict(canned)
            r = self._make_r12()
            r._self_history = [("stale",)]   # pollute from "previous run"
            r.run(time_budget=5, verbose=False)
            # run() reset history to empty list
            assert isinstance(r._self_history, list)
            # Stale marker gone
            assert not any(e == ("stale",) for e in r._self_history)
        finally:
            Sentinel.run = original


# -------------------------------------------------------------
# Phase 3: Batched eval_fn (GPU Path 1)
# -------------------------------------------------------------

class TestReigenBatch:
    def test_batch_eval_fn_accepted(self):
        """Reigen __init__ が batch_eval_fn + batch_size を受け入れる."""
        r = Reigen(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            user_param_ranges=[(0.1, 1.0)] * 2,
            batch_eval_fn=lambda ps: [0.0] * len(ps),
            batch_size=4,
        )
        assert r.batch_eval_fn is not None
        assert r.batch_size == 4

    def test_tunable_batch_collect_size_5(self):
        """_TunableSentinel._collect が batch_size=5 で分割し n_samples=20 を一括評価."""
        calls = {"n": 0, "sizes": []}

        def spy_batch(params_list):
            calls["n"] += 1
            calls["sizes"].append(len(params_list))
            return [-sum(x * x for x in p) for p in params_list]

        ts = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(-1, 1)] * 3,
            _batch_eval_fn=spy_batch, _batch_size=5,
        )
        m = ts._collect(20, verbose=False)
        assert len(m) == 20
        assert calls["n"] == 4   # 4 batches of 5
        assert calls["sizes"] == [5, 5, 5, 5]

    def test_tunable_batch_fallback_when_none(self):
        """_batch_eval_fn=None なら従来の Sentinel._collect が呼ばれる."""
        single_calls = {"n": 0}

        def single_eval(p):
            single_calls["n"] += 1
            return -sum(x * x for x in p)

        ts = _TunableSentinel(
            eval_fn=single_eval, guard_fn=single_eval,
            param_ranges=[(-1, 1)] * 3,
            _batch_eval_fn=None,
        )
        m = ts._collect(20, verbose=False)
        assert len(m) == 20
        assert single_calls["n"] == 20   # all single calls

    def test_batch_score_length_mismatch_raises(self):
        """batch_eval_fn が params_list と違う長さを返したら ValueError."""
        def bad_batch(params_list):
            return [0.0] * (len(params_list) - 1)   # one short

        ts = _TunableSentinel(
            eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
            param_ranges=[(-1, 1)] * 2,
            _batch_eval_fn=bad_batch, _batch_size=5,
        )
        with pytest.raises(ValueError, match="must match"):
            ts._collect(20, verbose=False)

    def test_reigen_propagates_batch_to_inner(self):
        """Reigen 経由で作られる inner _TunableSentinel に batch_eval_fn が届く.

        minimal_4 固定で joint 次元を 2+4=6 に揃える。
        """
        def batch_fn(params_list):
            return [-sum(x * x for x in p) for p in params_list]

        r = Reigen(
            eval_fn=lambda p: -sum(x * x for x in p),
            guard_fn=lambda p: -sum(x * x for x in p),
            user_param_ranges=[(0.1, 1.0)] * 2,
            self_dim_preset="minimal_4",   # pin 4 self dims
            batch_eval_fn=batch_fn, batch_size=4,
        )
        # Call the combined_eval factory, which constructs inner Sentinels
        combined = r._make_combined_eval()
        captured = {}
        original_init = _TunableSentinel.__init__

        def capturing_init(self, *args, **kwargs):
            captured.update(kwargs)
            original_init(self, *args, **kwargs)

        try:
            _TunableSentinel.__init__ = capturing_init
            # invoke with a joint vector; it should try to construct inner
            try:
                combined([0.5, 0.5, 0.3, 0.6, 0.5, 1.0])  # user(2) + self(4)
            except Exception:
                pass  # inner may fail due to partial mock, but init must be called
        finally:
            _TunableSentinel.__init__ = original_init

        assert "_batch_eval_fn" in captured
        assert captured["_batch_eval_fn"] is batch_fn
        assert captured["_batch_size"] == 4


# -------------------------------------------------------------
# Integration: one actual end-to-end Reigen run
# -------------------------------------------------------------

class TestReigenIntegration:
    def test_end_to_end_smoke(self):
        """1 回だけ実 Reigen 実行.

        注意: 外側 owl autonomous が verify_fn (= 内側 Sentinel 1 回分) を
        最大 50 回ほど呼ぶため、outer time_budget は wall time の厳密な
        上限にならない。inner_time_budget=1 + 5 dim joint で、wall time は
        およそ 150-350s を想定し、assert は 600s 以内.
        """
        fn = _quadratic()
        t0 = time.time()
        r = Reigen(
            eval_fn=fn, guard_fn=fn,
            user_param_ranges=[(-3, 3)],   # 1 user + 4 self = 5 dim via minimal_4
            self_dim_preset="minimal_4",    # explicit: keep integration fast (4 dims)
            inner_time_budget=1,
            verbose=False,
        ).run(time_budget=30, verbose=False)
        elapsed = time.time() - t0

        assert elapsed < 600, f"Took too long: {elapsed:.0f}s"
        assert r["verdict"] in ("approved", "pivoted", "failed")
        assert r["n_user_dims"] == 1
        assert r["n_self_dims"] == 4
        assert len(r["user_best_params"]) == 1
        assert len(r["self_best_params"]) == 4
        # target=2, range=[-3,3]. Midpoint=0 gives score=-4. Near-opt → 0.
        assert r["user_best_score"] > -10, \
            f"user_best_score={r['user_best_score']} too far from optimum"
