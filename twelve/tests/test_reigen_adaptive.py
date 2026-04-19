"""Tests for kathara_17_adaptive preset — additive Reigen²."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import twelve.agent.reigen as reigen_mod
from twelve.agent.reigen import Reigen


# -----------------------------------------------------------------
# Preset structure
# -----------------------------------------------------------------

def test_kathara_17_adaptive_exists():
    p = Reigen.PRESETS["kathara_17_adaptive"]
    assert len(p["ranges"]) == 17
    assert len(p["names"]) == 17
    assert len(p["defaults"]) == 17


def test_kathara_17_adaptive_extra_dims_layout():
    """Indices 13-17 must be the 5 Reigen-level outer constants in this order."""
    names = Reigen.PRESETS["kathara_17_adaptive"]["names"]
    assert names[12] == "self_batch_size"
    assert names[13] == "self_hebbian_lr"
    assert names[14] == "self_hebbian_min_history"
    assert names[15] == "self_hebbian_percentile_gate"
    assert names[16] == "self_hebbian_k_winners"


def test_kathara_17_adaptive_first_12_match_kathara_12():
    """First 12 dims must be identical to kathara_12 (same ranges/names/defaults)."""
    p17 = Reigen.PRESETS["kathara_17_adaptive"]
    p12 = Reigen.PRESETS["kathara_12"]
    assert p17["ranges"][:12] == p12["ranges"]
    assert p17["names"][:12] == p12["names"]
    assert p17["defaults"][:12] == p12["defaults"]


# -----------------------------------------------------------------
# _NAME_TO_TUNABLE_KWARG additions
# -----------------------------------------------------------------

def test_self_batch_size_routes_to_inner_batch_size():
    mapping = Reigen._NAME_TO_TUNABLE_KWARG
    # _batch_size (underscore prefix matches other _TunableSentinel kwargs)
    assert mapping.get("self_batch_size") == "_batch_size"


def test_outer_only_hebbian_names_not_in_mapping():
    """The 4 outer-only dims must NOT be in the inner mapping (they are
    consumed by outer _hebbian_propagate, not passed to inner Sentinel)."""
    mapping = Reigen._NAME_TO_TUNABLE_KWARG
    assert "self_hebbian_lr" not in mapping
    assert "self_hebbian_min_history" not in mapping
    assert "self_hebbian_percentile_gate" not in mapping
    assert "self_hebbian_k_winners" not in mapping


# -----------------------------------------------------------------
# Reigen construction with adaptive preset
# -----------------------------------------------------------------

def _build_adaptive(**kwargs):
    return Reigen(
        eval_fn=lambda p: -sum((x - 1.0) ** 2 for x in p),
        guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)] * 2,
        self_dim_preset="kathara_17_adaptive",
        enable_hebbian=True,
        inner_time_budget=1,
        **kwargs,
    )


def test_construction_n_self_17():
    r = _build_adaptive()
    assert r.n_self == 17
    assert len(r.self_param_ranges) == 17


def test_build_inner_kwargs_includes_batch_size():
    r = _build_adaptive()
    self_p = list(r.self_param_defaults)
    # Force self_batch_size to 12.7 → should become int(13) in kwargs
    self_p[12] = 12.7
    kwargs = r._build_inner_kwargs(self_p)
    # _batch_size (underscore prefix) — routes to inner _TunableSentinel kwarg
    assert kwargs.get("_batch_size") == 13


def test_build_inner_kwargs_no_hebbian_extras():
    """Extra 4 hebbian dims must NOT appear in inner kwargs."""
    r = _build_adaptive()
    self_p = list(r.self_param_defaults)
    kwargs = r._build_inner_kwargs(self_p)
    # None of the outer-only hebbian attributes should be passed to inner.
    assert "hebbian_lr" not in kwargs
    assert "hebbian_min_history" not in kwargs
    assert "hebbian_percentile_gate" not in kwargs
    assert "hebbian_k_winners" not in kwargs


# -----------------------------------------------------------------
# Adaptive Hebbian: reads from best-history's self_p
# -----------------------------------------------------------------

def _seed_adaptive_history(r, lr_at_best=0.2, min_hist_at_best=3,
                            pct_at_best=0.3, kw_at_best=2):
    """Seed history so that best entry carries specific hebbian settings."""
    hist = []
    # 9 low-score entries with DEFAULT hebbian settings
    for i in range(9):
        sp = list(r.self_param_defaults)
        sp[13] = 0.05  # hebbian_lr (low)
        sp[14] = 5     # min_history
        sp[15] = 0.75  # percentile_gate
        sp[16] = 3     # k_winners
        hist.append((tuple(sp), -1.0 - i))
    # 1 BEST entry with custom hebbian settings at dims 13..16
    best_sp = list(r.self_param_defaults)
    best_sp[13] = lr_at_best
    best_sp[14] = min_hist_at_best
    best_sp[15] = pct_at_best
    best_sp[16] = kw_at_best
    hist.append((tuple(best_sp), 100.0))
    r._self_history = hist


def test_get_adaptive_hebbian_params_from_best_history():
    r = _build_adaptive()
    _seed_adaptive_history(r,
                           lr_at_best=0.123, min_hist_at_best=7,
                           pct_at_best=0.42, kw_at_best=5)
    best_p = r._self_history[-1][0]
    lr, min_hist, pct, kw = r._get_adaptive_hebbian_params(best_p)
    assert lr == 0.123
    assert min_hist == 7
    assert abs(pct - 0.42) < 1e-9
    assert kw == 5


def test_get_adaptive_hebbian_params_falls_back_for_kathara_12():
    """kathara_12 preset should fall back to static self attributes."""
    r = Reigen(
        eval_fn=lambda p: 0, guard_fn=lambda p: 0,
        user_param_ranges=[(-1, 1)] * 2,
        self_dim_preset="kathara_12",
        hebbian_lr=0.07, hebbian_min_history=8,
        hebbian_percentile_gate=0.6, hebbian_k_winners=2,
    )
    dummy_best_p = [0.0] * 12
    lr, min_hist, pct, kw = r._get_adaptive_hebbian_params(dummy_best_p)
    assert lr == 0.07
    assert min_hist == 8
    assert pct == 0.6
    assert kw == 2


def test_hebbian_propagate_uses_adaptive_lr():
    """If best-history has larger lr, propagation delta should be larger."""
    r_static = _build_adaptive()
    _seed_adaptive_history(r_static, lr_at_best=0.05)  # static-like
    out_static = r_static._hebbian_propagate([0.5] * 17)

    r_hot = _build_adaptive()
    _seed_adaptive_history(r_hot, lr_at_best=0.5)     # 10x lr
    out_hot = r_hot._hebbian_propagate([0.5] * 17)

    # Pick a dim that was actually updated and compare magnitude.
    # Dim 0 should see propagation since best_p[0]=default (0.30)
    # vs current 0.5 → delta -0.20, which is among largest abs deltas.
    # But winners are picked by abs delta; more of the propagation
    # depends on exact self_p. Use summed abs change as rough signal.
    total_static = sum(abs(a - 0.5) for a in out_static[:12])
    total_hot = sum(abs(a - 0.5) for a in out_hot[:12])
    assert total_hot > total_static, \
        f"10x lr should give larger updates: static={total_static} vs hot={total_hot}"


def test_hebbian_propagate_respects_adaptive_min_history():
    """If best-history demands min_history > actual hist length, propagation skips."""
    r = _build_adaptive()
    _seed_adaptive_history(r, min_hist_at_best=100)  # hist has 10, best demands 100
    out = r._hebbian_propagate([0.5] * 17)
    # No propagation → output unchanged
    assert out == [0.5] * 17


def test_hebbian_does_not_touch_dims_13_16():
    """Kathara propagation should only update first 12 dims (adjacency).
    The 5 extra dims (including 13-16) are NOT nudged by Hebbian."""
    r = _build_adaptive()
    _seed_adaptive_history(r, lr_at_best=0.5)
    input_self_p = list(r.self_param_defaults)
    # Set dim 13-16 to distinct values
    input_self_p[13] = 0.123
    input_self_p[14] = 7
    input_self_p[15] = 0.42
    input_self_p[16] = 5
    out = r._hebbian_propagate(input_self_p)
    # Dims 12-16 must be identical pre/post (NOT touched by Hebbian)
    for i in range(12, 17):
        assert out[i] == input_self_p[i], \
            f"dim {i} was modified (shouldn't be): {input_self_p[i]} -> {out[i]}"


# -----------------------------------------------------------------
# Backward compat: kathara_12 behavior unchanged
# -----------------------------------------------------------------

def test_kathara_17_adaptive_is_default():
    """Default preset is kathara_17_adaptive (2026-04-19 A/B promotion).
    kathara_12 remains available as explicit opt-in for legacy static Hebbian."""
    r = Reigen(
        eval_fn=lambda p: 0, guard_fn=lambda p: 0,
        user_param_ranges=[(-1, 1)] * 2,
    )
    assert r.self_dim_preset == "kathara_17_adaptive"
    assert r.n_self == 17

    # Explicit kathara_12 still works as legacy opt-in
    r12 = Reigen(
        eval_fn=lambda p: 0, guard_fn=lambda p: 0,
        user_param_ranges=[(-1, 1)] * 2,
        self_dim_preset="kathara_12",
    )
    assert r12.self_dim_preset == "kathara_12"
    assert r12.n_self == 12
