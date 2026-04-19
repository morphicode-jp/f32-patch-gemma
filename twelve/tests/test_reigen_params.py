"""Tests for reigen_params.json externalization + Reigen² kwarg override contract.

Covers:
  - `_rp()` accessor fallback behavior (missing file / corrupt / partial)
  - `_load_reigen_params()` permissive error handling
  - Reigen kwargs default via `_rp()` (JSON > hardcoded fallback)
  - Explicit kwarg > JSON override (Reigen² meta-tuning contract)
  - `0.0` falsy explicit kwarg preserved (None-sentinel correctness)
  - PRESETS dict untouched by refactor
  - load-once semantics (disk edit after import is ignored)
"""
import json
import os
import sys
import tempfile
import importlib

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import twelve.agent.reigen as reigen_mod
from twelve.agent.reigen import (
    Reigen, _rp, _load_reigen_params, _REIGEN_PARAMS_PATH,
)


# -----------------------------------------------------------------
# 1-5: _rp() accessor + _load_reigen_params() contract
# -----------------------------------------------------------------

def test_rp_reads_json_when_present(monkeypatch):
    """_rp returns injected value from _REIGEN_PARAMS."""
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS",
                        {"outer": {"outer_min_r_squared": 0.9}})
    assert reigen_mod._rp("outer", "outer_min_r_squared", 0.3) == 0.9


def test_rp_fallback_when_section_missing(monkeypatch):
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS", {})
    assert reigen_mod._rp("nonexistent", "k", 42) == 42


def test_rp_fallback_when_key_missing(monkeypatch):
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS",
                        {"outer": {"other_key": 1.0}})
    assert reigen_mod._rp("outer", "outer_min_r_squared", 0.3) == 0.3


def test_load_returns_empty_on_missing_file(tmp_path, monkeypatch):
    missing = str(tmp_path / "nope.json")
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS_PATH", missing)
    assert reigen_mod._load_reigen_params() == {}


def test_load_returns_empty_on_corrupt_json(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("not json{{", encoding="utf-8")
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS_PATH", str(bad))
    assert reigen_mod._load_reigen_params() == {}


# -----------------------------------------------------------------
# 6-9: Reigen.__init__ default-resolution + kwarg-override contract
# -----------------------------------------------------------------

def _stub_reigen(monkeypatch, params=None, **kwargs):
    """Helper — build a Reigen with given _REIGEN_PARAMS injected."""
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS", params or {})
    return Reigen(
        eval_fn=lambda p: 0.0,
        guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)] * 3,
        self_dim_preset="minimal_4",
        **kwargs,
    )


def test_init_default_matches_prior_literals(monkeypatch):
    """Empty JSON → every new attr == prior hardcoded literal."""
    r = _stub_reigen(monkeypatch, params={})
    assert r.outer_min_r_squared == 0.3
    assert r.hebbian_lr == 0.05
    assert r.batch_size == 8
    assert r.hebbian_min_history == 5
    assert r.hebbian_percentile_gate == 0.75
    assert r.hebbian_k_winners == 3


def test_init_json_override(monkeypatch):
    r = _stub_reigen(
        monkeypatch,
        params={
            "outer": {"outer_min_r_squared": 0.9, "batch_size": 16},
            "hebbian": {"hebbian_lr": 0.1, "min_history": 10,
                        "percentile_gate": 0.5, "k_winners": 5},
        },
    )
    assert r.outer_min_r_squared == 0.9
    assert r.batch_size == 16
    assert r.hebbian_lr == 0.1
    assert r.hebbian_min_history == 10
    assert r.hebbian_percentile_gate == 0.5
    assert r.hebbian_k_winners == 5


def test_init_kwarg_beats_json(monkeypatch):
    """Reigen² contract: explicit kwarg overrides JSON."""
    r = _stub_reigen(
        monkeypatch,
        params={"outer": {"outer_min_r_squared": 0.9}},
        outer_min_r_squared=0.5,
    )
    assert r.outer_min_r_squared == 0.5


def test_init_kwarg_beats_json_for_zero_valued_override(monkeypatch):
    """Falsy explicit kwarg (0.0) must NOT be overwritten by JSON.
    Proves None-sentinel (not literal-default) is used in __init__.
    """
    r = _stub_reigen(
        monkeypatch,
        params={"hebbian": {"hebbian_lr": 0.9}},
        hebbian_lr=0.0,
    )
    assert r.hebbian_lr == 0.0


# -----------------------------------------------------------------
# 10-11: wall_time_factor (run() + reigen() wrapper) — _UNSET sentinel
# -----------------------------------------------------------------

def test_run_wall_time_factor_json_override(monkeypatch):
    """run() called without wall_time_factor kwarg → picks up JSON value.

    Stub out Sentinel.run to short-circuit before actual eval; we just
    want to inspect `_abort_at` set at the top of Reigen.run.
    """
    r = _stub_reigen(
        monkeypatch,
        params={"outer": {"wall_time_factor": 7.0}},
    )
    import time as _time
    captured = {}

    # Sentinel.run is the first thing called after _abort_at assignment.
    from twelve.agent.sentinel import Sentinel
    orig_sentinel_run = Sentinel.run

    def stop_run(self, *a, **kw):
        captured["abort_at"] = r._abort_at
        captured["t0"] = _time.time()
        raise RuntimeError("stop")

    monkeypatch.setattr(Sentinel, "run", stop_run)
    with pytest.raises(RuntimeError, match="stop"):
        r.run(time_budget=10)
    # _abort_at ≈ t0 + 10 * 7.0 = t0 + 70 (allow 2s slack for timing).
    delta = captured["abort_at"] - captured["t0"]
    assert 68 < delta < 72, f"expected ~70s abort offset, got {delta:.2f}"


def test_reigen_wrapper_wall_time_factor_kwarg(monkeypatch):
    """reigen(..., wall_time_factor=7.0) forwards to Reigen.run()."""
    # Intercept Reigen.run to capture the kwarg.
    from twelve.agent.reigen import reigen as reigen_fn
    captured = {}

    orig_run = Reigen.run

    def spy_run(self, time_budget=600, verbose=None, wall_time_factor=reigen_mod._UNSET):
        captured["wall_time_factor"] = wall_time_factor
        captured["time_budget"] = time_budget
        # Don't actually run; return stub dict.
        return {"verdict": "stub", "user_best_params": [], "user_best_score": 0.0}

    monkeypatch.setattr(Reigen, "run", spy_run)
    reigen_fn(
        eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)], time_budget=30,
        wall_time_factor=7.0, self_dim_preset="minimal_4",
    )
    assert captured["wall_time_factor"] == 7.0


# -----------------------------------------------------------------
# 12-13: _hebbian_propagate uses new kwargs
# -----------------------------------------------------------------

def _seed_history(r, n=10):
    hist = []
    for i in range(n):
        sp = tuple(0.1 + 0.05 * i + 0.01 * k for k in range(12))
        score = -((i - 5) ** 2) / 10.0
        hist.append((sp, score))
    r._self_history = hist


def test_hebbian_min_history_respected(monkeypatch):
    """min_history=1 should allow propagation even with just 2 history entries."""
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS", {})
    r = Reigen(
        eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)],
        self_dim_preset="kathara_12",
        enable_hebbian=True,
        hebbian_min_history=1,
        hebbian_percentile_gate=0.0,  # permissive gate so propagation fires
    )
    _seed_history(r, n=2)
    self_p = [0.5] * 12
    out = r._hebbian_propagate(self_p)
    # At least one dim must have moved (some winner's neighbor)
    assert out != self_p


def test_hebbian_k_winners_respected(monkeypatch):
    """k_winners=1 should update fewer dims than k_winners=3 (usually)."""
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS", {})
    r1 = Reigen(
        eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)],
        self_dim_preset="kathara_12",
        enable_hebbian=True,
        hebbian_k_winners=1,
        hebbian_percentile_gate=0.0,
    )
    _seed_history(r1)
    out1 = r1._hebbian_propagate([0.5] * 12)
    changed1 = sum(1 for a, b in zip(out1, [0.5] * 12) if a != b)

    r3 = Reigen(
        eval_fn=lambda p: 0.0, guard_fn=lambda p: 0.0,
        user_param_ranges=[(-1.0, 1.0)],
        self_dim_preset="kathara_12",
        enable_hebbian=True,
        hebbian_k_winners=3,
        hebbian_percentile_gate=0.0,
    )
    _seed_history(r3)
    out3 = r3._hebbian_propagate([0.5] * 12)
    changed3 = sum(1 for a, b in zip(out3, [0.5] * 12) if a != b)

    # k=3 nudges 3 winners × 5 Kathara neighbors (with overlap), typically > k=1's 5.
    assert changed3 >= changed1


# -----------------------------------------------------------------
# 14: PRESETS dict intact
# -----------------------------------------------------------------

def test_presets_dict_unchanged():
    """Refactor must not touch PRESETS. Tests this by exact-name/defaults check."""
    assert Reigen.PRESETS["minimal_4"]["names"] == [
        "self_min_r_squared", "self_owl_budget_ratio",
        "self_pivot_budget_ratio", "self_collect_min_ratio",
    ]
    assert Reigen.PRESETS["minimal_4"]["defaults"] == [0.30, 0.60, 0.50, 1.0]
    assert len(Reigen.PRESETS["kathara_12"]["names"]) == 12
    assert Reigen.PRESETS["kathara_12"]["defaults"] == [
        0.30, 0.60, 0.50, 1.0,
        10.0, 5.0, 3.0, 0.33,
        0.3064, 0.1411, 0.156, 0.1377,
    ]


# -----------------------------------------------------------------
# 15: load-once semantics (JSON edits post-import ignored)
# -----------------------------------------------------------------

def test_rp_load_once_semantics(tmp_path, monkeypatch):
    """Mirror _mp contract: JSON is read at import time; disk edits are NOT picked up
    on subsequent _rp() calls. Users must re-import the module to pick up changes."""
    jf = tmp_path / "rp.json"
    jf.write_text(json.dumps({"x": {"y": 1}}), encoding="utf-8")
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS_PATH", str(jf))
    # Manually invoke load (simulates import) and capture
    first = reigen_mod._load_reigen_params()
    assert first == {"x": {"y": 1}}
    # Now edit disk content
    jf.write_text(json.dumps({"x": {"y": 999}}), encoding="utf-8")
    # _REIGEN_PARAMS (cached at import) is unchanged
    monkeypatch.setattr(reigen_mod, "_REIGEN_PARAMS", first)
    assert reigen_mod._rp("x", "y", -1) == 1
