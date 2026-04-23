"""Tests for eval_fn sanity checker.

Covers:
  - good eval_fn passes
  - constant eval_fn flagged as fatal
  - exception-raising eval_fn flagged
  - non-numeric return flagged
  - dict return handled
  - format_report runs without error
  - stochastic eval_fn flagged with CV > threshold + wrap recommendation
  - deterministic eval_fn has CV ~0 (no false positive)
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.eval_check import check_eval_fn, format_report


# -----------------------------------------------------------------
# good eval_fn: should pass
# -----------------------------------------------------------------

def test_check_good_quadratic_passes():
    """Smooth quadratic should pass with reasonable proxy_r2."""
    def eval_fn(p):
        return -sum((x - 0.5) ** 2 for x in p)

    diag = check_eval_fn(
        eval_fn,
        [(-1, 1)] * 3,
        time_budget=15,
        experience_id="test_check_good",
    )
    assert diag["severity"] in ("ok", "warn"), (
        f"good quadratic flagged: {diag['issues']}")
    # Structure keys present
    assert "proxy_r2" in diag
    assert "active_dims" in diag
    assert "dead_dims" in diag
    assert diag["probe_score"] is not None


# -----------------------------------------------------------------
# constant eval_fn: should be flagged
# -----------------------------------------------------------------

def test_check_constant_eval_flagged():
    """Constant eval_fn should be flagged (all dims dead)."""
    def eval_fn(p):
        return 42.0  # never changes

    diag = check_eval_fn(
        eval_fn,
        [(-1, 1)] * 3,
        time_budget=10,
        experience_id="test_check_constant",
    )
    assert not diag["ok"]
    assert diag["severity"] in ("warn", "fatal")
    # Issue mentions constant/dead
    joined = " ".join(diag["issues"])
    assert ("constant" in joined or "応答しない" in joined
            or "active_dims" in joined or "proxy_r2" in joined), (
        f"constant eval_fn not diagnosed: {diag}")


# -----------------------------------------------------------------
# raising eval_fn: fatal
# -----------------------------------------------------------------

def test_check_raising_eval_fatal():
    """eval_fn that raises on midpoint gets fatal severity."""
    def eval_fn(p):
        raise RuntimeError("broken")

    diag = check_eval_fn(
        eval_fn,
        [(-1, 1)] * 2,
        time_budget=5,
        experience_id="test_check_raise",
    )
    assert not diag["ok"]
    assert diag["severity"] == "fatal"
    assert diag["eval_fn_returned"] == "error"


# -----------------------------------------------------------------
# non-numeric return: fatal
# -----------------------------------------------------------------

def test_check_nonnumeric_return_fatal():
    """eval_fn returning NaN/None gets fatal severity."""
    def eval_fn(p):
        return float("nan")

    diag = check_eval_fn(
        eval_fn,
        [(-1, 1)] * 2,
        time_budget=5,
        experience_id="test_check_nan",
    )
    assert not diag["ok"]
    assert diag["severity"] == "fatal"


# -----------------------------------------------------------------
# dict return: handled
# -----------------------------------------------------------------

def test_check_dict_return_handled():
    """Multi-observer dict return is accepted."""
    def eval_fn(p):
        return {"primary": -sum(x * x for x in p),
                "secondary": sum(x for x in p)}

    diag = check_eval_fn(
        eval_fn,
        [(-1, 1)] * 3,
        time_budget=15,
        experience_id="test_check_dict",
    )
    # Shouldn't crash; dict path works
    assert diag["eval_fn_returned"] == "dict"
    assert diag["probe_score"] is not None


# -----------------------------------------------------------------
# format_report: runs without error
# -----------------------------------------------------------------

def test_format_report_runs():
    """format_report produces readable output."""
    def eval_fn(p):
        return -sum(x ** 2 for x in p)

    diag = check_eval_fn(
        eval_fn,
        [(-1, 1)] * 2,
        time_budget=8,
        experience_id="test_format",
    )
    report = format_report(diag)
    assert isinstance(report, str)
    assert len(report) > 20
    assert "proxy_r2" in report


# -----------------------------------------------------------------
# stochasticity probe: detect noisy eval_fn
# -----------------------------------------------------------------

def test_check_stochastic_eval_flagged():
    """Stochastic eval_fn should be flagged with high CV and wrap_* recommendation."""
    rng = random.Random(42)

    def stochastic_eval(p):
        # small signal + large noise → high CV at midpoint
        return 1.0 + rng.gauss(0, 2.0)

    diag = check_eval_fn(
        stochastic_eval,
        [(-1, 1)] * 2,
        time_budget=5,
        experience_id="test_check_stochastic",
        n_stability_probes=8,
    )
    assert diag.get("stochastic_detected") is True, (
        f"stochastic eval not flagged: cv={diag.get('stochastic_cv')}"
    )
    assert diag["stochastic_cv"] is not None and diag["stochastic_cv"] > 0.1
    # Recommendation must point at wrap_stochastic / wrap_multi_obs
    rec_text = " ".join(diag["recommendations"])
    assert ("wrap_stochastic" in rec_text
            or "wrap_multi_obs" in rec_text
            or "n_samples_per_eval" in rec_text)


def test_check_deterministic_no_false_positive():
    """Pure deterministic eval_fn should not be flagged as stochastic."""
    def det_eval(p):
        return -sum((x - 0.5) ** 2 for x in p)

    diag = check_eval_fn(
        det_eval,
        [(-1, 1)] * 2,
        time_budget=8,
        experience_id="test_check_det_no_fp",
        n_stability_probes=5,
    )
    assert diag.get("stochastic_detected") is False
    # CV should be effectively zero
    cv = diag.get("stochastic_cv")
    assert cv is not None
    assert cv < 1e-6, f"deterministic eval has non-zero CV: {cv}"


def test_check_stochastic_disabled_when_n_lt_2():
    """n_stability_probes<2 disables the probe; stochastic_cv should be None."""
    rng = random.Random(0)

    def noisy(p):
        return rng.gauss(0, 1.0)

    diag = check_eval_fn(
        noisy,
        [(-1, 1)] * 2,
        time_budget=5,
        experience_id="test_check_sp_off",
        n_stability_probes=1,
    )
    assert diag.get("stochastic_cv") is None
    assert diag.get("stochastic_detected") is False
