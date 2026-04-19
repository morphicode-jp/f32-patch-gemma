"""Tests for Sentinel.initial_measurements — curated-data path bypassing
_collect's uniform-random sampling (2026-04-19 refactor).
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent.sentinel import Sentinel


def test_curated_measurements_replaces_collect():
    calls = {"n": 0}
    def eval_fn(p):
        calls["n"] += 1
        return -sum(x * x for x in p)

    curated = [{"params": [0.1 * i, 0.1 * i], "score": -0.02 * i}
               for i in range(22)]   # >= n_samples
    s = Sentinel(eval_fn, eval_fn, [(-1, 1)] * 2,
                 initial_measurements=curated)
    ms = s._collect(max(20, 2), verbose=False)
    assert len(ms) == 22
    assert calls["n"] == 0, f"curated path should not call eval_fn; got {calls['n']}"


def test_partial_curated_tops_up_with_random():
    """Curated < n_samples → top up via eval_fn random samples."""
    calls = {"n": 0}
    def eval_fn(p):
        calls["n"] += 1
        return 0.0

    curated = [{"params": [0.1, 0.1], "score": 0.0}] * 5  # only 5 curated
    s = Sentinel(eval_fn, eval_fn, [(-1, 1)] * 2,
                 initial_measurements=curated)
    ms = s._collect(20, verbose=False)
    assert len(ms) == 20
    assert calls["n"] == 15  # 20 - 5 top-up via random


def test_legacy_path_unchanged_when_no_curated():
    calls = {"n": 0}
    def eval_fn(p):
        calls["n"] += 1
        return 0.0
    s = Sentinel(eval_fn, eval_fn, [(-1, 1)] * 2,
                 initial_params=[0.0, 0.0])
    ms = s._collect(20, verbose=False)
    assert len(ms) == 20
    assert calls["n"] == 20  # full random collection


def test_curated_first_entry_preserved():
    """Curated measurements appear verbatim (not re-evaluated)."""
    def eval_fn(p):
        return -99.0  # distinguishable sentinel

    curated = [{"params": [0.1, 0.2], "score": 42.0},
               {"params": [0.3, 0.4], "score": 55.0}]
    s = Sentinel(eval_fn, eval_fn, [(-1, 1)] * 2,
                 initial_measurements=curated)
    ms = s._collect(20, verbose=False)
    # First two entries should be the curated ones with their original scores
    assert ms[0]["score"] == 42.0
    assert ms[1]["score"] == 55.0
    # Top-up entries come from eval_fn → score == -99
    assert ms[2]["score"] == -99.0


def test_malformed_curated_entries_skipped():
    """Curated entries missing required keys are skipped gracefully."""
    calls = {"n": 0}
    def eval_fn(p):
        calls["n"] += 1
        return 0.0

    curated = [
        {"params": [0.1, 0.2], "score": 1.0},  # valid
        {"params": [0.3, 0.4]},                  # missing score
        "not a dict",                             # wrong type
        {"score": 2.0},                           # missing params
        {"params": [0.5, 0.6], "score": 3.0},    # valid
    ]
    s = Sentinel(eval_fn, eval_fn, [(-1, 1)] * 2,
                 initial_measurements=curated)
    ms = s._collect(20, verbose=False)
    # Only 2 valid curated, rest topped up
    assert ms[0]["score"] == 1.0
    assert ms[1]["score"] == 3.0
    assert calls["n"] == 18  # 20 - 2
