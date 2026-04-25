"""Tests for kathara_mimir ODIN integration and curated sharing."""
import os
import pickle
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from twelve.agent import kathara_mimir as km


def _worker_eval(p):
    return float(p[0])


def test_share_via_kathara_respects_share_weight():
    """share_weight=0 keeps self, share_weight=1 uses neighbor blend."""
    params = [[float(i)] for i in range(12)]
    scores = [float(i) for i in range(12)]

    own = km._share_via_kathara(params, scores, 0, share_weight=0.0)
    shared = km._share_via_kathara(params, scores, 0, share_weight=1.0)

    assert own.tolist() == [0.0]
    assert shared.tolist() != [0.0]


def test_share_via_kathara_handles_non_finite_neighbor_scores():
    """Failed neighbor scores must not poison the share vector with NaN."""
    params = [[float(i)] for i in range(12)]
    scores = [float(i) for i in range(12)]
    scores[1] = float("-inf")
    scores[4] = float("nan")

    shared = km._share_via_kathara(params, scores, 0, share_weight=1.0)

    assert np_is_finite_list(shared.tolist())


def np_is_finite_list(values):
    return all(value == value and value not in (float("inf"), float("-inf"))
               for value in values)


def test_kathara_mimir_use_odin_rescores_curated_per_target_node(monkeypatch):
    """Kathara shared params must be scored by the receiving node's eval_fn."""
    import twelve.agent.mimir_odin_stable as stable_mod

    eval_fns = []
    for i in range(12):
        def eval_fn(p, node=i):
            return float(1000 * node + p[0])

        eval_fn.node_i = i
        eval_fns.append(eval_fn)

    captured = []

    def fake_stable(eval_fn, param_ranges, **kwargs):
        curated = kwargs.get("curated_measurements")
        if curated is not None:
            captured.append((eval_fn.node_i, curated))
        params = [eval_fn.node_i + 0.25]
        return {
            "best_params": params,
            "best_score": eval_fn(params),
            "tool_used": "fake_stable",
        }

    monkeypatch.setattr(stable_mod, "mimir_odin_stable", fake_stable)

    results = km.kathara_mimir(
        eval_fns,
        [(0, 20)],
        time_budget=10,
        share_rounds=2,
        max_workers=1,
        use_odin=True,
    )

    assert len(results) == 12
    assert len(captured) == 12
    for node_i, curated in captured:
        assert curated
        assert len(curated) <= 4
        for measurement in curated:
            assert measurement["score"] == pytest.approx(
                eval_fns[node_i](measurement["params"])
            )


def test_kathara_mimir_rejects_invalid_share_weight():
    """share_weight is a blend ratio, so values outside [0, 1] are invalid."""
    eval_fns = [_worker_eval] * 12

    with pytest.raises(ValueError, match="share_weight"):
        km.kathara_mimir(
            eval_fns,
            [(0, 1)],
            time_budget=1,
            share_weight=1.5,
        )


def test_kathara_mimir_rejects_too_small_curated_limit():
    """Need at least self + blended share for curated sharing."""
    eval_fns = [_worker_eval] * 12

    with pytest.raises(ValueError, match="max_curated_per_node"):
        km.kathara_mimir(
            eval_fns,
            [(0, 1)],
            time_budget=1,
            max_curated_per_node=1,
        )


def test_node_budget_accounts_for_worker_waves():
    """max_workers < 12 must reduce per-node budget to respect wall time."""
    budget = km._node_budget_for_round(
        round_budget=30.0,
        remaining=60.0,
        max_workers=4,
        rounds_left=2,
    )

    assert budget == pytest.approx(9.5)


def test_kathara_mimir_auto_throttles_odin(monkeypatch):
    """ODIN defaults avoid 12×4 worker oversubscription."""
    import twelve.agent.mimir_odin_stable as stable_mod

    monkeypatch.setattr(km.os, "cpu_count", lambda: 8)

    def fake_stable(eval_fn, param_ranges, **kwargs):
        return {
            "best_params": [0.25],
            "best_score": eval_fn([0.25]),
            "tool_used": "fake_stable",
        }

    monkeypatch.setattr(stable_mod, "mimir_odin_stable", fake_stable)

    results = km.kathara_mimir(
        [_worker_eval] * 12,
        [(0, 1)],
        time_budget=10,
        share_rounds=1,
        max_workers=12,
        use_odin=True,
    )

    assert results[0]["kathara_effective_max_workers"] == 2
    assert results[0]["kathara_inner_executor"] == "thread"


def test_process_worker_can_select_odin_stable(monkeypatch):
    """use_processes=True path must not silently fall back to plain mimir."""
    import twelve.agent.mimir_odin_stable as stable_mod

    def fake_stable(eval_fn, param_ranges, **kwargs):
        return {
            "best_params": [0.25],
            "best_score": eval_fn([0.25]),
            "tool_used": "fake_stable",
        }

    monkeypatch.setattr(stable_mod, "mimir_odin_stable", fake_stable)

    i, result, share_counted = km._process_worker((
        3,
        pickle.dumps(_worker_eval),
        [(0, 1)],
        None,
        None,
        1.0,
        "test_worker",
        {},
        True,
    ))

    assert i == 3
    assert share_counted is False
    assert result["tool_used"] == "fake_stable"
    assert result["best_score"] == pytest.approx(0.25)


def test_process_worker_rescores_candidates_before_curating(monkeypatch):
    """Process path re-scores candidates in the child worker before ODIN sees them."""
    import twelve.agent.mimir_odin_stable as stable_mod

    captured = []

    def fake_stable(eval_fn, param_ranges, **kwargs):
        captured.extend(kwargs.get("curated_measurements", []))
        return {
            "best_params": [0.25],
            "best_score": eval_fn([0.25]),
            "tool_used": "fake_stable",
        }

    monkeypatch.setattr(stable_mod, "mimir_odin_stable", fake_stable)

    i, result, share_counted = km._process_worker((
        4,
        pickle.dumps(_worker_eval),
        [(0, 1)],
        None,
        [[0.1], [0.2], [0.2]],
        1.0,
        "test_worker_rescore",
        {},
        True,
        True,
    ))

    assert i == 4
    assert result["tool_used"] == "fake_stable"
    assert share_counted is True
    assert captured == [
        {"params": [0.1], "score": pytest.approx(0.1)},
        {"params": [0.2], "score": pytest.approx(0.2)},
    ]
