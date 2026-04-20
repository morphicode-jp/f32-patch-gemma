"""hagen (覇玄) — owl と Reigen の上位層 meta-dispatcher (placeholder name).

役目:
  - 問題の eval コストを測定 (1-call tick)
  - 高コスト/curated あり → owl 直行 (L-BFGS-B + multi-start + random-restart 全部盛り)
  - 安 eval + confidence 不足 → Reigen へ escalate (owl best を warm-start)
  - 構造発見情報 (dead_dims, importance, fragility) は最終 dict にマージして保持

設計原則:
  - 既存 owl() / reigen() の動作は変更しない (呼び出しのみ)
  - 全 opt-in 強化機能を デフォルト ON で呼ぶ (use_lbfgs/multi/random_restart=5)
  - Reigen escalation は opt-in 判定 (confidence / budget 残量)
  - GP+EI 型 overhead の再発防止: bounded cost (1 owl + 高々 1 Reigen)

LaD (Logic-as-Data):
  dispatch 閾値と owl/Reigen kwargs は hagen_params.json で JSON 制御可能。
  precedence: 明示 kwarg > JSON > hardcode fallback。K² 自己最適化対応。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional, Sequence


_HAGEN_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "hagen_params.json")

_HARDCODE_DEFAULTS = {
    "dispatch": {
        "eval_cost_threshold": 0.5,
        "owl_share": 0.4,
        "escalation_min_remaining": 5.0,
        "confidence_skip_threshold": 0.7,
    },
    "owl_kwargs": {
        "random_restart_count": 5,
        "max_iterations": 30,
        "min_r_squared": 0.1,
    },
    "reigen_kwargs": {
        "inner_time_budget": 2.0,
        "wall_time_factor": 1.0,
    },
}


def _load_hagen_params() -> dict:
    """Load hagen_params.json, falling back to hardcoded defaults silently."""
    try:
        with open(_HAGEN_PARAMS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {k: dict(v) for k, v in _HARDCODE_DEFAULTS.items()}
    # Merge with hardcode (missing keys fall through to hardcode)
    merged = {k: dict(v) for k, v in _HARDCODE_DEFAULTS.items()}
    for section, values in (data or {}).items():
        if section.startswith("_") or not isinstance(values, dict):
            continue
        if section in merged:
            merged[section].update({
                k: v for k, v in values.items() if not k.startswith("_")
            })
    return merged


def _resolve(kwarg_val, json_val, hardcode_val):
    """Precedence: explicit kwarg > JSON > hardcode. None-sentinel for kwarg."""
    if kwarg_val is not None:
        return kwarg_val
    if json_val is not None:
        return json_val
    return hardcode_val


def _measure_eval_cost(eval_fn: Callable, ranges: Sequence[tuple[float, float]]) -> float:
    """Best-effort 1-call timing measurement. Returns seconds/call."""
    midpoint = [(lo + hi) / 2.0 for lo, hi in ranges]
    t0 = time.time()
    try:
        eval_fn(midpoint)
    except Exception:
        return 0.001  # assume cheap on error
    return max(time.time() - t0, 1e-6)


def hagen(
    eval_fn: Callable[[list[float]], float],
    param_ranges: Sequence[tuple[float, float]],
    *,
    param_names: Optional[list[str]] = None,
    curated_measurements: Optional[list[dict]] = None,
    guard_fn: Optional[Callable] = None,
    experience_id: str = "hagen",
    time_budget: float = 300.0,
    eval_cost_hint: Optional[float] = None,
    # LaD dispatch overrides (None = load from JSON/hardcode)
    eval_cost_threshold: Optional[float] = None,
    owl_share: Optional[float] = None,
    escalation_min_remaining: Optional[float] = None,
    confidence_skip_threshold: Optional[float] = None,
    random_restart_count: Optional[int] = None,
    reigen_inner_time_budget: Optional[float] = None,
    reigen_wall_time_factor: Optional[float] = None,
    # Misc
    force_cascade: bool = False,
    n_seed_samples: Optional[int] = None,
    hagen_cfg: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """Meta-dispatcher routing to owl or cascading owl→Reigen.

    Args:
      eval_fn: f(params) -> float, higher is better.
      param_ranges: [(lo, hi), ...]
      curated_measurements: optional precomputed points {"params": list, "score": float}
      guard_fn: optional safety metric (passed to owl).
      experience_id: cross-call learning namespace.
      time_budget: total wall budget (seconds).
      eval_cost_hint: if known, seconds/call; else auto-measured once.
      owl_share: fraction of budget owl gets before cascading to Reigen (cheap eval only).
      owl_confidence_accepted: confidence values that skip Reigen escalation.

    Returns:
      dict merging owl's structure info (dead_dims, importance, fragility,
      proxy_type, proxy_r2) with best_params/best_score. Extra keys:
        tool_used: "owl" | "owl+reigen"
        eval_cost_s: measured or provided
        route: "expensive_single" | "cheap_cascade"
        owl_result: raw owl return
        reigen_result: raw reigen return (if escalated)
    """
    from twelve.optimize import owl as _owl

    t_start = time.time()

    # ---- LaD: resolve dispatch params (kwarg > hagen_cfg > JSON > hardcode) ----
    _p = _load_hagen_params()
    _cfg_override = dict(hagen_cfg or {})
    _d = _p["dispatch"]
    _o = _p["owl_kwargs"]
    _r = _p["reigen_kwargs"]
    _eval_cost_threshold = _resolve(
        eval_cost_threshold, _cfg_override.get("eval_cost_threshold"),
        _d.get("eval_cost_threshold",
               _HARDCODE_DEFAULTS["dispatch"]["eval_cost_threshold"]))
    _owl_share = _resolve(
        owl_share, _cfg_override.get("owl_share"),
        _d.get("owl_share",
               _HARDCODE_DEFAULTS["dispatch"]["owl_share"]))
    _escalation_min_remaining = _resolve(
        escalation_min_remaining, _cfg_override.get("escalation_min_remaining"),
        _d.get("escalation_min_remaining",
               _HARDCODE_DEFAULTS["dispatch"]["escalation_min_remaining"]))
    _confidence_skip_threshold = _resolve(
        confidence_skip_threshold, _cfg_override.get("confidence_skip_threshold"),
        _d.get("confidence_skip_threshold",
               _HARDCODE_DEFAULTS["dispatch"]["confidence_skip_threshold"]))
    _owl_random_K = int(_resolve(
        random_restart_count, _cfg_override.get("random_restart_count"),
        _o.get("random_restart_count",
               _HARDCODE_DEFAULTS["owl_kwargs"]["random_restart_count"])))
    _reigen_inner = float(_resolve(
        reigen_inner_time_budget, _cfg_override.get("reigen_inner_time_budget"),
        _r.get("inner_time_budget",
               _HARDCODE_DEFAULTS["reigen_kwargs"]["inner_time_budget"])))
    _reigen_wtf = float(_resolve(
        reigen_wall_time_factor, _cfg_override.get("reigen_wall_time_factor"),
        _r.get("wall_time_factor",
               _HARDCODE_DEFAULTS["reigen_kwargs"]["wall_time_factor"])))

    # ---- Phase 0: eval cost probing ----
    if eval_cost_hint is not None:
        t_eval = float(eval_cost_hint)
    else:
        t_eval = _measure_eval_cost(eval_fn, param_ranges)

    # Expensive eval OR curated data provided → owl single-shot with full enhancements
    expensive_route = (t_eval > _eval_cost_threshold) or (
        curated_measurements is not None and len(curated_measurements) > 0)

    # ---- Phase 1: owl (always) ----
    owl_budget = (time_budget if expensive_route
                  else max(5.0, time_budget * _owl_share))
    owl_kwargs = dict(
        measurements=curated_measurements if curated_measurements else [],
        param_ranges=list(param_ranges),
        verify_fn=eval_fn,
        autonomous=True,
        max_iterations=30,
        time_budget=owl_budget,
        min_r_squared=0.1,
        experience_id=f"{experience_id}_owl",
        use_lbfgs_refinement=True,
        use_multistart_fallback=True,
        random_restart_count=_owl_random_K,
        verbose=verbose,
    )
    if param_names is not None:
        owl_kwargs["param_names"] = list(param_names)
    if guard_fn is not None:
        owl_kwargs["guard_fn"] = guard_fn
        owl_kwargs["safe_dim_analysis"] = True
    if n_seed_samples is not None:
        owl_kwargs["n_seed_samples"] = int(n_seed_samples)

    r_owl = _owl(**owl_kwargs)

    owl_best_params = r_owl.get("best_params")
    owl_best_score = r_owl.get("verified_score")
    if owl_best_score is None:
        owl_best_score = r_owl.get("best_score")
    owl_conf = r_owl.get("confidence", "")

    # ---- Phase 2: decide whether to escalate ----
    elapsed = time.time() - t_start
    remaining = max(0.0, time_budget - elapsed)
    # LaD escalation gate: use proxy_r2 continuous value (was categorical "high").
    # proxy_r2 ≥ confidence_skip_threshold → trust owl, skip Reigen.
    # Lower threshold = more aggressive cascade (Reigen fires more often).
    _owl_proxy_r2 = r_owl.get("proxy_r2") or 0.0
    _proxy_good_enough = (_owl_proxy_r2 >= _confidence_skip_threshold)
    escalate = (
        not expensive_route
        and (force_cascade or not _proxy_good_enough)
        and remaining > _escalation_min_remaining
    )

    result = {
        "best_params": owl_best_params,
        "best_score": owl_best_score,
        "tool_used": "owl",
        "route": "expensive_single" if expensive_route else "cheap_cascade",
        "eval_cost_s": t_eval,
        "owl_result": r_owl,
        "confidence": owl_conf,
        # Structure-discovery passthrough
        "dead_dims": r_owl.get("dead_dims"),
        "active_dims": r_owl.get("active_dims"),
        "fragility": r_owl.get("fragility"),
        "proxy_type": r_owl.get("proxy_type"),
        "proxy_r2": r_owl.get("proxy_r2"),
    }

    if not escalate:
        result["elapsed_s"] = time.time() - t_start
        return result

    # ---- Phase 3: Reigen escalation ----
    from twelve.agent.reigen import Reigen

    # Reigen needs guard_fn; if user didn't provide, use eval_fn as guard (single-metric case)
    _guard = guard_fn if guard_fn is not None else eval_fn

    # Use owl's best as warm-start for user params
    _init_user_params = None
    if owl_best_params is not None:
        if isinstance(owl_best_params, dict):
            names_ = param_names or list(owl_best_params.keys())
            _init_user_params = [float(owl_best_params.get(n, 0.0)) for n in names_]
        else:
            _init_user_params = [float(x) for x in owl_best_params]

    try:
        reigen_obj = Reigen(
            eval_fn=eval_fn,
            guard_fn=_guard,
            user_param_ranges=list(param_ranges),
            user_param_names=param_names,
            initial_user_params=_init_user_params,
            experience_id=f"{experience_id}_reigen",
            inner_time_budget=_reigen_inner,
            verbose=verbose,
        )
        # wall_time_factor from LaD (default 1.0 keeps total wall close to
        # time_budget; Reigen's native default 3.0 would blow user's budget).
        r_reigen = reigen_obj.run(
            time_budget=max(5.0, remaining),
            wall_time_factor=_reigen_wtf,
        )
    except Exception as e:
        r_reigen = {"error": f"{type(e).__name__}: {e}",
                    "user_best_score": float("-inf"),
                    "user_best_params": None}

    reigen_best = r_reigen.get("user_best_score", float("-inf"))
    reigen_params = r_reigen.get("user_best_params")

    # Pick winner (keep owl if reigen didn't improve)
    owl_score_val = owl_best_score if owl_best_score is not None else float("-inf")
    if reigen_best > owl_score_val and reigen_params is not None:
        result["best_params"] = reigen_params
        result["best_score"] = float(reigen_best)
        result["tool_used"] = "owl+reigen"
    else:
        result["tool_used"] = "owl(reigen_tried)"

    result["reigen_result"] = r_reigen
    result["elapsed_s"] = time.time() - t_start
    return result


__all__ = ["hagen"]
