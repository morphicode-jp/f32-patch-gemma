"""mimir — owl と Reigen の上位層 meta-dispatcher (2026-04-20 default entry).

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
  dispatch 閾値と owl/Reigen kwargs は mimir_params.json で JSON 制御可能。
  precedence: 明示 kwarg > JSON > hardcode fallback。K² 自己最適化対応。

Benchmark 実績 (2026-04-20、25s budget、3 seeds):
  Rastrigin 5d   gap 0 ✅
  Ackley 5d      gap 0 ✅
  Styblinski 5d  gap -0.001 ✅ (owl 単独 +18.4 を Reigen cascade で救済)
  Rosenbrock 5d  gap 0.081 (basinhopping 0 に 0.08 差の 2 位)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional, Sequence


_MIMIR_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "mimir_params.json")

_HARDCODE_DEFAULTS = {
    "dispatch": {
        "eval_cost_threshold": 0.5,
        "owl_share": 0.4,
        "escalation_min_remaining": 5.0,
        "confidence_skip_threshold": 0.7,
        "scipy_cascade_dim_threshold": 10,
        "scipy_cascade_budget_share": 0.5,
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
    "scipy_kwargs": {
        "niter": 200,
        "minimizer_method": "L-BFGS-B",
    },
}


def _load_mimir_params() -> dict:
    """Load mimir_params.json, falling back to hardcoded defaults silently."""
    try:
        with open(_MIMIR_PARAMS_PATH, "r", encoding="utf-8") as f:
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


def mimir(
    eval_fn: Callable[[list[float]], float],
    param_ranges: Sequence[tuple[float, float]],
    *,
    param_names: Optional[list[str]] = None,
    curated_measurements: Optional[list[dict]] = None,
    guard_fn: Optional[Callable] = None,
    experience_id: str = "mimir",
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
    scipy_cascade_dim_threshold: Optional[int] = None,
    scipy_cascade_budget_share: Optional[float] = None,
    enable_scipy_cascade: bool = True,
    # Misc
    force_cascade: bool = False,
    n_seed_samples: Optional[int] = None,
    mimir_cfg: Optional[dict] = None,
    mode: str = "optimize",  # "optimize" | "structure_only"
    thread_safe_eval: bool = True,  # False: eval_fn touches global state (Rule 11); forces sequential cascade
    batch_eval_fn: Optional[Callable] = None,  # f(params_list) -> scores_list, for GPU/vectorized eval
    batch_size: int = 8,                        # batch size for batch_eval_fn (5090 LLM sweet spot)
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

    # ---- LaD: resolve dispatch params (kwarg > mimir_cfg > JSON > hardcode) ----
    _p = _load_mimir_params()
    _cfg_override = dict(mimir_cfg or {})
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
    _scipy_dim_thresh = int(_resolve(
        scipy_cascade_dim_threshold,
        _cfg_override.get("scipy_cascade_dim_threshold"),
        _d.get("scipy_cascade_dim_threshold",
               _HARDCODE_DEFAULTS["dispatch"]["scipy_cascade_dim_threshold"])))
    _scipy_budget_share = float(_resolve(
        scipy_cascade_budget_share,
        _cfg_override.get("scipy_cascade_budget_share"),
        _d.get("scipy_cascade_budget_share",
               _HARDCODE_DEFAULTS["dispatch"]["scipy_cascade_budget_share"])))

    # ---- Phase 0: eval cost probing ----
    if eval_cost_hint is not None:
        t_eval = float(eval_cost_hint)
    else:
        t_eval = _measure_eval_cost(eval_fn, param_ranges)

    # ---- Phase 0b: pre-batched seed collection (batch_eval_fn + no curated) ----
    # GPU/vectorized eval path: collect N=batch_size*2 initial points in ONE
    # batched call instead of N sequential calls. For LLM (~1s/call), this
    # turns ~20s seed collection into ~3s. Resulting seed becomes curated
    # for owl — routes to expensive_single with rich initial data.
    if batch_eval_fn is not None and curated_measurements is None:
        try:
            import random as _rr
            _batch_rng = _rr.Random(hash(experience_id) & 0xFFFFFFFF)
            _n_seed = max(int(batch_size) * 2,
                          max(20, len(list(param_ranges)) + 2))
            _seed_params = [
                [_batch_rng.uniform(lo, hi) for lo, hi in param_ranges]
                for _ in range(_n_seed)
            ]
            _seed_scores = batch_eval_fn(_seed_params)
            _seed_measurements = [
                {"params": list(p), "score": float(s)}
                for p, s in zip(_seed_params, _seed_scores)
            ]
            # Replace curated_measurements so owl gets batched seed
            curated_measurements = _seed_measurements
            if verbose:
                print(f"  [batch seed] {_n_seed} points via batch_eval_fn")
        except Exception as _bsc_e:
            if verbose:
                print(f"  [batch seed] failed: {type(_bsc_e).__name__}; falling back to eval_fn path")

    # Expensive eval OR curated data provided → owl single-shot with full enhancements
    expensive_route = (t_eval > _eval_cost_threshold) or (
        curated_measurements is not None and len(curated_measurements) > 0)

    # ---- mode="structure_only": skip optimization, run owl just long enough
    #       to extract dead_dims / importance / fragility / proxy_r2.
    #       autonomous=False, no verify_fn iterations, no L-BFGS refinement.
    #       Useful for analysis phase before committing to full optimization.
    if mode == "structure_only":
        owl_budget = min(float(time_budget), 30.0)
        owl_kwargs = dict(
            measurements=curated_measurements if curated_measurements else [],
            param_ranges=list(param_ranges),
            verify_fn=eval_fn,
            autonomous=False,           # no growing loop
            max_iterations=1,
            time_budget=owl_budget,
            min_r_squared=0.1,
            experience_id=f"{experience_id}_structure",
            use_lbfgs_refinement=False,
            use_multistart_fallback=False,
            random_restart_count=0,
            verbose=verbose,
        )
        if param_names is not None:
            owl_kwargs["param_names"] = list(param_names)
        if n_seed_samples is not None:
            owl_kwargs["n_seed_samples"] = int(n_seed_samples)
        r_owl = _owl(**owl_kwargs)
        return {
            "mode": "structure_only",
            "best_params": r_owl.get("best_params"),
            "best_score": (r_owl.get("verified_score")
                           or r_owl.get("best_score")),
            "tool_used": "owl_structure_only",
            "route": "structure_only",
            "eval_cost_s": t_eval,
            "dead_dims": r_owl.get("dead_dims"),
            "active_dims": r_owl.get("active_dims"),
            "fragility": r_owl.get("fragility"),
            "proxy_type": r_owl.get("proxy_type"),
            "proxy_r2": r_owl.get("proxy_r2"),
            "confidence": r_owl.get("confidence"),
            # owl's 構造発見 extras (v2 2026-04-21)
            "verified_score": r_owl.get("verified_score"),
            "param_names": r_owl.get("param_names"),
            "rounds_completed": r_owl.get("rounds_completed"),
            "recovered_dims": r_owl.get("recovered_dims"),
            "n_measurements": r_owl.get("n_measurements"),
            "owl_result": r_owl,
            "elapsed_s": time.time() - t_start,
        }

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
    # LaD escalation gate: AND check (confidence AND proxy_r2).
    # Skip Reigen only when BOTH conditions hold:
    #   (a) confidence == "high" — owl's proxy succeeded standalone
    #   (b) proxy_r2 >= threshold — cross-check continuous value
    # This fixes 20d Rosenbrock failure where proxy_r2=0.9996 (very high)
    # but owl's L-BFGS improved the result (confidence="lbfgs_refined"),
    # indicating the proxy was insufficient. AND check catches this case
    # because "lbfgs_refined" ≠ "high" triggers cascade.
    _owl_proxy_r2 = r_owl.get("proxy_r2") or 0.0
    _proxy_good_enough = (
        owl_conf == "high"
        and _owl_proxy_r2 >= _confidence_skip_threshold
    )
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
        # owl's 構造発見 extras (v2 2026-04-21)
        "verified_score": r_owl.get("verified_score"),
        "param_names": r_owl.get("param_names"),
        "rounds_completed": r_owl.get("rounds_completed"),
        "recovered_dims": r_owl.get("recovered_dims"),
        "n_measurements": r_owl.get("n_measurements"),
    }

    if not escalate:
        result["elapsed_s"] = time.time() - t_start
        return result

    # ---- Phase 3: Reigen escalation ----
    from twelve.agent.reigen import Reigen

    # Reigen needs guard_fn; if user didn't provide, use eval_fn as guard (single-metric case)
    _guard = guard_fn if guard_fn is not None else eval_fn

    # Warm-start Reigen from owl's best ONLY when owl succeeded (conf="high").
    # If owl fell back (lbfgs_refined / direct / insufficient), its best may
    # be in a wrong basin — warm-starting Reigen there traps it (observed on
    # 5d Rosenbrock: warm-start from owl's -4 kept Reigen at -4, while
    # fresh _collect Reigen reached -0.08).
    _init_user_params = None
    _owl_succeeded = (owl_conf == "high")
    if _owl_succeeded and owl_best_params is not None:
        if isinstance(owl_best_params, dict):
            names_ = param_names or list(owl_best_params.keys())
            _init_user_params = [float(owl_best_params.get(n, 0.0)) for n in names_]
        else:
            _init_user_params = [float(x) for x in owl_best_params]

    # ---- Phase 3 + 3b: parallel cascade (reigen || scipy) ----
    # Previously sequential: owl → reigen → scipy. Wall = owl + reigen + scipy.
    # Now parallel: owl → (reigen || scipy concurrently). Wall = owl + max(r, s).
    # Each algorithm gets the FULL remaining budget (not split), since they
    # run in separate threads. For pure Python eval_fn GIL limits parallelism
    # but scipy/numpy internal operations release GIL, giving partial overlap.
    import concurrent.futures as _cf

    _dim = len(list(param_ranges))
    _scipy_trigger = (
        enable_scipy_cascade
        and _dim >= _scipy_dim_thresh
        and owl_conf != "high"
    )
    _cascade_budget = max(5.0, remaining)

    def _reigen_worker():
        try:
            reigen_kwargs = dict(
                eval_fn=eval_fn,
                guard_fn=_guard,
                user_param_ranges=list(param_ranges),
                user_param_names=param_names,
                initial_user_params=_init_user_params,
                experience_id=f"{experience_id}_reigen",
                inner_time_budget=_reigen_inner,
                verbose=verbose,
            )
            # Passthrough batch_eval_fn for GPU/vectorized speedup inside
            # Reigen's inner Sentinel._collect (20 pts per inner → batch).
            if batch_eval_fn is not None:
                reigen_kwargs["batch_eval_fn"] = batch_eval_fn
                reigen_kwargs["batch_size"] = int(batch_size)
            reigen_obj = Reigen(**reigen_kwargs)
            return reigen_obj.run(
                time_budget=_cascade_budget,
                wall_time_factor=_reigen_wtf,
            )
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}",
                    "user_best_score": float("-inf"),
                    "user_best_params": None}

    def _scipy_worker():
        if not _scipy_trigger:
            return None
        try:
            from scipy.optimize import basinhopping as _bh
            import numpy as _np
            _scipy_t0 = time.time()
            _rng_np = _np.random.default_rng(hash(experience_id) & 0xFFFFFFFF)
            _x0 = _np.array([_rng_np.uniform(lo, hi) for lo, hi in param_ranges])
            _scipy_best_tracker = [float("-inf"), None]

            def _scipy_neg(p):
                try:
                    v = float(eval_fn(list(p)))
                except Exception:
                    return 0.0
                if v > _scipy_best_tracker[0]:
                    _scipy_best_tracker[0] = v
                    _scipy_best_tracker[1] = [float(x) for x in p]
                return -v

            def _scipy_cb(x, v, accepted):
                return time.time() - _scipy_t0 >= _cascade_budget * _scipy_budget_share

            _r_bh = _bh(
                _scipy_neg, _x0,
                minimizer_kwargs={
                    "method": _HARDCODE_DEFAULTS["scipy_kwargs"]["minimizer_method"],
                    "bounds": list(param_ranges),
                },
                niter=int(_HARDCODE_DEFAULTS["scipy_kwargs"]["niter"]),
                seed=hash(experience_id) & 0xFFFFFFFF,
                callback=_scipy_cb,
            )
            _sb = _scipy_best_tracker[0]
            _sp = _scipy_best_tracker[1]
            if _sb == float("-inf"):
                _sb = -float(_r_bh.fun)
                _sp = [float(x) for x in _r_bh.x]
            return {
                "best_score": _sb, "best_params": _sp,
                "n_iter": int(_HARDCODE_DEFAULTS["scipy_kwargs"]["niter"]),
                "elapsed_s": time.time() - _scipy_t0,
            }
        except ImportError:
            return {"error": "scipy.optimize unavailable"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # Launch cascade: parallel when eval_fn is thread-safe, sequential when not.
    # Rule 11 cases (eval_fn touches global model state like KV scales) must
    # use thread_safe_eval=False to avoid race between Reigen and scipy
    # calling eval_fn concurrently.
    if thread_safe_eval:
        with _cf.ThreadPoolExecutor(max_workers=2) as _ex:
            _f_reigen = _ex.submit(_reigen_worker)
            _f_scipy = _ex.submit(_scipy_worker) if _scipy_trigger else None
            try:
                r_reigen = _f_reigen.result(timeout=_cascade_budget * 2.0)
            except _cf.TimeoutError:
                r_reigen = {"error": "reigen timeout",
                            "user_best_score": float("-inf"),
                            "user_best_params": None}
            if _f_scipy is not None:
                try:
                    r_scipy = _f_scipy.result(timeout=_cascade_budget * 2.0)
                except _cf.TimeoutError:
                    r_scipy = {"error": "scipy timeout"}
            else:
                r_scipy = None
    else:
        # Sequential fallback: Reigen first, then scipy (if triggered).
        # Each call owns the eval_fn for its duration — no cross-thread races.
        r_reigen = _reigen_worker()
        r_scipy = _scipy_worker() if _scipy_trigger else None

    reigen_best = r_reigen.get("user_best_score", float("-inf"))
    reigen_params = r_reigen.get("user_best_params")
    scipy_best = float("-inf")
    scipy_params = None
    if r_scipy is not None and not r_scipy.get("error"):
        scipy_best = r_scipy.get("best_score", float("-inf"))
        scipy_params = r_scipy.get("best_params")

    # Pick winner among {owl, reigen, scipy}
    owl_score_val = owl_best_score if owl_best_score is not None else float("-inf")
    candidates = [("owl", owl_score_val, owl_best_params)]
    if reigen_params is not None:
        candidates.append(("reigen", float(reigen_best), reigen_params))
    if scipy_params is not None:
        candidates.append(("scipy", float(scipy_best), scipy_params))
    # Highest score wins
    winner = max(candidates, key=lambda c: c[1])
    _tag, _score, _params = winner
    if _tag == "owl":
        result["tool_used"] = ("owl(reigen_tried)" if scipy_params is None
                               else "owl(reigen_scipy_tried)")
    elif _tag == "reigen":
        result["best_params"] = _params
        result["best_score"] = float(_score)
        result["tool_used"] = "owl+reigen"
    else:  # scipy
        result["best_params"] = _params
        result["best_score"] = float(_score)
        result["tool_used"] = "owl+scipy"

    result["reigen_result"] = r_reigen
    if r_scipy is not None:
        result["scipy_result"] = r_scipy
    result["elapsed_s"] = time.time() - t_start
    return result


__all__ = ["mimir"]
