"""eval_fn sanity checker — mimir 本番前に eval_fn の品質を自動診断する.

使い方:
    from twelve.agent.eval_check import check_eval_fn
    diag = check_eval_fn(my_eval_fn, param_ranges)
    if not diag["ok"]:
        print("⚠ eval_fn 問題:", diag["issues"])
        # 修正してから本番 mimir() を呼べ

60 秒で bad eval_fn パターンを検出、本番最適化の無駄走行を防ぐ.

検出する症状:
  - constant eval_fn (全 dim dead)
  - noisy / 多峰 / 不連続 (proxy_r2 < threshold)
  - 崩壊 factor (fragility spike)
  - 1 次元的 eval_fn (active_dims 少)
  - score 爆発 (range が不安定、PPL=262144 型)
  - dict multi-observer での observer 間非整合

背景: eval_fn = 人間の価値観定義、完全自動化は原理不可能 (bootstrap paradox)。
しかし bad pattern の自動検出で 80% の失敗は事前回避できる。
詳細: docs/全論の公式の活用.md §11.2 "universal tracker of environment"。
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional, Sequence


def _safe_finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _single_probe(eval_fn, midpoint, param_ranges, eps_frac, n_probes, rng_state):
    rng_widths = [hi - lo for (lo, hi) in param_ranges]
    scores: list[float] = []
    for _ in range(n_probes):
        delta = [eps_frac * w * rng_state.gauss(0, 1) for w in rng_widths]
        p = [m + d for m, d in zip(midpoint, delta)]
        try:
            v = eval_fn(p)
            if isinstance(v, dict):
                v = next((x for x in v.values() if _safe_finite(x) is not None), None)
            f = _safe_finite(v)
            if f is not None:
                scores.append(f)
        except Exception:
            continue
    if len(scores) < 2:
        return {"n_unique": len(scores), "n_probes": len(scores), "spread": 0.0}
    s_arr = sorted(scores)
    spread = float(s_arr[-1] - s_arr[0])
    rounded = [round(s, 12) for s in scores]
    return {
        "n_unique": int(len(set(rounded))),
        "n_probes": int(len(scores)),
        "spread": spread,
    }


def _probe_discreteness(
    eval_fn: Callable,
    midpoint: list[float],
    param_ranges: Sequence[tuple],
    *,
    eps_frac: float = 0.001,
    n_probes: int = 8,
    seed: int = 13,
) -> dict:
    """2 段階摂動で「丸め/整数化/mask decode」の離散性を検出.

    判定:
      Tier 1 (微小 eps_frac=0.001): 滑らか連続なら全 unique、丸めで潰れるなら
        n_unique 少ない or spread ほぼゼロ → 強い離散
      Tier 2 (中規模 eps_frac=0.1):  整数化 k のジャンプ境界を渡るのを期待。
        n_unique <= n_probes/2 かつ spread > 0 → step-like 弱い離散

    判定優先度:
      Tier 1 likely_discrete           → "small" 強い離散
      Tier 1 連続 + Tier 2 step-like   → "mid" 弱い離散 (整数 mask 等)
      両方とも all-unique               → 純連続
    """
    import random as _random
    rng_state = _random.Random(seed)
    small = _single_probe(eval_fn, midpoint, param_ranges,
                          eps_frac=eps_frac, n_probes=n_probes, rng_state=rng_state)

    rng_state2 = _random.Random(seed + 1)
    mid_eps = max(eps_frac * 100, 0.05)
    mid = _single_probe(eval_fn, midpoint, param_ranges,
                        eps_frac=mid_eps, n_probes=n_probes, rng_state=rng_state2)

    threshold_small = max(2, small["n_probes"] // 4)
    small_discrete = (
        small["n_probes"] >= 2 and
        (small["spread"] < 1e-9 or small["n_unique"] <= threshold_small)
    )
    threshold_mid = max(2, mid["n_probes"] // 2)
    mid_steplike = (
        mid["n_probes"] >= 4 and
        mid["n_unique"] <= threshold_mid and
        mid["spread"] > 1e-9
    )
    likely_discrete = small_discrete or mid_steplike
    if small_discrete:
        signal = "small"
    elif mid_steplike:
        signal = "mid"
    else:
        signal = "none"
    return {
        "likely_discrete": bool(likely_discrete),
        "signal": signal,
        "small": small,
        "mid": mid,
        "n_unique": small["n_unique"],     # backward compat (small tier)
        "n_probes": small["n_probes"],
        "spread": small["spread"],
        "eps_frac": eps_frac,
    }


def check_eval_fn(
    eval_fn: Callable,
    param_ranges: Sequence[tuple],
    *,
    time_budget: float = 60.0,
    proxy_r2_threshold: float = 0.2,
    fragility_spike_ratio: float = 5.0,
    min_active_dims: int = 2,
    min_dims_for_active_check: int = 5,
    n_stability_probes: int = 3,
    stochastic_cv_threshold: float = 0.10,
    check_discrete: bool = True,
    discreteness_eps_frac: float = 0.001,
    discreteness_n_probes: int = 8,
    experience_id: str = "eval_check",
    verbose: bool = False,
) -> dict:
    """Diagnose eval_fn quality via mimir structure_only scan.

    Limitation: integer + continuous mixed objectives (e.g. ``k = round(p[0])``
    plus continuous tail) often look continuous to the discreteness probe
    because score varies smoothly even when the integer flips. In those cases
    the user should call ``mimir_odin_structure_policy(param_decoder=...)``
    explicitly. Verified accuracy on 6 synthetic cases: 83% (5/6); the missed
    case is exactly this mixed integer/continuous pattern.

    Args:
      eval_fn: f(params: list[float]) -> float | dict
      param_ranges: [(lo, hi), ...]
      time_budget: diagnosis budget (default 60s is enough for most).
      proxy_r2_threshold: flag if proxy_r2 below this.
      fragility_spike_ratio: flag if max(fragility) > ratio * median.
      min_active_dims: flag if fewer active dims (when problem is >= min_dims_for_active_check).
      min_dims_for_active_check: apply active_dims check only above this dim.
      n_stability_probes: # repeated same-params calls to detect stochasticity
                         (default 3; set 1 to disable).
      stochastic_cv_threshold: robust CV threshold above which eval_fn is
                               flagged as stochastic (default 0.10).
      experience_id: namespace (kept short for sanity check runs).

    Returns:
      {
        "ok": bool,                  # no issues detected
        "issues": list[str],         # human-readable problem descriptions
        "severity": "ok" | "warn" | "fatal",
        "proxy_r2": float,
        "active_dims": list,
        "dead_dims": list,
        "fragility_max": float,
        "fragility_median": float,
        "n_dims": int,
        "elapsed_s": float,
        "probe_score": float|None,   # 1-call probe result
        "eval_fn_returned": "scalar"|"dict"|"error",
        "stochastic_cv": float|None, # robust CV across n_stability_probes repeats
        "stochastic_detected": bool, # True if CV > stochastic_cv_threshold
        "recommendations": list[str],
      }
    """
    from twelve.agent.mimir import mimir

    t0 = time.time()
    n_dims = len(list(param_ranges))
    issues = []
    recommendations = []
    severity = "ok"
    stochastic_cv = None
    stochastic_detected = False
    discreteness = None
    likely_discrete = False

    # --- Step 1: 2-point probe to catch immediate failures + constant detection ---
    midpoint = [(lo + hi) / 2.0 for lo, hi in param_ranges]
    # 2nd probe: shift each dim by 30% of range toward hi
    corner = [(lo + hi) / 2.0 + 0.3 * (hi - lo) for lo, hi in param_ranges]
    eval_returned = "scalar"
    probe_score = None
    probe_raw = None
    probe_score_2 = None
    try:
        probe_raw = eval_fn(midpoint)
        probe_raw_2 = eval_fn(corner)
        if isinstance(probe_raw, dict):
            eval_returned = "dict"
            for v in probe_raw.values():
                f = _safe_finite(v)
                if f is not None:
                    probe_score = f
                    break
            if isinstance(probe_raw_2, dict):
                for v in probe_raw_2.values():
                    f = _safe_finite(v)
                    if f is not None:
                        probe_score_2 = f
                        break
        else:
            probe_score = _safe_finite(probe_raw)
            probe_score_2 = _safe_finite(probe_raw_2)
    except Exception as e:
        eval_returned = "error"
        issues.append(f"midpoint eval raised: {type(e).__name__}: {e}")
        severity = "fatal"

    if severity == "fatal":
        return {
            "ok": False,
            "issues": issues,
            "severity": "fatal",
            "proxy_r2": 0.0,
            "active_dims": [],
            "dead_dims": [],
            "fragility_max": 0.0,
            "fragility_median": 0.0,
            "n_dims": n_dims,
            "elapsed_s": time.time() - t0,
            "probe_score": probe_score,
            "eval_fn_returned": eval_returned,
            "stochastic_cv": stochastic_cv,
            "stochastic_detected": stochastic_detected,
            "discreteness": None,
            "likely_discrete": False,
            "recommended_optimizer": None,
            "recommended_reason": "fatal: midpoint exception",
            "recommendations": ["eval_fn が midpoint で例外、範囲か実装を修正せよ"],
        }

    if probe_score is None:
        issues.append("eval_fn が non-finite or non-numeric を返した")
        severity = "fatal"
        return {
            "ok": False,
            "issues": issues,
            "severity": "fatal",
            "proxy_r2": 0.0,
            "active_dims": [],
            "dead_dims": [],
            "fragility_max": 0.0,
            "fragility_median": 0.0,
            "n_dims": n_dims,
            "elapsed_s": time.time() - t0,
            "probe_score": None,
            "eval_fn_returned": eval_returned,
            "stochastic_cv": stochastic_cv,
            "stochastic_detected": stochastic_detected,
            "discreteness": None,
            "likely_discrete": False,
            "recommended_optimizer": None,
            "recommended_reason": "fatal: non-finite return",
            "recommendations": [
                "eval_fn は finite float か全 value finite の dict を返すこと",
            ],
        }

    # Constant detection: 2 probes with different params returning same value.
    # Catches apply/restore bugs and truly constant functions that mimir's
    # correlation-based dead detection misses (corr of constant = NaN → fallback).
    constant_2pt_detected = False
    if (probe_score_2 is not None and
            abs(probe_score - probe_score_2) < 1e-12):
        issues.append(
            "2 点 probe で同一 score: eval_fn が constant の疑い "
            "(apply/restore bug / range 無関係 / 対称軸ヒット / argsort decode など)"
        )
        recommendations.append(
            "midpoint と corner で別 score が返るか debug 出力で確認。"
            "apply が効いていない/restore で巻き戻ってないか疑え。"
            " argsort/policy decode を使ってる場合は対称軸 (全 dim 同値) で tie 起こりうる、"
            "structure_policy 経路を検討。"
        )
        constant_2pt_detected = True
        severity = "fatal"

    # --- Step 1.5: stochasticity probe (repeat same midpoint N times) ---
    # Detect stochastic eval_fn (LLM sampling, RL rollout, Monte Carlo) so
    # the user knows they should wrap with lad_wrappers before running mimir.
    # Use robust CV (MAD / |median|) to handle zero-mean eval_fns.
    if n_stability_probes >= 2:
        repeat_vals = [probe_score]
        for _ in range(n_stability_probes - 1):
            try:
                rv = eval_fn(midpoint)
            except Exception:
                break
            if isinstance(rv, dict):
                f = None
                for v in rv.values():
                    f = _safe_finite(v)
                    if f is not None:
                        break
            else:
                f = _safe_finite(rv)
            if f is None:
                break
            repeat_vals.append(f)

        if len(repeat_vals) >= 2:
            s = sorted(repeat_vals)
            nr = len(s)
            med = s[nr // 2] if nr % 2 else (s[nr // 2 - 1] + s[nr // 2]) / 2
            abs_dev = sorted(abs(v - med) for v in repeat_vals)
            na = len(abs_dev)
            mad = abs_dev[na // 2] if na % 2 else (abs_dev[na // 2 - 1] + abs_dev[na // 2]) / 2
            denom = max(abs(med), 1e-9)
            stochastic_cv = mad / denom
            if stochastic_cv > stochastic_cv_threshold:
                stochastic_detected = True
                issues.append(
                    f"同一 params で {len(repeat_vals)} 回 probe、robust CV={stochastic_cv:.2f} > "
                    f"{stochastic_cv_threshold:.2f} = stochastic eval_fn の疑い強 "
                    "(LLM 生成 / RL reward / Monte Carlo 等)"
                )
                recommendations.append(
                    "wrap_stochastic(eval_fn, n=20) で集約 (scalar 互換) or "
                    "wrap_multi_obs(eval_fn, n=20) で LaD 多観測化 (mimir の multi-observer path 活性) を推奨。"
                    " 詳細: twelve/agent/lad_wrappers.py、もしくは mimir(..., n_samples_per_eval=20) で自動集約。"
                )
                if severity == "ok":
                    severity = "warn"

    # --- Step 1.7: discreteness probe (微小摂動で score 潰れるか) ---
    # 丸め / 整数化 / mask decode は近傍 params で同じ score を返す。
    # 連続的 eval_fn なら 8 点 probe で 8 unique score が出るはず。
    if check_discrete and not stochastic_detected:
        discreteness = _probe_discreteness(
            eval_fn, midpoint, param_ranges,
            eps_frac=discreteness_eps_frac,
            n_probes=discreteness_n_probes,
        )
        likely_discrete = discreteness.get("likely_discrete", False)
        if likely_discrete and discreteness["n_probes"] >= 4:
            issues.append(
                f"微小摂動 ({discreteness_eps_frac*100:.1f}% range) {discreteness['n_probes']} 点で "
                f"unique score = {discreteness['n_unique']} 個 (spread {discreteness['spread']:.2e}) "
                "= 丸め/整数化/mask decode の疑い、純連続じゃない"
            )
            recommendations.append(
                "近傍点で score が潰れるなら mimir_odin_structure_policy() を推奨。"
                " 連続 carrier を policy に decode する離散構造問題と判断 (Rule 17)。"
            )
            if severity == "ok":
                severity = "warn"

    # --- Step 2: mimir structure_only for dead/active/fragility/proxy_r2 ---
    # Use scalar-wrapping eval for structure scan if original returned dict
    # (mimir's structure_only path uses owl single-obs primarily).
    if eval_returned == "dict":
        # Wrap: pick same key as probe's first numeric to keep consistent
        first_key = None
        for k, v in probe_raw.items():
            if _safe_finite(v) is not None:
                first_key = k
                break

        def _scalar_eval(params, _key=first_key):
            r = eval_fn(params)
            if isinstance(r, dict):
                return _safe_finite(r.get(_key, 0.0)) or 0.0
            return _safe_finite(r) or 0.0

        scan_fn = _scalar_eval
    else:
        def _scalar_eval(params):
            return _safe_finite(eval_fn(params)) or 0.0
        scan_fn = _scalar_eval

    try:
        r = mimir(
            scan_fn,
            list(param_ranges),
            time_budget=time_budget,
            mode="structure_only",
            experience_id=experience_id,
            verbose=verbose,
        )
    except Exception as e:
        issues.append(f"mimir structure_only が失敗: {type(e).__name__}: {e}")
        return {
            "ok": False,
            "issues": issues,
            "severity": "fatal",
            "proxy_r2": 0.0,
            "active_dims": [],
            "dead_dims": [],
            "fragility_max": 0.0,
            "fragility_median": 0.0,
            "n_dims": n_dims,
            "elapsed_s": time.time() - t0,
            "probe_score": probe_score,
            "eval_fn_returned": eval_returned,
            "stochastic_cv": stochastic_cv,
            "stochastic_detected": stochastic_detected,
            "discreteness": discreteness,
            "likely_discrete": likely_discrete,
            "recommended_optimizer": None,
            "recommended_reason": "fatal: structure scan failed",
            "recommendations": ["param_ranges または eval_fn 実装を確認"],
        }

    proxy_r2 = float(r.get("proxy_r2") or 0.0)
    active_dims = list(r.get("active_dims") or [])
    dead_dims = list(r.get("dead_dims") or [])
    fragility = list(r.get("fragility") or [])

    # --- Step 3: diagnose ---
    # (a) constant (全 dim dead or active 空)
    if len(active_dims) == 0 or (
        len(dead_dims) == n_dims and n_dims > 0
    ):
        issues.append(
            "active_dims が空 = eval_fn が param に応答しない "
            "(constant / state leak / range 不適 等)"
        )
        recommendations.append(
            "param を変えた時に eval_fn が実際に違う値を返すか debug 出力で確認。"
            "state leak (apply/restore 漏れ) を疑え。"
        )
        severity = "fatal"

    # (b) proxy_r2 低 (noisy / 多峰 / 不連続)
    if proxy_r2 < proxy_r2_threshold:
        issues.append(
            f"proxy_r2={proxy_r2:.2f} < {proxy_r2_threshold} "
            "= eval_fn が noisy / 多峰 / 不連続"
        )
        recommendations.append(
            "eval_fn に平均化や log 変換を追加すると proxy_r2 改善することがある "
            "(ex: return -log1p(-score))。"
        )
        if severity == "ok":
            severity = "warn"

    # (c) fragility spike (崩壊 factor)
    if fragility:
        finite_frag = [_safe_finite(x) for x in fragility]
        finite_frag = [x for x in finite_frag if x is not None]
        if finite_frag:
            fmax = max(finite_frag)
            sorted_f = sorted(finite_frag)
            fmedian = sorted_f[len(sorted_f) // 2]
            if fmedian > 0 and fmax > fragility_spike_ratio * fmedian:
                spike_idx = fragility.index(fmax)
                issues.append(
                    f"param[{spike_idx}] fragility={fmax:.3f} が median={fmedian:.3f} "
                    f"の {fmax / fmedian:.1f}× = 崩壊因子 (scale=0 型 PPL 爆発リスク)"
                )
                recommendations.append(
                    f"param[{spike_idx}] の range が 0 を含まないか確認 (Rule 7)。"
                    f"安全な range (例 0.5〜1.5) に狭めるか、fragility 対策として "
                    f"対数変換・clipping を入れる。"
                )
                if severity == "ok":
                    severity = "warn"
        else:
            fmax = 0.0
            fmedian = 0.0
    else:
        fmax = 0.0
        fmedian = 0.0

    # (d) 1 次元的 eval_fn
    if (n_dims >= min_dims_for_active_check
            and len(active_dims) < min_active_dims):
        issues.append(
            f"active_dims={len(active_dims)} 個 (全 {n_dims} dim 中) "
            "= eval_fn が 1 次元的、他の param が情報に貢献してない"
        )
        recommendations.append(
            "eval_fn に追加の測定成分を加えるか、dead な param を range から除く "
            "(次元削減)。"
        )
        if severity == "ok":
            severity = "warn"

    # (e) dict eval で observer 非整合 (multi-observer の stable_active チェック)
    if eval_returned == "dict":
        owl_raw = r.get("owl_result", {})
        stable_active = owl_raw.get("stable_active")
        if stable_active is not None and len(stable_active) == 0:
            issues.append(
                "multi-observer で stable_active が空 = observer 間で "
                "「何が効くか」が全く一致してない = eval_fn 指標として非整合"
            )
            recommendations.append(
                "dict の各 metric が同じ物理量を測ってるか見直し。"
                "1 つを guard_fn に分離する方が適切かも。"
            )
            if severity == "ok":
                severity = "warn"

    ok = len(issues) == 0
    if ok:
        recommendations.append(
            f"eval_fn OK (proxy_r2={proxy_r2:.2f}, active={len(active_dims)}/"
            f"{n_dims} dim)。本番 mimir に進んでよい。"
        )

    # --- Step 4: optimizer recommendation (stable vs structure_policy) ---
    # 判定優先度:
    #   fatal             → (修正後再診断、optimizer 推奨は保留)
    #   likely_discrete   → mimir_odin_structure_policy
    #   stochastic        → mimir_odin_stable + lad wrapper
    #   default           → mimir_odin_stable
    if severity == "fatal":
        # 2 点 constant fatal だが discreteness probe で動きが見えるなら
        # 対称軸 tie / argsort decode の誤発動 → structure_policy 推奨に救済
        if (constant_2pt_detected and discreteness and
                discreteness["small"]["n_unique"] >= 2):
            recommended_optimizer = "mimir_odin_structure_policy"
            recommended_reason = (
                "2 点 probe constant 誤発動の疑い (discrete probe で n_unique="
                f"{discreteness['small']['n_unique']}/{discreteness['small']['n_probes']} 反応)、"
                "argsort/policy decode の対称軸 tie で structure_policy 推奨 (Rule 17)"
            )
        else:
            recommended_optimizer = None
            recommended_reason = "fatal issue を修正してから再診断"
    elif likely_discrete:
        recommended_optimizer = "mimir_odin_structure_policy"
        recommended_reason = (
            f"discrete probe で n_unique={discreteness['n_unique']}/"
            f"{discreteness['n_probes']}、近傍 score 潰れ検出 (Rule 17)"
        )
    elif stochastic_detected:
        recommended_optimizer = "mimir_odin_stable"
        recommended_reason = (
            f"stochastic CV={stochastic_cv:.2f} 検出、wrap_multi_obs / "
            "n_samples_per_eval=20 で LaD 化してから stable 投入 (Rule 12)"
        )
    else:
        recommended_optimizer = "mimir_odin_stable"
        recommended_reason = "純連続検出、stable が default (Rule 9)"
    if recommended_optimizer:
        recommendations.append(f"→ 推奨: {recommended_optimizer}() ({recommended_reason})")

    return {
        "ok": ok,
        "issues": issues,
        "severity": severity,
        "proxy_r2": proxy_r2,
        "active_dims": active_dims,
        "dead_dims": dead_dims,
        "fragility_max": float(fmax),
        "fragility_median": float(fmedian),
        "n_dims": n_dims,
        "elapsed_s": time.time() - t0,
        "probe_score": probe_score,
        "eval_fn_returned": eval_returned,
        "stochastic_cv": stochastic_cv,
        "stochastic_detected": stochastic_detected,
        "discreteness": discreteness,
        "likely_discrete": likely_discrete,
        "recommended_optimizer": recommended_optimizer,
        "recommended_reason": recommended_reason,
        "recommendations": recommendations,
    }


def format_report(diag: dict) -> str:
    """Pretty-print a check_eval_fn() result for CLI usage."""
    lines = []
    icon = {"ok": "✅", "warn": "⚠", "fatal": "❌"}[diag.get("severity", "ok")]
    lines.append(f"{icon} eval_fn check: severity={diag.get('severity')}")
    lines.append(f"   proxy_r2      : {diag['proxy_r2']:.3f}")
    lines.append(f"   active / dead : {len(diag['active_dims'])}/{diag['n_dims']} "
                 f"active, {len(diag['dead_dims'])} dead")
    lines.append(f"   fragility     : max={diag['fragility_max']:.3f} "
                 f"median={diag['fragility_median']:.3f}")
    lines.append(f"   elapsed       : {diag['elapsed_s']:.1f}s")
    lines.append(f"   probe_score   : {diag.get('probe_score')}")
    cv = diag.get("stochastic_cv")
    if cv is not None:
        tag = " (stochastic!)" if diag.get("stochastic_detected") else ""
        lines.append(f"   stochastic_cv : {cv:.3f}{tag}")
    disc = diag.get("discreteness")
    if disc:
        tag = " (discrete!)" if diag.get("likely_discrete") else ""
        lines.append(f"   discrete probe: n_unique={disc['n_unique']}/"
                     f"{disc['n_probes']} spread={disc['spread']:.2e}{tag}")
    rec_opt = diag.get("recommended_optimizer")
    if rec_opt:
        lines.append(f"   recommended   : {rec_opt}() — "
                     f"{diag.get('recommended_reason', '')}")
    if diag["issues"]:
        lines.append("   issues:")
        for i in diag["issues"]:
            lines.append(f"     - {i}")
    if diag["recommendations"]:
        lines.append("   recommendations:")
        for r in diag["recommendations"]:
            lines.append(f"     → {r}")
    return "\n".join(lines)


__all__ = ["check_eval_fn", "format_report"]
