"""Sentinel — eval_fnで最適化、guard_fnで見張る。NGなら自動修正。

Usage:
    result = Sentinel(
        eval_fn=ppl_eval,           # これで最適化（いつも通り）
        guard_fn=hellaswag_eval,    # これで見張る（Sentinelの追加分）
        param_ranges=[(0.5, 1.5)] * 61,
        initial_params=known_good,  # 離散空間用の warm-start (optional)
        learn=True,                 # 経験蓄積 (optional)
    ).run(time_budget=900)

    result["verdict"]        # "approved" / "pivoted" / "failed"
    result["best_params"]    # 最適パラメータ
    result["safe_dims"]      # 全指標で安全な次元 (pivoted時)
    result["conflict_dims"]  # 指標間で矛盾する次元 (pivoted時)

内部動作:
    Step 1: owl(eval_fn) で最適化 ← いつものowl()フルパワー
        proxy R² < min_r_squared なら optimize() に自動フォールバック (離散空間対応)
    Step 2: guard_fn(best_params) で見張り ← ベースライン以上か？
    Step 3: NG → multi-observer → safe_dims特定 → safe_dimsだけ再最適化
        multi-observer が無効 (R² 低) なら全dimsをsafe扱いで保守的に継続
"""
import random
import time

from twelve.optimize import owl, _owl_multi_observer


class Sentinel:
    """eval_fnで最適化、guard_fnで見張る。NGなら自動修正。"""

    def __init__(self, eval_fn, guard_fn, param_ranges,
                 param_names=None, experience_id=None,
                 initial_params=None, learn=False, min_r_squared=0.3,
                 meta=False, meta_config=None):
        self.eval_fn = eval_fn
        self.guard_fn = guard_fn
        self.param_ranges = param_ranges
        self.n_dims = len(param_ranges)
        self.param_names = param_names or [f"p{i}" for i in range(self.n_dims)]
        self.experience_id = experience_id
        self.initial_params = initial_params
        self.learn = learn
        self.min_r_squared = min_r_squared
        # Phase3 meta-evolution (K7-K12). Only applies to optimize() fallback path.
        # Use for long budgets (>1hr) or repeated runs where meta-evolving the
        # optimizer itself pays off.
        self.meta = meta
        self.meta_config = meta_config
        self.baseline_params = [(lo + hi) / 2 for lo, hi in param_ranges]
        # seed for search ≠ baseline for guard comparison
        self.seed_params = (list(initial_params)
                            if initial_params is not None
                            else self.baseline_params)

    def run(self, time_budget=600, verbose=True):
        """eval_fnで最適化 → guard_fnで見張り → NG時は自動修正。"""
        t0 = time.time()

        if verbose:
            print("=" * 60)
            print(f"  Sentinel | {self.n_dims}D, {time_budget}s")
            print(f"  eval_fn:  {_fn_name(self.eval_fn)}")
            print(f"  guard_fn: {_fn_name(self.guard_fn)}")
            if self.initial_params is not None:
                print(f"  initial_params: {self.initial_params[:6]}"
                      f"{'...' if len(self.initial_params) > 6 else ''}")
            print("=" * 60)

        # Step 1: eval_fnで初期データ収集 + owl()フルパワー最適化
        measurements = self._collect(max(20, self.n_dims), verbose)
        opt_result = self._optimize(measurements, time_budget * 0.6, verbose)

        best_params = opt_result.get("best_params")
        if isinstance(best_params, dict):
            best_params = [best_params.get(f"p{i}", self.baseline_params[i])
                           for i in range(self.n_dims)]
        if best_params is None:
            best_params = self.baseline_params[:]

        opt_score = opt_result.get("verified_score") or opt_result.get("best_score") or 0.0

        # Track best-across-all-measurements (any eval_fn call during the run)
        # measurements + best_params from optimizer = all points evaluated.
        best_ever_score = float("-inf")
        best_ever_params = None
        for m in measurements:
            s = float(m.get("score", float("-inf")))
            if s > best_ever_score:
                best_ever_score = s
                best_ever_params = list(m["params"])
        # Include the optimizer's verified best
        if opt_score > best_ever_score:
            best_ever_score = float(opt_score)
            best_ever_params = list(best_params) if best_params is not None else None

        # Step 2: guard_fnで見張り
        baseline_guard = float(self.guard_fn(self.baseline_params))
        best_guard = float(self.guard_fn(best_params))

        if verbose:
            print(f"\n  [Guard] baseline={baseline_guard:.4f}, "
                  f"optimized={best_guard:.4f}")

        if best_guard >= baseline_guard:
            if verbose:
                print(f"  [Verdict] APPROVED")
            return self._build_result(
                best_params, opt_score, best_guard, baseline_guard,
                verdict="approved", elapsed=time.time() - t0,
                proxy_r2=opt_result.get("proxy_r2", 0),
                optimization_mode=opt_result.get("confidence", "owl"),
                best_ever_params=best_ever_params,
                best_ever_score=best_ever_score)

        # Step 3: NG → 診断 → 自動修正
        if verbose:
            print(f"  [Verdict] NG — guard dropped "
                  f"({best_guard:.4f} < {baseline_guard:.4f})")
            print(f"  [Diagnose] multi-observer分析中...")

        mo_result = self._diagnose(measurements, verbose)

        remaining = max(30, time_budget - (time.time() - t0))
        pivot_params, pivot_score = self._pivot(
            mo_result, measurements, remaining, verbose)

        pivot_guard = float(self.guard_fn(pivot_params))
        verdict = "pivoted" if pivot_guard >= baseline_guard else "failed"

        if verbose:
            safe = mo_result.get("stable_active", [])
            conflict = mo_result.get("observer_dependent", [])
            print(f"\n  [Pivot] guard={pivot_guard:.4f} "
                  f"(baseline={baseline_guard:.4f})")
            print(f"  [Pivot] safe={len(safe)}, conflict={len(conflict)}")
            print(f"  [Verdict] {verdict.upper()}")

        # Update best_ever to include pivot result too
        if pivot_score > best_ever_score:
            best_ever_score = float(pivot_score)
            best_ever_params = list(pivot_params) if pivot_params is not None else None

        return self._build_result(
            pivot_params, pivot_score, pivot_guard, baseline_guard,
            verdict=verdict, elapsed=time.time() - t0,
            proxy_r2=opt_result.get("proxy_r2", 0),
            optimization_mode=opt_result.get("confidence", "owl"),
            best_ever_params=best_ever_params,
            best_ever_score=best_ever_score,
            mo_result=mo_result)

    def _collect(self, n_samples, verbose=False):
        """n点をeval_fnで測定。seed_paramsがあれば最初のサンプルに使う。"""
        rng = random.Random(42)
        measurements = []
        # First: seed (== initial_params if provided, else midpoint)
        s = float(self.eval_fn(self.seed_params))
        measurements.append({"params": list(self.seed_params), "score": s})
        # Add midpoint too if different from seed (diversity)
        if self.initial_params is not None:
            s2 = float(self.eval_fn(self.baseline_params))
            measurements.append({"params": list(self.baseline_params), "score": s2})
        # Random samples
        while len(measurements) < n_samples:
            sample = [rng.uniform(lo, hi) for lo, hi in self.param_ranges]
            s = float(self.eval_fn(sample))
            measurements.append({"params": sample, "score": s})
        if verbose:
            print(f"  [Collect] {len(measurements)} measurements")
        return measurements

    def _optimize(self, measurements, budget, verbose=False):
        """owl first, optimize() fallback if proxy insufficient.

        owl is fast (proxy-based, ~20 evals) but requires smooth landscape.
        optimize() is slower but works on discrete/non-smooth spaces.
        """
        t_start = time.time()

        if verbose:
            print(f"  [Optimize] owl(eval_fn, autonomous, {budget * 0.5:.0f}s)")

        # Phase A: Try owl (50% of budget)
        result = owl(
            measurements=measurements,
            param_ranges=self.param_ranges,
            param_names=self.param_names,
            verify_fn=self.eval_fn,
            autonomous=True,
            time_budget=budget * 0.5,
            experience_id=self.experience_id,
            verbose=verbose,
        )

        # Detect insufficiency
        insufficient = (
            result.get("best_params") is None
            or result.get("confidence") == "insufficient"
            or float(result.get("proxy_r2") or 0.0) < self.min_r_squared
        )

        # Additional check: if owl's verified score is worse than the best raw
        # measurement, owl's proxy is misleading (common in discrete spaces where
        # R² looks OK but predicted optimum is wrong). Fall back.
        if not insufficient:
            best_m = max(measurements, key=lambda m: m.get("score", float("-inf")))
            best_m_score = float(best_m.get("score", float("-inf")))
            verified = result.get("verified_score")
            if verified is not None and float(verified) < best_m_score:
                if verbose:
                    print(f"  [Fallback] owl verified_score={float(verified):.3f} "
                          f"< best measurement {best_m_score:.3f}. Proxy misleading.")
                insufficient = True

        if not insufficient:
            return result

        # Phase B: Fallback to optimize() (direct search)
        from twelve.optimize import optimize

        elapsed = time.time() - t_start
        remaining = max(30, budget - elapsed)

        # Pick warm-start: best measurement seen so far
        best_m = max(measurements, key=lambda m: m.get("score", float("-inf")))
        warm = (best_m["params"] if isinstance(best_m["params"], list)
                else list(best_m["params"]))

        if verbose:
            print(f"  [Fallback] owl insufficient "
                  f"(R²={float(result.get('proxy_r2') or 0):.3f} < {self.min_r_squared}). "
                  f"Direct optimize({remaining:.0f}s) from warm-start")

        best_params, best_score, info = optimize(
            eval_fn=self.eval_fn,
            param_ranges=self.param_ranges,
            time_budget=remaining,
            initial_params=warm,
            learn=self.learn,
            experience_id=self.experience_id,
            meta=self.meta,
            meta_config=self.meta_config,
            verbose=verbose,
        )

        # Wrap optimize's 3-tuple to owl-dict schema so downstream is unchanged
        return {
            "best_params": list(best_params) if best_params is not None else None,
            "best_score": float(best_score) if best_score is not None else 0.0,
            "verified_score": float(best_score) if best_score is not None else 0.0,
            "proxy_r2": 0.0,
            "proxy_type": "direct_search",
            "active_dims": list(range(self.n_dims)),
            "dead_dims": [],
            "param_names": self.param_names,
            "confidence": "direct",
            "n_measurements": len(measurements) + int(info.get("total_evals", 0) or 0),
            "rounds_completed": 1,
        }

    def _diagnose(self, measurements, verbose=False):
        """eval_fnデータにguard_fnスコア追加 → multi-observer分析。"""
        multi_data = []
        for m in measurements:
            guard_score = float(self.guard_fn(m["params"]))
            multi_data.append({
                "params": m["params"],
                "scores": {"eval": m["score"], "guard": guard_score},
            })
        return _owl_multi_observer(multi_data, param_names=self.param_names,
                                   verbose=verbose)

    def _pivot(self, mo_result, orig_measurements, budget, verbose=False):
        """safe_dimsだけでguard_fnを使って再最適化。多観測者無効時は全dim保守的に。"""
        safe = mo_result.get("stable_active", [])

        # CONSERVATIVE FALLBACK: multi-observer inconclusive -> treat all dims as safe
        observers = mo_result.get("observers") or {}
        max_r2 = max(
            (float(o.get("proxy_r2") or 0) for o in observers.values()),
            default=0.0
        )
        if not safe or max_r2 < self.min_r_squared:
            if verbose:
                print(f"  [Pivot] multi-observer inconclusive "
                      f"(max R²={max_r2:.3f}). Treating all {self.n_dims} dims as safe")
            safe = list(range(self.n_dims))

        safe_ranges = [self.param_ranges[d] for d in safe]
        default = self.baseline_params[:]

        def safe_guard(safe_params):
            full = default[:]
            for i, d in enumerate(safe):
                full[d] = safe_params[i]
            return self.guard_fn(full)

        safe_meas = []
        for m in orig_measurements:
            guard_score = float(self.guard_fn(m["params"]))
            safe_params = [m["params"][d] for d in safe]
            safe_meas.append({"params": safe_params, "score": guard_score})

        if verbose:
            print(f"  [Pivot] {len(safe)} dims, guard_fn re-optimize, "
                  f"{budget:.0f}s")

        # Try owl first (50% of budget)
        exp_pivot = f"{self.experience_id}_pivot" if self.experience_id else None
        result = owl(
            measurements=safe_meas,
            param_ranges=safe_ranges,
            verify_fn=safe_guard,
            autonomous=True,
            time_budget=budget * 0.5,
            experience_id=exp_pivot,
            verbose=verbose,
        )

        # Fallback if owl insufficient
        bp = result.get("best_params")
        insufficient = (
            bp is None
            or result.get("confidence") == "insufficient"
            or float(result.get("proxy_r2") or 0.0) < self.min_r_squared
        )

        if insufficient:
            from twelve.optimize import optimize
            remaining = max(30, budget * 0.5)

            # Warm-start from best guard-score measurement
            best_m = max(safe_meas, key=lambda m: m.get("score", float("-inf")))
            warm = best_m["params"]

            if verbose:
                print(f"  [Pivot-Fallback] owl insufficient, "
                      f"direct optimize({remaining:.0f}s)")

            bp, bs, info = optimize(
                eval_fn=safe_guard,
                param_ranges=safe_ranges,
                time_budget=remaining,
                initial_params=warm,
                learn=self.learn,
                experience_id=exp_pivot,
                meta=self.meta,
                meta_config=self.meta_config,
                verbose=verbose,
            )
            score = float(bs) if bs is not None else 0.0
        else:
            score = float(result.get("verified_score")
                          or result.get("best_score") or 0.0)

        # Reconstruct full-length params
        full_params = default[:]
        if bp is not None:
            if isinstance(bp, dict):
                bp = [bp.get(f"p{i}", (safe_ranges[i][0] + safe_ranges[i][1]) / 2)
                      for i in range(len(safe))]
            for i, d in enumerate(safe):
                full_params[d] = bp[i]

        return full_params, score

    def _build_result(self, best_params, eval_score, guard_score,
                      baseline_guard, verdict, elapsed, proxy_r2,
                      optimization_mode="owl", mo_result=None,
                      best_ever_params=None, best_ever_score=None):
        # best_ever = max over (all measurements, optimizer result, pivot result)
        # If None (shouldn't happen), fall back to returned best_params/eval_score
        if best_ever_score is None:
            best_ever_score = float(eval_score)
            best_ever_params = list(best_params) if best_params is not None else None

        r = {
            "verdict": verdict,
            "best_params": best_params,
            "eval_score": float(eval_score),
            "guard_score": float(guard_score),
            "baseline_guard": float(baseline_guard),
            "proxy_r2": float(proxy_r2),
            "optimization_mode": optimization_mode,
            "elapsed_s": round(elapsed, 1),
            # best across every eval_fn call in this run (catches the real peak)
            "best_ever_score": float(best_ever_score),
            "best_ever_params": best_ever_params,
            "safe_dims": None,
            "conflict_dims": None,
            "multi_observer": None,
        }
        if mo_result:
            r["safe_dims"] = mo_result.get("stable_active")
            r["conflict_dims"] = mo_result.get("observer_dependent")
            r["multi_observer"] = mo_result
        return r


def _fn_name(fn):
    return getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
