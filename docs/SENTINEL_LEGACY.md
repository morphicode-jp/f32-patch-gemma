# Sentinel — safe 2-metric optimization (legacy)

**用途**: 2026-04-19 以前の 2 指標最適化 API。2026-04-20 以降は `mimir(fn, ranges, guard_fn=my_guard)` で代替可。このドキュメントは既存 Sentinel コード を保守する時のために残す。

`optimize eval_fn, protect guard_fn; if guard < baseline → auto-diagnose → auto-pivot`.

```python
from twelve.agent.sentinel import Sentinel
r = Sentinel(
    eval_fn=ppl_eval, guard_fn=hellaswag_eval,
    param_ranges=[(0.5, 1.5)] * 61, param_names=[...],
    experience_id="llm_calib",       # cross-session learning
    initial_params=warm_start,       # required for discrete
    learn=True,                       # accumulate across runs
    min_r_squared=0.3,                # owl→optimize fallback threshold
).run(time_budget=1800, verbose=True)
```

## Flow

```
1a. _collect 20 pts (seed = initial_params OR midpoint)
1b. owl(eval_fn, autonomous, 30% budget)
    insufficient IF: best_params is None OR proxy_r² < min_r_squared
                  OR verified_score < max(measurements)    ← proxy misleading
    → fallback optimize(eval_fn, warm=best_m, 30% budget)
2.  guard_fn(best) >= guard_fn(baseline)?
    YES → verdict="approved"
    NO  → Step 3
3.  multi-observer({eval,guard}) → safe_dims + conflict_dims
    IF max R² < min_r_squared → treat all dims as safe (conservative)
    owl(guard_fn, safe_dims, 20% budget) → fallback optimize
    guard_fn(pivot) >= baseline? → "pivoted" : "failed"
```

## Return

```python
{
    "verdict":           "approved"|"pivoted"|"failed",
    "best_params":       list, "eval_score": float, "guard_score": float,
    "baseline_guard":    float,                # guard at midpoint
    "optimization_mode": "owl"|"direct"|"high"|"low",
    "safe_dims":         list|None,            # pivoted: dims safe for both
    "conflict_dims":     list|None,            # pivoted: metrics conflict
    "multi_observer":    dict|None,
    "proxy_r2":          float,                # 0 if direct fallback
    "best_ever_score":   float, "best_ever_params": list,
    "elapsed_s":         float,
}
```

## Example domains (eval / guard)

PPL / HellaSwag · binding / toxicity · character power / meta diversity · expected return / max drawdown · new capability / existing capability.

**Single metric**: `guard_fn=eval_fn` (+2 eval calls overhead).

**Direct-mode fires when**: discrete spaces · non-smooth landscapes · proxy R² < 0.3 · verified_score < max measurement. Proven on brain_growth 04-17 (ISS 92-107 on 8D discrete topo where owl alone returned 0.0).

**2026-04-19 以降、owl 自身にも direct-HC fallback 移植済** (commit 911997e)。autonomous + verify_fn があれば Sentinel wrap 不要で同じ動作。新規 code では **`mimir()` 推奨**、Sentinel は legacy 互換で残置。
