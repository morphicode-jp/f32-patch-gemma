# CLAUDE.md -- Operational Rules

All responses in Japanese.

---
# Part 1: Usage
---

## The Stack

```
Sentinel  →  "Optimize X, protect Y. Auto-pivot on failure."
               Auto-picks tool: owl (continuous) or optimize (discrete)
  owl()   →  "Find structure in data. Build proxy. Optimize." Continuous only.
  optimize() → Phase1(TwelveParallel) + Phase2(KatharaParamOptimizer). Direct search, discrete OK.
```

```python
# 1. Two metrics (default — safe optimization)
from twelve.agent.sentinel import Sentinel
result = Sentinel(
    eval_fn=fast_metric,         # optimize with this (fast, cheap)
    guard_fn=important_metric,   # protect this (what actually matters)
    param_ranges=[(0.5, 1.5)] * 61,
).run(time_budget=900)
# → verdict: "approved" / "pivoted" / "failed"

# 2. Single metric (structure discovery + optimize from data)
from twelve.optimize import owl
result = owl([
    {"params": {"temp": 0.7, "top_p": 0.9}, "score": 85},
    {"params": {"temp": 0.5, "top_p": 0.95}, "score": 90},
    # ... 20+ entries (min 5, recommended 20+)
], verify_fn=eval_fn, autonomous=True, time_budget=300)

# 3. Raw engine (internal — used by owl() on proxy_fn)
from twelve.optimize import optimize
best_params, best_score, info = optimize(eval_fn, param_ranges, time_budget=60)

# All return:
result["best_params"]   # optimal parameters
result["proxy_r2"]      # proxy accuracy (trust if >= 0.7)
result["dead_dims"]     # structurally meaningless dimensions
```

| | Sentinel | owl() | optimize() |
|---|---|---|---|
| Input | eval_fn + guard_fn + ranges | measurements (data) | eval_fn + ranges |
| Calls eval_fn | Yes (via owl AND/OR optimize) | No (uses proxy) | Yes (directly, 1000s of times) |
| Continuous space | **Yes** (owl path) | **Yes** | Yes (slower) |
| Discrete space | **Yes** (auto-fallback to optimize) | No (proxy fails) | **Yes** |
| Cross-validation | **Yes** (guard_fn) | No | No |
| Auto-pivot on failure | **Yes** | No | No |
| Use when | 2 metrics or want safety | Have data, smooth landscape | Discrete-only without guard |

> **MirrorAgent** (`twelve.agent.mirror_agent`) is a thin owl() wrapper: 20 random points → `owl(autonomous=True)`. Use Sentinel or owl() directly instead.

## Sentinel -- Safe Optimization

Optimize with eval_fn. Protect with guard_fn. If guard_fn drops below baseline → auto-diagnose → auto-pivot.

```python
from twelve.agent.sentinel import Sentinel

result = Sentinel(
    eval_fn=ppl_eval,            # optimize this
    guard_fn=hellaswag_eval,     # protect this
    param_ranges=[(0.5, 1.5)] * 61,
    param_names=[f"L{i}" for i in range(61)],
    experience_id="llm_calib",   # cross-session learning
    initial_params=warm_start,   # warm-start (important for discrete problems)
    learn=True,                  # accumulate experience across runs
    min_r_squared=0.3,           # owl proxy threshold; below -> fallback
).run(time_budget=1800, verbose=True)
```

### Internal flow

```
Step 1a: _collect 20 points (seed=initial_params if given, else midpoint)
Step 1b: Try owl(eval_fn, autonomous=True, 30% budget)
         Is owl insufficient?
           - best_params is None, OR
           - proxy R² < min_r_squared, OR
           - verified_score < max(measurements) (proxy misleading)
         Yes -> fallback: optimize(eval_fn, initial_params=best_m, 30% budget)
         No  -> use owl's result
Step 2:  guard_fn(best_params) >= guard_fn(baseline)?
           YES -> verdict="approved", done
           NO  -> Step 3
Step 3:  multi-observer({eval, guard}) -> safe_dims + conflict_dims
         If multi-observer inconclusive (R² too low) -> treat all dims safe
         Try owl(guard_fn, safe_dims only, 20% budget), fallback to optimize
         guard_fn(pivot_params) >= baseline?
           YES -> verdict="pivoted"
           NO  -> verdict="failed"
```

### Return structure

```python
{
    "verdict":           "approved" | "pivoted" | "failed",
    "best_params":       list,           # final parameters
    "eval_score":        float,          # eval_fn score
    "guard_score":       float,          # guard_fn score
    "baseline_guard":    float,          # guard_fn at range midpoint
    "optimization_mode": "owl" | "direct" | "high" | "low",  # which path fired
    "safe_dims":         list | None,    # dims safe for both metrics (pivoted)
    "conflict_dims":     list | None,    # dims where metrics conflict (pivoted)
    "multi_observer":    dict | None,    # full diagnosis (pivoted)
    "proxy_r2":          float,          # 0.0 when optimize fallback fired
    "elapsed_s":         float,
}
```

### When fallback fires (optimization_mode = "direct")

- Discrete/integer parameter spaces (graph topology, layer counts, quantization bits)
- Highly non-smooth landscapes
- owl's proxy gets `R² < 0.3` OR `verified_score < max measurement`
- Proven: brain_growth (04-17) hit ISS 92-107 on 8D discrete topology params where owl alone returned ISS 0.0

### What Sentinel enables

| eval_fn (optimize) | guard_fn (protect) | Outcome |
|---|---|---|
| PPL (fast) | HellaSwag (matters) | Push capability without breaking alignment |
| Binding affinity | Toxicity | Effective drugs that stay safe |
| Character power | Meta diversity | Buff without breaking the game |
| Expected return | Max drawdown | Higher returns within risk budget |
| New capability | Existing capability | Smarter without forgetting |

**Single metric?** Set `eval_fn=guard_fn`. Sentinel degrades to owl+rubber-stamp. Overhead: 2 extra eval_fn calls.

### Sentinel constructor options

| Option | Default | Purpose |
|---|---|---|
| `eval_fn` | required | Optimization target |
| `guard_fn` | required | Protection target |
| `param_ranges` | required | `[(lo, hi), ...]` |
| `param_names` | auto | Name parameters for readability |
| `experience_id` | None | Cross-session learning (owl + optimize) |
| `initial_params` | None | Warm-start. Required for discrete problems |
| `learn` | False | Accumulate experience. Set True for repeated runs |
| `min_r_squared` | 0.3 | Proxy threshold. Below -> optimize() fallback |

## owl() -- Structure Discovery + Optimization

| Option | Effect | When to use |
|--------|--------|-------------|
| `verify_fn=eval_fn` | Verify proxy optimum with real eval. 5-round auto-growth | Preventing proxy hallucination |
| `autonomous=True` | Stagnation detect → range perturb → auto-expand | verify_fn + long-running tasks |
| `experience_id='task1'` | Gets smarter each call with same ID | Repeating the same problem |
| `time_budget=300` | Time limit (seconds). Default 60s | Heavy eval_fn |
| `max_iterations=10` | Round limit for autonomous. Default 10 | Want longer runs |
| `param_names=[...]` | Name your parameters | Result readability |
| `n_rounds=5` | Number of verify_fn rounds | Auto 5 when verify_fn set |
| `min_r_squared=0.7` | owl's proxy confidence threshold (high/low split) | Lower for noisy data |
| `kathara="auto"` | 6-node parallel K² optimization | Auto: verify_fn + budget>=120s |

```python
# Full power (smartest single-metric invocation)
result = owl(data,
    verify_fn=eval_fn,
    autonomous=True,
    max_iterations=10,
    experience_id='my_task',
    time_budget=300)
```

Everything is optional. `owl(data)` alone gives an answer.

### How verify_fn works

1. Proxy optimization → best_params
2. verify_fn(best_params) → real verified_score
3. Add measurement to pool → rebuild proxy (more data = smarter)
4. Repeat for 5 rounds

With `autonomous=True`: 3 stagnations → MS re-analysis + range perturbation → repeat up to max_iterations.

### How to read R²

- >= 0.7 → confidence="high". Trustworthy. verify_fn effective
- 0.3-0.7 → confidence="low". Usable but exercise caution
- < 0.3 → confidence="insufficient". Add more measurements

## 5 Readings from 1 Computation

owl() computes structure once. Read different fields = different answers.

| Read this | Get this | Example |
|---|---|---|
| `importance` | What matters (understanding) | "Layer 24, 27 drive intelligence" |
| `dead_dims` | What's irrelevant (discovery) | "logP doesn't affect binding -- contradicts textbooks" |
| `fragility` | What's vulnerable (protection) | "hidden_size is fragile -- no backup if it breaks" |
| `proxy_fn(x)` | What-if prediction | "This config -> score ~ 85" |
| `best_params` | Optimal values (optimization) | "Set these scales for best PPL" |

Read top to bottom. Understanding first, optimization last.
best_params is a side effect of structure discovery, not the main output.

## Multi-Observer (multiple scores)

```python
result = owl([
    {"params": {"layers": 28, "hidden": 3584}, "scores": {"reasoning": 0.85, "math": 0.72}},
    # ... 20+ entries
])

result["stable_active"]      # active in ALL observers (universal structure)
result["stable_dead"]        # dead in ALL observers (true noise)
result["observer_dependent"] # changes by observer (context-dependent)
result["observers"]          # per-observer detail (importance, fragility, dead_dims)
```

Single `"score"` field still works (backward compatible).
Sentinel uses this internally: when guard_fn fails, it runs multi-observer on `{"eval": ..., "guard": ...}` to find safe_dims vs conflict_dims.

## Recipe: Quantify and Optimize (LaD)

Everything expressible can be quantified. Everything quantified can be optimized.

```python
# Step 1: Write eval_fn (the only human job)
def eval_fn(params):
    apply_scales(params)        # apply parameters
    ppl = measure_ppl()         # measure quality
    restore()                   # restore original state
    return -ppl                 # higher = better. Negate PPL
    # CAUTION: never pass test answers to eval_fn (prevents cheating)

ranges = [(0.5, 1.5)] * 12

# Step 2: Submit
# Two metrics? Sentinel (recommended)
result = Sentinel(eval_fn=eval_fn, guard_fn=bench_eval,
                  param_ranges=ranges).run(time_budget=300)

# Single metric? owl() with verify
result = owl(data, verify_fn=eval_fn, autonomous=True, time_budget=300)
```

Manual workflow:
```python
from twelve.measure import sensitivity_scan, random_sample
data = sensitivity_scan(eval_fn, ranges)     # perturb 1 param at a time (2n+1)
data += random_sample(eval_fn, ranges, n=20) # 20 random measurements
result = owl(data)
```

## eval_fn Design Guide

| Good | Bad | Reason |
|------|-----|--------|
| `return -ppl` (higher=better) | `return ppl` | Violates score convention |
| Discrete params with `initial_params` via Sentinel | owl() alone on discrete | owl proxy collapses; Sentinel auto-fallbacks |
| Range `[0.5, 1.5]` | Range includes 0 | scale=0 -> catastrophic (proven: PPL=262144) |
| apply -> measure -> restore | State leaks | Measurement contamination |
| Eval on training data | Eval on test answers | Cheating |

## API & Server

```
Python:  owl(data, verify_fn=..., autonomous=...) <- full features
         Sentinel(eval_fn, guard_fn, ranges).run() <- safe optimization
MCP:     owl(measurements_json, time_budget, experience_id) <- data-to-answer only
HTTP:    POST http://localhost:8284/owl (Bearer owl2026) <- same as MCP
```

```bash
python twelve/agent/owl_server.py --api-key owl2026  # Owl only (safe for external)
python twelve/agent/api.py --port 8282               # internal full toolset
```

---
# Part 2: Principles -- Why This Works
---

## The Zenron Formula

```
x_i <- best( perturb(x_i), share(neighbors_i) )
```

"Change yourself, compare with neighbors, keep the better one."
DNA, galaxies, neurons -- all follow this.

Owl: measurements=perturb, proxy=share, optimize=keep.
Sentinel: owl=perturb, guard_fn=compare, pivot=keep the safe one.

## Why Data Alone Gives Answers

MirrorScan extracts hidden structure from measurements in 3 layers:

### Layers 1-3: The Importance Formula

```
truth[i]        = |corr(param_i, scores)|              # does this param affect score?
connectivity[i] = mean(|corr(param_i, param_j)|) j!=i  # does it move with others?
importance[i]   = (truth * max(connectivity, floor))^exp  # multiplication kills noise
```

Current values (JSON-optimized via `configs/ma_meta_params.json`):
- exp = 0.3064, floor = 0.1411
- `_mp()` resolves: JSON exists -> JSON wins, else hardcode (exp=0.5, floor=0.01)

| truth | connectivity | importance | meaning |
|-------|-------------|------------|---------|
| high | high | **high** | real structure. optimize this |
| high | low | low | accidental correlation. overfitting risk |
| low | high | low | co-moves but no effect |
| low | low | ~0 | dead dim. safe to ignore |

**Multiplication kills noise.** This is the core.

### Layer 4: Interaction (pair extension)

Same formula on parameter PAIRS:
```
interaction_imp[i,j] = (|corr(x_i * x_j, scores)| * max(|corr(x_i, x_j)|, floor))^exp
```
High-importance pairs get interaction terms (x_i * x_j) in proxy.
Proven: R² 0.38 -> 0.79 on same 53 measurements.

### Proxy Generation

```
{zenron, zenron_interact, linear} x {raw, log} = max 6 candidates -> best R² wins
zenron: importance-weighted | zenron_interact: + pair terms | linear: plain
raw: linear systems | log: multiplicative (PPL, neural nets). Log requires same-sign scores
```

Proxy extracts structure, not noise -> stable under 100K+ optimizations.
Extrapolation not guaranteed -> use verify_fn.

Dead dims compress search: 206D -> 21D = 10^185x reduction.

### Fragility (death-side dual)

```
isolation[i] = 1.0 - connectivity[i]
fragility[i] = (truth * max(isolation, floor))^exp    # active dims only
```

Important AND isolated = most fragile. Proven: 306 LLM analysis, hidden_size most fragile despite highest connectivity.

---
# Part 3: Reference
---

## Architecture

```
Sentinel: eval_fn + guard_fn + (optional) initial_params
  _collect(20 points): seed_params first, then random
  _optimize: try owl -> detect insufficient -> fallback to optimize(eval_fn)
    insufficient = best_params is None OR R² < min_r_squared
                   OR verified_score < max(measurements)  [proxy misleading]
  _diagnose: _owl_multi_observer({eval, guard}) for safe/conflict dims
  _pivot: same owl-first/optimize-fallback on safe_dims with guard_fn
    Multi-observer inconclusive (max R² low) -> treat all dims as safe

Owl: owl() -> MS.from_measurements() -> build_proxy(3-6 candidates) -> optimize(proxy_fn)
  verify_fn -> 5-round auto-growth | autonomous -> stagnation detect + range perturb

Internal:
  MS (MirrorScan) -> importance formula -> dead_dims -> proxy auto-gen
    Dead dims: importance < max(global_max * 0.156, 0.1377)  [ma_meta_params.json]
    Fragility: sqrt(truth * isolation). Active dims only
    3-seed consensus: majority vote across 3 random splits
  TL (optimize) -> Phase1(TwelveParallel) + Phase2(KatharaParamOptimizer)
    Phase3(meta-evolution): only when meta=True (K7-K12)
    Sentinel invokes this directly when owl proxy insufficient
  K²: owl(kathara="auto") -- 6-node Circulant parallel, auto when importable verify_fn + budget>=120s
  Experience: UnifiedExperience(id) -> dead_dims hint + warm_start + fossil rollback
    Sentinel passes experience_id through to BOTH owl and optimize paths
```

## Rules

| # | Rule |
|---|------|
| -1 | Strip to essence: x_i, perturb, share, eval_fn |
| 0 | Measure, don't guess: 5+ data points -> owl() -> read the numbers |
| 1 | Ask Oracle: structural questions -> `oracle` MCP |
| 2 | No manual tuning: data -> owl() |
| 3 | LaD: no if/else -> convert to numeric parameters |
| 4 | importance = (truth * max(connectivity, floor)) ^ exp |
| 5 | Overfitting prevention: n_problems > n_params. Optimize on train -> verify on bench |
| 6 | Discrete/integer params: OK via Sentinel (auto-fallback to optimize). Pass `initial_params` for warm-start |
| 7 | scale=0 forbidden. Never include 0 in parameter ranges |
| 8 | Two metrics? Sentinel. eval_fn optimizes, guard_fn protects. One metric = eval_fn=guard_fn |

## Proven Results

| Date | Technique | Result |
|---|---|---|
| 04-10 | **ISS proxy + V3 bench** | **96.0% (Reasoning +33pts), 6 min vs GPU256x3wks** |
| 04-10 | **Owl** | **R²=0.998, 320K eval/2.6s** |
| 04-11 | **MixQ Gemma4 31B** | **8.7GB, +6.6pt. Cross-arch universal** |
| 04-12 | **F32 calibration** | **PPL 1554->23.9, 21KB (98.5%)** |
| 04-12 | **Interaction proxy** | **R² 0.38->0.79, same data** |
| 04-12 | **Owl self-optimize** | dead_dims F1 +155%, experience +33% |
| 04-14 | **Fragility + Multi-observer** | 6obs x 306LLM: 4 stable, 1 dead, 3 dependent |
| 04-17 | **Sentinel v1** | Auto cross-validation + pivot. HellaSwag disaster -> automated |
| 04-17 | **Sentinel v2 (fallback)** | owl -> optimize auto-fallback for discrete spaces |
| 04-17 | **brain_growth** | 3-scale safe evolution (48N/144N/576N). ISS 92/107/85 with guard maintained |
| **Failures** | Rule 7: scale=0 -> PPL=262144 | verify_fn catches hallucination |

## Hardware & Safety

- i7-12700K, RTX 5090 32GB, Win11
- llama.cpp pre-built: `c:/Users/user/llm/llama-bin/` (b8795, CUDA 12.4)
- Models: `c:/Users/user/llm/models/`
- Never commit: `unified_memory.py`, `evaluator*.py`, `_legacy/`, patent docs

## Docs

| Need | Read |
|---|---|
| Theory | `@docs/全論.md` |
| Oracle | `@twelve/ORACLE.md` |
| Techniques / LLM surgery | `@docs/HANDBOOK.md` (tech catalog, specs, F32 calibration) |
| Experiments | `@docs/LAB_NOTES.md` (experiment log, chronological) |
| Kathara math | `@docs/KATHARA_NOTE.md` |
| Engine API | `@twelve/TWELVE_API.md` (optimize/AT/surrogate design) |
| Optimization log | `@twelve/OPTIMIZATION_LOG.md` (Phase 14-17 results) |
