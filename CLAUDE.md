# CLAUDE.md — Operational Rules

All responses in Japanese.

---
# Part 1: Usage
---

## Basics: 2 Patterns

```python
from twelve.optimize import owl

# 1. Have data → owl()
result = owl([
    {"params": {"temp": 0.7, "top_p": 0.9}, "score": 85},
    {"params": {"temp": 0.5, "top_p": 0.95}, "score": 90},
    # ... 20+ entries (min 5, recommended 20+)
], time_budget=60)

# 2. Have eval_fn → MirrorAgent (measure → owl → optimize, fully automatic)
from twelve.agent.mirror_agent import MirrorAgent
result = MirrorAgent(eval_fn=my_eval, param_ranges=[(0.5, 1.5)] * 12).run(time_budget=300)

# Both return the same structure:
result["best_params"]   # optimal parameters
result["proxy_r2"]      # proxy accuracy (trust if >= 0.7)
result["proxy_type"]    # winning proxy name ("zenron" / "zenron_interact" / "linear" etc.)
result["confidence"]    # "high" / "low" / "insufficient"
result["dead_dims"]     # structurally meaningless dimensions
```

## Options: When to use what

| Option | Effect | When to use |
|--------|--------|-------------|
| `experience_id='task1'` | Gets smarter each call with same ID | Repeating the same problem |
| `verify_fn=eval_fn` | Verifies proxy optimum with real eval. 5-round auto-growth | Preventing proxy hallucination |
| `autonomous=True` | Stagnation detect → range perturb → auto-expand. Runs unattended | verify_fn + long-running tasks |
| `time_budget=300` | Time limit (seconds) | Heavy eval_fn. Default 60s |
| `max_iterations=10` | Round limit for autonomous | Want longer runs. Default 10 |
| `param_names=['a','b']` | Name your parameters | Result readability |
| `n_rounds=5` | Number of verify_fn rounds | Auto 5 when verify_fn set. Manual override |
| `min_r_squared=0.7` | Proxy trust threshold | Lower when you want results despite low-quality proxy |
| `kathara="auto"` | 6-node parallel K² optimization | Auto: verify_fn + budget≥120s. True/False to force |

MirrorAgent.run() additional options:

| Option | Effect | When to use |
|--------|--------|-------------|
| `mr_samples=20` | Initial measurement count | Reduce for faster runs with less data |

```python
# Full power (smartest invocation)
result = owl(data,
    verify_fn=eval_fn,
    autonomous=True,
    max_iterations=10,
    experience_id='my_task',
    time_budget=300)
```

Everything is optional. `owl(data)` alone gives an answer.

## How verify_fn works

1. Proxy optimization → obtain best_params
2. verify_fn(best_params) → real verified_score
3. Add real measurement to measurements pool
4. Rebuild proxy (more data = smarter proxy)
5. Repeat 1-4 for 5 rounds (n_rounds=5, auto-set when verify_fn provided)

With autonomous=True: 3 stagnations → MS re-analysis + range perturbation → repeat up to max_iterations

## How to read R²

- >= 0.7 → confidence="high". Trustworthy. verify_fn effective
- 0.3–0.7 → confidence="low". Usable but exercise caution
- < 0.3 → confidence="insufficient". Not enough data. Add measurements

## 5 readings from 1 computation

owl() computes structure once. Read different fields = different answers.

| Read this | Get this | Example |
|---|---|---|
| `importance` | What matters (understanding) | "Layer 24, 27 drive intelligence" |
| `dead_dims` | What's irrelevant (discovery) | "logP doesn't affect binding — contradicts textbooks" |
| `fragility` | What's vulnerable (protection) | "hidden_size is fragile — no backup if it breaks" |
| `proxy_fn(x)` | What-if prediction | "This config → score ≈ 85" |
| `best_params` | Optimal values (optimization) | "Set these scales for best PPL" |

Read top to bottom. Understanding first, optimization last.
best_params is a side effect of structure discovery, not the main output.

## Multi-observer (multiple scores)

```python
# Multiple scores → automatic multi-observer analysis
result = owl([
    {"params": {"layers": 28, "hidden": 3584}, "scores": {"reasoning": 0.85, "math": 0.72}},
    # ... 20+ entries
])

result["stable_active"]      # active in ALL observers (universal structure)
result["stable_dead"]        # dead in ALL observers (true noise)
result["observer_dependent"] # changes by observer (context-dependent)
result["observers"]          # per-observer detail (importance, fragility, dead_dims)
```

Single "score" field still works (backward compatible).

## Recipe: Quantify and optimize (LaD)

Everything expressible can be quantified. Everything quantified can be optimized.

```python
# Step 1: Write eval_fn (the only human job)
def eval_fn(params):
    apply_scales(params)        # apply parameters
    ppl = measure_ppl()         # measure quality
    restore()                   # restore original state
    return -ppl                 # higher = better. Negate PPL
    # CAUTION: never pass test answers to eval_fn (prevents cheating)

ranges = [(0.5, 1.5)] * 12     # search range

# Step 2: Submit (everything else is automatic)
result = MirrorAgent(eval_fn, ranges).run(
    time_budget=300, experience_id='llm_scale')
```

For manual workflow:
```python
from twelve.measure import sensitivity_scan, random_sample
data = sensitivity_scan(eval_fn, ranges)     # perturb 1 param at a time (2n+1 measurements)
data += random_sample(eval_fn, ranges, n=20) # 20 random measurements
result = owl(data)
```

## eval_fn Design Guide

| Good | Bad | Reason |
|------|-----|--------|
| `return -ppl` (higher=better) | `return ppl` | Violates score convention |
| Continuous parameters | Discontinuous (type switching) | Proxy collapse (proven: MixQ PPL 2x worse) |
| Range `[0.5, 1.5]` | Range includes 0 | scale=0 → catastrophic (proven: PPL=262144) |
| apply → measure → restore | State leaks | Measurement contamination |
| Eval on training data | Eval on test answers | Cheating |

## 3 Ways to call (same internals)

```
Python:  owl(data, verify_fn=..., autonomous=...) ← full features
MCP:     owl(measurements_json, time_budget, experience_id)
         ← no verify_fn/autonomous. Data-to-answer only
HTTP:    POST http://localhost:8284/owl  (Bearer owl2026)
         ← same as MCP. Data-to-answer only
```

## Server

```bash
python twelve/agent/owl_server.py --api-key owl2026  # Owl only (safe for external)
python twelve/agent/api.py --port 8282               # internal full toolset
```

---
# Part 2: Principles — Why this works
---

## The Zenron formula

```
x_i ← best( perturb(x_i), share(neighbors_i) )
```

"Change yourself, compare with neighbors, keep the better one."
This is the operating principle. DNA, galaxies, neurons — all follow this.

Owl applies this principle: measurements = perturb results, proxy = share structure, optimize = keep the best.

## Why data alone gives answers

Hidden structure exists in measurement data. MirrorScan extracts it in 3 layers using the Zenron-derived importance formula.

### Layer 1: truth (individual power)

Correlation between each parameter and score. "Does changing this parameter affect the score?"

```
truth[i] = |corr(param_i, scores)|
```

truth alone is not used. It picks up accidental correlations.

### Layer 2: connectivity (linkage)

Correlation between parameters. "Does this parameter move with others?"

```
connectivity[i] = mean(|corr(param_i, param_j)|) for j≠i
```

Parameters that move together = the system has structure.

### Layer 3: Importance formula (multiplication eliminates noise)

```
importance[i] = (truth[i] × max(connectivity[i], floor)) ^ exp

Hardcode default: exp=0.5, floor=0.01  (mirror_agent.py)
JSON-optimized:   floor=0.078 (configs/ma_meta_params.json)
_mp() resolves: JSON exists → JSON wins, else hardcode.
exp=0.5 fixed. exp depends on eval_fn weighting (R²→0.46, equal→0.54, F1→0.40).
0.5 ≈ equal-weight optimum. Universal safe default. No "correct exp" exists.
```

| truth | connectivity | importance | meaning |
|-------|-------------|------------|---------|
| high | high | **high** | real structure. optimize this |
| high | low | low | accidental correlation. overfitting risk |
| low | high | low | co-moves but no effect |
| low | low | ≈0 | dead dim. safe to ignore |

**Multiplication kills noise.** This is the core of the importance formula.

### Layer 4: Interaction importance (pair extension)

Same formula applied to parameter PAIRS:

```
pair_connectivity[i,j] = |corr(param_i, param_j)|
pair_truth[i,j]        = |corr(param_i × param_j, scores)|
interaction_imp[i,j]   = (pair_truth[i,j] × max(pair_connectivity[i,j], floor)) ^ exp
```

Only pairs with high interaction_importance get interaction terms (x_i × x_j) in the proxy.
Same double-filter. Same noise elimination. Natural extension of the zenron formula.

**Proven:** R²=0.38 (linear only) → R²=0.79 (with interactions) on same 53 measurements.

### Proxy generation

3-6 candidates auto-selected:

```
{zenron, zenron_interact, linear} × {raw, log} = max 6 candidates → best R² wins
(log is only available when all scores have same sign. Otherwise 3 candidates)

zenron:          importance-weighted linear
zenron_interact: + connectivity-guided interaction terms (x_i × x_j)
linear:          plain linear regression

raw: linear systems (direct proportion)
log: multiplicative systems (PPL, neural nets, chemical reactions)
```

### Why 100K optimizations don't break it

Proxy extracts structure only. Individual data noise is excluded.
Stable within data range no matter how many times called.

Extrapolation (outside data range) is not guaranteed → use verify_fn.

### Power of dead dims

```
Measured: 206 dims → MS reduces to 21 dims
        = search space 10^206 → 10^21 (10^185x reduction)
```

---
# Part 3: Reference
---

## Architecture

```
Owl: owl() → MS.from_measurements() → build_proxy(3-6 candidates) → optimize()
  with verify_fn → auto-growth loop (5 rounds)
  with autonomous=True → stagnation detect + range perturb + neighbor sampling
  Proxy optimization only. Does not use categories/meta/Phase3

MirrorAgent: eval_fn → initial measurements → owl(autonomous) → result
  Internals fully delegated to owl()

AT (EvolutionAgent): diagnose → prescribe → owl() → verify → evolve
  Uses owl() internally for MS Proxy + optimize (Dual Accel implicit)
  AT-specific: categories/Phase3/prescription table/AutoSurrogate/compose_eval_fn
  Fallback: when MS history < 5, calls optimize() directly

Internal engine:
  MS (MirrorScan) → importance formula → dead_dims → proxy auto-gen
  TL (optimize)   → Phase1(TwelveParallel) + Phase2(KatharaParamOptimizer) + Phase3(meta)
    Phase1: per-category HC + Kathara search
    Phase2: 12-candidate precision search
    Phase3: K7-K12 meta-evolution (AT: only when meta=True)
    categories: parameter group names. 2+ groups enable Phase1 parallelization

Dead dims: importance < max(global_max × ratio, floor)
  Hardcode: ratio=0.1, floor=0.05
  JSON-optimized: ratio=0.233, floor=0.131 (configs/ma_meta_params.json)

K² mode: owl(kathara="auto") — verify_fn + budget≥120s で自動6ノード並列。
  Circulant(6,{1,3}): 6ノード, 3隣接, 直径2。各ノードがowl()でproxy加速。

Cross-session: UnifiedExperience(experience_id)
  → dead_dims hint, warm_start, best_params carry-over
```

## Rules

| # | Rule |
|---|------|
| -1 | Strip to essence: x_i, perturb, share, eval_fn |
| 0 | Measure, don't guess: 5+ data points → owl() → read the numbers |
| 1 | Ask Oracle: structural questions → `oracle` MCP |
| 2 | No manual tuning: data → owl() |
| 3 | LaD: no if/else → convert to numeric parameters |
| 4 | importance = (truth × max(connectivity, floor)) ^ exp |
| 5 | Overfitting prevention: n_problems > n_params. Optimize on PPL → verify on bench |
| 6 | Discontinuous parameters → proxy collapse. Convert to continuous via LaD |
| 7 | scale=0 forbidden. Never include 0 in parameter ranges |

## Proven results

| Date | Technique | Result |
|---|---|---|
| 2026-04-10 | **ISS proxy + V3 bench** | **96.0% (Reasoning 93.3%, +33pts)** |
| | *(ISS=Intelligence Structure Score, V3=ARC-Challenge v3)* | |
| 2026-04-10 | ISS proxy workflow | **6 min (world: GPU256 × 3 weeks)** |
| 2026-04-10 | **Owl** | **R²=0.998 (self-proxy), 320K eval/2.6s** |
| 2026-04-11 | **MixQ Gemma4 31B** | **8.7GB, 93.3% (+6.6pt vs Q2_K 12GB)** |
| 2026-04-11 | MixQ cross-arch | Qwen/Gemma both +6.6pt. Universal principle |
| 2026-04-12 | **F32 scale calibration** | **PPL 1554→23.9, 21KB change (98.5%)** |
| 2026-04-12 | **Interaction proxy** | **R² 0.38→0.79, same data, zenron pair extension** |
| 2026-04-12 | **Owl self-optimize** | dead_dims F1: 0.22→0.56 (+155%), connectivity_floor 8x |
| 2026-04-12 | **MA→owl() delegation** | 352→60 lines. Same results, zero duplication |
| 2026-04-12 | **Dual acceleration** | 1114x proxy speedup, 20x effective. Now implicit via owl() |
| 2026-04-12 | **Experience carry-over** | 2nd run +33%, verified_score 979→1304 |
| 2026-04-13 | **AT→owl() delegation** | -159 lines. AT uses owl() internally. K² deprecated |
| 2026-04-13 | **exp depends on eval_fn** | R²→0.46, equal→0.54, F1→0.40. No "correct exp". 0.5≈equal default |
| 2026-04-14 | **Fragility (death-side)** | √(truth×isolation). 306 LLM: hidden_size most fragile despite highest connectivity |
| 2026-04-14 | **Multi-observer Owl** | 6 observers × 306 LLM: 4 stable_active, 1 stable_dead, 3 observer_dependent |
| **Failures** | *(reflected in Rule 6, 7)* | |
| MixQ type mixing | PPL 2x worse | Discontinuous → proxy collapse (Rule 6) |
| Layer zero-out | PPL=262144 | scale=0 → catastrophic (Rule 7) |
| Proxy hallucination | proxy=3907, real=-17 | Caught by verify_fn + autonomous |

## LLM surgery & F32 calibration

Details in `@docs/HANDBOOK.md`. Key commands and architecture findings documented there.

## Hardware & Safety

- i7-12700K, RTX 3080 Ti 12GB, Win11
- llama.cpp modified: `c:/Users/mukic/llama.cpp/build/bin/Release/`
- Models: `c:/Users/mukic/models/`
- Never commit: `unified_memory.py`, `evaluator*.py`, `_legacy/`, patent docs

## Docs

| Need | Read |
|---|---|
| Theory | `@docs/全論.md` |
| Oracle | `@twelve/ORACLE.md` |
| Techniques | `@docs/HANDBOOK.md` (tech catalog, specs, results) |
| Experiments | `@docs/LAB_NOTES.md` (experiment log, chronological) |
| Kathara math | `@docs/KATHARA_NOTE.md` |
| Engine API | `@twelve/TWELVE_API.md` (optimize/AT/surrogate design) |
| Optimization log | `@twelve/OPTIMIZATION_LOG.md` (Phase 14-17 results) |
