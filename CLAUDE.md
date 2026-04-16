# CLAUDE.md — Operational Rules

All responses in Japanese.

---
# Part 1: Usage
---

## Basics: 3 Levels

**Nous** = environment → autonomous data collection → structure discovery → optimal policy.
Owl is the brain (give it data). Nous is brain + body + learning loop (it acts, collects, learns).

```python
from twelve.agent.nous import Nous

# 1. Have environment (RL) or eval_fn → Nous (recommended)
#    Autonomous: explores → understands structure → optimizes. 4-phase adaptive.

# RL mode: pass an environment. Nous auto-constructs policy + eval_fn.
from twelve.agent.nous import BalanceEnv
result = Nous(env=BalanceEnv()).run(time_budget=60)

# eval_fn mode: pass a function + ranges (backward compatible)
result = Nous(eval_fn=my_eval, param_ranges=[(0.5, 1.5)] * 12).run(time_budget=300)

# 2. Have eval_fn, want simple auto → MirrorAgent (random sampling → owl)
from twelve.agent.mirror_agent import MirrorAgent
result = MirrorAgent(eval_fn=my_eval, param_ranges=[(0.5, 1.5)] * 12).run(time_budget=300)

# 3. Have data already → owl() (structure analysis + optimize from measurements)
from twelve.optimize import owl
result = owl([
    {"params": {"temp": 0.7, "top_p": 0.9}, "score": 85},
    {"params": {"temp": 0.5, "top_p": 0.95}, "score": 90},
    # ... 20+ entries (min 5, recommended 20+)
], time_budget=60)

# All return the same core structure:
result["best_params"]   # optimal parameters
result["proxy_r2"]      # proxy accuracy (trust if >= 0.7)
result["proxy_type"]    # winning proxy name ("zenron" / "zenron_interact" / "linear" etc.)
result["dead_dims"]     # structurally meaningless dimensions
result["importance"]    # per-dimension importance scores

# Nous extras:
result["phase_log"]              # phase transition history
result["policy_interpretation"]  # (RL mode only) obs→action connection analysis
```

Core insight: **An RL environment is just an eval_fn auto-generator** — `env + policy_params → run_episode → total_reward = eval_fn(params)`. All Owl machinery works on RL without modification.

| | Nous | MirrorAgent | owl() |
|---|---|---|---|
| Input | env OR eval_fn | eval_fn + ranges | measurements (data) |
| Who collects data | Nous (importance-guided) | MirrorAgent (random) | Human |
| Exploration | 4-phase adaptive | Random sampling | N/A |
| RL environments | **Yes** | No | No |
| Use when | env or eval_fn + want max autonomy | Quick auto-optimize | Already have data |

**RL mode vs eval_fn mode**: Use RL mode (`env=`) when the problem has sequential decisions (actions → state changes → rewards over time). Use eval_fn mode when score = f(params) with no time dimension (e.g., PPL from static weights, benchmark accuracy from config).

## Options

Nous() constructor:

| Option | Effect | When to use |
|--------|--------|-------------|
| `env=BalanceEnv()` | RL mode. Auto-constructs policy eval_fn | You have an RL environment |
| `eval_fn=fn` | Direct eval_fn mode (backward compatible) | No environment, just a scoring function |
| `param_ranges=[...]` | Search ranges. Auto-set to [(-2,2)] in RL mode | eval_fn mode (required) |
| `param_names=[...]` | Name parameters. Auto-generated in RL mode | Result readability |
| `experience_id='task1'` | Cross-session learning. dead_dims + warm_start carry-over | Repeating the same problem |
| `policy_type="linear"` | Linear: `argmax(W@obs+b)`. Fast, few params | Default. Simple environments |
| `policy_type="kathara"` | Kathara(12,{1,4,6}): `obs→W_in→tanh→Kathara_propagate→tanh→W_out→action`. Non-linear | Complex environments needing non-linear policy |

Nous.run():

| Option | Effect | When to use |
|--------|--------|-------------|
| `time_budget=300` | Total time limit (seconds). Default 300 | Control runtime |
| `verbose=True` | Print phase transitions + structure updates | Debugging/monitoring |

owl() options:

| Option | Effect | When to use |
|--------|--------|-------------|
| `experience_id='task1'` | Gets smarter each call with same ID | Repeating the same problem |
| `verify_fn=eval_fn` | Verifies proxy optimum with real eval. 5-round auto-growth | Preventing proxy hallucination |
| `autonomous=True` | Stagnation detect → range perturb → auto-expand | verify_fn + long-running tasks |
| `time_budget=300` | Time limit (seconds). Default 60s | Heavy eval_fn |
| `max_iterations=10` | Round limit for autonomous. Default 10 | Want longer runs |
| `param_names=[...]` | Name your parameters | Result readability |
| `n_rounds=5` | Number of verify_fn rounds | Auto 5 when verify_fn set |
| `min_r_squared=0.7` | Proxy trust threshold | Lower for low-quality proxy |
| `kathara="auto"` | 6-node parallel K² optimization | Auto: verify_fn + budget≥120s |

MirrorAgent.run(): `mr_samples=20` (initial measurement count).

```python
# Full power owl (smartest invocation)
result = owl(data,
    verify_fn=eval_fn,
    autonomous=True,
    max_iterations=10,
    experience_id='my_task',
    time_budget=300)
```

Everything is optional. `owl(data)` alone gives an answer.

## How it works

### verify_fn loop

1. Proxy optimization → obtain best_params
2. verify_fn(best_params) → real verified_score
3. Add real measurement to pool → rebuild proxy
4. Repeat for 5 rounds (auto-set when verify_fn provided)

With autonomous=True: 3 stagnations → MS re-analysis + range perturbation → repeat up to max_iterations

### Nous 4-phase loop

3-layer architecture (RL + LaD + Owl):
- **RL layer**: Environment protocol (reset/step), linear policy (W@obs + b), episode runner
- **LaD layer**: Auto-converts actions/rewards to `{"params": policy_weights, "score": avg_reward}`
- **Owl layer**: MirrorScan for importance/dead_dims, owl() for final optimization

```
EXPLORE (0-20%)  → Random sampling. Whole-space survey.
                   Exit: dead_dims stable 2 rounds, or 20% time
FOCUS   (20-40%) → Importance-proportional sampling. Dead dims pinned at best.
                   Exit: proxy R² >= 0.7, or 40% time
DEEPEN  (40-60%) → Interaction-pair targeted sampling. Top pairs full sweeps.
                   Exit: 60% time
EXPLOIT (60-100%)→ Proxy-screened: 100 candidates → top 5 real-evaluated.
                   Final: owl() with all accumulated measurements.
```

RL mode auto-construction:
1. Policy dims = `n_actions × obs_size + n_actions` (W matrix + bias)
2. `param_ranges = [(-2.0, 2.0)]` for all policy weights
3. `eval_fn(params) = mean(run_episode(params) for 3 episodes)`

### R² confidence

- >= 0.7 → "high". Trustworthy
- 0.3–0.7 → "low". Usable with caution
- < 0.3 → "insufficient". Add measurements

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

In RL mode, same 5 readings apply to policy connections (e.g., importance = which obs→action links matter, dead_dims = prunable connections).

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
# Option A: Nous (recommended — importance-guided 4-phase exploration)
result = Nous(eval_fn, ranges, experience_id='llm_scale').run(time_budget=300)

# Option B: MirrorAgent (simpler — random exploration → owl)
result = MirrorAgent(eval_fn, ranges).run(time_budget=300, experience_id='llm_scale')
```

For manual workflow:
```python
from twelve.measure import sensitivity_scan, random_sample
data = sensitivity_scan(eval_fn, ranges)     # perturb 1 param at a time (2n+1 measurements)
data += random_sample(eval_fn, ranges, n=20) # 20 random measurements
result = owl(data)
```

## eval_fn & Environment

### eval_fn design rules

| Good | Bad | Reason |
|------|-----|--------|
| `return -ppl` (higher=better) | `return ppl` | Violates score convention |
| Continuous parameters | Discontinuous (type switching) | Proxy collapse (proven: MixQ PPL 2x worse) |
| Range `[0.5, 1.5]` | Range includes 0 | scale=0 → catastrophic (proven: PPL=262144) |
| apply → measure → restore | State leaks | Measurement contamination |
| Eval on training data | Eval on test answers | Cheating |

### Environment protocol (RL mode)

```python
from twelve.agent.nous import Environment

class MyEnv(Environment):
    @property
    def obs_size(self): return 4          # observation vector length

    @property
    def n_actions(self): return 2         # number of discrete actions

    def reset(self):
        return [0.0] * self.obs_size      # initial observation (list[float])

    def step(self, action):               # action: int index
        obs = [...]                       # new observation (list[float])
        reward = ...                      # float (higher = better)
        done = ...                        # bool (episode ended?)
        return obs, reward, done
```

Built-in environments:
- `BalanceEnv` — CartPole-v1 equivalent (obs=4D, actions=2, max 200 steps). Proven: 200/200 perfect
- `FlyWorldEnv` — 2D insect navigation (obs=12D, actions=9 (3×3 turn×speed), max 60 steps). Proven: 93.58/100
No gym dependency. No external packages required.

**Any system with observe→act→reward loop is an Environment:**
game AI (screen→move→score), trading (price→buy/sell→PnL), robot control (sensors→torque→distance),
manufacturing (temp/pressure→valve→quality), LLM tuning (scales→adjust→-PPL).

## API & Server

```
Python:  owl(data, verify_fn=..., autonomous=...) ← full features
MCP:     owl(measurements_json, time_budget, experience_id) ← data-to-answer only
HTTP:    POST http://localhost:8284/owl (Bearer owl2026) ← same as MCP
```

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
Nous extends this: act in environment = perturb, understand structure = share, improve policy = keep the best.

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
Nous: eval_fn/env → 4-phase autonomous loop → owl() → result
  RL mode: env → auto-construct policy eval_fn
    policy_type="linear": argmax(W@obs+b). dims = n_actions × obs_size + n_actions
    policy_type="kathara": obs→W_in→tanh→Kathara(12,{1,4,6})→tanh→W_out→action
      dims = obs×12 + 30(edges) + 12×actions + actions
    eval_fn(params) = mean(run_episode(params) × 3 episodes)
  Built-in envs: BalanceEnv (4D/2act), FlyWorldEnv (12D/9act)
  eval_fn mode: direct eval_fn (backward compatible with MirrorAgent)
  EXPLORE: random sampling (whole-space survey)
  FOCUS:   importance-guided sampling (dead_dims pinned at best)
  DEEPEN:  interaction-pair targeted sampling (top pairs full-range)
  EXPLOIT: proxy-screened (100→5) + final owl() with all measurements
  Phase transitions: dead_dims stability → R² threshold → time budget %
  Uses MirrorScan.from_measurements() directly for continuous structure updates

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

Hierarchical Kathara: n_clusters × Kathara(12) feedforward pipeline
  Cluster 0: sensors. Cluster c: proj(cluster[c-1]). Motor: last cluster
  396 params (vs fetal 1248). Proven: 4×12=94.00. Skip/fb hurt.
  Principle: preserve Kathara topology at each cluster = scaling success
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
| 8 | RL environment = eval_fn auto-generator. Same Owl machinery, no modification needed |
| 9 | Topology preservation > feature addition. Scale by preserving Kathara, not adding bio features |

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
| 2026-04-15 | **Nous v1 (FlyWorld)** | **93.58/100, food到達, 91D自律探索 (351 evals, 4 phases)** |
| 2026-04-15 | **kathara_brain_sim v4→v8** | **94.37% (12N), 94.00% (48N hier 4×12). Kathara = neural scaling law** |
| 2026-04-16 | **Nous v2 (BalanceEnv)** | **200/200 perfect, 8/10 dead_dims, textbook-optimal policy auto-discovered, 60s** |
| | *RL + LaD + Owl unified* | *angular_velocity→action = only relevant connection (physics textbook rediscovered)* |
| 2026-04-16 | **Hierarchical Kathara 4×12** | **48N=94.00 (12N=94.37, gap=0.37). Topology preservation > biology** |
| 2026-04-16 | v8 ablation (6 conditions) | inhib/skip/fb全て中立〜有害. baseline hier4最適. 396 params (68% reduction) |
| **Failures** | *(reflected in Rule 6, 7)* | |
| MixQ type mixing | PPL 2x worse | Discontinuous → proxy collapse (Rule 6) |
| Layer zero-out | PPL=262144 | scale=0 → catastrophic (Rule 7) |
| Proxy hallucination | proxy=3907, real=-17 | Caught by verify_fn + autonomous |

## LLM surgery & F32 calibration

Details in `@docs/HANDBOOK.md`. Key commands and architecture findings documented there.

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
| Techniques | `@docs/HANDBOOK.md` (tech catalog, specs, results) |
| Experiments | `@docs/LAB_NOTES.md` (experiment log, chronological) |
| Kathara math | `@docs/KATHARA_NOTE.md` |
| Engine API | `@twelve/TWELVE_API.md` (optimize/AT/surrogate design) |
| Optimization log | `@twelve/OPTIMIZATION_LOG.md` (Phase 14-17 results) |
