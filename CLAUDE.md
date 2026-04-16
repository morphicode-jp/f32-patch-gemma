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
| `policy_type="auto"` | 10回ずつlinear/kathara試行 → 勝者で継続 | どちらが良いか不明な時 |
| `envs=[env1,env2,...]` | 全環境の平均スコアで評価。汎化圧力 | マルチ環境で過学習防止 |
| `memory_len=3` | 過去N stepのobsを結合。時間パターン学習 | 部分観測・時系列依存の環境 |
| `curiosity=True` | proxy不確実性×距離で未探索領域優先 (FOCUS phase) | 広い探索空間、局所解が多い |
| `n_workers=6` | Circulant(N,{1,⌊N/2⌋})で並列探索→知見共有 | 並列計算で高速化 |

Nous.run(): `time_budget=300` (seconds, default 300), `verbose=True` (phase transitions log).

Nous.meta_optimize() — LaD究極形: Nous自身の内部定数をowl()で最適化:

```python
# Nous自身のフェーズ閾値・バッチサイズ等23パラメータを自動最適化
Nous.meta_optimize(env_factory=BalanceEnv, time_budget=300)
# → configs/nous_params.json に最適値を保存。以降の全Nous実行に反映
```

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

3-layer: **RL** (env→episode) → **LaD** (episode→`{params, score}`) → **Owl** (MS + proxy + optimize)

```
EXPLORE (0-20%)  → Random, whole-space. Exit: dead_dims stable
FOCUS   (20-40%) → Importance-guided, dead pinned. Exit: R²≥0.7
DEEPEN  (40-60%) → Interaction-pair targeted sweeps
EXPLOIT (60-100%)→ Proxy-screened (100→5) + final owl()
```

RL auto-construction: dims = `n_actions × obs_size × (1+memory_len) + n_actions`, ranges=(-2,2), eval=mean(3 episodes).

### R² confidence

R² >= 0.7 → "high" (trustworthy) | 0.3–0.7 → "low" (caution) | < 0.3 → "insufficient" (add data)

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

# For continuous actions, override these:
    @property
    def action_type(self): return "continuous"  # default: "discrete"
    @property
    def action_dim(self): return 2              # number of continuous action dims
    @property
    def action_range(self): return (-1.0, 1.0)  # (low, high) per dimension
    # step() receives np.array instead of int
```

Built-in: `BalanceEnv`(4D/2act, 200/200), `SwingUpEnv`(3D/1cont), `FlyWorldEnv`(12D/9act, 108/110). No gym dependency.
**Any observe→act→reward loop is an Environment**: game AI, trading, robotics, manufacturing, LLM tuning.

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

"Change yourself, compare with neighbors, keep the better one." DNA, galaxies, neurons — all follow this.
Owl: measurements=perturb, proxy=share, optimize=keep. Nous: act=perturb, understand=share, improve=keep.

## Why data alone gives answers

MirrorScan extracts hidden structure from measurements in 3 layers:

### Layer 1-3: Importance formula

```
truth[i]        = |corr(param_i, scores)|              # does this param affect score?
connectivity[i] = mean(|corr(param_i, param_j)|) j≠i   # does it move with others?
importance[i]   = (truth × max(connectivity, floor))^exp  # multiplication kills noise
```

exp=0.5 (universal default), floor: 0.01 (hardcode) / 0.078 (JSON-optimized).
`_mp()` resolves: JSON exists → JSON wins, else hardcode.

| truth | connectivity | importance | meaning |
|-------|-------------|------------|---------|
| high | high | **high** | real structure |
| high | low | low | accidental correlation |
| low | high | low | co-moves, no effect |
| low | low | ≈0 | dead dim |

### Layer 4: Interaction (pair extension)

Same formula on pairs: `interaction_imp[i,j] = (|corr(x_i×x_j, scores)| × max(|corr(x_i,x_j)|, floor))^exp`
High-importance pairs get interaction terms (x_i × x_j) in proxy. R² 0.38→0.79 on same 53 measurements.

### Proxy generation

```
{zenron, zenron_interact, linear} × {raw, log} = max 6 → best R² wins
zenron: importance-weighted | zenron_interact: + pair terms (x_i×x_j) | linear: plain
raw: linear systems | log: multiplicative (PPL, neural nets). Log requires same-sign scores
```

Proxy extracts structure, not noise → stable under 100K+ optimizations. Extrapolation not guaranteed → use verify_fn.

Dead dims compress search: 206D → 21D = 10^185x reduction.

---
# Part 3: Reference
---

## Architecture

```
Nous → 4-phase loop (EXPLORE/FOCUS/DEEPEN/EXPLOIT) → owl() → result
  RL: env → policy eval_fn auto-construct. eval_fn: → same Owl pipeline
  LaD config: configs/nous_params.json (JSON > hardcode, _cfg(section, key))
  meta_optimize(): owl()でNous自身の23パラメータを自動最適化

Owl: owl() → MS.from_measurements() → build_proxy(3-6) → optimize()
  verify_fn → 5-round auto-growth | autonomous → stagnation detect + range perturb

MirrorAgent: eval_fn → random measurements → owl() → result

AT: diagnose → prescribe → owl() → verify → evolve
  AT-specific: categories/Phase3/prescription table/AutoSurrogate

Internal:
  MS (MirrorScan) → importance formula → dead_dims → proxy auto-gen
  TL (optimize) → Phase1(Twelve) + Phase2(Kathara) + Phase3(meta-evolution)
  Dead dims: importance < max(global_max × 0.233, 0.131)  [ma_meta_params.json]
  K²: owl(kathara="auto") — 6-node Circulant parallel, auto when verify_fn + budget≥120s
  Experience: UnifiedExperience(id) → dead_dims hint + warm_start carry-over
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

## Proven results

| Date | Technique | Result |
|---|---|---|
| 04-10 | **ISS proxy + V3 bench** | **96.0% (Reasoning +33pts), 6 min vs GPU256×3wks** |
| 04-10 | **Owl** | **R²=0.998, 320K eval/2.6s** |
| 04-11 | **MixQ Gemma4 31B** | **8.7GB, +6.6pt. Cross-arch universal** |
| 04-12 | **F32 calibration** | **PPL 1554→23.9, 21KB (98.5%)** |
| 04-12 | **Interaction proxy** | **R² 0.38→0.79, same data** |
| 04-12 | **Owl self-optimize** | dead_dims F1 +155%, experience +33% |
| 04-14 | **Fragility + Multi-observer** | √(truth×isolation). 6obs×306LLM: 4 stable, 1 dead, 3 dependent |
| 04-15 | **Nous v1 (FlyWorld)** | **108/110, 117D自律探索** |
| 04-16 | **Nous v2 (BalanceEnv)** | **200/200, angular_velocity→action自動発見 (物理教科書再発見)** |
| 04-16 | **Nous v3 (全機能)** | **continuous/memory/curiosity/auto/K²/multi-env. 1800行** |
| 04-16 | **Nous LaD** | **23定数→JSON + meta_optimize(). Rastrigin +54%** |
| **Failures** | Rule 6: MixQ type→PPL 2x | Rule 7: scale=0→PPL=262144 | verify_fn catches proxy hallucination |

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
