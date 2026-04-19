# CLAUDE.md — Operational Rules

All responses in Japanese.

---

## Quick Reference

| situation | use | why |
|---|---|---|
| 2 metrics (eval+guard), **N_user ≤ 8**, repeated/related runs | **`reigen(...)`** | self-tuning default, cross-task learning |
| 2 metrics, N_user > 16, or 1-shot | `Sentinel(...)` | smaller overhead, no self-dim bloat |
| **Internal model state** (KV/weight/LoRA) | `Sentinel` + multi-obs | Rule 11: can't batch shared state |
| Have measurement data already | `owl(data)` | direct structure + optimization |
| Discrete-only, no guard | `optimize(eval, ranges)` | primitive direct search |

```python
from twelve.agent.reigen import reigen            # shortest (recommended)
from twelve.agent.sentinel import Sentinel        # bare Sentinel
from twelve.optimize import owl, optimize          # primitives
```

`experience_id`: 任意の task 名札を付ける。self_params cross-task 学習は `reigen_meta_knowledge.json` 経由で自動共有される (2026-04-19 以降、ID 共有不要)。fossil は per-ID 分離で並列時の衝突回避。

---

## Reigen (零玄) — default tool (集大成)

Dimension-additive self-application: one outer Sentinel over (N_user + N_self)-dim joint space (N_self = 17 for kathara_17_adaptive default, 12 for kathara_12 legacy).
Internally composes Sentinel → owl → MS → multi-observer → Kathara K² → optimize → UnifiedExperience.
`twelve/agent/sentinel.py` is **never modified** (uses `_TunableSentinel` subclass + `mirror_agent._mp()` monkey-patch).

### 3 ways to invoke

```python
# (1) shortest
from twelve.agent.reigen import reigen
r = reigen(my_eval, my_guard, [(-5, 5)] * 8, experience_id="genesis")

# (2) class form — defaults already optimal (kathara_17_adaptive + adaptive Hebbian + auto meta_knowledge)
from twelve.agent.reigen import Reigen
r = Reigen(
    eval_fn=..., guard_fn=..., user_param_ranges=[(-5, 5)] * 8,
    experience_id="my_task_v1",       # ← ID は好きに付けろ。meta_knowledge 経由で学習は自動共有
    inner_time_budget=2,              # sec per inner Sentinel
    # default: self_dim_preset="kathara_17_adaptive" (2026-04-19 昇格), enable_hebbian=True
    batch_eval_fn=None,               # opt-in: GPU/vectorized eval (see Reigen Reference)
    batch_size=8,
    # legacy opt-in: self_dim_preset="kathara_12" (static Hebbian)
    #                self_dim_preset="minimal_4"   (N_user > 16)
).run(time_budget=300, wall_time_factor=3.0)  # wall cap = budget × factor

# (3) HTTP: POST http://localhost:8282/reigen {eval_module, eval_fn, guard_fn,
#                                              user_param_ranges, experience_id, time_budget}
```

### Return (extends Sentinel)

```
user_best_params   list   user answer (N-dim)
user_best_score    float  = eval_score
self_best_params   dict   Reigen-discovered optimal Sentinel hparams
self_diagnostic    dict   {self_informative, self_dead_ratio, recommend_freeze, drift_from_defaults}
self_dead_dims     list   self knobs flagged dead
user_dead_dims     list   user dims flagged dead
self_dim_importance dict  per-self-dim MS importance
combined_best_params list  raw [user | self] concat
n_user_dims / n_self_dims  int
```
Plus all Sentinel keys: `verdict | best_params | eval_score | guard_score | baseline_guard | proxy_r2 | optimization_mode | safe_dims | conflict_dims | multi_observer | best_ever_* | elapsed_s`.

### Presets (`self_dim_preset`)

| preset | n_self | Hebbian | cost vs legacy | when |
|---|---|---|---|---|
| `"kathara_17_adaptive"` **(default)** | 17 | ✓ first 12 dims (adaptive lr/gate/winners from best-history) | +25-30% dim search | default since 2026-04-19 A/B: 3/3 wins (Rosenbrock 5d / Ackley 8d / Styblinski 6d), 2.7-3.3× faster + 15/15 approved (vs kathara_12's 8/15) |
| `"kathara_12"` (legacy) | 12 | ✓ Circulant(12,{1,4,6}) 30 edges, static lr | same (N_user ≤ 8), +20% (N_user > 8) | bit-identical canary baseline; explicit opt-in when you want static Hebbian |
| `"minimal_4"` | 4 | — | baseline | N_user > 16 (Rule 5 pressure); 1-shot throwaway |

`kathara_17_adaptive` の extra 5 dims (13: `self_batch_size`, 14-17: `self_hebbian_lr/min_history/percentile_gate/k_winners`) は Kathara 隣接グラフ上に**無い** — Hebbian 伝播は先頭 12 dims のみに適用 (Rule 10 保護)。14-17 は `best_p` から動的参照される outer-level パラメータ。

Switching mid-stream `minimal_4 → kathara_12 → kathara_17_adaptive`: inner fossils transfer ✓、meta_knowledge は preset 別 key なので cross-leak なし。

### Wall time model

Outer owl autonomous calls verify_fn (= 1 inner Sentinel) up to ~50×. `total_wall ≈ time_budget × wall_time_factor` (default 3). For 90s outer + inner=2s → typical 5 min. Past deadline, `combined_eval` returns cached best (soft cap).

### meta_knowledge (cross-task self-param inheritance)

Reigen reads `twelve/configs/reigen_meta_knowledge.json` at `__init__` and overlays learned best self-param values onto preset defaults. On `verdict in {"approved","pivoted"}`, `Reigen.run()` writes the run's self_best_params back (atomic + quality-gated: new score ≥ existing score).

- Per-preset keyed (`self_param_best.kathara_12`, `kathara_17_adaptive`, etc.) — no cross-preset leak
- Explicit `self_param_defaults=...` kwarg still wins over meta_knowledge (user control)
- Out-of-range stored values silently rejected (stale entries can't break validation)
- File corrupt / missing → silent fallback to preset defaults (fail-open)
- Atomic write (`tmp + os.replace`) — parallel processes never see partial JSON
- Tests isolated via `twelve/tests/conftest.py` autouse fixture (production file never written by pytest)
- Reset: delete `reigen_meta_knowledge.json` to go back to preset defaults

### When NOT to use Reigen
- N_user > 16 → bare Sentinel (preset `minimal_4` loses cross-task benefit)
- Ultra-cheap eval_fn (< 1ms) → inner Sentinel overhead dominates, use `owl()` directly
- Internal model state → Rule 11 (batch forbidden); keep eval_fn fast + multi-obs

---

## Reigen — Reference

### 12 self-dims (`kathara_12` default, auto-tuned)

| # | name | range | default | controls |
|---|---|---|---|---|
| 1 | self_min_r_squared | 0.15–0.75 | 0.30 | owl→optimize fallback threshold |
| 2 | self_owl_budget_ratio | 0.35–0.85 | 0.60 | owl share of `_optimize` budget |
| 3 | self_pivot_budget_ratio | 0.30–0.75 | 0.50 | owl share of `_pivot` budget |
| 4 | self_collect_min_ratio | 1.0–4.0 | 1.0 | multiplier on `max(20, n_dims)` |
| 5 | self_owl_max_iterations | 5–20 | 10 | autonomous round cap (continuous→int) |
| 6 | self_owl_n_rounds | 1–15 | 5 | verify_fn rounds (continuous→int) |
| 7 | self_stagnation_threshold | 2–8 | 3 | autonomous range-perturb trigger |
| 8 | self_proxy_pref | 0.01–1.0 | 0.33 | 0:zenron / 0.5:interact / 1:linear |
| 9 | self_ms_exp | 0.1–0.6 | 0.3064 | importance exponent |
| 10 | self_ms_floor | 0.05–0.30 | 0.1411 | connectivity floor |
| 11 | self_dead_threshold_ratio | 0.05–0.40 | 0.156 | dead-dim threshold |
| 12 | self_dead_threshold_floor | 0.01–0.25 | 0.1377 | dead-dim absolute floor |

Override via `self_dim_preset="minimal_4"` or explicit `self_param_ranges / _names / _defaults` kwargs.

### 7 JSON-overridable Reigen constants (`reigen_params.json`)

Static defaults for non-self-tuned knobs. Edit file for persistent change, or pass kwargs for per-call override.

| kwarg | JSON path | default | controls |
|---|---|---|---|
| `outer_min_r_squared` | `outer.outer_min_r_squared` | 0.3 | outer owl→optimize fallback threshold |
| `wall_time_factor` | `outer.wall_time_factor` | 3.0 | soft wall cap = budget × factor |
| `batch_size` | `outer.batch_size` | 8 | batch_eval_fn default size |
| `hebbian_lr` | `hebbian.hebbian_lr` | 0.05 | Hebbian propagation step size |
| `hebbian_min_history` | `hebbian.min_history` | 5 | min fossil count before Hebbian fires |
| `hebbian_percentile_gate` | `hebbian.percentile_gate` | 0.75 | reward gate (best must exceed this quantile) |
| `hebbian_k_winners` | `hebbian.k_winners` | 3 | top-K winner dims for propagation |

Precedence (strongest first): explicit kwarg → `reigen_params.json` → hardcoded fallback. Falsy kwarg (e.g., `hebbian_lr=0.0`) is respected via None-sentinel — Reigen² meta-tuning relies on this.

### Hebbian propagation (`kathara_12` and `kathara_17_adaptive`)

Top-K "winner" self-dims (largest delta vs best history) propagate `hebbian_lr × delta` to their 5 Kathara neighbors via Circulant(12,{1,4,6}). Fires only when best score > percentile-gate of history (min `hebbian_min_history` entries). Disable via `enable_hebbian=False`.

For `kathara_17_adaptive`: `hebbian_lr`, `min_history`, `percentile_gate`, `k_winners` are read from **best-history entry's self_p[13..16]** (dynamic per-call) instead of static attributes. The extra 5 dims are NOT on Kathara adjacency — propagation stays on first 12 dims only (Rule 10 preserved).

Cost: < 1ms numpy update per outer eval. Benefit: λ₂=4.0 → 3-step full propagation across all 12 dims.

**Kathara usage note**: we use the **graph-theoretic properties** (5-regular, λ₂=4.0 Ramanujan, diameter 2). We do NOT rely on `symmetric placement` (Rule 10) because Reigen's 12 dims are semantically distinct, not geometrically interchangeable. Chaos-game uniformity is irrelevant here.

### batch_eval_fn (GPU path)

`batch_eval_fn(params_list) -> scores_list` kwarg. When provided, inner Sentinel's `_collect` submits `batch_size` params at once. User supplies; Reigen/Sentinel/owl/MS all run CPU. Falls back to single `eval_fn` if None.

**Speedup** ∝ eval_fn latency:

| eval_fn latency | example | speedup |
|---|---|---|
| 0.1ms | synthetic | ~1× (Python overhead dominates) |
| 10ms | small NN | 1.2-1.5× |
| 100ms | GPU LLM 7B short | 3-8× |
| 1-5s | GPU LLM long / HellaSwag | 10-50× |

**VRAM** ≈ `model_weights + batch × seq_len × n_layers × d_model × 4B + 1GB`. RTX 5090 (32GB) practical:

| model | batch | seq=128 | seq=512 | seq=2048 |
|---|---|---|---|---|
| **Gemma 4 31B Q4_K_M** (18GB) | 8 | 19 GB | 21 GB | 27 GB |
|  | 16 | 20 GB | 23 GB | OOM |
|  | 32 | 22 GB | 27 GB | OOM |
| **Qwen 35B Q4** (21GB) | 8 | 22 GB | 24 GB | 30 GB (tight) |
|  | 16 | 24 GB | 27 GB | OOM |
| **7B Q4** (4GB) | 32 | 8 GB | 14 GB | — |
|  | 64 | 10 GB | 22 GB | OOM |
|  | 128 | 15 GB | OOM | OOM |

**Recommended batch_size**: HellaSwag 31B → **8** (21GB, 10-15×); HellaSwag 7B → **32-64** (30-50×); PPL long 31B → **4** (23GB, 5-8×); chat short 31B → **16-32** (20-40×); **safe default = 8**.

For llama.cpp: use `llama-server --parallel N`, not subprocess-per-call (current bench path is slow due to model reload).

---

## Sentinel — safe 2-metric optimization

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

### Flow

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

### Return

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

### Example domains (eval / guard)
PPL / HellaSwag · binding / toxicity · character power / meta diversity · expected return / max drawdown · new capability / existing capability.

**Single metric**: `guard_fn=eval_fn` (+2 eval calls overhead).

**Direct-mode fires when**: discrete spaces · non-smooth landscapes · proxy R² < 0.3 · verified_score < max measurement. Proven on brain_growth 04-17 (ISS 92-107 on 8D discrete topo where owl alone returned 0.0).

---

## owl() — structure discovery + proxy optimization

```python
r = owl(data, verify_fn=eval_fn, autonomous=True, max_iterations=10,
        experience_id="my_task", time_budget=300)
# data: list of {"params": {...}|list, "score": float}            (single-observer)
#       or [{"params": ..., "scores": {"obs1": ..., "obs2": ...}}] (multi-observer)
```

| option | effect | notes |
|---|---|---|
| verify_fn | 5-round auto-growth: verify proxy optimum with real eval | prevents proxy hallucination |
| autonomous=True | stagnation → range perturb → re-expand | pair with verify_fn for long runs |
| experience_id | cross-call learning with same ID | smarter per call |
| time_budget | seconds (default 60) | |
| max_iterations | autonomous round cap (default 10) | |
| n_rounds | verify_fn rounds; auto=5 when verify_fn set | |
| stagnation_threshold | autonomous range-perturb trigger (default 3) | added for Reigen |
| min_r_squared | proxy confidence threshold; owl default 0.7, **Sentinel overrides to 0.3** | lower for noisy data |
| force_proxy_type | "zenron"/"zenron_interact"/"linear"; default R²-best | added for Reigen |
| kathara="auto" | 6-node K² parallel; auto if verify_fn importable + budget ≥ 120s | |

### Confidence bands
`r² ≥ 0.7` → high (trustworthy, verify_fn effective). `0.3–0.7` → low (use with caution). `< 0.3` → insufficient (add more data).

### 5 readings from 1 owl() computation

| read | meaning | example |
|---|---|---|
| importance | what matters | "L24, L27 drive intelligence" |
| dead_dims | what's irrelevant | "logP doesn't affect binding" |
| fragility | what's vulnerable | "hidden_size is fragile" |
| proxy_fn(x) | what-if prediction | "this config → score≈85" |
| best_params | optimal values | "use these scales for best PPL" |

Read top→bottom. `best_params` is a side effect of structure discovery, not the main output.

### Multi-observer

```python
r = owl([{"params": {...}, "scores": {"reasoning": 0.85, "math": 0.72}}, ...])
r["stable_active"]       # active in ALL observers (universal structure)
r["stable_dead"]         # dead in ALL observers (true noise)
r["observer_dependent"]  # varies by observer (context-dependent)
r["observers"]           # per-observer: importance, fragility, dead_dims
```
Single `"score"` key remains backward-compatible. Sentinel uses this internally on pivot: `{eval, guard}` → safe/conflict dims.

---

## eval_fn design

| good | bad | why |
|---|---|---|
| `return -ppl` (higher=better) | `return ppl` | score convention |
| discrete params via Sentinel + `initial_params` | owl() alone on discrete | owl proxy collapses; Sentinel auto-fallbacks |
| range `[0.5, 1.5]` | range includes 0 | scale=0 catastrophic (PPL=262144, proven) |
| apply → measure → restore | state leaks | measurement contamination |
| eval on training data | eval on test answers | cheating |

---

## Principles — Zenron + MirrorScan

### Formula

```
x_i ← best( perturb(x_i), share(neighbors_i) )
```
"Perturb self, compare with neighbors, keep the better." DNA / galaxies / neurons all follow this.
Mapping: owl = (measurements, proxy, optimize); Sentinel = (owl, guard_fn, pivot).

### MirrorScan — importance (3-layer + pairs)

```
truth[i]         = |corr(param_i, scores)|             # does it affect score?
connectivity[i]  = mean(|corr(param_i, param_j)|) j≠i  # does it co-move?
importance[i]    = (truth × max(connectivity, floor))^exp     # multiplication kills noise
```
Values from `configs/ma_meta_params.json`: exp=0.3064, floor=0.1411 (fallback hardcode: 0.5, 0.01). `_mp()` resolves JSON > hardcode.

| truth | conn | importance | meaning |
|---|---|---|---|
| hi | hi | **hi** | real structure — optimize this |
| hi | lo | lo | accidental correlation (overfit risk) |
| lo | hi | lo | co-moves, no effect |
| lo | lo | ~0 | dead dim — safe to ignore |

**Multiplication kills noise.** Core principle.

### Pair interaction (Layer 4)
```
interaction_imp[i,j] = (|corr(x_i*x_j, scores)| × max(|corr(x_i,x_j)|, floor))^exp
```
High-importance pairs → add `x_i × x_j` terms to proxy. Proven R² 0.38→0.79 on same 53 meas.

### Proxy generation
```
{zenron, zenron_interact, linear} × {raw, log} = up to 6 candidates
best R² wins (or force via force_proxy_type).
  zenron: importance-weighted   zenron_interact: +pair terms   linear: plain
  raw: linear systems           log: multiplicative (PPL, NN); requires same-sign scores
```
Proxy extracts structure not noise → stable under 100K+ opts. Extrapolation not guaranteed → use verify_fn. Dead dims compress search: 206D → 21D ≈ 10^185× reduction.

### Fragility (death-side dual)
```
isolation[i] = 1.0 - connectivity[i]
fragility[i] = (truth × max(isolation, floor))^exp   # active dims only
```
Important AND isolated = fragile. Proven on 306-LLM: `hidden_size` most fragile despite highest connectivity.

---

## API & servers

```
Python:  reigen(eval, guard, user_ranges)                ← recommended (local)
         Sentinel(eval, guard, ranges).run()
         owl(data, verify_fn=..., autonomous=...)
         optimize(eval, ranges, ...)                      ← primitive
MCP:     owl(measurements_json, ...)                      ← data-to-answer only

HTTP (local/ngrok-shareable, auth via REIGEN_API_KEY env var):
  POST /owl                                 ← measurements-only optimization
  POST /reigen                              ← local eval_module (server-side fn)
  POST /reigen/start                        ← session: start → {session_id}
  GET  /reigen/next?sid=X&timeout=15        ← session: poll → {action: eval|wait|done}
  POST /reigen/score                        ← session: submit {sid, score, params}
```

```bash
# Launch (requires .reigen_api_key file with your bearer token):
./serve_reigen.sh [PORT]
# In another terminal:
ngrok http 8282

# Without launcher (manual):
REIGEN_API_KEY="your_token" python twelve/agent/api.py --port 8282
```

### Session API — remote friend's eval_fn runs CLIENT-side

`/reigen/start + /next + /score` let a remote client keep their `eval_fn` local (on their GPU/LLM) while the server does Reigen bookkeeping only (CPU, ~1 core).

Client helper: `reigen_friend_client.py` — `run_reigen_remote(server, key, eval_fn, ranges, ...)`.
Distribute `reigen_friend_client.py` + `FRIEND_README.md` to friends; they `pip install` nothing (stdlib only).

**experience_id sharing**: everyone (you + all friends) should use `experience_id="genesis"` (the default). Self-params accumulate across tasks and users; user-param fossils auto-filter by dim/range mismatch. More shared runs → smarter Reigen for everyone (Rule 9).

### /reigen body (legacy local-module mode)

```json
{
  "eval_module": "my_pkg.evals",            // importable Python module on server FS
  "eval_fn": "my_eval",
  "guard_fn": "my_guard",                    // optional (defaults to eval_fn)
  "user_param_ranges": [[0.5,1.5], [0.1,1.0]],
  "user_param_names": ["layer_scale","dropout"],
  "experience_id": "genesis",
  "time_budget": 300,
  "inner_time_budget": 2,
  "wall_time_factor": 3.0
}
```
Local-only (`importlib.import_module`, no code eval). HTTPS + auth needed beyond localhost.

### /reigen/start body (session mode — recommended for remote friends)

```json
{
  "user_param_ranges": [[-5, 5], ...],       // required
  "user_param_names": ["x1", "x2", ...],      // optional
  "experience_id": "genesis",                 // default (shared learning)
  "time_budget": 300,
  "inner_time_budget": 2,
  "inner_learn": false,
  "client_eval_timeout": 600                  // per-eval wait on server (sec)
}
```

Response: `{"session_id": "abc12345"}`. Session TTL 30 min idle; all 3 endpoints need `Authorization: Bearer <REIGEN_API_KEY>` when env var is set.

---

## Rules

| # | rule | see |
|---|---|---|
| -1 | Strip to essence: `x_i, perturb, share, eval_fn` | Principles |
| 0 | Measure don't guess: ≥5 pts → owl() → read numbers | owl() |
| 1 | Ask Oracle: structural questions → oracle MCP (`arc_oracle.py`, `kathara_oracle.py`, `@twelve/ORACLE.md`) | — |
| 2 | No manual tuning: data → owl() | owl() |
| 3 | LaD: no if/else — convert to numeric params | Principles |
| 4 | `importance = (truth × max(connectivity, floor))^exp` | Principles |
| 5 | Overfitting: n_problems > n_params. Optimize on train → verify on bench | — |
| 6 | Discrete/int params OK via Sentinel (auto-fallback). Pass `initial_params` for warm-start | Sentinel |
| 7 | **scale=0 forbidden**. Never include 0 in parameter ranges (proven: PPL=262144) | eval_fn |
| 8 | Two metrics? Sentinel/Reigen. eval_fn optimizes, guard_fn protects. 1 metric: `guard_fn=eval_fn` | Reigen, Sentinel |
| 9 | **Default to Reigen** (`kathara_17_adaptive` as of 2026-04-19). self_params cross-task inheritance via `reigen_meta_knowledge.json` (auto read at __init__, auto write on approved/pivoted runs). Fossils stay task-local; use any `experience_id` per task (no need to share — meta_knowledge handles shared learning). N_user ≤ 8 optimal. | Reigen |
| 10 | Kathara chaos-game uniformity (0.993) requires N=12 + 5-regular + **symmetric placement**. Break any → collapse. Applying Kathara to a new domain: check all three. **Reigen uses graph properties only, not uniformity** | Reigen Ref |
| 11 | `batch_eval_fn` works only for **external params** (lr, dropout, prompt). **Internal model state** (KV scale, weight scale, LoRA) forbids batching — shared global state. Strategy: fast eval_fn (≤2s) + **multi-observer** `{nll, hs, mmlu, ...}` → owl() for max info/eval. | Reigen, owl() |

### Rule 11 concrete example (Qwen3.6-NVFP4 KV calibration)

```python
# FORBIDDEN: all 8 configs share global model state → cannot parallelize
# def batch_eval_fn(params_list):
#     for p in params_list: apply_kv_scales(p); score = measure(); restore()  ← sequential only!

# RECOMMENDED: single eval_fn + multi-observer
def eval_fn_multi(params):
    apply_kv_scales(params)
    nll  = measure_nll()         # fast primary
    hs   = measure_hs_small()
    mmlu = measure_mmlu_small()
    restore()
    return {"nll": -nll, "hs": hs, "mmlu": mmlu}   # dict triggers multi-obs in owl

data = [{"params": p, "scores": eval_fn_multi(p)} for p in samples]
r = owl(data)   # finds stable_active / observer_dependent / stable_dead
```

---

## Proven results (key lessons)

| date | technique | result |
|---|---|---|
| 04-10 | ISS proxy + V3 bench | 96.0% (Reasoning +33pts), 6 min vs GPU256×3wks |
| 04-10 | Owl | R²=0.998, 320K eval/2.6s |
| 04-12 | F32 calibration | PPL 1554→23.9, 21KB (98.5%) |
| 04-12 | Interaction proxy | R² 0.38→0.79, same data |
| 04-14 | Fragility + Multi-observer | 6obs × 306LLM: 4 stable / 1 dead / 3 dependent |
| 04-17 | Sentinel v1 + v2 (fallback) | auto cross-validation + pivot; owl→optimize for discrete |
| 04-17 | brain_growth | 3-scale safe evolution 48/144/576 N, ISS 92/107/85 guard-maintained |
| 04-18 | **Reigen v1+v2+v3 unified** | dim-additive self-application, kathara_12 default (12 knobs + Hebbian 30-edges), batch_eval_fn API. 45+12 tests all green. Sentinel untouched. |
| 04-18 | Kathara chaos-game uniformity | **0.993** on 12-icosahedron (p=0.17, indistinguishable from uniform). Rule 10. |
| 04-18 | Qwen3.6-NVFP4 KV-Reigen | Single-obs NLL hurt HS -2pt (hit boundary 0.01/1.5 = proxy overfit). Birth of Rule 11. |
| failures | scale=0 → PPL=262144 | Rule 7. verify_fn catches hallucination |
| failures | single-obs on internal model state | fix: multi-observer + fast eval (Rule 11) |

Archival entries (individual runs, method evolution): `@docs/LAB_NOTES.md`.

---

## Hardware & safety

- i7-12700K, RTX 5090 32GB, Win11
- llama.cpp pre-built: `c:/Users/user/llm/llama-bin/` (b8795, CUDA 12.4)
- Models: `c:/Users/user/llm/models/`
- **Never commit**: `unified_memory.py`, `evaluator*.py`, `_legacy/`, patent docs

---

## Docs

| need | read |
|---|---|
| Theory | `@docs/全論.md` |
| Oracle | `@twelve/ORACLE.md` |
| Techniques / LLM surgery | `@docs/HANDBOOK.md` |
| Experiments (chronological log) | `@docs/LAB_NOTES.md` |
| Kathara math | `@docs/KATHARA_NOTE.md` |
| Engine API design | `@twelve/TWELVE_API.md` |
| Optimization log | `@twelve/OPTIMIZATION_LOG.md` |
| 全論 formal audit (T1–T4 proved, OP1–OP4 open) | `@docs/ZENRON_FORMAL.md` |
| Void instability theorem (Ch4 σ=0 禁止) | `@docs/ZENRON_VOID_INSTABILITY.md` |
| Time-arrow theorem (Ch17 §Time; M(t) monotone + F irreversible) | `@docs/ZENRON_TIME_ARROW.md` |
| Space-emergence theorem (Ch4 chaos game → uniform space, Moran dim) | `@docs/ZENRON_SPACE_EMERGENCE.md` |
| Multiplication/√ uniqueness (Ch17 axioms, min² counter-example excluded) | `@docs/ZENRON_MULT_UNIQUENESS.md` |
| Kathara mixing time (Ramanujan → τ_mix bound; 15 steps ≤ 17.6 theoretical) | `@docs/ZENRON_MIXING_TIME.md` |
| Dead dims = Noether symmetries (Ch17 + F-equivariance + affine invariance) | `@docs/ZENRON_DEAD_DIMS_NOETHER.md` |
| Kathara Kuramoto sync (Ch7 numerical test; specific values refuted, qualitative gap ≤ 0.10) | `@docs/ZENRON_KURAMOTO_SYNC.md` |
| Parallelism theorems P1-P3 (KATHARA_K2: LaD+Zenron=parallel, cost additive, 67% consensus) | `@docs/ZENRON_PARALLELISM.md` |
| Circuit invariants C1-C4 (KATHARA_CIRCUITS: amplification/triangles; "current disparity" = degree ratio, not Kirchhoff) | `@docs/ZENRON_CIRCUITS.md` |
| Self-reference (Ch17 "formula discovers itself": weak attraction only, Banach contraction fails) | `@docs/ZENRON_SELF_REFERENCE.md` |
| **Uniqueness axioms A8+A9** (OP1-OP2 candidate: k=5 via integer Ramanujan / N=12 via τ≥6; **OP3 partial: λ₂=4=spacetime dim**) | `@docs/ZENRON_UNIQUENESS_AXIOMS.md` |
| OP3+OP4 attempt (α⁻¹≈N²-7 = 137 の 0.026% near-match / m_p/m_e≈N³+9N の 0.008% near-match、likely numerology; OP4 Circulant 圧縮 + \|Aut\|=24 で partial) | `@docs/ZENRON_OP3_OP4_ATTEMPT.md` |
| **Numerology bound NB1-NB3** (near-match の統計的棄却基準; α⁻¹, m_p/m_e マッチが NB3 再現性で失敗 = numerology 判定、spacetime=4 のみ principled 残存) | `@docs/ZENRON_NUMEROLOGY_BOUND.md` |
| **Hierarchy + Discovery (H1, D1)** (Kathara^n 階層は OP3 を解決せず; 3-stage 公式発見は statistical consistent = 未知法則の discoverer) | `@docs/ZENRON_HIERARCHY_DISCOVERY.md` |
| **Final open problems** (OP3 系統 framework 見つからず honest null; **OP4: Type C が K-minimum, Type B は sub-optimal**; OP-S3 局所接触条件 L1-L4) | `@docs/ZENRON_FINAL_OPEN_PROBLEMS.md` |
| Verification scripts (`python <file>`) | `@zenron_proof_verify.py`, `@zenron_void_verify.py`, `@zenron_time_space_verify.py`, `@zenron_mult_mixing_verify.py`, `@zenron_noether_verify.py`, `@zenron_kuramoto_verify.py`, `@zenron_parallelism_verify.py`, `@zenron_circuits_verify.py`, `@zenron_self_reference_verify.py`, `@zenron_uniqueness_search.py`, `@zenron_op3_op4_search.py` |

## Source files

| module | path |
|---|---|
| Reigen (default) | `@twelve/agent/reigen.py` |
| Sentinel | `@twelve/agent/sentinel.py` |
| owl / optimize | `@twelve/optimize.py` |
| MirrorAgent / MS | `@twelve/agent/mirror_agent.py` |
| UnifiedExperience | `@twelve/agent/unified_experience.py` |
| Reigen tests | `@twelve/tests/test_reigen.py` · `@twelve/tests/test_reigen_params.py` · `@twelve/tests/test_reigen_adaptive.py` · `@twelve/tests/test_reigen_meta_knowledge.py` |
| Reigen params JSON | `@twelve/configs/reigen_params.json` (static defaults) |
| Reigen meta_knowledge | `@twelve/configs/reigen_meta_knowledge.json` (learned cross-task) |
| Test isolation fixture | `@twelve/tests/conftest.py` (meta_knowledge autouse) |
| Sentinel tests | `@twelve/tests/test_sentinel.py` |
| HTTP API | `@twelve/agent/api.py` |
| Launcher (ngrok) | `@serve_reigen.sh` |
| Friend client | `@reigen_friend_client.py` |
| Friend docs | `@FRIEND_README.md` |
| Protocol tests | `@test_reigen_api_protocol.py` |

---

## Historical notes

- **`MirrorAgent`** (export in `twelve/agent/mirror_agent.py`): thin `owl(autonomous=True)` wrapper on 20 random pts. Kept for legacy callers; prefer `reigen`/`Sentinel`/`owl` directly.
- **`meta_sentinel_hparam.py`**: multiplicative-nesting predecessor of Reigen. outer(2D) × inner(N_user) ≈ 5000 evals. Superseded by dimension-additive Reigen (+2% per self-dim vs +100%). Kept for paper reproducibility.
- **`optimize(meta=True)` Phase 3 (K7-K12 evolution)**: meta-evolution layer in `twelve/optimize.py`. Reigen does **not** use this path (Reigen's self-tuning is dimension-additive at the outer Sentinel, not via nested K7-K12). Invoke directly only for very long (>1hr) runs of primitive `optimize()`.
- **`_TunableSentinel` subclass**: internal to `reigen.py`. Overrides Sentinel's hardcoded knobs without editing `sentinel.py`. Not for direct use.
- **Experience namespace convention**: Reigen outer = `"{id}_recursive"`, inner = `"{id}_inner"`. Self-params persist cross-task via the shared outer fossil record.
