# CLAUDE.md — Operational Rules

All responses in Japanese.

---

## Quick Reference

### 2026-04-20 以降: `apex()` 1 呼出しが基本

**`apex()` が全ての入口**。内部で eval コストを計測、`owl` (精鋭) → `Reigen` (cascade) を自動分岐。BBOB 4 問題で **gap≈0 達成** (3/4 perfect、Rosenbrock 0.08 で basinhopping に 2 位)。

```python
from twelve.agent.apex import apex

r = apex(eval_fn, param_ranges, time_budget=300)
# r["best_params"]   : 最適パラメータ
# r["best_score"]    : 最高スコア
# r["tool_used"]     : "owl" | "owl+reigen" | "owl(reigen_tried)"
# r["dead_dims"]     : 効果のない次元 (owl 構造発見)
# r["fragility"]     : 脆い次元
# r["proxy_r2"]      : proxy 信頼度
# r["route"]         : "expensive_single" | "cheap_cascade" | "structure_only"
```

### 使用例 3 パターン

```python
# (1) 何も知らない、とりあえず最適化
r = apex(my_eval_fn, [(-5, 5)] * 8, time_budget=300)

# (2) ドメイン知識 (過去実験 20 点あり) — 高 proxy R² 期待、直接 owl 路線
r = apex(my_eval_fn, [(0.5, 2.0)] * 61,
         curated_measurements=past_lab_results,
         time_budget=600)

# (3) LLM キャリブ等 eval コスト重い場合 — 自動検出、owl 全力 (L-BFGS + multi + random-restart)
r = apex(ppl_eval_fn, [(0.5, 1.5)] * 60, time_budget=1800)
# 自動で eval_cost_hint > 0.5s → expensive_single route

# (4) 分析だけしたい (高次元で dead_dims 知るため) — 最適化 skip
r = apex(my_eval_fn, [(0.5, 1.5)] * 60,
         curated_measurements=past_data,
         mode="structure_only",        # 最適化スキップ、構造情報のみ 15-30s で返す
         time_budget=30)

# (5) 安全指標を守りたい (2 指標)
r = apex(my_eval_fn, ranges, guard_fn=my_guard_fn, time_budget=300)
```

### apex 内部分岐ロジック

```
eval コスト 1-call 実測 (midpoint で 1 回呼んで time.time 差分)
│
├── mode="structure_only"
│      └── owl autonomous=False、max 30s → 構造情報のみ return
│
├── eval > 0.5s OR curated_measurements あり → expensive_single route
│      └── owl 100% budget + 全強化 ON (L-BFGS + multi-start + random-restart=5)
│
└── eval ≤ 0.5s → cheap_cascade route
        ├── Phase 1: owl 40% budget
        │      ├── proxy_r2 ≥ 0.7 → 即終了 (tool_used="owl")
        │      └── proxy_r2 < 0.7 → Phase 2 へ
        └── Phase 2: Reigen 残 budget、wall_time_factor=1.0
               owl best を initial_user_params で warm-start、
               winner 選んで tool_used="owl+reigen"
```

### LaD (dispatch 閾値の JSON 制御)

`twelve/configs/apex_params.json` で以下 6 値を制御可能。precedence: **kwarg > JSON > hardcode**。

```json
{
    "dispatch": {
        "eval_cost_threshold": 0.5,
        "owl_share": 0.4,
        "escalation_min_remaining": 5.0,
        "confidence_skip_threshold": 0.7
    },
    "owl_kwargs": {"random_restart_count": 5, "max_iterations": 30, "min_r_squared": 0.1},
    "reigen_kwargs": {"inner_time_budget": 2.0, "wall_time_factor": 1.0}
}
```

### ツール役割階層

```
apex()              ← 表舞台 (ユーザー呼び口)
  │
  ├── owl()         ← Phase 1 + 構造発見専用でも直呼び可
  │    └── optimize()  ← primitive HC engine
  │
  └── Reigen        ← Phase 2 cascade + cross-task 学習
       └── Sentinel    ← legacy、互換維持
            └── owl
```

| 層 | 用途 | いつ直呼び? |
|---|---|---|
| **`apex()`** | 表舞台 | **全 new code で default** |
| `owl()` | 構造発見精鋭 | dead_dims/fragility/importance/proxy_fn だけ欲しい時 (apex mode="structure_only" でも OK) |
| `reigen()` | cross-task 学習累積 | 複数 task で meta_knowledge 蓄積を明示的に指定したい時のみ |
| `Sentinel` | 2 指標 legacy | 新規ほぼ不要、apex の guard_fn 経由で代替可 |
| `optimize()` | primitive | apex/owl で包めない特殊事情のみ |

### experience_id

任意の task 名札。self_params cross-task 学習は `reigen_meta_knowledge.json` 経由で自動共有される (ID 共有不要、preset 別 key で分離)。fossil は per-ID 分離で並列衝突回避。apex は内部で `{experience_id}_owl` と `{experience_id}_reigen` に suffix 付けて分離管理。

### 実測値問題 (最重要)

**curated 20 点 ≈ random 200-2000 点の情報量**。ドメインエキスパートの 20 点は proxy R² を 0.5→0.85 に引き上げる。
apex は `curated_measurements=` を受けたら expensive_single route に切替、owl に直接渡す。random `_collect` でこの価値を捨てない。

### owl 2026-04-19〜04-20 強化まとめ (apex 内で常時 ON)

| kwarg / 機能 | 分類 | 説明 |
|---|---|---|
| `measurements=[]` + `verify_fn=` + `param_ranges=` | usability | 空データ → 自動で N 点 seed |
| `curated_measurements=` | usability | intent-明示 alias |
| `guard_fn=` / `guard_threshold=` / `safe_dim_analysis=` | safety | Sentinel 秘密兵器を owl 内に移植 |
| `n_seed_samples=` / `seed_rng_state=` | usability | 空データ seed 数・RNG 制御 |
| **budget-aware autonomous loop** | **algorithm** | `time_budget` 厳守、cheap eval で max_iterations 超え可 |
| **direct-HC fallback** | **algorithm** ★ | proxy 不能時に `optimize(eval_fn=verify_fn)` 発動、Rastrigin 5d gap 45→0 の主犯 |
| **`use_lbfgs_refinement=True`** | **algorithm** ★★ | scipy L-BFGS-B 1-shot、Styblinski score -6→195.83、Rosenbrock gap 173→3 |
| **`use_multistart_fallback=True`** | **algorithm** | direct-HC fallback の warm-start を L2 diverse top-3 に |
| **`random_restart_count=K`** | **algorithm** ★★ | K 個の random warm-start 注入、basinhopping-style、Rosenbrock 救済 |

★ = 04-19 算法強化、★★ = 04-20 benchmark 駆動追加。apex は上記すべてを default ON で呼ぶ。

### 世界 Benchmark 実績 (2026-04-20、25s budget、3 seeds)

| 問題 | cma_es | basinhopping | reigen_k17 | owl | **apex** |
|---|---|---|---|---|---|
| Rastrigin 5d | +5.98 | +22.9 | **0 ✅** | **0 ✅** | **0 ✅** |
| Ackley 5d | +0.002 | +1.65 | **0 ✅** | **0 ✅** | **0 ✅** |
| Styblinski 5d | +28.3 | +28.3 | **-0.001 ✅** | +18.4 | **-0.001 ✅** |
| Rosenbrock 5d | +2.74 | **0 🏆** | +0.081 | +3.19 | +0.081 |

apex の勝ち: 3/4 perfect gap≈0 + Rosenbrock で basin に 2 位追走。
optuna_tpe / skopt_gp は既存 benchmark で圧倒敗北で除外 (Rastrigin gap +10〜+20)。

---

## apex — 全パラメータリファレンス

```python
apex(
    eval_fn,                    # f(params: list[float]) -> float, higher is better
    param_ranges,               # [(lo, hi), ...]
    *,
    # 基本
    param_names=None,           # 次元名 (optional)
    curated_measurements=None,  # 過去実験データ [{"params":..., "score":...}]; 提供時 expensive_single
    guard_fn=None,              # 安全指標 f(params) -> float、owl に safe_dim_analysis=True 経由
    experience_id="apex",       # cross-call 学習 namespace
    time_budget=300.0,          # wall 予算 (秒)
    eval_cost_hint=None,        # None = 1-call 自動測定
    mode="optimize",            # "optimize" | "structure_only"
    # LaD 個別 override (None = JSON→hardcode で解決)
    eval_cost_threshold=None,         # expensive route 切替閾値 (default 0.5s)
    owl_share=None,                    # cheap cascade の Phase 1 share (default 0.4)
    escalation_min_remaining=None,     # Reigen 起動最小残時間 (default 5s)
    confidence_skip_threshold=None,    # Reigen skip の proxy_r2 閾値 (default 0.7)
    random_restart_count=None,         # owl の random restart 数 (default 5)
    reigen_inner_time_budget=None,     # Reigen inner Sentinel budget (default 2s)
    reigen_wall_time_factor=None,      # Reigen wall cap factor (default 1.0)
    # 稀に使う
    force_cascade=False,        # True で confidence 無視し必ず Reigen 走らせる
    n_seed_samples=None,        # 空データ seed 数
    apex_cfg=None,              # 上記 LaD 値を dict でまとめて指定
    verbose=False,
) -> dict
```

返り値 dict:
```python
{
    "best_params":   list|dict,
    "best_score":    float,
    "tool_used":     "owl" | "owl+reigen" | "owl(reigen_tried)" | "owl_structure_only",
    "route":         "expensive_single" | "cheap_cascade" | "structure_only",
    "mode":          "optimize" | "structure_only",   # structure_only の時のみ
    "eval_cost_s":   float,                           # 自動測定 or hint
    "confidence":    str,                             # owl の confidence
    "dead_dims":     list,
    "active_dims":   list,
    "fragility":     list,
    "proxy_type":    str,
    "proxy_r2":      float,
    "owl_result":    dict,                            # 生の owl 返り値
    "reigen_result": dict,                            # escalation 時のみ
    "elapsed_s":     float,
}
```

---

## Reigen (零玄) — apex の Phase 2 cascade 実行体

Dimension-additive self-application: one outer Sentinel over (N_user + N_self)-dim joint space (N_self = 17 for kathara_17_adaptive default, 12 for kathara_12 legacy).
Internally composes Sentinel → owl → MS → multi-observer → Kathara K² → optimize → UnifiedExperience.
`twelve/agent/sentinel.py` は 2026-04-19 に **`initial_measurements=` kwarg 追加のみ**修正 (curated data 経路、その他不変)。`_TunableSentinel` subclass + `mirror_agent._mp()` monkey-patch は維持。

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

**2026-04-19 以降、owl 自身にも direct-HC fallback 移植済** (commit 911997e)。autonomous + verify_fn があれば Sentinel wrap 不要で同じ動作。新規 code では owl 直使い推奨、Sentinel は legacy 互換で残置。

---

## owl() — structure discovery + proxy optimization + direct-HC fallback

```python
r = owl(data, verify_fn=eval_fn, autonomous=True, max_iterations=10,
        experience_id="my_task", time_budget=300)
# data: list of {"params": {...}|list, "score": float}            (single-observer)
#       or [{"params": ..., "scores": {"obs1": ..., "obs2": ...}}] (multi-observer)
```

| option | effect | notes |
|---|---|---|
| measurements=[] | empty-data autonomous start — seed via verify_fn + param_ranges | 2026-04-19 added |
| curated_measurements= | explicit alias for `measurements=` (domain-curated intent) | 2026-04-19 added |
| verify_fn | proxy optimum を real eval で verify; data 成長に使用 | prevents proxy hallucination |
| guard_fn= / guard_threshold= / safe_dim_analysis= | 安全指標ゲート + multi-observer safe_dim 分解 | 2026-04-19 added (Sentinel pivot 移植) |
| autonomous=True | 停滞 → range perturb → 継続、**budget-aware** (time_budget 厳守) | cheap eval で max_iterations 超え可 |
| experience_id | cross-call learning with same ID | smarter per call |
| time_budget | seconds (default 60) | **autonomous 時は TOTAL wall budget** |
| max_iterations | autonomous round cap (default 10) — budget 残ってれば hard cap 200 まで継続 | |
| n_rounds | verify_fn rounds; auto=5 when verify_fn set | |
| stagnation_threshold | autonomous range-perturb trigger (default 3) | added for Reigen |
| min_r_squared | proxy confidence threshold; default 0.7 | lower for noisy data |
| force_proxy_type | "zenron"/"zenron_interact"/"linear"; default R²-best | |
| n_seed_samples / seed_rng_state | empty-data path 制御 | 2026-04-19 added |
| kathara="auto" | 6-node K² parallel; auto if verify_fn importable + budget ≥ 120s | |

### Confidence bands
`r² ≥ 0.7` → high. `0.3–0.7` → low. `< 0.3` → **insufficient (proxy 不能) だが autonomous + verify_fn があれば direct-HC fallback が発動**。

### Direct-HC fallback (2026-04-19 追加、commit 911997e) ★

**owl 最大の算法強化**。proxy 不能な多峰 / deceptive / plateau 地形で effect 大。

**発動条件**:
- `autonomous=True` + `verify_fn` 提供
- 以下のどちらかが真:
  - A. proxy_r2 < 0.3 (insufficient): insufficient branch で自動発動
  - B. proxy_r2 < min_r_squared + verified_score < max(measurements) (proxy 騙された): Step 5b で発動

**動作**:
1. `growing_data` から best measurement 特定
2. その params を warm-start として `optimize(eval_fn=verify_fn, initial_params=warm, ...)` を直接起動
3. 結果が既存 best を上回れば best_result 更新 (`confidence="direct"`)

**効果実測 (A/B 2026-04-19、Rastrigin 5d、eval_fn=20ms/call)**:
- Before: gap=45.6 (proxy 頼み、諦め)
- After: **gap=0.000** (global optimum 到達)
- eval 数 1/5 (279 vs reigen 1465)

これで owl は verify_fn あれば reigen/Sentinel 不要、多峰 landscape も突破できる汎用最適化器に昇格。

### Budget-aware autonomous loop (2026-04-19 追加、commit 1addbd9)

従来: `autonomous=True` で max_iterations=10 固定、cheap eval で予算余らせ。
現行: `time_budget` を TOTAL 予算として厳守。budget 残っていれば max_iterations 超え (hard cap 200) で継続。inner optimize の budget も残時間に応じて動的分配。

### L-BFGS-B gradient refinement (2026-04-20 追加、commit edef9f5) ★

**smooth curved valley (Rosenbrock 型) / 局所解停滞 (Styblinski 型) の決定打**。owl 末尾 1-shot で scipy.optimize.minimize(L-BFGS-B) を発動、finite-diff gradient で頂点まで登り切る。

**発動条件** (opt-in、デフォ OFF で既存挙動完全保護):
- `use_lbfgs_refinement=True` kwarg
- `verify_fn` 提供 + `best_params` 非 None
- scipy import 成功

**動作**:
1. 全 autonomous round 完了後、best_params を x0 として `minimize(method="L-BFGS-B", bounds=ranges, maxfun=50)`
2. 改善したら `best_result` 更新、`confidence="lbfgs_refined"` (既に "direct" なら "direct+lbfgs")
3. scipy 失敗 / エラー → silent fallback (既存 best 保持)

**コスト**: +50 eval/run fixed cap。GP+EI のような per-round 爆発なし。

**実測効果 (2026-04-20)**:
- Styblinski 5d: **score -6.0 → 195.83** (gap 201 → -0.11、global optimum 到達)
- Rosenbrock 5d: **gap 173 → 3.0** (57× 改善)

### Multi-start warm-start diversification (2026-04-20 追加、commit a3c8a4c)

direct-HC fallback の warm-start を「互いに離れた top-K 点」にして wrong-basin 脱出。

**発動条件** (opt-in):
- `use_multistart_fallback=True` kwarg
- `growing_data >= 3`
- fallback 発動時のみ (通常経路は単一 warm-start のまま)

**動作**:
1. growing_data を score 降順 sort
2. L2 距離 >= `0.15 * mean(hi-lo)` で重複排除しつつ top-3 diverse 選出
3. 各 warm-start で `optimize()` を `total_budget / K` 実行
4. 全 warm-start の best を採用

**コスト**: fallback 発動時のみ +3× warm-start (budget 分割で合計時間は維持)。

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
| discrete / non-smooth params via owl() + autonomous | 旧: Sentinel でラップ必要 | 2026-04-19 以降 owl に direct-HC fallback 移植、単体で対応可 |
| range `[0.5, 1.5]` | range includes 0 | scale=0 catastrophic (PPL=262144, proven) |
| apply → measure → restore | state leaks | measurement contamination |
| eval on training data | eval on test answers | cheating |
| curated 過去 data → `owl(measurements=past, ...)` | Reigen で random `_collect` | curated data は 10-100× 情報量、捨てるな |

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

Quick-reference table (see per-rule subsections below for the prose form):

| # | rule | see |
|---|---|---|
| -1 | Strip to essence: `x_i, perturb, share, eval_fn` | Principles |
| 0 | Measure don't guess: ≥5 pts → owl() → read numbers | owl() |
| 1 | Ask Oracle for structural questions | — |
| 2 | No manual tuning: data → owl() | owl() |
| 3 | LaD: no if/else — convert to numeric params | Principles |
| 4 | `importance = (truth × max(connectivity, floor))^exp` | Principles |
| 5 | Overfitting: n_problems > n_params | — |
| 6 | Discrete/int params OK via Sentinel (auto-fallback) | Sentinel |
| 7 | **scale=0 forbidden**. Never include 0 in ranges | eval_fn |
| 8 | Two metrics → Sentinel / Reigen. 1 metric → `guard_fn=eval_fn` | Reigen, Sentinel |
| 9 | **Default to owl** (2026-04-19). Reigen only for cross-task or deceptive multi-peak | owl / Reigen |
| 10 | Kathara 0.993 uniformity requires N=12 + 5-regular + symmetric placement | Reigen Ref |
| 11 | `batch_eval_fn` is forbidden for internal model state | Reigen, owl() |

### Rule -1: Strip to essence

Reduce any problem to the four Zenron primitives before touching tools. Identify
the state variable `x_i`, the perturbation operator on it, the sharing relation
with neighbors, and the scalar eval_fn that scores the outcome. If you cannot
name all four cleanly, you are not ready to optimize yet.

### Rule 0: Measure, don't guess

Never guess at parameter importance or interaction structure. Collect at least
five concrete measurements, feed them to `owl()`, and read the numbers. Human
intuition about 20+ dimensional landscapes is unreliable; the proxy-R² score
tells you when you have enough data to trust a recommendation.

### Rule 1: Ask the Oracle for structural questions

When the question is about topology, graph structure, or problem classification,
invoke the oracle MCPs rather than guessing. The relevant tools are
`arc_oracle.py`, `kathara_oracle.py`, and the documentation at
`@twelve/ORACLE.md`. Oracle answers are cached and cheap.

### Rule 2: No manual tuning

If you have a scalar metric and a parameter range, you almost never need to
hand-tune. Collect data and pass it to `owl()`; let the proxy-extraction layer
find structure. Manual grid searches are only justified when you need auditable
intermediate steps for a report.

### Rule 3: LaD means no if/else

Logic-as-Data: replace conditional branches with numeric parameters that the
optimizer can vary. Every `if` in eval_fn becomes a dimension. Every threshold
becomes a range. This lets `owl()` discover the cutover points itself instead
of you guessing them.

### Rule 4: Importance formula

The canonical MirrorScan score is `importance = (truth × max(connectivity, floor))^exp`
with defaults `exp=0.3064` and `floor=0.1411` from `configs/ma_meta_params.json`.
Multiplication of truth and connectivity is what kills noise: a dimension that
correlates with the score but never co-moves with other dimensions is flagged
as accidental correlation and falls out of the ranking.

### Rule 5: Guard against overfitting

Keep `n_problems > n_params`, optimize on the training split, and verify on a
held-out bench. With more parameters than problems, the proxy can memorize
noise; its R² becomes meaningless. When in doubt, widen the problem set
before adding parameters.

### Rule 6: Discrete and integer params are fine

`Sentinel` auto-falls-back from owl proxy to direct HC when the proxy collapses
on non-smooth landscapes. Pass `initial_params` to warm-start the fallback.
Integer and discrete spaces handle exactly this way; no special encoding is
needed.

### Rule 7: scale=0 is forbidden

Never include zero in a parameter range. A verified LLM calibration run with
`scale=0` produced PPL=262144 in one step, wiping the model. Use ranges like
`(0.5, 1.5)` instead. `verify_fn` is the second line of defense: it catches
proxy hallucinations, but Rule 7 is the first line, and the only cost-free one.

### Rule 8: Two-metric flow

When you have two metrics, use `Sentinel` or `Reigen`: one metric is the
`eval_fn` to optimize, the other is the `guard_fn` to protect. Sentinel will
auto-pivot if optimizing eval_fn harms guard_fn. For the one-metric case, pass
`guard_fn=eval_fn` so the guard is satisfied by construction.

### Rule 9: Default to owl

As of 2026-04-19, `owl()` is the default optimization entry point. The standard
call is `owl(measurements=curated, verify_fn=..., autonomous=True)`. The
direct-HC fallback inside owl now handles multi-peak landscapes directly, and
owl is 5-16× more eval-efficient than Reigen on expensive eval_fns. Reigen
remains worthwhile only when you need cross-task learning accumulation or when
eval is cheap and the landscape is deceptive enough that brute-force optimize()
wins. The Reigen default preset is `kathara_17_adaptive`; it writes learned
self-params to `reigen_meta_knowledge.json` automatically.

### Rule 10: Kathara uniformity conditions

The chaos-game uniformity result of 0.993 requires three simultaneous
properties: N=12 nodes, 5-regular graph, and symmetric placement. Break any
one and uniformity collapses. When applying Kathara to a new domain, audit all
three. Inside Reigen we use the graph properties only (Circulant(12,{1,4,6}),
λ₂=4.0, diameter 2), not the placement uniformity, so Rule 10 does not bind
Reigen even though it cites Kathara.

### Rule 11: batch_eval_fn is for external params only

`batch_eval_fn` is reserved for external parameters such as learning rate,
dropout, or prompt tokens. It is forbidden for anything that touches internal
model state — KV cache scale, weight scale, LoRA adapters — because those
share a single global state and cannot be evaluated in parallel. When you hit
this case, keep `eval_fn` under two seconds, return a multi-observer dict such
as `{"nll": -ppl, "hs": hs_score, "mmlu": mmlu_score}`, and call `owl()`
directly to read `stable_active` and `stable_dead`.

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
| 04-19 | **kathara_17_adaptive A/B 3/3 勝利** | Rosenbrock 5d / Ackley 8d / Styblinski 6d で vs kathara_12: 2.7-3.3× 速 + 15/15 approved (vs 8/15)。default に昇格 (commit 9381df1) |
| 04-19 | **Rule 9 (genesis 学習累積) 実装** | meta_knowledge.json 新設 + atomic write、Reigen.run() 成功時に self_param_best 書戻し。これまで「累積」は建前だけだった (commit 888dfc1) |
| 04-19 | **owl direct-HC fallback 移植** | Sentinel 秘密兵器を owl 本体に。Rastrigin 5d で gap 45.6 → 0.000 (global optimum)、1/5 eval 数。autonomous + verify_fn で多峰 landscape 突破可能に (commit 911997e) |
| 04-19 | **owl budget-aware autonomous loop** | time_budget を TOTAL 予算として厳守、cheap eval で max_iterations 超え継続 (commit 1addbd9) |
| 04-19 | **Sentinel curated-data path** | `initial_measurements=` kwarg、Sentinel/Reigen 経由でもドメイン知識注入可能に (commit 6ed20c0) |
| 04-19 | **A/B realistic (20ms/eval)** | owl vs reigen: 同 budget で reigen 2/3 勝ち gap 小だが eval 5-16× 多 (= LLM 換算で実用不能)。owl は eval 効率で実戦優位 |
| 04-20 | **世界 benchmark: Reigen 3/4 勝利** | Rastrigin/Ackley/Styblinski で Reigen_k17 が optuna/skopt/cma/basinhopping 全てを圧倒 gap=0。Rosenbrock のみ basinhopping に僅差負け (gap 0.08 vs 0). owl_direct は Rastrigin/Ackley で eval 効率最強 (332 eval で gap=0) |
| 04-20 | **owl L-BFGS-B refinement** | scipy L-BFGS-B を owl 末尾 1-shot で発動、`use_lbfgs_refinement=True` opt-in。Styblinski 5d で score -6→195.83 (global optimum 到達)、Rosenbrock で gap 173→3 (57× 改善) (commit edef9f5) |
| 04-20 | **owl multi-start fallback** | direct-HC fallback の warm-start を L2 diverse top-3 に、`use_multistart_fallback=True` opt-in。wrong-basin 脱出機構 (commit a3c8a4c) |
| 04-20 | **owl random_restart_count** | multi-start に K 個の uniform random warm-start を注入 (basinhopping 模倣、L-BFGS-B 直行)。curved-valley 救済 (commit 62d20c1) |
| 04-20 | **apex 上位層誕生** | owl+Reigen cascade の meta-dispatcher。eval コスト 1-call 測定 → 自動分岐。BBOB 3/4 gap=0 + Rosenbrock 0.08 (basin 追走)。新 default (commit 62d20c1 + 8472c6d + rename) |
| 04-20 | **apex LaD 化** | dispatch 6 閾値を `apex_params.json` で制御可能、kwarg > JSON > hardcode 優先度。K² 自己最適化は病的 runtime で失敗 (Ch17 self-application は turnkey でない、remediation 必要) |
| 04-20 | **apex mode="structure_only"** | 最適化 skip、owl の dead_dims/fragility/proxy_r2 のみ返す高速分析経路 (max 30s cap)。高次元 LLM 事前分析用 |
| failures | scale=0 → PPL=262144 | Rule 7. verify_fn catches hallucination |
| failures | single-obs on internal model state | fix: multi-observer + fast eval (Rule 11) |

Archival entries (individual runs, method evolution): `@docs/LAB_NOTES.md`.

---

## Hardware & safety

### Workstation

The development workstation runs Windows 11 Pro on an Intel i7-12700K with an
NVIDIA RTX 5090 (32 GB VRAM). The pre-built llama.cpp binaries live at
`c:/Users/user/llm/llama-bin/` (version b8795, CUDA 12.4). Model GGUF files
live at `c:/Users/user/llm/models/`. Visual Studio and GCC are not installed,
so llama.cpp is never built from source here.

### Commit safety

Never commit any of the following: `unified_memory.py`, `evaluator*.py`,
anything under `_legacy/`, patent drafts, or credentials. Check `git status`
before every commit. Prefer staging specific files by name over `git add -A`
so that accidental `.env` or credential commits are avoided.

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
| **apex (new default, 2026-04-20)** | `@twelve/agent/apex.py` |
| apex params (LaD JSON) | `@twelve/configs/apex_params.json` |
| Reigen (cascade backend) | `@twelve/agent/reigen.py` |
| Sentinel | `@twelve/agent/sentinel.py` |
| owl / optimize | `@twelve/optimize.py` |
| MirrorAgent / MS | `@twelve/agent/mirror_agent.py` |
| UnifiedExperience | `@twelve/agent/unified_experience.py` |
| **apex tests** | `@twelve/tests/test_apex.py` (smoke / cascade / guard / curated / structure_only / wall-time, 10/10 green) |
| Reigen tests | `@twelve/tests/test_reigen.py` · `@twelve/tests/test_reigen_params.py` · `@twelve/tests/test_reigen_adaptive.py` · `@twelve/tests/test_reigen_meta_knowledge.py` |
| owl enhancements tests | `@twelve/tests/test_owl_enhancements.py` (empty-data / curated / guard_fn / safe_dim) |
| owl refinements tests | `@twelve/tests/test_owl_refinements.py` (L-BFGS-B / multi-start、2026-04-20) |
| World benchmark scripts | `@benchmark_apex.py` (apex vs cma/basin/reigen/owl、2026-04-20) · `@benchmark_apex_meta.py` (K² self-opt 試作、⚠ 病的 runtime で失敗) · `@benchmark_owl_vs_world.py` (legacy、vs optuna/skopt) · `@benchmark_refinements.py` (04-20 refinement 効果測定) |
| Sentinel curated-data tests | `@twelve/tests/test_sentinel_curated.py` (initial_measurements) |
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
