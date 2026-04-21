# Reigen (零玄) — Internal Reference

**用途**: Reigen を**編集・拡張する時**に読む。通常利用は `mimir()` 経由で済む。

Dimension-additive self-application: one outer Sentinel over (N_user + N_self)-dim joint space (N_self = 17 for kathara_17_adaptive default, 12 for kathara_12 legacy).
Internally composes Sentinel → owl → MS → multi-observer → Kathara K² → optimize → UnifiedExperience.
`twelve/agent/sentinel.py` は 2026-04-19 に **`initial_measurements=` kwarg 追加のみ**修正 (curated data 経路、その他不変)。`_TunableSentinel` subclass + `mirror_agent._mp()` monkey-patch は維持。

## 3 ways to invoke

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

## Return (extends Sentinel)

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

## Presets (`self_dim_preset`)

| preset | n_self | Hebbian | cost vs legacy | when |
|---|---|---|---|---|
| `"kathara_17_adaptive"` **(default)** | 17 | ✓ first 12 dims (adaptive lr/gate/winners from best-history) | +25-30% dim search | default since 2026-04-19 A/B: 3/3 wins (Rosenbrock 5d / Ackley 8d / Styblinski 6d), 2.7-3.3× faster + 15/15 approved (vs kathara_12's 8/15) |
| `"kathara_12"` (legacy) | 12 | ✓ Circulant(12,{1,4,6}) 30 edges, static lr | same (N_user ≤ 8), +20% (N_user > 8) | bit-identical canary baseline; explicit opt-in when you want static Hebbian |
| `"minimal_4"` | 4 | — | baseline | N_user > 16 (Rule 5 pressure); 1-shot throwaway |

`kathara_17_adaptive` の extra 5 dims (13: `self_batch_size`, 14-17: `self_hebbian_lr/min_history/percentile_gate/k_winners`) は Kathara 隣接グラフ上に**無い** — Hebbian 伝播は先頭 12 dims のみに適用 (Rule 10 保護)。14-17 は `best_p` から動的参照される outer-level パラメータ。

Switching mid-stream `minimal_4 → kathara_12 → kathara_17_adaptive`: inner fossils transfer ✓、meta_knowledge は preset 別 key なので cross-leak なし。

## Wall time model

Outer owl autonomous calls verify_fn (= 1 inner Sentinel) up to ~50×. `total_wall ≈ time_budget × wall_time_factor` (default 3). For 90s outer + inner=2s → typical 5 min. Past deadline, `combined_eval` returns cached best (soft cap).

## meta_knowledge (cross-task self-param inheritance)

Reigen reads `twelve/configs/reigen_meta_knowledge.json` at `__init__` and overlays learned best self-param values onto preset defaults. On `verdict in {"approved","pivoted"}`, `Reigen.run()` writes the run's self_best_params back (atomic + quality-gated: new score ≥ existing score).

- Per-preset keyed (`self_param_best.kathara_12`, `kathara_17_adaptive`, etc.) — no cross-preset leak
- Explicit `self_param_defaults=...` kwarg still wins over meta_knowledge (user control)
- Out-of-range stored values silently rejected (stale entries can't break validation)
- File corrupt / missing → silent fallback to preset defaults (fail-open)
- Atomic write (`tmp + os.replace`) — parallel processes never see partial JSON
- Tests isolated via `twelve/tests/conftest.py` autouse fixture (production file never written by pytest)
- Reset: delete `reigen_meta_knowledge.json` to go back to preset defaults

## When NOT to use Reigen
- N_user > 16 → bare Sentinel (preset `minimal_4` loses cross-task benefit)
- Ultra-cheap eval_fn (< 1ms) → inner Sentinel overhead dominates, use `owl()` directly
- Internal model state → Rule 11 (batch forbidden); keep eval_fn fast + multi-obs

---

## 12 self-dims (`kathara_12` default, auto-tuned)

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

## 7 JSON-overridable Reigen constants (`reigen_params.json`)

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

## Hebbian propagation (`kathara_12` and `kathara_17_adaptive`)

Top-K "winner" self-dims (largest delta vs best history) propagate `hebbian_lr × delta` to their 5 Kathara neighbors via Circulant(12,{1,4,6}). Fires only when best score > percentile-gate of history (min `hebbian_min_history` entries). Disable via `enable_hebbian=False`.

For `kathara_17_adaptive`: `hebbian_lr`, `min_history`, `percentile_gate`, `k_winners` are read from **best-history entry's self_p[13..16]** (dynamic per-call) instead of static attributes. The extra 5 dims are NOT on Kathara adjacency — propagation stays on first 12 dims only (Rule 10 preserved).

Cost: < 1ms numpy update per outer eval. Benefit: λ₂=4.0 → 3-step full propagation across all 12 dims.

**Kathara usage note**: we use the **graph-theoretic properties** (5-regular, λ₂=4.0 Ramanujan, diameter 2). We do NOT rely on `symmetric placement` (Rule 10) because Reigen's 12 dims are semantically distinct, not geometrically interchangeable. Chaos-game uniformity is irrelevant here.

## batch_eval_fn (GPU path)

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
