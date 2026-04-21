# owl() — Internal Reference

**用途**: owl を**編集・拡張する時**、もしくは `mimir()` 経由せず直呼びする特殊ケースで読む。通常は `mimir()` 経由 (mimir が owl を内部で呼ぶ)。

## Overview

`owl()` — structure discovery + proxy optimization + direct-HC fallback.

```python
r = owl(data, verify_fn=eval_fn, autonomous=True, max_iterations=10,
        experience_id="my_task", time_budget=300)
# data: list of {"params": {...}|list, "score": float}            (single-observer)
#       or [{"params": ..., "scores": {"obs1": ..., "obs2": ...}}] (multi-observer)
```

## Options

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
| use_lbfgs_refinement | scipy L-BFGS-B 末尾 refinement (opt-in) | 2026-04-20 added |
| use_multistart_fallback | fallback warm-start を L2 diverse top-3 に (opt-in) | 2026-04-20 added |
| random_restart_count | uniform random warm-start K 個注入 (opt-in、default 0) | 2026-04-20 added |

## Confidence bands

`r² ≥ 0.7` → high. `0.3–0.7` → low. `< 0.3` → **insufficient (proxy 不能) だが autonomous + verify_fn があれば direct-HC fallback が発動**。

## Direct-HC fallback (2026-04-19 追加、commit 911997e) ★

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

## Budget-aware autonomous loop (2026-04-19 追加、commit 1addbd9)

従来: `autonomous=True` で max_iterations=10 固定、cheap eval で予算余らせ。
現行: `time_budget` を TOTAL 予算として厳守。budget 残っていれば max_iterations 超え (hard cap 200) で継続。inner optimize の budget も残時間に応じて動的分配。

## L-BFGS-B gradient refinement (2026-04-20 追加、commit edef9f5) ★

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

## Multi-start warm-start diversification (2026-04-20 追加、commit a3c8a4c)

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

## Random restart injection (2026-04-20 追加、commit 62d20c1)

multi-start fallback に `K` 個の uniform random warm-start を追加、basinhopping 模倣。K 個それぞれ L-BFGS-B 直行 (optimize() スキップ、maxfun=100)。curved-valley 問題 (Rosenbrock) 救済。

**発動条件**:
- `random_restart_count=K` (K > 0) opt-in
- `use_multistart_fallback=True` と併用

## 5 readings from 1 owl() computation

| read | meaning | example |
|---|---|---|
| importance | what matters | "L24, L27 drive intelligence" |
| dead_dims | what's irrelevant | "logP doesn't affect binding" |
| fragility | what's vulnerable | "hidden_size is fragile" |
| proxy_fn(x) | what-if prediction | "this config → score≈85" |
| best_params | optimal values | "use these scales for best PPL" |

Read top→bottom. `best_params` is a side effect of structure discovery, not the main output.

## Multi-observer

```python
r = owl([{"params": {...}, "scores": {"reasoning": 0.85, "math": 0.72}}, ...])
r["stable_active"]       # active in ALL observers (universal structure)
r["stable_dead"]         # dead in ALL observers (true noise)
r["observer_dependent"]  # varies by observer (context-dependent)
r["observers"]           # per-observer: importance, fragility, dead_dims
```

Single `"score"` key remains backward-compatible. Sentinel / mimir uses this internally on guard_fn pivot: `{eval, guard}` → safe/conflict dims.
