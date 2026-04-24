# CLAUDE.md — Operational Rules

All responses in Japanese.

## 最優先ルール: `mimir_odin_stable()` を呼べ

全ての最適化は `mimir_odin_stable(eval_fn, param_ranges, time_budget=N)` で始まる。2026-04-24 以降の新 default エントリーポイント。

`mimir_odin_stable()` はオーディン (4 specialist 並列で peak 発見) → stabilizer (peak 周辺を Metropolis で探索し plateau 変換) の 2 段 pipeline。実測で**摂動耐性 8% → 96%** (10 倍改善)。実世界で「1mm ズレても壊れない答え」を保証。

**日常呼称**: 「**オーディン (stable)**」または「**安定オーディン**」。関数呼出は `mimir_odin_stable()`。

**名前の由来**: オーディン (北欧神話の主神) = 知恵を求めて片目を捧げた決断者。mimir (ミーミル: 知恵の神) から 4 specialist 経由で最適解候補を得て、その中から**最良を選択する決断者**。stable 版は更にその最良を Metropolis + cooling で**安定化** (plateau 化) する。

**時間コスト**: `time_budget` に対し実測 +25% (stabilize stage 分)。time_budget=300 なら実測 ~400s 想定。

**例外 — 旧 default `mimir_odin()` を直呼びする場面 2 つのみ**:
1. 論文 / 理論ベンチマーク (BBOB 等) で peak が fragile でも問題にならない
2. 時間が極端に厳しく 25% overhead が致命的

更に下層の `mimir()` 直呼びは 3 例外 (単純問題 / 1 core / GGUF patch、Rule 9 詳解)。

**ミーミル家族** (上から default 優先):
- `mimir_odin_stable()` — ★ default (Rule 9/16、peak→plateau、実世界推奨)
- `mimir_odin()` — 旧 default (Rule 13、論文ベンチ用、time -25%)
- `mimir()` — 内部実装 (3 例外時のみ直呼び)
- `mimir_cardinal_hierarchy()` — Council → GA (Rule 14、特殊)
- `mimir_cardinal_coevolution()` — params × weights 共進化 (Rule 15、特殊)
- `stabilizer.stabilize()` — 汎用 peak→plateau 変換 (stand-alone、他 optimizer に後付け可)

`mimir_odin()` は 4 specialist の mimir (default / lad / expensive / scipy-forced) を並列実行し、最良 best_score を採用する。単独 mimir の弱点 (低/高 noise / 多峰 / 高次元 gradient) を相互補完、「どの specialist の成績 ≤ council 成績」が保証される (取り方が max なので理論的下限 = 最強 specialist 単独)。

単独 `mimir()` が適切な例外は 3 つのみ: (a) 2-3d convex の超単純問題、(b) CPU 1 core / メモリ < 4GB 環境、(c) GGUF patch 系 eval_fn (disk/メモリ競合で council 逆効果)。

**ベンチ実績**:
- stochastic Rastrigin 10d (3 seeds × 2 noise): **council gap ≈ 0 全 6 run**、単独 mimir は gap 31-201 (不安定)
- LLM eval (Qwen 3.6): wall time 1.06× (GPU concurrent 並列化成功)
- BBOB 4 問題で gap≈0 達成、Rastrigin 20d で cma_es の 128× 優位、Rosenbrock 20d scipy cascade で gap=0 (単独 mimir ベースの実績、council はさらに安定)

## 最小使用法 (stable)

```python
from twelve.agent.mimir_odin_stable import mimir_odin_stable

r = mimir_odin_stable(eval_fn, param_ranges, time_budget=400)
print(r["best_params"])            # ★ plateau centroid (robust、実用推奨)
print(r["best_score"])             # plateau の score
print(r["plateau_robustness"])     # 摂動耐性 (通常 0.7-0.9)
print(r["peak_robustness"])        # 元の peak 耐性 (通常 0.1-0.3、比較用)
print(r["specialist"])             # 勝った odin specialist (diagnostic)
```

**旧 default (`mimir_odin()` 直呼び、論文ベンチ時のみ)**:
```python
from twelve.agent.mimir_odin import mimir_odin
r = mimir_odin(eval_fn, param_ranges, time_budget=300)
```

**内部実装 (`mimir()` 直呼び、3 例外時のみ)**:
```python
from twelve.agent.mimir import mimir
r = mimir(eval_fn, param_ranges, time_budget=300)
```

返り値 dict 主要キー (stable): `best_params` (★ plateau centroid) / `best_score` / `peak_params` / `peak_robustness` / `plateau_robustness` / `plateau_width` / `plateau_particles` / `robustness_improvement` + odin 由来 (`specialist` / `council` / `council_variance_std`) + mimir 由来 (`dead_dims` / `active_dims` / `proxy_r2` / `tool_used` / `route`) 全部引き継ぎ。構造発見系は最適化と同時に得られる。

## mimir_odin_stable 使用 6 パターン

全て `mimir_odin_stable()` 1 関数で扱える。extra_kwargs が odin → 全 specialist に passthrough される。time_budget に +25% 余裕を持たせる (stabilize stage 分)。

第1. **何も知らない最適化**: `mimir_odin_stable(fn, ranges, time_budget=400)` で終わる。
第2. **過去 data あり**: `curated_measurements=past` extra_kwargs で odin stage の seed を底上げ。
第3. **LLM キャリブ等 eval 重い**: そのまま渡す。specialist "lad" が n_samples=20 集約で noise 除去、stabilizer が plateau 化。
第4. **構造分析のみ**: `mode="structure_only"` は単独 `mimir(mode="structure_only")` 推奨 (stable 要らない、overkill)。
第5. **2 指標 guard**: `guard_fn=my_guard` extra_kwargs で odin 全 specialist に伝搬、安全指標 pivot 発動。
第6. **stochastic / 自動集約**: specialist "lad" が `n_samples_per_eval=20` を自動担当、ユーザー側追加 kwarg 不要。

```python
r1 = mimir_odin_stable(fn, [(-5, 5)] * 8, time_budget=400)                             # (1) 一般
r2 = mimir_odin_stable(fn, ranges, curated_measurements=past_data, time_budget=800)    # (2) 過去 data
r3 = mimir_odin_stable(ppl_eval, ranges, time_budget=2400)                             # (3) LLM キャリブ
r4 = mimir(fn, ranges, mode="structure_only", time_budget=30)                          # (4) 分析のみ (単独 OK)
r5 = mimir_odin_stable(fn, ranges, guard_fn=my_guard, time_budget=400)                 # (5) 2 指標
r6 = mimir_odin_stable(llm_eval, ranges, time_budget=2400)                             # (6) stochastic (lad 自動)
```

**旧 default `mimir_odin()` を直呼びする 2 例外**:
- (a) BBOB / 論文ベンチマークで peak fragile でも gap=0 のみ評価される
- (b) 時間が極端に厳しく 25% overhead が致命的 (典型: 1h 予算の中で stable は 15 分無駄になる)

**更に内側 `mimir()` が適切な 3 例外** (stable / odin 両方 overkill):
- (c) 2-3d convex な超単純問題 (stable / odin の並列 overhead で逆に遅い)
- (d) CPU 1 core / メモリ < 4GB 環境 (4 プロセス並列の恩恵なし)
- (e) GGUF patch + 評価系 eval_fn (disk/メモリ競合で逐次化、並列破綻)

## mimir の返り値

```
best_params, best_score, tool_used, route, eval_cost_s, confidence, elapsed_s
# 構造発見 (owl 由来)
dead_dims, active_dims, fragility, proxy_type, proxy_r2
verified_score (実測スコア), param_names, rounds_completed, recovered_dims, n_measurements
# multi-observer side-analysis (LaD seed / dict eval_fn / n_samples_per_eval>1 時のみ、Rule 12)
stable_active, stable_dead, observer_dependent
dead_observers, observers, observer_correlations
multi_observer_side_analysis (True flag)
# raw 生 dict (詳細読み取り)
owl_result, reigen_result (escalation 時), scipy_result (dim≥10 時)
```

tool_used 値: `"owl"` / `"owl+reigen"` / `"owl+scipy"` / `"owl(reigen_tried)"` / `"owl(reigen_scipy_tried)"` / `"owl_structure_only"`
route 値: `"expensive_single"` / `"cheap_cascade"` / `"structure_only"`

## mimir_odin_stable の返り値 (default 出力)

stable は odin の全キー + stabilizer 由来キーを追加。**`best_params` は plateau centroid に上書き** (実用推奨)、元の peak は `peak_params` に保存:

```
# ★ 実用推奨 (plateau = robust な答え)
best_params        : plateau centroid (list)
best_score         : centroid の実測 score
plateau_score      : 同上 (alias)
plateau_robustness : 摂動耐性 0-1 (期待 >0.7)
plateau_width      : 各次元の std (list、信頼区間的幅)
plateau_particles  : stabilizer 12 粒子の最終位置 (ensemble 用)

# 比較用 (peak = 元のfragile な最適)
peak_params        : odin が見つけた sharp peak (list)
peak_score         : peak の実測 score (通常 plateau_score より少し高い)
peak_robustness    : peak の摂動耐性 (通常 0.1-0.3)
robustness_improvement: plateau - peak (改善差分、典型 +0.5-0.8)

# diagnostics
stabilize_elapsed_s
stage1_elapsed_s
stabilize_applied  : True (fallback 時のみ False)

# odin 由来 (内部の council から引き継ぎ)
specialist                 : 勝者名 ("default" / "lad" / "expensive" / "scipy-forced")
specialist_role            : 勝者の役割説明 (日本語)
council                    : [(name, score), ...] 全 specialist の score 降順
council_variance_std       : best_score の std (問題難易度 signal)
council_elapsed_s, n_specialists_ran, n_specialists_failed, council_errors

# mimir 由来 (更に内部、勝者 specialist から引き継ぎ)
dead_dims / active_dims / fragility / proxy_r2 / proxy_type / tool_used / route /
rounds_completed / recovered_dims / n_measurements / verified_score ...
# multi-observer (LaD seed / dict eval_fn 時のみ)
stable_active / stable_dead / observer_dependent / dead_observers / observers /
observer_correlations / multi_observer_side_analysis
```

**council_variance_std の解釈**:
- std < 0.1 → 全 specialist 同じ答え、問題易しい、次回 odin 単独で十分
- std > 10 → specialist 毎に大差、問題難しい、stable 継続要

## 実測は自動化される — 手で 1 点ずつ測るな

`mimir_odin_stable(eval_fn, ranges, time_budget=400)` の **1 行で 2 stage 全自動**:
Stage 1 (75%): odin 4 specialist 並列で peak 発見 (seed 生成 → 実測 → proxy fit → argmax 再実測 → 収束判定 → range 拡張)
Stage 2 (25%): stabilizer 12 粒子 × 20 gens Metropolis で peak → plateau 変換
→ robust な `best_params` 返却 (摂動耐性 8%→96%)
ユーザーの仕事は `eval_fn` 書くだけ、手動ループ不要 (詳細は Rule 0/2)。

### 過去データは必ず `curated_measurements=` に渡せ

過去に手動で測った点があるなら**絶対に捨てるな**。seed にして精度を爆上げできる。

```python
past = [{"params": [0.8, 1.2], "score": 0.71},
        {"params": [1.0, 1.3], "score": 0.68}, ... ]   # 過去の実測 20 点
r = mimir_odin(eval_fn, ranges, curated_measurements=past, time_budget=600)
```

### proxy_r2 の違い (同じ実測数でも桁違い)

| 始点 | 初期 proxy_r2 |
|---|---|
| empty start (mimir が random 20 点実測) | 0.5 〜 0.7 |
| curated 20 点 (ドメイン知識) | **0.80 〜 0.90** |
| curated 50 点 | **0.90 〜 0.95** |
| stochastic eval_fn, `n_samples_per_eval=1` (Rule 12) | 0.3 〜 0.5 (noise fit、崩壊) |
| stochastic eval_fn, `n_samples_per_eval=20` (Rule 12) | **0.7 〜 0.9** (N 集約で noise 除去) |

curated = random の **10-100× 情報量**。ドメインエキスパートの選んだ 20 点は random 200-2000 点に相当する。

### 推測では始めない、常に実測で proxy 構築

mimir の proxy は**全て実測値から fit** される。推測・合成データは混ざらない。proxy_r2 が低い (< 0.3) 時は **direct-HC fallback** が自動発動して proxy を捨て、実 eval_fn で直接最適化に切り替わる。**推測で押し切ることは設計上ない**。

## eval_fn を書いたらまず check_eval_fn() で診断

**本番 `mimir_odin_stable()` (or `mimir_odin()` / `mimir()`) の前に必ず走らせろ**。30-60 秒で bad eval_fn を自動検出、本番 1 時間の無駄走を防ぐ。

```python
from twelve.agent.eval_check import check_eval_fn, format_report
from twelve.agent.mimir_odin_stable import mimir_odin_stable

diag = check_eval_fn(my_eval_fn, param_ranges, time_budget=60)
print(format_report(diag))

if not diag["ok"]:
    # issue を修正してから本番へ。fatal なら止まる
    raise ValueError(f"eval_fn bad: {diag['issues']}")

# OK なら本番 (stable が default、time_budget に +25% 余裕)
r = mimir_odin_stable(my_eval_fn, param_ranges, time_budget=2400)
```

検出する bad パターン:

| 症状 | 検出方法 | severity |
|---|---|---|
| midpoint で exception | 1-call probe | **fatal** |
| NaN / None 返す | 1-call probe | **fatal** |
| 2 点で同値 (constant / state leak) | 2-call probe 差分 | **fatal** |
| proxy_r2 < 0.2 (noisy / 多峰 / 不連続) | mimir structure_only | warn |
| fragility 突出 (崩壊因子、scale=0 型) | fragility max/median | warn |
| active_dims < 2 (高 dim で 1 次元的) | active_dims 数 | warn |
| multi-obs dict で stable_active 空 | observer 整合性 | warn |
| 同一 params で高 CV (stochastic eval_fn) | N-call stability probe (MAD/\|median\|) | warn |

**fatal なら本番走らせるな**。原因修正が先。stochastic 検出時 (`stochastic_cv > 0.10`) は `wrap_stochastic(n=20)` / `wrap_multi_obs(n=20)` / `n_samples_per_eval=20` を recommendations に自動追加 (Rule 12)。

**背景**: eval_fn = 人間の価値観定義、完全自動生成は原理不可能 (docs/全論の公式の活用.md §11.2)。**bad パターン検出**で 80% の失敗を事前回避できる。

## eval_fn 設計

| good | bad | why |
|---|---|---|
| `return -ppl` (higher=better) | `return ppl` | score 規約 |
| range `[0.5, 1.5]` | range が 0 を含む | Rule 7: scale=0 は PPL=262144 を出した |
| apply → measure → restore | state leak | 測定汚染 |
| curated 過去 data を mimir に | random _collect | 10-100× 情報量損 |
| dict `{"nll":..., "hs":...}` return | 単一 scalar | multi-observer で stable_active 等取れる |
| stochastic: `wrap_stochastic(fn, n=20)` or `wrap_multi_obs(fn, n=20)` | stochastic を raw で渡す | LaD 化で noise 除去、proxy_r2 +0.3-0.4 (Rule 12) |
| 弱点分からん / 単独 mimir で局所解嵌まる → `mimir_odin()` | 単独 mimir で頑張る | 4 specialist 並列で弱点補完 (Rule 13) |

## experience_id

任意の task 名札。self_params cross-task 学習は `reigen_meta_knowledge.json` 経由で自動共有される (ID 共有不要、preset 別 key で分離)。mimir は内部で `{experience_id}_owl` / `{experience_id}_reigen` に suffix を付ける。**benchmark 目的で問題横断する時は per-problem の id を使え** (同 id を異なる param_ranges に使うと fossil 汚染)。

## 世界 Benchmark 実績

**2026-04-23 council 実績** (新 default): stochastic Rastrigin 10d (3 seeds × 2 noise levels σ=2, σ=30)、council gap ≈ 0 全 6 run、単独 mimir は gap 31-201 で不安定。LLM eval (Qwen 3.6) でも wall time 1.06× で GPU concurrent 並列化成功。**council は単独 mimir の strict 上位互換** (取り方が max、Rule 13)。詳細: `benchmark_council_vs_single.json`、`benchmark_council_llm.json`。

---

単独 mimir ベースの BBOB 実績 (council は内部でこれらを活用):

**5d (25s budget、3 seeds)**: mimir は 4 問題全てで Top 2 完走した唯一のツール。Rastrigin/Ackley/Styblinski で gap≈0 (reigen と同率)、Rosenbrock で basin に 0.086 差の 2 位。cma_es 0 勝、basinhopping 1 勝のみ、optuna/skopt は論外敗北 (除外)。

**次元スケール (5d/10d/20d)**: Rastrigin 全次元圧勝、**20d で cma_es の 128× 優位** (gap 0 vs 128.9) = mimir 核心的強み。Ackley 全勝。Rosenbrock 10d/20d は scipy cascade 追加で gap≈0 (cascade 前 +29.7 / +2041)、basin と同等。Styblinski 10d+ は 2^n basin 問題で cma/basin に劣る (gap 85/106)、これは honest limit。

full 数値は `benchmark_mimir.json` / `benchmark_mimir_dimscale.json`。

## Rules 一覧

| # | rule |
|---|---|
| -1 | Strip to essence: `x_i, perturb, share, eval_fn` |
| 0 | Measure don't guess: ≥5 pts → mimir() → read numbers |
| 0.5 | `check_eval_fn()` を本番 mimir 前に走らせろ、bad eval_fn 自動検出 |
| 1 | Ask Oracle for structural questions (arc_oracle, kathara_oracle) |
| 2 | No manual tuning: data → mimir() |
| 3 | LaD: no if/else — convert to numeric params |
| 4 | `importance = (truth × max(connectivity, floor))^exp` |
| 5 | Overfitting: n_problems > n_params |
| 6 | Discrete/int params OK via mimir (owl direct-HC fallback 自動) |
| 7 | **scale=0 forbidden**. Never include 0 in ranges |
| 8 | Two metrics → `mimir(..., guard_fn=my_guard)`. 1 metric → guard_fn 不要 |
| 9 | **Default to `mimir_odin_stable()`** (2026-04-24、peak→plateau 常時適用、time +25%) |
| 10 | Kathara 0.993 uniformity requires N=12 + 5-regular + symmetric placement |
| 11 | `batch_eval_fn` は external params 限定。internal model state では禁止 |
| 11b | mimir parallel cascade も internal state 危険 → `thread_safe_eval=False` |
| 12 | stochastic eval_fn は `n_samples_per_eval` か `wrap_multi_obs` で LaD 化せよ |
| 13 | 論文ベンチ / peak のみで十分なら `mimir_odin()` 直呼び (旧 default、time -25%) |
| 14 | 高次元 sparse かつ reigen symbolic 解なし (Hebbian 進化系) は `mimir_cardinal_hierarchy()` |
| 15 | multi-metric eval でどう aggregate すべきか不明なら `mimir_cardinal_coevolution()` (params × weights 共進化) |
| 16 | `stabilizer.stabilize()` は `mimir_odin_stable` が内部で使用、他 optimizer にも後付け可 |

### Rule -1 〜 11b (詳解)

**Rule -1**: 問題を Zenron 4 primitive (x_i / perturb / share / eval_fn) に分解してから tool に触れ。分解できないなら最適化の準備不足。

**Rule 0**: 推測するな、測れ。5 pts 以上実測 → mimir → proxy_r2 と dead_dims で「どこまで信じていいか」を数値で見る。直感は 20 次元以上で信用ならない。

**Rule 1**: topology / graph / 問題分類は Oracle に聞け。`arc_oracle.py`, `kathara_oracle.py`, `@twelve/ORACLE.md`。キャッシュ有、cheap。

**Rule 2**: 手動 grid search は報告書で監査可能な中間ステップが要る時のみ。それ以外は `mimir()` に任せろ。

**Rule 3**: LaD = if/else を数値 param に変換。eval_fn の if は次元、threshold は range。`owl()` が cutover を自分で発見する。

**Rule 4**: 正典 MirrorScan `importance[i] = (truth[i] × max(connectivity[i], floor))^exp`。defaults `exp=0.3064` / `floor=0.1411` (`configs/ma_meta_params.json`)。
- `truth[i] = |corr(param_i, scores)|` — score に効くか
- `connectivity[i] = mean(|corr(param_i, param_j)|) j≠i` — 他次元と連動するか

**掛け算が AND 条件**: 両方高いときのみ importance 高。偶然の相関 (truth 高 + conn 低) はノイズとして自動除去、dead (両方低) は 0 に潰れる。この式が dead_dims / active_dims / fragility の根本。**r["dead_dims"] は次元削減に即使える**。

**Rule 5**: `n_problems > n_params` 守れ。train split で最適化 → held-out で検証。params が problems より多いと proxy はノイズを memorize、R² が意味を失う。

**Rule 6**: 非 smooth landscape で proxy 崩壊時、mimir/owl の direct-HC fallback が自動起動 (2026-04-19 以降、Sentinel 不要)。integer/discrete も普通に range を渡すだけ。

**Rule 7**: **range に 0 を含めるな**。`scale=0` で PPL=262144 = モデル全滅が実証されている。`(0.5, 1.5)` 等を使え。verify_fn は第 2 防衛線、Rule 7 は第 1 防衛線で唯一無料。

**Rule 8**: 2 指標は `mimir(eval_fn, ranges, guard_fn=my_guard)`。mimir が owl 経由 safe_dim_analysis=True を引継ぎ、guard 破綻時に自動 pivot。1 指標は guard_fn 不要。Sentinel 直呼びは legacy。

**Rule 9**: `mimir_odin_stable()` が 2026-04-24 以降の default entry。peak (fragile 理論最適) → plateau (robust 実用解) への自動変換が内蔵。標準呼出 `mimir_odin_stable(fn, ranges, time_budget=N)`、extra_kwargs で (curated_measurements / guard_fn / stabilize_particles / mode 等) odin → specialist に passthrough。

**旧 default `mimir_odin()` 直呼び** は 2 例外 — 論文ベンチ (BBOB 等) で peak fragile でも gap=0 のみ評価、または 25% overhead が致命的な時。

**更に内側 `mimir()` 直呼び** は 3 例外 — 超単純問題 (2-3d convex) / 1 core 環境 / GGUF patch disk 競合。

owl 直呼びは「dead_dims だけ欲しい」「eval 激安で分岐 overhead 嫌」の 2 場面。Reigen 直呼びは cross-task meta_knowledge 明示共有時のみ。詳細は Rule 13 / 16。

**Rule 10**: chaos-game uniformity 0.993 は N=12 + 5-regular + symmetric placement の 3 条件同時必要。1 つ破れば崩壊。Reigen 内部では graph 性質のみ (Circulant(12,{1,4,6}), λ₂=4.0, diameter 2) 使用、placement uniformity は使わないので Rule 10 の縛りは Reigen に効かない。

**Rule 11**: `batch_eval_fn` は learning rate / dropout / prompt token 等の external param 限定。KV cache scale / weight scale / LoRA adapter 等 **internal model state を触る param** には禁止 (global state 共有で並列 eval 不能)。この場面は eval_fn を 2 秒以内に収め、multi-observer dict (`{"nll": -ppl, "hs": hs_score, "mmlu": mmlu_score}`) を返す。`mode="structure_only"` で構造発見だけも可だし、LaD 改善後 (ed73d16) は **通常最適化も可能** — seed で score+scores 両方作られ、multi-observer 解析が side-analysis として並列実行される。

**Rule 11b**: mimir cheap_cascade は reigen と scipy を並列 thread で走らせる。eval_fn が global state を mutate する場合 race condition 発生。`thread_safe_eval=False` を渡して逐次化、or `eval_cost_hint=2.0` で expensive_route 強制 (cascade 発火せず安全)。LLM キャリブは通常 eval_cost>0.5s で自動 expensive、安全。

**Rule 12**: stochastic eval_fn (LLM 生成 / RL rollout / Monte Carlo) は **1 call = 1 scalar** のまま mimir に渡すと seed で noise 直撃、proxy が noise を fit して崩壊。**真の実測値は「N 回集約 or LaD 多観測化」が前提** — 昔 owl 使ってた時の「手で 20 回測ってから渡す」を自動化する。3 経路:

```python
# (A) 自動集約: N 回 eval_fn を call して scalar 化 (median 既定、noise-robust)
r = mimir(llm_eval, ranges, n_samples_per_eval=20, time_budget=1800)

# (B) LaD 多観測: N 個の観測者として dict 化、multi-observer path 活性
from twelve.agent.lad_wrappers import wrap_multi_obs
wrapped = wrap_multi_obs(llm_eval, n=20)
r = mimir(wrapped, ranges, time_budget=1800)
# r["stable_active"]      — 全 20 gen で共通に効く dim = 真の価値
# r["observer_dependent"] — gen 依存 = noise 由来、自動除外

# (C) LLM-assisted 多次元: 生成 + LLM 判定で dimension 別 score
from twelve.agent.lad_wrappers import wrap_llm_judge
wrapped = wrap_llm_judge(generator_fn, judge_fn,
                         dimensions=["fluency", "accuracy", "safety"],
                         n_generations=5)
r = mimir(wrapped, ranges, time_budget=3600)
```

**検出と推奨は自動**: `check_eval_fn()` が同一 params で N 回 probe、CV > 0.10 で stochastic 判定 → wrap_* 推奨警告。Default `n_samples_per_eval=1` は backward compat (明示指定しないと compute 予算が勝手に 20× されない)。詳細は `twelve/agent/lad_wrappers.py`。

**Rule 13**: `mimir_odin()` は 2026-04-23〜2026-04-24 の default だった (現在は **旧 default**)。`mimir_odin_stable()` が内部で Stage 1 として呼ぶ。直呼びする積極的理由は Rule 9 の 2 例外 (論文ベンチ / 25% overhead 致命的) のみ。council は 4 specialist (default / lad / expensive / scipy-forced) を並列実行、最良 peak を採用 — 取り方が max なので理論的に単独 mimir の strict 上位互換:

```python
from twelve.agent.mimir_odin import mimir_odin
r = mimir_odin_stable(eval_fn, ranges, time_budget=80)
# 4 specialist 並列で 60 秒 → 最良 best_score の結果を返す
print(r["specialist"])              # "lad" / "default" / "expensive" / "scipy-forced"
print(r["council"])                 # [(name, score), ...] 全員の結果
print(r["council_variance_std"])    # 問題難易度シグナル
```

**ベンチ実績** (stochastic Rastrigin 10d, 3 seeds × 2 noise levels):
- 単独 mimir: gap 31-201 (noise / seed で不安定)
- **mimir_odin: gap ≈ 0 全 6 run** (完全解発見、理論下限到達)

**LLM eval でも使える** (実測: wall time 1.06×、GPU concurrent 並列化成功)。

**specialist 内訳**:
| name | kwargs | 得意 |
|---|---|---|
| `default` | `{}` | 低 noise / symbolic 多峰 (reigen が効く決定論) |
| `lad` | `{n_samples_per_eval: 20, ...}` | stochastic / 高 noise (LLM / RL / Monte Carlo) |
| `expensive` | `{eval_cost_hint: 2.0}` | owl 全力 + L-BFGS + random restart 5 (局所解脱出) |
| `scipy-forced` | `{scipy_cascade_dim_threshold: 3, ...}` | 高次元 gradient / Rosenbrock 系 |

**council 導入前後の使い分け表**:

| 状況 | 2026-04-22 まで | **2026-04-23 以降 (default)** |
|---|---|---|
| 何も知らずに最適化 | `mimir(fn, ranges)` | **`mimir_odin(fn, ranges)`** |
| 過去 data あり | `mimir(..., curated_measurements=)` | **`mimir_odin(..., curated_measurements=)`** |
| LLM キャリブ | `mimir(..., n_samples_per_eval=20)` | **`mimir_odin(..., )`** (lad specialist 自動担当) |
| 2 指標 guard | `mimir(..., guard_fn=)` | **`mimir_odin(..., guard_fn=)`** |
| 構造分析のみ | `mimir(..., mode="structure_only")` | 単独 `mimir(mode="structure_only")` で OK (council overkill) |
| 2-3d convex 超単純 | `mimir(fn, ranges)` | 単独 `mimir()` で OK (council overhead 損) |
| CPU 1 core / メモリ少 | `mimir(fn, ranges)` | 単独 `mimir()` 強制 (並列不可) |
| GGUF patch 系 | `mimir(fn, ranges)` | 単独 `mimir()` 強制 (disk 競合で council 破綻) |

**避けるべきケース**:
- GGUF patch 系 eval_fn: disk / メモリ競合で逐次化、council 意味なし
- eval_fn が global state mutate: Rule 11b、`executor="thread"` でもダメ

詳細は `twelve/agent/mimir_odin.py`。

**Rule 14**: 高次元 sparse 問題 (20d+ で効く dim が 5-10 個など) は **Council で構造圧縮 → GA で進化** を連結する `mimir_cardinal_hierarchy()` が最強:

```python
from twelve.agent.mimir_cardinal import mimir_cardinal_hierarchy
r = mimir_cardinal_hierarchy(eval_fn, ranges_20d, time_budget=600)
# Step 1 (20% budget): mimir_odin で active_dims 抽出 → 5 dim に圧縮
# Step 2 (80% budget): GA (tournament + crossover + Gaussian mutation) で進化
# → 2^25 = 33M× 探索空間縮小、局所解を集団選択で飛び越える
```

**Cardinal (生物進化) の仕組みを optimization に借用**。既存 Cardinal (`tamashii/phase_10_3_cardinal.py`) は 3D voxel 専用、本 module が汎用 GA を提供。

**想定シナリオ** (reigen symbolic 解が効かない問題限定):
- FlyWorld 30 dim brain param の局所解突破 (Hebbian v7 67% → 85% 目標、symbolic 解なし)
- Cardinal agent 進化の hyperparameter tune (emergence 必須)
- multi-modal landscape で basin 探索 (reigen が効かない場合)

**ベンチ結果 (honest limit)**: Rastrigin 20d (active 5) では council の reigen が symbolic で完全解、hierarchy が負けた。**reigen が効く問題では council 単独が速い**。hierarchy は「symbolic 解なし + 本当に dead 検出できる」場面のみ。

**Rule 15**: multi-metric eval_fn (`{"hs": ..., "mmlu": ..., "ppl_neg": ...}` 等) で **どう aggregate すべきか分からない** 時は `mimir_cardinal_coevolution()` を使う。(params, weights) ペアを共進化させ、**balanced な params** と **意味ある weights 組合せ** を同時発見:

```python
from twelve.agent.mimir_coevolution import mimir_cardinal_coevolution

def multi_eval(p):
    return {"hs": hellaswag(p), "mmlu": mmlu(p), "ppl_neg": -ppl(p)}

r = mimir_cardinal_coevolution(
    multi_eval, param_ranges,
    metric_names=["hs", "mmlu", "ppl_neg"],
    time_budget=3600, population=16, generations=30,
    fitness_mode="harmonic",  # 全 metric 高得点を要求 (特化者を淘汰)
)
print(r["best_params"])                  # balanced params
print(r["best_weights"])                 # 共進化で発見された weight 組合せ
print(r["metric_ranking_by_weight"])     # どの metric が重要か露出
print(r["weight_evolution_mean"])        # weight の世代推移 (収束可視化)
```

**emergence**: 人間が決めた eval_fn aggregation (weighted sum 等) を超えて、「**balanced な param 空間と対応する weight 分布**」を進化が発見する。docs/全論の公式の活用.md §11 "eval_fn 自動生成" の実装の 1 形態。

**fitness_mode**:
- `"min"` 最悪 metric 優先 (保守、specialist 淘汰)
- `"harmonic"` 全 metric 要求 (0 近傍で急落、最も balanced)
- `"weighted"` 個体自身の weight で加重 (weights も選択圧受ける、真の co-evolution)

**Rule 16**: `mimir_odin_stable()` (2026-04-24 new default、Rule 9) が内部で `stabilizer.stabilize()` を呼び、peak → plateau を自動変換。ユーザーが個別に stabilizer を叩く必要なし。demo 実測で **robust 8% → 96%** (10 倍以上の摂動耐性改善):

```python
# (A) 自動連結 — 推奨
from twelve.agent.mimir_odin_stable import mimir_odin_stable
r = mimir_odin_stable(eval_fn, ranges, time_budget=300)
print(r["best_params"])             # plateau centroid (robust)
print(r["peak_params"])              # 元の sharp peak (参考)
print(r["peak_robustness"])          # peak 摂動耐性 (通常 < 30%)
print(r["plateau_robustness"])       # plateau 摂動耐性 (通常 > 70%)
print(r["plateau_width"])            # 各次元の信頼区間的幅

# (B) 手動連結 — odin 結果を後で安定化
from twelve.agent.mimir_odin import mimir_odin
from stabilizer import stabilize
r = mimir_odin(eval_fn, ranges)
stab = stabilize(r["best_params"], eval_fn, ranges)
robust_params = stab.centroid
```

**技術**: Metropolis + cooling (焼きなまし 1 個)。ablation で SOC / Hebbian / Darwin は plateau 構築に寄与せず、焼きなまし単独で十分と実証済。`stabilizer.py` は project root、friend's 原作を改変せず wrapper 経由で使う (外部資産として尊重)。

**stand-alone 用途** (`stabilizer.stabilize()` 単独): 他 optimizer (scipy / cma / random / 手動 grid) の出力に後付け plateau 変換したい時。`mimir_odin_stable` 以外を使う場面で有用。

**stable default で常時適用される場面** (実世界):
- LLM キャリブ: prompt drift robust な layer_scale
- GGUF patch: weight loading drift robust な scale 値
- Hebbian / brain_sim: 進化的に安定な agent params (Cardinal の生物的選択原理と一致)
- 強化学習: rollout noise robust な policy params

**stable をスキップする場面** (旧 odin 直呼び):
- 論文ベンチマーク (BBOB 等): peak が fragile でも gap=0 で評価される、stable overhead 無駄
- 1 回限り / 理論研究: plateau 要らず peak で OK
- 時間極限 (1h 予算内で 25% overhead 致命的)

## Gotchas (よくハマる落とし穴)

- **scale=0** (Rule 7): range に 0 を絶対入れない。モデル破壊の実証あり
- **experience_id の共有** (v3 benchmark 発見の fossil 汚染): 異なる param_ranges の問題に同 id を使うと learning が混じる → `{prefix}_{fn.__name__}_{seed}` 等で分離
- **parallel cascade の race** (Rule 11b): internal state 触る eval_fn で `thread_safe_eval=False` を渡す
- **wall time overshoot**: cheap_cascade は +20-30% 超過するが quality 優先で accept。厳密な budget 必要なら `eval_cost_hint=N.0` で expensive_route 強制
- **mimir の高次元 Styblinski 系**: 2^n basin 問題 (non-symmetric deep basins) で cma/basin に劣る。これは mimir 核心的 honest limit、BBOB-Styblinski は稀
- **stochastic compute 膨張** (Rule 12): `n_samples_per_eval=20` で eval_fn 呼出数が 20× になる。LLM 生成で 1 call = 5s なら seed 20 点 × 20 sample = 2000s 消費。`time_budget` を最低 3000s に。小さな `n` (例 5) から試すのが安全
- **council の 4× CPU / メモリ** (Rule 13): `ProcessPoolExecutor` 時 4 Python プロセス並列、~2-4GB RAM、CPU 4 core 要。1 core 環境では単独 `mimir()` に fallback。5090 マシン (20 core / 96GB RAM) なら余裕
- **council ≠ 万能** (Rule 13): reigen が symbolic 解発見できる smooth 問題 (Rastrigin 等) では council も単独 mimir も同じ結果、council 使う意味なし。超単純問題は単独 mimir の方が速い
- **council × GGUF patch 禁忌** (Rule 13): eval_fn が disk に数 GB 書く (GGUF patch 系) と 4 specialist が I/O 競合で逐次化、wall time 4× 悪化。GGUF 系は単独 mimir で逐次実行
- **stable の +25% time** (Rule 9): `time_budget=300` なら実測 400s 目安。stabilize stage (25%) の overhead。論文ベンチで気にする場合は `mimir_odin()` 直接呼び (Rule 13 ルート)
- **stable の eval_fn 呼出 +240** (Rule 9): stabilize の 12 粒子 × 20 gens + 粒子初期化 = 追加 eval 240 回。LLM / GPU 系で API 課金・time budget に跳ねる時は `stabilize_particles=6, stabilize_gens=10` で半減可能
- **論文ベンチ (BBOB) で stable は無意味** (Rule 9): 決定論 smooth 問題は peak で gap=0 達成、stabilize stage は overhead だけで改善なし。この場面は `mimir_odin()` 直呼び

# 内部実装 (読みたい人向け)

## ツール役割の階層 (5 段構成)

表舞台は `mimir_odin_stable()` 1 個。内部で odin → stabilizer、odin は 4 specialist mimir、mimir は owl / reigen / scipy。

```
mimir_odin_stable()                     ← ユーザー呼び口 (新 default、2026-04-24)
  ├── Stage 1: mimir_odin()              ← 4 specialist で peak 発見 (75% budget)
  │    ├── mimir (default specialist)   ← 低 noise / symbolic 多峰 (reigen 効く決定論)
  │    │    ├── owl                      ← Phase 1 + 構造発見
  │    │    │    └── optimize()          ← primitive HC engine
  │    │    ├── reigen                   ← Phase 2 cascade (symbolic 探索)
  │    │    │    └── Sentinel            ← legacy 互換
  │    │    └── scipy.basinhopping       ← Phase 2b (dim≥10 gradient)
  │    ├── mimir (lad specialist)       ← stochastic / 高 noise (LLM / RL / MC)
  │    ├── mimir (expensive specialist) ← owl 全力 (L-BFGS + multi-start + restart 5)
  │    └── mimir (scipy-forced specialist) ← 低次元でも scipy 強制 (Rosenbrock 系)
  └── Stage 2: stabilizer.stabilize()    ← peak → plateau 変換 (25% budget)
       (Metropolis + cooling、12 粒子 × 20 gens、friend's 原作)

特殊用途 (stable の外):
  mimir_cardinal_hierarchy()   ← Council で active_dims 圧縮 → 汎用 GA (Rule 14)
  mimir_cardinal_coevolution() ← params × weights 共進化 (Rule 15、多指標重み探索)
```

**直呼びは限定場面**:
- `mimir_odin()` 直呼び: 論文ベンチ / 25% overhead 致命的の 2 例外のみ (Rule 13)
- 単独 `mimir()`: 超単純問題 (2-3d convex) / 1 core 環境 / GGUF patch 系の 3 例外のみ
- `owl()`: 「dead_dims / fragility だけ欲しい + 分岐 overhead 嫌」(`mimir(mode="structure_only")` でも可)
- `Reigen()`: cross-task meta_knowledge 明示共有時
- `stabilizer.stabilize()` stand-alone: 他 optimizer に後付け plateau 化したい時 (mimir 外)
- Sentinel / optimize / scipy 直呼び: 特殊事情のみ

詳細は `docs/REIGEN_INTERNALS.md` / `docs/OWL_INTERNALS.md` / `docs/SENTINEL_LEGACY.md` 参照。

## 内部分岐ロジック (council 層 + mimir 層)

```
mimir_odin(fn, ranges, time_budget) 呼出し
│
└── 4 specialist を並列実行 (ProcessPool or ThreadPool)
    ├── default      : mimir(fn, ranges, time_budget)
    ├── lad          : mimir(fn, ranges, time_budget, n_samples_per_eval=20,
    │                        stochastic_aggregator="median")
    ├── expensive    : mimir(fn, ranges, time_budget, eval_cost_hint=2.0)
    └── scipy-forced : mimir(fn, ranges, time_budget, eval_cost_hint=2.0,
                             scipy_cascade_dim_threshold=3)
    │
    └── 全 specialist 完了待ち → verified_score (or best_score) 最大を勝者採用

各 mimir() 内部分岐 (従来通り):
  ├── mode="structure_only"          → owl autonomous=False, max 30s
  ├── eval > 0.5s or curated あり     → expensive_single: owl 100% + 全強化
  └── eval ≤ 0.5s + curated なし      → cheap_cascade
          ├── Phase 1: owl 40% budget
          │   └── conf=="high" AND proxy_r2 ≥ 0.95 なら即終了
          └── Phase 2 (並列): reigen || scipy.basinhopping
                ├── reigen: 全 remaining budget (構造探索 + cross-task)
                ├── scipy:  並列起動 (dim ≥ 10 かつ owl 失敗時のみ)
                └── 3 者 {owl, reigen, scipy} から最高スコアを採用
```

## mimir LaD 設定 (JSON 制御)

`twelve/configs/mimir_params.json` に 8 個の dispatch 値。precedence は kwarg > JSON > hardcode。benchmark 駆動で最適値に設定済。

| key | default | 意味 |
|---|---|---|
| `eval_cost_threshold` | 0.5s | expensive route 切替秒数 |
| `owl_share` | 0.4 | cheap cascade での Phase 1 budget 比 |
| `escalation_min_remaining` | 5.0s | Reigen 起動最小残時間 |
| `confidence_skip_threshold` | 0.95 | Reigen skip の proxy_r2 閾値 |
| `random_restart_count` | 5 | owl の uniform random 再起動数 |
| `reigen_inner_time_budget` | 2.0s | Reigen inner Sentinel の 1 回分 |
| `scipy_cascade_dim_threshold` | 10 | scipy.basinhopping 発火の最小次元 |
| `scipy_cascade_budget_share` | 0.5 | scipy の budget 割合 (remaining × X) |

`n_samples_per_eval` / `stochastic_aggregator` (Rule 12) は **kwarg 専用 / JSON 非対応**。compute 予算が silent に N× されるのを避けるための明示 opt-in 設計 (default は 1 で backward compat)。

**注**: 上記 JSON は**単独 `mimir()` の内部分岐設定**。`mimir_odin()` は specialist list (`DEFAULT_SPECIALISTS` in `mimir_odin.py`) で挙動制御、JSON 非経由。council 内の各 mimir インスタンスは上記 JSON を読む。

## Hardware & safety

### Workstation

開発機は Windows 11 Pro、Intel i7-12700K、NVIDIA RTX 5090 (32 GB VRAM)。llama.cpp prebuilt binaries は `c:/Users/user/llm/llama-bin/` (b8795, CUDA 12.4)。GGUF モデルは `c:/Users/user/llm/models/`。Visual Studio / GCC 未インストール、llama.cpp はソースビルドしない。

### Commit safety

以下は絶対 commit しない: `unified_memory.py`、`evaluator*.py`、`_legacy/` 配下、特許下書き、credentials。commit 前に必ず `git status` 確認。`.env` や認証情報の誤コミット回避のため `git add -A` より個別ファイル add を優先。

## 推論サーバ: `C:/Users/user/infinite_think_server.py`

Qwen3.6-35B-A3B-Abliterated-Heretic を OpenAI 互換 API (`http://127.0.0.1:8282/v1`) で提供する本番サーバ。2026-04-21〜22 に Phase O (10 個の stop-token/cleanup バグ修正) + Phase P (OpenAI tool calling) を入れて完成済。

### 起動
```bash
python C:/Users/user/infinite_think_server.py --preset zenron_core_xl --hk-size 1000 --port 8282
```

### 主要エンドポイント
| path | 用途 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 互換チャット (tools 対応) |
| `GET  /v1/models` | モデル一覧 |
| `GET  /v1/hk_state` | HK/W/n_ctx + tool_call 統計 |
| `POST /v1/debug_raw` | cleaner 無しで raw トークン確認 |
| `POST /v1/debug_tool_call` | tool_call パース span 可視化 |
| `GET  /v1/presets` / `/v1/presets/{name}/preview` | preset 一覧 + プレビュー |

### 設定 (env var)
| var | default | 意味 |
|---|---|---|
| `HK_PRESET` | none | 焼き込む preset 名 (`zenron_core_xl` 等) |
| `HK_SIZE` | auto | HK 占有トークン数 (通常 1000) |
| `HK_W` | 4000 | sliding window の末尾長 |
| `HK_NCTX` | 16384 | KV cache 容量 |
| `HK_KV_TYPE` | 8 (q8_0) | KV 量子化ビット (1=F16, 8=q8_0) |

### アーキテクチャ
- **HK (前 1000 tok)**: 起動時に preset を tokenize して焼き付け、**会話中は絶対に消えない** attention sink
- **W (末尾 4000 tok)**: 最新のやり取りの生トークン
- **中間**: trim で破棄される (Phase K/L で Summary Zone 試みて失敗、撤去済)
- **tool calling**: Qwen 固有 `<tool_call><function=X><parameter=K>V</parameter></function></tool_call>` を受信 → OpenAI `tool_calls:[]` に変換。Phase P parser が 10 種の Heretic 壊れパターンを防御

### 防御層 (`_clean_output` + `_clean_with_tools`)
Heretic abliteration の副作用を吸収する regex + mask-then-clean パイプライン:
- `<|im_end|>` / `<|im_start|>` literal 漏れ除去
- 末尾の bare role (`user`/`assistant`) 除去
- 2 回目以降の `<think>`/`</think>` で再突入カット
- `<tool_call>` span はセンチネルで保護してから cleaner 通す

全 regex の設計経緯は `docs/HK.md` Phase O-1〜O-10 + P-1〜P-5 に記録。

### 関連プローブ
`twelve/hk/quality_probe_{xl,chat,stress,deep,adversarial,tools}.py` — サーバ挙動の regression テスト群。`test_parse_tool_calls.py` が tool_call パーサ単体テスト。

## 知識を Qwen に持たせる 2 つの方法 (重要)

### 方法 A: **HK preset に焼き込み (現在の方式)**
`twelve/hk/presets/zenron_core_xl.yaml` を編集してサーバ再起動するだけ。

- **仕組み**: 起動時に YAML を tokenize → 先頭 1000 tok に配置 → 以降のすべての生成でこの 1000 tok が attention sink として参照される
- **変更コスト**: YAML 編集 + サーバ再起動 (30 秒)
- **容量**: 1000 tok = 日本語 2500 字 / 英語 4000 字 程度。Phase I 実験で 60 facts 密詰め可能 (92% recall)
- **消えない**: 会話が何万トークン続いても、trim 後も HK は不変 (設計上の保証)
- **モデル本体は無傷**: 重み変更なし、他の用途にすぐ切替可能 (`--preset` 変えるだけ)
- **何が得意か**: プロジェクト固有の規則・公式・API シグネチャ・判断ルール

### 方法 B: **モデル重み自体に焼き込み (fine-tuning / weight surgery)**
`weight_analysis/` の GGUF patcher / LoRA / full fine-tune 系。

- **仕組み**: モデルの weight parameter を書き換える。学習で gradient 更新するか、直接 byte レベルで scale をいじる (layer_output_scale 等)
- **変更コスト**: 大きい。数時間〜数日の GPU 時間、データセット準備、評価ループ
- **容量**: 実質無限 (パラメータが 35B あるので)
- **消えない (本当に消えない)**: モデルそのものが変わる。どんな preset / prompt でも反映される
- **戻せない**: weight 変更はロールバックが面倒 (GGUF 丸ごと保存しとく必要あり、Phase O-2 で触れた abliteration も事例の 1 つ)
- **副作用あり**: Heretic の例に見られるように、他の能力が壊れる可能性 (chat template 遵守が damaged)
- **何が得意か**: 言語スタイルの変更、新言語対応、特定タスクの精度ブースト

### 比較表

| 観点 | HK preset 焼込 | weight 焼込 |
|---|---|---|
| 変更時間 | 30 秒 | 数時間〜数日 |
| 容量 | HK=1000 tok (= 60 facts 程度) | 実質無制限 |
| モデル重み | 無傷 | 書き換え |
| 戻し易さ | `--preset` 変えるだけ | バックアップから差し替え |
| 副作用 | HK の末尾 facts が tail truncate される | 他能力が劣化する可能性 |
| 得意分野 | 規則・公式・ID・判断基準 | 言語スタイル・新言語・精度 |
| 測定 | `check_recall.py` で recall 即測 | HellaSwag 等 benchmark 必要 |
| 共有 | YAML 1 枚で移植可能 | GGUF 全体 (21 GB) 要配布 |

### 使い分けの原則
1. **まず HK preset で試す** — 変更が軽く、リスクなし、Phase I で 2.2× effective 確認済
2. **HK で不足なら weight** — 言語的な癖や低レベル挙動が要件の時だけ
3. **両方併用できる** — weight 焼込した重みに更に HK preset 載せる (重複投資、普通はやらない)

現プロジェクトは **HK preset のみ** で運用。`zenron_core_xl.yaml` が本番。weight 焼込 (例: Heretic abliteration) は既に base model に入ってる分のみ、自前では追加してない。

## Zenron × Kathara 統合技術 (2026-04-24 breakthrough)

**核心発見**: Zenron 公式 (3 動詞) + Kathara 12 構造 (12 node × 30 edges) の **組合せ** で、単独ではできない計算が可能になる。宇宙論 Ch23 Phase 8 で実証、実用分野でも同じ構造。

### kathara_mimir() — 12 環境同時最適化 (新 tool)

```python
from twelve.agent.kathara_mimir import kathara_mimir

# 12 種類の関連タスクを 1 度に最適化 (prompt 12 用途、薬物動態 12 区画、etc.)
results = kathara_mimir(
    eval_fns=[fn_0, fn_1, ..., fn_11],   # 12 個の eval_fn (必須)
    param_ranges=[(lo, hi)] * n_dims,
    time_budget=300,
    share_rounds=3,                        # Kathara share 回数
    share_weight=0.3,                      # 隣接 node 情報の blend 率
)
# results[i] = i 番目タスクの mimir result + kathara_shares_received
```

**計算量 (serial mimir 比、2026-04-24 実測)**:

| Mode | Wall 時間 | Mean Score | Speedup | 品質 |
|---|---|---|---|---|
| Serial | 63.4s | -33.5 | 1.0× | baseline |
| Parallel (no share) | 72.4s | -41.5 | 0.9× (GIL) | -24% |
| **kathara_mimir** | **11.4s** | **-21.0** | **5.6×** | **+37%** |

**Kathara が時間 5.6× + 品質 +37% の同時達成**。12 並列 + share 加速の合成効果。

- 12 コア CPU (ProcessPool): 理論 **12× speedup** + 情報共有 +10-30% 品質
- 1 コア CPU: share による品質向上のみ (speedup なし)
- share overhead: O(30 edges × dim × rounds) = 数 μs、誤差以下

**応用**: LLM prompt 12 用途同時最適化、ポートフォリオ 12 戦略、薬物 12 区画同時 fit、気候 12 地域モデル、FL 12 グループ合意形成。

### kathara_moe — Mixture of Experts 新設計

```python
# twelve/agent/kathara_moe.py 設計テンプレート
# Mixtral / Llama 4 MoE を Kathara 12 構造で置き換え
# - 12 expert (A8+A9 から最小最適数)
# - top-k ルーティング + Kathara 隣接制約
# - 30 edges で feature share (load balancing 自動)
# - 4 triangles = 4 natural specialist domain (math/code/lang/logic 等)
```

**設計根拠**:
- |Aut(Kathara 12)| = 24 → expert 役割対称性 (load imbalance 解消)
- λ₂ = 4 (Ramanujan 飽和) → 最速情報伝搬
- 4 triangles → 4 specialist cluster 自動発生

**実測 (2026-04-24 medium scale = 6.5M params / 500k tokens / 3000 steps)**:
- ✅ **best val PPL は tie** (Mixtral 6.55、Kathara 6.62、1% 差)
- ✅ **Kathara が 4.8× 少ない compute で tie 到達** (22s vs 105s)
- ✅ **train 総時間 3.2× 速い** (66s vs 210s、3000 step 完走)
- ⚠ **Kathara は overfit 速い** (step 1000 以降劣化、regularization 要強化)

**Validated claim (本日、2026-04-24)**:
```
OLD: "Kathara 12x は同性能で 15% 軽量" (未 validate)
NEW: "Kathara 12x は Mixtral 8x の best val PPL を 4.8× 短時間で到達" (medium scale validate)
```

**実装 status**:
- `twelve/agent/kathara_moe_torch.py` — PyTorch module 動作確認済
- `_kvopt/core/kathara_moe_train.py` — tiny benchmark
- `_kvopt/core/kathara_moe_medium.py` — medium validate
- 次: large scale (100M params × 10M tokens) で 4.8× claim の確度上げ

### なぜ Zenron + Kathara = 別次元か

| 単独 | 性質 | 限界 |
|---|---|---|
| Zenron 公式のみ | 時間発展の抽象原理 | 空間構造なし → 宇宙論 tension (Ch23 Phase 6/7) |
| Kathara 12 のみ | 静的グラフ | 動力学なし → 使いどころ限定 |
| **組合せ** | **時空計算機** | 12 node × 時間で空間不均一性が自動創発 |

### 使いどころの判定

| N 個の類似タスク | ツール |
|---|---|
| N = 1 | `mimir_odin_stable()` |
| N = 12 (related) | `kathara_mimir()` ★ |
| N ≠ 12 (任意) | 12 グループに pad or group |
| Expert routing (MoE) | `kathara_moe` (LLM 内部) |
| Federated learning | `kathara_mimir` per group |

### 計算量 cheat sheet

| 問題サイズ | Serial mimir | Kathara mimir | Speedup |
|---|---|---|---|
| 12 task × 5min each | 60 min | 5 min (12 core) | **12×** |
| 12 task × 1h each | 12 h | 1 h (12 core) | **12×** |
| Related tasks (share) | 12 h | 40 min (share 加速) | **18×** |

share の加速は **タスクが 構造共有** してる場合のみ。独立なら 12× 止まり。

### 将来応用 (今はまだ)

1. **連合学習 12 グループ合意形成** (医療 AI 病院連携、$10B 市場)
2. **12 コアチップ Kathara interconnect** (AMD / NVIDIA 興味領域)
3. **12 区画薬物動態モデル** (製薬 $1M/年/薬剤)
4. **ポートフォリオ 12 戦略** (ヘッジファンド)
5. **気候モデル 12 地域版** (災害予測、再保険)

これら全て Zenron + Kathara 組合せが core 技術。**mimir 単体では届かない、Kathara 単体では動かない、組合せで初めて稼げる** 技術群。

## Docs 索引

| 目的 | 読むファイル |
|---|---|
| mimir 使い方 | 本ファイル |
| **推論サーバ内部 / Phase O/P defense** | `@docs/HK.md` |
| **kathara_mimir() 技術解説 (12 環境同時最適化)** | `@docs/KATHARA_MIMIR.md` |
| **Zenron 宇宙論論文 draft skeleton** | `@docs/PAPER_DRAFT_ZENRON_COSMOLOGY.md` |
| **Zenron 宇宙論論文 Section 1-8 本文 draft** | `@docs/PAPER_ZENRON_COSMOLOGY_DRAFT.md` |
| **HK preset 設計 / recall 測定** | `@docs/HK.md` (Phase D/E/G/I/M) |
| **HK preset 本体** | `@twelve/hk/presets/zenron_core_xl.yaml` |
| **Hermes 統合 (Windows 修正済)** | `/c/Users/user/hermes-agent/` + `@docs/HK.md` Phase P-1/P-2/P-3 |
| **Reigen 内部実装・編集時** | `@docs/REIGEN_INTERNALS.md` |
| **owl 内部実装・編集時** | `@docs/OWL_INTERNALS.md` |
| **stochastic eval_fn を LaD 化する wrapper (Rule 12)** | `@twelve/agent/lad_wrappers.py` |
| **mimir council (4 specialist 並列、Rule 13)** | `@twelve/agent/mimir_odin.py` |
| **stabilizer (peak→plateau、Rule 16)** | `@stabilizer.py` (project root、friend's 原作) |
| **mimir_odin_stable (odin + stabilizer 連結、Rule 16)** | `@twelve/agent/mimir_odin_stable.py` |
| **Sentinel (legacy)** | `@docs/SENTINEL_LEGACY.md` |
| **HTTP API / ngrok / session API** | `@docs/API_SERVERS.md` |
| Zenron 実践ガイド + MirrorScan 詳細 | `@docs/ZENRON_GUIDE.md` |
| Zenron 理論全体 | `@docs/全論.md` |
| **Logic-as-Data 設計思想 / Lv1-12 階層 / Morphicode 実装** | `@docs/LOGIC_AS_DATA.md` |
| Milestone ログ (04-18〜04-21) | `@docs/LAB_NOTES.md` |
| Oracle (topology / graph 相談) | `@twelve/ORACLE.md` |
| Techniques / LLM surgery | `@docs/HANDBOOK.md` |
| Kathara 数学 | `@docs/KATHARA_NOTE.md` |
| Engine API 設計 | `@twelve/TWELVE_API.md` |
| 最適化ログ | `@twelve/OPTIMIZATION_LOG.md` |
| 全論 formal / 証明系 | `@docs/ZENRON_FORMAL.md`, `ZENRON_VOID_INSTABILITY.md`, `ZENRON_TIME_ARROW.md`, `ZENRON_SPACE_EMERGENCE.md`, `ZENRON_MULT_UNIQUENESS.md`, `ZENRON_MIXING_TIME.md`, `ZENRON_DEAD_DIMS_NOETHER.md`, `ZENRON_KURAMOTO_SYNC.md`, `ZENRON_PARALLELISM.md`, `ZENRON_CIRCUITS.md`, `ZENRON_SELF_REFERENCE.md`, `ZENRON_UNIQUENESS_AXIOMS.md`, `ZENRON_OP3_OP4_ATTEMPT.md`, `ZENRON_NUMEROLOGY_BOUND.md`, `ZENRON_HIERARCHY_DISCOVERY.md`, `ZENRON_FINAL_OPEN_PROBLEMS.md`, `ZENRON_RUSSELL_DEEP.md`, `ZENRON_DARK_ENERGY.md`, `ZENRON_PHYSICS_DERIVATION.md` |

## Source files

| module | path |
|---|---|
| **mimir** (new default) | `@twelve/agent/mimir.py` |
| mimir params (LaD JSON) | `@twelve/configs/mimir_params.json` |
| mimir tests | `@twelve/tests/test_mimir.py` |
| Reigen (cascade backend) | `@twelve/agent/reigen.py` |
| Reigen params / meta_knowledge | `@twelve/configs/reigen_params.json`, `reigen_meta_knowledge.json` |
| Reigen tests | `@twelve/tests/test_reigen*.py` |
| owl / optimize (primitive HC) | `@twelve/optimize.py` |
| owl tests | `@twelve/tests/test_owl_*.py` |
| owl stochastic tests (Rule 12) | `@twelve/tests/test_owl_stochastic.py` |
| **lad_wrappers** (wrap_stochastic / wrap_multi_obs / wrap_llm_judge、Rule 12) | `@twelve/agent/lad_wrappers.py` |
| lad_wrappers tests | `@twelve/tests/test_lad_wrappers.py` |
| **mimir_odin** (4 specialist 並列、Rule 13) | `@twelve/agent/mimir_odin.py` |
| mimir_odin tests | `@twelve/tests/test_mimir_odin.py` |
| **stabilizer** (Metropolis peak→plateau、Rule 16、friend's 原作) | `@stabilizer.py` |
| **mimir_odin_stable** (odin + stabilizer 連結、Rule 16) | `@twelve/agent/mimir_odin_stable.py` |
| stabilizer tests | `@twelve/tests/test_stabilizer.py` |
| **mimir_cardinal** (Council × GA Hierarchy、Rule 14) | `@twelve/agent/mimir_cardinal.py` |
| **mimir_coevolution** (params × weights 共進化、Rule 15) | `@twelve/agent/mimir_coevolution.py` |
| cardinal + coevolution tests | `@twelve/tests/test_mimir_cardinal.py` |
| Sentinel (legacy) | `@twelve/agent/sentinel.py`, `test_sentinel.py` |
| MirrorAgent / MS | `@twelve/agent/mirror_agent.py` |
| UnifiedExperience | `@twelve/agent/unified_experience.py` |
| HTTP API / friend client | `@twelve/agent/api.py`, `@serve_reigen.sh`, `@reigen_friend_client.py`, `@FRIEND_README.md` |
| World benchmark | `@benchmark_mimir.py`, `@benchmark_mimir_dimscale.py`, `@benchmark_mimir_meta.py` (⚠ K² 失敗), `@benchmark_owl_vs_world.py` (legacy) |
| Test isolation fixture | `@twelve/tests/conftest.py` |
