# CLAUDE.md — Operational Rules

All responses in Japanese.

## 最優先ルール: `mimir()` を呼べ

全ての最適化は `mimir(eval_fn, param_ranges, time_budget=N)` で始まる。2026-04-20 以降の公式エントリーポイント。

`mimir()` は内部で eval コストを測定、owl (精鋭) と Reigen (cascade) と scipy.basinhopping (高次元 gradient) を自動分岐する。ユーザーは tool 選択で悩まない。

BBOB 4 問題で gap≈0 達成。Rastrigin 20d で cma_es の 128× 優位。Rosenbrock 20d scipy cascade で gap=0。

## mimir 最小使用法

```python
from twelve.agent.mimir import mimir

r = mimir(eval_fn, param_ranges, time_budget=300)
print(r["best_params"], r["best_score"], r["tool_used"])
```

返り値 dict 主要キー: `best_params` / `best_score` / `tool_used` / `dead_dims` / `fragility` / `proxy_r2` / `route`。構造発見系 (dead_dims, fragility, proxy_r2) は最適化と同時に得られる。

## mimir 使用 5 パターン

第1に、何も知らない状態で最適化したい時。`mimir(fn, ranges, time_budget=300)` で終わる。

第2に、過去実験データがある時。`curated_measurements=past_data` を渡すと expensive_single route に切替、owl に直接データを食わせる。20 点の curated は random 200-2000 点相当の情報量を持つ。

第3に、LLM キャリブ等 eval が重い時。mimir は 1 call 実測で >0.5s を検知、自動で expensive_single へ。owl 全力モード (L-BFGS + multi-start + random-restart=5) が起動する。

第4に、分析だけしたい時。`mode="structure_only"` を指定する。最適化 skip、dead_dims / fragility / proxy_r2 だけを 15-30s で返す。

第5に、安全指標を守りたい時。`guard_fn=my_guard` を渡す。owl に safe_dim_analysis=True 経由で引き継がれ、2 指標 pivot が発動する。

```python
r1 = mimir(fn, [(-5, 5)] * 8, time_budget=300)                           # (1) 一般
r2 = mimir(fn, ranges, curated_measurements=past_data, time_budget=600)  # (2) 過去 data
r3 = mimir(ppl_eval, ranges, time_budget=1800)                           # (3) 自動 expensive
r4 = mimir(fn, ranges, mode="structure_only", time_budget=30)            # (4) 分析のみ
r5 = mimir(fn, ranges, guard_fn=my_guard, time_budget=300)               # (5) 2 指標
```

## mimir 内部分岐ロジック

```
mimir(fn, ranges, time_budget) 呼出し
│
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

## mimir の返り値

```
best_params, best_score, tool_used, route, eval_cost_s, confidence, elapsed_s
# 構造発見 (owl 由来)
dead_dims, active_dims, fragility, proxy_type, proxy_r2
verified_score (実測スコア), param_names, rounds_completed, recovered_dims, n_measurements
# raw 生 dict (詳細読み取り)
owl_result, reigen_result (escalation 時), scipy_result (dim≥10 時)
```

tool_used 値: `"owl"` / `"owl+reigen"` / `"owl+scipy"` / `"owl(reigen_tried)"` / `"owl(reigen_scipy_tried)"` / `"owl_structure_only"`
route 値: `"expensive_single"` / `"cheap_cascade"` / `"structure_only"`

## ツール役割の階層

表舞台は `mimir()` 1 個。裏方は owl (Phase 1 + 構造発見) と Reigen (Phase 2 cascade + cross-task) と scipy.basinhopping (高次元 gradient)。

```
mimir()                      ← ユーザー呼び口
  ├── owl()                  ← Phase 1、構造発見単体でも直呼び可
  │    └── optimize()        ← primitive HC engine
  ├── reigen()               ← Phase 2 cascade (並列)
  │    └── Sentinel          ← legacy 互換
  │         └── owl
  └── scipy.basinhopping     ← Phase 2b cascade (dim≥10、並列)
```

直呼びは限定場面。owl は「dead_dims / fragility だけ欲しい時」(`mimir(mode="structure_only")` でも可)、Reigen は「cross-task meta_knowledge 明示共有」時、Sentinel は legacy。optimize と scipy 直呼びは特殊事情のみ。

詳細は `docs/REIGEN_INTERNALS.md` / `docs/OWL_INTERNALS.md` / `docs/SENTINEL_LEGACY.md` 参照。

## 実測は自動化される — 手で 1 点ずつ測るな

`mimir(eval_fn, ranges, time_budget=300)` の **1 行で下記が全自動**:

- seed 20 点の param 選定 (uniform random)
- 各 param で eval_fn 呼出し、score 収集
- 実測点から proxy (近似式) を fit
- proxy argmax の新 param で再実測 (verify_fn)
- growing_data 拡張 → proxy 再 fit → 次の点で実測 ... のループ
- 収束判定・停滞検知・range 拡張

**ユーザーの仕事は `eval_fn` 書くだけ**。下記の手動ループは書くな:

```python
# ❌ 昔やってた手作業 (不要)
for p in preset_points:
    s = eval_fn(p)
    results.append((p, s))
# 近似書く / 最適点選ぶ / 再測る ... 全部手動

# ✅ 今はこれだけ
r = mimir(eval_fn, ranges, time_budget=300)
```

### 過去データは必ず `curated_measurements=` に渡せ

過去に手動で測った点があるなら**絶対に捨てるな**。mimir の seed にして精度を爆上げできる。

```python
past = [{"params": [0.8, 1.2], "score": 0.71},
        {"params": [1.0, 1.3], "score": 0.68}, ... ]   # 過去の実測 20 点
r = mimir(eval_fn, ranges, curated_measurements=past, time_budget=600)
```

### proxy_r2 の違い (同じ実測数でも桁違い)

| 始点 | 初期 proxy_r2 |
|---|---|
| empty start (mimir が random 20 点実測) | 0.5 〜 0.7 |
| curated 20 点 (ドメイン知識) | **0.80 〜 0.90** |
| curated 50 点 | **0.90 〜 0.95** |

curated = random の **10-100× 情報量**。ドメインエキスパートの選んだ 20 点は random 200-2000 点に相当する。

### 推測では始めない、常に実測で proxy 構築

mimir の proxy は**全て実測値から fit** される。推測・合成データは混ざらない。proxy_r2 が低い (< 0.3) 時は **direct-HC fallback** が自動発動して proxy を捨て、実 eval_fn で直接最適化に切り替わる。**推測で押し切ることは設計上ない**。

## eval_fn 設計

| good | bad | why |
|---|---|---|
| `return -ppl` (higher=better) | `return ppl` | score 規約 |
| range `[0.5, 1.5]` | range が 0 を含む | Rule 7: scale=0 は PPL=262144 を出した |
| apply → measure → restore | state leak | 測定汚染 |
| curated 過去 data を mimir に | random _collect | 10-100× 情報量損 |
| dict `{"nll":..., "hs":...}` return | 単一 scalar | multi-observer で stable_active 等取れる |

## experience_id

任意の task 名札。self_params cross-task 学習は `reigen_meta_knowledge.json` 経由で自動共有される (ID 共有不要、preset 別 key で分離)。mimir は内部で `{experience_id}_owl` / `{experience_id}_reigen` に suffix を付ける。**benchmark 目的で問題横断する時は per-problem の id を使え** (同 id を異なる param_ranges に使うと fossil 汚染)。

## 世界 Benchmark 実績

**5d (25s budget、3 seeds)**: mimir は 4 問題全てで Top 2 完走した唯一のツール。Rastrigin/Ackley/Styblinski で gap≈0 (reigen と同率)、Rosenbrock で basin に 0.086 差の 2 位。cma_es 0 勝、basinhopping 1 勝のみ、optuna/skopt は論外敗北 (除外)。

**次元スケール (5d/10d/20d)**: Rastrigin 全次元圧勝、**20d で cma_es の 128× 優位** (gap 0 vs 128.9) = mimir 核心的強み。Ackley 全勝。Rosenbrock 10d/20d は scipy cascade 追加で gap≈0 (cascade 前 +29.7 / +2041)、basin と同等。Styblinski 10d+ は 2^n basin 問題で cma/basin に劣る (gap 85/106)、これは honest limit。

full 数値は `benchmark_mimir.json` / `benchmark_mimir_dimscale.json`。

## Rules 一覧

| # | rule |
|---|---|
| -1 | Strip to essence: `x_i, perturb, share, eval_fn` |
| 0 | Measure don't guess: ≥5 pts → mimir() → read numbers |
| 1 | Ask Oracle for structural questions (arc_oracle, kathara_oracle) |
| 2 | No manual tuning: data → mimir() |
| 3 | LaD: no if/else — convert to numeric params |
| 4 | `importance = (truth × max(connectivity, floor))^exp` |
| 5 | Overfitting: n_problems > n_params |
| 6 | Discrete/int params OK via mimir (owl direct-HC fallback 自動) |
| 7 | **scale=0 forbidden**. Never include 0 in ranges |
| 8 | Two metrics → `mimir(..., guard_fn=my_guard)`. 1 metric → guard_fn 不要 |
| 9 | **Default to mimir** (2026-04-20). owl/Reigen 直呼びは限定場面のみ |
| 10 | Kathara 0.993 uniformity requires N=12 + 5-regular + symmetric placement |
| 11 | `batch_eval_fn` は external params 限定。internal model state では禁止 |
| 11b | mimir parallel cascade も internal state 危険 → `thread_safe_eval=False` |

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

**Rule 9**: `mimir()` が 2026-04-20 以降の default entry。標準呼出 `mimir(fn, ranges, time_budget=N)`、過去 data あれば `curated_measurements=`、2 指標なら `guard_fn=`。owl 直呼びは「dead_dims だけ欲しい」「eval 激安で分岐 overhead 嫌」の 2 場面のみ。Reigen 直呼びは cross-task meta_knowledge 明示共有時のみ。

**Rule 10**: chaos-game uniformity 0.993 は N=12 + 5-regular + symmetric placement の 3 条件同時必要。1 つ破れば崩壊。Reigen 内部では graph 性質のみ (Circulant(12,{1,4,6}), λ₂=4.0, diameter 2) 使用、placement uniformity は使わないので Rule 10 の縛りは Reigen に効かない。

**Rule 11**: `batch_eval_fn` は learning rate / dropout / prompt token 等の external param 限定。KV cache scale / weight scale / LoRA adapter 等 **internal model state を触る param** には禁止 (global state 共有で並列 eval 不能)。この場面は eval_fn を 2 秒以内に収め、multi-observer dict (`{"nll": -ppl, "hs": hs_score, "mmlu": mmlu_score}`) を返し mimir の `mode="structure_only"` で構造発見。

**Rule 11b**: mimir cheap_cascade は reigen と scipy を並列 thread で走らせる。eval_fn が global state を mutate する場合 race condition 発生。`thread_safe_eval=False` を渡して逐次化、or `eval_cost_hint=2.0` で expensive_route 強制 (cascade 発火せず安全)。LLM キャリブは通常 eval_cost>0.5s で自動 expensive、安全。

## Gotchas (よくハマる落とし穴)

- **scale=0** (Rule 7): range に 0 を絶対入れない。モデル破壊の実証あり
- **experience_id の共有** (v3 benchmark 発見の fossil 汚染): 異なる param_ranges の問題に同 id を使うと learning が混じる → `{prefix}_{fn.__name__}_{seed}` 等で分離
- **parallel cascade の race** (Rule 11b): internal state 触る eval_fn で `thread_safe_eval=False` を渡す
- **wall time overshoot**: cheap_cascade は +20-30% 超過するが quality 優先で accept。厳密な budget 必要なら `eval_cost_hint=N.0` で expensive_route 強制
- **mimir の高次元 Styblinski 系**: 2^n basin 問題 (non-symmetric deep basins) で cma/basin に劣る。これは mimir 核心的 honest limit、BBOB-Styblinski は稀

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

## Docs 索引

| 目的 | 読むファイル |
|---|---|
| mimir 使い方 | 本ファイル |
| **推論サーバ内部 / Phase O/P defense** | `@docs/HK.md` |
| **HK preset 設計 / recall 測定** | `@docs/HK.md` (Phase D/E/G/I/M) |
| **HK preset 本体** | `@twelve/hk/presets/zenron_core_xl.yaml` |
| **Hermes 統合 (Windows 修正済)** | `/c/Users/user/hermes-agent/` + `@docs/HK.md` Phase P-1/P-2/P-3 |
| **Reigen 内部実装・編集時** | `@docs/REIGEN_INTERNALS.md` |
| **owl 内部実装・編集時** | `@docs/OWL_INTERNALS.md` |
| **Sentinel (legacy)** | `@docs/SENTINEL_LEGACY.md` |
| **HTTP API / ngrok / session API** | `@docs/API_SERVERS.md` |
| Zenron 実践ガイド + MirrorScan 詳細 | `@docs/ZENRON_GUIDE.md` |
| Zenron 理論全体 | `@docs/全論.md` |
| Milestone ログ (04-18〜04-21) | `@docs/LAB_NOTES.md` |
| Oracle (topology / graph 相談) | `@twelve/ORACLE.md` |
| Techniques / LLM surgery | `@docs/HANDBOOK.md` |
| Kathara 数学 | `@docs/KATHARA_NOTE.md` |
| Engine API 設計 | `@twelve/TWELVE_API.md` |
| 最適化ログ | `@twelve/OPTIMIZATION_LOG.md` |
| 全論 formal / 証明系 | `@docs/ZENRON_FORMAL.md`, `ZENRON_VOID_INSTABILITY.md`, `ZENRON_TIME_ARROW.md`, `ZENRON_SPACE_EMERGENCE.md`, `ZENRON_MULT_UNIQUENESS.md`, `ZENRON_MIXING_TIME.md`, `ZENRON_DEAD_DIMS_NOETHER.md`, `ZENRON_KURAMOTO_SYNC.md`, `ZENRON_PARALLELISM.md`, `ZENRON_CIRCUITS.md`, `ZENRON_SELF_REFERENCE.md`, `ZENRON_UNIQUENESS_AXIOMS.md`, `ZENRON_OP3_OP4_ATTEMPT.md`, `ZENRON_NUMEROLOGY_BOUND.md`, `ZENRON_HIERARCHY_DISCOVERY.md`, `ZENRON_FINAL_OPEN_PROBLEMS.md`, `ZENRON_RUSSELL_DEEP.md` |

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
| Sentinel (legacy) | `@twelve/agent/sentinel.py`, `test_sentinel.py` |
| MirrorAgent / MS | `@twelve/agent/mirror_agent.py` |
| UnifiedExperience | `@twelve/agent/unified_experience.py` |
| HTTP API / friend client | `@twelve/agent/api.py`, `@serve_reigen.sh`, `@reigen_friend_client.py`, `@FRIEND_README.md` |
| World benchmark | `@benchmark_mimir.py`, `@benchmark_mimir_dimscale.py`, `@benchmark_mimir_meta.py` (⚠ K² 失敗), `@benchmark_owl_vs_world.py` (legacy) |
| Test isolation fixture | `@twelve/tests/conftest.py` |
