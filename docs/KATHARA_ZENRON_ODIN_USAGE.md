# Kathara × Zenron × ODIN — 実践使い分けガイド

**作成日**: 2026-04-26
**根拠**: `experiments/zenron_eml_bridge/integrated_report.md` (commit `b120725`)
で 12 target × 7 modes の数値検証完了。

このドキュメントは **「どの道具をいつ使うか」** を 3-5 分で読み切る実践ガイド。
理論の詳細は `docs/ZENRON_*.md`、Kathara 数学は `docs/KATHARA_NOTE.md`、
Reigen / OWL / Sentinel 内部は `docs/REIGEN_INTERNALS.md` 等を参照。

## 1. 概要 (1 行 × 3)

- **Zenron** = 最適化 + 物理 + 意識を 3-4 動詞 (`x_i / perturb / share / eval_fn`) で記述する **理論基盤**
- **Kathara** = 12 ノード × 30 エッジで対象を圧縮する **active architecture compression** (構造の地図)
- **ODIN** = 4 specialist 並列 + stabilizer + control_law で **連続 + 一部離散を最適化する実装**

3 つは独立じゃなく **層を成す相補関係**:

```
Zenron 全論          「なぜ動くか」 ← 理論基盤
   │ 3 動詞
   ↓
Kathara              「何を見るか」 ← 構造圧縮の地図
   │ 12 ロール + 30 接続
   ↓
ODIN (mimir_odin_*)  「どうなるか」 ← 実測探索エンジン
   ↓
データ / 実問題
```

## 2. 判定フロー (決定木)

```
質問: 何を最適化したい?

┌─ パラメータの数値だけ調整 (LLM scale, 薬量, 学習率)
│  └→ mimir_odin_stable()                        ★ Rule 9 default

├─ どのニューロン残す? どの mask? どの順列? (TSP, Lottery)
│  └→ mimir_odin_structure_policy()              Rule 17

├─ 12 個の似たタスクを並列で同時に解く (12 prompt 用途、12 区画薬物)
│  └→ kathara_mimir()                            12 環境同時最適化

├─ データから「式そのもの」を発見したい (sin, exp, 物理法則)
│  └→ experiments/zenron_eml_bridge/             symbolic regression

└─ 物理 + 意識 + 最適化を統一する理論探求
   └→ docs/ZENRON_*.md (40+ docs)                理論レイヤー
```

判定の質問が混ざってる場合 (例: 層選び + scale 同時) は
**`mimir_odin_structure_policy()`** が混合に強い。

## 3. ヒエラルキー (実測数値)

`experiments/zenron_eml_bridge/integrated_report.md` の Section 4 より。

12 target (exp, log, log(log), x², x·log(x), exp∘log, log∘exp, exp+log, x²+log,
sin, cos, tanh) で 7 modes 比較:

| mode | success rate | 用途 |
|---|---:|---|
| **kathara_mined (bridge)** | **9/12 (75%)** | symbolic regression、最強 |
| lad_seeded / kathara_lad / macro_mined | 7/11 (64%) | bridge ablation |
| raw_eml | 5/12 (42%) | bridge なしのベースライン |
| mimir_structure_policy | **0/12 (0%)** | symbolic に**届かない** |
| mimir_stable | **0/12 (0%)** | symbolic に**届かない** |

精度差: bridge `test_loss < 1e-8`、mimir 単独 `test_loss 1〜10⁴³`、
**gap 7-12 桁**。bridge は ODIN の **symbolic 拡張ユニット**として位置付け。

ただし bridge も sin / cos / tanh は **既存 budget では 0/5 失敗** —
universality 主張は理論的、深い tree が必要な関数は探索コスト指数的。

## 4. 5 つの道具 詳細

### 4.1 `mimir_odin_stable()` — default 連続最適化エンジン

**いつ使う**: 純連続な数値の最適化全般。LLM scale tuning、薬量、学習率、PID
制御ゲイン、エアコン温度。世界の AutoML (Vizier, Optuna, CMA-ES) と同等用途。

**何が独自**:
- **plateau 化** (peak → robust 解、摂動耐性 8% → 96%)
- **control_law gate** (per-dim σ/√n contract、high-dim 実用判定)
- **auto_check** (Rule 0.5 強制、本番前 sanity check)
- **4 specialist 並列** (default / lad / expensive / scipy-forced)

**最小例**:
```python
from twelve.agent.mimir_odin_stable import mimir_odin_stable

r = mimir_odin_stable(eval_fn, [(0.5, 1.5)] * 10, time_budget=300)
print(r["best_params"])         # plateau centroid
print(r["plateau_robustness"])   # 摂動耐性
print(r["practical_deployable"]) # 実用判定
```

詳細: CLAUDE.md Rule 9。

### 4.2 `mimir_odin_structure_policy()` — 離散構造発見

**いつ使う**: mask 選び、Lottery Ticket、TSP、レイヤー pruning、quantization
block、rewind epoch。「数字」じゃなく「種類選び」が含まれる問題。

**何が独自**:
- **policy_decoder** で連続 carrier を離散 policy に decode
- **policy_response_probe** で安全帯 (設計外利用に raise abort)
- **混合問題 hint** (`declared_structural=True`)

**最小例**:
```python
from twelve.agent.mimir_odin_structure_policy import mimir_odin_structure_policy

r = mimir_odin_structure_policy(
    eval_fn, ranges,
    policy_points=curated_policies,
    param_decoder=decode_to_mask,
    detail_fn=evaluate_with_metrics,
    time_budget=600,
)
print(r["best_policy"])  # 実際使う離散構造
```

**実測**: TSP 10 cities で gap +26.3% (純 stable は +72.9%)、tiny Transformer
Lottery Ticket で 6.6× 圧縮で満点維持、20.8× で精度 -3pt。

詳細: CLAUDE.md Rule 17。

### 4.3 `kathara_mimir()` — 12 環境同時最適化

**いつ使う**: 似た構造を持つ 12 個の独立タスクを並列で解きたい。LLM の 12 prompt
用途、薬物動態 12 区画、ポートフォリオ 12 戦略、気候 12 地域、連合学習 12 グループ。

**何が独自**:
- **Kathara 30 edges** で隣接 node から best params を share (探索 memory 圧縮)
- **5.6× speedup + 37% 質向上** (smoke で実証)
- **ProcessPool 12 並列** で CPU full 活用

**最小例**:
```python
from twelve.agent.kathara_mimir import kathara_mimir

results = kathara_mimir(
    eval_fns=[fn_0, fn_1, ..., fn_11],   # 12 個必須
    param_ranges=[(lo, hi)] * n_dims,
    time_budget=300,
    share_rounds=3, share_weight=0.3,
)
# results[i]["best_params"] for each task
```

**注意**: `eval_fns` ちょうど 12 個必須。N≠12 の時は 12 にパディング or グループ化。

詳細: docs/KATHARA_MIMIR.md。

### 4.4 `experiments/zenron_eml_bridge/` — シンボリック回帰

**いつ使う**: データから「**式そのもの**」を発見したい。物理法則、化学反応式、
回路設計式、統計モデルの構造。

**何が独自**:
- **EML primitive** (`eml(x,y) = exp(x) - ln(y)`) の universal substrate (arxiv:2603.21852)
- **Kathara 12 ロール sharing** で並列 macro mining
- **LaD witness** (hand 知識 + mined macro) で memory 学習
- **transfer warm-start** で 16-97× speedup (既存 target → 新 target)

**最小例**:
```bash
python experiments/zenron_eml_bridge/run_eml_zenron_smoke.py
```
- 実装は `make_benchmarks()` に target 追加するだけ
- mode 11 種から best を比較

**世界対応**: PySR, AI Feynman, DSR (Genetic Programming / RL ベース) と別アプローチ。
本 bridge は **「1 演算子 + Kathara mining」** の minimalist 路線。

**限界**: 現 budget では sin/cos/tanh 不可 (深 tree 必要、探索コスト指数)。

### 4.5 Zenron 理論 (背景レイヤー)

**いつ参照する**: 直接コードでは呼ばない。**理論的根拠**を確認したい時。

- 3 動詞 (`x_i, perturb, share, eval_fn`) を Reigen が proxy として使ってる
  (`a + b·x + c·x·y` という zenron 多項式形)
- Landauer 連接 (Rank A 数値 validate 済) で熱力学 bridge
- Kathara 12 = IIT Φ sparse class #1 で意識基盤と整合 (Rank B)

**主要 doc**:
- `docs/全論.md` — 理論本体
- `docs/全論の公式の活用.md` — 応用ガイド
- `docs/ZENRON_LANDAUER.md` — 熱力学連接
- `docs/ZENRON_CONSCIOUSNESS.md` — IIT Φ
- `docs/ZENRON_PROOFS_MASTER.md` — 全証明索引 (Rank A 5 + B 7)

## 5. 7 つの実用シナリオ

### シナリオ 1: LLM の layer scale 調整 (Gemma 4 / Qwen)

```python
# 「どの層に何 倍 を掛ければ HellaSwag 70.8% 達成?」
def my_eval(p):
    apply_scales(p)
    score = run_hellaswag()
    restore_scales()
    return score

r = mimir_odin_stable(my_eval, [(0.5, 1.5)] * 32, time_budget=3600)
```
→ **stable** 一択 (連続値、LLM eval は LaD wrapper 推奨、Rule 12)。

### シナリオ 2: Lottery Ticket (ニューラルネット圧縮)

```python
def eval_with_mask(p):
    mask = decode_top_k_mask(p)
    return train_and_test_with_mask(mask)

r = mimir_odin_structure_policy(
    eval_with_mask, ranges,
    param_decoder=decode_top_k_mask,
    detail_fn=lambda p: {"score": eval_with_mask(p),
                          "policy": {"k": ..., "indices": ...}},
)
```
→ **structure_policy** (離散 mask 選び)、experiments/lottery_ticket_odin_v0/ 参照。

### シナリオ 3: 12 prompt の同時最適化 (LLM API)

```python
prompts = ["math", "code", "ja", "en", ..., "creative"]  # 12 用途
eval_fns = [make_eval_for_prompt(p) for p in prompts]

results = kathara_mimir(eval_fns, ranges, time_budget=600,
                       share_rounds=3, share_weight=0.3)
```
→ **kathara_mimir**、隣接プロンプトの best params を share で加速。

### シナリオ 4: 物理データから式を発見

```python
# 観測 (x, y) ペアから y = f(x) を発見
# experiments/zenron_eml_bridge/run_eml_zenron_smoke.py の make_benchmarks() に追加
Benchmark("my_target", my_target_fn, (0.25, 3.0), None)
```
→ **zenron_eml_bridge** (純 mimir では届かない、本検証で実証)。

### シナリオ 5: TSP / 配送ルート最適化

```python
def eval_route(p):
    order = np.argsort(p)
    return -tour_distance(order)

r = mimir_odin_structure_policy(
    eval_route, [(-1, 1)] * n_cities,
    param_decoder=lambda p: {"order": list(np.argsort(p))},
    detail_fn=lambda p: {"score": eval_route(p),
                          "policy": {"order": list(np.argsort(p))}},
)
```
→ **structure_policy** (純 stable で TSP は対称軸 tie で fatal raise、auto_check
が誘導する)。

### シナリオ 6: 創薬・薬物動態 12 区画モデル fit

```python
# 12 組織 (脳、肝、腎、肺、心、筋、皮膚、消化、骨、脂肪、血液、その他) の
# 薬物濃度時系列を同時 fit
eval_fns = [make_pkpd_eval(tissue) for tissue in TISSUES_12]
results = kathara_mimir(eval_fns, [(0.1, 10)] * 5, time_budget=1800)
```
→ **kathara_mimir** (12 区画 PK/PD は文字通り Kathara 設計と一致)。

### シナリオ 7: ハイパーパラメータ tune (RNN/Transformer)

```python
def eval_hyperparams(p):
    lr, dropout, batch_log = p
    return train_and_validate(lr=lr, dropout=dropout,
                              batch_size=int(2 ** batch_log))

r = mimir_odin_stable(
    eval_hyperparams, [(1e-5, 1e-2), (0.0, 0.5), (4, 9)],
    time_budget=3600,
)
```
→ **stable** (純連続 hyperparameter tune、Optuna / Vizier 同等用途)。

## 6. よくある失敗と救済策

### 失敗 1: TSP / 順列を `mimir_odin_stable` に投げる
- **症状**: `auto_check detected fatal eval_fn issues: 2 点 probe で同一 score`
- **原因**: midpoint と corner で argsort tie → constant 検出 fatal
- **救済**: ✅ `auto_check` が 5-15 秒で誘導、エラーメッセージで `structure_policy`
  推奨される (Rule 0.5)。素直に switch する

### 失敗 2: 50d 純連続を `structure_policy` に identity decode で渡す
- **症状**: `policy_response_probe.score_spread = 1.29e-10` で 5 秒 abort
- **原因**: 高次元 random seed が peak から遠く全 score ≈ 0、policy 反応なし判定
- **救済**: 純連続なら **`mimir_odin_stable` を使え**。それが守備機能の意図

### 失敗 3: 整数 + 連続混合問題で probe が混合検出できない
- **症状**: `recommended_optimizer = mimir_odin_stable` と出るが本当は混合
- **原因**: probe の本質的限界 (`abs()` の kink と `round()` のジャンプは数学的に
  区別不能)
- **救済**: ✅ `check_eval_fn(declared_structural=True)` で hint、
  または直接 `structure_policy` を呼ぶ

### 失敗 4: `sin(x)` / `cos(x)` を bridge で fit できない
- **症状**: 全 mode で 0/5 失敗 (本検証で実証)
- **原因**: EML 表現に必要な depth が大きく、現 budget では探索不可
- **救済**: 論文の手法 (PyTorch 勾配 + Gumbel-softmax) で別 GPU 実装、または
  PySR 等の世界標準 symbolic regression を使う

### 失敗 5: mimir 単独 (`stable` / `structure_policy`) で symbolic regression
- **症状**: 全 12 target で 0/3 success、test_loss 0.01〜10⁴³
- **原因**: 連続最適化は EML tree 構造発見に届かない (本検証で 7-12 桁 gap)
- **救済**: bridge (`experiments/zenron_eml_bridge/`) 必須、ODIN の symbolic 拡張
  ユニット

### 失敗 6: kathara_mimir に 12 個未満の eval_fns を渡す
- **症状**: `Need exactly 12 eval_fns (Kathara 12), got X`
- **救済**: 12 にパディング (重複 OK)、または別 problem を 12 個まとめてグループ化

### 失敗 7: stochastic eval_fn を raw で `mimir_odin_stable` に投げる
- **症状**: proxy_r2 < 0.3 で direct-HC fallback、収束遅い
- **救済**: `wrap_multi_obs(eval_fn, n=20)` で LaD 化 (Rule 12)、または
  `n_samples_per_eval=20` で自動集約。`auto_check` も `stochastic_cv > 0.10` で
  自動推奨を出す

## 7. 世界比較 (1 段階要約)

| 領域 | 世界トップ | 本プロジェクト |
|---|---|---|
| **連続最適化** | CMA-ES, Vizier, Optuna | ODIN (Rastrigin 20d で cma_es 128× 優位) |
| **MoE** | Mixtral 8x7B, Switch Transformer | kathara_moe (Mixtral 比 4.8× 短時間、medium scale) |
| **Symbolic regression** | PySR (Cranmer) — 標準 | zenron_eml_bridge (1 演算子 + Kathara mining、minimalist) |
| **NAS / Pruning** | DARTS, Lottery Ticket Hypothesis | structure_policy (TSP gap 1/3、Lottery 6.6× 圧縮) |
| **Connectome 解析** | Google/Harvard H01 | Kathara で 12 ロール圧縮 (別軸) |
| **意識理論** | IIT (Tononi) | Zenron + Kathara 12 = sparse class #1 (Rank B) |

## 8. 関連ドキュメント

### 必読 (本ガイドの根拠)
- `experiments/zenron_eml_bridge/integrated_report.md` — 12 target × 7 modes
  数値検証 (本ガイドの一次出典)
- `CLAUDE.md` — Rule 9 (default to stable), Rule 13 (odin), Rule 14 (cardinal),
  Rule 16 (stabilizer), Rule 17 (structure_policy)

### 各ツール詳細
- `docs/KATHARA_MIMIR.md` — `kathara_mimir()` 技術解説
- `docs/REIGEN_INTERNALS.md` — Reigen 内部 (zenron 多項式 proxy)
- `docs/OWL_INTERNALS.md` — owl primitive (mimir 内部)
- `docs/SENTINEL_LEGACY.md` — Sentinel (legacy)

### Zenron 理論
- `docs/全論.md` — 理論本体
- `docs/全論の公式の活用.md` — 応用ガイド
- `docs/ZENRON_PROOFS_MASTER.md` — 全証明索引 (Rank A 5 + B 7)
- `docs/ZENRON_GUIDE.md` — 実践 + MirrorScan 詳細
- `docs/ZENRON_LANDAUER.md` — 熱力学 bridge (Rank A)
- `docs/ZENRON_CONSCIOUSNESS.md` — IIT Φ (Rank B)
- `docs/ZENRON_GR_DERIVATION.md` — Einstein 方程式導出

### Logic-as-Data
- `docs/LOGIC_AS_DATA.md` — Lv1-12 階層論

### 実験例
- `experiments/zenron_eml_bridge/` — symbolic regression
- `experiments/structure_policy_demo/` — TSP / sparse / 混合検証
- `experiments/kathara_mimir_smoke/` — 12 並列 demo
- `experiments/lottery_ticket_odin_v0/` — Lottery Ticket
- `experiments/h01_kathara_compression_v0/` — H01 brain 解析

## 9. 1 行サマリー

> **「数字なら stable、種類選びなら structure_policy、12 個なら kathara_mimir、
> 式の発見なら zenron_eml_bridge、理論なら Zenron docs」**

迷ったら `check_eval_fn()` を本番前に走らせて recommended_optimizer に従う
(Rule 0.5、`mimir_odin_stable(auto_check=True)` で default 強制)。
