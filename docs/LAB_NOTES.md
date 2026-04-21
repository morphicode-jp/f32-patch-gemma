# LAB_NOTES — Experiment Log

Chronological record of experiments, discoveries, and failures.
For technique specs, see `HANDBOOK.md`. For API usage, see `CLAUDE.md`.

---

## Phase 1-3: Qwen 9B Surrogate Exploration (2026-03)

Hourglass pattern discovered. 4bit optimal for surrogates. Weight stats work, activation stats fail.

---

## Phase 4-7: Qwen 9B Weight Surgery (2026-03~04)

| Phase | Method | PPL | Verdict |
|---|---|---|---|
| 4 | KV quant (real) | -4.23 | VRAM 17.8→9.6GB |
| 5-7 | Pruning | varies | 20 layers partially prunable |
| **8** | **Deep (residual+MLP+attn)** | **-2.3%** | L0 temp 1.81, L28 residual 160% |
| 9 | Weight bias surgery | +0.3% | Failed — weights already optimal |
| **10** | **Head surgery (32L×8G)** | **-2.79%** | Kill L14-G2(0.27), boost L22-G7(1.99) |
| 11 | Activation shape | +0.5% | Failed — unstable |
| 12 | Layer skip | +59% PPL | Speed only, quality catastrophic |
| **13** | **Unified (head+deep+scale)** | **-4.6%** | **Best. 104 dims, 6 cat** |

**Lessons:** weight stats work, activation stats fail · KV quant regularizes · block scales no surrogate (cliffs) · degenerate solutions need constraints

---

## Phase 8: GGUF Production (2026-04-05)

Three-stage F32 optimization on Qwen 9B Q4_K_M:
1. High-sensitivity F32 (104p, 5 cat): -1.71%
2. Low-sensitivity F32 (+168p, +7 cat): -2.88%
3. Block scales (uniform ±1.1%): **-3.75%** total

KV Cache: q8_0 winner (PPL improved + VRAM -73%).
Production: `Qwen3.5-9B-Q4_K_M-Twelve-V3.gguf`

---

## Phase 9: LoRA + ARC-Challenge (2026-04-06)

| Method | Result |
|---|---|
| V2 General LoRA | 93.00% |
| Oracle LoRA V2 | 93.77% |
| **Thinking mode** | **95.82% (> Opus 94%)** |

Optimal: 1.3 epochs (3 overfits on 492 samples).
**Confirmed:** Q4 ≈ Q8 ≈ FP16 → intelligence = topology, not precision.

---

## Phase 10: Kathara-Transformer (2026-04-06~07)

Derived from FlyWire connectome. 63:24:13 local:FF:FB ratio.

**Key experiments:**
- Kathara-Attention mask: compute -71%, PPL +1.9%
- Mask-then-Trim: KV VRAM -69.6%
- 25% global = optimal (matches fly brain 24%)
- KV O(1): 268 tokens fixed, -99.7% at 100K context

**Findings:**
- PPL identical across all KV trim configs → must also test recall
- Layer skip NOT viable (1 layer +1.4%, 4 layers +64%, 8+ catastrophic)
- Output similarity ≠ skippable

---

## Phase 11: Qwen 27B Full Stack (2026-04-07~09)

### F32 Scale Optimization

449/851 tensors are F32. 8-category sensitivity analysis:
- post_norm most sensitive (8.40), ssm_alpha dead (0.56)
- MA circle: 184 evals, 63 min → PPL **-3.02%**
- Key: most scales shrink (compensating quant inflation), ssm_conv1d/beta grow

### Kathara Skip (llama.cpp C++)

Modified qwen35.cpp (~80 lines). 72 edges, AT-optimized alpha.
- 12g: PPL -2.80%
- 24g: PPL **-6.48%**, ISS 82.2→**92.2** (7.3GB > 54GB FP16)
- Beta: group 4=1.256 (intelligence core). Gamma: no effect.

### KV Trim (SSM-Aware)

hk=82, w=5 → 96% KV reduction, recall 100% (SSM retains memory).
Infinite thinking: 10+ rounds, KATHARA-ZENRON-42 exact recall, O(1) KV.

### Combined: PPL -5.88% (580-chunk), ISS 91.7, all features simultaneous.

---

## Phase 12: ISS Proxy Discovery (2026-04-10)

**Biggest discovery.** Weight stats → JSON → ISS proxy (0.1ms) → 100K evals → burn to GGUF → verify once.
Total: 6 minutes. World: GPU 256 × 3 weeks.

Reasoning 60%→93.3% (+33pts). V3 benchmark: 96.0%.
ISS proxy matched real LLM performance. GPU-free intelligence optimization.

---

## Phase 13: MixQ (2026-04-11)

### Qwen 9B
SSM layers (75%) → Q2_K: +6.6pt reasoning, -11% size.
Compression removes noise → SNR improves → reasoning quality up.

### Gemma 4 31B
FFN tensors (72%) → IQ2_XXS+imat: same +6.6pt.
Pure Transformer = flat (std ratio 1.041), imatrix within-tensor allocation is key.
**+6.6pt is universal across architectures.**

PPL vs benchmark diverge for instruct models: PPL favors uniform quant, benchmarks favor imatrix.

---

## Phase 14: F32 Calibration + Owl Upgrade (2026-04-12)

### F32 Post-Quantization Calibration

| Step | Time | PPL | Bytes |
|---|---|---|---|
| Baseline Q2_K | — | 1554 | — |
| Sensitivity scan (25 meas) | 28 min | found G4/G5 | — |
| Grid search G4×G5 (42 meas) | 46 min | 32.8 | 40 B |
| Full validation | 4 min | 41 | 40 B |
| output_norm sweep (17 meas) | 22 min | 27.9 | +21 KB |
| NaN repair (Q3_K data) | instant | **23.9** | +756 KB |

### Owl Interaction Proxy

7-dim norm optimization failed (R²=0.38). Root cause: parameter interactions invisible to linear proxy.

Fix: connectivity-guided interaction terms.
`interaction_imp[i,j] = sqrt(pair_truth × pair_connectivity)`
Only high-connectivity pairs get x_i×x_j terms. Same zenron double-filter.

**Result: R² 0.38→0.79 on same data.** Confidence: low→high. verify_fn enabled.

### KV Optimization

- KV trim (hk=64, w=128): recall 4/4
- KV cache q8_0: PPL 23.9→23.8 (improved), VRAM -47%
- PPL measurement can't detect KV trim effect (single-chunk processing)

### Kathara Skip for Gemma 4

C++ implementation in gemma4-iswa.cpp (~120 lines).
- Uniform alpha: all failed (Gemma 4 too sensitive)
- **Per-edge: negative alpha improves PPL** (inhibitory connections)
- Measurement in progress (71 samples)

### What Failed

| Method | Result | Lesson |
|---|---|---|
| Owl linear on norms | R²=0.38 | Needs interaction terms |
| MixQ tensor mixing | PPL 2x worse | Discontinuous types → proxy fails |
| Q6_K→Q3_K requant | Model collapse | attn_q precision critical |
| Layer pruning (scale=0) | PPL=vocab_size | Transformer can't skip via scale |
| Uniform Kathara alpha | All degraded | Per-edge required for Gemma 4 |
| ISS on Gemma 4 | Invalid (ISS v2=1.8) | ISS assumes graph structure, Gemma is flat |

### Key Discoveries

1. Post-quantization F32 calibration is universal (EfficientQAT validates)
2. Owl interaction proxy: zenron pair extension doubles R²
3. Pure Transformers need inhibitory skip connections
4. LaD→Owl: 125min manual → 32min automated
5. Published GGUFs can have corrupted data (NaN in Q3_K)

### Compression Paper Survey

| Paper | Key Technique |
|---|---|
| EfficientQAT (ACL 2025) | Scale factor optimization (validates our approach) |
| RAMP (2026) | Layer-wise bit allocation (matches our findings) |
| ShortGPT | Layer pruning (llama-quantize --prune-layers) |
| SoLA (2026) | Sparse + low-rank hybrid, training-free |

---

## Phase 15: Gemma 4 Q4_K_M on RTX 5090 (2026-04-15)

**環境移行:** i7-12700K + 3080Ti 12GB → RTX 5090 32GB (CUDA 13.2)

### HellaSwag正しい評価方法の発見

llama-perplexity `--hellaswag` が唯一正確な評価方法。
サーバーAPI top-N logprobs → miss率21-41%で使えない。lm-eval `gguf`モデル型 → echo非対応。

| 評価方法 | 精度 | 速度 |
|---|---|---|
| **llama-perplexity --hellaswag** | **正確** | 100tasks ~9s |
| サーバーAPI top-N logprobs | miss 21-41% | 速い |
| lm-eval gguf | 動作しない | — |

### F32キャリブレーション: PPLベース vs HellaSwagベース

**PPLベース（失敗）:**
- PPL 12,734→46 (-99.6%) だが対話崩壊
- L34×0.45が支配的 → IT priorを破壊 → `flu flu de la`ループ
- eval_fnがITモデルに不適切だった

**HellaSwagベース（成功）:**
- HS 57%→70.8% (+13.8pt) greedy、最終確認で62.25→69.75% (+7.5pt)
- 3層のみ: L18×1.5, L25×1.5, L26×1.5（深層スケールアップ）
- 対話品質完全保持
- 4層目以降は層間相互作用で崩壊

**教訓: eval_fnが全てを決める。同じ技術でも測定対象で結果が真逆。**

### Multi-Observer構造分析（Owl最深分析）

153測定（sensitivity scan 123 + random sample 30）、88分。

| 発見 | 値 |
|---|---|
| HS×PPL相関 | 0.139 (ほぼ無相関) |
| HS proxy R² | 0.841 (zenron_interact) |
| PPL proxy R² | 0.440 (zenron_interact) |
| PPL dead_dims | 56/61 |
| HS dead_dims | 5/61 |
| stable_active | L52, L54, L55, L58 |
| stable_dead | L3, L11, L41, L51 |

**Phase 3 検証結果（5候補すべてPPL崩壊）:**

| 候補 | HS | HS変化 | PPL | PPL変化 |
|---|---|---|---|---|
| Top5_moderate | 71.00% | +10.25 | 39 | -99.7% |
| Top3_wide | 70.50% | +9.75 | 27 | -99.8% |
| AllPositive_slight | 52.50% | -8.25 | 65 | -99.5% |

**核心的発見:**
- layer_output_scaleはIT alignmentのON/OFFスイッチ
- PPL-dead dims（Owl分類）でも累積変更でPPL崩壊
- 「品質の連続的なツマミ」ではなく「モード切替」
- **Gemma 4のlayer_output_scale surgeryは3層が天井**

### Q6_K比較

Q6_K baseline: HS 59.5% (+7GB)。Q4_K_M+キャリブ(69.75%)のほうが上。
→ 量子化レベルを上げるより、重要層のスケール補正のほうが効果的。

### What Worked / What Failed

| 手法 | 結果 | 判定 |
|---|---|---|
| HellaSwag eval_fn | HS +7.5pt, IT保持 | ✅ |
| greedy層追加 | 3層まで有効、4層目崩壊 | ✅ (限界あり) |
| Multi-Observer Owl | 構造発見に成功 | ✅ (分析目的) |
| PPL eval_fn on IT model | IT崩壊 | ❌ |
| Owl autonomous (61dim) | proxy hallucination | ❌ |
| PPL-dead dimsでの安全最適化 | 累積でPPL崩壊 | ❌ |

### スクリプト

| ファイル | 用途 |
|---|---|
| `weight_analysis/gemma4_hellaswag_eval.py` | HellaSwag eval_fn |
| `weight_analysis/gemma4_hellaswag_greedy.py` | greedy層最適化 |
| `weight_analysis/gemma4_multiobs_analysis.py` | Multi-Observer 3-phase分析 |
| `weight_analysis/gemma4_calib_eval.py` | PPL eval_fn (⚠️ ITモデルに不適切) |

### 結論

**Gemma 4 Q4_K_M + 3層キャリブ (L18,L25,L26 ×1.5) が確定版。**
12バイト変更で+7.5pt、コスパ異常。これ以上はlayer_output_scaleでは不可能。
Qwenのほうが改造の余地が大きい（構造が分離している）。

---

## 2026-04-16: 階層型Kathara Brain Sim (v8) + 活用可能性分析

### 問題: ニューロンスケーリング
12N Kathara(12,{1,4,6}) = 94.37% だが、48Nにスケールすると劣化:
- 48N fetal (ランダムトポロジー) = 85.38 (-8.99)
- 抑制、ハブ配置、スキップ接続 — 全て失敗

### 解決: 階層型Kathara (4×12)
4つの独立 Kathara(12) クラスタを順次接続:
- Cluster 0: sensors → Kathara(12)
- Cluster c: proj(cluster[c-1]) → Kathara(12)
- Motor: last cluster nodes 5, 11
- Params: 4×90 edges + 36 projections = **396** (vs fetal ~1248)

### 結果

| Config | Score | Params | vs 12N |
|---|:---:|:---:|:---:|
| 12N Kathara (ref) | 94.37 | 90 | — |
| 48N fetal (random) | 85.38 | ~1248 | -8.99 |
| **48N hier 4×12** | **94.00** | **396** | **-0.37** |
| 48N hier + inhib 20% | 93.99 | 396 | -0.38 |
| 48N hier + skip | 86.69 | 420 | -7.68 |
| 48N hier + feedback | 85.68 | 408 | -8.69 |
| 24N hier 2×12 | 86.69 | 192 | -7.68 |

### 核心発見
**トポロジー保存 > 機能追加**
- 抑制 (Dale's law): 中立 (-0.01)
- スキップ接続: 有害 (-7.31)
- フィードバック: 有害 (-8.32)
- Kathara トポロジーそのものが知性の源泉。追加構造は有害。

### 活用可能性 (7方向)

| # | 活用先 | 方法 | 根拠 |
|---|---|---|---|
| 1 | Nous policy network | 線形→Kathara(12)隠れ層 | 非線形RL問題で構造保証 |
| 2 | Owl 階層パラメータ最適化 | 大次元→4×12クラスタ分割 | 68%パラメータ削減で同等性能 |
| 3 | LLM Attention Head構造 | Head間をKathara接続で制約 | ランダム接続の劣化防止 |
| 4 | MoE Expert Routing | Expert間をKathara接続 | expert collapse防止 |
| 5 | F32キャリブ相互作用予測 | 層間相互作用グラフ化 | 危険な組合せの事前予測 |
| 6 | チップ設計 | 12コア×Circulant配線 | diameter 2, λ₂=4.0保証 |
| 7 | Rule 9: トポロジー保存原則 | スケーリング時の設計指針 | 実験で証明済み |

### スクリプト
- `kathara_brain_sim_v8.py` — 全実装 (run_hierarchical_kathara, run_hier_full_test)
- 結果: `kathara_brain_sim_v8_hier_result.json`, `kathara_brain_sim_v8_hier_full_result.json`

---

## Milestone ログ (2026-04-18 〜 2026-04-21)

2026-04-21 に CLAUDE.md "Proven results" 表から移管。個別 commit 記録と重複する細部は git log 参照。

| 日付 | 技術 | 結果 / 要点 |
|---|---|---|
| 04-18 | **Reigen v1+v2+v3 unified** | dim-additive self-application、kathara_12 default (12 knobs + Hebbian 30-edges)、batch_eval_fn API。45+12 tests all green |
| 04-18 | Kathara chaos-game uniformity | **0.993** on 12-icosahedron (p=0.17)。Rule 10 の数値的根拠 |
| 04-18 | Qwen3.6-NVFP4 KV-Reigen | Single-obs NLL hurt HS -2pt (hit boundary 0.01/1.5 = proxy overfit)。**Rule 11 の誕生契機** |
| 04-19 | kathara_17_adaptive A/B 3/3 勝利 | Rosenbrock 5d / Ackley 8d / Styblinski 6d で vs kathara_12: 2.7-3.3× 速 + 15/15 approved (vs 8/15)。default に昇格 (commit 9381df1) |
| 04-19 | Rule 9 (genesis 学習累積) 実装 | reigen_meta_knowledge.json 新設 + atomic write、Reigen.run() 成功時に self_param_best 書戻し (commit 888dfc1) |
| 04-19 | **owl direct-HC fallback 移植** | Sentinel 秘密兵器を owl 本体に。Rastrigin 5d で **gap 45.6 → 0.000**、eval 数 1/5。autonomous + verify_fn で多峰 landscape 突破可能に (commit 911997e) |
| 04-19 | owl budget-aware autonomous loop | time_budget を TOTAL 予算として厳守 (commit 1addbd9) |
| 04-20 | 世界 benchmark: Reigen 3/4 勝利 | Rastrigin/Ackley/Styblinski で Reigen_k17 が optuna/skopt/cma/basinhopping を圧倒 gap=0。Rosenbrock のみ basinhopping に僅差負け (gap 0.08 vs 0) |
| 04-20 | owl L-BFGS-B refinement | scipy L-BFGS-B を owl 末尾 1-shot で発動 opt-in。**Styblinski 5d score -6→195.83 (global 到達)**、Rosenbrock gap 173→3 (57×) (commit edef9f5) |
| 04-20 | owl multi-start fallback | fallback の warm-start を L2 diverse top-3 に (commit a3c8a4c) |
| 04-20 | owl random_restart_count | K 個の uniform random warm-start 注入、basinhopping 模倣 (commit 62d20c1) |
| 04-20 | **mimir 上位層誕生** | owl+Reigen cascade の meta-dispatcher。eval コスト 1-call 測定 → 自動分岐。BBOB 3/4 gap=0 + Rosenbrock 0.08。新 default |
| 04-20 | mimir LaD 化 | dispatch 閾値を mimir_params.json で制御、kwarg > JSON > hardcode。K² 自己最適化は病的 runtime で失敗 |
| 04-20 | mimir mode="structure_only" | 最適化 skip、owl の dead_dims/fragility/proxy_r2 のみ返す高速分析経路 |
| 04-20 | **mimir scipy cascade 追加** | dim ≥ 10 + conf != "high" で scipy.basinhopping を Phase 2 に並列配置。**Rosenbrock 20d gap 2041 → 0** (commit 5d8f204) |
| 04-20 | mimir parallel cascade | reigen と scipy を ThreadPoolExecutor で並列実行 (commit 6c02df3) |
| 04-21 | mimir thread_safe_eval + Rule 11b | parallel cascade の race condition 文書化 + 回避 kwarg (commit de09c59) |
| 04-21 | mimir batch_eval_fn passthrough | 5090 GPU 活用、seed 20 点を 1 batch call で 10× 速く (commit baf6a3b) |
| failures | scale=0 → PPL=262144 | Rule 7。verify_fn catches hallucination |
| failures | single-obs on internal model state | fix: multi-observer + fast eval (Rule 11) |

---

## Pending / Future

- Qwen 27B Q4_K_M on 5090（全技術スタック適用）
- 100B+モデルでの極限圧縮仮説検証
- IT品質を直接測るeval_fn（MT-Bench等）の開発
- imatrix再生成（5090 GPU高速化の活用）
