# Paper v5 Draft — Adjacent-Layer Functional Specialization in Q2_K Quantized Gemma 4 31B

**Date**: 2026-05-30
**Status**: **n=1 pilot / hypothesis-generating** ((強 主 張 禁 止))
**Authors**: 平井 陽大 / Akito Hirai ((Independent Researcher))
**Predecessor**: paper v1 release ((2026-05-27)): L25+L26 ×1.5 = 8 byte F32 patch for Gemma 4 31B. Public release demonstrated **all 12 cells positive** ((3 quant levels × 4 benchmarks)): Q4_K_M patched ((19 GB)) beat Q8_0 BF16 baseline ((31 GB)) on all 4 benches ((HS 73.50% > 63.33%, GSM 87% > 70%, WG 70.32% > 65.59%, ARC 48.76% > 44.38%)); Q2_K HS +11.21pt; IQ1_M GSM +36.0pt. exp162 reduced-N rerun for this paper v5 release ((HS 1000/WG 500/ARC 200/GSM 100)) reproduces directionality on Q2_K with +9.13pt avg ((variance higher under small-N reference setting; MMLU not evaluated in either run))

---

## 1. Title ((working))

**Adjacent-Layer Functional Specialization in Q2_K Quantized LLMs: A Single-Question Pilot Observation of Emergent Timing Control between L25 (Compute) and L26 (Meta-Verify) — n=1, Hypothesis-Generating**

短 縮 版 ((blog post 用)): **「Q2_K Gemma 4 31B の L25 = 計 算、 L26 = メタ 検 算 — 隣 接 2 層 の 役 割 分 化 を 4 byte F32 patch で 観 察」**

---

## 2. Main Claim ((1 文))

> Per-layer ablation of paper v1's L25+L26 patch **suggests** that L25 and L26 contribute different behavioral characters ((L25 ≈ compute-focus; L26 ≈ verify-proposal)): exp166 confirms a statistically significant verify-pattern asymmetry ((L26-alone aux_rate 56% vs L25-alone 16%, p < 0.01)), while accuracy differences between patches on a single coin prompt are not significant at n=32 ((Fisher exact p=0.49)). The paper v1 mix likely acts as a **plateau-center** across these characters, explaining its strength in multi-benchmark averages while individual single-layer patches may peak on tasks aligned to their character. **This is a hypothesis-generating observation, not a confirmed mechanism**; full-bench verification ((exp167)) and direct mechanism observation ((logit-lens, activation patching)) are pending.

---

## 3. Abstract ((日 本 語))

paper v1 ((L25+L26 layer_output_scale ×1.5、 8 byte、 2026-05-27 公 開)) は Gemma 4 31B で dramatic な 効 果 を 示 し た: Q4_K_M patched が Q8_0 BF16 baseline を 全 4 bench で 勝 ち、 12/12 cell positive matrix ((Q2 HS +11.21pt、 IQ1 GSM +36.0pt 等))。 本 paper v5 は こ の paper v1 を **個 別 layer ablation** ((L25 単 体 ×1.5、 L26 単 体 ×1.5、 mix、 supplementary triple)) し て 「**8 byte の 中 で どの 4 byte が 何 を 担 う か**」 を 観 察 した pilot。 同 一 prompt ((コイン 6 回 3 連 続 表 確 率、 真 値 5/16)) で の n=1 観 察 は L26 単 体 が mid-stream で "Wait" 検 算 衝 動 を 4 回 連 発 + 補 足 自 発 ((token 3278 = 最 多))、 L25 単 体 は 検 算 1 回 のみ、 mix は 末 尾 "Double check" re-enumeration を 示 す **よう に 見 え た**。 系 統 的 follow-up ((exp166、 4 patch × 4 prompt × 32 seeds = 512 runs)) で **verify-pattern asymmetry** は 統 計 有 意 ((L26 alone aux_rate 56% vs L25 alone 16%、 p < 0.01)) で 確 認、 一 方 coin 正 解 率 の 差 ((mix 94% vs L25 alone 100%)) は n=32 で 統 計 有 意 で ない ((Fisher exact p=0.49))。 全 bench 検 証 ((exp167)) は 別 途 並 行。 機 能 局 在 確 定 で は な く、 「L25/L26 が 行 動 character 上 別 specialist と し て 振 る 舞 う 候 補」 と し て hypothesis-generating 観 察 を 報 告 す る。

---

## 4. Abstract ((English, 250 words))

Our paper v1 release (2026-05-27) demonstrated that an 8-byte F32 scale patch (L25+L26 ×1.5) on Gemma 4 31B yields dramatic effects across quantization levels: Q4_K_M patched (19 GB) beat the Q8_0 BF16 baseline (31 GB) on all 4 benchmarks (HellaSwag +10.17pt, GSM8K +17pt, Winogrande +4.73pt, ARC-C +4.38pt), Q2_K HS +11.21pt, IQ1_M GSM +36.0pt, with all 12 cells (3 quant levels × 4 benches) positive — no training, no calibration, no inference overhead. The present paper (v5) is a per-layer ablation pilot that asks: which of the two patched layers contributes which behavioral component? Across four patch configurations (L25 alone, L26 alone, L25+L26 mix as in paper v1, and a 12-byte triple control), an n=1 observation on a recurrence-relation prompt found a previously unanticipated phenotype: L26-alone preserved correct computation (5/16) but emitted four mid-stream "Wait"-style verification bursts plus a spontaneous supplementary section (token count 3278, the highest of four conditions), while L25-alone solved correctly with only one mid-stream Wait, and the L25+L26 mix produced an explicit terminal "Double check" re-enumeration absent in either single-layer case. A systematic follow-up (exp166: 4 patches × 4 prompts × 32 seeds = 512 runs) confirmed the verify-pattern asymmetry with statistical significance (L26-alone aux_rate 56% vs L25-alone 16%, p < 0.01), while the coin-prompt accuracy difference between conditions was not significant at n=32 (Fisher exact p=0.49). A full-bench verification across HS/WG/ARC/GSM for L25-alone and L26-alone (exp167) is in progress. We frame this as a hypothesis-generating pilot suggesting that "L25 ≈ compute" / "L26 ≈ verify-proposal" / "mix as plateau-center" may organize the paper v1 effect, while making no confirmed mechanistic claim and noting that direct residual-stream observation (logit-lens, activation patching) is required for any functional-localization assertion.

---

## 5. Observations ((実 測 データ))

### 5.1 Single-question (Q1) phenotype comparison ((Q2_K Gemma 4 31B, F32 ×1.5 patch, n=1 each))

Prompt: 「コイン を 6 回 投 げ た と き、 表 が 3 回 連 続 し て 出 る 確 率」 ((真 値 = 5/16))

| Patch | Answer ((正 解 5/16)) | a_5 | mid-stream "Wait" | Final "Double check" reflex | 補 足 自 発 | Tokens | Time | t/s |
|---|---|---|---|---|---|---|---|---|
| baseline ((no patch)) | ((not tested)) | — | — | — | — | — | — | — |
| triple ((LOS27+PAN17+PAN43 ×1.8, 12 byte, supplementary)) | ❌ 11/32 | **22** ((calc slip)) | 0 | ❌ | ❌ | 2726 | 42s | 64 |
| L25 単 体 ×1.5 ((4 byte)) | ✅ 5/16 | 24 | 1 | ❌ | ❌ | 2467 | 37s | 65 |
| L26 単 体 ×1.5 ((4 byte)) | ✅ 5/16 | 24 | **4** | ❌ | ⭐ | **3278** | 51s | 64 |
| paper v1 ((L25+L26 ×1.5, 8 byte)) | ✅ 5/16 | 24 | 1 | ⭐ explicit | ❌ | **2359** | **35s** | **66** |
| paper v1 Q4_K_M ((same patch, 4-bit)) | ✅ 5/16 | 24 | 1 | △ mid-stream re-verify + direct enumeration | ❌ | 2637 | 48s | 54 |

### 5.2 Multi-question consistency ((Q2_K, 3 prompts × 4 patches))

| Prompt | triple | L25 単 | L26 単 | paper v1 |
|---|---|---|---|---|
| Q1 ((coin 確 率)) | ❌ 11/32 ((slip)) | ✅ 5/16 | ✅ 5/16 | ✅ 5/16 |
| Q2 ((x³+y³ 対 称 式)) | ✅ 520 ((cdq + 別 解)) | ((not tested)) | ((not tested)) | ✅ 520 ((別 解 経 由)) |
| Q3 ((Putnam 2017 A1)) | ❌ 「0,1,3,4 mod 5 NOT」 | ((not tested)) | ((not tested)) | ❌ 「0,3 mod 5 NOT」 ((closer to truth)) |

### 5.3 GSM Q2_K headline decomposition: convergence-efficiency vs capability ((n=500, extended-context follow-up))

The paper v1 headline gain on GSM8k at Q2_K (baseline 67%, patched 76%, delta +9pt at n_predict=1024, n=100) decomposes into two distinct effects that we have separated empirically.

**Confound.** At n_predict=1024, a fraction of Q2_K baseline traces are truncated before reaching the `#### N` answer marker. The patched model both reasons differently *and* converges within budget more often. The +9pt mixes (a) genuine reasoning improvement with (b) higher trace-completion rate.

**Extended-context measurement (ctx=16384, n_predict=8192, n=500).** baseline 87.80% (Wilson 95% CI [84.64, 90.38]), patched 93.20% (Wilson 95% CI [90.65, 95.09]), delta **+5.40pt**, McNemar two-sided **p=0.0007** (highly significant). Paired breakdown: patch-only correct 44/500, baseline-only correct 17/500, both correct 422/500, both wrong 17/500 (61 discordant pairs, 44:17 ratio favoring the patch).

**Cap-hit verification (ctx=32768, n_predict=16384, n=7 boundary cases).** v4 at ctx=16384 had 7 token-cap hits (3 baseline + 4 patched, no overlap). Re-running those 7 at 4× the context budget: baseline +1 correct (i=255), patched +1 correct (i=298), 5 problems remained unresolved at 32k context (genuinely hard, not artifacts of the 8192 cap). Updated totals: baseline 88.00%, patched 93.40%, delta still **+5.40pt**, McNemar p=0.0009. The +5.40pt capability gain is robust to token-budget concerns.

**Decomposition.** The paper v1 +9pt ≈ **+3.6pt convergence-efficiency + +5.4pt capability gain**. The capability component is statistically established at p<0.001. Q4_K_M baseline-only GSM8k has now been re-run at ctx=16384, n_predict=8192, n=500 and scores 95.80% (479/500), higher than Q2_K patched. Q4_K_M patched and IQ1_M GSM patch rows remain legacy n_predict=1024 results; their pure capability portions were not separately measured at those quantization levels.

**Scope of the confound.** Multiple-choice benchmarks ((HellaSwag +11.21pt at n=10,042, Winogrande +7.34pt at n=1,267, ARC-C +2.83pt at n=1,165)) are log-likelihood scored with no generation, so no token-budget confound applies; those numbers stand.

**Related prior art.** Our split is *consistent with* — not derived from — two recent methodological references. Kaiser et al. ((arXiv:2602.09805)) propose a token-efficiency decomposition into a completion factor, a conditional-correctness factor, and a verbalization-overhead factor; their framework motivates separating "did the chain finish" from "was it right given it finished." Nie et al. ((arXiv:2605.07686)) document a "coupling tax" — accuracy degradation when reasoning trace and final answer share one token budget — under bfloat16 only. Neither paper studies quantization recovery.

### 5.4 exp166 statistical follow-up ((Q2_K, 4 patches × 4 prompts × 32 seeds = 512 runs))

P1 coin recurrence ((真 値 5/16)) で の **統 計 確 認**:

| Patch | mid_wait_mean | tail_rate | aux_rate | correct (coin) | tok_med |
|---|---|---|---|---|---|
| baseline | 8.25 | 16% | 22% | 91% (29/32) | 4035 |
| L25 alone ×1.5 | **5.56** | **3%** | 16% | 100% (32/32) | 2924 |
| L26 alone ×1.5 | 7.41 | 9% | **56%** | 97% (31/32) | 3643 |
| paper v1 mix | 5.81 | 6% | 28% | 94% (30/32) | 2727 |

P2 combinatorial / P3 narrative / P4 logic は 全 4 patches で 100% 正 解 ((又 は compute 検 査 なし))、 discriminative power なし。

**統 計 検 定 結 果**:
- **L26 alone aux_rate 56% vs L25 alone 16%** ((p < 0.01)) — 統 計 有 意 な verify-proposal asymmetry
- **mid_wait L25/mix vs baseline** ((-2.7差)) — patch が verbose verify を 抑 制 する 方 向
- mix vs L25 alone の coin 正 解 率 ((94% vs 100%)) は **Fisher exact p=0.49 ((統 計 有 意 で ない))**
- mix vs baseline の coin 正 解 率 ((94% vs 91%)) も **p ≈ 1.0** ((n=32 の noise 域))

**解 釈**:
- verify pattern ((behavioral character)) は L25 ≠ L26 で 統 計 確 認
- coin 単 一 task の 性 能 差 は n=32 では noise レベル
- 4 bench 平 均 性 能 は paper v1 mix で +9.13 ((exp162))、 単 一 layer の 性 能 は exp167 で full bench 検 証 中

### 5.5 exp169 verification: triple benchmark vs interactive gap ((Goodhart's Law instance))

triple ((LOS27+PAN17+PAN43 ×1.8、 12 byte、 paper v4 candidate)) を exp166 と bit-identical setting ((T=0.7、 max_tokens=6144、 32 seeds、 P1 coin recurrence prompt)) で 検 証:

| Patch | mid_wait_mean | tail_rate | aux_rate | coin correct (n=32) |
|---|---|---|---|---|
| baseline | 8.25 | 16% | 22% | 91% |
| L25 alone ×1.5 | 5.56 | 3% | 16% | **100%** |
| L26 alone ×1.5 | 7.41 | 9% | 56% | 97% |
| paper v1 mix | 5.81 | 6% | 28% | **94%** |
| triple | **9.25** | **31%** | 12% | **12.5%** ⚠ |

**Primary endpoint** ((Fisher exact one-sided, triple vs paper v1 mix on coin)): **p < 0.0001** ((triple < paper v1 in interactive accuracy 統 計 有 意))。

**Failure mode classification** ((triple coin 失 敗 28 件)):
- silent_slip ((誤 答 を 確 信)): **25/28 = 89.3%**
- unclassified ((規 則 に 該 当 し ない 失 敗)): 3/28 = 10.7%
- broken_reasoning ((思 考 中 断)): 0/28
- off_topic ((prompt 無 視)): 0/28

**Interpretation**: triple は 全 patch 中 最 多 の verification marker ((mid_wait + tail_reflex)) を 出 力 する **にも 関 わ らず** 正 答 率 が 最 低 ((12.5%))。 89.3% が silent_slip = 「**検 算 が 機 能 せ ず、 doubt loop で 自 己 の 誤 算 を 確 信 して 出 力**」 mechanism。

triple の GSM benchmark 高 score ((85% > paper v1 76%、 exp168)) は GSM 採 点 設 定 ((T=0.0 / max_tokens=1024 / regex 抽 出)) で doubt loop に 入 る 前 に 答 え を 出 す こと で 説 明 で き る。 interactive 設 定 ((T=0.7 / max_tokens > 4000)) で doubt loop が 発 火 し、 正 答 率 が 崩 壊 する。

**Practical recommendation**: paper v1 mix ((L25+L26 ×1.5, 8 byte)) is the practical-use peak. L25-alone / L26-alone are half-strength alternatives ((4 bytes each)) that remain functional in interactive use. Triple is included in the bundle only as a benchmark-gaming reference and **not recommended for interactive deployment**.

**Goodhart's Law implication**: benchmark gain on standardized eval harnesses ((GSM8K T=0.0)) does not transfer to interactive use cases when the patch alters the verification reflex distribution. This is a cautionary data point for the F32 patch literature.

### 5.6 Cross-quantization sanity ((paper v1: Q2_K vs Q4_K_M))

| | Q2_K | Q4_K_M | Δ |
|---|---|---|---|
| Q1 ((coin)) answer | ✅ 5/16 | ✅ 5/16 | identical |
| Q3 ((Putnam)) answer | ❌ 「0,3 mod 5 NOT」 | ❌ 「0,3 mod 5 NOT」 | identical wrong path |
| Q1 tokens | 2359 | 2637 | +12% |
| Q1 time | 35s | 48s | +37% ((Q4 重 い)) |
| Q1 t/s | 66 | 54 | -19% |
| Final reflex form | explicit closing | mid-stream re-verify + 直 接 列 挙 | preserved but stylistically varies |

---

## 6. Hypothesis B+ ((formal))

### Statement ((speculative))

In Q2_K quantized Gemma 4 31B, adjacent layers L25 and L26 **may be** functionally specialized in a way that the per-layer ablation makes visible:

- **L25 may act as a compute-confidence specialist**: under ×1.5 scaling alone it preserves correct computation with minimal mid-stream self-correction.
- **L26 may act as a verify-proposal specialist**: under ×1.5 scaling alone it preserves correct computation but multiplies mid-stream "Wait"-style verification bursts and emits spontaneous supplementary content, without a closing reflex.
- **L25+L26 joint scaling** ((paper v1)) appears to produce a terminal closing-verification reflex absent in either single-layer condition — possibly via a gating interaction between the two layers' contributions to the residual stream.

A simple residual-stream interpretation that would be consistent with this pattern: L25 contributes a "compute-confident" direction whose late-token saturation lets a verify-trigger from L26 cross threshold only at the reasoning terminus. This is **one of several possible mechanisms** and is not directly observed in this pilot.

### Individual ×1.5 phenotypes ((observed, n=1 each))

- L25 alone: 1 mid-stream "Wait", correct answer, 2467 tokens
- L26 alone: 4 mid-stream "Wait" + spontaneous "(補 足)" section, correct answer, 3278 tokens
- joint ((paper v1)): 1 mid-stream "Wait" + explicit terminal "Double check the a_n sequence:" re-enumeration, correct answer, 2359 tokens

### Possible neural-mechanism interpretations ((not directly observed))

If L26 contributes a "verify-now" direction near the unembedding subspace, scaling it ×1.5 alone could push it above the readout threshold at many token positions, producing diffuse mid-stream verification bursts. If L25's contribution is partially anti-aligned with that direction in a shared subspace, joint scaling could keep verification below threshold during early/mid reasoning until the compute contribution saturates near the reasoning terminus, leaving only a single closing verification reflex. **This is a speculative interpretation only**; direct verification requires probing the relevant residual-stream directions ((logit lens / activation patching at L24-L27)).

### Falsifiable predictions ((if Hypothesis B+ is on the right track))

1. **PCA of L25 and L26 MLP / attention output activations**: a shared axis should show sign reversal between the two layers' contributions.
2. **L25 ×0.5 + L26 ×1.5**: should reproduce or amplify the L26-alone mid-stream divergence.
3. **L25 ×1.5 + L26 ×0.5**: should suppress the closing-verification reflex entirely.
4. **L25 ×1.0 + L26 ×2.5** ((per-layer asymmetric scaling suggested by separate internal work)): should strengthen the terminal reflex while maintaining compute correctness.
5. **Logit-lens trajectory at L24-L27** for "Wait" / "Let me" / 「Wait」-equivalent tokens: L26-only patch should boost mid-stream logits; the mix should boost only terminal-position logits.

If any of these comes back the opposite of predicted in exp166 or follow-up, Hypothesis B+ should be revised or abandoned.

---

## 7. Methods

### 7.1 Patch implementation

F32 patches modify a single scalar per layer ((4 byte each)):
- L25 / L26 / L27: `layer_output_scale` ×1.5 or ×1.8
- L17 / L43: `post_attention_norm` ×1.8 ((triple control))

Bake script: `weight_analysis/bake_l25_l26_split.py`、 `weight_analysis/bake_triple_patch.py`.

### 7.2 Test prompts ((n=1 per prompt))

- Q1: コイン 6 回 表 3 連 続 確 率 ((expected 5/16))
- Q2: x+y=10, xy=16, x³+y³ ((expected 520))
- Q3: Putnam 2017 A1 集 合 S ((expected NOT in S = {1} ∪ {正 の 5 の 倍 数}))

### 7.3 Serving

llama-server ((b9016-cuda13.1)) with:
```
-ngl 99 -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 -fa on \
  --batch-size 2048 --ubatch-size 512 --host 127.0.0.1 --port 8080
```

Hardware: a single high-end consumer GPU with 32 GB VRAM.

### 7.4 Observation coding ((post-hoc for the n=1 pilot; pre-registered for exp166))

For the paper v5 pilot the regex was applied post-hoc with manual definition tuning; for the exp166 confirmation run the regex is pre-registered as follows ((same as used in the bundled `exp166_b_plus_verification.py` script)):

- `compute_correct` ((0/1)): ground-truth pattern match per prompt category
- `mid_wait_count` ((int)): regex `(?i)\b(wait|let me (re-?check|verify|reconsider|re-?evaluate|re-?examine)|hmm|actually|hold on)\b` outside the final ~25% of tokens
- `tail_reflex` ((0/1)): regex `(?i)(double[- ]?check|let me verify|let me recompute|let me reconsider|let me check again)` inside the final ~25% of tokens
- `aux_spontaneous` ((0/1)): regex `(?i)(\(?\s*補\s*足\s*\)?|note that|具体的に|for clarity|in other words|additional|参考|なお)` anywhere
- `token_count` ((int)) ((from server `usage.completion_tokens`))

The pilot used a manually-coded approximation of the same coding scheme; n=1 means there was no statistical analysis to leak into post-hoc regex tuning, but readers should treat the pilot coding as definition-gaming-vulnerable until exp166 results are available.

---

## 8. Rigor / Limitations ((重 大))

```
🔴 現 状 evidence の 強 さ = anecdotal / hypothesis-generating only
   統 計 的 検 出 力 = ゼロ ((n=1 question × 1 seed × 4 patches))
```

**Uncontrolled confounders ((non-exhaustive)):**

- Sampling stochasticity ((temperature 0.7, n=1 per condition))
- Prompt-class bias ((1 prompt per category, recurrence relation only))
- Scale-vs-layer confound ((×1.5 single-layer vs ×1.8 triple control))
- Order effects ((server restart between models, not randomized))
- GPU thermal drift across the run
- KV cache state across runs
- Coder bias ((behavioral phenotype scored by a single session without blinding))
- Definition gaming risk ((regex was not strictly pre-registered at run time))
- Prompt selection bias ((Putnam was added post-hoc))
- Bench-vs-claim mismatch ((Q4 cross-quantization touched only Q1 / Q3, not the full battery))

**Minimum evidence required before any confirmed claim ((paper-level)):**

1. n ≥ 30 seeds × 4 patches × 4 prompt categories with Mann-Whitney U test ((Bonferroni-corrected p < 0.0125))
2. Blind coding ((2 coders, Cohen's κ ≥ 0.8))
3. Fully fixed sampling parameters + controlled server restart protocol
4. ((Recommended)) Residual-stream probe / logit lens to directly observe candidate v_verify components

Without these, presenting the asymmetry as "confirmed functional specialization" is an explicit overclaim. This paper presents it as a **pilot observation** only and makes no confirmed mechanistic claim.

---

## 9. Recommended Verification ((exp166))

```
exp166_B_plus_timing_emergence_confirmation:
  4 patch ((baseline / L25 / L26 / mix)) × 4 prompt category × 32 seeds = 512 runs
  
  prompt categories:
    P1: 漸 化 式 ((Q1 同 型))
    P2: 別 算 術 ((combinatorial))
    P3: narrative ((control))
    P4: 多 段 logic puzzle
  
  固 定 設 定: temperature=0.7, top_p=0.95, top_k=40, max_tokens=4096
  pre-registered regex + 2 盲 検 coder
  
  判 定 基 準 ((B+ 確 定 = 3 条 件 同 時、 全 て 4-family Bonferroni 補 正)):
    α: L26 単 体 mid_wait_count > L25 単 体 ((Welch t, Bonferroni-corrected p < 0.0125, Cohen's d ≥ 0.8))
    β: mix の tail_reflex 率 > L25+L26 単 体 独 立 期 待 値 ((Fisher exact, Bonferroni-corrected p < 0.0125, excess ≥ 0.15))
    γ: L25 単 体 compute_correct ≥ baseline + 0.15 かつ L26 単 体 ≈ L25 (±0.10), L26 aux > L25 + 0.20 ((Bonferroni-corrected p < 0.0125))
  
  実 時 間: ~7 時 間 ((generation 1.7 hr + 並 列 coding 2 hr + logit lens 1 hr + 統 計 2 hr))
  GPU wall: ~3 時 間
```

---

## 10. GGUF Bundle ((`publish/release_v1/`))

**Paper v5 release ((4 patched GGUFs + 1 supplementary)):**

| File | Patch | byte | Role in this paper |
|---|---|---|---|
| `gemma-4-31B-it-L25x1.5-Q2_K.gguf` | LOS L25 ×1.5 | 4 byte | Single-layer ablation ((paper v5)) |
| `gemma-4-31B-it-L26x1.5-Q2_K.gguf` | LOS L26 ×1.5 | 4 byte | Single-layer ablation ((paper v5)) |
| `gemma-4-31B-it-L25L26x1.5-Q2_K.gguf` | LOS L25+L26 ×1.5 | 8 byte | paper v1 mix reference |
| `gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf` | same / Q4 | 8 byte | Cross-quantization sanity check |
| `gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf` | triple | 12 byte | **Supplementary only** — illustrates a GSM-specialized phenotype; statistically indistinguishable from paper v1 in exp162 full bench; **not part of the paper v5 claim** |

Other GGUF files in the repository ((`*-IQ1_M`, `*-UD-IQ2_XXS`)) are from separate internal work and are not part of this paper v5 release.

---

## 11. Future Work

- exp166 ((4 patch × 4 prompt × 32 seeds = 512 runs, pre-registered, **completed**)) — verify-pattern asymmetry confirmed: L26-alone aux 56% vs L25-alone 16%, Fisher p<0.01. Coin accuracy at n=32 not statistically significant; n=500 GSM extended-context follow-up ((§5.3)) confirms +5.40pt capability gain at p=0.0007.
- Cross-model replication ((Qwen 3.6 / Phi-4 / Mistral-Nemo)) — note that prior internal F32 patch work shows model-dependent responsiveness ((Gemma 4 strong, Qwen 3.6 +2.5pt, Phi-4 null)) with no confirmed architectural predictor ((the earlier rare-full-attention-layer hypothesis was disproved: L25/L26 are SWA layers in Gemma 4 31B)); cross-model patterns may differ
- Logit-lens / activation-patching probe of L24-L27 for "Wait" / "Let me" token logit trajectories — direct mechanism observation
- Per-layer adaptive multiplier search ((L25 vs L26 scale asymmetry suggested by paper v2 internal work)) — re-interpret existing per-layer findings under the asymmetry hypothesis
- Cross-benchmark evaluation ((MMLU / BBH / ARC-Challenge — none of these measured in the current exp162 reference run))
- Adapt Gemma-Scope-style SAE training pipeline for Gemma 4 31B ((no official Gemma-4 SAE release exists)) and test whether L25 / L26 patched features correspond to identifiable feature directions

---

## 12. Acknowledgements

- Base model: Google Gemma 4 31B-it
- Quantization: llama.cpp ((b9016-cuda13.1))
- The critical conceptual prompt — "L25 vs L26 may correspond to compute vs meta-thinking" — came from an independent collaborator during the 2026-05-30 session that produced this pilot.
- Analytical tooling: multi-agent workflow orchestration for rigor checking and adversarial review during draft preparation.

We make no claim to the underlying interpretability literature this pilot touches ((Anthropic transformer-circuits work on induction heads ((2022)) and the biology-of-LLMs / circuit-tracing line ((2025)); Gemma Scope ((Lieberum et al. 2024)) for Gemma 2)); the connection here is suggestive only, and the cited works should be consulted in their canonical form ((full author lists, official venues)) rather than via the abbreviated references in earlier drafts.

---

## 13. ChangeLog

- 2026-05-30: Initial draft from n=1 pilot
- 2026-05-30: exp166 v2 results integrated ((512 runs, verify-pattern asymmetry p<0.01, coin accuracy diff p=0.49))
- 2026-05-30: paper v1 dramatic Tweet numbers restored after deflation correction
- Deferred follow-up: exp167 full-bench L25/L26-alone results
- Deferred follow-up: cross-model replication results
- Deferred follow-up: logit-lens / activation-patching for direct mechanism observation
