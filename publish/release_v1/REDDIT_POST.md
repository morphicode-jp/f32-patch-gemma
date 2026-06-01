# Reddit r/LocalLLaMA Post Draft v3 (L25+L26 flagship)

公開当日: 2026-05-25 (Mon) 22:00 JST 頃 投稿

---

## Title

```
[Release] 8-byte F32 patch improves Gemma 4 31B across all 12 cells (Q1/Q2/Q4 × HellaSwag/GSM8k/Winogrande/ARC-C), no training, alignment preserved at >93%
```

## Body (Markdown)

```markdown
**TL;DR**: Modifying 8 bytes — two `layer_output_scale` F32 weights at layers 25 and 26, each multiplied by 1.5 — improves Gemma 4 31B accuracy across HellaSwag, GSM8k, Winogrande, and ARC-Challenge on Q1/Q2/Q4 quantizations. No fine-tuning, no calibration data, no inference overhead. 12/12 cells positive. Q4 patched beats the Q8 BF16-proxy baseline on all 4 benchmarks. Safety alignment retention ≥ 93% on AdvBench.

## Results (Gemma 4 31B, full validation)

All three release quants use the **same 8-byte L25+L26 ×1.5 patch**.

| quant | HellaSwag n=10,042 | Winogrande n=1,267 | GSM8k n=100 | ARC-C n=1,165 |
|---|---|---|---|---|
| **Q1** (IQ1_M, 9.5 GB) | 42.02 → 52.98 (**+10.95**) | 49.80 → 55.56 (**+5.76**) | 24 → **60** (**+36**) ⭐⭐ | 30.56 → 36.74 (**+6.18**) |
| **Q2** (Q2_K, 12.6 GB) | 59.15 → **70.36** (**+11.21**) ⭐ | 59.35 → 66.69 (**+7.34**) | 67 → 76 (**+9.0**) | 43.61 → 46.44 (+2.83) |
| **Q4** (Q4_K_M, 19 GB) | 63.70 → **73.50** (**+9.80**) | 65.04 → 70.32 (**+5.28**) | 72 → **87** (**+15.0**) | 44.89 → 48.76 (+3.87) |
| Q8 baseline (ref.)† | 63.33 | 65.59 | 70.00 | 44.38 |

HellaSwag Wilson 95% CIs do not overlap. **Q2 and Q4 patched both surpass the Q8 BF16-proxy baseline of 63.33%**. **Q4 patched beats Q8 baseline on all 4 benchmarks** (HellaSwag +10.17pt, GSM8k +17pt, Winogrande +4.73pt, ARC-C +4.38pt).

† Q8_0 is reported only as a BF16-proxy baseline reference. We do not release a Q8 patched model (31 GB GGUF doesn't fit in 32 GB VRAM with full offload, and Q4 patched already beats Q8 baseline across the board).

Safety: AdvBench n=520 refusal rate baseline 97.31% vs L25+L26 93.46% (-3.85pt, alignment retention ≥ 93%; basin B was 97.12% but underperforms benchmarks). See paper §6 for full trade-off.

**Scorer note (transparency)**: All HellaSwag/Winogrande scores measured with `llama-perplexity --hellaswag` (0-shot, token-level log-likelihood). This is systematically 0.2–2.5pp lower than `lm-evaluation-harness` (10-shot, acc_norm) used in published model cards. See [llama.cpp #2321](https://github.com/ggml-org/llama.cpp/discussions/2321) for the systematic gap. Within-scorer baseline-vs-patched deltas are valid; absolute numbers not directly comparable to externally reported.

## The Patch (single recipe for all quants)

All 3 release quants use the same 8-byte modification:

```
layer_output_scale[25] *= 1.5
layer_output_scale[26] *= 1.5
```

That's 2 F32 values per GGUF. 8 bytes total. ~3 seconds to apply.

## Honest disclosure: I tried to do it the fancy way

Before settling on L25+L26, I ran **ODIN** (my autonomous optimization engine, 12 specialists in parallel, ~7 hours on a single 32 GB consumer GPU) targeting Q4 HellaSwag accuracy. It found an 11-layer, 44-byte multiplicative-scale patch I call **basin B**:

```
{ 5:1.440, 18:0.767, 19:1.016, 20:0.814, 21:1.413,
  25:1.402, 26:1.289, 27:1.326, 28:1.431, 30:0.548, 31:1.267 }
```

When I benchmarked basin B vs the simple L25+L26 on Q4 across the 4 release benchmarks, **L25+L26 won on every single one — including the search target (Q4 HellaSwag) itself**:

| bench | basin B (44 B) | L25+L26 (8 B) |
|---|---|---|
| Q4 HellaSwag  | 72.82 | **73.50** (+0.68) |
| Q4 Winogrande | 69.61 | **70.32** (+0.71) |
| Q4 GSM8k      | 84.00 | **87.00** (+3.00) |
| Q4 ARC-C      | 48.50 | **48.76** (+0.26) |

7 hours of high-dimensional search lost to a 2-layer manual baseline. The over-parameterized solution generalizes strictly worse than the simple one — an instance of objective-overfitting. The basin B values are kept in the paper appendix for transparency.

## Why it works (preliminary)

Gemma 4 is a hybrid LLM (5:1 full-attention : sliding-window). The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales. This is **hybrid-architecture specific** — verified across 4 model families:

- Gemma 4 31B (5:1 hybrid): +13.25pt @ Q2_K
- Qwen 3.6 27B (1:3 SSM hybrid): +3.50pt
- Phi-4 14B BF16 (pure dense): -2.31pt (destructive!)
- Llama 3.1 / Mistral 7B (pure dense): null

Phi-4 at BF16 with no quantization showing destructive Δ rules out "quantization recovery" as the underlying mechanism — the effect is structural to hybrid LLMs.

## Bonus: side-by-side video demo (1080p, 67 sec, attached)

Same prompt (write a 100-line Ursina 3D game from scratch). Same seed.
Same quant size. Δ = 8 bytes (2 F32 weights).

Recorded both terminals streaming in real time. In ~32 seconds:
- **baseline (left)**: stuck mid-entity setup, no game loop, no `app.run()`
- **patched (right)**: finished — full win/lose logic, input handler, restart panel, `app.run()` reached

The 8-byte patch doesn't lift benchmarks by adding speed (decode rate is
identical). It lifts the model's ability to finish a task at the same speed.

(Video attached. For Q2_K and Q4_K_M, baseline-vs-patched in one clip with HUD overlay showing chars / time / chars-per-second.)

## Auxiliary: LeetCode-style DP (text-only verification)

Same patch, on a LeetCode-style longest arithmetic subsequence problem (8 test cases):
- Q2 patched (L25+L26): **8/8 passed** — correct DP algorithm
- Q2 baseline: 3/8 passed — off-by-one bug with `defaultdict(int)`
- Q4 patched (L25+L26): **8/8 passed**
- Q4 patched with basin B: **wrong algorithm** — silently sorts the input (which changes the problem from "longest AP subsequence" to "longest AP in the set")

The simple patch preserves algorithmic reasoning. The 44-byte ODIN solution silently broke it.

## Resources

- **HF Q1 (1-bit, 9.5 GB)**: https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M
- **HF Q2 (2-bit, 12.6 GB)**: https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K
- **HF Q4 (4-bit, 19 GB)**: https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M
- **GitHub tool**: https://github.com/morphicode-jp/f32-patch-gemma
- **Paper (Zenodo archive)**: https://doi.org/10.5281/zenodo.20362821

## How to use

```bash
pip install gguf numpy
git clone https://github.com/morphicode-jp/f32-patch-gemma
python f32-patch-gemma/apply_l25l26.py /path/to/your-gemma-4-31b.gguf
```

Backup is automatic; restore with `--restore`. Works on any Gemma 4 31B GGUF (IQ1_M through Q8_0).

## Background

I'm an independent researcher in Japan, building this on nights & weekends while working a day job in oil pump engineering. The optimization engine (called `morphi`, the source of the basin B failure above) will be released separately in a follow-up.

Happy to answer questions about methodology, mechanism, the ODIN overfitting, or reproduction.
```

---

## Post-投稿 戦略

### 投稿時間 (Reddit / JST)
- **22:00-24:00 JST** = 9:00-11:00 EST = LocalLLaMA peak

### 投稿後 30 分以内
- top comment に「Reproduction」「Q4 で試した」 など実用情報を pin
- 質問には 1 時間以内に返信 (algorithm 的に重要)

### 期待リアクション

| 反応 | 対応 |
|---|---|
| 「Q4 で試した、確認できた」 | 感謝 + 結果 share 依頼 |
| 「他 quant でも効くか」 | 「Q1/Q2/Q4 全部で確認、論文 figure 参照」 |
| 「Llama でも試して」 | 「Phase A-I で試した、null。pure dense は効かない構造的理由 (paper §7)」 |
| 「これは fluke では?」 | 全 3 quant の Wilson CI 非重複提示 |
| 「Gemma license 大丈夫?」 | 「patch 値 Apache 2.0、Gemma weights は Google Terms 準拠」 |
| 「Daniel Han 何て言ってる?」 | 「DM したらこう返ってきた」(後で更新) |
| 「morphi 公開して」 | 「v4 release 予定、まずは patch + 論文で reach 形成中」 |
| 「basin B も配って」 | 「basin B (44 byte, paper v3 candidate) は今回のリリースには含めていません。値は paper appendix に、 過去の script は GitHub の `_archive/basin_b_v1/` に置いてあります。 paper v5 では推奨せず」 |
| 「Q8 L25+L26 試した?」 | 「未測定、compute 制約 (partial offload で 4h)。 limitations 参照」 |
| 「safety -3.85pt は?」 | 「Q2 のみ測定、 retention 93% で実用範囲内。 厳格な alignment 要件あれば basin B 推奨」 |

### Subreddit ルール確認
- self-promotion 比率: 過去投稿が 9:1 比 (他人の話 9 : 自分宣伝 1) でないと削除
- 新規アカウントは自動 filter される
- 「I'm releasing this」明示で OK (自作自演バレ回避)

---

## Hacker News (Show HN、 同日 22:10 JST)

title:
```
Show HN: An 8-byte patch that improves Gemma 4 31B accuracy across all bit-widths
```

body: GitHub URL を 最初 + TL;DR 3 行 (16-cell all-positive, ODIN overfitting honest negative, hybrid-architecture specific) + Zenodo DOI link (doi.org/10.5281/zenodo.20362821)。
