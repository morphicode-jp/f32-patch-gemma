# Reddit r/LocalLLaMA Post Draft

公開当日に投稿。フォーマットは LocalLLaMA の community 慣習に合わせる。

---

## Title (一行のみ)

```
[Release] 44-byte F32 patch improves Gemma 4 31B Q2_K HellaSwag by +13.25pt (no training, no calibration, no inference overhead)
```

代替 title (もう少しキャッチー):
```
I found a 44-byte patch that boosts Gemma 4 31B Q2_K HellaSwag by +13.25pt. Training-free, GGUF-compatible, alignment-preserved.
```

## Body (Markdown)

```markdown
**TL;DR**: Modifying 11 specific `layer_output_scale` F32 weights in Gemma 4 31B (44 bytes total) improves HellaSwag accuracy by +13.25pt on Q2_K. No fine-tuning, no calibration data, no inference overhead.

## Results (Gemma 4 31B Q2_K, HellaSwag n=400)

| condition | accuracy | Wilson 95% CI |
|---|---|---|
| baseline Q2_K | 57.00% | [52, 62] |
| **basin B patched** | **70.25%** | **[65, 75]** |
| Δ | **+13.25pt** | CIs non-overlapping |

Safety preserved: AdvBench refusal rate 100% in both conditions (n=50).

## What is basin B?

A set of 11 multiplicative scales applied to F32 `layer_output_scale` weights in Gemma 4 31B:

```
{5: 1.440, 18: 0.767, 19: 1.016, 20: 0.814, 21: 1.413,
 25: 1.402, 26: 1.289, 27: 1.326, 28: 1.431, 30: 0.548, 31: 1.267}
```

Discovered via an autonomous optimization engine I built (12-specialist parallel search, 7 hours on RTX 5090).

## Why it works (preliminary)

Gemma 4 is a hybrid LLM (5:1 full-attention : sliding-window). The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales. This is **hybrid-architecture specific** — verified across 4 model families:

- Gemma 4 31B (5:1 hybrid): **+13.25pt**
- Qwen 3.6 27B (1:3 SSM hybrid): +3.50pt
- Phi-4 14B BF16 (pure dense): -2.31pt (destructive!)
- Llama 3.1 / Mistral 7B (pure dense): null

## Resources

- **HF model** (pre-patched, 12.6 GB): https://huggingface.co/morphicode_jp/gemma-4-31B-it-basin-b-Q2_K
- **GitHub tool** (apply to any Gemma 4 31B quant): https://github.com/morphicode_jp/f32-patch-gemma
- **arXiv preprint**: forthcoming

## How to use

```bash
pip install gguf numpy
git clone https://github.com/morphicode_jp/f32-patch-gemma
python f32-patch-gemma/apply_basin_b.py /path/to/your-gemma-4-31b.gguf
```

Backup is automatic; restore with `--restore`.

## Background

I'm an independent researcher in Japan, building this on nights & weekends while working a day job in oil pump engineering. The optimization engine that discovered basin B (called `morphi`) will be released separately.

Happy to answer questions about methodology, mechanism, or reproduction.
```

---

## Post-投稿 戦略

### 投稿時間 (Reddit / JST)
- **米国 morning slot**: 21:00-24:00 JST = 8:00-11:00 EST
- LocalLLaMA のピーク

### 投稿後 30 分以内
- top comment に「Reproduction」「Try on different quants」など実用情報を pin
- 質問には 1 時間以内に返信 (アルゴリズム的に重要)

### Subreddit ルール確認
- self-promotion 比率: 過去投稿が 9:1 比 (他人の話 9 : 自分宣伝 1) でないと削除
- 新規アカウントは自動 filter される → 数日前から community participation 推奨

### 過去アカウントなければ
- Twitter で先に thread 投稿 → 1 日待つ → Reddit に「I saw this on Twitter and thought you'd find it interesting」風で投稿しない (自作自演バレる)
- 代わりに最初から「I'm releasing this」明示で OK

## 期待リアクション

| 反応 | 対応 |
|---|---|
| 「Reproduce 試した」 | 感謝 + 結果 share 依頼 |
| 「他 quant でも効くか」 | 「同じ patch 適用で OK、結果 share してほしい」 |
| 「Llama でも試して」 | 「Phase 9-I で試した、null。詳細 arXiv で」 |
| 「これは fluke では?」 | Wilson CI + n=10042 結果提示 |
| 「Gemma license 大丈夫?」 | 「patch 値 Apache 2.0、Gemma weights は Google Terms 準拠」 |
| 「Daniel Han 何て言ってる?」 | 「DM したらこう返ってきた」(後で更新) |
