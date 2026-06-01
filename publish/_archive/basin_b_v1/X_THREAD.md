# X (Twitter) 公開 thread draft (4 tweets)

公開タイミング:
- HF model upload 完了
- GitHub repo public
- arXiv preprint submit 完了 (or 同日)

ハッシュタグ統一: `#LLM #quantization #gemma #llama_cpp`

---

## Tweet 1/4 (主張 + 数字、最重要)

```
A 44-byte F32 patch improves Gemma 4 31B Q2_K HellaSwag accuracy by +13.25pt — no training, no calibration, no inference overhead.

baseline: 57.00%
patched:  70.25%

Wilson 95% CIs do not overlap.

Released today: HF model + apply script. 🧵 1/4
```

JP 訳 (別言語版として thread に追加 or 別 thread):
```
44 byte の F32 patch だけで Gemma 4 31B Q2_K の HellaSwag を +13.25pt 改善 — 訓練なし、calibration なし、推論オーバーヘッドなし。

baseline: 57.00%
patched:  70.25%

Wilson 95% CI 完全分離。

本日公開: HF モデル + 適用スクリプト。🧵 1/4
```

---

## Tweet 2/4 (mechanism + cross-model)

```
Why does it work?

Gemma 4 is a hybrid (5:1 full-attention : sliding window) LLM. The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales — conservatively constrained during training.

Tested 4 architectures:
- Gemma 4 31B (5:1 hybrid):  +13.25pt
- Qwen 3.6 27B (1:3 SSM):    +3.50pt
- Phi-4 14B (pure dense):    -2.31pt (destructive!)
- Llama / Mistral:           null

→ F32 slack is hybrid-architecture specific.

2/4
```

---

## Tweet 3/4 (safety + tooling)

```
Safety: alignment fully preserved (Wilson 95% CIs overlap baseline).

AdvBench refusal rate (n=520, full):
- baseline:        97.31% (506/520), CI [95.53, 98.39]
- basin B patched: 97.12% (505/520), CI [95.30, 98.24]

The 44-byte patch does NOT compromise RLHF guardrails.

Tools released:
- HF model: huggingface.co/morphicode_jp/gemma-4-31B-it-basin-b-Q2_K
- GitHub script: github.com/morphicode_jp/f32-patch-gemma
- 44 bytes, single Python file, no dependencies

3/4
```

---

## Tweet 4/4 (theory + next + branding)

```
Discovered by morphi, an autonomous optimization engine I built (12-specialist parallel search).

Coming next:
- arXiv preprint with full mechanistic analysis
- morphi engine OSS release
- per-tensor patch (sub-44-byte)

Built independently as a 油圧ポンプ engineer in 🇯🇵 spending nights & weekends on LLM research.

If this helps your work, RT/star/⭐ welcome.

4/4
```

---

## Post-thread engagement plan

### Pin この thread を固定ツイート化
- プロフィール → 「固定ツイートに表示」

### 直接 mention 候補 (1-2 人だけ、控えめに)
- @danielhanchen (LLM 量子化界の最大ハブ)
- @bartowski1182 (GGUF 個人職人、同類)

Mention 文面:
```
@danielhanchen @bartowski1182 — sharing my first public release on Gemma 4 quant recovery. Curious to hear thoughts whenever you have a spare moment. (thread above)
```

→ thread の reply に投稿、本 thread の URL 添える形

### Reddit r/LocalLLaMA 同日投稿
タイトル案:
```
[Release] 44-byte F32 patch improves Gemma 4 31B Q2_K HellaSwag by +13.25pt (no training, no calibration)
```

### Zenn 日本語版 (公開後 1-2 日以内)
タイトル案:
```
44 byte の F32 patch で Gemma 4 31B Q2_K の精度を +13.25pt 改善した話
```

---

## 投稿時間帯 (JST)

- **22:00-24:00 JST** = 米国 morning 9:00-11:00 EST
- AI Twitter の活発時間 = この slot
- Daniel Han / Bartowski がオンライン確率最高
- 平日 (火-木) 推奨、月曜は週末ノイズで埋もれる、金曜は週末モード

---

## 公開後 24h 観測指標

| 指標 | 目標 |
|---|---|
| Tweet 1 impressions | 1k+ |
| Tweet 1 likes | 50+ |
| Tweet 1 RT/quote | 5+ |
| GitHub stars | 20+ |
| HF model DL | 100+ |
| 新規 X followers | 50+ |

達成しなければ翌日 + 翌々日に追加 push (例: 図表追加投稿、解説 thread、reply 戦略)。
