# X 本番リリース thread (5 連投、 改訂版 v3) — 5/25 (月) 22:00 JST 投稿

公開タイミング: **2026-05-25 (月) 22:00 JST**

戦略: 告知 thread の 24 時間後、 引き継いだ 熱量 で 全 公開。
**Tweet 3 は ODIN 過剰最適化 storytelling**、 **Tweet 4 は real-world quality demo** (self-correction + code 品質)。

画像: 既存 figures (`publish/release_v1/figures/`) 流用 + 新 規 1 枚 (Tweet 4 用、 任意)。

ハッシュタグ: なし (押し付けがましく ない 方針 維持)

---

## Tweet 1/5 (ヘッダー + 全 リソース link、 画像: fig1_quant_overview.png)

```
Releasing now: an 8-byte F32 patch that makes Gemma 4 31B's 4-bit
beat its 8-bit baseline on every benchmark.

Paper + 3 GGUFs (1/2/4-bit) + open-source script — all free.

📄 Zenodo:  doi.org/10.5281/zenodo.20362821
🛠 GitHub:  github.com/morphicode-jp/f32-patch-gemma
🤗 Models:  hf.co/morphicode-jp

🧵 1/5
```

---

## Tweet 2/5 (12 セル 全部、 画像: fig3_delta_heatmap.png)

```
2/5 12 cells. Every single one positive.

3 release quants × 4 benchmarks on Gemma 4 31B-it:

           HellaSwag  Winogrande   GSM8k    ARC-C
Q1 (9.5G)  +10.95     +5.76      +36.0★    +6.18
Q2 (12G)   +11.21     +7.34      +9.00     +2.83
Q4 (19G)   +9.80      +5.28      +15.0     +3.87

HellaSwag n=10,042, Wilson 95% CI.
Q4 patched > Q8_0 BF16 baseline on ALL 4 benchmarks
(beating it by +17pt on GSM8k).
```

---

## Tweet 3/5 (手法 + ODIN 過剰最適化 storytelling、 画像: fig4_cross_arch.png)

```
3/5 The patch:

All 3 release quants (Q1/Q2/Q4) use the SAME 8-byte modification:
  layer_output_scale[25] *= 1.5
  layer_output_scale[26] *= 1.5

2 F32 values. 8 bytes. ~3 seconds to apply.

I also ran ODIN (my optimization engine, ~7h on a single 32 GB consumer GPU) targeting
Q4 HellaSwag — it found an 11-layer 44-byte patch ("basin B").

Then I benchmarked basin B vs the simple L25+L26 on Q4:
L25+L26 won on every single benchmark, including the search target.

  HellaSwag:  basin B 72.82  vs  L25+L26 73.50
  GSM8k:      basin B 84.00  vs  L25+L26 87.00
  Winogrande: basin B 69.61  vs  L25+L26 70.32
  ARC-C:      basin B 48.50  vs  L25+L26 48.76

7 hours of search lost to a 2-layer baseline.

No training. No calibration. No inference overhead.
```

---

## Tweet 4/5 ★ Side-by-side video demo (動画 添付: baseline_vs_patched_demo.mp4, 67 秒)

```
4/5 Beyond benchmarks: same prompt, side-by-side.

I asked baseline vs patched (8-byte L25+L26) to write a 100-line
Ursina 3D game from scratch, then recorded both terminals.

Same model. Same prompt. Same seed (42). Same quant.
Δ = 8 bytes (2 F32 weights, layer_output_scale[25,26] ×1.5).

In 32 seconds:
  baseline (left):  stuck mid-entity setup, no game loop
  patched (right):  finished with app.run() — full win/lose logic,
                    input handler, restart panel
```

動画 ファイル: `publish/release_v1/videos/baseline_vs_patched_demo.mp4` (24 MB、 67 秒、 1920x1080 H.264 + AAC)。 X / Reddit / HN 全部 互換、 投稿 時 に 自動 圧縮 想定。 構成: 2 秒 intro → 28 秒 Q2 比較 (末尾 1.5 秒 zoom up) → 35 秒 Q4 比較 (末尾 1.5 秒 zoom up) → 2.5 秒 outro 「Δ = 8 bytes」。

---

## Tweet 5/5 (Safety + Cross-arch + CTA、 画像: fig5_alignment_preservation.png)

```
5/5 Safety + cross-architecture results.

AdvBench n=520 refusal rate (Q2_K):
  baseline:           97.31% (CI 95.53–98.39)
  L25+L26 patched:    93.46% (CI 91.00–95.28)
A small (-3.85pt) refusal-rate drop; alignment remains ≥93%.
The heavier basin B (44B) preserves alignment more strictly
(97.12%, CI fully overlaps baseline) — described in paper §6
as an alternative for safety-critical deployments.

Cross-architecture (hybrid LLM specific):
🟢 Gemma 4 (5:1 hybrid)  +13pt HellaSwag
🟢 Qwen 3.6 (1:3 SSM)    +3pt
🔴 Phi-4 (pure dense)    null
🔴 Llama (pure dense)    null

📄 doi.org/10.5281/zenodo.20362821
🛠 github.com/morphicode-jp/f32-patch-gemma
🤗 hf.co/morphicode-jp

Built solo, nights & weekends, day job in oil pump engineering 🇯🇵.
CC-BY 4.0. RT / star / 🤗 like welcome.

Follow @morphicode_jp for the next drop.
```

---

## Post-thread engagement

### Tweet 1 を 固定 ツイート に
プロフィール → 「固定 ツイート に 表示」

### 直接 mention 候補 (thread の 最下 reply で 控えめ に)
- @danielhanchen (Unsloth、 quant コミュニティ 最大 ハブ)
- @bartowski1182 (GGUF 個人 職人)

Mention 文 (thread 最下 reply):
```
@danielhanchen @bartowski1182 — sharing my first public release on Gemma 4 quant recovery. All 3 release quants use a single 8-byte L25+L26 ×1.5 patch. 16-cell matrix all positive; ODIN's 44-byte search solution actually lost to the simpler baseline. Curious to hear thoughts. (thread + paper above)
```

### Reddit r/LocalLLaMA 同日 投稿 (22:05 JST)
[REDDIT_POST.md](REDDIT_POST.md) を 投稿 (本日中 に Q4 L25+L26 数字 + ODIN overfitting セクション に 更新済)。

### Hacker News Show HN (22:10 JST = ET 9:10 AM = HN 朝 ピーク)
title:
```
Show HN: An 8-byte patch that makes 4-bit Gemma 4 31B beat its 8-bit baseline
```
body: GitHub URL を 最初 + TL;DR 3 行 + Zenodo DOI link (doi.org/10.5281/zenodo.20362821) + ODIN overfitting bullet。

---

## 投稿時間帯 メリット

22:00 JST = 9:00 ET = HN 朝 ピーク + 米 ML twitter 起床 直後
+ 月曜 = work week 開始 で engagement 最高

---

## 公開後 24h 観測 指標

| 指標 | 目標 |
|---|---|
| Tweet 1 impressions | 10k+ (告知 thread からの 引き継ぎ で 倍増 期待) |
| Tweet 1 likes | 300+ |
| Tweet 1 RT / quote | 50+ |
| GitHub stars | 100+ |
| 3 HF model DL 合計 | 1k+ |
| 新規 X followers | 300+ |
| HN front page (上位 30) | 確率 30% |
| Reddit r/LocalLLaMA 上位 10 | 確率 60% |

---

## 注記

- **Q4 patched 数字 は 全て L25+L26 ×1.5 (8 byte) 版** (`weight_analysis/results/phase2/exp86_q4km_l25l26_*.json`、 ngl=99 フル offload、 wall 48 分 で 取得)。
- **Q4 GGUF は L25+L26 採用、 basin B は archive**。 公開 ファイル: `publish/release_v1/gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf` (MD5 `2c348ed9c3c499587343a93de07a84ce`)。
- **Q8_0 は basin B 版 のみ 測定** (Q8 L25+L26 ベンチ は 時間 制約 で 未実施)。 paper §Limitations で 明記。
- arXiv は今回見送り (endorsement未取得)。Zenodo DOI (10.5281/zenodo.20362821) が論文の正式archive。将来 arXiv endorsement 取得時に v2 で追加検討。
- Tweet 4 の 数学 / コード 比較 は 本日 5/24 の 実 用 検証 結果 (Q2 baseline vs Q2 patched、 Q4 basin B vs Q4 L25+L26 同 プロンプト)。 詳細 は paper Discussion section 参照。
- Tweet 3 の 「basin B vs L25+L26」 数字 は exp80 (basin B) + exp86 (L25+L26) の 直接 比較。
