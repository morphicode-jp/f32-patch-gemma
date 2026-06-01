---
title: "Gemma 4 31B Q2_K の L25/L26 機 能 分 化 を 4 byte F32 patch で 観 察 した ((n=1 pilot 報 告))"
emoji: "🧠"
type: "tech"
topics: ["LLM", "量 子 化", "Gemma", "ローカル LLM", "interpretability"]
published: false
---

## TL;DR

- Gemma 4 31B を 2 bit 量 子 化 ((Q2_K)) した model に、 layer 25 と layer 26 の F32 scale 値 を ×1.5 する だ け の patch ((**4 byte/layer**)) を 適 用。
- **L25 だ け 強 化** すると 「計 算 で きる が 検 算 reflex 弱 い」
- **L26 だ け 強 化** すると 「計 算 で きる が 「Wait, 確 認 し よ う」 を **連 発** する meta-obsessive」
- **L25+L26 同 時** 強 化 すると 単 体 で は 見 ら れ な い 「**結 論 直 前 の 検 算 reflex**」 が 現 れる **よう に 見 え る** ((単 体 観 測 で は 観 察 さ れ ず))
- 同 じ paper v1 patch を Q4_K_M ((4 bit 量 子 化)) で 試 して も 同 様 の パターン が **観 察 さ れる ように 見 え る** ((quantization-robustness の **示 唆**、 ただ し n=1 で は 確 定 不 能))
- **ただ し n=1 single question pilot、 hypothesis-generating only**。 確 定 主 張 で は な い。

GGUF ((5 種 全 部)) + bake script + paper draft 全 て 公 開: ((HuggingFace link TBD))

---

## 背 景: paper v1 から の 派 生

2026-05-27 に paper v1 を 公 開 した。 「Gemma 4 31B の F32 weight 60 個 中 layer 25 と 26 の **layer_output_scale** ((1 weight = 4 byte)) を **×1.5** する だ け で、 全 quant level で 顕 著 な 効 果」 と い う finding。 公 開 時 の 主 要 数 値:

```
patch 総 size: 4 byte × 2 layer = **8 byte**、 訓 練 / calibration / 推 論 overhead **全 部 ゼロ**

⭐ Q4_K_M patched ((19 GB)) が Q8_0 BF16 baseline ((31 GB)) を **4 / 4 全 勝**:
   HellaSwag    73.50% > 63.33% ((+10.17))
   GSM8k        87.00% > 70.00% ((+17.00))
   Winogrande   70.32% > 65.59% ((+4.73))
   ARC-C        48.76% > 44.38% ((+4.38))
   ((サイズ も Q4 patched が **約 40% 小**))

⭐ Q2_K HellaSwag +11.21pt
⭐ IQ1_M ((1 bit、 9.5 GB)) GSM 24% → 60% ((+36 pt = 12 cell 中 最 大))
   = 3 quant × 4 bench の **12 cell 全 部 positive**

exp162 reference rerun ((本 paper v5 release 用、 限 定 N: HS 1000/WG 500/ARC 200/GSM 100)):
   Q2_K patched ((paper v1)) 平 均 +9.13 pt ((HS+11.30, WG+7.20, ARC+9.00, GSM+9.00))
   → 公 開 時 fullN 値 より 小 さ め に 出 る ((小 N variance 由 来))、 方 向 性 は 一 致
```

この paper v1 の dramatic な 効 果 を **個 別 layer に 分 解** ((L25 だ け / L26 だ け)) して 「どこ が どう 効 い て いる の か」 を 観 察 した の が 本 稿 の pilot。

---

## 観 察: 4 model phenotype 比 較

同 じ プロンプト 「コイン を 6 回 投 げ て 表 が 3 回 連 続 する 確 率」 ((真 値 5/16)) を 4 model で 試 用 した ((n=1 each)):

| Patch | 答 え ((正 解 5/16)) | mid-stream "Wait" | 末 尾 検 算 reflex | 補 足 自 発 | tokens |
|---|---|---|---|---|---|
| triple ((LOS27+PAN17+PAN43 ×1.8、 12 byte)) | ❌ 11/32 ((計 算 ミス)) | 0 | なし | なし | 2726 |
| **L25 単 体 ×1.5** ((4 byte)) | ✅ 5/16 | 1 | なし | なし | 2467 |
| **L26 単 体 ×1.5** ((4 byte)) | ✅ 5/16 | **4 ((多 い))** | なし | ⭐ あり | **3278 ((最 多))** |
| **L25+L26 ×1.5** ((paper v1、 8 byte)) | ✅ 5/16 | 1 | ⭐ **explicit** | なし | **2359 ((最 少))** |

### 注 目 ポイント

1. **L25 単 体** と **L26 単 体** で **行 動 phenotype が 明 確 に 違 う**:
   - L25 単 体 = 1 回 だ け 「Wait!」 mid-stream self-correct ((軽 い))
   - L26 単 体 = 「Wait, let me re-verify the logic」 「Wait, let me double check f(n) calculation」 等 を **4 回 連 発**
   - L26 単 体 だ け 「**(補 足) 具 体 的 に どの よ う な ケース**」 と 自 発 的 に 説 明 追 加

2. **paper v1 ((L25+L26))** だ け が 「**Double check the a_n sequence:**」 を 末 尾 に 明 示 し、 a_1 〜 a_6 を 全 列 挙 で 再 検 算 する canonical reflex を 出 す ((単 体 で は emerge し な い))

3. paper v1 は **token 数 最 少** ((2359)) = 「**いつ 検 算 する か**」 の 判 断 が 効 い て いる = 過 剰 検 算 を 抑 え られ て いる

---

## 仮 説 ((Hypothesis B+))

```
L25 = 「compute-confidence specialist」
       残 差 流 に v_compute ((計 算 confident 方 向)) を 注 入

L26 = 「verify-proposal specialist」
       残 差 流 に v_verify ((「Wait」 token を 提 案 する 方 向)) を 注 入
       ↑ 自 己 抑 制 機 構 を 持 たな い ((mid-stream で diffuse))

v_compute と v_verify は **部 分 anti-aligned** ((内 積 < 0))

L25+L26 連 携 ×1.5:
   compute 残 差 飽 和 を 待 って 末 尾 で の み v_verify が 閾 値 超 過
   → **emergent timing controller**
```

= 「**機 能 が 別 の layer に 局 在 して いて、 同 時 強 化 で 新 し い behavior が emerge する**」 と い う 解 釈。

---

## さら に: Q4_K_M でも 同 じ パターン

同 じ paper v1 patch ((L25+L26 ×1.5)) を Q4_K_M に 適 用 して 同 じ 質 問 を 試 した:

| | Q2_K paper v1 | Q4_K_M paper v1 |
|---|---|---|
| 答 え ((Q1 coin)) | ✅ 5/16 | ✅ 5/16 |
| 計 算 ミス | なし | なし |
| 末 尾 reflex | explicit "Double check" | mid-stream "let me double check" + 直 接 列 挙 attempt |
| token | 2359 | 2637 ((+12%)) |
| 時 間 | 35s | 48s ((+37%)) |
| 速 度 | 66 t/s | 54 t/s ((-19%、 Q4 重 い)) |

**観 察**: **2 bit vs 4 bit ((4 倍 の 精 度 差))** で も 同 prompt の 答 え + 検 算 行 動 の 「型」 は 一 致 する **よう に 見 え る**。 = 「**cognitive structure は weights 精 度 に 依 存 しな い**」 という hypothesis と 整 合 ((n=1 で 確 定 で は な い))。

---

## 重 大 な caveat ((必 読))

```
🔴 これ は n=1 single-question pilot observation。 確 定 主 張 で は な い。

🔴 11 種 の 交 絡 ((sampling stochasticity、 prompt 偏 り、 scale-vs-layer confound 等))
   未 統 制。

🔴 paper として 確 定 する なら exp166 ((4 patch × 4 prompt × 32 seeds = 512 runs))
   が 必 要 ((現 在 並 行 で 実 行 中、 2 週 で v5.1 update 予 定))

🔴 cross-model 一 般 性 ((Qwen 3.6 / Phi-4)) は 未 検 証
```

---

## なぜ 公 開 する か ((hypothesis-generating として))

1. **再 現 障 壁 ゼロ**: 5 GGUF + bake script + 1 命 令 で 試 用 可
2. **誰 か が 検 証 / 反 証** できる: 公 開 する こと で 仮 説 が ablated される 可 能 性
3. **priority claim** ((2026-05-30 timestamp)): 後 続 研 究 で 引 用 可 能
4. **Anthropic / DeepMind interpretability community** に 直 接 query
5. **n=1 で confident 主 張 し ない こと** で intellectual honesty 維 持

---

## 公 開 artifacts

- **HuggingFace**: paper v1 mix を3 quantで配布 — [IQ1_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) / [Q2_K](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) / [Q4_K_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M)
- **GitHub**: https://github.com/morphicode-jp/f32-patch-gemma — bake script + reproduction code
- **Paper (Zenodo)**: https://doi.org/10.5281/zenodo.20362821 — paper本体 + 再現用artifact
- **次 update**: exp166 結 果 ((2 週 後 予 定))

---

## 試 し て み た い 方 へ

```powershell
# llama.cpp で 起 動
& "$env:USERPROFILE\llm\llama-b9016-cuda13.1\llama-server.exe" `
  -m "gemma-4-31B-it-L26x1.5-Q2_K.gguf" `
  -ngl 99 -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 -fa on `
  --host 127.0.0.1 --port 8080
```

ブラウザ で http://localhost:8080 を 開 い て、 上 の Q1 ((コイン)) を 投 げ て み て く だ さ い。 L26 単 体 で 「Wait, let me ...」 が 連 発 する ((multiple times)) こと が 観 察 でき れ ば、 paper v5 の phenotype が 再 現 さ れ た こと に な り ます。

---

## 関 連 / 引 用

- paper v1 ((2026-05-27 release)): Gemma 4 31B 向 け F32 scale patch 初 出
- Gemma Scope ((Lieberum et al. 2024)): **Gemma 2 用** SAE 公 開 ((Gemma 4 用 SAE は 公 式 release な し、 本 稿 仮 説 は その style の adapt が future work と い う 位 置 付 け))
- Anthropic transformer-circuits: induction heads ((2022)) と biology of LLM ((2025)) line を 参 考 ((本 稿 で は specific author 引 用 は 避 け 一 般 form で 言 及))

---

## ⚠ Practical Use Note ((重 要、 benchmark vs 実 用 の gap))

exp169 ((triple × 4 prompt × 32 seeds、 exp166 と 同 一 設 定)) で 驚 く べ き Goodhart's Law 例 を 観 察:

| Patch | mid_wait | tail_reflex | aux | coin 正 答 率 ((n=32)) |
|---|---|---|---|---|
| baseline | 8.25 | 16% | 22% | 91% |
| L25 alone | 5.56 | 3% | 16% | **100%** |
| L26 alone | 7.41 | 9% | 56% | 97% |
| paper v1 mix | 5.81 | 6% | 28% | **94%** |
| triple ((paper v4 candidate)) | **9.25** | **31%** | 12% | **12.5%** ⚠ |

triple は **検 算 ((mid + tail)) を 全 patch 中 最 多** で 出 す 一 方、 **正 答 率 は 12.5%** ((Fisher exact p < 0.0001 で paper v1 と 統 計 有 意 差))。 失 敗 28 件 中 **89.3% が silent_slip ((確 信 持 って 違 う 答 え を 出 す))**。

triple の GSM benchmark 85% は GSM 採 点 設 定 ((T=0.0 / max_tok=1024 / regex 抽 出)) で の 結 果 で あ り、 doubt loop に 入 る 前 に 答 え を 出 す か ら 強 い。 interactive 設 定 ((T=0.7 / max_tok=6144)) では doubt loop に 入 って 正 答 率 が 崩 壊 する。

**実 用 推 奨: paper v1 mix ((L25+L26 ×1.5、 8 byte))**。 L25/L26 単 体 ((各 4 byte)) は half-strength な が ら interactive で も 機 能。 triple は 「benchmark-gaming phenotype」 の 再 現 用 reference として のみ 含 む。

---

## まとめ

```
✅ 観 察 ((n=1 → n=32)): L25/L26 で 行 動 phenotype が 異 な る ((aux 56% vs 16%、 p<0.01))
✅ paper v1 mix = 真 の 実 用 best ((coin 94%、 GSM/HS/WG/ARC で peak))
✅ triple = benchmark gaming phenotype ((interactive で 12.5%、 silent_slip 89%))
✅ Q2 / Q4 で paper v1 同 一 答 え + 検 算 型 一 致
🔴 n=32 は P1 coin のみ ((P2/P3/P4 は 全 patch 飽 和))、 cross-model 検 証 未 実 施
📦 4 patched GGUF + supplementary triple + script 全 公 開、 反 証 大 歓 迎
```

質 問 や 反 証 ある 方 は HuggingFace discussion / GitHub issue / Twitter ((TBD)) で フィードバック お 願 い します。 特 に Anthropic / DeepMind の mechanistic interpretability 研 究 者 の 方 が この 観 察 を 「既 知 か 新 規 か」 判 断 して く だ さる と 助 か り ます。
