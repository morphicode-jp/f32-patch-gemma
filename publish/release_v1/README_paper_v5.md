# Paper v5 GGUF bundle — L25/L26 Functional Specialization + Practical Use Verification

**Date**: 2026-05-30
**Status**: pilot ((n=32 follow-up complete on P1 coin prompt))
**Predecessor**: paper v1 ((L25+L26 ×1.5、 8 byte))
  - public release demonstrated Q4 patched > Q8 baseline on all 4 benchmarks
  - 12/12 cell positive matrix across 3 quant levels
  - Q2 HS +11.21pt, IQ1 GSM +36pt
**Key finding ((exp169))**: triple ((paper v4 candidate)) wins GSM benchmark ((85%)) but **collapses to 12.5%** in interactive use ((Fisher p < 0.0001 vs paper v1's 94%)). silent_slip rate 89.3%. Goodhart's Law instance.
**Recommendation**: use **paper v1 mix ((L25+L26 ×1.5, 8 byte))** for interactive deployment. Triple is bundled only as a reference for the benchmark-gaming phenotype.
**Full draft**: `docs/PAPER_V5_DRAFT_2026-05-30.md`

---

## What this bundle is

Q2_K quantized Gemma 4 31B-it に F32 4-byte ((per layer)) patch を 個 別 適 用 した 5 種 の GGUF。

**main observation**: L25 alone と L26 alone で **行 動 phenotype が 異 なる**:
- **L25 ×1.5 単 体** → 計 算 正 確、 mid-stream "Wait" 1 回、 token 2467
- **L26 ×1.5 単 体** → 計 算 正 確、 mid-stream "Wait" **4 回**、 補 足 自 発、 token **3278 ((最 多))**
- **paper v1 ((L25+L26 mix))** → 計 算 正 確、 末 尾 explicit "Double check" reflex、 token **2359 ((最 少))**

→ 「**L25 ≈ compute, L26 ≈ verify-proposal、 mix で 末 尾 reflex の timing 制 御 が 現 れ る よう に 見 え る**」 と い う speculative Hypothesis B+。 ただし **n=1 pilot** で 確 定 で は な く、 mechanism は 直 接 観 測 して いない。

---

## 配布GGUF (HuggingFace上)

paper v1 patch (L25+L26 ×1.5、 8バイト) を3つの量子化レベルで配布:

| HuggingFace repo | Quant | Size | Highlight |
|---|---|---|---|
| [morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) | IQ1_M | ~10 GB | 1-bit revival、 GSM +36pt (paper v1) |
| [morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) | Q2_K | ~13 GB | paper v5 default、 HellaSwag +11.21pt (paper v1) |
| [morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M) | Q4_K_M | ~19 GB | Q4 patched が Q8 BF16 baseline を全 4 bench で打破 (paper v1) |

## per-layer ablation GGUF (配布なし、 bake script で再生成可)

paper本文で言及する L25単体 / L26単体 / triple (LOS27+PAN17+PAN43 ×1.8) は HuggingFaceに配布していません。GitHub repoのbake scriptで unmodified base GGUF から再生成できます:

- `bake_l25_l26_split.py` → `gemma-4-31B-it-L25x1.5-Q2_K.gguf` または `gemma-4-31B-it-L26x1.5-Q2_K.gguf` (4 byte each)
- `bake_triple_patch.py` → `gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf` (12 byte)

GitHub: https://github.com/morphicode-jp/f32-patch-gemma

これら per-layer 系を配布しない理由: paper本文で「triple は Goodhart's Law instance、 interactive で 12.5% に崩落」 と説明している通り (§Practical Use Note)、 配布するとreader が deployment patchと誤解するリスクがある。 paper v5 narrative の主張 (L25/L26 functional asymmetry) は paper v1 mix 3 repo を見れば説明済み、 per-layer ablationは「説明のための実験」 という位置付け。

---

## How to run

### llama.cpp / llama-server

```powershell
# 8080 で server 起 動 (32K context, Q8 KV)
& "$env:USERPROFILE\llm\llama-b9016-cuda13.1\llama-server.exe" `
  -m "gemma-4-31B-it-L25x1.5-Q2_K.gguf" `
  -ngl 99 -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 -fa on `
  --batch-size 2048 --ubatch-size 512 `
  --host 127.0.0.1 --port 8080

# ブラウザ で http://localhost:8080 を 開 く
```

### LM Studio

models フォルダ に .gguf を コピー。 「Local Server」 タブ で 選 択 起 動。

### Ollama

```
ollama create gemma4-31b-L25 -f Modelfile
# Modelfile:
#   FROM ./gemma-4-31B-it-L25x1.5-Q2_K.gguf
```

---

## Recommended user test ((paper v5 phenotype 確 認))

ブラウザ で 以 下 を 投 げる:

```
コイン を 6 回 投 げ た と き、 表 が 3 回 連 続 し て 出 る 確 率 を 求 め て く だ さ い。
((「途 中 に 出 て も OK」 で は な く 「最 低 1 回 連 続 3 回」 の 意 味))
```

期 待:
- L25 単 体: 計 算 正 確 ((5/16))、 mid-stream "Wait" 1 回
- L26 単 体: 計 算 正 確 ((5/16))、 mid-stream "Wait" **複 数 回**、 「(補 足)」 等 の 自 発 拡 張
- paper v1: 計 算 正 確、 末 尾 で "Double check" + 全 値 再 列 挙

「**L26 単 体 で 検 算 衝 動 が 多 い こと**」 が 観 察 で きれば paper v5 phenotype 再 現。

---

## Reproducibility

bake scripts は `weight_analysis/bake_l25_l26_split.py` と `weight_analysis/bake_triple_patch.py`。

Base GGUF: `google_gemma-4-31B-it-Q2_K.gguf` (llama.cpp標準量子化、HuggingFaceから取得)。

Patch適用のコアロジックは本リリースに含まれる bake scripts (`bake_l25_l26_split.py` / `bake_triple_patch.py`) に実装されています。CLI 使用例とパラメータの詳細は各スクリプトの docstring と `REPRODUCE.md` を参照してください。

---

## Important caveats ((必 読))

```
🔴 これ は n=1 single-question pilot observation。 確 定 主 張 で は な い。

🔴 「機 能 分 化」 は 観 察 された 行 動 phenotype の 解 釈 仮 説 ((B+))、
   neural mechanism は 直 接 観 測 して な い。

🔴 sampling stochasticity / prompt 偏 り / scale-vs-layer confound 等
   11 種 の confounder は 統 制 して ない。

🔴 paper として 確 定 主 張 する なら exp166 ((n=32 seeds × 4 conditions × 4 prompts = 512 runs))
   が 必 要。

🔴 cross-model 一 般 性 ((Qwen/Phi-4/Mistral)) は 未 検 証。
   F32 patch 研 究 既 知: Gemma 4 31B = 強 効 果、 Qwen 3.6 = +2.5pt、 Phi-4 = null。
   hybrid LLM の rare full attention 層 特 異 性 が key と 推 定 さ れる。
```

---

## License & Attribution

- Base weights: Google Gemma 4 31B-it ((Gemma 4 は 2026-05-31 確認時点で Apache 2.0))
- Quantization: llama.cpp ((MIT))
- F32 patch / bake code: 本 研 究 ((Apache 2.0; see `LICENSE-CODE`))
- Patched GGUF model files: Gemma 4 Apache 2.0 basisの派生重み。配布前に `LICENSE-WEIGHTS` と上流model cardを再確認する。
- Paper / explanatory docs: Zenodo (DOI 10.5281/zenodo.20362821) にて CC-BY-4.0 で公開。code / weights とは別スコープ。
- 観 察 ((credit)): ユーザー の 「L25/L26 = 計 算 / メタ 思 考」 仮 説 提 示 ((2026-05-30 session))

---

## Citation ((draft))

```
@misc{gemma4-l25l26-functional-asymmetry-pilot-2026,
  title  = {Adjacent-Layer Functional Specialization in Q2_K Quantized Gemma 4 31B:
            A Single-Question Pilot of L25 (Compute) and L26 (Meta-Verify)},
  author = {Hirai, Akito},
  year   = {2026},
  month  = {May 30},
  note   = {n=1 hypothesis-generating observation, replication pending (exp166)},
}
```
