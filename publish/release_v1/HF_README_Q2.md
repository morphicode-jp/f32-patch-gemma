---
license: apache-2.0
base_model: google/gemma-4-31b-it
tags:
  - quantization
  - gguf
  - gemma
  - llama.cpp
  - f32-patch
  - L25L26
  - 2-bit
language:
  - en
  - ja
pipeline_tag: text-generation
---

# Gemma 4 31B-it Q2_K + L25+L26 ×1.5 (8-byte F32 patch) — flagship low-spec release

**The flagship low-spec release: Q2_K is the sweet spot for low-spec rescue + reliable accuracy. Only 8 bytes of F32 patch yield +11.21pt HellaSwag improvement (full validation n=10,042).**

A Q2_K quantized Gemma 4 31B-it model — only **~12.6 GB** — with the minimal **L25+L26 ×1.5** patch applied to 2 F32 `layer_output_scale` weights. **Training-free, calibration-free, zero inference overhead.**

## TL;DR

| metric | baseline Q2_K | L25+L26 ×1.5 patched | Δ |
|---|---|---|---|
| **HellaSwag** (n=10042 full) | 59.15% [58.19, 60.11] | **70.36%** [69.46, 71.25] | **+11.21pt** ⭐ (CIs separated 9.35pt) |
| **GSM8k** (n=100) | 67.0% [57.31, 75.44] | **76.0%** [66.77, 83.31] | **+9.00pt** |
| **Winogrande** (n=1267 full) | 59.35% | **66.69%** | **+7.34pt** |
| **ARC-Challenge** (n=1165) | 43.61% | **46.44%** | **+2.83pt** |
| safety | 97.31% refusal | 93.46% refusal | -3.85pt (still >93%) |

**Striking result**: 70.36% HellaSwag (patched Q2_K) **surpasses the BF16-proxy baseline of 63.33%** by **+7.03pt**, demonstrating the patch unlocks capacity beyond simple quantization recovery — a phenomenon unique to hybrid LLMs (see paper §5–7).

- **Patch size**: 8 bytes (2 layers × 4 bytes F32 scalar)
- **Model size**: ~12.6 GB
- **Recommended for**: GPUs with ≥16 GB VRAM, low-spec rescue
- **Same patch used uniformly for Q1/Q2/Q4** — a single 8-byte recipe across all release quantizations

## What is L25+L26 ×1.5?

```python
l25_l26_patch = {
    25: 1.5,
    26: 1.5,
}
```

Two multiplicative scales on F32 `layer_output_scale` weights at layers 25 and 26. The simplest possible patch that consistently unlocks capacity across all three release quantizations.

Mechanism: Gemma 4 is a hybrid LLM (5:1 full-attention to sliding-window-attention ratio). The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales. Validated across 4 model architectures (see paper).

## Honest note: I tried the fancy way first

Before settling on L25+L26, I ran an autonomous optimization engine (`morphi`, ~7h on a single 32 GB consumer GPU) targeting Q4 HellaSwag. It found an 11-layer 44-byte patch called `basin B`. On Q2 GSM8k, basin B underperformed the baseline (-4 pt drop), while L25+L26 improved it (+9 pt). The simple 2-layer patch wins. The basin B values are kept in the paper appendix for transparency.

## Safety (AdvBench n=520)

| condition | refusal rate | Wilson 95% CI |
|---|---|---|
| baseline (Q2_K) | 97.31% (506/520) | [95.53%, 98.39%] |
| **L25+L26 ×1.5 patched** | **93.46% (486/520)** | [91.00%, 95.28%] |

The L25+L26 patch shows a small (~3.85pt) refusal-rate decrease versus baseline; alignment remains above 93%. For users with strict alignment requirements, the heavier `basin B` patch (44 bytes, refusal rate 97.12% — CI fully overlaps baseline) is provided in the paper appendix as an alternative.

## Files

- `gemma-4-31B-it-L25L26x1.5-Q2_K.gguf` (~12.6 GB)
  - MD5: `28f13a6f87d01fdaebb72cdc8aa9faec`
- `gemma-4-31B-it-L25L26x1.5-Q2_K.gguf.md5`
- `apply_l25l26.py`
- `README.md` (this file)
- `LICENSE`

## How to use

```bash
huggingface-cli download morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K \
    --local-dir ./gemma
./llama-cli -m ./gemma/gemma-4-31B-it-L25L26x1.5-Q2_K.gguf -ngl 99 -c 4096
```

### Apply to your own GGUF

```bash
pip install gguf numpy
git clone https://github.com/morphicode-jp/f32-patch-gemma
python f32-patch-gemma/apply_l25l26.py /path/to/google_gemma-4-31b-it-Q2_K.gguf
```

`--restore` undoes the patch via the auto-created `.backup` file.

## Sister releases

- [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) — 1-bit, ~9.5 GB, +36pt GSM8k showcase
- [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M) — 4-bit, ~19 GB, mainstream quality (beats Q8_0 BF16 baseline on all 4 benchmarks)

All three use the identical 8-byte L25+L26 ×1.5 patch.

## Citation

```bibtex
@misc{hirai2026f32patch,
  title  = {Why Some LLMs Have a Hidden Reasoning Knob:
            Rare Full-Attention Bottlenecks in Hybrid Architectures
            and an 8-byte Quantization Recovery},
  author = {Hirai, Akito},
  year   = {2026},
  doi    = {10.5281/zenodo.20362821},
  url    = {https://doi.org/10.5281/zenodo.20362821}
}
```

## Limitations & Methodology Notes

- Patch values are calibrated for Gemma 4 31B; other Gemma sizes (9B, 27B) not tested.
- Cross-family transfer is weak (Qwen 3.6 +3.5pt; Phi-4 / Llama / Mistral null).
- Alignment shows a ~3.85pt decrease vs baseline (still >93%). For maximum alignment preservation, the basin B alternative (44 bytes, CI fully overlaps baseline) is described in the paper appendix.
- **Scorer caveat**: HellaSwag/Winogrande accuracies measured with
  `llama-perplexity --hellaswag` mode, systematically 0.2–2.5 pp lower than
  `lm-evaluation-harness` standard
  [(llama.cpp discussion #2321)](https://github.com/ggml-org/llama.cpp/discussions/2321).
  Our +11.21 pt HellaSwag improvement greatly exceeds typical Q2_K quantization
  degradation, suggesting structural capacity unlock.

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- GitHub: [github.com/morphicode-jp](https://github.com/morphicode-jp)
- Zenodo (paper + code archive): [doi.org/10.5281/zenodo.20362821](https://doi.org/10.5281/zenodo.20362821)

DMs open for research collaboration.

License: Apache 2.0 for the patch tooling. Gemma 4 base weights are licensed under [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license) (verified 2026-05-31; Gemma 4 was moved off the older Gemma Terms of Use). The patched-GGUF derivative notice is in `LICENSE-WEIGHTS`.
