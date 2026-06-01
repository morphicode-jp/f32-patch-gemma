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
  - 4-bit
language:
  - en
  - ja
pipeline_tag: text-generation
---

# Gemma 4 31B-it Q4_K_M + L25+L26 ×1.5 (8-byte F32 patch)

**The mainstream-quality release. Q4_K_M is the LLM community's go-to quantization, with an 8-byte L25+L26 patch that beats both its own baseline AND the 8-bit BF16 reference on every benchmark.**

A Q4_K_M quantized Gemma 4 31B-it model — **~19 GB** — with `L25+L26 ×1.5` patch applied to 2 F32 `layer_output_scale` weights. **Training-free, calibration-free, zero inference overhead.**

## TL;DR

| metric | baseline Q4_K_M | L25+L26 ×1.5 patched | Δ |
|---|---|---|---|
| **HellaSwag** (n=10042 full) | 63.70% [62.76, 64.64] | **73.50%** [72.63, 74.36] | **+9.80pt** (CIs separated 7.99pt) |
| **GSM8k** (n=100) | 72.00% [62.51, 79.86] | **87.00%** [79.02, 92.24] | **+15.00pt** ⭐ |
| **Winogrande** (n=1267 full) | 65.04% [62.37, 67.61] | **70.32%** [67.75, 72.77] | **+5.28pt** |
| **ARC-Challenge** (n=1165) | 44.89% [42.06, 47.76] | **48.76%** [45.89, 51.63] | **+3.87pt** |
| safety | (Q2_K AdvBench L25+L26 retention 93.46%) | — | — |

**Striking result**: this 19 GB Q4_K_M patched at 73.50% **surpasses the BF16 reference (Q8_0 baseline) of 63.33% by +10.17pt on HellaSwag**, and beats Q8 baseline on all 4 benchmarks (GSM8k by +17pt). The 8-byte patch unlocks capacity beyond simple quantization recovery — a phenomenon unique to hybrid LLMs (see paper §5–7).

- **Patch size**: 8 bytes (2 layers × 4 bytes F32 scalar)
- **Model size**: ~19 GB
- **Recommended for**: GPUs with ≥24 GB VRAM, mainstream quality demand

## What is L25+L26 ×1.5?

```python
l25_l26_patch = {
    25: 1.5,
    26: 1.5,
}
```

Two multiplicative scales on F32 `layer_output_scale` weights at layers 25 and 26. The simplest possible patch that consistently unlocks capacity.

**This same 8-byte patch is also used for [IQ1_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) and [Q2_K](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K)** — single recipe across all 3 release quantizations.

Mechanism: Gemma 4 is a hybrid LLM (5:1 full-attention to sliding-window-attention ratio). The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales. Validated across 4 model architectures (see paper).

## Honest note: I tried the fancy way first

Before settling on L25+L26, I ran an autonomous optimization engine (`morphi`, ~7h on a single 32 GB consumer GPU) targeting Q4 HellaSwag. It found an 11-layer 44-byte patch called `basin B`. On Q4, basin B is **outperformed by L25+L26 on every benchmark**:

| bench | basin B (44 B) | L25+L26 (8 B) |
|---|---|---|
| HellaSwag | 72.82 | **73.50** (+0.68) |
| Winogrande | 69.61 | **70.32** (+0.71) |
| GSM8k | 84.00 | **87.00** (+3.00) |
| ARC-C | 48.50 | **48.76** (+0.26) |

7 hours of search lost to a 2-layer baseline. The over-parameterized solution generalizes worse. Paper §3.3 has the honest writeup.

## Files

- `gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf` (~19 GB)
  - MD5: `2c348ed9c3c499587343a93de07a84ce`
- `gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf.md5`
- `apply_l25l26.py`
- `README.md` (this file)
- `LICENSE`

## How to use

```bash
huggingface-cli download morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M \
    --local-dir ./gemma
./llama-cli -m ./gemma/gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf -ngl 99 -c 4096
```

### Apply to your own GGUF

```bash
pip install gguf numpy
git clone https://github.com/morphicode-jp/f32-patch-gemma
python f32-patch-gemma/apply_l25l26.py /path/to/google_gemma-4-31b-it-Q4_K_M.gguf
```

`--restore` undoes the patch via the auto-created `.backup` file.

## Sister releases

- [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) — 1-bit, ~9.5 GB, +36pt GSM8k showcase
- [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) — 2-bit, ~12 GB, flagship low-spec

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
- Alignment was measured on Q2_K (L25+L26: 93.46% refusal retention vs baseline 97.31%, a 3.85pt drop). Q4 alignment not yet measured; users with strict safety requirements may prefer the basin B variant (see paper §6 trade-off).
- **Scorer caveat**: HellaSwag/Winogrande accuracies measured with
  `llama-perplexity --hellaswag` mode, systematically 0.2–2.5 pp lower than
  `lm-evaluation-harness` standard
  [(llama.cpp discussion #2321)](https://github.com/ggml-org/llama.cpp/discussions/2321).
  Our +9.80 pp Q4 improvement greatly exceeds typical Q4_K_M quantization
  degradation on Llama-2-7B (~0.5 pp), suggesting structural capacity unlock.

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- GitHub: [github.com/morphicode-jp](https://github.com/morphicode-jp)
- Zenodo (paper + code archive): [doi.org/10.5281/zenodo.20362821](https://doi.org/10.5281/zenodo.20362821)

DMs open for research collaboration.

License: Apache 2.0 for the patch tooling. Gemma 4 base weights are licensed under [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license) (verified 2026-05-31; Gemma 4 was moved off the older Gemma Terms of Use). The patched-GGUF derivative notice is in `LICENSE-WEIGHTS`.
