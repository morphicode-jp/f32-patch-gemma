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
  - 1-bit
language:
  - en
  - ja
pipeline_tag: text-generation
---

# Gemma 4 31B-it IQ1_M + L25+L26 ×1.5 (8-byte F32 patch)

**1-bit ULTRA-quantized Gemma 4 31B revived with an 8-byte F32 patch — GSM8k 24% → 60% (+36pt), HellaSwag +10.95pt, training-free.**

A 1-bit (IQ1_M) quantized Gemma 4 31B-it model — only **~9.5 GB** — with the minimal **L25+L26 ×1.5** patch applied to 2 F32 `layer_output_scale` weights. **Training-free, calibration-free, zero inference overhead.**

## TL;DR

| metric | baseline IQ1_M | L25+L26 ×1.5 patched | Δ |
|---|---|---|---|
| **GSM8k** (n=100) | 24.0% [16.69, 33.23] | **60.0%** [50.20, 69.06] | **+36.0pt** ⭐⭐ (CIs separated 16.99pt) |
| **HellaSwag** (n=10042 full) | 42.02% [41.06, 42.99] | **52.98%** [52.00, 53.95] | **+10.95pt** (CIs separated) |
| **Winogrande** (n=1267 full) | 49.80% | **55.56%** | **+5.76pt** |
| **ARC-Challenge** (n=1165) | 30.56% | **36.74%** | **+6.18pt** |

**Striking result**: GSM8k jumps from 24% to 60% — a **+36 percentage point** improvement, the largest single-cell gain in our 12-cell evaluation matrix. This is the strongest evidence that the F32 patch unlocks structural reasoning capacity rather than merely recovering quantization loss.

- **Patch size**: 8 bytes (2 layers × 4 bytes F32 scalar)
- **Model size**: ~9.5 GB
- **Recommended for**: CPU-only laptops, low-VRAM GPUs (≤8 GB)
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

Before settling on L25+L26, I ran an autonomous optimization engine (`morphi`, ~7h on a single 32 GB consumer GPU) targeting Q4 HellaSwag. It found an 11-layer 44-byte patch called `basin B`. On Q1 GSM8k specifically, basin B was **worse than baseline** (-8 pt, dropping to 16%), while L25+L26 reached 60%. The simple 2-layer patch wins. The basin B values are kept in the paper appendix for transparency.

## Files

- `gemma-4-31B-it-L25L26x1.5-IQ1_M.gguf` (~9.5 GB)
  - MD5: `0985dd00c5408169d77cd4c3c021fda6`
- `gemma-4-31B-it-L25L26x1.5-IQ1_M.gguf.md5`
- `apply_l25l26.py`
- `README.md` (this file)
- `LICENSE`

## How to use

```bash
huggingface-cli download morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M \
    --local-dir ./gemma
./llama-cli -m ./gemma/gemma-4-31B-it-L25L26x1.5-IQ1_M.gguf -ngl 99 -c 4096
```

### Apply to your own GGUF

```bash
pip install gguf numpy
git clone https://github.com/morphicode-jp/f32-patch-gemma
python f32-patch-gemma/apply_l25l26.py /path/to/google_gemma-4-31b-it-IQ1_M.gguf
```

`--restore` undoes the patch via the auto-created `.backup` file.

## Sister releases

- [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) — 2-bit, ~12 GB, flagship low-spec
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
- Alignment was measured on Q2_K (L25+L26: 93.46% AdvBench refusal retention vs baseline 97.31%, a 3.85pt drop). Q1 alignment not separately measured.
- **Scorer caveat**: HellaSwag/Winogrande accuracies measured with
  `llama-perplexity --hellaswag` mode, systematically 0.2–2.5 pp lower than
  `lm-evaluation-harness` standard
  [(llama.cpp discussion #2321)](https://github.com/ggml-org/llama.cpp/discussions/2321).
  Within-scorer baseline-vs-patched deltas remain valid.

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- GitHub: [github.com/morphicode-jp](https://github.com/morphicode-jp)
- Zenodo (paper + code archive): [doi.org/10.5281/zenodo.20362821](https://doi.org/10.5281/zenodo.20362821)

DMs open for research collaboration.

License: Apache 2.0 for the patch tooling. Gemma 4 base weights are licensed under [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license) (verified 2026-05-31; Gemma 4 was moved off the older Gemma Terms of Use). The patched-GGUF derivative notice is in `LICENSE-WEIGHTS`.
