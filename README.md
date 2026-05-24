# f32-patch-gemma

**An 8-byte F32 patch that improves Gemma 4 31B accuracy across all evaluated quantizations and benchmarks (12/12 cells positive) — no training, no calibration, no inference overhead.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Zenodo DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20362821-blue)](https://doi.org/10.5281/zenodo.20362821)
[![HuggingFace Q1](https://img.shields.io/badge/🤗-Q1_IQ1__M-yellow)](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M)
[![HuggingFace Q2](https://img.shields.io/badge/🤗-Q2_K-yellow)](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K)
[![HuggingFace Q4](https://img.shields.io/badge/🤗-Q4_K__M-yellow)](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M)

## What this is

A single-file Python script (`apply_l25l26.py`, ~150 lines) that modifies **8 bytes** of any Gemma 4 31B GGUF file (across all quantizations: IQ1_M, Q2_K, Q4_K_M, Q5_K_M, Q8_0) and yields measurable benchmark improvements, without any retraining or fine-tuning.

The patch: multiply `layer_output_scale` at layers 25 and 26 by 1.5. That's it. 8 bytes (2 × F32). ~3 seconds to apply.

## Headline results (Gemma 4 31B-it)

| quant | HellaSwag n=10,042 | Winogrande n=1,267 | GSM8k n=100 | ARC-C n=1,165 |
|---|---|---|---|---|
| **Q1** (IQ1_M, 9.5 GB) | 42.02 → 52.98 (**+10.95**) | 49.80 → 55.56 (+5.76) | 24 → **60** (**+36** ⭐) | 30.56 → 36.74 (+6.18) |
| **Q2** (Q2_K, 12.6 GB) | 59.15 → **70.36** (**+11.21** ⭐) | 59.35 → 66.69 (+7.34) | 67 → 76 (+9.0) | 43.61 → 46.44 (+2.83) |
| **Q4** (Q4_K_M, 19 GB) | 63.70 → **73.50** (**+9.80**) | 65.04 → 70.32 (+5.28) | 72 → **87** (**+15**) | 44.89 → 48.76 (+3.87) |
| Q8 baseline (ref.)† | 63.33 | 65.59 | 70.00 | 44.38 |

**Q4 patched beats Q8 baseline on all 4 benchmarks** (HellaSwag +10.17pt, GSM8k +17pt). HellaSwag Wilson 95% CIs do not overlap. See [paper](paper/main.pdf) for full details + alignment analysis.

† Q8_0 reported only as a BF16-proxy baseline reference. We do not release a Q8 patched model.

## Quick start

```bash
pip install gguf numpy

# Dry-run first to see the plan
python apply_l25l26.py /path/to/gemma-4-31b.gguf --dry-run

# Apply (creates .backup automatically)
python apply_l25l26.py /path/to/gemma-4-31b.gguf

# Undo
python apply_l25l26.py /path/to/gemma-4-31b.gguf --restore
```

## Pre-patched GGUF releases on HuggingFace

| Quant | Size | Link |
|---|---|---|
| IQ1_M (1-bit) | 9.5 GB | [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) |
| Q2_K (2-bit) | 12.6 GB | [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) |
| Q4_K_M (4-bit) | 19 GB | [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M) |

All three use the identical 8-byte L25+L26 ×1.5 patch.

## How it works (one sentence)

Gemma 4 is a hybrid LLM (5:1 full-attention to sliding-window-attention ratio). The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales — a hidden post-hoc tuning knob that the conservative training process leaves on the table.

Cross-architecture validation:
- 🟢 Gemma 4 31B (5:1 hybrid): +13.25pt
- 🟢 Qwen 3.6 27B (1:3 SSM hybrid): +3.5pt
- 🔴 Phi-4 14B BF16 (pure dense): null/destructive
- 🔴 Llama / Mistral (pure dense): null

The Phi-4 BF16 result (no quantization, null effect) rules out "quantization recovery" — the effect is structural to hybrid LLMs.

## Honest negative result: ODIN search

Before settling on the simple 2-layer L25+L26 patch, we ran an autonomous optimization engine (`morphi`, 12-specialist parallel search, 7 hours on RTX 5090) targeting Q4 HellaSwag. It found an 11-layer 44-byte patch we call `basin B` (see `apply_basin_b.py` for the values, kept for transparency).

When benchmarked on Q4 across all 4 release benchmarks, **L25+L26 outperforms basin B on every single one — including the search target itself**:

| bench | basin B (44 B) | L25+L26 (8 B) |
|---|---|---|
| Q4 HellaSwag | 72.82 | **73.50** (+0.68) |
| Q4 Winogrande | 69.61 | **70.32** (+0.71) |
| Q4 GSM8k | 84.00 | **87.00** (+3.00) |
| Q4 ARC-C | 48.50 | **48.76** (+0.26) |

7 hours of high-dimensional search lost to a 2-layer manual baseline. The over-parameterized solution generalizes strictly worse than the simple one — an instance of objective-overfitting. The basin B script is included here for reproducibility but **L25+L26 is the recommended flagship**.

## Repository contents

```
.
├── README.md               this file
├── apply_l25l26.py        ★ flagship patch script (8 bytes, recommended)
├── apply_basin_b.py        legacy / ODIN-discovered patch script (44 bytes, kept for transparency)
├── requirements.txt        pip dependencies
├── CITATION.cff            BibTeX-compatible citation file
├── REPRODUCE.md            full reproduction instructions
├── LICENSE                 Apache 2.0
└── paper/                  paper source + compiled PDF
    ├── main.tex
    ├── main.pdf
    ├── refs.bib
    └── figures/
```

## Citation

```bibtex
@misc{hirai2026f32patch,
  title  = {Why Some LLMs Have a Hidden Reasoning Knob:
            Rare Full-Attention Bottlenecks in Hybrid Architectures
            and an 8-byte Quantization Recovery},
  author = {Hirai, Akito},
  year   = {2026},
  doi    = {10.5281/zenodo.20362821},
  url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

Apache 2.0 for the patch tooling and paper. Gemma 4 weights subject to [Google's Gemma Terms](https://ai.google.dev/gemma/terms).

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- ORCID: TBD
- Email: morphicode.jp@gmail.com

Independent researcher in Tokyo, Japan. Built on nights & weekends while working a day job in oil pump engineering. DMs open for research collaboration.
