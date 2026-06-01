# f32-patch-gemma

**44-byte F32 patch that improves Gemma 4 31B Q2_K HellaSwag accuracy by +13.25pt — no training, no calibration, no inference overhead.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/morphicode_jp/gemma-4-31B-it-basin-b-Q2_K)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-red)](#)

## What this is

A single-file Python script that modifies **44 bytes** of any Gemma 4 31B GGUF file (across all quantizations: IQ1_M, Q2_K, Q4_K_M, Q5_K_M, Q8_0) and yields measurable HellaSwag accuracy improvement, without any retraining or fine-tuning.

## Quick start

```bash
pip install gguf numpy

# Dry-run first to see the plan
python apply_basin_b.py /path/to/gemma-4-31b.gguf --dry-run

# Apply (creates .backup automatically)
python apply_basin_b.py /path/to/gemma-4-31b.gguf

# Undo
python apply_basin_b.py /path/to/gemma-4-31b.gguf --restore
```

## Results (Gemma 4 31B Q2_K, HellaSwag n=400)

| condition | accuracy | Wilson 95% CI |
|---|---|---|
| baseline Q2_K | 57.00% | [52.0%, 61.9%] |
| **basin B patched** | **70.25%** | **[65.5%, 74.6%]** |
| Δ | **+13.25pt** | CIs do not overlap |

Safety alignment preserved on full AdvBench (n=520):
- baseline: 97.31% refusal (Wilson 95% CI [95.53, 98.39])
- basin B: 97.12% refusal (Wilson 95% CI [95.30, 98.24])
- CIs fully overlap → statistically indistinguishable, alignment fully preserved.

## What is basin B?

A specific multiplicative-scale configuration on 11 layers' F32 `layer_output_scale` weights, discovered by the `morphi` autonomous optimization engine (12-specialist parallel search, 7-hour run on RTX 5090).

```python
basin_b = {
    5:  1.440,   # ↑
    18: 0.767,   # ↓
    19: 1.016,
    20: 0.814,   # ↓
    21: 1.413,   # ↑
    25: 1.402,   # ↑
    26: 1.289,   # ↑
    27: 1.326,   # ↑
    28: 1.431,   # ↑
    30: 0.548,   # ↓
    31: 1.267,   # ↑
}
```

## Why it works (preliminary mechanism)

Gemma 4 is a **hybrid LLM** with 5:1 full-attention to sliding-window-attention layer ratio. The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales, which appear to be conservatively constrained during training.

This is a **structural** phenomenon, not a quantization artifact:
- **Gemma 4 31B** (5:1 hybrid, Q2_K): **+13.25pt**
- **Qwen 3.6 27B** (1:3 SSM hybrid, Q2_K): +3.50pt
- **Phi-4 14B** (pure dense, BF16): -2.31pt (destructive)
- **Llama 3.1 / Mistral 7B** (pure dense): null

→ F32 slack is **hybrid-architecture specific**. Pure dense transformers do not benefit.

(Full mechanistic analysis in the upcoming arXiv preprint.)

## Reproducibility

```bash
# 1. Get any Gemma 4 31B GGUF
huggingface-cli download bartowski/google_gemma-4-31b-it-Q2_K \
    --local-dir ./gemma

# 2. Apply patch
python apply_basin_b.py ./gemma/google_gemma-4-31b-it-Q2_K.gguf

# 3. Evaluate (requires llama.cpp)
~/llm/llama.cpp/llama-perplexity \
    -m ./gemma/google_gemma-4-31b-it-Q2_K.gguf \
    -f ~/llm/benchmarks/hellaswag_val.txt \
    -ngl 99 --hellaswag --hellaswag-tasks 400 \
    -c 512
```

Expected: baseline ~57%, patched ~70%.

## Files

- [`apply_basin_b.py`](apply_basin_b.py) — the entire tool, single file
- [`LICENSE`](LICENSE) — Apache 2.0
- [`CITATION.cff`](CITATION.cff) — citation metadata
- [`README.md`](README.md) — this file

No dependencies beyond `gguf` and `numpy`. No conda env. No build step.

## Background

This work was developed independently while I worked my day job as an engineer. The discovery uses an autonomous optimization engine I built called **morphi** (12-specialist parallel search), which will be released separately.

The mechanistic finding (rare full-attention bottleneck in hybrid LLMs) was validated across 4 model families before release.

## Releases planned

- **v1** (this release): basin B patch + HF model card (Gemma 4 31B Q2_K)
- **v2**: arXiv preprint with full 4-model comparison + mechanistic analysis
- **v3**: per-tensor exploration (sub-44-byte patches)
- **v4**: morphi optimization engine OSS release

## Citation

```bibtex
@misc{hirai2026f32patch,
  title  = {Why Some LLMs Have a Hidden Reasoning Knob:
            Rare Full-Attention Bottlenecks in Hybrid Architectures
            and a 44-byte Quantization Recovery},
  author = {Hirai, Akito},
  year   = {2026},
  url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

Apache 2.0 for this tooling. Gemma 4 weights are subject to Google's [Gemma Terms](https://ai.google.dev/gemma/terms). Users redistributing patched Gemma 4 models must comply with both.

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- HF: [huggingface.co/morphicode_jp](https://huggingface.co/morphicode_jp)

DMs open for research collaboration and questions.
