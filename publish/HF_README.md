---
license: gemma
language:
  - en
base_model:
  - google/gemma-4-31B-it
tags:
  - gemma
  - quantization
  - gguf
  - calibration
  - F32-patch
  - training-free
  - reasoning
library_name: gguf
pipeline_tag: text-generation
---

> ⚠ **This is the paper v1 generic README (2026-05-27).** For paper v5 (per-layer L25/L26 ablation + Practical Use Note + Goodhart's Law triple finding), see [`release_v1/HF_MODEL_CARD.md`](release_v1/HF_MODEL_CARD.md) and [`release_v1/README_paper_v5.md`](release_v1/README_paper_v5.md).

# Gemma 4 31B-it — F32 Patch (L25+L26 ×1.5, 8 byte)

> **8 bytes. +10pt mean across 3 benchmarks. No retraining. No GPU. No fine-tuning.**
>
> A training-free F32 calibration patch for Google Gemma 4 31B-it that recovers ~10pt accuracy across 7 quantization levels (IQ1_M to Q8_0) by modifying only **2 F32 scalars** in the GGUF binary.

## TL;DR

- **Patch**: L25 and L26 `layer_output_scale` × 1.5 (= 8 bytes total, 2 × float32)
- **Effect**: HellaSwag +9.00pt / GSM8k +15pt (n=100, 95% CI [79.0, 92.2]) / MMLU +6pt (lm-eval 5-shot)
- **Universal**: All 7 quantizations (IQ1_M, Q2_K, Q4_K_M, Q5_K_M, Q8_0, UD-IQ2_XXS, UD-Q2_K_XL) show +9 to +14pt mean +10.71pt
- **IQ1_M revival**: 34.75% → 48.25% (+13.50pt) — first technique to make 1-bit Gemma 4 practically usable
- **Gemma-specific**: Llama 3.1 / Mistral 7B show no comparable effect (Google calibration peculiarity)
- **Inference overhead**: 0% (binary modification only, no compute change)

## Available files

| File | Quant | Size | HellaSwag baseline → patched | GSM8k Δ |
|---|---|---|---|---|
| `gemma-4-31B-it-F32Patch-Q4_K_M.gguf` | Q4_K_M | 19.6 GB | 63.75% → 72.50% (+8.75pt) | +15pt |
| `gemma-4-31B-it-F32Patch-Q5_K_M.gguf` | Q5_K_M | 22.6 GB | 63.25% → 73.75% (+10.50pt) | (TBD) |
| `gemma-4-31B-it-F32Patch-Q8_0.gguf` | Q8_0 | 32.6 GB | 62.50% → 73.75% (+11.25pt) | (TBD) |
| `gemma-4-31B-it-F32Patch-IQ1_M.gguf` ⭐ | IQ1_M | 10.1 GB | 34.75% → 48.25% (+13.50pt) | (TBD) |

(Optional `_basinA.gguf` and `_basinB.gguf` variants are also provided as research artifacts; see paper Section 5.4.)

## Why this matters

### For practitioners
- **Mobile / Edge**: IQ1_M revival (10 GB) makes Gemma 4 deployable on Pixel-class devices. Previously chance-level (35%), now usable (48%).
- **Self-hosting**: Q4_K_M users get +8.75pt HellaSwag and +15pt GSM8k for free. No re-quantization, no fine-tuning.
- **Inference servers**: Drop-in replacement for existing GGUF, zero compute overhead.

### For researchers
- A new category of post-quantization recovery: **F32 scalar patching** (orthogonal to QAT, GPTQ, AWQ, SmoothQuant, LoRA).
- Evidence that Gemma 4 ships with conservative calibration; ×1.5 boost on critical layers (L25, L26) is the optimal rescaling.
- Multi-basin landscape: at least 2 independent solutions (8-byte L25+L26 and 44-byte ODIN basin B) yield comparable improvements, suggesting structural redundancy in the calibration error.

## How to use

### Option 1: Download patched GGUF (this repo)
```bash
hf download TwelveQuant/Gemma-4-31B-it-F32Patch-L25L26x1.5 \
  gemma-4-31B-it-F32Patch-Q4_K_M.gguf --local-dir ~/llm/models
```

Drop into your existing llama.cpp / Ollama / LM Studio setup. Drop-in compatible with bartowski/Q4_K_M.

### Option 2: Patch your existing GGUF (recommended for verification)
```bash
git clone https://github.com/<author>/f32-patch-gemma
cd f32-patch-gemma
pip install -r requirements.txt

# 1. Snapshot original F32 scalars
python weight_analysis/snapshot_originals.py \
  ~/llm/models/google_gemma-4-31B-it-Q4_K_M.gguf gemma4_q4km

# 2. Apply the 8-byte patch
python -c "
from weight_analysis._eval_lib import patch_layer_scales
patch_layer_scales('gemma4_q4km', {25: 1.5, 26: 1.5})
"

# 3. Verify with HellaSwag
python weight_analysis/exp16_l25l26_universal_quant.py
```

To revert: `python weight_analysis/restore_from_snapshot.py gemma4_q4km`.

## Method (1-paragraph summary)

In Gemma 4's GGUF format, each transformer block stores a single F32 scalar named `blk.{N}.layer_output_scale` that normalizes the dequantized weight outputs. We discovered that increasing the L25 and L26 scalars by 1.5× produces dramatic accuracy improvements across all quantization levels, while leaving L0–L24 and L27–L59 unchanged. The total binary modification is 8 bytes (2 F32 scalars). The change has no inference-time cost and is reversible by restoring the original snapshot. Cross-model experiments on Llama 3.1 8B and Mistral 7B v0.3 show that this phenomenon is specific to Google Gemma's calibration: those models do not exhibit comparable improvements with analogous patches on `attn_norm` or `ffn_norm`.

For the full theoretical and empirical treatment, including ablations (L25 alone, L26 alone), inference latency benchmarks, robustness to seed/temperature/prompt format, and a preliminary mechanistic investigation of why L25 and L26 are pivot layers, see the paper:

📄 **Paper**: [arxiv.org/abs/2506.XXXXX](https://arxiv.org/abs/2506.XXXXX)
🔬 **Code & data**: [github.com/<author>/f32-patch-gemma](https://github.com/<author>/f32-patch-gemma)
🇯🇵 **Japanese version (Zenn)**: [zenn.dev/<author>/articles/...](https://zenn.dev/...)

## Citation

```bibtex
@article{f32patch2026,
  title={8-Byte F32 Calibration: Reviving 1-bit Quantization in Gemma 4 with Training-Free Layer-Output Scale Patching},
  author={<著者本名>},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2026}
}
```

## Limitations

- Verified on Gemma 4 31B-it (60 layer) and partially on Gemma 4 E4B (42 layer); 27B and Gemma 2/3 series untested
- Mechanistic explanation of why L25/L26 are pivot layers is preliminary
- HellaSwag/GSM8k/MMLU benchmarks; long-form generation quality not evaluated
- Statistical 95% CI confirmed for n=100 (GSM8k) and n=400 (HellaSwag); MMLU n=100 may have wider CI

## Ethics & License

- Inherits the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) from the base model
- This patched variant is research output, distributed under the same license
- The 8-byte modification does not alter model identity (still Gemma 4 31B-it), only adjusts calibration

## Acknowledgments

- bartowski for the original Q4_K_M / Q5_K_M / Q8_0 / IQ1_M / Q2_K GGUFs
- Unsloth team for UD-IQ2_XXS / UD-Q2_K_XL Dynamic 2.0 GGUFs
- llama.cpp (ggml.ai) team for the GGUF format and inference engine
- Google DeepMind for the Gemma 4 base model
