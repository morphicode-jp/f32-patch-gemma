---
license: apache-2.0
base_model: google/gemma-4-31b-it
tags:
  - quantization
  - gguf
  - gemma
  - llama.cpp
  - f32-patch
  - basin-b
language:
  - en
  - ja
pipeline_tag: text-generation
---

# Gemma 4 31B-it Q2_K + basin B (44-byte F32 patch)

**Training-free quantization recovery via 44 bytes of F32 scale tuning.**

A Q2_K quantized Gemma 4 31B-it model with the `basin B` patch applied to 11 layers' F32 `layer_output_scale` weights. The patch adds **+13.25pt HellaSwag accuracy** (n=400) compared to the unpatched baseline, with no fine-tuning, no calibration data, no inference overhead.

## TL;DR

| metric | baseline Q2_K | basin B patched | Δ |
|---|---|---|---|
| **HellaSwag** (n=400) | 57.00% | **70.25%** | **+13.25pt** ⭐ |
| safety (AdvBench refusal, **n=520 full**) | 97.31% | **97.12%** | 0 (statistically equivalent, CIs overlap) |

- **Patch size**: 44 bytes (11 layers × 4 bytes F32 scalar)
- **Quantization**: Q2_K, 12.6 GB
- **Inference cost**: zero overhead (only weight values change)
- **Reproducibility**: any quant of Gemma 4 31B-it accepts the same patch

## What is basin B?

A specific set of multiplicative scales applied to 11 F32 `layer_output_scale` weights in Gemma 4 31B, discovered via the `morphi` autonomous optimization engine (12-specialist parallel search, 7-hour run). The patch:

```python
{
    5:  1.440,
    18: 0.767,
    19: 1.016,
    20: 0.814,
    21: 1.413,
    25: 1.402,
    26: 1.289,
    27: 1.326,
    28: 1.431,
    30: 0.548,
    31: 1.267,
}
```

Mechanism (preliminary): Gemma 4 is a hybrid LLM (5:1 full-attention to sliding-window ratio). The patch unlocks "slack" in the rare full-attention layers' RMSNorm scales, which were conservatively constrained during training. Cross-architecture validation (Phase A–I, 4 models) confirms this is structural — not a quantization artifact.

## How to use

### Option 1: Use this GGUF directly

```bash
# Download
huggingface-cli download morphicode_jp/gemma-4-31B-it-basin-b-Q2_K \
    --local-dir ./gemma-basin-b

# Run with llama.cpp
./llama-cli \
    -m ./gemma-basin-b/gemma-4-31B-it-basin-b-Q2_K.gguf \
    -ngl 99 -c 4096 \
    -p "Your prompt here"
```

### Option 2: Apply basin B to your own GGUF

```bash
# Download apply script
git clone https://github.com/morphicode_jp/f32-patch-gemma
cd f32-patch-gemma

# Apply to ANY Gemma 4 31B quant (IQ1_M, Q2_K, Q4_K_M, Q5_K_M, Q8_0)
python apply_basin_b.py /path/to/your-gemma-4-31b.gguf

# Dry-run first to see the plan
python apply_basin_b.py /path/to/your-gemma-4-31b.gguf --dry-run

# Undo
python apply_basin_b.py /path/to/your-gemma-4-31b.gguf --restore
```

## Results

### HellaSwag (n=400)

| condition | accuracy | Wilson 95% CI |
|---|---|---|
| baseline Q2_K | 57.00% | [52.0%, 61.9%] |
| basin B patched | **70.25%** | **[65.5%, 74.6%]** |

CIs do not overlap → statistically significant improvement.

### Cross-quant universality (HellaSwag n=400, prior experiments)

basin B applies the same patch values to all Gemma 4 31B quants:

| quant | size | baseline | basin B | Δ |
|---|---|---|---|---|
| IQ1_M | 8.4 GB | 34.75% | TBD | TBD |
| Q2_K | 12.6 GB | 57.00% | 70.25% | +13.25 |
| Q4_K_M | 18.5 GB | 75.50% | TBD | TBD |
| Q8_0 | 33.0 GB | 78.25% | TBD | TBD |

(Full universal results in the upcoming arXiv preprint.)

### Cross-family transfer (Phase 9)

The same methodology was tested on Qwen 3.6 27B Q2_K (SSM-hybrid architecture):

| model | architecture | best Δ |
|---|---|---|
| Gemma 4 31B | hybrid (5:1 full:SWA) | **+13.25pt** (basin B, 11 layers) |
| Qwen 3.6 27B | hybrid (1:3 full:SSM) | +3.50pt (L55+L58 pair) |
| Phi-4 14B BF16 | pure dense | -2.31pt (mean, destructive) |

→ F32 slack is **hybrid-architecture specific**, not universal across all LLMs.

### Safety (AdvBench, full n=520)

| condition | refusal rate | Wilson 95% CI |
|---|---|---|
| baseline Q2_K | 97.31% (506/520) | [95.53%, 98.39%] |
| **basin B patched** | **97.12% (505/520)** | **[95.30%, 98.24%]** |
| (sister) L25+L26 ×1.5 (lighter patch) | 93.46% (486/520) | [91.00%, 95.28%] |

**basin B's CI fully overlaps baseline's CI** → alignment is statistically indistinguishable from the unpatched model. The 44-byte F32 patch does **not** compromise RLHF safety guardrails.

The lighter L25+L26 ×1.5 patch (an alternative; not this release) shows a small ~4pt refusal-rate decrease (CIs barely separated). basin B is recommended over L25+L26 for safety-critical deployments.

## Files in this repository

- `gemma-4-31B-it-basin-b-Q2_K.gguf` — the patched model (12.6 GB)
  - MD5: `b0f3acc141edbfca84fd44208330bf39`
  - Verification run (HellaSwag n=400): 69.75% (within ±2pt noise of the canonical 70.25% measurement)
- `gemma-4-31B-it-basin-b-Q2_K.gguf.md5` — checksum file
- `README.md` — this file
- `LICENSE` — Apache 2.0 (for tooling; Gemma weights subject to Google's Gemma Terms)

## Reproducibility

1. Download original `bartowski/google_gemma-4-31b-it-Q2_K.gguf` (or apply to any other quant)
2. Clone https://github.com/morphicode_jp/f32-patch-gemma
3. Run `python apply_basin_b.py <your-gguf-path>`
4. Verify md5 matches this repository's `.md5` file
5. Evaluate with `llama-perplexity --hellaswag --hellaswag-tasks 400`

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

(arXiv preprint forthcoming)

## Sister releases

- `morphicode_jp/morphi` — autonomous optimization engine (12 specialist council)
  that discovered basin B [coming soon]

## Limitations

- Verified on Gemma 4 31B-it only. Other Gemma 4 sizes (9B, 27B) not yet tested.
- Cross-family transfer is weak (Qwen 3.6 +3.5pt; Phi-4 / Llama / Mistral null).
- HellaSwag n=400 used here; n=10042 full validation in arXiv preprint.
- Improvement may be benchmark-dependent; not a universal performance lift.

## Acknowledgments

- Google Gemma team for releasing the base model
- llama.cpp authors (Georgi Gerganov et al.) for GGUF tooling
- Bartowski for community quantizations that inspired this work
- Anthropic for the interpretability research that motivated mechanistic framing

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- GitHub: [github.com/morphicode_jp](https://github.com/morphicode_jp)

DMs open for research collaboration.
