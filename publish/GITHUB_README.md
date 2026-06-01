# f32-patch-gemma — 8-byte F32 Calibration for Gemma 4

> ⚠ **This is the paper v1 generic README (2026-05-27).** For paper v5 (per-layer L25/L26 ablation + Practical Use Note + Goodhart's Law triple finding), see [`publish/release_v1/HF_MODEL_CARD.md`](publish/release_v1/HF_MODEL_CARD.md) and [`publish/release_v1/README_paper_v5.md`](publish/release_v1/README_paper_v5.md).

> Reviving 1-bit quantization in Gemma 4 31B-it with a training-free, GPU-free, 8-byte binary patch.

[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/TwelveQuant)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Headline numbers

| Benchmark | Baseline | After patch | Δ |
|---|---|---|---|
| HellaSwag (Q4_K_M, n=400) | 63.75% | 72.50% | **+8.75pt** |
| HellaSwag (IQ1_M, 10 GB) ⭐ | 34.75% | 48.25% | **+13.50pt** (chance level → usable) |
| GSM8k (n=100, 95% CI) | 72.0% [62.5, 79.9] | 87.0% [79.0, 92.2] | **+15.00pt** |
| MMLU (5-shot, lm-eval-harness) | TBD | TBD | TBD |

All 7 quantization levels (IQ1_M to Q8_0): **mean +10.71pt HellaSwag**.

## What is this

Two F32 scalars in Gemma 4 31B-it's GGUF binary — `blk.25.layer_output_scale` and `blk.26.layer_output_scale` — when multiplied by 1.5, dramatically boost benchmark accuracy across all quantization levels. Total binary modification: **8 bytes** (2 × float32). No retraining, no GPU, no calibration data, no inference overhead.

This works on Google Gemma 4 31B-it but **not** on Llama 3.1 8B (+0.75pt only) or Mistral 7B v0.3 (no effect), suggesting Gemma's training pipeline ships with conservative `layer_output_scale` values that can be locally re-calibrated post-hoc.

## Quick start

```bash
git clone https://github.com/<author>/f32-patch-gemma
cd f32-patch-gemma
pip install -r requirements.txt

# Download Gemma 4 31B-it Q4_K_M (~19.6 GB)
hf download bartowski/google_gemma-4-31B-it-GGUF \
  google_gemma-4-31B-it-Q4_K_M.gguf --local-dir ~/llm/models

# Snapshot original F32 scalars (one-time, ~30 sec)
python weight_analysis/snapshot_originals.py \
  ~/llm/models/google_gemma-4-31B-it-Q4_K_M.gguf gemma4_q4km

# Apply the 8-byte patch (in-place modify)
python -c "
import sys; sys.path.insert(0, '.')
from weight_analysis._eval_lib import patch_layer_scales
patch_layer_scales('gemma4_q4km', {25: 1.5, 26: 1.5})
print('Patched: L25 and L26 layer_output_scale × 1.5 (8 bytes modified)')
"

# Verify with HellaSwag (~40 sec)
python weight_analysis/exp29_l25_l26_individual_ablation.py
```

To revert:
```bash
python weight_analysis/restore_from_snapshot.py gemma4_q4km
```

## Repository structure

```
f32-patch-gemma/
├── README.md                      # this file
├── paper/                         # NeurIPS-style LaTeX paper
│   ├── main.tex
│   ├── refs.bib
│   ├── draft_ja.md                # Japanese draft
│   └── figures/                   # 8 figures + reproduction scripts
├── weight_analysis/
│   ├── _eval_lib.py               # HellaSwag eval, snapshot/restore/patch API
│   ├── snapshot_originals.py      # extract F32 scalars to JSON
│   ├── restore_from_snapshot.py   # in-place restore from snapshot
│   ├── exp16_l25l26_universal_quant.py     # Tier 1: 7 quant verification
│   ├── exp27_gsm8k_100q.py                 # Tier 4: GSM8k 100q + Wilson CI
│   ├── exp29_l25_l26_individual_ablation.py  # Phase 7 A1: ablation
│   ├── exp30_hellaswag_n1000.py            # Phase 7 A2: CI tighten
│   ├── exp31_mmlu_lmeval.py                # Phase 7 A3: MMLU rigorous
│   ├── exp32_gemma4_e4b.py                 # Phase 7 A4: E4B family
│   ├── exp33_latency_bench.py              # Phase 7 A5: tokens/sec
│   ├── exp34_robustness.py                 # Phase 7 A6: seed/prompt sweep
│   └── snapshots/                          # F32 originals JSON per quant
├── results/                       # JSON outputs (raw experimental data)
└── docs/
    ├── F32_PATCH_RESEARCH.md      # original 1898-line research notebook (JP)
    └── F32_PATCH_OVERVIEW.html    # visual summary
```

## Reproducing all results

Each experiment is a self-contained script with deterministic output (with the exception of `exp34_robustness.py` which sweeps temperature). To reproduce:

```bash
# Tier 1: 7 quant universal (~30 min on RTX 5090)
python weight_analysis/exp16_l25l26_universal_quant.py

# Tier 4: GSM8k 100q (~2 hours)
python weight_analysis/exp27_gsm8k_100q.py

# Phase 7: additional verification
python weight_analysis/exp29_l25_l26_individual_ablation.py  # ablation
python weight_analysis/exp30_hellaswag_n1000.py              # CI tighten
python weight_analysis/exp31_mmlu_lmeval.py                  # MMLU rigorous
python weight_analysis/exp32_gemma4_e4b.py                   # E4B family
python weight_analysis/exp33_latency_bench.py                # latency
python weight_analysis/exp34_robustness.py                   # robustness
```

Result JSONs land in `results/phase2/`. Aggregate plots are generated by `paper/figures/gen_fig*.py`.

## Hardware tested

- NVIDIA RTX 5090 (32 GB VRAM)
- llama.cpp build b9016 (CUDA 13.1)
- Windows 11 Pro

The patch itself works on any CPU/GPU configuration (it is a 8-byte file modification). Only the verification benchmarks need GPU.

## License

- This repository: MIT License
- Patched GGUFs (released on HuggingFace): inherit Gemma Terms of Use
- Base model: Google Gemma 4 31B-it

## Citation

```bibtex
@article{f32patch2026,
  title={8-Byte F32 Calibration: Reviving 1-bit Quantization in Gemma 4 with Training-Free Layer-Output Scale Patching},
  author={<著者本名>},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2026}
}
```

## Acknowledgments

- bartowski (HF) for the original Q4_K_M / Q5_K_M / Q8_0 / IQ1_M / Q2_K GGUFs of Gemma 4 31B-it
- Unsloth team (Daniel Han et al.) for UD-IQ2_XXS / UD-Q2_K_XL Dynamic 2.0 quantizations
- llama.cpp / ggml.ai (Georgi Gerganov et al.) for the GGUF format, perplexity tool, and inference engine
- Google DeepMind for releasing Gemma 4
