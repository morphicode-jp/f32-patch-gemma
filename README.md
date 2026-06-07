# f32-patch-gemma

**8-byte F32 patch on Gemma 4 31B. Q4_K_M beats the Q8 BF16 reference on HellaSwag, Winogrande, and ARC-Challenge. For GSM8k patch effect, cite the Q2_K paper-grade n=500 same-quant re-measurement; Q4_K_M baseline-only n=500 is stronger at 95.80%.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20362821-blue)](https://doi.org/10.5281/zenodo.20362821)
[![HuggingFace Q1](https://img.shields.io/badge/🤗-Q1__IQ1__M-yellow)](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M)
[![HuggingFace Q2](https://img.shields.io/badge/🤗-Q2__K-yellow)](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K)
[![HuggingFace Q4](https://img.shields.io/badge/🤗-Q4__K__M-yellow)](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M)

## What this is

A small Python script that modifies **8 bytes** of any Gemma 4 31B GGUF file (two F32 scalars on `layer_output_scale` at layers 25 and 26, each multiplied by 1.5) and measurably improves benchmark accuracy across all standard llama.cpp quantizations (IQ1_M through Q8_0), without any retraining, calibration, or inference overhead.

This repo also ships the bake scripts for the per-layer ablation experiments described in the paper: `bake_l25_l26_split.py` (L25-alone or L26-alone, 4 bytes each) and `bake_triple_patch.py` (a 12-byte triple-tensor control that is a documented Goodhart's Law instance).

## Quick start

```bash
pip install gguf numpy

# Apply the paper v1 patch (L25+L26 ×1.5, 8 bytes) to any Gemma 4 31B GGUF
python apply_l25l26.py /path/to/gemma-4-31b.gguf

# Optional: bake the per-layer ablation variants (L25-alone or L26-alone)
python bake_l25_l26_split.py /path/to/gemma-4-31b-Q2_K.gguf --layer 25
python bake_l25_l26_split.py /path/to/gemma-4-31b-Q2_K.gguf --layer 26

# Undo (apply_l25l26.py creates .backup automatically)
python apply_l25l26.py /path/to/gemma-4-31b.gguf --restore
```

## Pre-patched GGUF releases on HuggingFace

The paper v1 patch (L25+L26 ×1.5, 8 bytes) is distributed at three quantization levels:

| Quant | Size | Notable result | HF link |
|---|---|---|---|
| **IQ1_M** (1-bit) | ~10 GB | GSM +36pt (chance-level → usable) | [morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) |
| **Q2_K** (2-bit) | ~13 GB | HS +11.21pt (flagship for paper v5) | [morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) |
| **Q4_K_M** (4-bit) | ~19 GB | Q4 patched beats Q8 BF16 baseline on HS/WG/ARC; GSM8k caveat in results | [morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M) |

L25-alone, L26-alone, and the triple control GGUFs are **not** distributed as separate downloads — they are reproducible from the bake scripts in this repo. Distribution would invite users to mistake the diagnostic ablation patches for recommended deployment patches.

## Results — paper v1 (publicly released 2026-05-27)

Gemma 4 31B-it, full validation:

| Benchmark | Baseline | Patched | Δ |
|---|---|---|---|
| HellaSwag (Q4_K_M vs Q8_0 BF16) | 63.33% | **73.50%** | **+10.17pt** |
| GSM8k (Q4 vs Q8 BF16, paper v1 short-budget legacy) | 70% | **87%** | **+17pt** (not n=500; mixes capability + token-budget) |
| GSM8k (Q4_K_M baseline-only, n=500 ctx=16384) | **95.80%** | — | 479/500, Wilson CI [93.66, 97.24]; Q4 patched not re-run |
| Winogrande (Q4 vs Q8 BF16) | 65.59% | **70.32%** | +4.73pt |
| ARC-Challenge (Q4 vs Q8 BF16) | 44.38% | **48.76%** | +4.38pt |
| HellaSwag (Q2_K vs same-quant baseline) | — | — | **+11.21pt** |
| GSM8k (IQ1_M vs same-quant baseline) | 24% | **60%** | **+36pt** |

→ 12/12 cells positive across 3 release quants × 4 benchmarks under their documented protocols. For Q4_K_M vs Q8_0, cite HellaSwag / Winogrande / ARC-Challenge first. For GSM8k patch effect, cite the Q2_K n=500 paper-grade same-quant result below; do not claim Q2_K beats Q4_K_M on GSM8k.

### Note on Q2_K GSM +9pt (convergence vs. capability)

The Q2_K GSM +9pt at `n_predict=1024` decomposes into two effects we separated via extended-context re-measurement. Re-running at `ctx=16384, n_predict=8192, n=500` yields baseline 87.80%, patched 93.20%, delta **+5.40pt** (McNemar two-sided **p=0.0007**, highly significant; paired breakdown 44/17 patch/baseline-only-correct out of 61 discordant pairs). Cap-hit verification at ctx=32768/n_predict=16384 on the 7 boundary cases (3 baseline + 4 patched cap-hits) shifts totals to baseline 88.00%/patched 93.40% with delta unchanged at +5.40pt (p=0.0009) — the +5.4pt is robust to token-budget concerns. The headline +9pt thus splits into approximately +3.6pt convergence-efficiency (the patched model finishes its chain-of-thought within budget more often) and **+5.4pt capability gain** (statistically established at p<0.001). The multi-choice benchmarks (HS/WG/ARC) are log-likelihood scored with no generation and are unaffected; those numbers stand. Q4_K_M baseline-only has now been re-run under the same GSM8k n=500/ctx=16384 protocol and scores 95.80%, higher than Q2_K patched; Q4_K_M patched and IQ1_M GSM patch rows remain legacy n=100 short-budget results. Related prior art: [arXiv:2602.09805](https://arxiv.org/abs/2602.09805) (token efficiency decomposition), [arXiv:2605.07686](https://arxiv.org/abs/2605.07686) (coupling tax under shared token budget). See `HF_MODEL_CARD.md` §Methodology note for the full version.

## Paper v5 contribution — per-layer functional specialization (n=1 pilot)

Splitting the same 8-byte patch into single-layer ablations on a recurrence-relation prompt (n=1 pilot, hypothesis-generating only):

| Patch | mid-stream "Wait" | tail reflex | aux explanation | Coin Q correct |
|---|---|---|---|---|
| baseline (no patch) | 8.25 | 16% | 22% | 91% |
| L25-alone ×1.5 (4 byte) | 5.56 | 3% | 16% | 100% |
| L26-alone ×1.5 (4 byte) | 7.41 | 9% | 56% | 97% |
| **L25+L26 mix (paper v1, 8 byte)** | 5.81 | 6% | 28% | **94%** |
| triple LOS27+PAN17+PAN43 ×1.8 (12 byte) | 9.25 | 31% | 12% | **12.5%** ⚠ |

The n=32 follow-up (exp166) statistically confirms an **asymmetric auxiliary-explanation rate** (L26-alone 56% vs L25-alone 16%, Fisher p<0.01) while finding accuracy differences not statistically significant at n=32.

**Speculative hypothesis (B+, pilot-only)**: L25 ≈ compute-confidence specialist, L26 ≈ verify-proposal specialist; the L25+L26 mix appears to produce an explicit terminal-verification reflex absent in either single-layer condition. Confirmation requires logit-lens / activation-patching at L24–L27.

## Practical Use Note — Goodhart's Law instance (triple patch)

exp169 (triple × 4 prompts × 32 seeds, interactive T=0.7, max_tokens=6144) found the triple patch solves a coin probability prompt only **4/32 (12.5%)** vs paper v1 mix **30/32 (93.8%)** (Fisher exact p < 0.0001). The triple emits the most verification markers (mid_wait 9.25, tail reflex 31%) yet has the **lowest** interactive accuracy — silent_slip rate 89.3%.

The triple's high GSM8K benchmark score under T=0.0 / max_tokens=1024 / regex extraction is explained by the benchmark setting short-circuiting the "doubt loop" that fires under interactive conditions. **Use the L25+L26 mix at Q2_K for interactive deployment**; the triple is documented here only to motivate the Goodhart's Law caution.

## Cross-architecture validation (paper v1)

| Model | Architecture | Best Δ |
|---|---|---|
| **Gemma 4 31B** | hybrid (5:1 full:SWA) | **+10pt mean across 4 benches** ⭐ |
| Qwen 3.6 27B | hybrid (1:3 full:SWA) | +2.5pt |
| Phi-4 14B BF16 | pure dense | null |
| Llama 3.1 / Mistral 7B | pure dense | null |

→ The F32 patch effect is concentrated on hybrid LLMs. Pure dense transformers do not benefit. The earlier "rare full-attention layer slack" framing has been revised: structural GGUF analysis shows the working L25/L26 sites are sliding-window (not full-attention) layers. The current accurate description: `layer_output_scale` is a per-layer F32 scalar that gates how strongly each block's normalized output is written back to the residual stream, and we amplify L25/L26's gates by 1.5×. *Why* this specific layer pair works on this specific architecture family remains an open question — see paper §6 for the discussion.

## Reproducibility

```bash
# 1. Get any Gemma 4 31B GGUF (any quantization)
hf download bartowski/google_gemma-4-31b-it-Q2_K \
    --local-dir ./models

# 2. Apply paper v1 patch (8 bytes)
python apply_l25l26.py ./models/google_gemma-4-31b-it-Q2_K.gguf

# 3. Evaluate (requires llama.cpp)
~/llm/llama.cpp/llama-perplexity \
    -m ./models/google_gemma-4-31b-it-Q2_K.gguf \
    -f ~/llm/benchmarks/hellaswag_val.txt \
    -ngl 99 --hellaswag --hellaswag-tasks 10042 \
    -c 512
```

Expected on Q2_K HellaSwag (n=10042): baseline approximately 60%, patched approximately +9–11pt.

## Files

- [`apply_l25l26.py`](apply_l25l26.py) — paper v1 L25+L26 ×1.5 patcher (single file, 8 bytes)
- [`bake_l25_l26_split.py`](bake_l25_l26_split.py) — per-layer ablation baker (L25-alone or L26-alone, 4 bytes each)
- [`bake_triple_patch.py`](bake_triple_patch.py) — triple control baker (LOS27+PAN17+PAN43 ×1.8, 12 bytes; not recommended for deployment)
- [`LICENSE`](LICENSE) — Apache 2.0 base text
- [`LICENSE-CODE`](LICENSE-CODE) — code/tooling scope (Apache 2.0)
- [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS) — patched GGUF/model-weight scope (Gemma 4 Apache 2.0 basis, verified 2026-05-31)
- [`CITATION.cff`](CITATION.cff) — citation metadata
- [`REPRODUCE.md`](REPRODUCE.md) — step-by-step reproduction guide
- [`ROLLBACK_PLAYBOOK.md`](ROLLBACK_PLAYBOOK.md) — release rollback playbook
- `README.md` — this file

No dependencies beyond `gguf` and `numpy`. No conda env. No build step.

## Background

This work was developed independently by a researcher in Japan on nights and weekends. The 8-byte patch (L25+L26 ×1.5) was identified after a wider exploration that produced an 11-layer 44-byte solution (basin B) via an in-house multi-specialist parallel optimization engine; the simpler 2-layer baseline beat that solution on every release benchmark and is what we ship here. The internal optimization engine itself is not included in this release. Basin B numerical values are kept in the paper appendix for transparency only.

## Releases planned

- **v1.0** (paper v1, 2026-05-27): 8-byte L25+L26 patch + 3 HF GGUFs (Q1/Q2/Q4)
- **v1.1** (paper v5, this release, 2026-06-07): per-layer ablation (L25/L26 alone, triple) + exp166 statistical confirmation + Goodhart's Law instance (triple) documentation
- **v1.2** (planned within ~2 weeks): exp166 full 512-run statistics integrated; cross-model replication (Qwen 3.6 / Phi-4)
- **v2.0** (future): logit-lens / activation-patching at L24–L27 for direct mechanism verification

## Citation

```bibtex
@misc{hirai2026-l25l26-functional-asymmetry,
  title  = {Adjacent-Layer Functional Specialization in Q2_K Quantized
            Gemma 4 31B: A Single-Question Pilot of L25 (Compute) and
            L26 (Meta-Verify)},
  author = {Hirai, Akito},
  year   = {2026},
  month  = {June},
  doi    = {10.5281/zenodo.20362821},
  url    = {https://doi.org/10.5281/zenodo.20362821}
}
```

## License

This release keeps three scopes separate:

1. Code/tooling: Apache 2.0. See `LICENSE-CODE`.
2. Patched Gemma 4 GGUF weights: Gemma 4 Apache 2.0 basis as listed by Google and Hugging Face (verified 2026-05-31; Gemma 4 was moved off the older Gemma Terms of Use to Apache 2.0). See `LICENSE-WEIGHTS`.
3. Paper/docs: CC-BY-4.0, archived on Zenodo (DOI 10.5281/zenodo.20362821).

Before redistributing patched GGUF files, re-check the upstream Gemma 4 model card and Google license page because model licensing is an external dependency.

## Contact

- X (Twitter): [@morphicode_jp](https://x.com/morphicode_jp)
- HF: [huggingface.co/morphicode-jp](https://huggingface.co/morphicode-jp)
- Zenodo: [doi.org/10.5281/zenodo.20362821](https://doi.org/10.5281/zenodo.20362821)
- Email: morphicode.jp@gmail.com

DMs open for research collaboration and questions. Feedback / reproductions / refutations welcome.
