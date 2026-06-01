# Reproduction Guide

Step-by-step instructions to reproduce the +13.25pt HellaSwag improvement on Gemma 4 31B Q2_K.

## Requirements

- Hardware: ~24 GB free disk + GPU with ≥16 GB VRAM (or CPU with 32 GB RAM, slower)
- Python 3.10+
- llama.cpp with HellaSwag support

## Steps

### 1. Install dependencies

```bash
pip install gguf numpy
```

llama.cpp build (or download prebuilt):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make GGML_CUDA=1   # or just `make` for CPU
```

### 2. Get the GGUF

```bash
# Option A: Original (untouched, then apply patch yourself)
huggingface-cli download bartowski/google_gemma-4-31b-it-Q2_K \
    --local-dir ./models

# Option B: Pre-patched (faster)
huggingface-cli download morphicode_jp/gemma-4-31B-it-basin-b-Q2_K \
    --local-dir ./models
```

### 3. Get HellaSwag dataset

```bash
wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl \
    -O ./hellaswag_val.txt
# (or use the .txt format llama-perplexity expects;
#  many community mirrors exist)
```

### 4. Baseline measurement

```bash
./llama.cpp/llama-perplexity \
    -m ./models/google_gemma-4-31b-it-Q2_K.gguf \
    -f ./hellaswag_val.txt \
    -ngl 99 --hellaswag --hellaswag-tasks 400 -c 512
```

Expected: **57.00% ± 5%** (Wilson 95% CI [52, 62])

### 5. Apply basin B (if Option A)

```bash
git clone https://github.com/morphicode_jp/f32-patch-gemma
python f32-patch-gemma/apply_basin_b.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf
```

### 6. Patched measurement

```bash
./llama.cpp/llama-perplexity \
    -m ./models/google_gemma-4-31b-it-Q2_K.gguf \
    -f ./hellaswag_val.txt \
    -ngl 99 --hellaswag --hellaswag-tasks 400 -c 512
```

Expected: **70.25% ± 5%** (Wilson 95% CI [65, 75])

### 7. Verify

CIs should not overlap. If they do, increase to n=1000 or n=10042 to tighten.

### 8. Restore (optional)

```bash
python f32-patch-gemma/apply_basin_b.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf --restore
```

## Reproducibility checks

- MD5 of patched file should match the value in HF model card
- Per-layer F32 scale values can be inspected with `python apply_basin_b.py <file> --dry-run`
- HellaSwag accuracy varies ±1-2pt across n=400 runs; use n=1000+ for tighter bounds

## Variations to test

- Apply to IQ1_M (8.4 GB): expected baseline ~35%, patched ~50%
- Apply to Q4_K_M (18.5 GB): expected baseline ~76%, patched ~88%
- Apply to Q8_0 (33 GB): expected baseline ~78%, patched ~89%

(Full cross-quant table in upcoming arXiv preprint.)

## Troubleshooting

- "No layer_output_scale tensors found": this is not Gemma 4. Other model families have different F32 tensor names.
- "Found only N tensors": likely a different Gemma 4 size (9B, 27B); patch is calibrated for 31B specifically.
- llama-perplexity returns 0%: check `--hellaswag-tasks` arg and dataset format.
- Acc is way off baseline: try restoring (`--restore`) and ensure no other patches were applied.
