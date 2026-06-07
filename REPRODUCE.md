# Reproduction Guide — Paper v5 (L25+L26 mix, 8 bytes)

Step-by-step instructions to reproduce the paper v1 patch (L25+L26 ×1.5, 8 bytes) on Gemma 4 31B Q2_K. Expected effect: HellaSwag improvement of approximately +9 to +11pt depending on the eval setting (Wilson CI varies with N; full N=10042 numbers are in the paper). For the per-layer ablation experiments described in paper v5 (L25-alone, L26-alone, triple), see Section 9 below.

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
# Option A: Original base GGUF (unpatched, then apply the patch yourself)
huggingface-cli download bartowski/google_gemma-4-31b-it-Q2_K \
    --local-dir ./models

# Option B: Pre-patched (faster) — pick the quant you want
huggingface-cli download morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K \
    --local-dir ./models
# (or -IQ1_M or -Q4_K_M)
```

### 3. Get HellaSwag dataset

```bash
wget https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl \
    -O ./hellaswag_val.txt
# (or use the .txt format llama-perplexity expects;
#  many community mirrors exist)
```

### 4. Baseline measurement (Option A path)

```bash
./llama.cpp/llama-perplexity \
    -m ./models/google_gemma-4-31b-it-Q2_K.gguf \
    -f ./hellaswag_val.txt \
    -ngl 99 --hellaswag --hellaswag-tasks 400 -c 512
```

Expected: approximately **57%–63%** depending on llama.cpp version and N (Wilson 95% CI varies; full N=10042 numbers in the paper).

### 5. Apply paper v1 patch (Option A path)

```bash
git clone https://github.com/morphicode-jp/f32-patch-gemma
python f32-patch-gemma/apply_l25l26.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf
```

This applies `layer_output_scale × 1.5` at layers 25 and 26 (8 bytes total) and creates a `.backup` of the original file. Use `--restore` to undo.

### 6. Patched measurement

```bash
./llama.cpp/llama-perplexity \
    -m ./models/google_gemma-4-31b-it-Q2_K.gguf \
    -f ./hellaswag_val.txt \
    -ngl 99 --hellaswag --hellaswag-tasks 400 -c 512
```

Expected: paper v1 reported approximately **+11.21pt over the same-quant baseline at Q2_K HellaSwag**. With N=400 the Wilson CI will be wide; for sharper bounds use `--hellaswag-tasks 10042`.

### 7. Verify

If CIs overlap at N=400, increase to N=1000 or N=10042 to tighten. The directionality (patched > baseline) should be visible even at smaller N.

### 8. Restore (optional)

```bash
python f32-patch-gemma/apply_l25l26.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf --restore
```

## Reproducibility checks

- MD5 of patched file should match the value in `HF_MODEL_CARD.md`
- Per-layer F32 scale values can be inspected with `python apply_l25l26.py <file> --dry-run`
- HellaSwag accuracy varies ±1-2pt across N=400 runs; use N=1000+ for tighter bounds

## Variations to test (paper v1 cross-quant)

- Apply to IQ1_M (10 GB): expected baseline ~24%, patched ~60% on GSM8k (paper v1 +36pt headline)
- Apply to Q4_K_M (19 GB): expected baseline ~63% Q8 BF16 reference, patched ~73.5% on HellaSwag — Q4 patched beats Q8 baseline 4-for-4 on the public release benchmarks
- Apply to Q8_0 (33 GB): expected baseline ~78%, patched ~89% (smaller absolute gap because Q8 is already strong)

(Full cross-quant table in the Zenodo archive paper, DOI 10.5281/zenodo.20362821.)

## 9. Paper v5 per-layer ablation (optional, advanced)

Paper v5 introduces three diagnostic ablation variants. They are NOT distributed as HF downloads and must be baked locally from an unmodified base GGUF.

```bash
# L25 alone (4 bytes, "compute specialist" hypothesis)
python f32-patch-gemma/bake_l25_l26_split.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf --layer 25 --scale 1.5

# L26 alone (4 bytes, "verify-proposal" hypothesis)
python f32-patch-gemma/bake_l25_l26_split.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf --layer 26 --scale 1.5

# Triple (12 bytes, paper v4 candidate, DOCUMENTED Goodhart's Law instance — wins GSM8k benchmark but collapses under interactive use)
python f32-patch-gemma/bake_triple_patch.py \
    ./models/google_gemma-4-31b-it-Q2_K.gguf --scale 1.8
```

After baking, run the same eval steps (4 and 6 above) to reproduce the per-layer phenotype observations described in paper §5. The exp166 follow-up (4 patches × 4 prompts × 32 seeds = 512 runs) is described in paper §5.5; the runner is at `exp166_b_plus_verification.py` in the same GitHub repo.

**Important**: the triple variant is included for documentation completeness only. It exhibits a strong Goodhart's Law failure mode (silent_slip 89.3% under interactive T=0.7) and should NOT be used for deployment. Use `apply_l25l26.py` (the paper v1 mix) for any actual interactive use.

## Troubleshooting

- "No layer_output_scale tensors found": this is not Gemma 4. Other model families have different F32 tensor names.
- "Found only N tensors": likely a different Gemma 4 size (9B, 27B); the patch is calibrated for 31B specifically.
- llama-perplexity returns 0%: check `--hellaswag-tasks` arg and dataset format.
- Accuracy is way off baseline: try restoring (`--restore`) and ensure no other patches were applied.
- "basin B" or `apply_basin_b.py` referenced somewhere in your local clone: that is the paper v3 candidate research line, archived under `publish/_archive/basin_b_v1/` and NOT part of this release. See `_archive/NOTE.md` for context.
