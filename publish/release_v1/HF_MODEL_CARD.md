---
license: apache-2.0
base_model: google/gemma-4-31b-it
tags:
  - gemma
  - quantization
  - f32-patch
  - per-layer-ablation
  - hypothesis-generating-pilot
language:
  - en
  - ja
library_name: gguf
pipeline_tag: text-generation
---

# Gemma 4 31B Q2_K with Per-Layer F32 Patches — Paper v5 Bundle

**Status**: `n=1 pilot, hypothesis-generating` ((NOT a confirmed mechanistic claim))
**Date**: 2026-05-30
**Predecessor**: paper v1 ((L25+L26 ×1.5, 8 bytes)) — public release showed dramatic gains across quantization levels:
- **Q4_K_M patched ((19 GB)) > Q8_0 BF16 baseline ((31 GB))** on all 4 benchmarks: HS 73.50% > 63.33% ((+10.17)), GSM 87% > 70% ((+17)), WG 70.32% > 65.59% ((+4.73)), ARC 48.76% > 44.38% ((+4.38))
- **Q2_K HellaSwag +11.21pt**, **IQ1_M GSM +36.0pt** ((largest gain across 12 cells))
- exp162 reference rerun for this paper v5 release ((reduced N: HS 1000 / WG 500 / ARC 200 / GSM 100)) reproduces the directionality with +9.13pt avg ((variance higher due to small-N reference setting))

## Quick start

```bash
# llama.cpp
llama-server -m gemma-4-31B-it-L25L26x1.5-Q2_K.gguf -ngl 99 -c 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 -fa on \
  --host 127.0.0.1 --port 8080

# Open http://localhost:8080
```

## The pilot observation ((n=1, hypothesis-generating))

Applying 4-byte F32 scale patches ((×1.5)) to **single layers** of Gemma 4 31B Q2_K **appears to give** different behavioral phenotypes on one recurrence-relation prompt ((n=1 each, single-question pilot)):

| Patch | Bytes | Coin Q ((正 解 5/16)) answer | mid-stream "Wait" | Final reflex | Tokens |
|---|---|---|---|---|---|
| L25 alone ×1.5 | 4 | ✅ 5/16 | 1 | ❌ none | 2467 |
| L26 alone ×1.5 | 4 | ✅ 5/16 | **4** ((appears obsessive)) | ❌ none | **3278 ((max))** |
| L25 + L26 ×1.5 ((paper v1)) | 8 | ✅ 5/16 | 1 | ⭐ **explicit closing** | **2359 ((min))** |
| triple LOS27+PAN17+PAN43 ×1.8 | 12 | ❌ 11/32 ((calc slip)) | 0 | ❌ | 2726 |

**Speculative Hypothesis B+**: L25 ≈ compute-confidence specialist, L26 ≈ verify-proposal specialist ((without intrinsic self-termination)). Joint scaling ((×1.5 of both)) **appears to** produce a terminal-verification reflex absent in either single-layer condition. This is a hypothesis suggested by the pilot, not a confirmed mechanism.

**Cross-quantization sanity check**: Same paper v1 patch tested on Q4_K_M gives the same answer ((5/16)) on the same prompt. The closing reflex is preserved in spirit ((appears as mid-stream "let me double check" + direct case enumeration)) rather than the exact Q2 form, suggesting the broad pattern survives the 4× weight-precision change — but a single prompt cannot establish quantization-robustness.

## Companion HuggingFace repos (distributed)

The paper v1 patch (L25+L26 ×1.5, 8 bytes) is distributed at three quantization levels:

| HF repo | Quant | Size | Highlight |
|---|---|---|---|
| [morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M) | IQ1_M | ~10 GB | 1-bit revival, +36pt GSM (paper v1) |
| [morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) | Q2_K | ~13 GB | Default for paper v5 narrative, +11.21pt HellaSwag (paper v1) |
| [morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M) | Q4_K_M | ~19 GB | Q4 patched beats Q8 BF16 baseline 4-for-4 (paper v1) |

Each file contains the same 8-byte F32 patch (layers 25 and 26 `layer_output_scale` × 1.5) applied to the corresponding llama.cpp standard quantization of `google/gemma-4-31B-it`.

## Per-layer ablation GGUFs (NOT distributed, reproducible via bake scripts)

The L25-alone, L26-alone, and triple (LOS27+PAN17+PAN43 ×1.8) variants discussed in this paper are NOT distributed as separate HuggingFace downloads. They are reproducible from the bake scripts in the GitHub repo `morphicode-jp/f32-patch-gemma`:

- `bake_l25_l26_split.py` → `gemma-4-31B-it-L25x1.5-Q2_K.gguf` or `gemma-4-31B-it-L26x1.5-Q2_K.gguf` (4 bytes each)
- `bake_triple_patch.py` → `gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf` (12 bytes)

Reason: the per-layer ablations are diagnostic — they were run to understand which part of the L25+L26 mix contributes what. Distribution would invite users to mistake them for recommended deployment patches. Triple in particular is a Goodhart's Law instance (see §Practical Use Note below): it wins GSM8K under benchmark conditions but collapses to 12.5% accuracy under interactive use.

## Reproducible test

Try this prompt with each model and observe the **behavioral differences**:

```
コイン を 6 回 投 げ た と き、 表 が 3 回 連 続 し て 出 る 確 率 を 求 め て く だ さ い。
((「途 中 に 出 て も OK」 で は な く 「最 低 1 回 連 続 3 回」 の 意 味))
```

Expected ((from our n=1 pilot)):
- **L25 alone**: solves correctly ((5/16)) with 1 mid-stream "Wait"
- **L26 alone**: solves correctly ((5/16)) but inserts **multiple** "Wait, let me re-verify" + spontaneous supplementary explanation
- **paper v1**: solves correctly ((5/16)) and inserts canonical closing "**Double check the a_n sequence:**" reflex with re-enumeration

## Practical Use Note ((重 要 — benchmark gain ≠ interactive reliability))

exp169 ((triple × 4 prompts × 32 seeds, exp166 bit-identical setting)) demonstrated a Goodhart's Law instance in this patch family:

| Patch | mid_wait | tail_reflex | aux | coin correct ((n=32)) |
|---|---|---|---|---|
| baseline | 8.25 | 16% | 22% | 91% |
| L25 alone | 5.56 | 3% | 16% | **100%** |
| L26 alone | 7.41 | 9% | 56% | 97% |
| paper v1 mix | 5.81 | 6% | 28% | **94%** |
| triple | **9.25** | **31%** | 12% | **12.5%** ⚠ |

triple emits the **most** verification markers ((mid_wait + tail_reflex)) yet has the **lowest** interactive accuracy ((Fisher exact p < 0.0001 vs paper v1)). Failure mode classification: 89.3% silent_slip ((confident wrong numeric answer))。 mechanism interpretation: triple's verify pathway is over-active **but mis-functions** — it enters a "doubt loop" where each re-check produces a new wrong answer rather than catching the original error.

triple's high GSM8K benchmark score ((85% > paper v1 76%)) is explained by the benchmark's T=0.0 / max_tokens=1024 / regex-extraction setting, which short-circuits the doubt loop. Under interactive conditions ((T=0.7, max_tokens > 4000)) the doubt loop fires and accuracy collapses.

**Recommendation: download [`morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K`](https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K) (paper v1 mix at Q2_K) for interactive use cases.** L25-alone and L26-alone are half-strength alternatives (4 bytes each) that remain functional in interactive use, but are NOT distributed as separate HF downloads — bake them locally with `bake_l25_l26_split.py` if you want to reproduce the per-layer phenotype. Triple is documented here as a Goodhart's Law instance and is similarly NOT distributed; reproduce with `bake_triple_patch.py` only if you specifically want to study the benchmark-gaming phenotype.

## Critical caveats ((please read))

```
🔴 This is an n=1 single-question pilot observation. NOT a confirmed mechanism.

🔴 Multiple uncontrolled confounders (sampling stochasticity, prompt-class bias,
   scale-vs-layer confound, order effects, GPU thermal drift, KV cache state,
   coder bias, definition gaming, selection bias, limited cross-quantization
   comparison). Detailed in paper §8.

🔴 For a confirmed claim, exp166 (4 patches × 4 prompt categories × 32 seeds = 512 runs
   with pre-registered regex and blind coding) is required. Currently running, results
   expected within ~2 weeks (paper v5.1 update).

🔴 Cross-model generality untested at this depth. Prior internal F32 patch work on
   hybrid LLMs with rare full-attention layers suggests effects concentrate on Gemma 4
   (stronger), Qwen 3.6 (weaker), Phi-4 (none observed) — but those measurements are
   separate from this paper v5 claim and not part of this bundle.
```

## How to bake your own patches

```python
import struct
import numpy as np
# (See bake_l25_l26_split.py / bake_triple_patch.py in the repo)

# Locate F32 tensor by name and layer index
info = find_f32_tensor("layer_output_scale", layer=25)

# Read original, multiply by scale, write back as F32
orig = read_f32_vector(info["offset"], info["n_elements"])
new = orig * 1.5  # scale factor
write_f32_vector(out_gguf_path, info["offset"], new)
```

Total patch size: 4 bytes per layer × number of patched layers. For paper v1: 2 layers = 8 bytes total.

## Related research

- **paper v1** ((2026-05-27 release)): initial L25+L26 ×1.5 = 8 byte F32 patch for Gemma 4 31B. Public release demonstrated Q4 patched beating Q8 BF16 baseline on all 4 benchmarks, Q2 HS +11.21pt, IQ1 GSM +36pt. All quantization levels showed positive effect ((12-cell matrix, all positive)). This is the established finding that paper v5 builds on.
- **paper v5** ((this release, 2026-05-30)): per-layer ablation of paper v1 — pilot observation of L25 vs L26 behavioral asymmetry; exp166 statistical follow-up confirms verify-pattern asymmetry ((L26 aux 56% vs L25 alone 16%, p < 0.01)) while finding the coin-prompt accuracy difference is not statistically significant at n=32 ((Fisher exact p=0.49)).

((Other internal work in this research line — basin B 11-layer patch, MoE patch, etc. — is separate from this paper v5 release and not represented in this bundle.))

## Citation ((draft, pending exp166 confirmation))

```bibtex
@misc{hirai2026-l25l26-functional-asymmetry,
  title  = {Adjacent-Layer Functional Specialization in Q2_K Quantized
            Gemma 4 31B: A Single-Question Pilot of L25 (Compute) and
            L26 (Meta-Verify)},
  author = {Hirai, Akito},
  year   = {2026},
  month  = {June},
  note   = {n=1 hypothesis-generating observation; n=32 follow-up (exp166)
            confirms verify-pattern asymmetry (L26-alone aux 56% vs
            L25-alone 16%, p<0.01). Hosted on HuggingFace and archived
            on Zenodo (DOI: 10.5281/zenodo.20362821).},
  doi    = {10.5281/zenodo.20362821},
  url    = {https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K},
}
```

## Honest changelog

- 2026-05-30: Initial release ((n=1 pilot, hypothesis-generating))
- TBD: exp166 statistical results ((v5.1 update planned within 2 weeks))
- TBD: cross-model replication ((Qwen 3.6 / Phi-4))
- TBD: logit-lens probe of L24-L27 for direct mechanism observation

## License & attribution

- Base model: Google Gemma 4 31B-it ((Apache 2.0 as listed by Google/Hugging Face for Gemma 4))
- Quantization: llama.cpp ((MIT))
- F32 patches & observation: this repo
- Cognitive observation credit: independent collaborator hypothesis ((2026-05-30))

See `LICENSE-WEIGHTS` for the model-weight notice and `LICENSE-CODE` for
the patch/reproduction code notice. The paper text is released under
CC-BY-4.0 via Zenodo (DOI 10.5281/zenodo.20362821) and is kept separate
from code and model-weight licensing.
