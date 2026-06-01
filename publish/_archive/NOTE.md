# publish/_archive/ — Prior research lines, NOT part of paper v5

This folder contains earlier release line drafts that are **separate from the paper v5 (2026-05-30) bundle** and should not be treated as recommended patches by downstream readers.

## Contents

### `POSTPONE_TWEET_UNUSED.md` — unused launch-postpone tweet draft

Originally written for a 2026-05-25 → 2026-05-27 paper v1 launch postponement. The actual paper v1 launch ran on 2026-05-27 without needing this tweet. Kept here so the historical narrative is intact but the file does not appear in the active publish set.

### `sns_draft_OLD_2026-06-10.md` — pre-paper-v1 SNS draft

Written for a planned 2026-06-10 launch (later moved to 2026-05-27 for paper v1). Contains numbers (HellaSwag +9pt mean, IQ1_M 34.75% → 48.25%, MMLU `+Xpt` placeholder) from an early calibration run that diverges from the paper v1 final public numbers (Q4 HS 73.50%, Q2 HS +11.21pt, IQ1_M GSM +36pt) and from paper v5 (no MMLU evaluation). Superseded by `publish/release_v1/REDDIT_POST.md`, `X_THREAD.md`, and `ZENN_ARTICLE_DRAFT.md`.

### `basin_b_v1/apply_basin_b.py` — bake script for basin B (NOT part of paper v5)

A copy of this script previously also lived at `publish/release_v1/apply_basin_b.py`. As of 2026-05-31 it was deleted from `release_v1/` because it is not part of the paper v5 release: basin B has not been verified under interactive conditions, paper v5 narrative recommends `bake_l25_l26_split.py` instead, and shipping the script alongside paper v5 would muddy the recommendation. The canonical copy here in `_archive/basin_b_v1/` is preserved for provenance and for any future basin B paper that might explicitly publish it.

### `basin_b_v1/` — paper v3 candidate (basin B, 11-layer 44-byte patch)

- 11 × `layer_output_scale` tensors scaled simultaneously, total 44 bytes
- Developed before paper v5's per-layer ablation work clarified the L25/L26 functional asymmetry
- **Not validated under interactive (T=0.7, max_tokens > 4000) settings** — given that the much smaller `triple` patch (12 byte) was found in exp169 to collapse from 85% GSM benchmark to 12.5% interactive coin accuracy (silent_slip 89.3%, Fisher p < 0.0001 vs paper v1), the basin B 11-layer patch is presumed to carry similar risk and has not been re-tested.
- Files include `apply_basin_b.py`, full HF/GitHub/Reddit/X release templates, CITATION.cff, LICENSE — these were prepared as a full publish bundle but **superseded by paper v5**

## Why archived (not deleted)

History matters: paper v5's Goodhart's Law finding (triple) is informative precisely because the patch-and-publish reflex on benchmark scores was a real prior research line. Keeping basin_b_v1/ in _archive preserves provenance while preventing accidental promotion to recommended status.

## What IS recommended for downstream use

See [`publish/release_v1/HF_MODEL_CARD.md`](../release_v1/HF_MODEL_CARD.md) and [`publish/release_v1/README_paper_v5.md`](../release_v1/README_paper_v5.md). The paper v1 mix (`gemma-4-31B-it-L25L26x1.5-Q2_K.gguf`, 8 byte) is the recommended interactive patch.
