# X Teaser Thread (5 tweets) — Posting May 24 (Sun) 22:00 JST

**Post time**: 2026-05-24 (Sun) 22:00 JST
**Strategy**: All English text. Teaser only — numbers visible, method (F32 / patch / byte / layer) completely hidden. Full reveal tomorrow May 25 (Mon) 22:00 JST.

**Images**: 4 generated (Tweet 1-4). Tweet 5 = text only.

---

## Tweet 1/5 (image 1: Two-upset hero)

```
1/5  1-bit jumps to 60%. 4-bit just beat Q8.

Gemma 4 31B-it just proved low-bit LLMs are back — and stronger than ever.

Remarkably:
1-bit (IQ1_M, 9.5 GB) on GSM8k → 24% → 60% (+36.0pt)
The largest improvement across all 12 cells.

And 4-bit patched beats Q8 BF16 baseline on every single benchmark.
No training. No calibration. No inference overhead.

The conventional wisdom on low-bit LLMs has been completely overturned.
```

---

## Tweet 2/5 (image 2: Q4 vs Q8 detail)

```
2/5  4-bit patched beats 8-bit baseline on every benchmark.

Q4_K_M patched (19 GB) vs Q8_0 BF16 baseline (31 GB):

HellaSwag    73.50% > 63.33%  (+10.17pt)
GSM8k        87.00% > 70.00%  (+17.00pt)
Winogrande   70.32% > 65.59%  (+4.73pt)
ARC-C        48.76% > 44.38%  (+4.38pt)

4-for-4 victory.
And ~40% smaller (19 GB vs 31 GB).
```

---

## Tweet 3/5 (image 3: 1-bit transformed)

```
3/5  1-bit patched took an extraordinary leap.

GSM8k:       24.0% → 60.0%   (+36.0pt) ← Largest improvement across all 12 cells
HellaSwag:   42.02% → 52.98% (+10.95pt)
Winogrande:  49.80% → 55.56% (+5.76pt)
ARC-C:       30.56% → 36.74% (+6.18pt)

Just 9.5 GB. Look how far it can fly.
```

---

## Tweet 4/5 (image 4: All 12 cells, stacked-bar visualization)

```
4/5  The patch works at every bit-level. Across the board.

1-bit  GSM8k +36.0pt
2-bit  HellaSwag +11.21pt
4-bit  GSM8k +15.0pt  (87% — beats Q8 baseline by +17pt)
4-bit  HellaSwag +9.80pt  (73.50% — beats Q8 baseline by +10.17pt)

12 cells, all positive. (See image for the full matrix.)
No training. No calibration. No inference overhead.
```

---

## Tweet 5/5 (no image)

```
5/5  Full reveal tomorrow (Mon May 25) · 22:00 JST.

• Paper
• 3 GGUFs (1-bit / 2-bit / 4-bit)
• Patch application script

All free, all open source.
The new era of low-bit LLMs begins.

Ready?

@morphicode_jp
Independent AI researcher (Japan)
```

---

# 🎨 Image Generation Prompts (4 images)

Common spec:
- **Size**: 1080×1080 px square (image 4 may go 1080×1440 if needed)
- **Background**: off-white `#fafafa`
- **Typography**: Inter or Helvetica Neue for labels; IBM Plex Mono for numbers
- **Color palette**: gray `#9ca3af` (baseline), emerald `#10b981` (gain), dark green `#047857` (delta labels), pale emerald `#ecfdf5` (callout bg)
- **Style**: flat, no gradients, no shadows, no 3D, no neon, no emoji, no decorative artwork
- **Forbidden text**: F32, scale, layer, byte, RMSNorm, 44, 8 byte
- **Date footer (all 4 images)**: `"Tomorrow (Mon 5/25) · 22:00 JST"`
- **Handle (all 4 images)**: `"@morphicode_jp · Independent AI researcher (Japan)"`
- **Spelling critical**: `@morphicode_jp`, `HellaSwag`, `Winogrande`, `ARC-C`, `GSM8k`, `Gemma 4 31B-it`, `Q1 (IQ1_M)`, `Q2 (Q2_K)`, `Q4 (Q4_K_M)`, `Q8 (Q8_0)` must render exactly

Recommended generator priority: **GPT-4o > Imagen 3 > Midjourney** (Midjourney often misspells text).

---

## 🎨 Image 1 Prompt — Two-Upset Hero (for Tweet 1/5)

```
Create a clean, premium tech research announcement infographic.
Square 1:1 (1080×1080 px). Off-white #fafafa background. Inter or
Helvetica Neue typography. IBM Plex Mono for all numbers.
Flat design — no gradients, shadows, 3D, emoji, or neon.

═════════════════════════════════════════════════════════════════
HEADER (top 20%)
═════════════════════════════════════════════════════════════════
Centered, two stacked lines:

Line 1 (massive bold black #0a0a0a, ~48pt):
    "1-bit jumps to 60%.   4-bit just beat Q8."

Line 2 (italic gray #6b7280, ~17pt):
    "Gemma 4 31B-it just proved low-bit LLMs are back — and stronger than ever."

Thin horizontal divider (#e5e7eb, 1px).

═════════════════════════════════════════════════════════════════
UPSET #1 BLOCK (next 26%) — "1-bit took the biggest leap"
═════════════════════════════════════════════════════════════════
Small label top-left (all caps, letter-spaced ~12pt, gray #6b7280):
    "UPSET #1   ·   GSM8k — largest gain of all 12 cells"

A single giant transformation:

LEFT:    "24.0%"   gray #9ca3af, mono, ~64pt
         tiny label below: "1-bit baseline"

CENTER:  "→"       emerald #10b981, ~80pt

RIGHT:   "60.0%"   emerald #047857 bold mono, ~88pt
         tiny label below: "1-bit patched"

Below this group, centered (~28pt, bold black):
    "+36.0 pt   ★"   (★ is solid black 5-point star)

═════════════════════════════════════════════════════════════════
UPSET #2 BLOCK (next 26%) — "4-bit beats 8-bit on every bench"
═════════════════════════════════════════════════════════════════
Small label top-left (~12pt, gray):
    "UPSET #2   ·   4-bit patched  >  Q8 BF16 baseline on ALL 4 benches"

Four compact horizontal bar-pair rows. Each row:
- Bench name (bold black, ~13pt, left)
- Gray Q8 baseline bar (#9ca3af)
- Emerald Q4 patched bar (#10b981), visibly longer than gray
- Right delta tag (bold #047857 mono, ~13pt)

  HellaSwag    [gray 63.33]  [emerald 73.50]   +10.17pt
  GSM8k        [gray 70.00]  [emerald 87.00]   +17.00pt
  Winogrande   [gray 65.59]  [emerald 70.32]   +4.73pt
  ARC-C        [gray 44.38]  [emerald 48.76]   +4.38pt

CRITICAL: in EVERY row, the emerald bar visibly extends beyond the
gray bar. Must be obvious at thumbnail (180×180 px).

═════════════════════════════════════════════════════════════════
CALLOUT (next 10%)
═════════════════════════════════════════════════════════════════
Two centered stacked lines:

Line 1 (bold dark green #064e3b, ~18pt):
    "Conventional wisdom on low-bit LLMs — overturned."

Line 2 (italic gray #6b7280, ~14pt):
    "No training. No calibration. No inference overhead."

═════════════════════════════════════════════════════════════════
FOOTER (bottom 18%)
═════════════════════════════════════════════════════════════════
Dotted divider (#e5e7eb).

Centered, three stacked lines:

Tiny eyebrow (all caps, ~11pt, gray #6b7280):
    "FULL REVEAL"

Massive bold (~32pt, black):
    "Tomorrow (Mon 5/25)  ·  22:00 JST"

Tiny gray italic (~11pt, #9ca3af):
    "@morphicode_jp   ·   Independent AI researcher (Japan)"

═════════════════════════════════════════════════════════════════
DO NOT INCLUDE: F32 / scale / layer / byte / RMSNorm / 44 / 8 byte
DO NOT INCLUDE: shadows, gradients, glow, 3D, emoji, photo backdrops

CRITICAL: render "@morphicode_jp", "HellaSwag", "Winogrande",
"Gemma 4 31B-it", all percentages EXACTLY as written.
```

---

## 🎨 Image 2 Prompt — Q4 vs Q8 Detail (for Tweet 2/5)

```
Create a focused 1-shot data comparison card. Square 1:1
(1080×1080 px). Off-white #fafafa, Inter for labels, IBM Plex Mono
for numbers. Flat design.

═════════════════════════════════════════════════════════════════
HEADER (top 20%)
═════════════════════════════════════════════════════════════════
Centered, two lines:

Line 1 (massive bold black, ~46pt):
    "4-bit patched beats 8-bit baseline."

Line 2 (gray #6b7280, ~20pt):
    "Q4_K_M patched (19 GB)   vs   Q8_0 BF16 baseline (31 GB)"

The "vs" rendered in bold emerald #10b981.

Thin divider (#e5e7eb).

═════════════════════════════════════════════════════════════════
DATA TABLE (middle 58%)
═════════════════════════════════════════════════════════════════
Four rows, evenly spaced. Each row contains:

LEFT (bench name): bold black ~22pt, ~150px column
CENTER (bars area): two stacked horizontal bars, shared x-axis 0-100%
  - Top: gray #9ca3af, tiny gray label above "Q8 baseline"
  - Bottom: emerald #10b981, tiny emerald label above "Q4 patched"
  - Bold white mono ~15pt percentage inside each bar tip
RIGHT (delta tag): bold #047857 mono, ~22pt, format "+X.XXpt"

Rows (top to bottom):

  HellaSwag    Q8 63.33%   →   Q4 73.50%     +10.17pt
  GSM8k        Q8 70.00%   →   Q4 87.00%     +17.00pt
  Winogrande   Q8 65.59%   →   Q4 70.32%     +4.73pt
  ARC-C        Q8 44.38%   →   Q4 48.76%     +4.38pt

EVERY row, emerald bar visibly extends past gray bar.

═════════════════════════════════════════════════════════════════
CALLOUT BAND (next 8%)
═════════════════════════════════════════════════════════════════
Pill-shaped horizontal band, pale emerald #ecfdf5 background,
emerald border #10b981 (1.5px), rounded corners (12px). Inside,
centered, bold dark green #064e3b, ~22pt:

    "4-for-4 victory  ·  ~40% smaller (19 GB vs 31 GB)"

═════════════════════════════════════════════════════════════════
FOOTER (bottom 14%)
═════════════════════════════════════════════════════════════════
Centered, two stacked lines:

Bold black ~26pt:
    "Tomorrow (Mon 5/25)  ·  22:00 JST"

Tiny gray ~11pt:
    "@morphicode_jp"

═════════════════════════════════════════════════════════════════
DO NOT INCLUDE: F32 / scale / byte / layer / RMSNorm
DO NOT INCLUDE: shadows, gradients, glow, 3D, emoji

CRITICAL: render "Q4_K_M", "Q8_0", "HellaSwag", "Winogrande",
"ARC-C", "@morphicode_jp", all percentages EXACTLY.
```

---

## 🎨 Image 3 Prompt — 1-bit Transformed (for Tweet 3/5)

```
Create a high-drama transformation card. Square 1:1 (1080×1080 px).
Off-white #fafafa, Inter for labels, IBM Plex Mono for numbers.
Flat design.

═════════════════════════════════════════════════════════════════
HEADER (top 14%)
═════════════════════════════════════════════════════════════════
Centered:

Tiny eyebrow (all caps, letter-spaced ~12pt, gray #6b7280):
    "1-BIT PATCHED   ·   IQ1_M (9.5 GB)   ·   Gemma 4 31B-it"

Massive bold black (~48pt):
    "1-bit, transformed."

Thin divider (#e5e7eb).

═════════════════════════════════════════════════════════════════
GIANT TRANSFORMATION (middle 26%) — GSM8k hero
═════════════════════════════════════════════════════════════════
Tiny centered label above (~14pt, gray): "GSM8k"

Single horizontal expression in giant numbers:

LEFT:    "24.0%"      gray #9ca3af, mono, ~88pt
         tiny gray label below (~12pt): "baseline"

CENTER:  "→"          emerald #10b981, ~108pt

RIGHT:   "60.0%"      emerald #047857, bold mono, ~120pt
         tiny emerald label below (~12pt): "patched"

Below this group, centered, ~36pt, bold black:
    "+36.0 pt   ★     extraordinary leap"

The "★" is a solid black 5-point star; "extraordinary leap" in
dark green #047857.

═════════════════════════════════════════════════════════════════
4-BENCH SUMMARY (next 28%)
═════════════════════════════════════════════════════════════════
Small label top-left (~12pt all caps gray): "ALL 4 BENCHMARKS"

Four compact horizontal bar-pair rows. Each row:
- Bench name (bold black ~14pt) left
- Gray baseline bar (#9ca3af) + emerald patched bar (#10b981) stacked
- Right delta tag (bold #047857 mono ~14pt)

  HellaSwag    [gray 42.02]  [emerald 52.98]    +10.95pt
  Winogrande   [gray 49.80]  [emerald 55.56]    +5.76pt
  GSM8k        [gray 24.0]   [emerald 60.0]     +36.0pt ★   ← highlight
  ARC-C        [gray 30.56]  [emerald 36.74]    +6.18pt

The GSM8k row has a pale yellow band (#fef9c3) wrapping it. Its
delta tag is ~18pt (larger than others).

═════════════════════════════════════════════════════════════════
CALLOUT (next 8%)
═════════════════════════════════════════════════════════════════
Pill horizontal band, pale emerald #ecfdf5, emerald border #10b981.
Inside centered, bold dark green #064e3b, ~17pt:

    "Largest gain across all 12 cells — just 9.5 GB."

═════════════════════════════════════════════════════════════════
FOOTER (bottom 14%)
═════════════════════════════════════════════════════════════════
Centered, two lines:

Bold black ~26pt:
    "Tomorrow (Mon 5/25)  ·  22:00 JST"

Tiny gray ~11pt:
    "@morphicode_jp   ·   Independent AI researcher (Japan)"

═════════════════════════════════════════════════════════════════
DO NOT INCLUDE: F32 / scale / byte / layer / RMSNorm
DO NOT INCLUDE: shadows, gradients, glow, 3D, emoji

CRITICAL: render "GSM8k" (not "GSM8K"), "HellaSwag", "Winogrande",
"IQ1_M", "Gemma 4 31B-it", "@morphicode_jp", percentages EXACTLY.
```

---

## 🎨 Image 4 Prompt — 12-Cell Stacked Bar Matrix (for Tweet 4/5)

**Design**: 3×4 grid of horizontal stacked bars (3 release quants × 4 benchmarks).
Each bar shows baseline (gray) + improvement chunk (emerald) on top. Reader sees
both the absolute value AND the gain at a single glance — uncluttered.
Q8_0 baseline shown only as a reference line below the matrix (not a row).

```
Create a data-dense yet airy result-summary infographic. Square 1:1
(1080×1080 px) — vertical (1080×1440) acceptable if needed for
breathing room. Off-white #fafafa background. Inter for labels,
IBM Plex Mono for numbers. Flat design — no gradients, shadows, 3D,
emoji.

═════════════════════════════════════════════════════════════════
HEADER (top 16%)
═════════════════════════════════════════════════════════════════
Centered, two stacked lines:

Line 1 (massive bold black #0a0a0a, ~44pt):
    "Every bit-level. Every benchmark."

Line 2 (italic gray #6b7280, ~16pt):
    "12 cells. All positive. Gemma 4 31B-it."

Thin divider (#e5e7eb).

═════════════════════════════════════════════════════════════════
MATRIX (middle 60%) — 3×4 stacked bars
═════════════════════════════════════════════════════════════════

Layout:
- 4 columns: HellaSwag · Winogrande · GSM8k · ARC-C
- 3 rows:    Q1 (9.5G) · Q2 (12G) · Q4 (19G)
  (Q8 baseline shown as a horizontal dotted reference line BELOW the matrix,
   not as a row — see "Q8 reference line" block.)

Column headers (top of grid, bold ~13pt, #1f2937, centered above each column):
    "HellaSwag"   "Winogrande"   "GSM8k"   "ARC-C"

Row labels (left of grid, bold ~13pt, #1f2937, right-aligned):
    "Q1 (9.5G)"
    "Q2 (12G)"
    "Q4 (19G)"

Each cell contains ONE horizontal stacked bar:
  • Bar height: ~22px (slightly taller now that there are 3 rows)
  • Bar total width: scales such that 100% fills the cell width
  • Gray segment (#9ca3af): from 0 to baseline%
  • Emerald segment (#10b981): from baseline% to patched%  ← the "gain chunk"
  • Generous whitespace between bars (vertical gap ~28px)
  • Right end of bar: delta tag in bold #047857 mono ~14pt, format "+X.XX"

Cell data (baseline% → patched% | delta tag shown):

  Q1 HellaSwag:    gray 0→42.02 + emerald 42.02→52.98    tag "+10.95"
  Q1 Winogrande:   gray 0→49.80 + emerald 49.80→55.56    tag "+5.76"
  Q1 GSM8k:        gray 0→24.0  + emerald 24.0→60.0      tag "+36.0 ★"  ← largest, gold star
  Q1 ARC-C:        gray 0→30.56 + emerald 30.56→36.74    tag "+6.18"

  Q2 HellaSwag:    gray 0→59.15 + emerald 59.15→70.36    tag "+11.21"
  Q2 Winogrande:   gray 0→59.35 + emerald 59.35→66.69    tag "+7.34"
  Q2 GSM8k:        gray 0→67.0  + emerald 67.0→76.0      tag "+9.0"
  Q2 ARC-C:        gray 0→43.61 + emerald 43.61→46.44    tag "+2.83"

  Q4 HellaSwag:    gray 0→63.70 + emerald 63.70→73.50    tag "+9.80"
  Q4 Winogrande:   gray 0→65.04 + emerald 65.04→70.32    tag "+5.28"
  Q4 GSM8k:        gray 0→72.0  + emerald 72.0→87.0      tag "+15.0"
  Q4 ARC-C:        gray 0→44.89 + emerald 44.89→48.76    tag "+3.87"

═════════════════════════════════════════════════════════════════
Q8 reference line (small annotation BELOW the 3x4 matrix)
═════════════════════════════════════════════════════════════════
Tiny gray italic line (~11pt, #6b7280, left-aligned under the matrix):
    "Q8_0 BF16-proxy baseline (reference):  HellaSwag 63.33 · Winogrande 65.59 · GSM8k 70.0 · ARC-C 44.38"

(No patched Q8 row — Q8 is not released. Q4 patched exceeds Q8 baseline
on all 4 benchmarks visible in the matrix above.)

CRITICAL: every cell has a VISIBLE emerald segment (even the smallest,
+2.83 for Q2 ARC-C). The 12-cell matrix reads as "all green, all positive"
at a single glance.

The Q1 GSM8k cell has a tiny gold star ★ next to the +36.0 tag
to mark "largest gain". Optionally a subtle pale-yellow row background
(#fef9c3) on the entire Q1 GSM8k cell to draw the eye.

═════════════════════════════════════════════════════════════════
CALLOUT BAND (next 8%)
═════════════════════════════════════════════════════════════════
Pill-shaped band, pale emerald #ecfdf5, emerald border #10b981
(1.5px), rounded corners (12px). Inside centered, bold dark green
#064e3b, ~18pt:

    "12 cells. All positive. No exceptions."

Tiny italic centered below (~13pt, gray #6b7280):
    "No training. No calibration. No inference overhead."

═════════════════════════════════════════════════════════════════
FOOTER (bottom 16%)
═════════════════════════════════════════════════════════════════
Dotted divider (#e5e7eb).

Centered, three stacked lines:

Tiny eyebrow (all caps, ~11pt, gray):
    "FULL REVEAL"

Massive bold (~30pt, black):
    "Tomorrow (Mon 5/25)  ·  22:00 JST"

Tiny gray italic (~11pt):
    "@morphicode_jp   ·   Independent AI researcher (Japan)"

═════════════════════════════════════════════════════════════════
DO NOT INCLUDE: F32 / scale / byte / layer / RMSNorm
DO NOT INCLUDE: shadows, gradients, glow, 3D, emoji

CRITICAL: render every quant label ("Q1 (9.5G)" etc.), every bench
name ("HellaSwag", "Winogrande", "GSM8k", "ARC-C"), every delta
("+X.XX"), and "@morphicode_jp" EXACTLY as written.

The visual punchline: 16 emerald gain segments, all clearly visible.
A wall of green chunks proving universal improvement.
```

---

# 📋 Posting procedure

1. **22:00 JST** — drop Tweet 1 with image 1 attached
2. **30-60 second intervals** — reply-thread Tweets 2 → 3 → 4 → 5 (image attached on 2/3/4; Tweet 5 text only)
3. **Pin Tweet 1** to profile (Profile → Pin to profile)
4. **Monitor replies / DMs** through the night.

Note: arXiv is not in scope for this launch (endorsement not obtained). The paper is archived on Zenodo (DOI 10.5281/zenodo.20362821) and that DOI is the canonical citation.

---

# 📊 Expected metrics (24h)

| Metric | Target |
|---|---|
| Tweet 1 impressions | 5k+ |
| Tweet 1 likes | 100+ |
| Tweet 1 RT / quote | 10+ |
| New X followers | 100+ |
| HF repo unique visitors | 200+ |
| GitHub stars | 50+ |

---

# ✅ Concurrent tasks (tonight, after teaser posts)

- [ ] Confirm Zenodo deposit is published with the correct paper PDF and DOI
- [ ] Confirm GitHub repo is public with the v1.0.0 tag attached
- [ ] Confirm all 3 HuggingFace repos (IQ1_M / Q2_K / Q4_K_M) are public and downloadable

---

# 🔧 If something goes wrong

| Issue | Mitigation |
|---|---|
| Tweet 1 not gaining traction in first 30 min | Reply to your own Tweet 1 with one extra bullet (e.g. cross-arch teaser) at 22:30 |
| Image AI misspells `@morphicode_jp` | Don't post the image. Regenerate, or post tweet text-only and add image later as a reply |
| HF repo accidentally still private after announcement | Switch public immediately via `huggingface-cli repo update --private false`, do not delete the tweet |
| Zenodo DOI page 500s under load | Mirror the PDF as a GitHub release asset and link that in a reply tweet |
