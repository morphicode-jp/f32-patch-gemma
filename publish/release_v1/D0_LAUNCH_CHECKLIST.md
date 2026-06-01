# Day 0 Launch Checklist — Paper v5 (2026-06-02 22:00 JST)

**Target Day 0**: 2026-06-02 (Mon) 22:00 JST
**Scope**: 3 HF repos + Zenodo deposit + GitHub repo simultaneous public switch, then Reddit / Zenn / Twitter announcements.
**Note**: arXiv is NOT in scope for this launch (endorsement not obtained). Paper canonical archive is Zenodo (DOI 10.5281/zenodo.20362821).

Release infrastructure (already uploaded as **private**):

| Service | URL | State |
|---|---|---|
| HF Q1 | `morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M` | 🔒 Private |
| HF Q2 | `morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K` | 🔒 Private |
| HF Q4 | `morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M` | 🔒 Private |
| Zenodo | `zenodo.org/uploads/20362821` | 🔒 Draft (DOI 10.5281/zenodo.20362821 reserved) |
| GitHub | `github.com/morphicode-jp/f32-patch-gemma` | 🔒 Private |

---

## D-1 (2026-05-31 〜 2026-06-01): Pre-launch work

### Step 1: Final factual grep (送信前 safety net)

Run from project root:

```powershell
# All 5 main artifacts plus the paper:
$files = @(
  "publish/release_v1/EMAIL_TEMPLATES.md",
  "publish/release_v1/HF_MODEL_CARD.md",
  "publish/release_v1/ZENN_ARTICLE_DRAFT.md",
  "publish/release_v1/D0_LAUNCH_CHECKLIST.md",
  "publish/release_v1/README_paper_v5.md",
  "publish/release_v1/GITHUB_README.md",
  "publish/release_v1/REDDIT_POST.md",
  "publish/release_v1/X_THREAD.md",
  "publish/release_v1/X_TEASER.md",
  "publish/release_v1/REPRODUCE.md",
  "publish/release_v1/CITATION.cff",
  "publish/release_v1/ROLLBACK_PLAYBOOK.md",
  "publish/release_v1/HF_PRIVATE_PUBLIC_UPLOAD_CHECKLIST.md",
  "publish/release_v1/HF_README_Q1.md",
  "publish/release_v1/HF_README_Q2.md",
  "publish/release_v1/HF_README_Q4.md",
  "docs/PAPER_V5_DRAFT_2026-05-30.md"
)
# Confirm zero matches for each pattern below:
```

- [ ] `MMLU` (paper v5 does not evaluate MMLU; only the "not evaluated" honest mention is allowed)
- [ ] `44.byte` / `44 byte` / `11.layer` / `11 層` (basin B narrative, NOT part of paper v5)
- [ ] `13.25` / `12.94` (old basin B numbers)
- [ ] `arxiv.org/abs/XXXX.XXXXX` / `<arXiv URL>` (replaced with Zenodo DOI throughout)
- [ ] `<TBD>` / `<YOUR_NAME>` / `<github URL>` / `<HF URL>` / `<email>` (must all be concrete by D0)
- [ ] `morphicode_jp/` (the underscore form should not appear in HF or GitHub URLs; only the `@morphicode_jp` X handle is acceptable)
- [ ] `reveals` / `direct evidence` / `直接証拠` / `emerges` (overclaim verbs incompatible with n=1 pilot framing)

### Step 2: Contact verification (1 文字単位)

- [ ] **Neel Nanda email**: re-confirm `neelnanda27@gmail.com` via neelnanda.io / Twitter @NeelNanda5 bio / Google Scholar
- [ ] **Joshua Batson contact**: re-confirm `@thebasepoint` Twitter DM is open, or find email via anthropic.com team page / transformer-circuits.pub
- [ ] (Optional, for followup phase) Arthur Conmy / Tom Lieberum / Adam Jermyn / Trenton Bricken contacts

### Step 3: Anthropic Fellows cohort verification

- [ ] Open `https://www.anthropic.com/anthropic-fellows-program`, confirm current cohort name and application deadline are still what `EMAIL_TEMPLATES.md §4` assumes. If different, update §4 cover letter.

### Step 4: GPG + git config (optional but recommended)

- [ ] `gpg --list-secret-keys` to check if a GPG key exists
- [ ] If yes: `git config --global user.signingkey <YOUR_KEY_ID>` then `git config --global commit.gpgsign true`
- [ ] If no: skip the `-S` flag in the commit command below

### Step 5: Token setup (do NOT paste tokens in chat)

- [ ] Re-issue fresh tokens for HuggingFace / Zenodo / GitHub (previous tokens shared in chat are considered compromised)
- [ ] `hf auth login` with the new HF token
- [ ] `gh auth login` with the new GitHub token
- [ ] `$env:ZENODO_TOKEN = '<your-zenodo-token>'` in PowerShell (or write to `.env` that is in `.gitignore`)
- [ ] Confirm: `hf auth whoami`, `gh auth status`, `Test-Path env:ZENODO_TOKEN`

### Step 6: Git priority claim commit

```powershell
cd c:/Users/morph/開発/パイプラインオートメーション

# Stage paper v5 artifacts (individual file specification, never `git add -A`):
git add docs/PAPER_V5_DRAFT_2026-05-30.md
git add publish/release_v1/CITATION.cff
git add publish/release_v1/README_paper_v5.md
git add publish/release_v1/GITHUB_README.md
git add publish/release_v1/HF_MODEL_CARD.md
git add publish/release_v1/HF_README_Q1.md
git add publish/release_v1/HF_README_Q2.md
git add publish/release_v1/HF_README_Q4.md
git add publish/release_v1/HF_PRIVATE_PUBLIC_UPLOAD_CHECKLIST.md
git add publish/release_v1/ZENN_ARTICLE_DRAFT.md
git add publish/release_v1/EMAIL_TEMPLATES.md
git add publish/release_v1/D0_LAUNCH_CHECKLIST.md
git add publish/release_v1/ROLLBACK_PLAYBOOK.md
git add publish/release_v1/REPRODUCE.md
git add publish/release_v1/REDDIT_POST.md
git add publish/release_v1/X_THREAD.md
git add publish/release_v1/X_TEASER.md
git add publish/release_v1/LICENSE publish/release_v1/LICENSE-CODE publish/release_v1/LICENSE-WEIGHTS
git add publish/release_v1/apply_l25l26.py
git add publish/release_v1/requirements.txt
git add weight_analysis/bake_l25_l26_split.py
git add weight_analysis/bake_triple_patch.py
git add weight_analysis/exp166_b_plus_verification.py

# Top-level README warnings (paper v1 generic, now points to release_v1):
git add publish/HF_README.md
git add publish/GITHUB_README.md

# Archive moves (separate commit for clarity):
git add publish/_archive/NOTE.md
# Plus the moved files in _archive/ (basin_b_v1/*, POSTPONE_TWEET_UNUSED.md, sns_draft_OLD_2026-06-10.md)

# Do NOT add: GGUF files (too large, on HF), credentials, .env

# Commit (with GPG if set up):
git commit -S -m "paper v5 launch v1.0.0 (2026-06-02)"
# Without GPG:
# git commit -m "paper v5 launch v1.0.0 (2026-06-02)"

git tag -a v1.0.0 -m "paper v5 launch — L25/L26 per-layer ablation pilot"
```

### Step 7: opentimestamps anchor (internal record only — do NOT advertise as "blockchain proof" in public)

```powershell
pip install opentimestamps-client
# Stamp the commit hash:
git log -1 --format='%H' > .ots-target.txt
ots stamp .ots-target.txt
# 24 hours later, verify upgrade:
# ots verify .ots-target.txt.ots
# Archive the receipt:
# move .ots-target.txt.ots proofs/v5_commit.ots
```

### Step 8: Server cleanup

- [ ] Stop all `llama-server` processes to free VRAM (`Get-Process llama-server | Stop-Process`)
- [ ] Make sure no scheduled bake/eval jobs collide with D0 launch window

### Step 9: GitHub push (still private at this point)

```powershell
# Add remote if not already (Claude can run this once user has token):
# gh repo create morphicode-jp/f32-patch-gemma --private --source=.
# Or if remote exists:
git push origin master
git push origin v1.0.0
```

The repo stays private until the D0 launch sequence below flips it public.

---

## D0 (2026-06-02 Mon): Launch sequence

### T-30 min (21:30 JST): final checks

- [ ] User: confirm with Claude in chat ("公開準備") → Claude runs final grep across all files
- [ ] User: confirm `hf auth whoami` / `gh auth status` / `$env:ZENODO_TOKEN` all succeed
- [ ] User: open the HF repo URLs in a private browser tab to confirm they 404 (still private) — establishes baseline

### T-5 min (21:55 JST): pre-flight

- [ ] User: pin a "going live in 5 minutes" message in a draft Reddit/X tab
- [ ] Claude: prepare CLI commands (do not execute yet)

### T+0 (22:00 JST): simultaneous public switch — Claude executes

Claude runs these in quick succession:

```powershell
# 1. HF: switch all 3 repos to public via huggingface_hub Python API
python -c "from huggingface_hub import update_repo_settings; \
  [update_repo_settings(r, private=False, repo_type='model') for r in \
   ['morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M', \
    'morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K', \
    'morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M']]"

# 2. Zenodo: publish the deposit (this issues the DOI and makes it citeable)
$body = '{}'
Invoke-RestMethod -Uri "https://zenodo.org/api/deposit/depositions/20362821/actions/publish" `
  -Method POST -Headers @{ "Authorization" = "Bearer $env:ZENODO_TOKEN" } `
  -ContentType "application/json" -Body $body

# 3. GitHub: switch repo to public
gh api -X PATCH /repos/morphicode-jp/f32-patch-gemma -F private=false
```

Claude then verifies all 4 are public:

```powershell
# HF
foreach ($r in @("morphicode-jp/gemma-4-31B-it-L25L26x1.5-IQ1_M",
                 "morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K",
                 "morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q4_K_M")) {
  $ok = (Invoke-WebRequest -Uri "https://huggingface.co/$r" -UseBasicParsing -ErrorAction SilentlyContinue).StatusCode
  Write-Output "HF $r -> $ok"
}
# Zenodo
Invoke-WebRequest -Uri "https://doi.org/10.5281/zenodo.20362821" -UseBasicParsing
# GitHub
gh api /repos/morphicode-jp/f32-patch-gemma --jq '.private'  # should print "false"
```

If any check fails, see `ROLLBACK_PLAYBOOK.md` immediately.

### T+5 min (22:05 JST): Reddit announcement (User)

- [ ] User: post `REDDIT_POST.md` body to `r/LocalLLaMA`
- [ ] User: keep the title under 300 chars
- [ ] User: pin the top comment with a reproduction link

### T+15 min (22:15 JST): Zenn article (User)

- [ ] User: flip Zenn article `published: false` → `true`
- [ ] User: verify tags: `LLM`, `量子化`, `Gemma`, `ローカルLLM`, `interpretability`
- [ ] User: add Zenn URL to a reply on the Reddit post and on Twitter

### T+20 min (22:20 JST): Twitter announcement (User)

- [ ] User: post `X_THREAD.md` Tweet 1 with image; reply-chain Tweets 2-5 at 30-60s intervals
- [ ] User: pin Tweet 1 to profile
- [ ] User: quote-tweet from `@morphicode_jp` linking the Reddit and Zenn posts

### T+30 min (22:30 JST): r/MachineLearning cross-post (User, optional)

- [ ] User: cross-post to `r/MachineLearning` with the same body, framed slightly more academic
- [ ] User: link the r/LocalLLaMA post as "discussion in progress"

---

## D0+1 morning (2026-06-03): direct outreach

- [ ] User: send Neel Nanda email per `EMAIL_TEMPLATES.md §1` with concrete URLs (Zenodo / GitHub / HF Q2)
- [ ] User: send Joshua Batson Twitter DM per `§2`
- [ ] User: post on HuggingFace Gemma Scope repo discussion (Q2_K patch x SAE feature question)
- [ ] User: drop link in EleutherAI Discord `#interpretability` channel with 1-line context
- [ ] User: cross-post to LessWrong / AlignmentForum per `§6` framing

Wait 24-48 hours for replies before sending followups.

---

## D+3〜D+7: Anthropic Fellows + LinkedIn

- [ ] User: apply to Anthropic Fellows Program (current cycle) — `EMAIL_TEMPLATES.md §4` as base
- [ ] User: LinkedIn InMail to Anthropic Interpretability Recruiter

---

## D+7〜D+10: exp166 monitoring + followup

- [ ] Claude: monitor `weight_analysis/results/phase2/exp166_runs.jsonl`
- [ ] Claude: when 256+ runs complete, compute preliminary stats
- [ ] User: if 0 replies by D+7, send polite followup to Neel/Joshua (one paragraph)
- [ ] Decision gate (D+10): is exp166 complete? If yes, proceed to v5.1 at D+14. If no, slip to D+21.

---

## D+14 (or D+21): paper v5.1 update

- [ ] Claude: write 1-page v5.1 update with exp166/167/168/169 statistical results
- [ ] User: upload new paper PDF to Zenodo (new version of the same DOI, or a v2 deposit)
- [ ] User: update HF model card and `CITATION.cff` with new identifiers
- [ ] User: update Zenn article or post a follow-up
- [ ] User: send second email to Neel/Joshua ("exp166 just finished, here are the numbers")

---

## Honest expectations (set before launch)

```
Reply rate from cold mail to top researchers: 1-5%
Anthropic Fellows acceptance rate: ~5% (submission itself is the signal)
Reddit r/LocalLLaMA upvote rate: highly variable, 50-500 range possible
HF download rate week 1: 10-100 typical for niche releases

Most likely outcome: small but real reach increase, paper v1 → v5 narrative starts to form.
Less likely: Neel / Joshua / Anthropic Fellows reads it and replies (even 1-line is significant).
Worst case: reach 0. But you still have a Zenodo DOI, HF model card, GitHub repo, Fellows submission, and exp166 follow-through commitment.
```

---

## Files prepared in `publish/release_v1/`

| File | Use |
|---|---|
| `HF_MODEL_CARD.md` | HuggingFace README content (paper v5) |
| `HF_README_Q1.md` / `Q2.md` / `Q4.md` | Per-quant HuggingFace README (paper v1 quants) |
| `README_paper_v5.md` | GGUF bundle usage guide (Japanese) |
| `GITHUB_README.md` | GitHub repo README (paper v5 narrative, English) |
| `ZENN_ARTICLE_DRAFT.md` | Zenn article draft (Japanese) |
| `EMAIL_TEMPLATES.md` | Neel/Josh emails + Fellows cover letter + Reddit/LessWrong + Critical response templates |
| `REDDIT_POST.md` | r/LocalLLaMA post template |
| `X_THREAD.md` / `X_TEASER.md` | Twitter announcement threads |
| `REPRODUCE.md` | Step-by-step reproduction guide |
| `CITATION.cff` | Citation metadata (paper v5, Zenodo DOI) |
| `D0_LAUNCH_CHECKLIST.md` | This document |
| `HF_PRIVATE_PUBLIC_UPLOAD_CHECKLIST.md` | Private upload → inspection → public switch procedure |
| `ROLLBACK_PLAYBOOK.md` | HF / Zenodo / GitHub / Zenn rollback procedure |
| `LICENSE` / `LICENSE-CODE` / `LICENSE-WEIGHTS` | Apache 2.0 three-scope split |
| `apply_l25l26.py` | L25+L26 patcher (8 bytes, single file) |
| `requirements.txt` | gguf, numpy |

And in `docs/`:

| File | Use |
|---|---|
| `PAPER_V5_DRAFT_2026-05-30.md` | Full paper draft (convert to PDF for Zenodo deposit) |

---

## What requires user execution (Claude cannot do these autonomously)

- `hf auth login` / `gh auth login` / `$env:ZENODO_TOKEN` setup (token handling)
- GPG key generation / signing setup
- Reddit / Zenn / Twitter / LinkedIn / LessWrong posting (account credentials)
- Email sending (mail client)
- LinkedIn InMail (LinkedIn premium)
- Anthropic Fellows application form

Claude prepares the content, runs the public-switch CLI at T+0, and monitors exp166. User does account-bound actions.
