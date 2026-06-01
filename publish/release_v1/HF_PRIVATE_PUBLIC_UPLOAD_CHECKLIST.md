# Hugging Face Private-to-Public Upload Checklist

Purpose: upload the large GGUF bundle privately first, inspect it, then
switch the repository public only after the model card, licenses, file list,
and checksums are correct.

Use `hf` as the primary CLI. This machine also has `huggingface-cli`, but
the current Hugging Face docs and local help prefer `hf`.

## 0. Preconditions

- [ ] Do not paste HF tokens in chat or commit them to files.
- [ ] Run `hf auth login` locally with a write-scoped token.
- [ ] Confirm the target repo name:
  - `HF_USER=<your-hf-user-or-org>`
  - `HF_REPO=$HF_USER/gemma-4-31b-l25-l26-patches-q2k`
- [ ] Confirm `LICENSE-WEIGHTS` still matches the current upstream Gemma 4 license.
- [ ] Confirm `HF_MODEL_CARD.md` has no `<TBD>` / placeholder URLs left.

PowerShell setup:

```powershell
$ReleaseDir = "C:\Users\morph\開発\パイプラインオートメーション\publish\release_v1"
$HFRepo = "<your-hf-user-or-org>/gemma-4-31b-l25-l26-patches-q2k"
hf auth whoami
```

## 1. Create Private Repo

```powershell
hf repos create $HFRepo --type model --private --exist-ok
hf repos settings $HFRepo --private
```

Expected result: repo exists and is private.

## 2. Upload Metadata First

Upload the model card as the Hugging Face README, plus license and usage
docs. This makes the private inspection meaningful before large files land.

```powershell
hf upload $HFRepo "$ReleaseDir\HF_MODEL_CARD.md" README.md --commit-message "add model card"
hf upload $HFRepo "$ReleaseDir\README_paper_v5.md" README_paper_v5.md --commit-message "add bundle usage guide"
hf upload $HFRepo "$ReleaseDir\LICENSE-WEIGHTS" LICENSE-WEIGHTS --commit-message "add model weights license notice"
hf upload $HFRepo "$ReleaseDir\LICENSE-CODE" LICENSE-CODE --commit-message "add code license notice"
```

## 3. Upload GGUF Files While Still Private

```powershell
hf upload $HFRepo "$ReleaseDir\gemma-4-31B-it-L25x1.5-Q2_K.gguf" "gemma-4-31B-it-L25x1.5-Q2_K.gguf" --commit-message "upload L25 Q2_K patch"
hf upload $HFRepo "$ReleaseDir\gemma-4-31B-it-L26x1.5-Q2_K.gguf" "gemma-4-31B-it-L26x1.5-Q2_K.gguf" --commit-message "upload L26 Q2_K patch"
hf upload $HFRepo "$ReleaseDir\gemma-4-31B-it-L25L26x1.5-Q2_K.gguf" "gemma-4-31B-it-L25L26x1.5-Q2_K.gguf" --commit-message "upload L25L26 Q2_K patch"
hf upload $HFRepo "$ReleaseDir\gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf" "gemma-4-31B-it-L25L26x1.5-Q4_K_M.gguf" --commit-message "upload L25L26 Q4_K_M patch"
hf upload $HFRepo "$ReleaseDir\gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf" "gemma-4-31B-it-tripleLOS27-PAN17-PAN43-x1.8-Q2_K.gguf" --commit-message "upload supplementary triple Q2_K patch"
```

If the connection is unstable, upload one file at a time and record the
completed commit URL after each command.

## 4. Private Inspection Gate

Do not make the repo public until every item is checked:

- [ ] Hugging Face repo is still private.
- [ ] README renders from `HF_MODEL_CARD.md`.
- [ ] YAML frontmatter says `license: apache-2.0`.
- [ ] `LICENSE-WEIGHTS` and `LICENSE-CODE` are present.
- [ ] File list has exactly the expected five GGUF files.
- [ ] No `.backup`, `_temp/`, local absolute cache path, token, or `.env` file is present.
- [ ] File sizes match local expectation:
  - [ ] L25 Q2_K: about 12.6 GB
  - [ ] L26 Q2_K: about 12.6 GB
  - [ ] L25L26 Q2_K: about 12.6 GB
  - [ ] L25L26 Q4_K_M: about 19.6 GB
  - [ ] triple Q2_K: about 12.6 GB
- [ ] Download smoke test works from a clean temp directory.

Download smoke test:

```powershell
$SmokeDir = Join-Path $env:TEMP "hf_v5_private_smoke"
New-Item -ItemType Directory -Force $SmokeDir | Out-Null
hf download $HFRepo README.md LICENSE-WEIGHTS --local-dir $SmokeDir
Get-ChildItem $SmokeDir
```

## 5. Switch Public

Only after the private inspection gate passes:

```powershell
hf repos settings $HFRepo --no-private
```

Immediately open the public URL and verify the model card, license, and
file list again.

## 6. Emergency Re-Private

If the repo was made public with a wrong file, wrong license, or wrong
number, switch it private first and then repair:

```powershell
hf repos settings $HFRepo --private
```

Then follow `ROLLBACK_PLAYBOOK.md`.
