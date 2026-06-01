# Paper v5 Release Rollback Playbook

Purpose: reduce public damage from a bad launch state without deleting
evidence or improvising under pressure.

## Severity Levels

| Level | Trigger | First action |
|---|---|---|
| S0 | typo, broken link, harmless wording issue | patch in place and add changelog note |
| S1 | wrong numeric claim, missing caveat, wrong file label | make affected channel non-public if possible, patch, then republish |
| S2 | wrong model file, wrong license notice, leaked private path, accidental public HF repo | immediately switch repo/private channel off, preserve local evidence, repair |
| S3 | credential/token leak, personal data leak, legal takedown, harmful model misuse issue | revoke credentials or remove access first, then freeze public posting and write incident note |

## Universal First 10 Minutes

1. Stop posting new links.
2. Save the current public URL, timestamp, and screenshot if possible.
3. Identify the exact artifact: GitHub, Hugging Face, Zenodo, Zenn, Reddit, X/Twitter, email, or release tag. (arXiv is not part of the 2026-06-02 launch; if a future release adds arXiv, see the arXiv section below.)
4. If exposure is ongoing, make the artifact private or unpublished before editing.
5. Write one local note in `publish/release_v1/archive/` with:
   - UTC/JST time
   - channel
   - bad state
   - action taken
   - remaining risk

Do not rewrite public history silently for S1+ issues. Fix the artifact and
leave a short changelog note where readers would reasonably look.

## Hugging Face

### Triggers

- Repo accidentally public before private inspection passed.
- Wrong GGUF uploaded.
- `README.md` / model card contains wrong license, overclaim, or placeholder.
- Local backup/temp files uploaded.

### Immediate actions

```powershell
$HFRepo = "<your-hf-user-or-org>/gemma-4-31b-l25-l26-patches-q2k"
hf repos settings $HFRepo --private
```

Then inspect and repair:

```powershell
Start-Process "https://huggingface.co/$HFRepo/tree/main"
hf upload $HFRepo "publish/release_v1\HF_MODEL_CARD.md" README.md --commit-message "fix model card after rollback"
hf upload $HFRepo "publish/release_v1\LICENSE-WEIGHTS" LICENSE-WEIGHTS --commit-message "fix weights license notice"
```

If a wrong file was uploaded, delete only the wrong path:

```powershell
hf repos delete-files $HFRepo "<wrong-file-name>"
```

Republish only after the private inspection gate in
`HF_PRIVATE_PUBLIC_UPLOAD_CHECKLIST.md` passes again.

## GitHub

### Triggers

- Credentials, `.env`, local private path, huge GGUF, or forbidden file was pushed.
- README contains wrong license or a materially wrong numeric claim.
- Release asset is wrong.

### Immediate actions

- If the repository is public and contains sensitive material, make the repo
  private in GitHub UI first.
- Revoke leaked tokens before editing Git history.
- Do not use `git reset --hard` in this workspace.

Repair path:

```powershell
git status --short
git diff -- publish/release_v1 docs/PAPER_V5_DRAFT_2026-05-30.md
```

Patch only the affected files. Stage by explicit file path only.

## Zenodo (paper archive, DOI 10.5281/zenodo.20362821)

### Triggers

- Wrong PDF version uploaded.
- Quantitative typo in the deposited paper.
- Missing caveat that changes the claim boundary.
- Wrong metadata (title, authors, license).

### Immediate actions

- For DOI-issued deposits, you cannot delete the deposit but you can publish a **new version** of the same record (Zenodo assigns a new versioned DOI and the original DOI now resolves to a "latest version" page).
- If the issue is metadata-only (and the DOI hasn't been used in citations yet), open a support request at `https://zenodo.org/support` for a metadata correction; do not rely on this for time-sensitive fixes.
- Once a corrected version is published, announce the new versioned DOI on the same channels (Reddit / Zenn / Twitter) where the original was posted.

### Minimum public wording for a quantitative correction

```text
Correction: an earlier version misstated <field>. The artifact has been
updated to <correct value>. The claim boundary is unchanged / changed as
follows: <one sentence>. New Zenodo version DOI: <new DOI>.
```

## arXiv (NOT in 2026-06-02 launch — future contingency only)

This section is retained for the future case where an arXiv preprint is added (after endorsement is obtained). It is not relevant to the 2026-06-02 launch.

### Triggers

- Wrong title, author, affiliation, abstract, license, or PDF.
- Quantitative typo in the submitted PDF.
- Missing caveat that changes the claim boundary.

### Immediate actions

- If submission is not finalized, withdraw or replace before public release.
- If public, prepare a replacement PDF and a concise change note.
- Do not email researchers with the bad arXiv link until the replacement path is clear.

Minimum public wording for a quantitative correction:

```text
Correction: an earlier version misstated <field>. The artifact has been
updated to <correct value>. The claim boundary is unchanged / changed as
follows: <one sentence>.
```

## Zenn / Reddit / Social Posts

### Triggers

- Overclaim wording, broken URL, wrong license, or obsolete benchmark framing.
- Community criticism identifies a real ambiguity.

### Immediate actions

- For minor typos: edit in place and add "修正" note.
- For S1+ issues: unpublish/delete the post if platform norms allow, then
  repost with correction note.
- Do not argue in comment threads before the artifact itself is corrected.

## Email Outreach

### Triggers

- Email sent with wrong link, wrong numeric value, or missing caveat.
- Wrong recipient/contact.

### Immediate actions

- Send one short correction. Do not send a chain of follow-ups.

Template:

```text
Subject: Correction to previous note: <paper/release name>

Hi <name>,

Quick correction to my previous email: <exact wrong item> should be
<correct item>. I have corrected the public artifact here: <link>.

The corrected claim boundary is: <one sentence>.

No reply needed; I wanted to avoid leaving a bad pointer in your inbox.

Akito Hirai
```

## License-Specific Response

If someone challenges the Gemma 4 license, do not debate from memory.

1. Re-open the upstream source:
   - https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
   - https://ai.google.dev/gemma/apache_2
   - https://huggingface.co/google/gemma-4-31B-it
2. Check whether the exact base GGUF source has a different model card.
3. If local docs are wrong, switch HF private, patch `LICENSE-WEIGHTS`,
   `HF_MODEL_CARD.md`, and `README_paper_v5.md`, then republish.

## Closeout Checklist

- [ ] Affected public channel repaired or made private.
- [ ] Local incident note written under `publish/release_v1/archive/`.
- [ ] Changelog or correction note added for S1+.
- [ ] Tokens revoked if any credential was exposed.
- [ ] Follow-up emails/posts paused for 24 hours after S2/S3.
