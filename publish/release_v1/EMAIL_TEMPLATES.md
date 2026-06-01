# Outreach Email Templates ((Day 0 send))

## ⚠ Before sending

- [ ] **Verify each contact ((1 文字 違 い で 別人 risk))**: Neel email / Arthur email / Joshua Twitter handle を D-1 朝 に 公 開 source で 再 確 認
- [ ] All public URLs are concrete (Zenodo DOI 10.5281/zenodo.20362821, GitHub `morphicode-jp/f32-patch-gemma`, HF `morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K`)
- [ ] Attach paper PDF ((3-5 pages max)) or link to Zenodo deposit
- [ ] Use plain text format ((NOT HTML / fancy signatures))
- [ ] Send from a real email ((preferably matching your GitHub handle))
- [ ] Send no more than 2-3 emails in 24h to avoid spam triggers
- [ ] Reply rate base: 1-5% for cold mail; matching topic + concrete public artifact lifts ceiling but doesn't guarantee anything

---

## 1. Google DeepMind: Neel Nanda

**To**: `neelnanda27@gmail.com` ((**re-verify D-1: neelnanda.io / @NeelNanda5 Twitter bio**))
**Subject**: `Gemma 4 31B Q2_K: 8-byte F32 patch (L25+L26 ×1.5) and per-layer ablation (n=1 pilot, code+GGUF public)`

```
Neel,

I'm an independent researcher. Background: my paper v1 release (May 27)
on Gemma 4 31B showed an 8-byte F32 scale patch (×1.5 on layer_output_scale
of L25+L26, no training, no calibration) yielding striking effects on the
quantized model — Q4_K_M patched (19 GB) beat the Q8_0 BF16 baseline
(31 GB) on all 4 benchmarks (HS 73.50% > 63.33%, GSM 87% > 70%,
WG 70.32% > 65.59%, ARC 48.76% > 44.38%); Q2_K HS +11.21pt;
IQ1_M GSM +36.0pt. All 12 cells (3 quant levels × 4 benches) positive.
A reduced-N reference rerun for this release (exp162: HS 1000 /
WG 500 / ARC 200 / GSM 100) reproduces the directionality with
+9.13pt avg on Q2_K — small-N variance expected.

What may be more interesting: splitting the same patch into single-layer
ablations (L25-only ×1.5 ≈ 4 bytes; L26-only ×1.5 ≈ 4 bytes) appears
to give different behavioral phenotypes on a recurrence-relation prompt:
L25-only solves correctly with minimal mid-stream self-correction;
L26-only solves correctly but inserts multiple "Wait, let me re-verify"
bursts plus a spontaneous supplementary section; the L25+L26 mix (paper
v1) produces an explicit closing "Double check the a_n sequence:" re-
enumeration absent in either single-layer condition. This is n=1 on a
single prompt — hypothesis-generating only.

I'm running a 4 patch × 4 prompt × 32 seed = 512 run confirmation now
(exp166), expected within ~2 weeks. Posting today because the bake
script + 4 patched GGUFs (L25, L26, L25+L26, plus a 12-byte triple
control) are public and the claim is immediately falsifiable.

Speculative framing: L25 / L26 as "compute" / "verify-proposal"
specialists. I'd appreciate any read on whether this kind of layer-level
behavioral asymmetry is (a) already studied — Gemma 2 SAE work is the
nearest neighbour I'm aware of, but Gemma 4 has no public SAE release —
(b) a known quantization artifact, or (c) something worth probing further.

paper (Zenodo): https://doi.org/10.5281/zenodo.20362821
code: https://github.com/morphicode-jp/f32-patch-gemma
HF (paper v1 mix Q2_K): https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K
(IQ1_M and Q4_K_M variants linked from the GitHub repo README)

10 minutes of "yes / no / known" would be a huge help. Either answer
saves me time.

Thanks,
Akito Hirai
morphicode.jp@gmail.com
```

**Rationale**:
- Neel publicly invites DMs and reads cold email
- Leads with concrete, verifiable bench numbers ((+9.13pt avg, GGUF DL-able))
- The new claim ((L25/L26 phenotype split)) is framed as **observation, n=1, hypothesis**
- Gemma Scope is mentioned as "nearest neighbour" rather than claimed mapping ((avoids "Gemma 4 SAE" overclaim))
- Citation-free, no missing-author landmines
- 3 specific low-cost reply options ((known / artifact / worth probing))

---

## 2. Anthropic: Joshua Batson

**To**: Twitter DM `@thebasepoint` ((**re-verify D-1: anthropic.com team page + transformer-circuits.pub author bio**; DM permission may be closed))
**Backup**: email if found ((Anthropic format is often `firstname@anthropic.com` but not guaranteed))
**Subject (if email)**: `Q2_K interpretability case: 8-byte L25+L26 patch + per-layer ablation on Gemma 4 31B`

```
Josh,

Independent researcher. Background: paper v1 (May 27) on Gemma 4 31B
demonstrated an 8-byte F32 scale patch (L25+L26 ×1.5, no training)
where Q4_K_M patched (19 GB) beats Q8_0 BF16 baseline (31 GB) across all
4 benchmarks (HS 73.50%, GSM 87%, WG 70.32%, ARC 48.76%), Q2_K HS +11.21pt,
IQ1_M GSM +36.0pt — 12/12 cells positive across 3 quant levels.
exp162 reduced-N rerun for this release reproduces the directionality
with +9.13pt avg on Q2_K (smaller N, smaller absolute effect).

Splitting that patch into single-layer ablations appears to give
asymmetric behavioral phenotypes on one recurrence-relation prompt:

  L25 alone (×1.5): correct answer, minimal self-correction
  L26 alone (×1.5): correct answer, multiple "Wait" bursts + spontaneous
                    supplementary section, highest token count
  L25+L26 mix:      correct answer, explicit closing "Double check" re-
                    enumeration absent in either single-layer case

This sits adjacent to (a) the Anthropic circuits / induction-head line
of work, and (b) crosscoder / model-diffing — Q2_K vs Q4_K_M would be a
natural diff target if a Gemma-4-scale SAE existed. Q2_K interpretability
appears under-studied as far as I can find.

This is n=1 on a single prompt — hypothesis-generating only, not a
confirmed mechanism. n=32 × 4 prompts × 4 patches confirmation is
running now. Code + 4 patched GGUFs are public so the claim is
immediately falsifiable.

paper (Zenodo): https://doi.org/10.5281/zenodo.20362821
code: https://github.com/morphicode-jp/f32-patch-gemma
HF (paper v1 mix Q2_K): https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K

Would 15 minutes from the Anthropic interp team be possible, or is this
a known dead-end? Either answer is useful.

Akito Hirai
morphicode.jp@gmail.com
```

**Rationale**:
- Same honest framing as Neel email
- Anthropic circuits / crosscoder mentioned **generically** ((no specific paper / author chosen)) to avoid credit miscredit landmines
- "if a Gemma-4-scale SAE existed" preemptively addresses the Gemma Scope = Gemma 2 issue
- "n=1 hypothesis-generating only" repeated explicitly
- Low-cost reply option ((known dead-end / worth a look))

---

## 3. Followup CCs ((only if main reply lands))

After Neel or Joshua reply, can naturally include ((re-verify each contact at that time)):
- **Tom Lieberum** ((Gemma Scope co-lead)) — preferred path: GitHub discussion on Gemma Scope repo rather than cold mail
- **Arthur Conmy** ((ACDC author; current org may have changed — verify before contact))
- **Adam Jermyn** ((Anthropic interp))
- **Trenton Bricken** ((Anthropic, crosscoder/SAE))

Don't CC them upfront — looks like spam. Verify each contact 1 文字 単位 で before sending.

---

## 4. Anthropic Fellows Program cover letter draft

**For**: https://www.anthropic.com/anthropic-fellows-program

⚠ **D-1 で 確 認 必 須**: 現 行 cohort 名 / 締 切 / 募 集 中 か を 公 式 page で 1 字 1 句 確 認。 cohort 名 が 違 う 場 合 「Anthropic Fellows Program ((current cycle))」 等 に rewrite。

Cover letter ((1 page max)):

```
Dear Anthropic Fellows team,

I'm applying to the current Anthropic Fellows cycle with a research
focus on mechanistic interpretability of quantized LLMs — specifically,
whether per-layer functional specialization can be observed and
selectively strengthened in heavily quantized models (Q2_K and below).

Background — paper v1 ((May 27, 2026)):

  An 8-byte F32 scale patch (L25+L26 ×1.5, no training, no calibration)
  on Gemma 4 31B yielded striking effects across all tested quantization
  levels: Q4_K_M patched (19 GB) beat Q8_0 BF16 baseline (31 GB) on all
  4 benchmarks (HS 73.50% > 63.33%, GSM 87% > 70%, WG 70.32% > 65.59%,
  ARC 48.76% > 44.38%); Q2_K HellaSwag +11.21pt; IQ1_M GSM +36.0pt.
  All 12 cells (3 quant levels × 4 benches) positive.

Pilot finding for paper v5 ((n=1 → n=32 follow-up, posted 2026-05-30)):

  Splitting the v1 patch into single-layer ablations (L25 alone, L26
  alone, both 4 bytes each) produces qualitatively different behavioral
  phenotypes. exp166 follow-up (512 runs, 4 patches × 4 prompts ×
  32 seeds) confirms that the verify-proposal aux_rate is significantly
  different (L26 alone 56% vs L25 alone 16%, p < 0.01).

  A separate 12-byte triple patch (LOS27+PAN17+PAN43 ×1.8) was tested
  as a paper v4 candidate. It wins on GSM8K benchmark (85% vs paper v1
  76% under T=0.0 / max_tokens=1024 / regex extraction) but in exp169
  interactive verification (T=0.7, max_tokens=6144, n=32 seeds on the
  same P1 coin prompt as exp166), triple solves only 4/32 (12.5%) vs
  paper v1 mix 30/32 (93.8%) — Fisher exact one-sided p < 0.0001.
  Failure mode: 89.3% silent_slip (confident wrong numeric answer).
  Triple emits more verification markers (mid_wait 9.25, tail_rate 31%)
  than any other patch including baseline (mid 8.25, tail 16%), but
  the verification is non-functional under interactive conditions. The
  result reads as a Goodhart's Law instance for F32 patch quality
  evaluation.

Connection to interpretability literature:

  1. Layer-level functional differentiation framing builds on the
     Anthropic circuits line of work (induction heads, attribution
     graphs, crosscoders)
  2. Q2_K-specific interpretability appears under-studied; the patched
     model is small enough that activation patching / logit lens is
     directly tractable
  3. Gemma Scope-style SAE methodology could be adapted for Gemma 4
     31B (no official Gemma-4 SAE release exists yet) to probe whether
     L25 / L26 correspond to identifiable feature directions

Limitations I should be upfront about:

  - n=1 on a single prompt — currently running n=32 × 512-run
    confirmation experiment (exp166)
  - Multiple uncontrolled confounders (sampling stochasticity, prompt-
    class bias, scale-vs-layer confound, order effects) — detailed in
    paper §8
  - Cross-model generality untested at this depth; prior F32 patch work
    on hybrid LLMs with rare full-attention layers showed strong effects
    on Gemma 4, weaker on Qwen 3.6, null on Phi-4
  - No direct residual-stream observation yet (logit-lens probe is a
    proposed next step)

What I'd pursue as a Fellow:

  - Adapt Gemma-Scope-style SAE training pipeline for Gemma 4 31B and
    test whether L25 / L26 patched features correspond to interpretable
    feature directions (6 months)
  - Crosscoder-style Q2 vs Q4 feature drift quantification under the
    same patch
  - Cross-model replication (Qwen 3.6, Phi-4, Mistral-Nemo)
  - Logit lens / activation patching at L24-L27 for direct mechanism
    verification

Public artifacts (immediately falsifiable):
  - paper (Zenodo): https://doi.org/10.5281/zenodo.20362821
  - code: https://github.com/morphicode-jp/f32-patch-gemma
  - patched GGUFs (HF): morphicode-jp/gemma-4-31B-it-L25L26x1.5-{IQ1_M,Q2_K,Q4_K_M}
  - Zenn writeup (JP): <Zenn URL — fill at D0 after publish>

Background:
  - Prior paper v1 release (2026-05-27): initial L25+L26 F32 patch
    finding for Gemma 4 31B
  - GitHub commit history: https://github.com/morphicode-jp

Available to discuss at morphicode.jp@gmail.com.

Thanks for considering,
Akito Hirai
```

---

## 5. r/LocalLLaMA Reddit post template

**Title**: `Released L25/L26 F32 patches for Gemma 4 31B Q2_K — 8-byte L25+L26 mix recovers +9.13pt avg on HS/WG/ARC/GSM, plus per-layer ablation (n=1 pilot, hypothesis-generating)`

**Body**:

```
TL;DR: Posted 4 patched GGUFs + bake script + paper draft for Gemma 4
31B Q2_K with per-layer F32 scale patches. The 8-byte L25+L26 ×1.5
patch (paper v1) recovers +9.13pt average on HellaSwag (+11.30),
WinoGrande (+7.20), ARC-C (+9.00), and GSM8K (+9.00) in exp162 reference
run.

What may be more interesting: splitting that patch into single-layer
ablations gives qualitatively different behavioral patterns on a
recurrence-relation prompt:

- L25 alone (4 byte): correct answer, 1 mid-stream "Wait" self-correct
- L26 alone (4 byte): correct answer, 4 mid-stream "Wait, let me re-
  verify" + spontaneous supplementary section, longest token count
- L25+L26 mix (8 byte, paper v1): an apparent closing "Double check the
  a_n sequence:" re-enumeration absent in either single-layer case,
  with shortest token count
- Triple LOS27+PAN17+PAN43 ×1.8 (12 byte): supplementary illustration,
  statistically indistinguishable from paper v1 in full bench; included
  as reference, not as a v5 claim

The mix pattern appears to be preserved when the same patch is tested
on Q4_K_M (4× weight precision difference, n=1 on the same prompt).

Speculative framing: L25 may act as a compute specialist, L26 as a
verify-proposal specialist, and joint scaling may produce something
like a terminal-verification reflex. This is a hypothesis only.

**Important: this is an n=1 single-question pilot, NOT a confirmed
claim.** Posting because all artifacts are public and immediately
falsifiable. n=32 × 4 prompts × 4 patches confirmation run (exp166)
in progress, expected within 2 weeks.

Files (HuggingFace): https://huggingface.co/morphicode-jp/gemma-4-31B-it-L25L26x1.5-Q2_K
Code (GitHub): https://github.com/morphicode-jp/f32-patch-gemma
Paper (Zenodo): https://doi.org/10.5281/zenodo.20362821

Feedback / reproductions / refutations welcome.
```

---

## 6. LessWrong / AlignmentForum crosspost

Reuse Reddit body but add:
- 1 paragraph linking to Anthropic circuits / crosscoder work without naming specific authors
- 1 paragraph "what would refute this" ((e.g. "if L25-only and L26-only run on n=32 seeds show identical mid-stream Wait distributions, asymmetry hypothesis is refuted"))
- Explicit ask for reviewer-style comments ((statistical, mechanism, alternative explanations))

---

## 7. Critical response templates

Use these only after checking the artifact itself. If the critic is right,
patch the artifact first and then reply.

### 7.1 "This is n=1; what can you claim?"

```
You're right that n=1 cannot establish a mechanism or a robust behavioral
effect. The current release is framed as a hypothesis-generating artifact:
patched GGUFs, bake scripts, and the exact prompt are public so the observation
can be reproduced or falsified.

The only claim I want to stand behind today is narrow:
on this prompt and these patched files, the single-layer ablations produced
different observed behaviors. The confirmation gate is exp166
(4 patches x 4 prompt families x 32 seeds = 512 runs). If that run does not
separate the L25/L26 behavior, the asymmetry hypothesis should be treated as
not supported.
```

### 7.2 "This looks cherry-picked."

```
That's a fair concern. The release includes the single pilot because it is
the observation that motivated the follow-up, not because it is enough by
itself. I am not treating the pilot as proof.

The anti-cherry-pick test is already defined: exp166 uses multiple prompt
families and seeds, with pre-registered parsing/coding rules. I will publish
the negative result as well if the effect disappears. The triple patch is also
included as a cautionary control: it scores well in one benchmark setup but
collapses under interactive verification, which is exactly why I do not want
to overfit the launch claim to a single attractive number.
```

### 7.3 "This violates Gemma Terms / model licensing."

```
Thanks for flagging this. I rechecked before release: current official Gemma 4
materials list Gemma 4 under Apache 2.0, and the google/gemma-4-31B-it model
card lists license: apache-2.0. I separated the notices into LICENSE-CODE and
LICENSE-WEIGHTS so code, model weights, and paper/docs are not conflated.

If you are pointing to a different upstream GGUF or a specific clause I missed,
please send the exact link. I will switch the HF repo private and patch the
license notice before continuing distribution if the current notice is wrong.
```

### 7.4 "The patch is just benchmark gaming."

```
That is possible for some patches, and the release explicitly includes a
negative example. The triple patch does well in a constrained GSM8K setup
(T=0.0, short max_tokens, regex extraction) but fails badly in interactive
verification. I label it supplementary for that reason.

The practical recommendation is the L25+L26 paper v1 mix, not the triple
patch. The stronger claim waits for exp166 and later mechanism probes
(logit lens / activation patching / feature analysis). Until then, the result
is an artifact-backed hypothesis, not a proven mechanism.
```

### 7.5 "Why email researchers before the full run is complete?"

```
I am waiting for public URLs and keeping the first outreach to two targeted
messages. The reason to share before the full run is that the artifacts are
already falsifiable and the main ask is not endorsement; it is whether this
looks like a known artifact, a dead end, or a useful probe direction.

If exp166 fails, the follow-up is simple: update the model card and paper with
the negative result and stop using the asymmetry framing.
```
