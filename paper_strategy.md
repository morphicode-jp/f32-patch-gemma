# Paper Strategy: Fastest Path to Publication

**Date:** 2026-04-18
**Author:** meta-analysis of commits, experiments, and novelty claims

---

## Executive Summary

Given current evidence base (~50 commits, 18 experiments, 46 literature claims),
**the single-shortest-to-ship paper** is:

> **"LaD v2: Propositional Logic-as-Data for Literature-Grounded Neural Design Search"**

Shipping this paper does NOT require new experiments. All content exists.
Estimated time to arxiv-ready: **5-10 days** (focused writing + figures).

## Decision criteria

For each candidate narrative, score on:

1. **Novelty** — is anyone else doing this?
2. **Defensibility** — can a reviewer easily poke holes?
3. **Existing evidence** — how much of the paper exists in code + JSON already?
4. **Missing experiments** — what must we still run?
5. **Writing effort** — days of focused prose + figures
6. **Venue fit** — workshop / conference / journal

---

## Option 1: Sentinel Methodology Paper

**Title draft**: "Sentinel: Safe Optimization via Dual-Metric Auto-Pivot"

**Claim**: A novel optimizer framework for constrained optimization with
(eval_fn, guard_fn) dual objectives + automatic pivot on failure.

### Evidence
- `twelve/agent/sentinel.py` (core implementation)
- `twelve/tests/test_sentinel.py` (9+ unit tests passing)
- Discrete-space fallback (owl → optimize)
- Applications: LLM calibration, brain topology, co-evolution, memory
- ~15 experiments across domains

### Pros
- Strong engineering claim
- Many empirical demos
- Replicable by others (open source)

### Cons (critical)
- "New optimizer" framing is **hard to sell** without strong benchmark comparison
- Need head-to-head vs: CMA-ES, Bayesian opt, gradient-free RL, Optuna
- Need ablation study (does fallback actually help?)

### Missing to ship
- Benchmark comparison (1 week of running CMA-ES / Optuna on same tasks)
- Statistical significance of "Sentinel > CMA-ES on discrete X"
- Unified benchmarks (2-3 problem suites)

### Effort to ship
- **10-14 days** (most of it benchmark work)
- Venue: NeurIPS / ICLR workshop on optimization
- Novelty score: **6/10** (good framework, not paradigm-shifting)

---

## Option 2: Phase 3.5 Cooperative Language Paper (RESCUE REQUIRED)

**Title draft**: "Small-Scale Emergence of Referential Signaling under
Cooperative vs Competitive Co-Evolution"

**Claim**: With 91-parameter evolved neural agents, cooperation
consistently produces more informative voice signals than competition
(MI vs shuffled null).

### Evidence
- `coop_sentinel.py`, `signal_sentinel.py`, `signal_world.py`, `coop_world.py`
- Original n=1 Phase 3.5 finding: coop gain +0.057 bits, compete +0.006
- `language_rigorous.py` n=3 attempted replication (confound with random init)
- `language_replication.py` n=5 replication (**failed mid-run, exit code 4**)

### Pros
- Clear narrative ("cooperation causes language")
- Connects to biological theory (Hamilton, Skyrms)
- Small-scale demo counters "need LLMs" narrative
- Novel combination: Sentinel + co-evolution + MI analysis

### Cons (critical)
- Replication evidence is currently **WEAK** (n=1 original, n=3 failed to
  replicate under different protocol, n=5 run CRASHED)
- We cannot claim the result is robust without fixing this
- Single-instance finding at n=1 is textbook "p-hacking risk"

### Missing to ship
- Rerun `language_replication.py` n=5 (~1-2 hours) + debug why it crashed
- Possibly n=10 replication for robustness (+3 hours)
- Confound check: matched training budget (65s for both conditions)
- Ablation: does the voice channel actually carry the signal,
  or do we see same MI from non-voice nodes?

### Effort to ship
- **2-3 weeks** (fix replication + robust statistics + writing)
- Venue: CoNLL / EMNLP (emergent communication workshop)
- Novelty score: **5/10** (not new claim; novelty is in method + smallest scale)

---

## Option 3: LaD v2 Methodology Paper ★ RECOMMENDED ★

**Title draft**: "Propositional Logic-as-Data: A Sentinel-Powered Framework
for Literature-Grounded Neural Design Search"

**Claim**: Encoding research claims as propositions (condition → outcome,
strength) enables automated rule extraction, contradiction detection, and
literature-grounded optimal design proposal. Demonstrated on 46 claims
from emergent communication literature.

### Evidence (all in commits)
- `meta_logic_claims.py` — 46 propositional claims dataset
- `meta_logic_analysis.py` — rule extraction + contradictions + owl
- `meta_optimal_design.py` — forward: design → score (13,542 candidates)
- `meta_reverse_design.py` — reverse: outcome → best design (7 goals)
- `meta_gap_analysis.py` — literature coverage gaps → search queries
- `meta_research_papers.py` — 72 papers as attribute vectors (LaD v1)
- `meta_sentinel_analysis.py` — owl on attribute space
- `kathara_topology_search.py` — architectural optimization application
- `CLAUDE.md` — methodology documentation

### Pros (strong)
- **Truly novel methodology** — no one has used Sentinel-style search on
  propositional claims as data
- **Working end-to-end pipeline** (papers → claims → rules → design)
- **Positive result verified**: literature correctly predicted
  `our_phase3_compete` would fail; it did
- **Generates falsifiable predictions** (reverse design suggests
  specific next experiments)
- Self-contained — does not require Phase 3.5 to be resolved

### Cons
- Dataset is n=46 (moderate). Reviewer will ask: is this representative?
- Claim encoding is somewhat subjective (paper → propositions)
- Spectral-only topology search (no behavioral validation yet)

### Missing to ship
- **None, strictly speaking** — all content exists
- Nice-to-have: one blinded domain-expert reviewing our claim encoding
- Nice-to-have: extend dataset to 100 claims for robustness
- Nice-to-have: compare against Pearson-only attribute analysis as baseline

### Effort to ship
- **5-10 days** of focused writing + figures
- Venue: NeurIPS D&B track, or AI/meta-research journal
- Novelty score: **8/10** (methodology is genuinely new at this integration)

---

## Recommended Paper Outline (Option 3)

### Title
Propositional Logic-as-Data: Sentinel-Powered Literature-Grounded Neural Design

### Abstract (150 words)
Literature meta-analysis in neural architecture research is typically
narrative, with sporadic attribute-based surveys (paper_count(method), etc).
We propose **Propositional Logic-as-Data (LaD v2)**: each paper's
empirical claim is encoded as a conditional rule
(condition_list → outcome, strength). Applied to 46 claims from 42 papers
on emergent neural communication, we extract:
(i) lifted IF-THEN rules with confidence ranking;
(ii) pairwise contradictions between papers (indicating active disputes);
(iii) owl-based importance of condition features;
(iv) a forward+reverse optimal-design pipeline that, given a target outcome,
returns the design configuration best supported by literature.
We verify this pipeline's validity on our own experimental records:
a configuration the pipeline marked "predicts failure" indeed failed.
This methodology is architecture-agnostic and extends to any field where
empirical claims can be structured as conditional propositions.

### Sections

1. Introduction (1p)
   - Problem: meta-analysis is narrative
   - Contribution: LaD v2 + Sentinel pipeline
   - Validation: literature-grounded negative prediction verified

2. Related Work (1p)
   - Literature analysis approaches (Pearson, co-citation)
   - Meta-research in ML (BO for hparam, NAS)
   - What we do differently: propositions with strength

3. Methodology (3p)
   - 3.1 Claim schema (condition, outcome, strength)
   - 3.2 Rule extraction (lift computation)
   - 3.3 Contradiction detector
   - 3.4 Optimal design search (forward + reverse)
   - 3.5 Gap analysis (literature coverage)

4. Application: Emergent Communication Literature (3p)
   - 4.1 Dataset (46 claims, 42 papers, 25 feature conditions, 7 outcomes)
   - 4.2 Extracted rules (top IF-THENs)
   - 4.3 Detected contradictions (reward_coop → compositional DISPUTED)
   - 4.4 Reverse design proposals (what causes deception? honesty?)

5. Validation (2p)
   - 5.1 Our own experiments as held-out test: `phase3_compete`
     predicted to fail, indeed failed
   - 5.2 Gap analysis correctly identifies under-sampled regions
   - 5.3 Pipeline recommends testable configurations

6. Discussion (1p)
   - Limitations: claim encoding is manual
   - Extensibility to other fields
   - Future: LLM-automated claim extraction

7. Conclusion (0.5p)

### Required Figures
- Fig 1: LaD v1 (attributes) vs v2 (propositions) schema diagram
- Fig 2: Rule lift distribution (top-10 bar chart)
- Fig 3: Contradiction graph (nodes=claims, edges=conditions_shared,
  color=agreement)
- Fig 4: Reverse-design table (7 outcomes × top-3 designs)
- Fig 5: Our experiments positioned on the design-score landscape

### Required Tables
- Tab 1: 46 claims sample (5 rows)
- Tab 2: Rule extraction top-10
- Tab 3: Gap-analysis priority (20 under-sampled cells)

### Page estimate
~12 pages (ICML/NeurIPS format) or ~8 pages (workshop).

---

## Timeline to Submit

| Day | Task |
|---|---|
| 1 | Finalize LaTeX template, section headers |
| 2 | Section 3 (Methodology) prose |
| 3 | Section 4 (Application) prose + Fig 2, 3 |
| 4 | Section 5 (Validation) prose + Fig 5 |
| 5 | Introduction + Abstract + Related Work |
| 6 | All figures polished, tables finalized |
| 7 | Full-read pass, rewriting weak transitions |
| 8 | External reader (any domain friend) |
| 9 | Revise based on feedback |
| 10 | Final proof + arxiv submission |

---

## Alternative: Combined Paper (if ambitious)

"Sentinel for Literature-Grounded Design: From Claim Propositions to
Neural Architecture Search" — combines Option 1 (Sentinel) + Option 3
(LaD v2) into a single longer paper.

- Pros: stronger story, shows methodology in context of tool
- Cons: 2-3x effort, higher rejection risk at top venue

**Recommendation**: Ship Option 3 first. If accepted, Option 1 becomes
natural follow-up with same tooling.

---

## Action Items (if user approves Option 3)

1. Create `paper/` directory with LaTeX skeleton
2. Export key figures from existing JSON:
   - `meta_logic_result.json` → Fig 2, 3
   - `reverse_design_result.json` → Fig 4
   - `optimal_design_result.json` → Fig 5
3. Draft Section 4 first (has most existing content)
4. User chooses venue (NeurIPS D&B, ICLR workshop, arxiv-first)

---

## Not Shipping This Year

These are interesting but need too much new work:

- **Red Queen ToM paper**: needs n=10 replication, deeper ToM analysis
- **Hippocampal memory paper**: Phase 6 result was PARTIAL (+10pt not +30pt)
- **Kathara brain survey**: needs systematic comparison to other small NN frameworks
- **Sentinel vs ES/BO benchmark**: 2-3 weeks benchmarking work

---

## TL;DR

**Ship Option 3 (LaD v2 paper) in 5-10 days.**
It's the only narrative where all necessary evidence exists and the
novelty is genuinely novel (no one else combines Sentinel with
propositional literature analysis).
