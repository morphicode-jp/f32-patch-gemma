"""meta_logic_claims.py - Logic-as-data: paper CLAIMS as structured propositions.

Unlike meta_research_papers.py (attribute vectors), this captures:
  - Each paper's core EMPIRICAL CLAIM as a propositional rule
    (conditions -> outcome)
  - Strength (how strongly paper supports it)
  - Whether contradicted by other papers

Example claim:
  paper P asserts: "if (cooperative=True AND iterated_learning=True)
                    then compositional_language_emerges"
  strength: 0.9 (clear claim)

Then we can:
  1. Find claims supported by many papers
  2. Find claims contradicted between papers
  3. Identify the LOGICAL structure of the field
  4. Use owl/Sentinel on the CLAIM SPACE (not paper metadata)
"""

# Each claim is a tuple:
#  conditions: list of (feature, operator, value) triples
#  outcome:    string outcome variable
#  strength:   -1 (strongly against), 0 (neutral/fail), +1 (strongly supports)
#  reason:     brief paper-specific justification

# Possible features in conditions:
#   reward_coop (bool)
#   reward_compete (bool)
#   reward_mixed (bool)
#   population (int: 1=2 agents, 2=3-10, 3=>10)
#   architecture (evolved / neural / bayesian)
#   has_embodiment (bool)
#   has_plasticity (bool)
#   iterated_learning (bool)
#   has_noise (bool)
#   discrete_messages (bool)
#   continuous_messages (bool)

# Possible outcomes:
#   emerges             (any signal with > 0 MI)
#   compositional       (systematic structure)
#   robust              (generalizes to new observers)
#   scales              (works at larger populations)
#   honest              (truthful signals, not deception)
#   deceptive           (agents learn to lie)
#   fails               (no emergence despite training)


CLAIMS = [
    # ========= Classical foundations =========
    dict(paper_id="lewis1969",
         condition=[("reward_coop", "=", True), ("discrete_messages", "=", True)],
         outcome="emerges", strength=+1,
         reason="Lewis proves cooperation + discrete signaling converges to common code"),

    dict(paper_id="skyrms2010",
         condition=[("reward_coop", "=", True)],
         outcome="emerges", strength=+1,
         reason="Signals evolve under common-interest pressure; converge even from Hebbian"),

    dict(paper_id="zahavi1975_handicap",
         condition=[("reward_compete", "=", True), ("has_noise", "=", True)],
         outcome="honest", strength=+1,
         reason="Competition PLUS costly signals -> honest signaling (counter to pure coop claim)"),

    dict(paper_id="smith1974_games",
         condition=[("reward_compete", "=", True)],
         outcome="deceptive", strength=+1,
         reason="Pure competitive game theory allows deception as ESS"),

    dict(paper_id="axelrod1984",
         condition=[("reward_mixed", "=", True), ("iterated_learning", "=", True)],
         outcome="emerges", strength=+1,
         reason="Iterated prisoner dilemma -> cooperation + reputation signals emerge"),

    # ========= ML era cooperative =========
    dict(paper_id="foerster2016",
         condition=[("reward_coop", "=", True), ("population", ">=", 2)],
         outcome="emerges", strength=+1,
         reason="Cooperation + DRL message channel produces useful communication"),

    dict(paper_id="lazaridou2017",
         condition=[("reward_coop", "=", True)],
         outcome="emerges", strength=+1,
         reason="Reference game with shared reward produces protocol"),

    dict(paper_id="mordatch2018",
         condition=[("reward_coop", "=", True), ("has_embodiment", "=", True),
                    ("population", ">=", 3)],
         outcome="compositional", strength=+1,
         reason="Grounded multi-agent with 3+ agents -> compositional language"),

    dict(paper_id="chaabouni2020_compositionality",
         condition=[("reward_coop", "=", True), ("discrete_messages", "=", True)],
         outcome="compositional", strength=0,
         reason="Emerges BUT not always compositional — context-dependent"),

    dict(paper_id="kottur2017",
         condition=[("reward_coop", "=", True),
                    ("memory_limit", "=", True)],
         outcome="compositional", strength=+1,
         reason="Cooperation + bottleneck -> forced compositionality"),

    dict(paper_id="ren2020_iterated",
         condition=[("iterated_learning", "=", True)],
         outcome="compositional", strength=+1,
         reason="Iteration across generations increases structure"),

    dict(paper_id="cogswell2020_compositional",
         condition=[("population", ">=", 3)],
         outcome="compositional", strength=+1,
         reason="Population pressure forces more general protocols"),

    dict(paper_id="li2019_ease",
         condition=[("listener_changes", "=", True)],
         outcome="compositional", strength=+1,
         reason="Changing listeners ease-of-teaching -> structured language"),

    # ========= ML era competitive / mixed =========
    dict(paper_id="noukhovitch2021",
         condition=[("reward_compete", "=", True)],
         outcome="emerges", strength=0,
         reason="Under competition, partial emergence - weaker than coop"),

    dict(paper_id="deception_tom2023",
         condition=[("reward_compete", "=", True), ("tom_active", "=", True)],
         outcome="deceptive", strength=+1,
         reason="Competition + ToM -> deception emerges"),

    dict(paper_id="cao2018",
         condition=[("reward_mixed", "=", True), ("population", "=", 2)],
         outcome="emerges", strength=+1,
         reason="Mixed negotiation task -> communication emerges"),

    dict(paper_id="cao_competitive2018",
         condition=[("reward_compete", "=", True)],
         outcome="fails", strength=-1,
         reason="Pure competition often FAILS to produce informative comm"),

    dict(paper_id="jaques2019",
         condition=[("reward_mixed", "=", True), ("social_influence", "=", True)],
         outcome="emerges", strength=0,
         reason="Social influence as intrinsic reward partially works"),

    # ========= Evolutionary robotics =========
    dict(paper_id="werner1991",
         condition=[("reward_coop", "=", True), ("evolution", "=", True)],
         outcome="emerges", strength=+1,
         reason="Evolved agents develop communication given shared reward"),

    dict(paper_id="nolfi1999",
         condition=[("reward_coop", "=", True), ("evolution", "=", True),
                    ("population", ">=", 3)],
         outcome="emerges", strength=+1,
         reason="Evolved NN + coop foraging -> signaling"),

    dict(paper_id="floreano2007",
         condition=[("reward_coop", "=", True), ("evolution", "=", True),
                    ("kin_selection", "=", True)],
         outcome="emerges", strength=+1,
         reason="Kin selection specifically predicts communication emergence"),

    dict(paper_id="floreano2007",
         condition=[("reward_compete", "=", True), ("evolution", "=", True),
                    ("kin_selection", "=", False)],
         outcome="fails", strength=-1,
         reason="Non-kin competitive populations FAIL to evolve communication"),

    dict(paper_id="mirolli2013",
         condition=[("reward_mixed", "=", True), ("evolution", "=", True)],
         outcome="emerges", strength=0,
         reason="Mixed coop/compete -> partial comm, depends on balance"),

    dict(paper_id="swarm_plos2016",
         condition=[("evolution", "=", True), ("reward_coop", "=", True),
                    ("population", ">=", 10)],
         outcome="emerges", strength=+1,
         reason="Swarm sizes produce signaling-based swarming"),

    dict(paper_id="swarm_spiking2023",
         condition=[("evolution", "=", True), ("reward_coop", "=", True)],
         outcome="scales", strength=+1,
         reason="Spiking NN swarm scales emergence with foraging task"),

    # ========= Iterated learning / cultural =========
    dict(paper_id="kirby2014",
         condition=[("iterated_learning", "=", True)],
         outcome="compositional", strength=+1,
         reason="Iterated learning alone sufficient for compositional structure"),

    dict(paper_id="smith2003_iterated",
         condition=[("iterated_learning", "=", True), ("population", ">=", 3)],
         outcome="compositional", strength=+1,
         reason="Population + iteration -> compositional language"),

    dict(paper_id="spatial_pop2023",
         condition=[("population", ">=", 10), ("spatial_structure", "=", True)],
         outcome="scales", strength=-1,
         reason="Spatial structure IMPEDES language amalgamation - negative finding"),

    # ========= Theory of Mind =========
    dict(paper_id="yuan2019_tom_collab",
         condition=[("tom_active", "=", True), ("reward_coop", "=", True)],
         outcome="emerges", strength=+1,
         reason="ToM accelerates cooperative comm emergence"),

    dict(paper_id="zhao2023_matom_snn",
         condition=[("tom_active", "=", True)],
         outcome="emerges", strength=+1,
         reason="Brain-inspired ToM works in BOTH coop and compete"),

    dict(paper_id="oguntola2025_tom",
         condition=[("tom_active", "=", True), ("population", ">=", 4)],
         outcome="robust", strength=+1,
         reason="ToM enables generalization to unseen partners"),

    # ========= Hebbian / plasticity =========
    dict(paper_id="najarro2020_hebbian",
         condition=[("has_plasticity", "=", True), ("evolution", "=", True)],
         outcome="robust", strength=+1,
         reason="Plasticity + evolution -> adaptation to novel morphology"),

    dict(paper_id="risi2021_voxel_hebbian",
         condition=[("has_plasticity", "=", True)],
         outcome="robust", strength=+1,
         reason="Evolved plasticity rules generalize better than fixed weights"),

    dict(paper_id="reward_mod_hebbian2015",
         condition=[("has_plasticity", "=", True),
                    ("reward_gated_plasticity", "=", True)],
         outcome="emerges", strength=+1,
         reason="Reward-modulated Hebbian in compliant robotics"),

    # ========= Scale & neural arch =========
    dict(paper_id="chaabouni2022",
         condition=[("population", ">=", 100)],
         outcome="scales", strength=+1,
         reason="100-agent populations produce emergence at scale"),

    dict(paper_id="kim2021",
         condition=[("population", ">=", 10), ("reward_coop", "=", True)],
         outcome="emerges", strength=+1,
         reason="Larger populations+connectivity -> more robust emergence"),

    dict(paper_id="taniguchi2024",
         condition=[("bayesian_inference", "=", True), ("iterated_learning", "=", True)],
         outcome="emerges", strength=+1,
         reason="MH naming game: analytical Bayesian path to emergence"),

    dict(paper_id="taniguchi2025",
         condition=[("large_neural", "=", True)],
         outcome="emerges", strength=+1,
         reason="LLM-scale collective world model shows emergence"),

    # ========= Negative / failure claims =========
    dict(paper_id="rita2022",
         condition=[("reward_coop", "=", True), ("no_bottleneck", "=", True)],
         outcome="fails", strength=-1,
         reason="Without generalization pressure, emerged language overfits (not robust)"),

    dict(paper_id="lowe2019_pitfalls",
         condition=[("emergence_claim", "=", True)],
         outcome="emerges", strength=0,
         reason="Warns: MI-based emergence claims can be artifacts"),

    dict(paper_id="chaabouni2019_anti",
         condition=[("reward_coop", "=", True)],
         outcome="compositional", strength=-1,
         reason="Emerged languages are anti-efficient (NOT compositional)"),

    # ========= Our own experiments =========
    dict(paper_id="our_phase35_coop",
         condition=[("reward_coop", "=", True), ("evolution", "=", True),
                    ("population", "=", 2), ("n_neurons_small", "=", True)],
         outcome="emerges", strength=+1,
         reason="n=1 Kathara coop showed +0.057 bits MI (but n>1 needed)"),

    dict(paper_id="our_phase3_compete",
         condition=[("reward_compete", "=", True), ("evolution", "=", True),
                    ("population", "=", 2)],
         outcome="fails", strength=-1,
         reason="Competitive 1-food Kathara showed no signal"),

    dict(paper_id="our_coevo_red_queen",
         condition=[("reward_compete", "=", True), ("co_evolution_alternating", "=", True)],
         outcome="emerges", strength=+1,
         reason="Alternating Sentinel co-evolution produces ToM signatures"),

    dict(paper_id="our_rigorous_n3_random",
         condition=[("reward_coop", "=", True), ("random_init", "=", True),
                    ("short_training", "=", True)],
         outcome="emerges", strength=0,
         reason="Random init + short training -> unstable emergence"),

    dict(paper_id="our_hebbian_memory",
         condition=[("has_plasticity", "=", True), ("reward_gated_plasticity", "=", True)],
         outcome="emerges", strength=0,
         reason="3-factor plasticity 30% reach vs 20% leaky baseline"),
]


# Collect all observed features and outcomes
def all_features():
    feats = set()
    for c in CLAIMS:
        for f, op, v in c["condition"]:
            feats.add(f)
    return sorted(feats)


def all_outcomes():
    return sorted(set(c["outcome"] for c in CLAIMS))


if __name__ == "__main__":
    print(f"Total claims: {len(CLAIMS)}")
    print(f"Unique features used as conditions: {len(all_features())}")
    for f in all_features():
        n = sum(1 for c in CLAIMS
                if any(ff == f for ff, _, _ in c["condition"]))
        print(f"  {f:<30s} appears in {n} claims")
    print(f"\nUnique outcomes: {all_outcomes()}")
    for o in all_outcomes():
        n = sum(1 for c in CLAIMS if c["outcome"] == o)
        pos = sum(1 for c in CLAIMS if c["outcome"] == o and c["strength"] > 0)
        neg = sum(1 for c in CLAIMS if c["outcome"] == o and c["strength"] < 0)
        neu = sum(1 for c in CLAIMS if c["outcome"] == o and c["strength"] == 0)
        print(f"  {o:<14s} n={n:2d}  (+{pos} /-{neg} /0={neu})")
