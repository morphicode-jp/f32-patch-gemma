"""meta_research_papers.py - Curated dataset of emergent communication papers.

Scope: signaling-game / emergent language / communication-in-multi-agent.
~25 canonical + recent papers, hand-curated for consistency.
Each paper represented as a structured record.

Our own experiments are ALSO added at the end so Sentinel can position
our results within the literature.

Schema:
  paper_id, title, year, venue/arxiv
  n_agents          : int
  n_params_log10    : float (log10 of learnable params; 0 for tabular)
  arch              : int (1=tabular/Bayesian, 2=small NN <1K, 3=RNN/LSTM medium,
                           4=deep transformer, 5=evolved NN, 6=other)
  learning          : int (1=gradient, 2=RL/policy-grad, 3=evolution/Sentinel,
                           4=Hebbian/plasticity, 5=Bayesian/analytical, 6=hybrid)
  task              : int (1=reference game, 2=navigation/nav+signal, 3=negotiation,
                           4=open-ended, 5=foraging/survival, 6=abstract)
  reward            : int (1=cooperative, 2=competitive, 3=mixed, 4=none/unsup)
  env_dim           : int (1=abstract/symbolic, 2=2D grid/plane, 3=3D)
  has_embodiment    : int (0=no, 1=yes)
  uses_plasticity   : int (0=no, 1=yes - online learning or Hebbian)
  emergence_reported: float (0.0=no, 0.5=partial/weak, 1.0=clear emergence claim)
"""

PAPERS = [
    # === Classical / Foundational ===
    dict(paper_id="lewis1969", title="Convention: A Philosophical Study",
         year=1969, venue="Harvard",
         n_agents=2, n_params_log10=0, arch=1, learning=5, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="skyrms2010", title="Signals: Evolution, Learning & Information",
         year=2010, venue="Oxford",
         n_agents=2, n_params_log10=0, arch=1, learning=3, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    # === Early ML / RL era ===
    dict(paper_id="steels2001", title="Language Games for autonomous robots",
         year=2001, venue="IEEE Intell Syst",
         n_agents=8, n_params_log10=3, arch=2, learning=2, task=1, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=1, emergence_reported=1.0),

    dict(paper_id="foerster2016", title="Learning to Communicate with DRL",
         year=2016, venue="NIPS",
         n_agents=3, n_params_log10=4.5, arch=3, learning=2, task=3, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="lazaridou2017", title="Multi-Agent Cooperation and the Emergence of (Natural) Language",
         year=2017, venue="ICLR",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="havrylov2017", title="Emergence of Language with Multi-agent Games: Sequences of Symbols",
         year=2017, venue="NIPS",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="mordatch2018", title="Emergence of Grounded Compositional Language in Multi-Agent Populations",
         year=2018, venue="AAAI",
         n_agents=3, n_params_log10=5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="das2017", title="Learning cooperative visual dialog agents with DRL",
         year=2017, venue="ICCV",
         n_agents=2, n_params_log10=6, arch=3, learning=2, task=3, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="cao2018", title="Emergent Communication through Negotiation",
         year=2018, venue="ICLR",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=3, reward=3,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="evtimova2017", title="Emergent Communication in a Multi-Modal, Multi-Step Referential Game",
         year=2017, venue="ICLR",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    # === Competitive / mixed conditions ===
    dict(paper_id="cao_competitive2018",
         title="Emergent Communication in Competitive Multi-Agent Games",
         year=2018, venue="workshop",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=3, reward=2,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="noukhovitch2021",
         title="Emergent Communication under Competition",
         year=2021, venue="AAMAS",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=3, reward=2,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    # === Open-ended / populations ===
    dict(paper_id="chaabouni2022",
         title="Emergent Communication at Scale",
         year=2022, venue="ICLR",
         n_agents=100, n_params_log10=7, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="rita2022",
         title="Emergent Communication: Generalization and Overfitting in Lewis Games",
         year=2022, venue="NeurIPS",
         n_agents=2, n_params_log10=6, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="kim2021",
         title="Emergent Communication under Varying Sizes and Connectivities",
         year=2021, venue="NeurIPS",
         n_agents=10, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    # === Embodied / grid world ===
    dict(paper_id="jaques2019",
         title="Social Influence as Intrinsic Motivation for Multi-Agent DRL",
         year=2019, venue="ICML",
         n_agents=5, n_params_log10=5.5, arch=3, learning=2, task=5, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="bogin2018",
         title="Emergence of Communication in an Interactive World with Consistent Speakers",
         year=2018, venue="workshop",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    # === Evolutionary NN approaches ===
    dict(paper_id="nolfi1999",
         title="Evolving communication with evolved NN agents",
         year=1999, venue="Artif Life",
         n_agents=10, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="werner1991",
         title="Evolution of communication in artificial organisms",
         year=1991, venue="ALife",
         n_agents=20, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="floreano2007",
         title="Evolutionary conditions for emergence of communication in robots",
         year=2007, venue="Curr Biol",
         n_agents=10, n_params_log10=3, arch=5, learning=3, task=5, reward=3,
         env_dim=3, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    # === 2023-2024 recent (from web search) ===
    dict(paper_id="rita2023",
         title="So many design choices: Improving neural agent communication",
         year=2023, venue="ACL Findings",
         n_agents=2, n_params_log10=6, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="taniguchi2024",
         title="Emergent Communication of Multimodal Deep Generative Models (MH Naming Game)",
         year=2024, venue="Front. Robot. AI",
         n_agents=2, n_params_log10=7, arch=4, learning=5, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="boldt2024",
         title="Emergent Language in Open-Ended Environments",
         year=2024, venue="arXiv",
         n_agents=4, n_params_log10=6, arch=4, learning=2, task=4, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="taniguchi2025",
         title="Generative Emergent Communication: LLM as Collective World Model",
         year=2025, venue="arXiv",
         n_agents=2, n_params_log10=10, arch=4, learning=1, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    # === Compositional / iterated learning ===
    dict(paper_id="ren2020_iterated",
         title="Compositional Languages Emerge in a Neural Iterated Learning Model",
         year=2020, venue="ICLR",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="cogswell2020_compositional",
         title="Emergence of Compositional Language via Population-based Training",
         year=2020, venue="ICLR",
         n_agents=8, n_params_log10=6, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="chaabouni2019_anti",
         title="Anti-efficient encoding in emergent communication",
         year=2019, venue="NeurIPS",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="chaabouni2020_compositionality",
         title="Compositionality and Generalization In Emergent Languages",
         year=2020, venue="ACL",
         n_agents=2, n_params_log10=6, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="li2019_ease",
         title="Ease-of-teaching and language structure from emergent communication",
         year=2019, venue="NeurIPS",
         n_agents=4, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="kharitonov2019_egg",
         title="EGG: Toolkit for research on Emergence of lanGuage in Games",
         year=2019, venue="EMNLP demo",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="andreas2017_analogs",
         title="Analogs of Linguistic Structure in Deep Representations",
         year=2017, venue="EMNLP",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="graesser2019",
         title="Emergent linguistic phenomena in Multi-Agent Communication Games",
         year=2019, venue="EMNLP",
         n_agents=4, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="lowe2019_pitfalls",
         title="On the Pitfalls of Measuring Emergent Communication",
         year=2019, venue="AAMAS",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    # === Larger-scale / biases / curricula ===
    dict(paper_id="eccles2019_biases",
         title="Biases for emergent communication in MARL",
         year=2019, venue="NeurIPS",
         n_agents=3, n_params_log10=5.5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="bullard2021_zeroshot",
         title="Exploring Zero-Shot Emergent Communication in Embodied Multi-Agent",
         year=2021, venue="arXiv",
         n_agents=2, n_params_log10=5.5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="lazaridou2020_survey",
         title="Emergent Multi-Agent Communication in the Deep Learning Era (survey)",
         year=2020, venue="arXiv",
         n_agents=3, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="ren2021_decouple",
         title="How to decouple semantic and syntactic in emergent communication",
         year=2021, venue="NAACL",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="vanneste2022_analysis",
         title="Analysis of emergent communication under varying behaviors",
         year=2022, venue="AAMAS",
         n_agents=3, n_params_log10=5, arch=3, learning=2, task=2, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    # === Theory of Mind ===
    dict(paper_id="yuan2019_tom_collab",
         title="Emergence of Theory of Mind Collaboration in Multiagent Systems",
         year=2019, venue="NeurIPS ws",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="oguntola2025_tom",
         title="Theory of Mind in Multi-Agent Systems (CMU PhD thesis)",
         year=2025, venue="CMU",
         n_agents=4, n_params_log10=6, arch=3, learning=2, task=2, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="zhao2023_matom_snn",
         title="Brain-inspired ToM spiking NN for multi-agent coop/compete",
         year=2023, venue="Patterns (Cell)",
         n_agents=4, n_params_log10=4, arch=5, learning=2, task=2, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=1, emergence_reported=1.0),

    dict(paper_id="mate2022",
         title="MATE: Multi-Agent Task Execution via Emergent Communication",
         year=2022, venue="NeurIPS",
         n_agents=4, n_params_log10=6, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="deception_tom2023",
         title="Emerging deception and skepticism via ToM",
         year=2023, venue="PMC",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=3, reward=2,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    # === Evolutionary / embodied robotics ===
    dict(paper_id="quinn2000_diff_com",
         title="Evolving communication without dedicated comm channel",
         year=2000, venue="ALife",
         n_agents=4, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="cangelosi2001_sg",
         title="Symbol grounding and symbolic theft hypothesis",
         year=2001, venue="Connection Science",
         n_agents=10, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="marocco2002",
         title="The emergence of communication in evolutionary robots",
         year=2002, venue="ALife",
         n_agents=6, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="wagner2003",
         title="Progress in the simulation of emergent communication",
         year=2003, venue="AI Magazine",
         n_agents=10, n_params_log10=3, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="mirolli2013",
         title="Evolution of language and cooperation in populations",
         year=2013, venue="Adapt Behav",
         n_agents=20, n_params_log10=3, arch=5, learning=3, task=5, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="swarm_spiking2023",
         title="Emergent communication enhances foraging in evolved spiking swarms",
         year=2023, venue="Swarm Intelligence",
         n_agents=20, n_params_log10=3, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="epuck_signaling2024",
         title="Improvement of signal communication for foraging (e-puck)",
         year=2024, venue="J. Appl Res Tech",
         n_agents=4, n_params_log10=3, arch=5, learning=3, task=5, reward=1,
         env_dim=3, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="swarm_plos2016",
         title="Emergence of Swarming Behavior: Foraging Agents with Signaling",
         year=2016, venue="PLOS ONE",
         n_agents=30, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    # === Hebbian / plasticity approaches ===
    dict(paper_id="najarro2020_hebbian",
         title="Meta-Learning through Hebbian Plasticity in Random Networks",
         year=2020, venue="NeurIPS",
         n_agents=1, n_params_log10=3, arch=5, learning=4, task=4, reward=4,
         env_dim=3, has_embodiment=1, uses_plasticity=1, emergence_reported=1.0),

    dict(paper_id="risi2021_voxel_hebbian",
         title="Evolving Hebbian Learning Rules in Voxel-based Soft Robots",
         year=2021, venue="GECCO",
         n_agents=1, n_params_log10=3, arch=5, learning=4, task=5, reward=4,
         env_dim=3, has_embodiment=1, uses_plasticity=1, emergence_reported=1.0),

    dict(paper_id="reward_mod_hebbian2015",
         title="Reward-Modulated Hebbian Plasticity in Compliant Robotics",
         year=2015, venue="Front. Neurorobotics",
         n_agents=1, n_params_log10=3, arch=5, learning=4, task=5, reward=4,
         env_dim=3, has_embodiment=1, uses_plasticity=1, emergence_reported=1.0),

    # === Iterated learning / cultural transmission ===
    dict(paper_id="kirby2014",
         title="Iterated learning and the evolution of language",
         year=2014, venue="Current Opin Neurobiol",
         n_agents=10, n_params_log10=0, arch=1, learning=5, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="smith2003_iterated",
         title="Iterated learning: Framework for emergence of language",
         year=2003, venue="ALife",
         n_agents=10, n_params_log10=0, arch=1, learning=5, task=1, reward=1,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="spatial_pop2023",
         title="Spatial community structure impedes language amalgamation",
         year=2023, venue="ALIFE Proc",
         n_agents=50, n_params_log10=3, arch=5, learning=5, task=1, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="social_learning2024",
         title="Framework for emergence/analysis of language in social learning agents",
         year=2024, venue="Nat Commun",
         n_agents=4, n_params_log10=5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    # === Classical biological / game theoretic foundations ===
    dict(paper_id="smith1974_games",
         title="The Theory of Games and Evolution of Animal Conflicts",
         year=1974, venue="J Theor Biol",
         n_agents=2, n_params_log10=0, arch=1, learning=5, task=6, reward=2,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="zahavi1975_handicap",
         title="Mate selection: a selection for a handicap",
         year=1975, venue="J Theor Biol",
         n_agents=2, n_params_log10=0, arch=1, learning=5, task=1, reward=2,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="axelrod1984",
         title="The Evolution of Cooperation",
         year=1984, venue="Basic Books",
         n_agents=2, n_params_log10=0, arch=1, learning=5, task=6, reward=3,
         env_dim=1, has_embodiment=0, uses_plasticity=0, emergence_reported=1.0),

    # === Swarm / collective ===
    dict(paper_id="embodied_evo2017",
         title="Learning collaborative foraging via embodied evolution",
         year=2017, venue="IEEE",
         n_agents=20, n_params_log10=3, arch=5, learning=3, task=5, reward=1,
         env_dim=3, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="diff_communicative2017",
         title="Evolution via differentiation of comm and goal-directed behaviors",
         year=2017, venue="Artif Life Robotics",
         n_agents=4, n_params_log10=2, arch=5, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="role_allocation2018",
         title="Communication for Scalable Role Allocation in Collective Robotics",
         year=2018, venue="ANTS",
         n_agents=10, n_params_log10=3, arch=5, learning=3, task=5, reward=1,
         env_dim=3, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    # === Additional MARL / networked comm ===
    dict(paper_id="networked_marl2020",
         title="Networked Multi-Agent RL with Emergent Communication",
         year=2020, venue="AAMAS",
         n_agents=8, n_params_log10=5.5, arch=3, learning=2, task=2, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="curiosity_comm2020",
         title="Emergent communication through curiosity-driven MARL",
         year=2020, venue="ANR",
         n_agents=2, n_params_log10=5, arch=3, learning=2, task=4, reward=4,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    # === Our experiments (for positioning) ===
    dict(paper_id="our_phase3_compete",
         title="Kathara Sentinel: competitive signaling (1-food)",
         year=2026, venue="self",
         n_agents=2, n_params_log10=1.96, arch=6, learning=3, task=5, reward=2,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.0),

    dict(paper_id="our_phase35_coop",
         title="Kathara Sentinel: cooperative signaling (2-food)",
         year=2026, venue="self",
         n_agents=2, n_params_log10=1.96, arch=6, learning=3, task=5, reward=1,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),

    dict(paper_id="our_rigorous_n3_random",
         title="Kathara rigorous test n=3 random init",
         year=2026, venue="self",
         n_agents=2, n_params_log10=1.96, arch=6, learning=3, task=5, reward=3,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=0.5),

    dict(paper_id="our_hebbian_memory",
         title="Kathara Hebbian memory (Phase 5) - not language, but related subsystem",
         year=2026, venue="self",
         n_agents=1, n_params_log10=1.96, arch=6, learning=4, task=5, reward=4,
         env_dim=2, has_embodiment=1, uses_plasticity=1, emergence_reported=0.5),

    dict(paper_id="our_dual_hippo",
         title="Kathara Dual brain + replay (Phase 6)",
         year=2026, venue="self",
         n_agents=1, n_params_log10=2.27, arch=6, learning=4, task=5, reward=4,
         env_dim=2, has_embodiment=1, uses_plasticity=1, emergence_reported=0.5),

    dict(paper_id="our_coevo_red_queen",
         title="Kathara co-evolution Red Queen (Phase 2)",
         year=2026, venue="self",
         n_agents=2, n_params_log10=1.96, arch=6, learning=3, task=5, reward=2,
         env_dim=2, has_embodiment=1, uses_plasticity=0, emergence_reported=1.0),
]


# Consistent numerical attribute list for analysis
ATTRIBUTES = [
    "n_agents", "n_params_log10", "arch", "learning", "task",
    "reward", "env_dim", "has_embodiment", "uses_plasticity",
]


def to_matrix():
    """Return (X, y, paper_ids) arrays for analysis."""
    import numpy as np
    X = np.array([[p[a] for a in ATTRIBUTES] for p in PAPERS], dtype=np.float64)
    y = np.array([p["emergence_reported"] for p in PAPERS], dtype=np.float64)
    ids = [p["paper_id"] for p in PAPERS]
    return X, y, ids


if __name__ == "__main__":
    import numpy as np
    X, y, ids = to_matrix()
    print(f"Dataset: {len(PAPERS)} papers x {len(ATTRIBUTES)} attributes")
    print(f"Emergence distribution: "
          f"{int((y == 0).sum())} zero, "
          f"{int((y == 0.5).sum())} partial, "
          f"{int((y == 1.0).sum())} clear")
    print(f"\nAttribute ranges:")
    for i, a in enumerate(ATTRIBUTES):
        print(f"  {a:<18s}: {X[:, i].min():.1f} - {X[:, i].max():.1f}, "
              f"mean={X[:, i].mean():.2f}")
    print(f"\nOur papers (for positioning):")
    for p in PAPERS:
        if p["venue"] == "self":
            print(f"  {p['paper_id']}: emergence={p['emergence_reported']}")
