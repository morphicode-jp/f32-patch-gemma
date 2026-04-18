"""meta_optimal_design.py - Use Sentinel on LaD v2 claims to find optimal design.

Given 46 empirical claims from the literature (see meta_logic_claims.py),
search the DESIGN SPACE (feature conjunctions) to find:

  1. Highest literature-supported design (what existing claims predict works)
  2. Highest NOVEL design (unexplored but implied by claim structure)
  3. Highest guard: supported AND novel (Sentinel's dual objective)

This is literal Sentinel application to paper claims: design_score (eval)
vs novelty (guard).

Output: top-K design blueprints with their supporting evidence.
"""
import os
import sys
import json
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta_logic_claims import CLAIMS, all_features, all_outcomes


# =====================================================================
# Feature encoding
# =====================================================================

FEATS = all_features()  # canonical order

# Mutually-exclusive feature groups (constraint: at most 1 can be True)
EXCLUSIVE_GROUPS = [
    ["reward_coop", "reward_compete", "reward_mixed"],  # reward type
]

# Continuous/numeric features (need to handle non-boolean)
# In our current dataset: "population" can be 2, 3, 10, 100
NUMERIC_FEATURES = {"population"}

# For population, define the discrete choice set
POPULATION_CHOICES = [1, 2, 3, 10, 100]  # log-spaced biological scales


def encode_design(feature_dict):
    """Convert {feature: value} dict to canonical vector.

    Returns list of floats in range [-1, 1].
    feature_dict contains bool or int values per FEATS entry.
    Missing features default to -1 (absent).
    """
    vec = []
    for f in FEATS:
        if f not in feature_dict:
            vec.append(-1.0)
        elif f in NUMERIC_FEATURES:
            # encode numeric via log scale
            v = feature_dict[f]
            vec.append(float(np.log10(max(1, v)) / 3.0))  # 0..~0.67
        else:
            v = feature_dict[f]
            vec.append(1.0 if v else -1.0)
    return vec


def decode_design(vec):
    """Convert canonical vector back to {feature: value} dict."""
    out = {}
    for i, f in enumerate(FEATS):
        v = vec[i]
        if f in NUMERIC_FEATURES:
            # inverse of log scale
            if v > -0.5:
                # Map back to nearest POPULATION_CHOICES
                exp_v = 10 ** (v * 3.0)
                best = min(POPULATION_CHOICES, key=lambda x: abs(x - exp_v))
                out[f] = best
        else:
            if v > 0.5:
                out[f] = True
            elif v < -0.5:
                out[f] = False
            # else: uncertain/absent, skip
    return out


# =====================================================================
# Design scoring (eval_fn on claim space)
# =====================================================================

def conditions_match(conditions, design):
    """Does this claim's condition list match the design?

    conditions: list of (feature, operator, value) triples from a claim
    design: dict of {feature: value} for the candidate design
    Returns: True if ALL conditions are satisfied.
    """
    for (f, op, v) in conditions:
        if f not in design:
            # feature not specified in design = cannot satisfy condition
            return False
        d = design[f]
        if op == "=":
            if d != v: return False
        elif op == ">=":
            if not (d >= v): return False
        elif op == "<=":
            if not (d <= v): return False
        elif op == ">":
            if not (d > v): return False
    return True


def design_score(design):
    """Score a design based on matching claims.

    Returns dict:
      score: net positive claim rate
      n_matches: total claims matching
      n_positive, n_negative, n_neutral
      novelty: 1 / (1 + n_matches)  (higher = more novel)
      matches: list of matched claim paper_ids
      outcomes: counter of predicted outcomes
    """
    pos = neg = neu = 0
    matches = []
    outcome_counts = {}
    for c in CLAIMS:
        if conditions_match(c["condition"], design):
            matches.append(c["paper_id"])
            if c["strength"] > 0: pos += 1
            elif c["strength"] < 0: neg += 1
            else: neu += 1
            outcome_counts[c["outcome"]] = outcome_counts.get(c["outcome"], 0) + 1
    n = pos + neg + neu
    if n == 0:
        # No direct match: soft scoring via feature-level priors
        feat_score = 0.0
        feat_count = 0
        for f, v in design.items():
            for c in CLAIMS:
                for (cf, cop, cv) in c["condition"]:
                    if cf == f and cv == v:
                        feat_score += c["strength"]
                        feat_count += 1
        if feat_count == 0:
            score = 0.0
        else:
            score = feat_score / feat_count
        return {
            "score": score,
            "n_matches": 0,
            "n_positive": 0, "n_negative": 0, "n_neutral": 0,
            "novelty": 1.0,
            "matches": [],
            "outcomes": {},
            "n_feature_support": feat_count,
        }
    # Net positive rate, weighted by strength
    score = (pos - neg) / n
    return {
        "score": score,
        "n_matches": n,
        "n_positive": pos, "n_negative": neg, "n_neutral": neu,
        "novelty": 1.0 / (1 + n),
        "matches": matches,
        "outcomes": outcome_counts,
    }


def validate_design(design):
    """Check mutually exclusive constraints satisfied."""
    for group in EXCLUSIVE_GROUPS:
        active = sum(1 for f in group if design.get(f, False) is True)
        if active > 1:
            return False
    return True


# =====================================================================
# Exhaustive top-K search
# =====================================================================

def enumerate_designs(max_features=5):
    """Enumerate designs of up to max_features (non-default) features.

    For tractability: enumerate CONJUNCTIONS of binary features + one
    population choice.
    """
    # Binary features (exclude numeric population)
    bin_feats = [f for f in FEATS if f not in NUMERIC_FEATURES]
    designs = []

    # Try each combination size from 1 to max_features
    for size in range(1, max_features + 1):
        for combo in itertools.combinations(bin_feats, size):
            # Check exclusivity
            for group in EXCLUSIVE_GROUPS:
                if sum(1 for f in combo if f in group) > 1:
                    break
            else:
                # No violation — enumerate with population choices
                for pop in POPULATION_CHOICES:
                    d = {f: True for f in combo}
                    d["population"] = pop
                    designs.append(d)
                # Also without population constraint
                d = {f: True for f in combo}
                designs.append(d)
    return designs


def rank_designs(designs, min_matches=2, mode="supported"):
    """Rank designs by score.

    mode:
      'supported' -- sort by score (higher = more positive claims match)
      'novel'     -- sort by novelty for unmatched-design scoring
      'balanced'  -- score * (1 + novelty_weight) for Sentinel-like dual
    """
    scored = []
    for d in designs:
        s = design_score(d)
        if mode == "supported":
            if s["n_matches"] >= min_matches:
                rank = s["score"]
            else:
                rank = -99  # skip
        elif mode == "novel":
            # Use feature-level score when no direct match
            rank = s["score"] - 0.1 * max(0, s["n_matches"])
        elif mode == "balanced":
            # Boost score by novelty, but require some feature support
            if s["n_matches"] >= 1 or s.get("n_feature_support", 0) >= 3:
                rank = s["score"] + 0.3 * s["novelty"]
            else:
                rank = -99
        else:
            rank = s["score"]
        scored.append((rank, d, s))
    scored.sort(key=lambda t: -t[0])
    return scored


def format_design(d):
    """Human-readable design string."""
    parts = []
    for k, v in sorted(d.items()):
        if k in NUMERIC_FEATURES:
            parts.append(f"{k}={v}")
        elif v is True:
            parts.append(k)
        elif v is False:
            parts.append(f"!{k}")
    return " & ".join(parts)


def main():
    print("=" * 70)
    print("  OPTIMAL DESIGN SEARCH via LaD v2")
    print("=" * 70)

    # Enumerate candidate designs
    print("\n[enum] generating candidate designs (conjunctions)...")
    designs = enumerate_designs(max_features=3)
    print(f"  Total candidate designs: {len(designs)}")

    # --- Supported designs (backed by existing claims) ---
    print("\n" + "=" * 70)
    print("  TOP LITERATURE-SUPPORTED DESIGNS (n_matches >= 2)")
    print("=" * 70)
    scored = rank_designs(designs, min_matches=2, mode="supported")
    for rank, d, s in scored[:10]:
        print(f"\n  score={s['score']:+.2f}  matches={s['n_matches']} "
              f"(+{s['n_positive']}/-{s['n_negative']}/={s['n_neutral']})  "
              f"novelty={s['novelty']:.2f}")
        print(f"    design: {format_design(d)}")
        print(f"    outcomes: {s['outcomes']}")
        if s['matches']:
            print(f"    supported by: {', '.join(s['matches'][:4])}"
                  f"{'...' if len(s['matches']) > 4 else ''}")

    # --- Balanced (supported + novel) ---
    print("\n" + "=" * 70)
    print("  TOP BALANCED DESIGNS (supported AND novel)")
    print("=" * 70)
    scored_bal = rank_designs(designs, mode="balanced")
    # Filter to keep distinct designs (no subsumption)
    seen = set()
    balanced_top = []
    for rank, d, s in scored_bal:
        key = tuple(sorted(d.items()))
        if key in seen: continue
        seen.add(key)
        balanced_top.append((rank, d, s))
        if len(balanced_top) >= 10: break
    for rank, d, s in balanced_top:
        print(f"\n  rank={rank:+.2f}  score={s['score']:+.2f}  "
              f"matches={s['n_matches']}  novelty={s['novelty']:.2f}")
        print(f"    design: {format_design(d)}")

    # --- Untried designs (high predicted, 0 matches) ---
    print("\n" + "=" * 70)
    print("  TOP UNEXPLORED DESIGNS (0 direct claim matches)")
    print("=" * 70)
    unmatched = [(d, s) for d in designs for s in [design_score(d)]
                 if s["n_matches"] == 0 and s.get("n_feature_support", 0) >= 3]
    # Rank by feature-level score
    unmatched.sort(key=lambda t: -t[1]["score"])
    for d, s in unmatched[:10]:
        print(f"\n  feature_score={s['score']:+.2f}  "
              f"n_feature_support={s.get('n_feature_support', 0)}")
        print(f"    design: {format_design(d)}")

    # --- Our own current designs ---
    print("\n" + "=" * 70)
    print("  OUR CURRENT DESIGNS in this design space")
    print("=" * 70)
    our = [
        ("our_phase35_coop", {"reward_coop": True, "evolution": True, "population": 2,
                              "n_neurons_small": True}),
        ("our_phase3_compete", {"reward_compete": True, "evolution": True, "population": 2}),
        ("our_coevo_red_queen", {"reward_compete": True, "co_evolution_alternating": True}),
        ("our_rigorous_n3_random", {"reward_coop": True, "random_init": True,
                                     "short_training": True}),
        ("our_hebbian_memory", {"has_plasticity": True, "reward_gated_plasticity": True}),
    ]
    for name, d in our:
        s = design_score(d)
        print(f"\n  [{name}]  score={s['score']:+.2f}  "
              f"matches={s['n_matches']} (+{s['n_positive']}/-{s['n_negative']})")
        print(f"    design: {format_design(d)}")

    # --- Final recommendation ---
    print("\n" + "=" * 70)
    print("  RECOMMENDED NEXT EXPERIMENT (highest balanced rank)")
    print("=" * 70)
    if balanced_top:
        top_rank, top_d, top_s = balanced_top[0]
        print(f"\n  Design: {format_design(top_d)}")
        print(f"  Literature score: {top_s['score']:+.2f}")
        print(f"  Existing support: {top_s['n_matches']} claims "
              f"(+{top_s['n_positive']}/-{top_s['n_negative']})")
        print(f"  Supporting papers: {', '.join(top_s['matches'][:5])}")
        print(f"  Novelty: {top_s['novelty']:.2f}")

    # Save
    out = {
        "n_candidates": len(designs),
        "top_supported": [
            {"rank": r, "design": format_design(d), "score": s["score"],
             "matches": s["matches"], "n_positive": s["n_positive"],
             "n_negative": s["n_negative"], "outcomes": s["outcomes"]}
            for r, d, s in scored[:15]
        ],
        "top_balanced": [
            {"rank": r, "design": format_design(d), "score": s["score"],
             "matches": s["matches"], "novelty": s["novelty"]}
            for r, d, s in balanced_top[:15]
        ],
        "top_unexplored": [
            {"design": format_design(d), "feature_score": s["score"],
             "n_feature_support": s.get("n_feature_support", 0)}
            for d, s in unmatched[:15]
        ],
        "our_designs": [
            {"name": name, "design": format_design(d),
             "score": design_score(d)["score"],
             "matches": design_score(d)["matches"]}
            for name, d in our
        ],
    }
    with open("optimal_design_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: optimal_design_result.json")


if __name__ == "__main__":
    main()
