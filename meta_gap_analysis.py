"""meta_gap_analysis.py - Literature coverage gap analysis (Priority B).

Identify (feature x outcome) cells where n < threshold (under-sampled).
Prioritize gaps in structurally important features. Generate concrete
paper-search queries to close gaps efficiently.
"""
import os
import sys
import json
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta_logic_claims import CLAIMS, all_features, all_outcomes


def coverage_matrix():
    """Return dict (feat_value_str, outcome) -> list of paper_ids."""
    cov = defaultdict(list)
    for c in CLAIMS:
        for (f, _, v) in c["condition"]:
            key = (f"{f}={v}", c["outcome"])
            cov[key].append(c["paper_id"])
    return cov


def coverage_summary(cov):
    """Per-feature coverage count across outcomes."""
    summary = defaultdict(lambda: {"total": 0, "outcomes": Counter()})
    for (fv, outcome), papers in cov.items():
        feat = fv.split("=")[0]
        summary[feat]["total"] += len(papers)
        summary[feat]["outcomes"][outcome] += len(papers)
    return summary


def find_gaps(cov, min_n=2):
    """Cells where fewer than min_n claims exist for (feature_value, outcome)."""
    gaps = []
    observed_keys = set(cov.keys())
    for outcome in all_outcomes():
        for f in all_features():
            # Only check True (active) values to keep space tractable
            key = (f"{f}=True", outcome)
            n = len(cov.get(key, []))
            if n < min_n:
                gaps.append({
                    "feature": f,
                    "outcome": outcome,
                    "n_claims": n,
                    "cells_shortfall": min_n - n,
                })
    return gaps


def feature_importance_from_outcomes(cov):
    """Rough importance: feature with many distinct outcome associations = important."""
    fv_outcomes = defaultdict(set)
    for (fv, outcome), papers in cov.items():
        feat = fv.split("=")[0]
        fv_outcomes[feat].add(outcome)
    return {f: len(os) for f, os in fv_outcomes.items()}


def prioritize_gaps(gaps, importance):
    """Rank gaps: prioritize those in features with many outcome associations
    (structurally important) AND outcomes with low total sample."""
    outcome_totals = Counter()
    for c in CLAIMS:
        outcome_totals[c["outcome"]] += 1

    for g in gaps:
        imp = importance.get(g["feature"], 0)
        # Priority: higher importance + rarer outcome + larger shortfall
        outcome_rarity = 1.0 / (1 + outcome_totals[g["outcome"]])
        g["priority_score"] = imp * 10 + outcome_rarity * 20 + g["cells_shortfall"] * 3
    gaps.sort(key=lambda g: -g["priority_score"])
    return gaps


def make_search_query(feature, outcome):
    """Translate (feature, outcome) gap into a targeted search query."""
    # Feature-specific phrasing
    feature_phrases = {
        "reward_coop": "cooperative reward",
        "reward_compete": "competitive",
        "reward_mixed": "mixed-motive",
        "reward_gated_plasticity": "reward-modulated plasticity",
        "has_plasticity": "synaptic plasticity",
        "has_noise": "noisy signals handicap",
        "iterated_learning": "iterated learning language transmission",
        "tom_active": "theory of mind multi-agent",
        "evolution": "evolutionary neural network",
        "co_evolution_alternating": "alternating co-evolution communication",
        "bayesian_inference": "Bayesian naming game",
        "discrete_messages": "discrete symbol communication",
        "kin_selection": "kin selection cooperation emergence",
        "large_neural": "large language model emergent communication",
        "n_neurons_small": "small neural agents minimal communication",
        "listener_changes": "varying listener population language",
        "spatial_structure": "spatial population emergent language",
        "social_influence": "social influence intrinsic reward",
        "has_embodiment": "embodied multi-agent communication",
        "memory_limit": "bottleneck compositional language",
    }
    outcome_phrases = {
        "emerges": "emergence evolution",
        "compositional": "compositional structure",
        "robust": "generalization transfer",
        "scales": "scaling laws population",
        "honest": "honest signaling handicap",
        "deceptive": "deception adversarial communication",
        "fails": "communication failure negative result",
    }
    feat_txt = feature_phrases.get(feature, feature.replace("_", " "))
    out_txt = outcome_phrases.get(outcome, outcome)
    return f"{feat_txt} {out_txt}"


def main():
    print("=" * 70)
    print("  LITERATURE GAP ANALYSIS")
    print("=" * 70)
    print(f"\n  Total claims in dataset: {len(CLAIMS)}")

    cov = coverage_matrix()
    summary = coverage_summary(cov)

    # Per-feature coverage
    print(f"\n  Coverage by feature (total claims using this feature):")
    for feat in sorted(summary.keys(), key=lambda k: -summary[k]["total"])[:15]:
        s = summary[feat]
        outs = ", ".join(f"{o}:{n}" for o, n in s["outcomes"].most_common())
        print(f"    {feat:<28s} total={s['total']:<3d}  breakdown: {outs}")

    # Per-outcome total
    print(f"\n  Coverage by outcome:")
    outcome_totals = Counter(c["outcome"] for c in CLAIMS)
    for o, n in outcome_totals.most_common():
        print(f"    {o:<16s} {n} claims")

    # Find gaps
    print(f"\n" + "=" * 70)
    print(f"  GAPS: (feature, outcome) with < 2 claims")
    print("=" * 70)
    gaps = find_gaps(cov, min_n=2)
    importance = feature_importance_from_outcomes(cov)
    gaps = prioritize_gaps(gaps, importance)

    print(f"\n  Total under-sampled cells: {len(gaps)}")
    print(f"  (out of {len(all_features()) * len(all_outcomes())} possible cells)")

    # Top-priority gaps (to close first)
    print(f"\n  TOP 20 PRIORITY GAPS TO CLOSE:")
    print(f"  {'priority':>8s}  {'feature':<28s}  {'outcome':<14s}  {'n':>3s}  {'search query':<50s}")
    for g in gaps[:20]:
        q = make_search_query(g["feature"], g["outcome"])
        print(f"  {g['priority_score']:>8.1f}  "
              f"{g['feature']:<28s}  {g['outcome']:<14s}  "
              f"{g['n_claims']:>3d}  {q[:45]:<50s}")

    # Generate 10 specific search queries
    print(f"\n  RECOMMENDED SEARCHES (top 10 highest priority, distinct features):")
    seen_feats = set()
    searches = []
    for g in gaps:
        if g["feature"] in seen_feats:
            continue
        if len(searches) >= 10:
            break
        seen_feats.add(g["feature"])
        q = make_search_query(g["feature"], g["outcome"])
        searches.append({
            "priority_rank": len(searches) + 1,
            "feature": g["feature"],
            "outcome_target": g["outcome"],
            "current_n": g["n_claims"],
            "query": q,
            "reason": f"gap in {g['feature']} -> {g['outcome']} ({g['n_claims']} claims)",
        })
        print(f"    {searches[-1]['priority_rank']}. \"{q}\"")
        print(f"       (fills: {g['feature']} -> {g['outcome']}, current n={g['n_claims']})")

    # Save
    out = {
        "n_claims": len(CLAIMS),
        "coverage_by_feature": {
            feat: {"total": s["total"], "outcomes": dict(s["outcomes"])}
            for feat, s in summary.items()
        },
        "coverage_by_outcome": dict(outcome_totals),
        "total_gaps": len(gaps),
        "top_priority_gaps": [
            {
                "priority_score": round(g["priority_score"], 2),
                "feature": g["feature"],
                "outcome": g["outcome"],
                "current_n": g["n_claims"],
                "shortfall": g["cells_shortfall"],
                "search_query": make_search_query(g["feature"], g["outcome"]),
            }
            for g in gaps[:30]
        ],
        "recommended_searches": searches,
    }
    with open("gap_analysis_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: gap_analysis_result.json")


if __name__ == "__main__":
    main()
