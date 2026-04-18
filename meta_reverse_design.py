"""meta_reverse_design.py - Outcome -> optimal design (Priority A).

Given a desired outcome (from CLAIMS), find the design (feature conjunction)
most predicted by the literature to PRODUCE that outcome.

This is the reverse direction of meta_optimal_design.py:
  forward: design -> score (across all outcomes)
  reverse: outcome -> best design (conditioned on that outcome)

Useful for answering: "I want to cause X. What should I build?"
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta_logic_claims import CLAIMS, all_features, all_outcomes
from meta_optimal_design import (
    enumerate_designs, conditions_match, format_design,
)


def design_score_for_outcome(design, target_outcome):
    """Score design's predicted likelihood of producing target_outcome.

    Filters CLAIMS to those with matching outcome, then counts supporting
    vs contradicting matches.
    """
    pos = neg = neu = 0
    matches = []
    for c in CLAIMS:
        if c["outcome"] != target_outcome:
            continue
        if conditions_match(c["condition"], design):
            matches.append(c["paper_id"])
            if c["strength"] > 0: pos += 1
            elif c["strength"] < 0: neg += 1
            else: neu += 1
    n = pos + neg + neu
    if n == 0:
        return 0.0, 0, 0, 0, []
    score = (pos - neg) / n
    return score, pos, neg, neu, matches


def rank_for_outcome(designs, target_outcome, min_n=1):
    """Return sorted list of (rank, n_matches, design, matches)."""
    ranked = []
    for d in designs:
        score, pos, neg, neu, matches = design_score_for_outcome(d, target_outcome)
        n = pos + neg + neu
        if n < min_n:
            continue
        # rank = score*10 + bonus for more matches (evidence strength)
        rank = score * 10 + n * 0.1
        ranked.append((rank, score, n, pos, neg, d, matches))
    ranked.sort(key=lambda t: -t[0])
    return ranked


def main():
    print("=" * 70)
    print("  REVERSE DESIGN: outcome -> best design")
    print("=" * 70)

    designs = enumerate_designs(max_features=3)
    print(f"\n  Candidate designs enumerated: {len(designs)}")

    all_outputs = {}
    for outcome in all_outcomes():
        ranked = rank_for_outcome(designs, outcome, min_n=1)
        print(f"\n{'=' * 70}")
        print(f"  TARGET OUTCOME: {outcome}")
        print(f"{'=' * 70}")

        # Count claims with this outcome
        n_claims_outcome = sum(1 for c in CLAIMS if c["outcome"] == outcome)
        if n_claims_outcome < 2:
            print(f"  [LOW EVIDENCE] Only {n_claims_outcome} claim(s) in dataset.")

        if not ranked:
            print(f"  No design with >= 1 match found.")
            all_outputs[outcome] = {
                "n_claims_in_dataset": n_claims_outcome,
                "top_designs": [],
            }
            continue

        top = ranked[:5]
        top_serialized = []
        for rank, score, n, pos, neg, d, matches in top:
            print(f"\n  score={score:+.2f}  n={n} (+{pos}/-{neg})")
            print(f"    design: {format_design(d)}")
            print(f"    sources: {matches[:4]}"
                  f"{'...' if len(matches) > 4 else ''}")
            top_serialized.append({
                "rank": round(rank, 3),
                "score": round(score, 3),
                "n_matches": n,
                "n_positive": pos,
                "n_negative": neg,
                "design": format_design(d),
                "supporting_papers": matches,
            })

        all_outputs[outcome] = {
            "n_claims_in_dataset": n_claims_outcome,
            "top_designs": top_serialized,
        }

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE: What design causes what")
    print("=" * 70)
    print(f"\n  {'Outcome':<16s} {'Top design (score, n)':<60s}")
    for outcome in all_outcomes():
        top = all_outputs[outcome]["top_designs"]
        if top:
            t = top[0]
            print(f"  {outcome:<16s} {t['design'][:45]:<46s}  "
                  f"({t['score']:+.2f}, n={t['n_matches']})")
        else:
            print(f"  {outcome:<16s} (no design meets min_n=1)")

    # Save
    with open("reverse_design_result.json", "w") as f:
        json.dump(all_outputs, f, indent=2, default=str)
    print(f"\n  Saved: reverse_design_result.json")


if __name__ == "__main__":
    main()
