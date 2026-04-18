"""meta_logic_analysis.py - Analyze logic-as-data claim propositions.

Proper "logic as data" analysis:
  1. Each paper's claim = propositional rule (condition => outcome, strength)
  2. Count which conditions SUPPORT each outcome (votes)
  3. Detect CONTRADICTIONS (same conditions, different outcomes)
  4. Find UNSUPPORTED conjunctions (logical gap)
  5. Feature pairs that co-occur → joint predictive power
  6. Use owl/Sentinel on the claim-feature space (not paper metadata)
"""
import os
import sys
import json
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta_logic_claims import CLAIMS, all_features, all_outcomes


def outcome_support_by_condition(claims):
    """For each (condition, outcome) pair, count support votes.

    Returns dict:  feature -> outcome -> (pos, neg, neu)
    """
    support = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for c in claims:
        for (feat, op, val) in c["condition"]:
            key = f"{feat}={val}"
            idx = 0 if c["strength"] > 0 else (1 if c["strength"] < 0 else 2)
            support[key][c["outcome"]][idx] += 1
    return support


def contradictions(claims):
    """Find condition-sets that have opposing outcomes across papers."""
    buckets = defaultdict(list)
    for c in claims:
        cond_key = tuple(sorted((f, str(v)) for f, _, v in c["condition"]))
        buckets[cond_key].append(c)
    contras = []
    for cond, group in buckets.items():
        if len(group) < 2:
            continue
        strengths = [g["strength"] for g in group]
        if min(strengths) * max(strengths) < 0:
            contras.append((cond, group))
    return contras


def outcome_by_feature_rate(claims):
    """For each feature-value, success rate per outcome.

    rate = (pos - 0.5*neutral) / total  (punishes negatives hard)
    """
    rates = defaultdict(dict)
    feat_outcomes = defaultdict(lambda: defaultdict(list))
    for c in claims:
        for (feat, _, val) in c["condition"]:
            feat_outcomes[f"{feat}={val}"][c["outcome"]].append(c["strength"])
    for key, outcomes in feat_outcomes.items():
        for outcome, strengths in outcomes.items():
            n = len(strengths)
            pos = sum(1 for s in strengths if s > 0)
            neg = sum(1 for s in strengths if s < 0)
            rates[key][outcome] = {
                "n": n, "pos": pos, "neg": neg,
                "net_rate": (pos - neg) / n if n else 0.0,
            }
    return rates


def feature_cooccurrence(claims):
    """Feature pair co-occurrence counts."""
    pairs = Counter()
    for c in claims:
        feats = tuple(sorted(f"{f}={v}" for f, _, v in c["condition"]))
        for i in range(len(feats)):
            for j in range(i+1, len(feats)):
                pairs[(feats[i], feats[j])] += 1
    return pairs.most_common(15)


def paper_logic_matrix(claims):
    """Build numeric matrix: row=claim, col=feature (1=condition present, 0=absent).

    Outcome encoded as integer category.
    Strength as real value.
    """
    feats = all_features()
    outcomes = all_outcomes()
    feat_idx = {f: i for i, f in enumerate(feats)}
    out_idx = {o: i for i, o in enumerate(outcomes)}

    X = np.zeros((len(claims), len(feats)))
    y_outcome = np.zeros(len(claims), dtype=np.int32)
    y_strength = np.zeros(len(claims))
    for i, c in enumerate(claims):
        for (f, _, v) in c["condition"]:
            # Encode as +1 if True, -1 if False, 0 if numeric
            if isinstance(v, bool):
                X[i, feat_idx[f]] = 1.0 if v else -1.0
            elif isinstance(v, (int, float)):
                X[i, feat_idx[f]] = float(v) / 10.0  # normalize
            else:
                X[i, feat_idx[f]] = 1.0
        y_outcome[i] = out_idx[c["outcome"]]
        y_strength[i] = c["strength"]
    return X, y_outcome, y_strength, feats, outcomes


def run_owl_on_logic(claims):
    """Apply owl structure analysis to the claim-feature -> strength mapping."""
    from twelve.optimize import owl
    X, y_outcome, y_strength, feats, outcomes = paper_logic_matrix(claims)

    # Use strength as score
    measurements = []
    for i in range(len(X)):
        measurements.append({
            "params": X[i].tolist(),
            "score": float(y_strength[i] * 100),  # -100..+100
        })

    # Conservative ranges: -1 to 1 (features are trinary)
    param_ranges = [(-1.0, 1.0) for _ in feats]

    result = owl(
        measurements=measurements,
        param_ranges=param_ranges,
        param_names=feats,
        autonomous=False,
        time_budget=20,
        verbose=False,
    )
    return result, feats


def main():
    print("=" * 70)
    print("  LOGIC-AS-DATA ANALYSIS: Claim propositions")
    print(f"  N claims = {len(CLAIMS)}")
    print("=" * 70)

    # --- 1. Overall outcome support ---
    print("\n1. Outcome support by condition")
    print("-" * 70)
    rates = outcome_by_feature_rate(CLAIMS)
    # Top features supporting "emerges"
    emerge_rates = []
    for key, outcomes in rates.items():
        if "emerges" in outcomes:
            d = outcomes["emerges"]
            if d["n"] >= 2:
                emerge_rates.append((key, d["net_rate"], d["n"], d["pos"], d["neg"]))
    emerge_rates.sort(key=lambda t: -t[1])
    print(f"\n  Conditions supporting 'emerges' (n>=2 claims):")
    for key, rate, n, pos, neg in emerge_rates[:15]:
        bar = "#" * int(max(0, rate) * 20)
        print(f"    {key:<40s}  net={rate:+.2f} (n={n}, +{pos}/-{neg}) {bar}")

    # Same for fails
    fail_rates = []
    for key, outcomes in rates.items():
        if "fails" in outcomes:
            d = outcomes["fails"]
            if d["n"] >= 1:
                fail_rates.append((key, d["n"], d["pos"]))
    fail_rates.sort(key=lambda t: -t[1])
    if fail_rates:
        print(f"\n  Conditions linked to 'fails':")
        for key, n, pos in fail_rates[:10]:
            print(f"    {key:<40s}  n={n}, {pos} claim failure")

    # --- 2. Contradictions ---
    print("\n\n2. Logical contradictions (same condition-set, different outcomes/strengths)")
    print("-" * 70)
    contras = contradictions(CLAIMS)
    if contras:
        for cond, group in contras[:5]:
            print(f"\n  Condition: {cond}")
            for g in group:
                print(f"    paper={g['paper_id']}  outcome={g['outcome']}  "
                      f"strength={g['strength']}")
    else:
        print("  None detected (each condition-set has consistent outcome)")

    # --- 3. Feature co-occurrence ---
    print("\n\n3. Most common feature-pair co-occurrences in claims")
    print("-" * 70)
    for (f1, f2), n in feature_cooccurrence(CLAIMS):
        print(f"    {f1:<30s} & {f2:<30s}  n={n}")

    # --- 4. owl structure analysis on logic matrix ---
    print("\n\n4. owl() structural analysis on claim-feature space")
    print("-" * 70)
    owl_result, feats = run_owl_on_logic(CLAIMS)
    print(f"  R^2 = {owl_result.get('proxy_r2', 0):.3f}")
    print(f"  confidence = {owl_result.get('confidence')}")

    importance = owl_result.get("importance")
    if importance:
        ranked = sorted(zip(feats, importance), key=lambda t: t[1], reverse=True)
        print(f"\n  Feature importance for CLAIM STRENGTH:")
        for name, imp in ranked[:10]:
            bar = "#" * int(max(0, imp) * 40)
            print(f"    {name:<30s} {imp:.3f} {bar}")

    dead = owl_result.get("dead_dims") or []
    if dead:
        print(f"\n  Dead conditions (owl says these don't predict claim strength):")
        for d in dead:
            if isinstance(d, (int, np.integer)) and 0 <= int(d) < len(feats):
                print(f"    - {feats[int(d)]}")

    # --- 5. Predictive rules extraction ---
    print("\n\n5. Strongest extractable IF-THEN rules")
    print("-" * 70)
    # For each feature, compute P(emerges | feat=True) vs baseline
    baseline_pos_rate = sum(1 for c in CLAIMS if c["strength"] > 0) / len(CLAIMS)
    rules = []
    feat_counts = defaultdict(lambda: {"total": 0, "pos": 0, "neg": 0})
    for c in CLAIMS:
        for (f, _, v) in c["condition"]:
            key = f"{f}={v}"
            feat_counts[key]["total"] += 1
            if c["strength"] > 0: feat_counts[key]["pos"] += 1
            elif c["strength"] < 0: feat_counts[key]["neg"] += 1
    for key, d in feat_counts.items():
        if d["total"] >= 2:
            rate = d["pos"] / d["total"]
            lift = rate - baseline_pos_rate
            rules.append((key, rate, lift, d["total"]))
    rules.sort(key=lambda t: -t[2])  # sort by lift
    print(f"  Baseline positive claim rate: {baseline_pos_rate:.2f}")
    print(f"\n  Top lift conditions (rule confidence = P(strength>0 | condition)):")
    for key, rate, lift, n in rules[:10]:
        direction = "+" if lift > 0 else "-"
        print(f"    IF {key:<35s} THEN pos_claim  rate={rate:.2f}  "
              f"lift={lift:+.2f} {direction}  (n={n})")

    # --- 6. Save ---
    out = {
        "n_claims": len(CLAIMS),
        "n_unique_features": len(all_features()),
        "outcomes_seen": all_outcomes(),
        "top_conditions_for_emergence": [
            {"condition": key, "net_rate": rate, "n": n, "pos": pos, "neg": neg}
            for key, rate, n, pos, neg in emerge_rates[:15]
        ],
        "contradictions": [
            {"conditions": list(cond),
             "papers": [g["paper_id"] for g in group],
             "strengths": [g["strength"] for g in group]}
            for cond, group in contras
        ],
        "top_rules": [
            {"condition": key, "rate": rate, "lift": lift, "n": n}
            for key, rate, lift, n in rules[:20]
        ],
        "owl_r2": owl_result.get("proxy_r2"),
        "owl_importance": dict(zip(feats, owl_result.get("importance") or [])),
    }
    with open("meta_logic_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: meta_logic_result.json")


if __name__ == "__main__":
    main()
