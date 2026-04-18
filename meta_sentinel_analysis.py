"""meta_sentinel_analysis.py - Sentinel-powered meta-analysis of signaling literature.

Applies MirrorScan + owl()'s structure-finding to a curated dataset of
emergent communication papers. Output:

  1. Attribute importance (which design choices correlate with emergence)
  2. Dead dims (which attributes don't matter)
  3. Best untried combinations (suggested regions of unexplored design space)
  4. Positioning our own experiments within the literature landscape
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta_research_papers import PAPERS, ATTRIBUTES, to_matrix


def correlation_analysis(X, y, attr_names):
    """Per-attribute correlation with emergence."""
    results = []
    for i, name in enumerate(attr_names):
        xi = X[:, i]
        if xi.std() < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(xi, y)[0, 1])
        results.append((name, r, float(xi.mean()), float(xi.std())))
    return sorted(results, key=lambda t: abs(t[1]), reverse=True)


def build_measurements(X, y):
    """Convert to owl() measurement format."""
    measurements = []
    for i in range(len(X)):
        measurements.append({
            "params": X[i].tolist(),
            "score": float(y[i]) * 100,  # owl prefers wider score range
        })
    return measurements


def attribute_ranges(X):
    """Returns per-attribute (lo, hi) based on data extent."""
    return [(float(X[:, i].min()), float(X[:, i].max()))
            for i in range(X.shape[1])]


def analyze_with_owl():
    """Use owl() structure analysis on paper dataset."""
    from twelve.optimize import owl

    X, y, ids = to_matrix()
    measurements = build_measurements(X, y)
    param_ranges = attribute_ranges(X)

    print("=" * 70)
    print("  owl() meta-analysis of emergent communication literature")
    print(f"  N papers: {len(X)}  |  Attributes: {len(ATTRIBUTES)}")
    print("=" * 70)

    # Run owl WITHOUT verify_fn (pure structure analysis from data)
    result = owl(
        measurements=measurements,
        param_ranges=param_ranges,
        param_names=ATTRIBUTES,
        autonomous=False,
        time_budget=30,
        verbose=False,
    )

    print(f"\n  proxy R² = {result.get('proxy_r2', 0):.3f}")
    print(f"  proxy type = {result.get('proxy_type')}")
    print(f"  confidence = {result.get('confidence')}")

    # Importance
    importance = result.get("importance", [])
    if importance:
        print(f"\n  Attribute importance (higher = correlates with emergence):")
        ranked = sorted(zip(ATTRIBUTES, importance), key=lambda t: t[1], reverse=True)
        for name, imp in ranked:
            bar = "█" * int(imp * 40)
            print(f"    {name:<18s} {imp:.4f} {bar}")

    # Dead dims
    dead_dims = result.get("dead_dims", [])
    if dead_dims is not None and len(dead_dims) > 0:
        print(f"\n  Dead dims (owl says these don't predict emergence):")
        for d in dead_dims:
            if isinstance(d, (int, np.integer)) and 0 <= int(d) < len(ATTRIBUTES):
                print(f"    - {ATTRIBUTES[int(d)]}")

    # Fragility
    fragility = result.get("fragility", [])
    if fragility:
        print(f"\n  Fragility (important AND isolated = most fragile):")
        ranked = sorted(zip(ATTRIBUTES, fragility), key=lambda t: t[1], reverse=True)
        for name, f in ranked[:5]:
            print(f"    {name:<18s} {f:.4f}")

    # best_params
    bp = result.get("best_params")
    if bp is not None:
        if isinstance(bp, dict):
            bp = [bp.get(a) for a in ATTRIBUTES]
        print(f"\n  owl's 'optimal' region (emergence-maximizing):")
        for a, v in zip(ATTRIBUTES, bp):
            if v is not None:
                print(f"    {a:<18s} = {v:.2f}")

    return result, X, y, ids


def pearson_table(X, y, attr_names):
    """Simple per-attribute correlation summary."""
    print("\n" + "=" * 70)
    print("  Pearson correlations with emergence (naive baseline)")
    print("=" * 70)
    corrs = correlation_analysis(X, y, attr_names)
    for name, r, mean, std in corrs:
        direction = "↑" if r > 0 else "↓" if r < 0 else " "
        print(f"    {name:<18s}  r = {r:+.3f} {direction}   "
              f"(range {mean - std:.2f}..{mean + std:.2f})")
    return corrs


def categorical_breakdown(attr_idx, attr_name):
    """For categorical attributes, show emergence rate per category."""
    X, y, ids = to_matrix()
    from collections import defaultdict
    by_cat = defaultdict(list)
    cat_ids = defaultdict(list)
    for i in range(len(X)):
        c = int(X[i, attr_idx])
        by_cat[c].append(y[i])
        cat_ids[c].append(ids[i])
    print(f"\n  Emergence rate by {attr_name}:")
    for c in sorted(by_cat.keys()):
        vals = by_cat[c]
        mean = float(np.mean(vals))
        print(f"    {attr_name}={c}: n={len(vals):2d}  "
              f"mean_emergence={mean:.2f}  "
              f"papers={cat_ids[c][:2]}{'...' if len(cat_ids[c]) > 2 else ''}")


def find_gap_configurations(X, attr_names, n_to_find=5):
    """Find unexplored configurations (product of observed unique values)."""
    from itertools import product
    unique_vals = [sorted(set(X[:, i].tolist())) for i in range(X.shape[1])]
    observed = set(tuple(row) for row in X.tolist())
    gaps = []
    # Check product space (but bound it)
    all_configs = list(product(*unique_vals))
    if len(all_configs) > 20000:
        print(f"\n  (product space = {len(all_configs)} — sampling 20000)")
        rng = np.random.RandomState(7)
        idx = rng.choice(len(all_configs), size=20000, replace=False)
        all_configs = [all_configs[i] for i in idx]
    for conf in all_configs:
        if conf not in observed:
            gaps.append(conf)
    print(f"\n  Total unexplored configs (from observed value combinations): "
          f"{len(gaps)}/{len(all_configs)}")
    # Score each gap by Pearson-based prediction
    X_arr, y_arr, _ = to_matrix()
    weights = []
    for i in range(X_arr.shape[1]):
        xi = X_arr[:, i]
        if xi.std() < 1e-9:
            weights.append(0.0)
        else:
            weights.append(float(np.corrcoef(xi, y_arr)[0, 1]))
    predictions = []
    for conf in gaps:
        pred = sum(w * v for w, v in zip(weights, conf))
        predictions.append((pred, conf))
    predictions.sort(reverse=True)
    print(f"\n  Top unexplored configurations by linear prediction:")
    for pred, conf in predictions[:n_to_find]:
        desc = ", ".join(f"{a}={v}" for a, v in zip(attr_names, conf))
        print(f"    pred={pred:+.2f}  |  {desc}")


def position_our_work(X, y, ids):
    """Show where our own experiments fall in the feature space."""
    print("\n" + "=" * 70)
    print("  Our experiments in the literature landscape")
    print("=" * 70)
    our_indices = [i for i, pid in enumerate(ids) if pid.startswith("our_")]
    for i in our_indices:
        print(f"\n  [{ids[i]}] emergence={y[i]}")
        for j, a in enumerate(ATTRIBUTES):
            col = X[:, j]
            percentile = float((col < X[i, j]).sum() / max(len(col), 1))
            marker = "" if 0.3 <= percentile <= 0.7 else (
                " (low)" if percentile < 0.3 else " (high)")
            print(f"    {a:<18s} = {X[i, j]:>5.2f}   "
                  f"pct={percentile * 100:3.0f}%{marker}")


def main():
    print("\n" + "#" * 70)
    print("  META-RESEARCH: Sentinel-powered literature analysis")
    print("#" * 70)

    X, y, ids = to_matrix()

    # Step 1: naive correlations
    corrs = pearson_table(X, y, ATTRIBUTES)

    # Step 2: owl() structure analysis
    print("\n")
    owl_result, X, y, ids = analyze_with_owl()

    # Step 3: categorical breakdowns for key dims
    print("\n" + "=" * 70)
    print("  Per-category emergence rates")
    print("=" * 70)
    for attr_idx, attr_name in [(5, "reward"), (3, "learning"),
                                 (2, "arch"), (7, "has_embodiment")]:
        categorical_breakdown(attr_idx, attr_name)

    # Step 4: positioning
    position_our_work(X, y, ids)

    # Step 5: find unexplored configs
    print("\n" + "=" * 70)
    print("  Gap analysis: unexplored design configurations")
    print("=" * 70)
    find_gap_configurations(X, ATTRIBUTES)

    # Save
    summary = {
        "n_papers": len(X),
        "n_attributes": len(ATTRIBUTES),
        "pearson_correlations": [
            {"attr": name, "r": r, "mean": m, "std": s}
            for name, r, m, s in corrs
        ],
        "owl_importance": list(zip(ATTRIBUTES, owl_result.get("importance", []))),
        "owl_dead_dims": [ATTRIBUTES[int(d)] for d in owl_result.get("dead_dims") or []
                          if isinstance(d, (int, np.integer)) and 0 <= int(d) < len(ATTRIBUTES)],
        "owl_proxy_r2": owl_result.get("proxy_r2"),
        "owl_confidence": owl_result.get("confidence"),
        "owl_best_params": list(zip(ATTRIBUTES, owl_result.get("best_params", []) or []))
    }
    with open("meta_research_result.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n  Saved: meta_research_result.json")


if __name__ == "__main__":
    main()
