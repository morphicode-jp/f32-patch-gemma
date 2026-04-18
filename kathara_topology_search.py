"""kathara_topology_search.py - Optimal circulant topology for Kathara brains.

Search over (n_nodes, skip_set) circulant graphs for optimal spectral +
structural properties. Current system uses Kathara(12, {1,4,6}); is there
a better choice?

Spectral criteria (from our hybrid_analysis finding that |lambda_2|
correlates with unseen generalization at r=+0.31-0.41):
  - lambda_1 = degree (connectedness)
  - lambda_2 = spectral gap (information mixing speed)
  - diameter = worst-case path (global integration)
  - clustering = local triangles (functional modularity)

We search n in {8, 12, 16, 20, 24} and skips of size 2-3.
"""
import os
import sys
import json
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_circulant_edges(n, skips):
    """Circulant graph C_n({skips}): node i connects to (i ± s) mod n."""
    edges = set()
    for i in range(n):
        for s in skips:
            j = (i + s) % n
            if i != j:
                key = tuple(sorted((i, j)))
                edges.add(key)
    return list(edges)


def adjacency(edges, n):
    A = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def spectral(A):
    """Return (|lambda_1|, |lambda_2|, all eigs).

    lambda_2 = second largest BY ABSOLUTE VALUE (matches
    hybrid_analysis.py convention - this is what correlates with
    generalization in our earlier experiment).
    """
    eigs = np.linalg.eigvalsh(A)
    eigs_abs_sorted = sorted(np.abs(eigs), reverse=True)
    lambda1 = float(eigs_abs_sorted[0])
    lambda2 = float(eigs_abs_sorted[1] if len(eigs_abs_sorted) > 1 else 0.0)
    return lambda1, lambda2, sorted(eigs, reverse=True)


def clustering_coefficient(A):
    """Global clustering coefficient: 3 * triangles / triples."""
    n = A.shape[0]
    A2 = A @ A
    A3 = A2 @ A
    triangles = np.trace(A3) / 6  # each triangle counted 6 times
    # Number of connected triples (v-u-w where u connects to both v, w)
    degrees = A.sum(axis=1)
    triples = sum(d * (d - 1) / 2 for d in degrees)
    if triples == 0:
        return 0.0
    return float(3 * triangles / triples)


def diameter(A):
    """Graph diameter via BFS (works for connected graphs)."""
    n = A.shape[0]
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if A[u, v] > 0 and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return float('inf')  # disconnected
        max_dist = max(max_dist, max(dist))
    return max_dist


def is_connected(A):
    """Check graph connectivity via BFS from node 0."""
    n = A.shape[0]
    visited = {0}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if A[u, v] > 0 and v not in visited:
                visited.add(v)
                queue.append(v)
    return len(visited) == n


def analyze_topology(n, skips):
    """Full analysis for a single (n, skips) topology."""
    edges = build_circulant_edges(n, skips)
    A = adjacency(edges, n)
    if not is_connected(A):
        return None
    lam1, lam2, eigs = spectral(A)
    cc = clustering_coefficient(A)
    diam = diameter(A)
    n_edges = len(edges)
    return {
        "n": n,
        "skips": list(skips),
        "n_edges": n_edges,
        "degree": int(A.sum(axis=1)[0]),  # regular graph, degree uniform
        "lambda_1": lam1,
        "lambda_2": lam2,
        "spectral_gap": lam1 - lam2,
        "clustering": cc,
        "diameter": diam,
        # Avg path length approximation via Wiener index
        "graph_edges": sorted([list(e) for e in edges]),
    }


def score_topology(props, target_lambda2=4.0):
    """Composite score favoring:
       - |lambda_2| close to target (our hybrid_analysis finding)
       - Low diameter (fast global mixing)
       - High clustering (modular structure)
       - Moderate degree (not too dense, not too sparse)
    """
    if props is None:
        return -1e9
    lam2_penalty = abs(props["lambda_2"] - target_lambda2)  # Gaussian-like
    degree_target = 5  # our Kathara is 5-regular
    degree_penalty = abs(props["degree"] - degree_target) * 0.2
    # Diameter: smaller = better (reach 2 is ideal for small graph)
    diam_bonus = 2.0 / (props["diameter"] + 0.5) if props["diameter"] != float('inf') else 0
    # Clustering: higher = better
    cc_bonus = props["clustering"] * 2.0
    score = (
        -lam2_penalty
        -degree_penalty
        +diam_bonus
        +cc_bonus
        +0.5  # baseline
    )
    return score


def search():
    print("=" * 70)
    print("  KATHARA TOPOLOGY SEARCH")
    print("=" * 70)

    results = []
    # Our current baseline
    baseline = analyze_topology(12, [1, 4, 6])
    if baseline:
        baseline["is_current"] = True
        baseline["score"] = score_topology(baseline)
        results.append(baseline)
        print(f"\n  Current: Kathara(12, {{1,4,6}}):")
        print(f"    lambda_1={baseline['lambda_1']:.3f}  lambda_2={baseline['lambda_2']:.3f}")
        print(f"    clustering={baseline['clustering']:.3f}  diameter={baseline['diameter']}")
        print(f"    score={baseline['score']:.3f}")

    # Search grid
    print(f"\n  Searching topologies...")
    search_space = []
    for n in [8, 10, 12, 16, 20, 24]:
        max_skip = n // 2
        for size in [2, 3]:
            for skips in itertools.combinations(range(1, max_skip + 1), size):
                search_space.append((n, list(skips)))

    print(f"  Total candidates: {len(search_space)}")

    for n, skips in search_space:
        props = analyze_topology(n, skips)
        if props is None:
            continue
        props["score"] = score_topology(props)
        props["is_current"] = (n == 12 and set(skips) == {1, 4, 6})
        if props["is_current"] and results[0]["score"] == props["score"]:
            continue  # already added as baseline
        results.append(props)

    # Sort by score
    results.sort(key=lambda r: -r["score"])
    return results


def main():
    results = search()

    # Top-20
    print(f"\n  {'rank':<5s} {'n':<4s} {'skips':<18s} {'deg':<4s} "
          f"{'|λ2|':<8s} {'cc':<7s} {'diam':<5s} {'score':<7s}")
    print("  " + "-" * 70)
    for i, r in enumerate(results[:20]):
        marker = " <-- current" if r.get("is_current") else ""
        print(f"  {i+1:<5d} {r['n']:<4d} {str(r['skips']):<18s} "
              f"{r['degree']:<4d} {r['lambda_2']:<8.3f} "
              f"{r['clustering']:<7.3f} {r['diameter']:<5d} "
              f"{r['score']:<+7.3f}{marker}")

    # Find current rank
    current_rank = next(
        (i for i, r in enumerate(results) if r.get("is_current")),
        None
    )
    if current_rank is not None:
        print(f"\n  Our current Kathara(12, {{1,4,6}}) ranks: "
              f"#{current_rank + 1} of {len(results)}")

    # Best alternative
    non_current = [r for r in results if not r.get("is_current")]
    if non_current:
        best = non_current[0]
        print(f"\n  Best alternative: Kathara({best['n']}, {best['skips']})")
        print(f"    lambda_2={best['lambda_2']:.3f} (target=4.0)")
        print(f"    clustering={best['clustering']:.3f}")
        print(f"    diameter={best['diameter']}")
        print(f"    degree={best['degree']}")

        if current_rank is not None:
            improvement = best["score"] - results[current_rank]["score"]
            print(f"    improvement over current: {improvement:+.3f}")

    # Save
    out = {
        "n_candidates": len(results),
        "current_kathara_rank": current_rank + 1 if current_rank is not None else None,
        "top_20": [
            {k: v for k, v in r.items() if k != "graph_edges"}
            for r in results[:20]
        ],
        "recommendation": (
            {
                "n": results[0]["n"],
                "skips": results[0]["skips"],
                "score": results[0]["score"],
                "reasoning": (
                    "topology with best composite score (|lambda_2| close to 4, "
                    "small diameter, high clustering)"
                ),
            }
            if results else None
        ),
    }
    with open("topology_search_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: topology_search_result.json")


if __name__ == "__main__":
    main()
