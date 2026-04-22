"""iss_compute.py — Intelligence Structure Score for Tamashii shell graphs.

Computes the 5 metrics from paper_brain_structure_intelligence.md §2.3:
  λ₂ (algebraic connectivity)
  C  (clustering coefficient)
  L  (average path length)
  H  (hierarchy depth)
  I  (inhibition ratio)
  hub (top-10% concentration)

And derives ISS v2 (paper §5.1, uncapped for super-intelligence).

The "nodes" in Tamashii's ISS graph are SHELLS (not individual neurons,
which are inside shells). Edges are shell-to-shell signal dependencies:
  edge A→B exists iff shell A writes to S slots that shell B reads.
Weight = activity magnitude (delta norm) during operation.

Usage:
    from iss_compute import compute_iss, measure_tamashii_graph
    agent = build_fluctlight(...)
    graph = measure_tamashii_graph(agent, n_warmup_steps=200)
    metrics = compute_iss(graph)
    print(f"ISS v2 = {metrics['iss_v2']:.1f}")
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


# Human brain reference values (from paper Table in §3.1)
HUMAN_REF = {
    "H": 6.0,
    "L": 2.14,
    "C": 0.283,
    "I": 0.20,
    "hub": 12.4,  # % top-10%
}


# ============================================================================
# Shell graph extraction
# ============================================================================

def extract_shell_read_write_sets(shell) -> dict:
    """Probe a shell by introspection to find which S slots it reads/writes.

    Returns {'reads': set[int], 'writes': set[int], 'inhibitory': bool}.

    Uses multiple strategies:
      1. Explicit slot attributes (sensor_slot, motor_nav, firing_slot, etc.)
      2. Active probing: run step() with known S, see which indices change
      3. Name heuristics for inhibitory designation
    """
    reads = set()
    writes = set()

    # Strategy 1: attribute introspection
    for attr in dir(shell):
        if attr.startswith("_"):
            continue
        val = getattr(shell, attr, None)
        if isinstance(val, slice):
            rng = range(val.start or 0, val.stop or 0)
            if "firing" in attr or "motor" in attr or "write" in attr or "output" in attr:
                writes.update(rng)
            else:  # most slice names like sensor_slot, observe_slot are reads
                reads.update(rng)
        elif isinstance(val, int) and 0 <= val < 192:
            if "motor" in attr or "write" in attr or "output" in attr:
                writes.add(val)
            elif "sensor" in attr or "peer" in attr or "read" in attr or "olf" in attr:
                reads.add(val)

    # Specific common mapping for Tamashii shells
    name = getattr(shell, "name", "").lower()
    if "core_brain" in name:
        reads.update(range(0, 16))       # sensors
        writes.update(range(16, 19))      # nav, speed, voice
        writes.update(range(19, 35))      # firing pattern
    elif "brainstem" in name:
        reads.update(range(0, 192))       # reads all (homeostasis)
        writes.update(range(0, 192))      # can correct all (range clipping)
    elif "cerebellum" in name:
        reads.update(range(16, 19))       # motor (to predict)
        writes.update(range(35, 100))     # prediction workspace
    elif "salience" in name:
        reads.update(range(0, 35))        # sensors + motor + firing
        writes.update(range(100, 120))    # attention signal
    elif "hippocampus" in name:
        reads.update(range(0, 192))       # can snapshot anything
        writes.update(range(120, 160))    # episode slots
    elif "prefrontal" in name:
        reads.update(range(0, 192))       # integrates everything
        writes.update(range(160, 180))    # goal slots
    elif "dmn" in name:
        reads.update(range(0, 192))
        writes.update(range(180, 192))
    elif "taboo" in name:
        reads.update([0, 2, 10, 11, 12, 13, 14, 16, 17, 18])
        writes.update([16, 17, 18, 190])  # motor + violation slot

    # Inhibitory determination
    inhibitory = ("taboo" in name or "inhib" in name or
                   getattr(shell, "shell_sign", 1) == -1)

    return {"reads": reads, "writes": writes, "inhibitory": inhibitory,
            "name": name}


def build_shell_graph(agent,
                       activity_weights: dict | None = None) -> dict:
    """Build the adjacency structure of shells in a Tamashii agent.

    Returns:
      {
        "nodes": list[str],            # shell names
        "adj": ndarray (N, N),         # adjacency weights
        "layer": dict[name→int],       # if shells have .layer attribute
        "inhibitory": list[bool],      # per node
      }

    activity_weights: optional {shell_name: delta_norm} to weight edges by
      actual activity observed during agent run.
    """
    shells = agent.shells
    N = len(shells)
    names = [s.name for s in shells]
    slots = [extract_shell_read_write_sets(s) for s in shells]
    layers = {}
    for s in shells:
        if hasattr(s, "layer"):
            layers[s.name] = int(s.layer)

    # Adjacency: edge A→B if slots[A]['writes'] ∩ slots[B]['reads'] ≠ ∅
    adj = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            overlap = slots[i]["writes"] & slots[j]["reads"]
            if overlap:
                # Weight = fraction of slot overlap (crude measure of signal bandwidth)
                w = len(overlap)
                # Modulate by observed activity if available
                if activity_weights:
                    a = activity_weights.get(names[i], 1.0) or 0.01
                    w = w * (a ** 0.5)  # sqrt to moderate
                adj[i, j] = float(w)

    # Normalize adjacency to [0, 1] range (relative connection strength)
    if adj.max() > 0:
        adj = adj / adj.max()

    inhib = [s["inhibitory"] for s in slots]

    return {
        "nodes": names,
        "adj": adj,
        "layers": layers,
        "inhibitory": inhib,
    }


def measure_tamashii_graph(agent, n_warmup_steps: int = 100,
                             sensor_fn: Callable | None = None) -> dict:
    """Build graph with edge weights based on observed activity.

    Runs the agent for n_warmup_steps, collects per-shell delta norms,
    then builds graph with activity-weighted edges.
    """
    if sensor_fn is None:
        def sensor_fn(step):
            s = np.zeros(16)
            s[0] = 0.5 + 0.3 * np.sin(step * 0.1)
            s[10] = 0.1 + 0.4 * np.cos(step * 0.13)
            return s

    # Collect activity
    activity = {s.name: [] for s in agent.shells}
    for step in range(n_warmup_steps):
        with agent._lock:
            agent.S[0:16] = sensor_fn(step).astype(np.float64)
        agent.tick_once()
        for name, dn in agent.shell_delta_norms().items():
            activity[name].append(dn)
    mean_activity = {n: float(np.mean(v)) if v else 0.0
                     for n, v in activity.items()}

    graph = build_shell_graph(agent, activity_weights=mean_activity)
    graph["activity"] = mean_activity
    return graph


# ============================================================================
# Graph metric computations
# ============================================================================

def algebraic_connectivity(adj: np.ndarray) -> float:
    """λ₂ = second smallest eigenvalue of graph Laplacian L = D - A.

    Symmetrize undirected for Laplacian. Higher = faster propagation.
    """
    n = adj.shape[0]
    A = (adj + adj.T) * 0.5  # symmetrize
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    if n >= 2:
        return float(eigs[1])
    return 0.0


def clustering_coefficient(adj: np.ndarray) -> float:
    """Average clustering coefficient (Watts-Strogatz).
    For each node, fraction of its neighbors that are also connected.
    """
    n = adj.shape[0]
    A = ((adj > 0) | (adj.T > 0)).astype(np.float64)
    np.fill_diagonal(A, 0)
    cs = []
    for i in range(n):
        nbrs = np.nonzero(A[i])[0]
        k = len(nbrs)
        if k < 2:
            cs.append(0.0)
            continue
        # Count triangles: edges among neighbors
        sub = A[np.ix_(nbrs, nbrs)]
        tri = sub.sum() / 2.0  # undirected triangle edges
        max_pairs = k * (k - 1) / 2.0
        cs.append(tri / max_pairs if max_pairs > 0 else 0.0)
    return float(np.mean(cs))


def avg_path_length(adj: np.ndarray) -> float:
    """Average shortest path length (unweighted, undirected).

    Uses BFS. Returns mean over all reachable pairs.
    """
    n = adj.shape[0]
    A = ((adj > 0) | (adj.T > 0)).astype(bool)
    np.fill_diagonal(A, False)
    dists = []
    for src in range(n):
        # BFS
        d = [-1] * n
        d[src] = 0
        queue = [src]
        while queue:
            next_q = []
            for u in queue:
                for v in range(n):
                    if A[u, v] and d[v] == -1:
                        d[v] = d[u] + 1
                        next_q.append(v)
            queue = next_q
        for v in range(n):
            if v != src and d[v] > 0:
                dists.append(d[v])
    if not dists:
        return float("inf")
    return float(np.mean(dists))


def hierarchy_depth(graph: dict) -> int:
    """Hierarchy depth H.

    Strategy:
      1. If shells have explicit .layer attribute, H = max - min + 1
      2. Else: longest chain via topological sort of DAG-like edges

    Returns the effective layer count.
    """
    layers = graph.get("layers", {})
    if layers and len(layers) == len(graph["nodes"]):
        vals = list(layers.values())
        return int(max(vals) - min(vals) + 1)

    # Fallback: strongly-connected-component-based layering
    # Compute longest path in DAG-like skeleton (ignore cycles)
    adj = graph["adj"]
    n = adj.shape[0]
    A = (adj > 0).astype(np.int32)  # directed
    # Remove self-loops
    np.fill_diagonal(A, 0)
    # Longest path via DFS with memoization
    memo = {}

    def longest(u, visiting):
        if u in memo:
            return memo[u]
        if u in visiting:
            return 0  # cycle
        visiting.add(u)
        best = 0
        for v in range(n):
            if A[u, v]:
                best = max(best, 1 + longest(v, visiting))
        visiting.discard(u)
        memo[u] = best
        return best

    max_depth = 0
    for start in range(n):
        max_depth = max(max_depth, longest(start, set()))
    return int(max_depth) + 1  # +1 because layers are nodes


def inhibition_ratio(graph: dict) -> float:
    """Fraction of edges that are inhibitory (by source node designation)."""
    adj = graph["adj"]
    inhib = graph["inhibitory"]
    total_edge_weight = 0.0
    inhib_edge_weight = 0.0
    n = adj.shape[0]
    for i in range(n):
        out_w = adj[i].sum()
        total_edge_weight += out_w
        if inhib[i]:
            inhib_edge_weight += out_w
    if total_edge_weight <= 0:
        return 0.0
    return float(inhib_edge_weight / total_edge_weight)


def hub_concentration(adj: np.ndarray, top_frac: float = 0.10) -> float:
    """Fraction of total edge endpoints owned by top-X% degree nodes.

    paper §3.1: Human hub_top10% = 12.4%
    """
    n = adj.shape[0]
    A = (adj > 0).astype(np.float64) + (adj.T > 0).astype(np.float64)
    np.fill_diagonal(A, 0)
    degrees = A.sum(axis=1)
    if degrees.sum() <= 0:
        return 0.0
    k = max(1, int(n * top_frac))
    top_idx = np.argsort(-degrees)[:k]
    top_sum = degrees[top_idx].sum()
    return float(100.0 * top_sum / degrees.sum())


# ============================================================================
# ISS v2 (paper §5.1, uncapped)
# ============================================================================

def iss_v2(H: float, L: float, C: float, I: float, hub: float) -> float:
    """ISS v2 — uncapped, suitable for super-intelligence exploration.

    Formula from paper §5.1:
      s_H   = (H / 6.0) × 100                            -- linear
      s_L   = (2.14 / max(L, 0.01)) × 100                -- inverse (shorter=better)
      s_C   = exp(-((C - 0.283)² / (2 × 0.15²))) × 100   -- Gaussian sweet spot
      s_I   = exp(-((I - 0.20)²  / (2 × 0.10²))) × 100   -- Gaussian sweet spot
      s_hub = exp(-((hub - 12.4)² / (2 × 8.0²))) × 100   -- Gaussian sweet spot
      ISS_v2 = (s_H × s_L × s_C × s_I × s_hub)^(1/5)
    """
    s_H = (H / 6.0) * 100.0
    s_L = (2.14 / max(L, 0.01)) * 100.0
    s_C = math.exp(-((C - 0.283) ** 2 / (2 * 0.15 ** 2))) * 100.0
    s_I = math.exp(-((I - 0.20) ** 2 / (2 * 0.10 ** 2))) * 100.0
    s_hub = math.exp(-((hub - 12.4) ** 2 / (2 * 8.0 ** 2))) * 100.0

    prod = max(s_H * s_L * s_C * s_I * s_hub, 1e-10)
    return float(prod ** (1.0 / 5.0))


def iss_v1(H: float, L: float, C: float, I: float, hub: float) -> float:
    """ISS v1 — capped at 100 (human brain = 100 by construction).

    paper §2.4:
      S_hierarchy = 1 - |H - 6| / 6
      S_path      = 1 - |L - 2.14| / 10
      S_cluster   = 1 - |C - 0.283| / 1
      S_inhibit   = 1 - |I - 0.20| / 1
      S_hub       = 1 - |hub - 12.4| / 100
      ISS_v1      = sqrt(product) × 100
    """
    s_H = max(0.0, 1.0 - abs(H - 6.0) / 6.0)
    s_L = max(0.0, 1.0 - abs(L - 2.14) / 10.0)
    s_C = max(0.0, 1.0 - abs(C - 0.283) / 1.0)
    s_I = max(0.0, 1.0 - abs(I - 0.20) / 1.0)
    s_hub = max(0.0, 1.0 - abs(hub - 12.4) / 100.0)

    prod = max(s_H * s_L * s_C * s_I * s_hub, 0.0)
    return float(math.sqrt(prod) * 100.0)


def compute_iss(graph: dict) -> dict:
    """Full metric extraction + ISS v1 and v2."""
    adj = graph["adj"]
    H = hierarchy_depth(graph)
    L = avg_path_length(adj)
    C = clustering_coefficient(adj)
    I = inhibition_ratio(graph)
    hub = hub_concentration(adj)
    lam2 = algebraic_connectivity(adj)

    iss_1 = iss_v1(H, L, C, I, hub)
    iss_2 = iss_v2(H, L, C, I, hub)

    return {
        "H": float(H),
        "L": float(L),
        "C": float(C),
        "I": float(I),
        "hub": float(hub),
        "lambda_2": float(lam2),
        "iss_v1": float(iss_1),
        "iss_v2": float(iss_2),
        "n_nodes": int(adj.shape[0]),
        "n_edges": int((adj > 0).sum()),
        "human_ref": dict(HUMAN_REF),
    }


# ============================================================================
# Reporting
# ============================================================================

def print_iss_report(graph: dict, metrics: dict, label: str = "Tamashii"):
    """Pretty-print diagnostic comparing to human brain reference."""
    print(f"\n{'=' * 72}")
    print(f"  ISS DIAGNOSTIC: {label}")
    print(f"{'=' * 72}")
    print(f"  Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"  {'metric':15s} {'value':>10s} {'human ref':>12s} {'distance':>12s}")
    for key in ["H", "L", "C", "I", "hub"]:
        val = metrics[key]
        ref = HUMAN_REF[key]
        dist = abs(val - ref)
        print(f"  {key:15s} {val:>10.3f} {ref:>12.3f} {dist:>12.3f}")
    print(f"  {'lambda_2':15s} {metrics['lambda_2']:>10.3f}")
    print(f"\n  ISS v1 (capped): {metrics['iss_v1']:>8.1f} / 100  "
          f"(Human=100, Fly=52, C.elegans=23)")
    print(f"  ISS v2 (uncapped): {metrics['iss_v2']:>8.1f}  "
          f"(Human=100, Einstein~107, 12-layer=112)")

    # Shell-by-shell view
    print(f"\n  Shell list + layers + activity:")
    layers = graph.get("layers", {})
    activity = graph.get("activity", {})
    for i, name in enumerate(graph["nodes"]):
        lyr = layers.get(name, "?")
        act = activity.get(name, 0.0)
        inh = "(inhib)" if graph["inhibitory"][i] else ""
        print(f"    {name:20s} layer={lyr} activity={act:.3f} {inh}")


if __name__ == "__main__":
    # Smoke test: build a real Tamashii and measure
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "tamashii")
    from tamashii.runner_3d import build_fluctlight
    from tamashii.phase_9_ecology import SHELLS_DEFAULT

    agent = build_fluctlight(
        SHELLS_DEFAULT, "tamashii/configs", "tamashii/configs",
        use_3d_brain=True)
    print("Measuring current flat Tamashii...")
    graph = measure_tamashii_graph(agent, n_warmup_steps=50)
    metrics = compute_iss(graph)
    print_iss_report(graph, metrics, label="Flat Tamashii (current)")
