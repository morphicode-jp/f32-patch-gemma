"""brain_growth.py — Multi-scale safe brain evolution with Sentinel + ISS v2

Grows a brain from 12N -> 48N -> 144N -> 576N, optimizing each scale with
Sentinel (eval=compute_iss v2, guard=graph connectivity + short paths).

At each scale we search 8 topology parameters:
  5 Kathara offsets (inter-layer bridges) + 3 skip distances (long-range)

Per-eval cost: ~50ms (graph build + networkx L/C/hub). 20 min budget gives
thousands of evals per stage.

Usage:
    python brain_growth.py --stage-budget 300
    # 4 stages x 300s = 1200s total (20 min)
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from twelve.agent.iss_calculator import compute_iss


# ============================================================
# 12-node Kathara adjacency (base pattern — fixed)
# ============================================================
KATHARA_ADJ = {
    0: [1, 3, 5, 10, 11], 1: [0, 2, 4, 7, 9],  2: [1, 3, 5, 8, 10],
    3: [0, 2, 4, 6, 9],   4: [1, 3, 5, 7, 11], 5: [0, 2, 4, 6, 8],
    6: [3, 5, 7, 9, 11],  7: [1, 4, 6, 8, 10], 8: [2, 5, 7, 9, 11],
    9: [1, 3, 6, 8, 10],  10: [0, 2, 7, 9, 11], 11: [0, 4, 6, 8, 10],
}
N_NODES_PER_LAYER = 12

PARAM_NAMES = [f"off{i}" for i in range(5)] + [f"skip{i}" for i in range(3)]


# ============================================================
# Graph construction
# ============================================================
def build_layered_graph(n_layers, offsets, skip_dists):
    """Build layered Kathara graph.

    Each layer: 12-node Kathara {1,4,6} pattern (intra-layer, 30 edges).
    Inter-layer: 5 offsets bridge layer L to L+1 (cyclic).
    Skip: 3 distances add long-range connections.
    """
    N_N = N_NODES_PER_LAYER
    N_TOT = n_layers * N_N
    G = nx.Graph()
    G.add_nodes_from(range(N_TOT))

    # Intra-layer edges
    for layer in range(n_layers):
        b = layer * N_N
        for n, nbs in KATHARA_ADJ.items():
            for nb in nbs:
                G.add_edge(b + n, b + nb)

    # Inter-layer bridges (offsets to next layer, cyclic)
    if n_layers > 1:
        for layer in range(n_layers):
            b = layer * N_N
            tl = (layer + 1) % n_layers
            tb = tl * N_N
            for off in offsets:
                for n in range(N_N):
                    G.add_edge(b + n, tb + (n + off) % N_N)

        # Skip connections
        for layer in range(n_layers):
            b = layer * N_N
            for k, (off, skip) in enumerate(zip(offsets[:3], skip_dists)):
                tl = (layer + skip) % n_layers
                if tl == (layer + 1) % n_layers or tl == layer:
                    continue
                tb = tl * N_N
                for n in range(0, N_N, 2):  # half density to avoid over-connection
                    G.add_edge(b + n, tb + (n + off) % N_N)

    return G


def measure_graph(G, n_samples=50):
    """Compute (L, C, hub_concentration) for graph G."""
    if not nx.is_connected(G):
        return None
    N_TOT = G.number_of_nodes()
    n_samp = min(n_samples, N_TOT)
    rng = np.random.RandomState(42)
    samp = rng.choice(N_TOT, n_samp, replace=False)
    total, cnt = 0, 0
    for i in range(n_samp):
        lens = nx.single_source_shortest_path_length(G, int(samp[i]))
        for j in range(i + 1, n_samp):
            if int(samp[j]) in lens:
                total += lens[int(samp[j])]
                cnt += 1
    L = total / max(cnt, 1)
    C = nx.average_clustering(G)
    degs = sorted([d for _, d in G.degree()], reverse=True)
    top10 = degs[: max(1, len(degs) // 10)]
    hub = sum(top10) / max(sum(degs), 1) * 100.0
    return L, C, hub


# ============================================================
# eval_fn / guard_fn builders
# ============================================================
def _decode_params(params):
    """Decode 8-D param vector to (offsets, skips)."""
    offsets = [int(round(float(p))) % 12 for p in params[:5]]
    skips = [max(2, int(round(float(p)))) for p in params[5:8]]
    return offsets, skips


def make_iss_eval(n_layers):
    """eval_fn: topology params -> ISS v2 score (maximize)."""
    def eval_fn(params):
        offsets, skips = _decode_params(params)
        G = build_layered_graph(n_layers, offsets, skips)
        m = measure_graph(G)
        if m is None:
            return 0.0
        L, C, hub = m
        return float(compute_iss(n_layers, L, C, 0.20, hub, v2=True))
    return eval_fn


def make_connectivity_guard(n_layers):
    """guard_fn: must stay connected + reasonable path length.

    Returns 0 if disconnected, else 100 - clamp(L*5, 0, 100).
    Short paths = better connected = higher guard.
    """
    def guard_fn(params):
        offsets, skips = _decode_params(params)
        G = build_layered_graph(n_layers, offsets, skips)
        m = measure_graph(G)
        if m is None:
            return 0.0
        L, _, _ = m
        return max(0.0, 100.0 - min(L * 5.0, 100.0))
    return guard_fn


# ============================================================
# Stage runner
# ============================================================
def get_ranges(n_layers):
    """Param ranges scaled to layer count. Skip distance max = n_layers/2."""
    skip_max = max(3, min(n_layers // 2, 24))
    return [
        (0.0, 11.0),  # off0
        (0.0, 11.0),  # off1
        (0.0, 11.0),  # off2
        (0.0, 11.0),  # off3
        (0.0, 11.0),  # off4
        (2.0, float(skip_max)),
        (2.0, float(skip_max)),
        (2.0, float(skip_max)),
    ]


def run_stage(n_layers, time_budget=300, progress_file=None):
    """Run Sentinel on one scale with warm-start + learn."""
    eval_fn = make_iss_eval(n_layers)
    guard_fn = make_connectivity_guard(n_layers)
    ranges = get_ranges(n_layers)

    # Known-good Kathara seed (from optimize_iss_140.py which achieved ISS=126)
    skip_max = max(3, min(n_layers // 2, 24))
    initial = [0.0, 1.0, 4.0, 6.0, 8.0,
               float(min(2, skip_max)),
               float(min(4, skip_max)),
               float(min(8, skip_max))]

    N_TOT = n_layers * N_NODES_PER_LAYER
    print(f"\n{'#' * 70}")
    print(f"#  STAGE: {n_layers}L x 12N = {N_TOT} nodes  (budget {time_budget}s)")
    print(f"#  warm-start: offsets={initial[:5]}, skips={initial[5:]}")
    print(f"{'#' * 70}\n")

    t_start = time.time()
    result = Sentinel(
        eval_fn=eval_fn,
        guard_fn=guard_fn,
        param_ranges=ranges,
        param_names=PARAM_NAMES,
        experience_id=f"brain_growth_{n_layers}L",
        initial_params=initial,
        learn=True,
    ).run(time_budget=time_budget, verbose=True)
    elapsed = time.time() - t_start

    # Measure final graph
    best = result["best_params"]
    offsets, skips = _decode_params(best)
    G = build_layered_graph(n_layers, offsets, skips)
    m = measure_graph(G)

    entry = {
        "n_layers": n_layers,
        "n_nodes": N_TOT,
        "verdict": result["verdict"],
        "iss_v2": round(float(result["eval_score"]), 1),
        "conn_guard": round(float(result["guard_score"]), 1),
        "baseline_guard": round(float(result["baseline_guard"]), 1),
        "L": round(m[0], 3) if m else None,
        "C": round(m[1], 3) if m else None,
        "hub": round(m[2], 1) if m else None,
        "n_edges": G.number_of_edges(),
        "offsets": offsets,
        "skips": skips,
        "proxy_r2": round(float(result["proxy_r2"]), 3),
        "optimization_mode": result.get("optimization_mode", "owl"),
        "elapsed_s": round(elapsed, 1),
        "safe_dims": result.get("safe_dims"),
        "conflict_dims": result.get("conflict_dims"),
    }

    print(f"\n  [{N_TOT}N] verdict={result['verdict']}  "
          f"ISS={entry['iss_v2']}  L={entry['L']}  C={entry['C']}  "
          f"hub={entry['hub']}  edges={entry['n_edges']}")

    # Live progress dump
    if progress_file is not None:
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"stages": []}
        data["stages"].append(entry)
        data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(progress_file, "w") as f:
            json.dump(data, f, indent=2)

    return entry


def main():
    ap = argparse.ArgumentParser(description="Multi-scale brain growth with Sentinel + ISS")
    ap.add_argument("--stage-budget", type=int, default=300,
                    help="Seconds per stage (default 300 = 5 min)")
    ap.add_argument("--layers", default="4,12,48",
                    help="Comma-separated layer counts (1L dropped: no inter-layer params)")
    args = ap.parse_args()

    layer_list = [int(x) for x in args.layers.split(",")]
    progress_file = "brain_growth_progress.json"

    # Reset progress file
    with open(progress_file, "w") as f:
        json.dump({"stages": [], "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    total_start = time.time()
    summary = []

    for n_layers in layer_list:
        entry = run_stage(n_layers, time_budget=args.stage_budget,
                          progress_file=progress_file)
        summary.append(entry)

    total_elapsed = time.time() - total_start

    # Final report
    print("\n" + "=" * 78)
    print("  BRAIN GROWTH PIPELINE SUMMARY")
    print("=" * 78)
    print(f"{'Scale':<10} {'Nodes':>7} {'ISS v2':>8} {'L':>6} {'C':>6} "
          f"{'Hub':>6} {'Edges':>8} {'Verdict':>10}")
    print("-" * 78)
    for e in summary:
        L = f"{e['L']:.2f}" if e['L'] is not None else "-"
        C = f"{e['C']:.2f}" if e['C'] is not None else "-"
        hub = f"{e['hub']:.1f}" if e['hub'] is not None else "-"
        print(f"{e['n_layers']:>3}L x 12N {e['n_nodes']:>7} {e['iss_v2']:>8.1f} "
              f"{L:>6} {C:>6} {hub:>6} {e['n_edges']:>8} {e['verdict']:>10}")
    print("-" * 78)
    print(f"\n  Total elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    # Save final
    with open("brain_growth_result.json", "w") as f:
        json.dump({
            "pipeline_elapsed_s": round(total_elapsed, 1),
            "stages": summary,
        }, f, indent=2)
    print("  Saved: brain_growth_result.json")


if __name__ == "__main__":
    main()
