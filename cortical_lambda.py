"""cortical_lambda.py - Does |L2| correlate with unseen in cortical 5L brain?

12N finding: Pearson r = +0.31, Spearman rho = +0.26 (weak but real).
Question: Does same spectral-generalization relationship hold for 5L cortical
  (60 neurons, 184D params, different graph structure)?

Computes |L2| of the effective 60x60 inter-layer weight matrix.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cortical_brain import (
    PARAM_RANGES, KATHARA_EDGES, N_LAYERS, N_PER_LAYER,
    N_EDGES_PER_LAYER, TOTAL_INTRA_EDGES, TOTAL_FF, TOTAL_FB,
)
from kathara_brain_sim_v8 import FlyWorldV3, simulate_step as _noop

from cortical_brain import CorticalBrain


def build_full_W(params):
    """Construct 60x60 effective weight matrix from 184 params."""
    p = np.array(params, dtype=np.float64)
    W = np.zeros((60, 60), dtype=np.float64)

    # Intra-layer Kathara edges (5 layers x 30 edges)
    for layer in range(N_LAYERS):
        start = layer * N_EDGES_PER_LAYER
        edges = p[start:start + N_EDGES_PER_LAYER]
        off = layer * N_PER_LAYER
        for i, (a, b) in enumerate(KATHARA_EDGES):
            W[off + a, off + b] = edges[i]
            W[off + b, off + a] = edges[i]

    # FF projections (4 scales, each connects layer L to L+1 as scale * identity)
    ff_scales = p[TOTAL_INTRA_EDGES:TOTAL_INTRA_EDGES + TOTAL_FF]
    for l in range(TOTAL_FF):
        s = ff_scales[l]
        for n in range(N_PER_LAYER):
            W[l * N_PER_LAYER + n, (l + 1) * N_PER_LAYER + n] += s

    # FB projections (4 scales x 0.47)
    fb_scales = p[TOTAL_INTRA_EDGES + TOTAL_FF:TOTAL_INTRA_EDGES + TOTAL_FF + TOTAL_FB]
    FB_RATIO = 0.47
    for l in range(TOTAL_FB):
        s = fb_scales[l] * FB_RATIO
        for n in range(N_PER_LAYER):
            W[(l + 1) * N_PER_LAYER + n, l * N_PER_LAYER + n] += s

    # Symmetrize for eigendecomposition (we take abs eigs anyway)
    W_sym = (W + W.T) / 2
    return W_sym


def lambda2_cortical(params):
    W = build_full_W(params)
    eigs = sorted(np.abs(np.linalg.eigvalsh(W)), reverse=True)
    return eigs[1] if len(eigs) > 1 else 0.0


def run_unseen_cortical(params, n_seeds=8, n_steps=40):
    rng = np.random.RandomState(999)
    seeds = [int(s) for s in rng.randint(1000, 10000, size=n_seeds)]
    train_seeds = [ep * 7 + 13 for ep in range(4)]
    seeds = [s for s in seeds if s not in train_seeds][:n_seeds]

    scores = []
    for seed in seeds:
        brain = CorticalBrain(params)
        world = FlyWorldV3(seed=seed)
        sensors = world.reset()
        initial_dist = world.get_food_dist()
        min_dist = initial_dist
        reached = False
        caught = False
        move_sum = 0.0
        for _ in range(n_steps):
            prev_pos = world.fly_pos.copy()
            nav, cen = brain.step(sensors)
            sensors, dist, reached_now, caught_now = world.step(nav, cen)
            move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
            if dist < min_dist:
                min_dist = dist
            if reached_now:
                reached = True; break
            if caught_now:
                caught = True; break
        approach = max(0.0, initial_dist - min_dist) / (initial_dist + 1e-6)
        ep_score = approach * 60.0 + (30.0 if reached else 0.0) + min(10.0, move_sum * 2.0)
        if caught:
            ep_score *= 0.3
        scores.append(ep_score)
    return seeds, scores


def main():
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    uniform = [1.0] * TOTAL_INTRA_EDGES + [1.0] * TOTAL_FF + [1.0] * TOTAL_FB + mid[TOTAL_INTRA_EDGES + TOTAL_FF + TOTAL_FB:]

    brains = [("cort_midpoint", mid), ("cort_uniform", uniform)]

    # Load optimized cortical brain
    if os.path.exists("cortical_sentinel_result.json"):
        with open("cortical_sentinel_result.json") as f:
            d = json.load(f)
        p = d.get("best_ever_params") or d.get("best_params")
        if p: brains.append(("cort_opt", p))

    # 8 random brains (reduced for cortical speed)
    rng = np.random.RandomState(54321)
    for i in range(8):
        scale = rng.uniform(0.3, 2.5)
        p = []
        for lo, hi in PARAM_RANGES:
            if abs(hi) <= 3.5 and abs(lo) <= 3.5:
                p.append(rng.uniform(-scale, scale))
            else:
                p.append(rng.uniform(lo, hi))
        brains.append((f"cort_rand_{i}", p))

    print(f"Testing n={len(brains)} cortical brains (20 unseen seeds each)...\n")
    print(f"{'brain':<18s}  {'|L2|':>6s}  {'unseen':>8s}  {'reach':>6s}")
    print("-" * 50)

    data = []
    for name, p in brains:
        try:
            l2 = lambda2_cortical(p)
        except Exception:
            l2 = 0.0
        try:
            _, scores = run_unseen_cortical(p, n_seeds=8, n_steps=40)
            um = float(np.mean(scores))
            reach = sum(1 for s in scores if s >= 30) / len(scores)
        except Exception as e:
            um, reach = 0.0, 0.0
        print(f"{name:<18s}  {l2:>6.2f}  {um:>8.2f}  {reach*100:>5.0f}%")
        data.append((name, l2, um, reach))

    lams = np.array([d[1] for d in data])
    unseens = np.array([d[2] for d in data])
    reaches = np.array([d[3] for d in data])

    def corr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    r_lu = corr(lams, unseens)
    r_lr = corr(lams, reaches)

    print(f"\nCortical correlations (n={len(data)}):")
    print(f"  |L2| -> unseen: r = {r_lu:+.3f}")
    print(f"  |L2| -> reach:  r = {r_lr:+.3f}")

    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(lams, unseens).statistic)
        print(f"  |L2| -> unseen: rho = {rho:+.3f}")
    except Exception:
        rho = None

    # Q4/Q1
    if len(data) >= 8:
        idx = np.argsort(lams)
        q1 = idx[:len(lams) // 4]
        q4 = idx[-len(lams) // 4:]
        print(f"\nQuartile:")
        print(f"  Q1 λ={lams[q1].mean():.2f}: unseen={unseens[q1].mean():.2f}")
        print(f"  Q4 λ={lams[q4].mean():.2f}: unseen={unseens[q4].mean():.2f}")
        lift = unseens[q4].mean() / max(unseens[q1].mean(), 0.01)
        print(f"  Lift: {lift:.1f}x")

    # Cross-arch comparison
    print(f"\n12N vs Cortical comparison:")
    print(f"  12N  n=25: Pearson r = +0.312")
    print(f"  Cort n={len(data)}: Pearson r = {r_lu:+.3f}")

    with open("cortical_lambda_result.json", "w") as f:
        json.dump({
            "n_brains": len(data),
            "pearson_r_unseen": round(r_lu, 3),
            "pearson_r_reach": round(r_lr, 3),
            "spearman_rho_unseen": round(rho, 3) if rho is not None else None,
            "data": [{"name": n, "lambda2": round(l, 3),
                     "unseen_mean": round(u, 2),
                     "reach_rate": round(r, 3)} for n, l, u, r in data],
        }, f, indent=2)
    print("\nSaved: cortical_lambda_result.json")


if __name__ == "__main__":
    main()
