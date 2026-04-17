"""validate_lambda_correlation.py - Does |λ₂| -> unseen r=+0.97 hold at n=25?

Previous finding (n=5): r = +0.970. But n=5 is fragile.
This script adds 20 random brains with varied edge magnitudes to test robustness.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, KATHARA_EDGES
from hybrid_eval_sentinel import run_unseen


def lambda2(params):
    p = np.array(params, dtype=np.float64)
    W = np.zeros((12, 12))
    for i, (a, b) in enumerate(KATHARA_EDGES):
        W[a, b] = p[i]; W[b, a] = p[i]
    eigs = sorted(np.abs(np.linalg.eigvalsh(W)), reverse=True)
    return eigs[1] if len(eigs) > 1 else 0.0


def load_best(path):
    if not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    return d.get("best_ever_params") or d.get("best_params")


def main():
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    brains = [("midpoint", mid),
              ("uniform_k1", [1.0] * 30 + mid[30:])]

    for name, path in [("iss_opt", "brain_sentinel_biological_result.json"),
                       ("lambda_opt", "lambda_eval_result.json"),
                       ("hybrid_opt", "hybrid_eval_result.json")]:
        p = load_best(path)
        if p: brains.append((name, p))

    # 20 random brains, varying edge scale to sweep |λ₂|
    rng = np.random.RandomState(12345)
    for i in range(20):
        edge_scale = rng.uniform(0.2, 3.0)  # spread to cover λ₂ range
        edges = [rng.uniform(-edge_scale, edge_scale) for _ in range(30)]
        node_params = [rng.uniform(lo, hi) for lo, hi in PARAM_RANGES[30:]]
        p = edges + node_params
        brains.append((f"rand_{i}", p))

    print(f"Testing n={len(brains)} brains (20 unseen FlyWorldV3 seeds each)...")
    print(f"{'brain':<15s}  {'|λ₂|':>6s}  {'unseen':>8s}  {'reach':>6s}")
    print("-" * 45)

    data = []
    for name, p in brains:
        l2 = lambda2(p)
        _, scores = run_unseen(p, n_seeds=20)
        um = float(np.mean(scores))
        reach = sum(1 for s in scores if s >= 30) / len(scores)
        print(f"{name:<15s}  {l2:>6.2f}  {um:>8.2f}  {reach*100:>5.0f}%")
        data.append((name, l2, um, reach))

    lams = np.array([d[1] for d in data])
    unseens = np.array([d[2] for d in data])
    reaches = np.array([d[3] for d in data])

    r_lu = float(np.corrcoef(lams, unseens)[0, 1])
    r_lr = float(np.corrcoef(lams, reaches)[0, 1])
    print(f"\nCorrelations (n={len(data)}):")
    print(f"  |λ₂| -> unseen_mean: r = {r_lu:+.3f}")
    print(f"  |λ₂| -> reach_rate:  r = {r_lr:+.3f}")

    # Rank correlation (more robust)
    from scipy.stats import spearmanr
    try:
        rho_lu = float(spearmanr(lams, unseens).statistic)
        rho_lr = float(spearmanr(lams, reaches).statistic)
        print(f"\nSpearman rank correlations (robust):")
        print(f"  |λ₂| -> unseen: rho = {rho_lu:+.3f}")
        print(f"  |λ₂| -> reach:  rho = {rho_lr:+.3f}")
    except Exception as e:
        rho_lu = rho_lr = None

    # Bin by λ₂ quartile
    sorted_idx = np.argsort(lams)
    q1 = sorted_idx[:len(lams) // 4]
    q4 = sorted_idx[-len(lams) // 4:]
    print(f"\nQuartile analysis:")
    print(f"  Q1 (low λ₂,  mean={lams[q1].mean():.2f}): unseen_mean={unseens[q1].mean():.2f}")
    print(f"  Q4 (high λ₂, mean={lams[q4].mean():.2f}): unseen_mean={unseens[q4].mean():.2f}")
    print(f"  Q4/Q1 lift: {unseens[q4].mean() / max(unseens[q1].mean(), 0.01):.1f}x")

    with open("validate_lambda_result.json", "w") as f:
        json.dump({
            "n_brains": len(data),
            "data": [{"name": n, "lambda2": round(l, 3),
                     "unseen_mean": round(u, 2),
                     "reach_rate": round(r, 3)} for n, l, u, r in data],
            "pearson_r_unseen": round(r_lu, 3),
            "pearson_r_reach": round(r_lr, 3),
            "spearman_rho_unseen": round(rho_lu, 3) if rho_lu is not None else None,
            "spearman_rho_reach": round(rho_lr, 3) if rho_lr is not None else None,
        }, f, indent=2)
    print("\nSaved: validate_lambda_result.json")


if __name__ == "__main__":
    main()
