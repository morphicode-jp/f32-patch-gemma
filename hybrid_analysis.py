"""hybrid_analysis.py - Direct landscape probe instead of Sentinel search.

Sentinel failed on hybrid_eval because multiplicative landscape has narrow
high-value regions (proxy hallucinates, real=0). Instead, take existing
optimized brains and measure their lambda/ISS/behavior profiles directly.

Brains under test:
  1. Uniform Kathara (edges=1.0) - the "natural" universe-aligned baseline
  2. ISS-optimized (from brain_sentinel_biological_result.json)
  3. Lambda-optimized (from lambda_eval_result.json)
  4. Midpoint baseline

Output: full triangle of (lambda, ISS, FlyWorld) for each.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, FlyWorldV3, simulate_step
from hybrid_eval_sentinel import lambda_score, iss_score_fn, run_unseen


def flyworld_perf(params, n_seeds=20):
    seeds, scores = run_unseen(params, n_seeds=n_seeds)
    return float(np.mean(scores)), float(np.std(scores)), \
        sum(1 for s in scores if s >= 30) / len(scores)


def load_best(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("best_ever_params") or d.get("best_params")


def lambda2_value(params):
    """Return raw |lambda_2| (NOT the score)."""
    from kathara_brain_sim_v8 import KATHARA_EDGES
    p = np.array(params, dtype=np.float64)
    W = np.zeros((12, 12), dtype=np.float64)
    for i, (a, b) in enumerate(KATHARA_EDGES):
        W[a, b] = p[i]; W[b, a] = p[i]
    eigs_abs = sorted(np.abs(np.linalg.eigvalsh(W)), reverse=True)
    return eigs_abs[1] if len(eigs_abs) > 1 else 0.0


def main():
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    uniform = [1.0] * 30 + mid[30:]

    iss_params = load_best("brain_sentinel_biological_result.json")
    lam_params = load_best("lambda_eval_result.json")
    hyb_params = load_best("hybrid_eval_result.json")  # may not exist

    brains = {
        "midpoint":     mid,
        "uniform_k1":   uniform,
        "iss_opt":      iss_params,
        "lambda_opt":   lam_params,
    }
    if hyb_params:
        brains["hybrid_opt"] = hyb_params

    print("=" * 90)
    print(f"  {'brain':<15s}  {'|λ₂|':>6s}  {'λ_sc':>6s}  {'ISS':>6s}  "
          f"{'hyb':>6s}  {'unseen':>8s}  {'reach':>6s}")
    print("=" * 90)

    results = {}
    for name, p in brains.items():
        if p is None:
            print(f"  {name:<15s}  (missing)")
            continue
        lam2 = lambda2_value(p)
        l = lambda_score(p)
        i = iss_score_fn(p, n_ep=4, n_steps=30)
        h = float(np.sqrt(max(l, 0) * max(i, 0)))
        mean, std, reach = flyworld_perf(p, n_seeds=20)
        print(f"  {name:<15s}  {lam2:>6.2f}  {l:>6.2f}  {i:>6.2f}  "
              f"{h:>6.2f}  {mean:>8.2f}  {reach*100:>5.0f}%")
        results[name] = {
            "lambda2_raw": round(lam2, 3),
            "lambda_score": round(l, 2),
            "iss": round(i, 2),
            "hybrid": round(h, 2),
            "unseen_mean": round(mean, 2),
            "unseen_std": round(std, 2),
            "reach_rate": round(reach, 3),
        }
    print("=" * 90)

    # Correlation analysis
    names = [n for n in brains if brains[n] is not None]
    if len(names) >= 3:
        lam2s = np.array([results[n]["lambda2_raw"] for n in names])
        lams = np.array([results[n]["lambda_score"] for n in names])
        isss = np.array([results[n]["iss"] for n in names])
        hyb = np.array([results[n]["hybrid"] for n in names])
        unseens = np.array([results[n]["unseen_mean"] for n in names])

        def corr(a, b):
            if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        print("\nCorrelations with unseen_mean (n={}):".format(len(names)))
        print(f"  |λ₂| raw  -> unseen:  r = {corr(lam2s, unseens):+.3f}")
        print(f"  λ score   -> unseen:  r = {corr(lams, unseens):+.3f}")
        print(f"  ISS       -> unseen:  r = {corr(isss, unseens):+.3f}")
        print(f"  hybrid    -> unseen:  r = {corr(hyb, unseens):+.3f}")

        # Find which lambda target would maximize correlation
        print("\nBest lambda_target sweep (n={}, target = fit peak):".format(len(names)))
        best_r, best_t = -1.0, 4.0
        for t in np.arange(2.0, 9.0, 0.25):
            scores = 100 * np.exp(-((lam2s - t) ** 2) / 2.0)
            r = corr(scores, unseens)
            if r > best_r:
                best_r, best_t = r, t
        print(f"  best target lambda = {best_t:.2f} (r = {best_r:+.3f})")

    with open("hybrid_analysis_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: hybrid_analysis_result.json")


if __name__ == "__main__":
    main()
