"""phase7_reach_threshold_sweep.py - Diagnostic: is reach ceiling at last 50cm?

Load trained N=5 optimal config agents. Evaluate at multiple reach
thresholds (0.5, 0.8, 1.0, 1.5, 2.0). If reach jumps at 1.0m from 7%
to >50%, confirms "architectural ceiling = inability to close last
50cm", not general navigation failure.

This is a 5-minute diagnostic with no new training — just re-evaluates
existing trained params with relaxed thresholds.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_coop_sentinel import run_coop_match


def evaluate_with_threshold(agent_params, threshold, n_seeds=20, n_steps=50):
    """Evaluate reach at a custom threshold by patching world.step detection.

    We run match normally, then recompute reach = (final_dist < threshold).
    """
    reach_counts = [0] * len(agent_params)
    for seed in range(2000, 2000 + n_seeds):
        _, _, final_dists = run_coop_match(
            agent_params, seed=seed, n_steps=n_steps
        )
        # final_dists: distance at end of episode (either hit or timed out)
        for i, d in enumerate(final_dists):
            if d < threshold:
                reach_counts[i] += 1
    reach_rates = [c / n_seeds for c in reach_counts]
    return {
        "threshold": threshold,
        "mean_reach": float(np.mean(reach_rates)),
        "per_agent_reach": reach_rates,
    }


def main():
    # Load trained N=5 optimal agents
    with open("phase7_N5_optimal_history.json") as f:
        data = json.load(f)
    agent_params = data["final_params"]
    print(f"  Loaded {len(agent_params)} trained agents")
    print(f"  Config: {data['reigen_config']}")

    print("\n" + "=" * 60)
    print("  Reach vs threshold sweep (20 unseen seeds)")
    print("=" * 60)

    thresholds = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    results = []
    for t in thresholds:
        r = evaluate_with_threshold(agent_params, threshold=t, n_seeds=20)
        results.append(r)
        per_a = " ".join(f"{x*100:.0f}%" for x in r["per_agent_reach"])
        bar = "#" * int(r["mean_reach"] * 40)
        print(f"  threshold={t:>4.1f}m  mean={r['mean_reach']*100:>3.0f}%  "
              f"per_agent=[{per_a}]  {bar}")

    # Save
    with open("phase7_reach_threshold_sweep.json", "w") as f:
        json.dump({
            "config": data["reigen_config"],
            "thresholds": thresholds,
            "results": results,
        }, f, indent=2)

    # Interpretation
    print("\n  Interpretation:")
    r05 = results[0]["mean_reach"]
    r10 = results[2]["mean_reach"]
    r20 = results[4]["mean_reach"]
    print(f"    threshold=0.5m:  {r05*100:.0f}%  <- current criterion")
    print(f"    threshold=1.0m:  {r10*100:.0f}%")
    print(f"    threshold=2.0m:  {r20*100:.0f}%")

    if r10 > 0.5 and r05 < 0.2:
        print("\n  CONFIRMED: reach ceiling = failure to close last 50-100cm.")
        print("    Navigation up to 1m works well. Final approach is the bottleneck.")
        print("    Fix: sensor gradient handling at close range, or smaller food/world.")
    elif r20 < 0.3:
        print("\n  Reach failure is GLOBAL, not just final approach.")
    else:
        print("\n  Intermediate: agents cluster near food but don't consistently close.")


if __name__ == "__main__":
    main()
