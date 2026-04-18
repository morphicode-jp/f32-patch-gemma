"""phase7_evaluation.py - Evaluation suite for Phase 7 intelligence.

Four evaluations (mapped to pre-registered success criteria):
  1. Standard task: N=5 unseen reach (>80% target)
  2. MI panel:      per-agent voice-to-food-direction MI (>0.08 bits target)
  3. ToM test:      can agent predict partner's action? (>45% accuracy target)
  4. Scales test:   trained on N=5, run on N=10 (>60% reach target)

Optional:
  5. Ablations:     Hebbian off, topology 12 (not yet wired but slot exists)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7_coop_sentinel import run_coop_match
from signal_sentinel import mutual_information


def evaluate_standard(agent_params, n_seeds=20, n_steps=50):
    """Reach rate + per-agent score on unseen seeds."""
    N = len(agent_params)
    scores_list = []
    reaches_list = []
    finals_list = []
    for seed in range(2000, 2000 + n_seeds):
        scores, reached, finals = run_coop_match(
            agent_params, seed=seed, n_steps=n_steps
        )
        scores_list.append(scores)
        reaches_list.append(reached)
        finals_list.append(finals)
    scores_arr = np.array(scores_list)  # (n_seeds, N)
    reaches_arr = np.array(reaches_list)  # bool
    finals_arr = np.array(finals_list)

    per_agent_reach = reaches_arr.mean(axis=0)
    group_reach = float(reaches_arr.any(axis=1).mean())
    all_reached = float(reaches_arr.all(axis=1).mean())

    return {
        "n_seeds": n_seeds,
        "per_agent_reach_rate": [float(r) for r in per_agent_reach],
        "mean_agent_reach_rate": float(per_agent_reach.mean()),
        "group_at_least_one_reach": group_reach,
        "group_all_reach": all_reached,
        "mean_score": float(scores_arr.mean()),
        "mean_final_dist": float(finals_arr.mean()),
    }


def evaluate_mi_panel(agent_params, n_episodes=30, n_perm=100, n_bins=4):
    """Per-agent MI(voice, food_direction). Returns list of MI + p-values."""
    N = len(agent_params)
    # Collect per-episode voice and direction traces
    all_voices = [[] for _ in range(N)]
    all_dirs = [[] for _ in range(N)]
    for ep in range(n_episodes):
        _, _, _, traces = run_coop_match(
            agent_params, seed=5000 + ep, n_steps=50, record_traces=True
        )
        for i in range(N):
            all_voices[i].extend(traces["voices"][i])
            all_dirs[i].extend(traces["dirs"][i])

    # Compute observed MI + permutation null
    import random
    results = []
    for i in range(N):
        if len(all_voices[i]) < 10:
            results.append({"agent": i, "mi": 0.0, "p": 1.0, "gain": 0.0})
            continue
        mi = mutual_information(all_voices[i], all_dirs[i], n_bins=n_bins)
        rng = random.Random(42 + i)
        nulls = []
        for _ in range(n_perm):
            shuffled = list(all_dirs[i])
            rng.shuffle(shuffled)
            nulls.append(mutual_information(all_voices[i], shuffled, n_bins=n_bins))
        nulls_arr = np.array(nulls)
        p = float((nulls_arr >= mi).sum() / n_perm)
        gain = float(mi - nulls_arr.mean())
        results.append({
            "agent": i, "mi": float(mi), "p": p, "gain": gain,
            "null_mean": float(nulls_arr.mean()),
        })

    gains = [r["gain"] for r in results]
    return {
        "per_agent": results,
        "max_gain": max(gains) if gains else 0.0,
        "mean_gain": float(np.mean(gains)) if gains else 0.0,
        "n_significant": sum(1 for r in results if r["p"] < 0.05),
        "n_samples_per_agent": [len(all_voices[i]) for i in range(N)],
    }


def evaluate_tom(agent_params, n_episodes=20, n_bins=3):
    """Theory-of-Mind test: does agent i's voice predict partner j's
    NEXT action (nav direction)?

    Collect (voice_i_at_t, nav_j_at_t+1) pairs. Compute MI.
    """
    N = len(agent_params)
    # voice_i[t] -> nav_j[t+1] predictions across all (i, j) pairs
    predictions = {}  # (i, j) -> {voice_i: [], nav_j: []}
    for ep in range(n_episodes):
        _, _, _, traces = run_coop_match(
            agent_params, seed=7000 + ep, n_steps=50, record_traces=True
        )
        for i in range(N):
            for j in range(N):
                if i == j: continue
                key = (i, j)
                if key not in predictions:
                    predictions[key] = {"voice_i": [], "nav_j_next": []}
                # Pair voice_i[t] with nav_j[t+1]
                v_i = traces["voices"][i]
                n_j = traces["navs"][j]
                T = min(len(v_i), len(n_j)) - 1
                for t in range(T):
                    predictions[key]["voice_i"].append(v_i[t])
                    # Discretize nav into {left, straight, right}
                    nav = n_j[t + 1]
                    if nav < 0.4: cat = "left"
                    elif nav > 0.6: cat = "right"
                    else: cat = "straight"
                    predictions[key]["nav_j_next"].append(cat)

    # Compute MI for each (i, j) pair
    results = []
    import random
    for (i, j), data in predictions.items():
        if len(data["voice_i"]) < 10:
            continue
        mi = mutual_information(data["voice_i"], data["nav_j_next"], n_bins=n_bins)
        # Null
        rng = random.Random(99 + i * N + j)
        shuffled = list(data["nav_j_next"])
        rng.shuffle(shuffled)
        null_mi = mutual_information(data["voice_i"], shuffled, n_bins=n_bins)
        results.append({
            "speaker": i, "listener": j,
            "mi_cross": float(mi),
            "mi_null": float(null_mi),
            "gain": float(mi - null_mi),
        })

    gains = [r["gain"] for r in results]
    # "Prediction accuracy" analogue: fraction of pairs with gain > 0.01
    n_predictive = sum(1 for r in results if r["gain"] > 0.01)
    return {
        "n_pairs_measured": len(results),
        "n_pairs_predictive": n_predictive,
        "pair_fraction_predictive": (n_predictive / len(results)
                                      if results else 0.0),
        "max_gain": float(max(gains)) if gains else 0.0,
        "mean_gain": float(np.mean(gains)) if gains else 0.0,
        "top_5_pairs": sorted(results, key=lambda r: -r["gain"])[:5],
    }


def evaluate_scales(agent_params, target_N, n_seeds=10):
    """Trained on N=|agent_params|, extended to target_N by replication.

    If target_N > len(agent_params), we replicate agents (wraparound).
    Measures degradation.
    """
    base_N = len(agent_params)
    if target_N <= base_N:
        # Sub-sample
        params_list = agent_params[:target_N]
    else:
        # Replicate (wraparound)
        params_list = [agent_params[i % base_N] for i in range(target_N)]

    scores_list = []
    reaches_list = []
    for seed in range(8000, 8000 + n_seeds):
        scores, reached, finals = run_coop_match(
            params_list, seed=seed, n_steps=60
        )
        scores_list.append(scores)
        reaches_list.append(reached)
    reaches_arr = np.array(reaches_list)
    per_agent_reach = reaches_arr.mean(axis=0)
    return {
        "target_N": target_N,
        "base_N": base_N,
        "mean_agent_reach_rate": float(per_agent_reach.mean()),
        "per_agent_reach_rate": [float(r) for r in per_agent_reach],
    }


def run_full_evaluation(agent_params, save_path="phase7_eval_result.json"):
    """Run all 4 evaluations + save."""
    N = len(agent_params)
    print("=" * 70)
    print(f"  PHASE 7 EVALUATION (N={N} trained agents)")
    print("=" * 70)

    print("\n[1/4] Standard task (20 unseen seeds)...")
    std = evaluate_standard(agent_params, n_seeds=20)
    print(f"  mean agent reach: {std['mean_agent_reach_rate']*100:.0f}%")
    print(f"  group at least 1: {std['group_at_least_one_reach']*100:.0f}%")
    print(f"  group all:        {std['group_all_reach']*100:.0f}%")
    print(f"  mean score:       {std['mean_score']:.2f}")

    print("\n[2/4] MI panel (30 episodes, per-agent voice->food direction)...")
    mi = evaluate_mi_panel(agent_params, n_episodes=30, n_perm=100)
    print(f"  max gain: {mi['max_gain']:+.4f}  mean: {mi['mean_gain']:+.4f}")
    print(f"  {mi['n_significant']}/{N} agents with p<0.05")

    print("\n[3/4] ToM test (20 episodes, voice_i -> nav_j[t+1])...")
    tom = evaluate_tom(agent_params, n_episodes=20)
    print(f"  predictive pairs: {tom['n_pairs_predictive']}/"
          f"{tom['n_pairs_measured']} "
          f"({tom['pair_fraction_predictive']*100:.0f}%)")
    print(f"  max gain: {tom['max_gain']:+.4f}")

    print("\n[4/4] Scales test (N=5 params -> N=10 world)...")
    target_N = max(10, N + 5)
    sc = evaluate_scales(agent_params, target_N=target_N, n_seeds=10)
    print(f"  N={sc['target_N']}: mean reach {sc['mean_agent_reach_rate']*100:.0f}%")

    # Pre-registered criteria
    print(f"\n" + "=" * 70)
    print(f"  PRE-REGISTERED SUCCESS CRITERIA")
    print(f"=" * 70)
    c1 = std["mean_agent_reach_rate"] > 0.8
    c2 = mi["mean_gain"] > 0.08
    c3 = tom["pair_fraction_predictive"] > 0.45
    c4 = sc["mean_agent_reach_rate"] > 0.6
    criteria = [
        ("1. Trained reach > 80%", c1, f"{std['mean_agent_reach_rate']*100:.0f}%"),
        ("2. Mean MI gain > 0.08", c2, f"{mi['mean_gain']:+.4f}"),
        ("3. ToM predictive > 45%", c3,
         f"{tom['pair_fraction_predictive']*100:.0f}%"),
        ("4. Scales N->2x reach > 60%", c4,
         f"{sc['mean_agent_reach_rate']*100:.0f}%"),
    ]
    n_met = sum(1 for _, v, _ in criteria if v)
    for name, met, value in criteria:
        mark = "v" if met else "x"
        print(f"  [{mark}] {name}  actual: {value}")
    print(f"\n  MET {n_met}/4 criteria")
    if n_met >= 2:
        verdict = "PHASE 7 SUCCESS (>=2 criteria met)"
    else:
        verdict = "PHASE 7 INCOMPLETE (<2 criteria met)"
    print(f"  VERDICT: {verdict}")

    # Save
    out = {
        "n_agents": N,
        "standard": std,
        "mi_panel": mi,
        "tom": tom,
        "scales": sc,
        "criteria_met": n_met,
        "verdict": verdict,
    }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")
    return out


if __name__ == "__main__":
    # Smoke: evaluate midpoint brains (expected low scores)
    from kathara16_brain import PARAM_RANGES_16
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_16]
    mid_params = [mid] * 3  # N=3 for quick smoke
    run_full_evaluation(mid_params, save_path="phase7_smoke_eval.json")
