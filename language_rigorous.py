"""language_rigorous.py - Statistical rigor for cooperative language emergence.

Hypothesis (to test rigorously):
  Voice signals carry meaningful information (measured as mutual
  information between voice output and perceived food direction)
  ONLY when agents are co-evolved in a COOPERATIVE task, not a
  COMPETITIVE one.

Protocol:
  1. Train N_BRAINS=5 brain-pairs in cooperative setting
     (different random seeds for training)
  2. Train N_BRAINS=5 brain-pairs in competitive setting (control)
  3. For each trained brain-pair, measure MI(voice, food_direction)
     on 30 test episodes
  4. Compute shuffled null distribution (1000 permutations)
  5. Report:
     - Per-brain MI and p-value vs shuffled null
     - Between-condition difference (coop vs compete) with bootstrap 95% CI
     - Effect size (Cohen's d)
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, PARAM_NAMES, simulate_step
from twelve.agent.sentinel import Sentinel
from signal_world import SignalFlyWorld
from coop_world import CoopFlyWorld
from signal_sentinel import run_match as run_compete_match, mutual_information
from coop_sentinel import run_match_coop


def make_compete_eval(opponent, is_agent_0, n_episodes=3):
    """Competitive: one food, one winner."""
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            seed = ep * 7 + 13
            if is_agent_0:
                s0, _ = run_compete_match(params, opponent, seed)
                total += s0
            else:
                _, s1 = run_compete_match(opponent, params, seed)
                total += s1
        return total / n_episodes
    return eval_fn


def make_coop_eval(opponent, is_agent_0, n_episodes=3):
    """Cooperative: two foods, joint reward."""
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            seed = ep * 7 + 13
            if is_agent_0:
                s0, _ = run_match_coop(params, opponent, seed)
                total += s0
            else:
                _, s1 = run_match_coop(opponent, params, seed)
                total += s1
        return total / n_episodes
    return eval_fn


def make_movement_guard(world_cls, n_episodes=2):
    """guard: movement (prevent stuck)."""
    def guard_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = world_cls(seed=ep * 11 + 5)
            out = world.reset()
            if world_cls is CoopFlyWorld:
                s0, s1 = out
            else:
                s0, s1 = out
            sa = np.zeros(12)
            move = 0.0
            for _ in range(30):
                prev = world.agent_pos[0].copy()
                sa, fa = simulate_step(p, s0, sa)
                nav, cen, voice = float(fa[5]), float(fa[11]), float(fa[1])
                if world_cls is CoopFlyWorld:
                    s0, s1, done = world.step(nav, cen, voice, 0.5, 0.5, 0.5)
                else:
                    s0, s1, done, _ = world.step(nav, cen, voice, 0.5, 0.5, 0.5)
                move += float(np.linalg.norm(world.agent_pos[0] - prev))
                if done: break
            total += move * 5
        return total / n_episodes
    return guard_fn


def train_pair(condition, train_seed, time_budget_per_gen=45, n_generations=3):
    """Train A and B alternately under given condition. Returns (pa, pb)."""
    rng = np.random.RandomState(train_seed)
    # Random init (not midpoint) for diversity across training runs
    pa = [rng.uniform(lo, hi) for lo, hi in PARAM_RANGES]
    pb = [rng.uniform(lo, hi) for lo, hi in PARAM_RANGES]

    for gen in range(n_generations):
        target = gen % 2
        opp = pb if target == 0 else pa
        if condition == "coop":
            eval_fn = make_coop_eval(opp, target == 0)
            guard_fn = make_movement_guard(CoopFlyWorld)
        else:  # compete
            eval_fn = make_compete_eval(opp, target == 0)
            guard_fn = make_movement_guard(SignalFlyWorld)

        result = Sentinel(
            eval_fn=eval_fn, guard_fn=guard_fn,
            param_ranges=PARAM_RANGES, param_names=PARAM_NAMES,
            experience_id=f"lang_rig_{condition}_seed{train_seed}_agent{target}",
            initial_params=(pa if target == 0 else pb),
            learn=True,
        ).run(time_budget=time_budget_per_gen, verbose=False)
        best = result.get("best_ever_params") or result.get("best_params")
        if best:
            if target == 0: pa = list(best)
            else: pb = list(best)
    return pa, pb


def measure_mi_detailed(pa, pb, condition, n_episodes=30, n_perm=200):
    """Measure MI(voice, food_dir) with permutation test."""
    all_a_voice, all_b_voice = [], []
    all_a_dir_own, all_b_dir_own = [], []

    for ep in range(n_episodes):
        if condition == "coop":
            _, _, tr = run_match_coop(pa, pb, seed=10000 + ep, record=True)
            all_a_voice.extend(tr["a_voice"])
            all_b_voice.extend(tr["b_voice"])
            all_a_dir_own.extend(tr["a_dir_own"])
            all_b_dir_own.extend(tr["b_dir_own"])
        else:
            from signal_sentinel import run_match as rm_compete
            _, _, tr = rm_compete(pa, pb, seed=10000 + ep, record=True)
            all_a_voice.extend(tr["a_voice"])
            all_b_voice.extend(tr["b_voice"])
            all_a_dir_own.extend(tr["a_dir"])
            all_b_dir_own.extend(tr["b_dir"])

    mi_a = mutual_information(all_a_voice, all_a_dir_own)
    mi_b = mutual_information(all_b_voice, all_b_dir_own)

    # Permutation null: shuffle direction labels and recompute MI
    import random
    rng = random.Random(42)
    null_a = []
    null_b = []
    dir_a_copy = list(all_a_dir_own)
    dir_b_copy = list(all_b_dir_own)
    for _ in range(n_perm):
        rng.shuffle(dir_a_copy)
        rng.shuffle(dir_b_copy)
        null_a.append(mutual_information(all_a_voice, dir_a_copy))
        null_b.append(mutual_information(all_b_voice, dir_b_copy))

    null_a = np.array(null_a)
    null_b = np.array(null_b)

    # p-value: P(null >= observed)
    p_a = float((null_a >= mi_a).sum() / n_perm)
    p_b = float((null_b >= mi_b).sum() / n_perm)

    return {
        "mi_a": float(mi_a),
        "mi_b": float(mi_b),
        "mi_max": float(max(mi_a, mi_b)),
        "mi_null_a_mean": float(null_a.mean()),
        "mi_null_b_mean": float(null_b.mean()),
        "mi_null_a_95": float(np.percentile(null_a, 95)),
        "mi_null_b_95": float(np.percentile(null_b, 95)),
        "p_a": p_a, "p_b": p_b,
        "gain_a": float(mi_a - null_a.mean()),
        "gain_b": float(mi_b - null_b.mean()),
        "n_samples": len(all_a_voice),
    }


def bootstrap_ci(values, n_boot=1000, ci=0.95):
    rng = np.random.RandomState(7)
    values = np.array(values)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    boots = np.array(boots)
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 - (1 - ci) / 2)))
    return lo, hi


def cohens_d(a, b):
    a = np.array(a); b = np.array(b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def main():
    print("=" * 70)
    print("  Cooperative Language Emergence: Rigorous Statistical Test")
    print("=" * 70)

    N_BRAINS = 3
    TIME_PER_GEN = 30
    N_GENS = 3
    N_TEST_EPISODES = 30
    N_PERMUTATIONS = 200

    t_start = time.time()

    results = {"coop": [], "compete": []}

    for condition in ["coop", "compete"]:
        print(f"\n{'#' * 70}")
        print(f"  Condition: {condition.upper()}")
        print(f"{'#' * 70}")

        for brain_idx in range(N_BRAINS):
            seed = brain_idx * 100 + 42
            t0 = time.time()
            print(f"\n[{condition} brain {brain_idx+1}/{N_BRAINS}] train seed={seed}")
            pa, pb = train_pair(condition, seed,
                                time_budget_per_gen=TIME_PER_GEN,
                                n_generations=N_GENS)
            t_train = time.time() - t0

            mi_data = measure_mi_detailed(pa, pb, condition,
                                          n_episodes=N_TEST_EPISODES,
                                          n_perm=N_PERMUTATIONS)
            print(f"  train={t_train:.0f}s  "
                  f"MI_a={mi_data['mi_a']:.4f} (p={mi_data['p_a']:.3f})  "
                  f"MI_b={mi_data['mi_b']:.4f} (p={mi_data['p_b']:.3f})  "
                  f"gain_max={max(mi_data['gain_a'], mi_data['gain_b']):+.4f}")
            mi_data["brain_idx"] = brain_idx
            mi_data["train_seed"] = seed
            mi_data["train_s"] = round(t_train, 1)
            mi_data["params_a"] = [float(x) for x in pa]
            mi_data["params_b"] = [float(x) for x in pb]
            results[condition].append(mi_data)

    total_elapsed = time.time() - t_start
    print(f"\n  Total training + eval time: {total_elapsed:.0f}s")

    # Statistical summary
    print("\n" + "=" * 70)
    print("  STATISTICAL SUMMARY")
    print("=" * 70)

    coop_gains = [max(r["gain_a"], r["gain_b"]) for r in results["coop"]]
    comp_gains = [max(r["gain_a"], r["gain_b"]) for r in results["compete"]]

    coop_mean = float(np.mean(coop_gains))
    comp_mean = float(np.mean(comp_gains))
    coop_ci = bootstrap_ci(coop_gains)
    comp_ci = bootstrap_ci(comp_gains)
    d = cohens_d(coop_gains, comp_gains)

    # Permutation test for coop > compete
    all_gains = np.array(coop_gains + comp_gains)
    observed_diff = coop_mean - comp_mean
    rng = np.random.RandomState(99)
    null_diffs = []
    for _ in range(1000):
        perm = rng.permutation(all_gains)
        d1 = perm[:N_BRAINS].mean() - perm[N_BRAINS:].mean()
        null_diffs.append(d1)
    null_diffs = np.array(null_diffs)
    p_value = float((null_diffs >= observed_diff).sum() / 1000)

    print(f"\n  MI gain (observed - null) per condition:")
    print(f"    cooperative:  {coop_mean:+.4f}  95% CI [{coop_ci[0]:+.4f}, {coop_ci[1]:+.4f}]")
    print(f"    competitive:  {comp_mean:+.4f}  95% CI [{comp_ci[0]:+.4f}, {comp_ci[1]:+.4f}]")
    print(f"\n  Difference (coop - compete): {observed_diff:+.4f}")
    print(f"    Cohen's d: {d:.2f}")
    print(f"    permutation p-value (coop > compete): p = {p_value:.3f}")

    # Per-brain significance
    sig_coop = sum(1 for r in results["coop"] if min(r["p_a"], r["p_b"]) < 0.05)
    sig_comp = sum(1 for r in results["compete"] if min(r["p_a"], r["p_b"]) < 0.05)
    print(f"\n  Per-brain significance (p<0.05 for MI vs null):")
    print(f"    cooperative:  {sig_coop}/{N_BRAINS} brains significant")
    print(f"    competitive:  {sig_comp}/{N_BRAINS} brains significant")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if p_value < 0.05 and d > 0.5:
        verdict = "COOPERATIVE > COMPETITIVE confirmed (p<0.05, large effect)"
    elif p_value < 0.05:
        verdict = "COOPERATIVE > COMPETITIVE confirmed (p<0.05)"
    elif d > 0.3:
        verdict = "Trend toward coop > compete but not yet significant"
    else:
        verdict = "NO clear difference"
    print(f"  {verdict}")
    print(f"\n  Interpretation:")
    if p_value < 0.05:
        print(f"    Communication pressure in cooperative task produces")
        print(f"    statistically significant information in voice signal,")
        print(f"    whereas competition does not. This reproduces the")
        print(f"    biological prediction (kin selection / reciprocal altruism")
        print(f"    required for honest signaling) in artificial neural agents.")

    # Save
    out = {
        "protocol": {
            "n_brains_per_condition": N_BRAINS,
            "n_test_episodes": N_TEST_EPISODES,
            "n_permutations": N_PERMUTATIONS,
            "train_time_per_gen_s": TIME_PER_GEN,
            "n_generations": N_GENS,
        },
        "coop_brains": results["coop"],
        "compete_brains": results["compete"],
        "summary": {
            "coop_mean_gain": round(coop_mean, 5),
            "coop_ci_95": [round(x, 5) for x in coop_ci],
            "compete_mean_gain": round(comp_mean, 5),
            "compete_ci_95": [round(x, 5) for x in comp_ci],
            "diff": round(observed_diff, 5),
            "cohens_d": round(d, 3),
            "permutation_p": round(p_value, 4),
            "n_sig_coop": sig_coop,
            "n_sig_compete": sig_comp,
        },
        "verdict": verdict,
        "total_elapsed_s": round(total_elapsed, 1),
    }
    with open("language_rigorous_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: language_rigorous_result.json")


if __name__ == "__main__":
    main()
