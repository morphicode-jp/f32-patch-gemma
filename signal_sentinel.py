"""signal_sentinel.py - Phase 3 MVP: co-evolve agents with voice signals.

Motor mapping:
  nav   = firing[5]
  speed = firing[11]
  voice = firing[1]   <- NEW: 0-1 continuous signal emitted each step

After co-evolution, measure mutual information:
  I(agent_A.voice_t ; agent_A.food_direction_t)
  If > chance, signals carry information -> proto-language seeded.
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


def run_match(params_a, params_b, seed, n_steps=40, record=False):
    pa = np.array(params_a, dtype=np.float64)
    pb = np.array(params_b, dtype=np.float64)
    world = SignalFlyWorld(seed=seed)
    s0, s1 = world.reset()
    initial_d0 = world.get_food_dist(0)
    initial_d1 = world.get_food_dist(1)
    min_d0, min_d1 = initial_d0, initial_d1
    move0 = move1 = 0.0
    sa = np.zeros(12); sb = np.zeros(12)
    winner = None
    traces = {"a_voice": [], "a_dir": [], "b_voice": [], "b_dir": []} if record else None

    for _ in range(n_steps):
        prev0 = world.agent_pos[0].copy()
        prev1 = world.agent_pos[1].copy()
        if record:
            traces["a_dir"].append(world.get_food_direction(0))
            traces["b_dir"].append(world.get_food_direction(1))
        sa, fa = simulate_step(pa, s0, sa)
        sb, fb = simulate_step(pb, s1, sb)
        nav0, cen0, voice0 = float(fa[5]), float(fa[11]), float(fa[1])
        nav1, cen1, voice1 = float(fb[5]), float(fb[11]), float(fb[1])
        if record:
            traces["a_voice"].append(voice0)
            traces["b_voice"].append(voice1)
        s0, s1, done, winner = world.step(nav0, cen0, voice0, nav1, cen1, voice1)
        move0 += float(np.linalg.norm(world.agent_pos[0] - prev0))
        move1 += float(np.linalg.norm(world.agent_pos[1] - prev1))
        min_d0 = min(min_d0, world.get_food_dist(0))
        min_d1 = min(min_d1, world.get_food_dist(1))
        if done: break

    approach0 = max(0.0, initial_d0 - min_d0) / (initial_d0 + 1e-6)
    approach1 = max(0.0, initial_d1 - min_d1) / (initial_d1 + 1e-6)
    score0 = (60.0 if winner == 0 else 0.0) + 30.0 * approach0 + min(10.0, move0 * 2.0)
    score1 = (60.0 if winner == 1 else 0.0) + 30.0 * approach1 + min(10.0, move1 * 2.0)
    if record:
        return score0, score1, traces
    return score0, score1


def make_eval_fn(opponent_params, is_agent_0, n_episodes=3):
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            seed = ep * 7 + 13
            if is_agent_0:
                s0, s1 = run_match(params, opponent_params, seed)
                total += s0
            else:
                s0, s1 = run_match(opponent_params, params, seed)
                total += s1
        return total / n_episodes
    return eval_fn


def make_guard_fn(opponent_params, is_agent_0, n_episodes=2):
    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            seed = ep * 11 + 5
            pa = np.array(params if is_agent_0 else opponent_params)
            pb = np.array(opponent_params if is_agent_0 else params)
            world = SignalFlyWorld(seed=seed)
            s0, s1 = world.reset()
            sa = np.zeros(12); sb = np.zeros(12)
            move = 0.0
            for _ in range(30):
                prev = world.agent_pos[0 if is_agent_0 else 1].copy()
                sa, fa = simulate_step(pa, s0, sa)
                sb, fb = simulate_step(pb, s1, sb)
                s0, s1, done, _ = world.step(float(fa[5]), float(fa[11]), float(fa[1]),
                                              float(fb[5]), float(fb[11]), float(fb[1]))
                curr = world.agent_pos[0 if is_agent_0 else 1]
                move += float(np.linalg.norm(curr - prev))
                if done: break
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def mutual_information(signals, labels, n_bins=4):
    """MI between continuous signal and categorical label. Returns bits."""
    if len(signals) < 10: return 0.0
    labels_unique = list(set(labels))
    # Bin signals
    sig_arr = np.array(signals)
    edges = np.linspace(sig_arr.min() - 1e-6, sig_arr.max() + 1e-6, n_bins + 1)
    bins = np.digitize(sig_arr, edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    # Joint distribution
    joint = np.zeros((n_bins, len(labels_unique)))
    for b, lbl in zip(bins, labels):
        j = labels_unique.index(lbl)
        joint[b, j] += 1
    joint /= joint.sum()

    # Marginals
    p_s = joint.sum(axis=1)
    p_l = joint.sum(axis=0)

    # MI
    mi = 0.0
    for b in range(n_bins):
        for j in range(len(labels_unique)):
            if joint[b, j] > 0 and p_s[b] > 0 and p_l[j] > 0:
                mi += joint[b, j] * np.log2(joint[b, j] / (p_s[b] * p_l[j]))
    return float(mi)


def measure_signal_information(params_a, params_b, n_episodes=30):
    """Run episodes and compute MI between voice and food direction."""
    all_a_voice, all_a_dir = [], []
    all_b_voice, all_b_dir = [], []
    for ep in range(n_episodes):
        _, _, traces = run_match(params_a, params_b, seed=1000 + ep, record=True)
        all_a_voice.extend(traces["a_voice"])
        all_a_dir.extend(traces["a_dir"])
        all_b_voice.extend(traces["b_voice"])
        all_b_dir.extend(traces["b_dir"])

    mi_a = mutual_information(all_a_voice, all_a_dir)
    mi_b = mutual_information(all_b_voice, all_b_dir)

    # Shuffled baseline (null hypothesis)
    import random
    shuffled_dir_a = list(all_a_dir); random.Random(42).shuffle(shuffled_dir_a)
    shuffled_dir_b = list(all_b_dir); random.Random(43).shuffle(shuffled_dir_b)
    mi_a_null = mutual_information(all_a_voice, shuffled_dir_a)
    mi_b_null = mutual_information(all_b_voice, shuffled_dir_b)

    return {
        "mi_a_voice_to_dir": mi_a,
        "mi_b_voice_to_dir": mi_b,
        "mi_a_null": mi_a_null,
        "mi_b_null": mi_b_null,
        "a_voice_stats": {"mean": float(np.mean(all_a_voice)),
                          "std": float(np.std(all_a_voice))},
        "b_voice_stats": {"mean": float(np.mean(all_b_voice)),
                          "std": float(np.std(all_b_voice))},
        "n_samples": len(all_a_voice),
    }


def co_evolve_signal(n_generations=3, gen_budget=75, verbose=True):
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    params_a = list(mid); params_b = list(mid)
    history = []
    t_start = time.time()

    # Baseline MI measurement (before any training)
    print("\n[baseline] Measuring pre-training signal informativeness...")
    pre_mi = measure_signal_information(params_a, params_b, n_episodes=10)
    print(f"  A: MI(voice;dir)={pre_mi['mi_a_voice_to_dir']:.4f} bits  "
          f"(null={pre_mi['mi_a_null']:.4f})")
    print(f"  B: MI(voice;dir)={pre_mi['mi_b_voice_to_dir']:.4f} bits  "
          f"(null={pre_mi['mi_b_null']:.4f})")

    for gen in range(n_generations):
        target = gen % 2
        opponent = params_b if target == 0 else params_a
        if verbose:
            print(f"\n{'='*70}\n  Gen {gen+1}/{n_generations}: optimizing Agent {target}\n{'='*70}")

        eval_fn = make_eval_fn(opponent, is_agent_0=(target == 0), n_episodes=3)
        guard_fn = make_guard_fn(opponent, is_agent_0=(target == 0), n_episodes=2)
        initial = params_a if target == 0 else params_b

        result = Sentinel(
            eval_fn=eval_fn, guard_fn=guard_fn,
            param_ranges=PARAM_RANGES, param_names=PARAM_NAMES,
            experience_id=f"signal_agent_{target}",
            initial_params=initial, learn=True,
        ).run(time_budget=gen_budget, verbose=False)

        best = result.get("best_ever_params") or result.get("best_params")
        if best is not None:
            if target == 0: params_a = list(best)
            else: params_b = list(best)

        gen_elapsed = time.time() - t_start
        print(f"  eval={result.get('best_ever_score', 0):.2f} "
              f"guard={result.get('guard_score', 0):.2f} "
              f"total_elapsed={gen_elapsed:.0f}s")
        history.append({
            "gen": gen + 1, "target": target,
            "eval": result.get("best_ever_score"),
            "guard": result.get("guard_score"),
            "elapsed": round(gen_elapsed, 1),
        })

    # Post-training MI measurement
    print("\n[post-training] Measuring signal informativeness...")
    post_mi = measure_signal_information(params_a, params_b, n_episodes=20)
    print(f"  A: MI(voice;dir)={post_mi['mi_a_voice_to_dir']:.4f} bits  "
          f"(null={post_mi['mi_a_null']:.4f})")
    print(f"  B: MI(voice;dir)={post_mi['mi_b_voice_to_dir']:.4f} bits  "
          f"(null={post_mi['mi_b_null']:.4f})")

    # Verdict
    signal_gain_a = post_mi['mi_a_voice_to_dir'] - post_mi['mi_a_null']
    signal_gain_b = post_mi['mi_b_voice_to_dir'] - post_mi['mi_b_null']
    print(f"\n{'='*70}\n  Signal emergence verdict\n{'='*70}")
    print(f"  Agent A signal info gain: {signal_gain_a:+.4f} bits")
    print(f"  Agent B signal info gain: {signal_gain_b:+.4f} bits")
    if max(signal_gain_a, signal_gain_b) > 0.05:
        verdict = "PROTO-LANGUAGE EMERGED (at least one agent uses voice meaningfully)"
    elif max(signal_gain_a, signal_gain_b) > 0.01:
        verdict = "WEAK SIGNAL EMERGENCE (above noise but not strong)"
    else:
        verdict = "NO EMERGENCE (voice is just noise)"
    print(f"  Verdict: {verdict}")

    return params_a, params_b, history, pre_mi, post_mi, verdict


def main():
    print("=" * 70)
    print("  Phase 3 MVP: Voice-enabled co-evolution (proto-language check)")
    print("=" * 70)

    pa, pb, history, pre_mi, post_mi, verdict = co_evolve_signal(
        n_generations=3, gen_budget=75, verbose=True)

    with open("signal_result.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "history": history,
            "pre_mi": pre_mi, "post_mi": post_mi,
            "final_params_a": [float(x) for x in pa],
            "final_params_b": [float(x) for x in pb],
        }, f, indent=2)
    print("\n  Saved: signal_result.json")


if __name__ == "__main__":
    main()
