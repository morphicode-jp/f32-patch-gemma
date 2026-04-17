"""coop_sentinel.py - Cooperative co-evolution with voice signals.

Scoring (for each agent):
  own_approach_ratio * 60           (reached own food)
  + partner_approach_ratio * 40     (partner reached their food)
  + 30 if both reached

Selection pressure: helping partner find their food improves own score.
Expected: voice signals evolve to encode useful info (e.g., partner's
  food direction relative to speaker).

MI measurement (more sophisticated):
  I(agent_A_voice ; direction_of_B_food_from_A)
  If A can see B's food (indirectly through B's proximity), it might
  learn to broadcast that direction to help B navigate.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, PARAM_NAMES, simulate_step
from twelve.agent.sentinel import Sentinel
from coop_world import CoopFlyWorld
from signal_sentinel import mutual_information


def run_match_coop(pa, pb, seed, n_steps=50, record=False):
    pa = np.array(pa, dtype=np.float64)
    pb = np.array(pb, dtype=np.float64)
    world = CoopFlyWorld(seed=seed)
    s0, s1 = world.reset()
    init_d0 = world.get_food_dist(0)
    init_d1 = world.get_food_dist(1)
    min_d0, min_d1 = init_d0, init_d1
    sa = np.zeros(12); sb = np.zeros(12)
    move0 = move1 = 0.0
    traces = {"a_voice": [], "b_voice": [],
              "a_dir_own": [], "a_dir_b": [],
              "b_dir_own": [], "b_dir_a": []} if record else None

    for _ in range(n_steps):
        prev0 = world.agent_pos[0].copy()
        prev1 = world.agent_pos[1].copy()
        if record:
            traces["a_dir_own"].append(world.get_food_direction_categorical(0, for_food_of=0))
            traces["a_dir_b"].append(world.get_food_direction_categorical(0, for_food_of=1))
            traces["b_dir_own"].append(world.get_food_direction_categorical(1, for_food_of=1))
            traces["b_dir_a"].append(world.get_food_direction_categorical(1, for_food_of=0))
        sa, fa = simulate_step(pa, s0, sa)
        sb, fb = simulate_step(pb, s1, sb)
        nav0, cen0, voice0 = float(fa[5]), float(fa[11]), float(fa[1])
        nav1, cen1, voice1 = float(fb[5]), float(fb[11]), float(fb[1])
        if record:
            traces["a_voice"].append(voice0)
            traces["b_voice"].append(voice1)
        s0, s1, done = world.step(nav0, cen0, voice0, nav1, cen1, voice1)
        move0 += float(np.linalg.norm(world.agent_pos[0] - prev0))
        move1 += float(np.linalg.norm(world.agent_pos[1] - prev1))
        min_d0 = min(min_d0, world.get_food_dist(0))
        min_d1 = min(min_d1, world.get_food_dist(1))
        if done: break

    app0 = max(0.0, init_d0 - min_d0) / (init_d0 + 1e-6)
    app1 = max(0.0, init_d1 - min_d1) / (init_d1 + 1e-6)
    reached_both = world.reached[0] and world.reached[1]
    # Cooperative scoring
    score0 = app0 * 60 + app1 * 40 + (30 if reached_both else 0) + min(5.0, move0)
    score1 = app1 * 60 + app0 * 40 + (30 if reached_both else 0) + min(5.0, move1)
    if record:
        return score0, score1, traces
    return score0, score1


def make_eval(opponent, is_a, n_ep=3):
    def eval_fn(p):
        total = 0.0
        for ep in range(n_ep):
            seed = ep * 7 + 13
            if is_a:
                s0, _ = run_match_coop(p, opponent, seed)
                total += s0
            else:
                _, s1 = run_match_coop(opponent, p, seed)
                total += s1
        return total / n_ep
    return eval_fn


def make_guard(opponent, is_a, n_ep=2):
    def guard_fn(p):
        total = 0.0
        for ep in range(n_ep):
            seed = ep * 11 + 5
            pa = np.array(p if is_a else opponent)
            pb = np.array(opponent if is_a else p)
            world = CoopFlyWorld(seed=seed)
            s0, s1 = world.reset()
            sa = np.zeros(12); sb = np.zeros(12)
            move = 0.0
            for _ in range(30):
                prev = world.agent_pos[0 if is_a else 1].copy()
                sa, fa = simulate_step(pa, s0, sa)
                sb, fb = simulate_step(pb, s1, sb)
                s0, s1, done = world.step(float(fa[5]), float(fa[11]), float(fa[1]),
                                          float(fb[5]), float(fb[11]), float(fb[1]))
                curr = world.agent_pos[0 if is_a else 1]
                move += float(np.linalg.norm(curr - prev))
                if done: break
            total += move * 5
        return total / n_ep
    return guard_fn


def measure_mi_coop(pa, pb, n_episodes=25):
    """MI: A's voice -> B's food direction (should encode partner help)."""
    all_a_voice, all_b_voice = [], []
    all_a_dir_own, all_a_dir_b = [], []
    all_b_dir_own, all_b_dir_a = [], []

    for ep in range(n_episodes):
        _, _, tr = run_match_coop(pa, pb, seed=1000 + ep, record=True)
        all_a_voice.extend(tr["a_voice"])
        all_b_voice.extend(tr["b_voice"])
        all_a_dir_own.extend(tr["a_dir_own"])
        all_a_dir_b.extend(tr["a_dir_b"])
        all_b_dir_own.extend(tr["b_dir_own"])
        all_b_dir_a.extend(tr["b_dir_a"])

    # Key MI: how much does A's voice tell about A's own food position?
    # (Lower-order signaling, still useful for B to hear)
    mi_a_own = mutual_information(all_a_voice, all_a_dir_own)
    # And: how much does A's voice tell about B's food direction?
    # (A cannot directly see B's food; this would require relay)
    mi_a_bfood = mutual_information(all_a_voice, all_a_dir_b)
    mi_b_own = mutual_information(all_b_voice, all_b_dir_own)
    mi_b_afood = mutual_information(all_b_voice, all_b_dir_a)

    # Null baseline
    import random
    sh = list(all_a_dir_own); random.Random(42).shuffle(sh)
    mi_a_null = mutual_information(all_a_voice, sh)
    sh = list(all_b_dir_own); random.Random(43).shuffle(sh)
    mi_b_null = mutual_information(all_b_voice, sh)

    return {
        "mi_a_voice_to_own_food": mi_a_own,
        "mi_a_voice_to_b_food":   mi_a_bfood,
        "mi_b_voice_to_own_food": mi_b_own,
        "mi_b_voice_to_a_food":   mi_b_afood,
        "mi_a_null": mi_a_null,
        "mi_b_null": mi_b_null,
        "a_voice_mean": float(np.mean(all_a_voice)),
        "b_voice_mean": float(np.mean(all_b_voice)),
        "n_samples": len(all_a_voice),
    }


def main():
    print("=" * 70)
    print("  Phase 3.5 MVP: COOPERATIVE voice co-evolution")
    print("=" * 70)

    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    pa = list(mid); pb = list(mid)

    print("\n[baseline MI]")
    pre = measure_mi_coop(pa, pb, n_episodes=10)
    print(f"  A voice -> own food: {pre['mi_a_voice_to_own_food']:.4f} bits "
          f"(null={pre['mi_a_null']:.4f})")
    print(f"  B voice -> own food: {pre['mi_b_voice_to_own_food']:.4f} bits "
          f"(null={pre['mi_b_null']:.4f})")

    history = []
    t0 = time.time()
    for gen in range(3):
        target = gen % 2
        opp = pb if target == 0 else pa
        print(f"\n=== Gen {gen+1}: optimizing Agent {target} ===")
        eval_fn = make_eval(opp, target == 0)
        guard_fn = make_guard(opp, target == 0)
        initial = pa if target == 0 else pb
        result = Sentinel(
            eval_fn=eval_fn, guard_fn=guard_fn,
            param_ranges=PARAM_RANGES, param_names=PARAM_NAMES,
            experience_id=f"coop_agent_{target}",
            initial_params=initial, learn=True,
        ).run(time_budget=60, verbose=False)
        best = result.get("best_ever_params") or result.get("best_params")
        if best:
            if target == 0: pa = list(best)
            else: pb = list(best)
        elapsed = time.time() - t0
        print(f"  eval={result.get('best_ever_score', 0):.2f} "
              f"guard={result.get('guard_score', 0):.2f} total={elapsed:.0f}s")
        history.append({
            "gen": gen+1, "target": target,
            "eval": result.get("best_ever_score"),
            "elapsed": round(elapsed, 1),
        })

    print("\n[post-training MI]")
    post = measure_mi_coop(pa, pb, n_episodes=25)
    print(f"  A voice -> own food: {post['mi_a_voice_to_own_food']:.4f} bits "
          f"(null={post['mi_a_null']:.4f})  "
          f"gain={post['mi_a_voice_to_own_food']-post['mi_a_null']:+.4f}")
    print(f"  B voice -> own food: {post['mi_b_voice_to_own_food']:.4f} bits "
          f"(null={post['mi_b_null']:.4f})  "
          f"gain={post['mi_b_voice_to_own_food']-post['mi_b_null']:+.4f}")

    gain_a = post['mi_a_voice_to_own_food'] - post['mi_a_null']
    gain_b = post['mi_b_voice_to_own_food'] - post['mi_b_null']
    max_gain = max(gain_a, gain_b)

    print(f"\n{'='*70}")
    print("  Cooperative signal emergence")
    print("=" * 70)
    if max_gain > 0.1:
        verdict = "STRONG EMERGENCE"
    elif max_gain > 0.05:
        verdict = "PROTO-LANGUAGE (above noise)"
    elif max_gain > 0.01:
        verdict = "WEAK SIGNAL"
    else:
        verdict = "NO EMERGENCE"
    print(f"  max MI gain = {max_gain:.4f} bits  -> {verdict}")
    print(f"  (Phase 3 competitive: max gain = +0.0062 bits)")

    with open("coop_signal_result.json", "w") as f:
        json.dump({
            "verdict": verdict,
            "max_gain_bits": round(max_gain, 4),
            "history": history,
            "pre_mi": pre,
            "post_mi": post,
            "final_params_a": [float(x) for x in pa],
            "final_params_b": [float(x) for x in pb],
        }, f, indent=2)
    print("\n  Saved: coop_signal_result.json")


if __name__ == "__main__":
    main()
