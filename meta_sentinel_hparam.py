"""meta_sentinel_hparam.py - Meta-Sentinel: Sentinel on Sentinel's hyperparameters.

Gödelian Level 3 (self-referential optimization):
  outer_Sentinel(eval=inner_sentinel_score, guard=stability)
  searching over (min_r_squared, collect_size_ratio)
  finds optimal hyperparameters that maximize inner Sentinel performance.

Inner task: simple FlyWorldV1 food-reach with 91D Kathara brain.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from kathara_brain_sim_v8 import (
    PARAM_RANGES as KATHARA_PR,
    PARAM_NAMES as KATHARA_PN,
    FlyWorldV1, simulate_step,
)


def make_simple_inner_eval(n_episodes=2, n_steps=30, seed_base=17):
    """Simple FlyWorldV1 reach-food eval. Used as inner Sentinel's eval_fn."""
    def eval_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=seed_base + ep * 7)
            sensors = world.reset()
            states = np.zeros(12)
            initial_d = world.get_food_dist()
            min_d = initial_d
            reached = False
            for _ in range(n_steps):
                states, firing = simulate_step(p, sensors, states)
                nav, cen = float(firing[5]), float(firing[11])
                sensors, d, reached_now, _ = world.step(nav, cen)
                min_d = min(min_d, d)
                if reached_now:
                    reached = True
                    break
            approach = max(0.0, initial_d - min_d) / (initial_d + 1e-6)
            total += approach * 70 + (30 if reached else 0)
        return total / n_episodes
    return eval_fn


def make_simple_inner_guard(n_episodes=1, n_steps=25):
    """Inner guard: movement activity."""
    def guard_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=100 + ep * 3)
            sensors = world.reset()
            states = np.zeros(12)
            move = 0.0
            for _ in range(n_steps):
                prev = world.fly_pos.copy()
                states, firing = simulate_step(p, sensors, states)
                nav, cen = float(firing[5]), float(firing[11])
                sensors, _, reached, _ = world.step(nav, cen)
                move += float(np.linalg.norm(world.fly_pos - prev))
                if reached: break
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def inner_sentinel_run(min_r_squared, learn, time_budget=15):
    """Run inner Sentinel with given hparams. Return best_ever_score."""
    eval_fn = make_simple_inner_eval()
    guard_fn = make_simple_inner_guard()
    # Random experience_id to prevent cross-contamination
    import random
    exp_id = f"meta_inner_{random.randint(0, 999999)}"
    result = Sentinel(
        eval_fn=eval_fn, guard_fn=guard_fn,
        param_ranges=KATHARA_PR, param_names=KATHARA_PN,
        experience_id=exp_id,
        learn=learn,
        min_r_squared=min_r_squared,
    ).run(time_budget=time_budget, verbose=False)
    return float(result.get("best_ever_score") or result.get("eval_score") or 0.0)


def meta_eval_fn(hparams):
    """Outer eval: single inner run (15s budget, actual ~60-90s due to
    autonomous loop extension). Noisy but trackable."""
    min_r2 = float(hparams[0])
    learn_val = float(hparams[1])  # >0.5 -> True
    learn = learn_val > 0.5
    if not (0.1 <= min_r2 <= 0.9):
        return 0.0
    try:
        return inner_sentinel_run(min_r2, learn, time_budget=8)
    except Exception as e:
        print(f"  [inner error] {e}")
        return 0.0


def meta_guard_fn(hparams):
    """Outer guard: just approve sensible regions (no crashes)."""
    min_r2 = float(hparams[0])
    if 0.1 <= min_r2 <= 0.9:
        return 50.0  # safe region
    return 0.0


def main():
    print("=" * 70)
    print("  META-SENTINEL: optimizing Sentinel's own hyperparameters")
    print("=" * 70)

    # Baseline: default hparams (min_r_squared=0.3, learn=False)
    print("\n[BASELINE] min_r_squared=0.3, learn=False, budget=8s/inner")
    t0 = time.time()
    baseline_scores = [inner_sentinel_run(0.3, False, time_budget=8)
                       for _ in range(3)]
    t_baseline = time.time() - t0
    baseline_mean = float(np.mean(baseline_scores))
    print(f"  baseline inner scores (3 runs): {[round(s, 2) for s in baseline_scores]}")
    print(f"  baseline mean: {baseline_mean:.2f}  (elapsed {t_baseline:.0f}s)")

    # Outer search
    print("\n[OUTER SEARCH] Sentinel searching (min_r_squared, learn) space")
    # Constrain outer: 2D search. learn encoded as continuous [0, 1].
    outer_ranges = [
        (0.1, 0.9),  # min_r_squared
        (0.0, 1.0),  # learn (>0.5 -> True)
    ]
    outer_names = ["inner_min_r_squared", "inner_learn"]

    t0 = time.time()
    outer_result = Sentinel(
        eval_fn=meta_eval_fn,
        guard_fn=meta_guard_fn,
        param_ranges=outer_ranges,
        param_names=outer_names,
        experience_id="meta_sentinel_hparam_outer",
        learn=True,
    ).run(time_budget=600, verbose=False)  # 10 min outer (inner runs are 30-90s each)
    t_outer = time.time() - t0

    best_hparams = outer_result.get("best_ever_params") or outer_result.get("best_params")
    best_meta_score = outer_result.get("best_ever_score") or outer_result.get("eval_score") or 0.0
    outer_verdict = outer_result.get("verdict")

    print(f"\n  Outer elapsed: {t_outer:.0f}s")
    print(f"  Best meta_eval score: {best_meta_score:.2f}")
    print(f"  Best hparams:")
    for n, v in zip(outer_names, best_hparams):
        print(f"    {n} = {v:.3f}")
    print(f"  Outer verdict: {outer_verdict}")

    # Decode best hparams
    best_min_r2 = float(best_hparams[0])
    best_learn = float(best_hparams[1]) > 0.5

    # Final validation: run 3 times with best hparams
    print(f"\n[VALIDATION] Running 3 inner runs with best hparams...")
    tuned_scores = [inner_sentinel_run(best_min_r2, best_learn, time_budget=8)
                    for _ in range(3)]
    tuned_mean = float(np.mean(tuned_scores))
    print(f"  tuned scores: {[round(s, 2) for s in tuned_scores]}")
    print(f"  tuned mean: {tuned_mean:.2f}")

    # Improvement
    improvement = tuned_mean - baseline_mean
    ratio = tuned_mean / max(baseline_mean, 1e-6)
    print(f"\n" + "=" * 70)
    print(f"  META-SENTINEL RESULT")
    print("=" * 70)
    print(f"  Baseline:  {baseline_mean:.2f}")
    print(f"  Tuned:     {tuned_mean:.2f}")
    print(f"  Improvement: {improvement:+.2f} ({ratio:.2f}x)")
    print(f"  Best min_r_squared: {best_min_r2:.3f}")
    print(f"  Best learn:         {best_learn}")

    if ratio > 1.1:
        verdict = "META-WORKS: tuned > baseline by >10%"
    elif ratio > 1.02:
        verdict = "META-MARGINAL: small improvement"
    elif ratio > 0.95:
        verdict = "META-NEUTRAL: no meaningful change"
    else:
        verdict = "META-HURT: outer search found worse hparams (noise?)"
    print(f"  Verdict: {verdict}")

    # Save
    out = {
        "baseline": {
            "hparams": {"min_r_squared": 0.3, "learn": False},
            "inner_scores": [round(s, 2) for s in baseline_scores],
            "mean": round(baseline_mean, 2),
        },
        "tuned": {
            "hparams": {"min_r_squared": round(best_min_r2, 3), "learn": best_learn},
            "inner_scores": [round(s, 2) for s in tuned_scores],
            "mean": round(tuned_mean, 2),
        },
        "improvement_delta": round(improvement, 2),
        "improvement_ratio": round(ratio, 3),
        "outer_total_elapsed_s": round(t_outer, 1),
        "outer_verdict": outer_verdict,
        "verdict": verdict,
    }
    with open("meta_sentinel_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: meta_sentinel_result.json")


if __name__ == "__main__":
    main()
