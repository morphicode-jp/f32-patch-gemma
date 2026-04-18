"""dual_memory_staged.py - Sentinel徹底活用版 Phase 6.

Strategy: exploit Sentinel's accumulative learning (experience_id + fossil_record
+ warm_start) via STAGED optimization on a much smaller parameter space.

Phase 6 original: 186D single run → 15% reach (too large for Sentinel budget).
Staged approach:
  Stage 1: Freeze main brain (Phase 5b best, 91D, reach 30%)
           Optimize ONLY hippo + coupling = 95D (half the size)
  Stage 2: Run Sentinel AGAIN on same experience_id (fossil record grows,
           warm-start improves)
  Stage 3: Run once more for convergence

Key leveraging of Sentinel features:
  1. experience_id="dual_staged_hippo" → fossil persists across runs
  2. learn=True → strategy weights adapt
  3. initial_params = Phase 6 hippo + coupling → warm start
  4. Multiple invocations = accumulated experience
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from kathara_brain_sim_v8 import PARAM_RANGES as KATHARA_PR
from memory_world import MemoryFlyWorld
from dual_brain import (
    DUAL_PARAM_RANGES, DUAL_TOTAL, N_MAIN, N_HIPPO,
    split_params, dual_step
)
from hippocampus_brain import replay_patterns
from dual_memory_sentinel import (
    run_dual_episode, evaluate_standard, evaluate_persistent_food,
    evaluate_no_memory_ablation,
)

# The hippo+coupling subspace: hippo (91D) + coupling (4D) = 95D
HIPPO_PART_RANGES = list(KATHARA_PR) + DUAL_PARAM_RANGES[N_MAIN + N_HIPPO:]
HIPPO_PART_NAMES = (
    [f"hippo_{i}" for i in range(N_HIPPO)] +
    ["m_to_h_gain", "h_to_m_gain", "k_wta", "persist_decay"]
)
SUBSPACE_D = len(HIPPO_PART_RANGES)  # = 95


def load_frozen_main():
    """Load the Phase 5b 3-factor best main brain (91D, reach 30%)."""
    path = "rgated_memory_result.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run Phase 5b first")
    with open(path) as f:
        d = json.load(f)
    p = d.get("best_params")
    assert len(p) == N_MAIN, f"expected {N_MAIN}D main brain, got {len(p)}"
    return list(p)


def make_dual_from_subspace(frozen_main, hippo_and_coupling):
    """Compose full 186D dual params from frozen main + optimized hippo+coupling."""
    return list(frozen_main) + list(hippo_and_coupling)


def make_eval_fn_staged(frozen_main, n_trials=3, persistent_trials=3):
    """eval_fn on 95D subspace. Main brain frozen. Tests cross-episode memory."""
    def eval_fn(sub_params):
        dual = make_dual_from_subspace(frozen_main, sub_params)
        total = 0.0
        for trial in range(n_trials):
            seed = 5000 + trial * 13
            w_adapt = None
            ep_scores = []
            for ep in range(persistent_trials):
                reveal = 5 if ep == 0 else 0
                s, _, _, _, w_adapt, _ = run_dual_episode(
                    dual, seed=seed, reveal_steps=reveal,
                    n_steps=40, w_adapt_carry=w_adapt
                )
                weight = 1.0 if ep == 0 else 2.0
                ep_scores.append(s * weight)
            total += sum(ep_scores) / 5.0  # weights sum = 1+2+2
        return total / n_trials
    return eval_fn


def make_guard_fn_staged(frozen_main, n_trials=2):
    """guard_fn: ensure movement (prevent stuck)."""
    def guard_fn(sub_params):
        dual = make_dual_from_subspace(frozen_main, sub_params)
        total_move = 0.0
        for trial in range(n_trials):
            w_adapt = None
            for ep in range(2):
                reveal = 5 if ep == 0 else 0
                _, _, _, _, w_adapt, move_sum = run_dual_episode(
                    dual, seed=6000 + trial, reveal_steps=reveal,
                    w_adapt_carry=w_adapt, enable_replay=False
                )
                total_move += move_sum * 5.0
        return total_move / n_trials
    return guard_fn


def extract_initial_from_phase6():
    """Pull hippo+coupling from Phase 6 result as warm start."""
    path = "dual_memory_result.json"
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        p = d.get("best_params")
        if p and len(p) == DUAL_TOTAL:
            return list(p[N_MAIN:])  # hippo + coupling = 95D
    except Exception:
        pass
    return None


def run_stage(stage_num, frozen_main, time_budget, initial, verbose=False):
    """One Sentinel stage. experience_id="dual_staged_hippo" persists across stages."""
    print(f"\n{'='*70}")
    print(f"  STAGE {stage_num}: Sentinel on 95D hippo+coupling subspace, {time_budget}s")
    print(f"{'='*70}")

    t0 = time.time()
    result = Sentinel(
        eval_fn=make_eval_fn_staged(frozen_main, n_trials=3),
        guard_fn=make_guard_fn_staged(frozen_main, n_trials=2),
        param_ranges=HIPPO_PART_RANGES,
        param_names=HIPPO_PART_NAMES,
        experience_id="dual_staged_hippo",  # SAME across stages - accumulates
        initial_params=initial,
        learn=True,
        min_r_squared=0.25,  # slightly lower threshold for 95D
    ).run(time_budget=time_budget, verbose=False)
    elapsed = time.time() - t0

    best_sub = result.get("best_ever_params") or result.get("best_params")
    eval_s = result.get("best_ever_score", result.get("eval_score"))
    guard_s = result.get("guard_score")
    sv = result.get("verdict")
    mode = result.get("optimization_mode", "?")

    print(f"  stage{stage_num}: eval={eval_s:.2f}  guard={guard_s:.2f}  "
          f"elapsed={elapsed:.0f}s  verdict={sv}  mode={mode}")
    return best_sub, eval_s, elapsed, sv


def main():
    print("=" * 70)
    print("  Phase 6 STAGED: Sentinel exploited via experience_id accumulation")
    print("=" * 70)

    frozen_main = load_frozen_main()
    print(f"  Frozen main brain: 91D from Phase 5b (baseline reach 30%)")
    print(f"  Optimization space: {SUBSPACE_D}D (hippo 91 + coupling 4)")

    # Warm-start from Phase 6's hippo if available
    initial = extract_initial_from_phase6()
    if initial:
        print(f"  Warm-start: Phase 6 hippo+coupling (previously reach 15%)")
    else:
        mid = [(lo + hi) / 2 for lo, hi in HIPPO_PART_RANGES]
        initial = mid
        print(f"  Warm-start: midpoint of 95D subspace")

    # Baseline: use initial (pre-staged) to confirm starting point
    dual_initial = make_dual_from_subspace(frozen_main, initial)
    print("\n[pre-stage baseline]")
    base_std = evaluate_standard(dual_initial, n_seeds=20)
    print(f"  standard: reach={base_std['reach_rate']*100:.0f}%  "
          f"final={base_std['mean_final_dist']:.2f}  "
          f"|w|={base_std['mean_w_adapt_mag']:.2f}")
    base_persist = evaluate_persistent_food(dual_initial, n_trials=10, episodes_per_trial=3)
    print(f"  persistent reach per ep: {base_persist['reach_per_ep']}")

    # Run 3 sequential Sentinel stages with same experience_id
    STAGES = [
        (1, 300),   # Stage 1: 5 min
        (2, 300),   # Stage 2: 5 min  (experience warmed up)
        (3, 300),   # Stage 3: 5 min  (further refinement)
    ]

    best_sub = initial
    stage_history = []
    total_elapsed = 0
    for stage_num, budget in STAGES:
        best_sub, eval_s, elapsed, sv = run_stage(
            stage_num, frozen_main, budget, best_sub
        )
        total_elapsed += elapsed
        # Intermediate evaluation
        dual_stage = make_dual_from_subspace(frozen_main, best_sub)
        std_res = evaluate_standard(dual_stage, n_seeds=15)
        persist_res = evaluate_persistent_food(dual_stage, n_trials=6, episodes_per_trial=3)
        print(f"  stage{stage_num} eval: std reach={std_res['reach_rate']*100:.0f}%  "
              f"persistent ep={persist_res['reach_per_ep']}  "
              f"|w|={std_res['mean_w_adapt_mag']:.1f}")
        stage_history.append({
            "stage": stage_num,
            "budget_s": budget,
            "elapsed_s": round(elapsed, 1),
            "train_eval": round(float(eval_s), 2),
            "sentinel_verdict": sv,
            "standard": std_res,
            "persistent": persist_res,
        })

    # Final full evaluation
    final_dual = make_dual_from_subspace(frozen_main, best_sub)
    print(f"\n{'='*70}")
    print("  FINAL EVALUATION")
    print(f"{'='*70}")
    final_std = evaluate_standard(final_dual, n_seeds=20)
    final_persist = evaluate_persistent_food(final_dual, n_trials=10, episodes_per_trial=3)
    final_ablation = evaluate_no_memory_ablation(final_dual, n_seeds=20)

    print(f"  STANDARD:         reach={final_std['reach_rate']*100:.0f}%  "
          f"final={final_std['mean_final_dist']:.2f}  "
          f"|w|={final_std['mean_w_adapt_mag']:.2f}")
    print(f"  PERSISTENT FOOD:  ep1={final_persist['reach_per_ep'][0]*100:.0f}%  "
          f"ep2={final_persist['reach_per_ep'][1]*100:.0f}%  "
          f"ep3={final_persist['reach_per_ep'][2]*100:.0f}%")
    print(f"  |w| per ep:       {final_persist['w_mag_per_ep']}")
    print(f"  ABLATION no-carry: reach={final_ablation['reach_rate']*100:.0f}%")

    # Verdicts
    ep1, ep2, ep3 = final_persist['reach_per_ep']
    lift_vs_p5b = final_std['reach_rate'] - 0.30
    lift_cross_ep = ep3 - ep1

    if final_std['reach_rate'] > 0.50 and lift_cross_ep > 0.20:
        overall = "FULL SUCCESS"
    elif final_std['reach_rate'] > 0.35 or lift_cross_ep > 0.10:
        overall = "PARTIAL SUCCESS"
    elif final_std['reach_rate'] >= 0.30:
        overall = "MATCH (matches Phase 5b, but plasticity active)"
    else:
        overall = "NO CLEAR IMPROVEMENT"

    print(f"\n  verdict: {overall}")
    print(f"    vs Phase 5b standard: {lift_vs_p5b*100:+.0f}pt")
    print(f"    cross-ep ep1->ep3:   {lift_cross_ep*100:+.0f}pt")
    print(f"    plasticity active:   {final_std['mean_w_adapt_mag'] > 0.5}")
    print(f"\n  Total training: {total_elapsed:.0f}s across {len(STAGES)} stages")

    # Save
    out = {
        "verdict": overall,
        "subspace_dim": SUBSPACE_D,
        "frozen_main_from": "rgated_memory_result.json (Phase 5b)",
        "stages": stage_history,
        "total_elapsed_s": round(total_elapsed, 1),
        "final": {
            "standard": final_std,
            "persistent": final_persist,
            "no_carry_ablation": final_ablation,
        },
        "coupling_learned": {
            "m_to_h": round(float(best_sub[N_HIPPO]), 3),
            "h_to_m": round(float(best_sub[N_HIPPO + 1]), 3),
            "k_wta": int(round(float(best_sub[N_HIPPO + 2]))),
            "decay": round(float(best_sub[N_HIPPO + 3]), 3),
        },
        "best_hippo_and_coupling": [float(x) for x in best_sub],
    }
    with open("dual_memory_staged_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: dual_memory_staged_result.json")


if __name__ == "__main__":
    main()
