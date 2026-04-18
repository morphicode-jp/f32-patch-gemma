"""phase7_reigen.py - Reigen auto-tunes Phase 7 hyperparameters.

Uses Reigen (from CLAUDE.md updated stack) as the outer meta-optimizer,
searching over Phase 7's control knobs. Reigen self-tunes its own 12
Sentinel hyperparameters in addition to our 3 user params.

User params (3):
  1. dale_law_enabled       (0-1, >0.5 = True)
  2. hebbian_enabled         (0-1, >0.5 = True)
  3. reward_reach_weight    (0.3-0.9, portion of reward that's pure reach)

Total joint dim: 3 + 12 self = 15D.

Inner eval: one full Phase 7 training run (N=3, 1 cycle, fast budget)
returning composite fitness:
  0.5 * reach_rate + 0.3 * ToM_pair_fraction + 0.2 * MI_gain

Goal: find configuration that breaks the 9% reach ceiling while preserving
ToM signal.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.reigen import reigen
import kathara16_brain
from phase7_coop_sentinel import train_phase7, run_coop_match
from phase7_evaluation import (
    evaluate_standard, evaluate_mi_panel, evaluate_tom,
)


# Global state for parameter switching
_ORIGINAL_DALE = kathara16_brain.DEFAULT_INHIBIT_16.copy()


def set_dale_law(enabled):
    """Monkey-patch Dale's law on/off globally."""
    if enabled:
        kathara16_brain.DEFAULT_INHIBIT_16 = _ORIGINAL_DALE.copy()
    else:
        kathara16_brain.DEFAULT_INHIBIT_16 = np.ones(16, dtype=np.float64)


_ORIGINAL_REWARD_WEIGHTS = None  # filled by patch


def patch_reward_weights(reach_weight):
    """Dynamically modify scoring in run_coop_match. reach_weight in [0.3, 0.9].

    Remaining weight split 50/50 between approach and synergy bonuses.
    """
    import phase7_coop_sentinel as pcs

    # Compute weights from reach_weight
    reach_w = reach_weight * 100   # 0-90
    partner_w = (1 - reach_weight) * 40  # residual
    approach_w = (1 - reach_weight) * 30
    synergy_w = (1 - reach_weight) * 20

    def patched_run_coop_match(params_list, seed, n_steps=50,
                                 use_hebbian=False, record_traces=False):
        """Copy of run_coop_match with parametric weights."""
        from kathara16_brain import N_NODES_16, KATHARA16_EDGES
        from multi_agent_world_N import MultiAgentCoopWorldN
        from phase7_coop_sentinel import brain_step, extract_motor

        N = len(params_list)
        world = MultiAgentCoopWorldN(n_agents=N, seed=seed)
        sensors_list = world.reset()

        initial_dists = [world.get_food_dist(i) for i in range(N)]
        min_dists = list(initial_dists)
        states = [np.zeros(N_NODES_16) for _ in range(N)]
        w_adapts = ([np.zeros(len(KATHARA16_EDGES)) for _ in range(N)]
                    if use_hebbian else [None] * N)
        prev_dists = list(initial_dists)
        lag_dists = list(initial_dists)

        traces = None
        if record_traces:
            traces = {
                "voices": [[] for _ in range(N)],
                "dirs":   [[] for _ in range(N)],
                "navs":   [[] for _ in range(N)],
                "speeds": [[] for _ in range(N)],
                "reached_steps": [-1] * N,
            }

        for step in range(n_steps):
            rewards = [lag_dists[i] - prev_dists[i] for i in range(N)]
            actions = []
            for i in range(N):
                if record_traces:
                    traces["dirs"][i].append(world.get_own_food_direction(i))
                states[i], firing = brain_step(
                    params_list[i], sensors_list[i], states[i],
                    w_adapt=w_adapts[i] if use_hebbian else None,
                    reward=rewards[i] if use_hebbian else None,
                    use_hebbian=use_hebbian
                )
                nav, speed, voice = extract_motor(firing)
                actions.append((nav, speed, voice))
                if record_traces:
                    traces["voices"][i].append(voice)
                    traces["navs"][i].append(nav)
                    traces["speeds"][i].append(speed)
            sensors_list, all_reached, reached_flags = world.step(actions)
            for i in range(N):
                d = world.get_food_dist(i)
                min_dists[i] = min(min_dists[i], d)
                lag_dists[i] = prev_dists[i]
                prev_dists[i] = d
                if record_traces and reached_flags[i] and traces["reached_steps"][i] == -1:
                    traces["reached_steps"][i] = step
            if all_reached:
                break

        own_approaches = [
            max(0.0, initial_dists[i] - min_dists[i]) / (initial_dists[i] + 1e-6)
            for i in range(N)
        ]
        reached = [world.reached[i] for i in range(N)]
        all_r = all(reached)

        per_agent_scores = []
        for i in range(N):
            own = own_approaches[i]
            reached_i = reached[i]
            others_reached = [reached[j] for j in range(N) if j != i]
            avg_partner_reached = (float(np.mean(others_reached))
                                    if others_reached else 0.0)
            score = (
                (reach_w if reached_i else 0)
                + avg_partner_reached * partner_w
                + own * approach_w
                + (synergy_w if all_r else 0)
            )
            per_agent_scores.append(score)

        final_dists = [world.get_food_dist(i) for i in range(N)]
        if record_traces:
            return per_agent_scores, reached, final_dists, traces
        return per_agent_scores, reached, final_dists

    pcs.run_coop_match = patched_run_coop_match


def inner_eval(user_params):
    """Full Phase 7 inner training + evaluation. Returns composite fitness."""
    dale_val, hebb_val, reach_w = user_params
    dale = dale_val > 0.5
    hebb = hebb_val > 0.5

    # Apply configuration
    set_dale_law(dale)
    patch_reward_weights(reach_w)

    # Small training (N=3, 1 cycle, 15s/agent)
    try:
        params_list, _ = train_phase7(
            N_AGENTS=3, n_cycles=1, budget_per_agent=15,
            use_hebbian=hebb, verbose=False,
        )
    except Exception as e:
        return 0.0

    # Quick eval (fewer seeds/episodes for speed)
    std = evaluate_standard(params_list, n_seeds=10)
    mi = evaluate_mi_panel(params_list, n_episodes=15, n_perm=50)
    tom = evaluate_tom(params_list, n_episodes=10)

    reach = float(std["mean_agent_reach_rate"])
    mi_gain = max(0.0, float(mi["mean_gain"]))
    tom_pair = float(tom["pair_fraction_predictive"])

    composite = 0.5 * reach + 0.3 * tom_pair + 0.2 * mi_gain
    print(f"  [eval] dale={dale} hebb={hebb} reach_w={reach_w:.2f} "
          f"-> reach={reach*100:.0f}% ToM={tom_pair*100:.0f}% "
          f"MI={mi_gain:+.3f} composite={composite:.3f}")
    return composite


def inner_guard(user_params):
    """Guard: protect against degenerate (all-zero, no motion) brains.
    Returns 0 (neutral) since we only care about eval."""
    return 0.0


def main():
    print("=" * 70)
    print("  REIGEN × PHASE 7: auto-tune Phase 7 hyperparameters")
    print("=" * 70)
    print(f"  User params: dale_law, hebbian_enabled, reward_reach_weight")
    print(f"  Joint search: 3 + 12 self = 15D")

    user_ranges = [
        (0.0, 1.0),   # dale_law (0 off, 1 on)
        (0.0, 1.0),   # hebbian (0 off, 1 on)
        (0.3, 0.9),   # reward_reach_weight
    ]
    user_names = ["dale_law", "hebbian", "reward_reach_weight"]

    t0 = time.time()
    result = reigen(
        eval_fn=inner_eval,
        guard_fn=inner_guard,
        user_param_ranges=user_ranges,
        user_param_names=user_names,
        experience_id="genesis",
        inner_time_budget=2,
        time_budget=1200,  # 20 min scheduled
        wall_time_factor=2.0,
    )
    elapsed = time.time() - t0

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Verdict: {result.get('verdict')}")
    print(f"\n  Best user params:")
    best_user = result.get("user_best_params") or []
    for n, v in zip(user_names, best_user):
        print(f"    {n} = {v:.3f}")
    print(f"  Best eval: {result.get('user_best_score', 0):.3f}")
    print(f"\n  Self-tuned Sentinel knobs (first 5):")
    self_best = result.get("self_best_params") or {}
    for k in list(self_best.keys())[:5]:
        print(f"    {k} = {self_best[k]:.3f}")

    diag = result.get("self_diagnostic") or {}
    print(f"\n  Self diagnostic:")
    for k, v in diag.items():
        print(f"    {k} = {v}")

    out = {
        "elapsed_s": round(elapsed, 1),
        "user_best_params": [float(x) for x in (best_user or [])],
        "user_best_names": user_names,
        "user_best_score": float(result.get("user_best_score", 0) or 0),
        "self_best_params": {k: float(v) for k, v in self_best.items()},
        "self_diagnostic": diag,
        "verdict": result.get("verdict"),
    }
    with open("phase7_reigen_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_reigen_result.json")


if __name__ == "__main__":
    main()
