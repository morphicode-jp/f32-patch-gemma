"""phase7_run_N5_optimal.py - N=5 with Reigen-discovered optimal config.

Config discovered by Reigen × batch (converged across partial + full runs):
  dale_law     = OFF
  hebbian      = OFF
  reach_w      = 0.60

This script scales from Reigen's N=3 discovery to N=5 for Phase 7
pre-registered criteria evaluation.

Goal: verify reach 17-20% + ToM 75-100% holds at N=5 scale.
If yes → Phase 7 criteria 2+/4 met → declare PARTIAL SUCCESS.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply learned config BEFORE importing phase7_coop_sentinel
import kathara16_brain
# dale_law = OFF
kathara16_brain.DEFAULT_INHIBIT_16 = np.ones(16, dtype=np.float64)

# Apply reach_w = 0.60 patch (same logic as phase7_reigen_batch worker)
import phase7_coop_sentinel as pcs
from phase7_coop_sentinel import brain_step, extract_motor
from kathara16_brain import N_NODES_16, KATHARA16_EDGES
from multi_agent_world_N import MultiAgentCoopWorldN

REACH_W = 0.60
_reach_w_bonus = REACH_W * 100
_partner_w = (1 - REACH_W) * 40
_approach_w = (1 - REACH_W) * 30
_synergy_w = (1 - REACH_W) * 20


def patched_run_coop_match(params_list, seed, n_steps=50,
                           use_hebbian=False, record_traces=False):
    N = len(params_list)
    world = MultiAgentCoopWorldN(n_agents=N, seed=seed)
    sensors_list = world.reset()
    initial_dists = [world.get_food_dist(i) for i in range(N)]
    min_dists = list(initial_dists)
    states = [np.zeros(N_NODES_16) for _ in range(N)]
    w_adapts = ([np.zeros(len(KATHARA16_EDGES)) for _ in range(N)]
                if use_hebbian else [None] * N)
    prev_dists = list(initial_dists); lag_dists = list(initial_dists)
    traces = None
    if record_traces:
        traces = {"voices": [[] for _ in range(N)],
                  "dirs":   [[] for _ in range(N)],
                  "navs":   [[] for _ in range(N)],
                  "speeds": [[] for _ in range(N)],
                  "reached_steps": [-1] * N}
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
            lag_dists[i] = prev_dists[i]; prev_dists[i] = d
            if record_traces and reached_flags[i] and traces["reached_steps"][i] == -1:
                traces["reached_steps"][i] = step
        if all_reached: break
    own_approaches = [
        max(0.0, initial_dists[i] - min_dists[i]) / (initial_dists[i] + 1e-6)
        for i in range(N)
    ]
    reached = [world.reached[i] for i in range(N)]
    all_r = all(reached)
    per_agent_scores = []
    for i in range(N):
        reached_i = reached[i]
        others_reached = [reached[j] for j in range(N) if j != i]
        avg_partner_reached = (float(np.mean(others_reached))
                                if others_reached else 0.0)
        score = (
            (_reach_w_bonus if reached_i else 0)
            + avg_partner_reached * _partner_w
            + own_approaches[i] * _approach_w
            + (_synergy_w if all_r else 0)
        )
        per_agent_scores.append(score)
    final_dists = [world.get_food_dist(i) for i in range(N)]
    if record_traces:
        return per_agent_scores, reached, final_dists, traces
    return per_agent_scores, reached, final_dists


pcs.run_coop_match = patched_run_coop_match

# NOW import training (with patches applied)
from phase7_coop_sentinel import train_phase7
from phase7_evaluation import run_full_evaluation


def main():
    print("=" * 70)
    print("  PHASE 7 N=5 OPTIMAL: Reigen-learned config at scale")
    print("=" * 70)
    print(f"  dale_law     = OFF (DEFAULT_INHIBIT_16 = all ones)")
    print(f"  hebbian      = OFF")
    print(f"  reach_w      = {REACH_W}")

    t0 = time.time()
    agent_params, history = train_phase7(
        N_AGENTS=5, n_cycles=3, budget_per_agent=45,
        use_hebbian=False, verbose=True,
    )
    train_time = time.time() - t0
    print(f"\n  Training: {train_time:.0f}s ({train_time/60:.1f} min)")

    with open("phase7_N5_optimal_history.json", "w") as f:
        json.dump({
            "reigen_config": {"dale_law": False, "hebbian": False,
                              "reward_reach_weight": REACH_W},
            "N_AGENTS": 5, "n_cycles": 3, "budget_per_agent_s": 45,
            "total_train_elapsed_s": round(train_time, 1),
            "history": history,
            "final_params": [[float(x) for x in p] for p in agent_params],
        }, f, indent=2)

    eval_result = run_full_evaluation(
        agent_params,
        save_path="phase7_N5_optimal_eval.json"
    )
    total = time.time() - t0
    print(f"\n  TOTAL: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
