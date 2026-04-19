"""phase7_wta_reward.py - Winner-Take-All reward (no social bonuses).

Based on reach-threshold diagnostic: agents converged to social-dominance
attractor because reward structure included partner bonuses. Test: if we
REMOVE all social reward components, does agents rediscover reach focus?

New reward (per agent):
  score = 100 if reached own food
        + 15 * own_approach_ratio    (tiny gradient for learning)
        + 0                          (NO partner bonus)
        + 0                          (NO synergy bonus)

Compare to prior Phase 7 reward structures:
  Original:        own*40 + partner*30 + reached*20 + synergy*30
  Reach-centric:   own*15 + partner*20 + reached*80 + synergy*10
  Winner-take-all: own*15 + reached*100 (no social)

Hypothesis: without partner rewards, agents can't exploit social-following
and must actually learn food-seeking.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kathara16_brain
kathara16_brain.DEFAULT_INHIBIT_16 = np.ones(16, dtype=np.float64)

import phase7_coop_sentinel as pcs
from phase7_coop_sentinel import brain_step, extract_motor
from kathara16_brain import N_NODES_16, KATHARA16_EDGES
from multi_agent_world_N import MultiAgentCoopWorldN


def wta_run_coop_match(params_list, seed, n_steps=50,
                       use_hebbian=False, record_traces=False):
    """Winner-take-all reward structure."""
    N = len(params_list)
    world = MultiAgentCoopWorldN(n_agents=N, seed=seed)
    sensors_list = world.reset()
    initial_dists = [world.get_food_dist(i) for i in range(N)]
    min_dists = list(initial_dists)
    states = [np.zeros(N_NODES_16) for _ in range(N)]
    traces = None
    if record_traces:
        traces = {"voices": [[] for _ in range(N)],
                  "dirs":   [[] for _ in range(N)],
                  "navs":   [[] for _ in range(N)],
                  "speeds": [[] for _ in range(N)],
                  "reached_steps": [-1] * N}

    for step in range(n_steps):
        actions = []
        for i in range(N):
            if record_traces:
                traces["dirs"][i].append(world.get_own_food_direction(i))
            states[i], firing = brain_step(
                params_list[i], sensors_list[i], states[i]
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
            if record_traces and reached_flags[i] and traces["reached_steps"][i] == -1:
                traces["reached_steps"][i] = step
        if all_reached: break

    own_approaches = [
        max(0.0, initial_dists[i] - min_dists[i]) / (initial_dists[i] + 1e-6)
        for i in range(N)
    ]
    reached = [world.reached[i] for i in range(N)]

    # WINNER-TAKE-ALL scoring (no partner/synergy terms):
    per_agent_scores = []
    for i in range(N):
        score = (100 if reached[i] else 0) + own_approaches[i] * 15
        per_agent_scores.append(score)

    final_dists = [world.get_food_dist(i) for i in range(N)]
    if record_traces:
        return per_agent_scores, reached, final_dists, traces
    return per_agent_scores, reached, final_dists


pcs.run_coop_match = wta_run_coop_match

from phase7_coop_sentinel import train_phase7
from phase7_evaluation import run_full_evaluation


def main():
    print("=" * 70)
    print("  PHASE 7 WINNER-TAKE-ALL reward (no social bonuses)")
    print("=" * 70)
    print(f"  score = 100 if reached + 15 * approach (pure individualist)")
    print(f"  Config: dale=OFF (from Reigen), hebb=OFF")

    t0 = time.time()
    agent_params, history = train_phase7(
        N_AGENTS=5, n_cycles=3, budget_per_agent=45,
        use_hebbian=False, verbose=True,
    )
    train_time = time.time() - t0
    print(f"\n  Training: {train_time:.0f}s ({train_time/60:.1f} min)")

    with open("phase7_wta_history.json", "w") as f:
        json.dump({
            "reward_type": "winner_take_all",
            "N_AGENTS": 5, "n_cycles": 3, "budget_per_agent_s": 45,
            "total_train_elapsed_s": round(train_time, 1),
            "history": history,
            "final_params": [[float(x) for x in p] for p in agent_params],
        }, f, indent=2)

    eval_result = run_full_evaluation(
        agent_params,
        save_path="phase7_wta_eval.json"
    )
    total = time.time() - t0
    print(f"\n  TOTAL: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
