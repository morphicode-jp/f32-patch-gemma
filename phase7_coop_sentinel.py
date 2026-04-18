"""phase7_coop_sentinel.py - N-way alternating co-evolution via Sentinel.

Trains N agents in a cooperative world, alternating one-at-a-time.
Supports 3-factor Hebbian plasticity per agent (Phase 7 design spec).

Training protocol:
  Cycle 1: train agent 0 (others frozen), then agent 1, ..., agent N-1
  Cycle 2: repeat (agents benefit from improvements in peers)
  Cycle 3: final refinement
  -> N * 3 Sentinel rounds per full run

Each round:
  eval_fn(params_i) = own cooperative score averaged over 3 episodes
  guard_fn(params_i) = movement (don't be stuck)
  Sentinel time_budget per round: configurable (default 45s)

After training:
  save best_params per agent
  save history of Sentinel verdicts
  ready for evaluation via phase7_evaluation.py
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
from kathara16_brain import (
    PARAM_RANGES_16, PARAM_NAMES_16, TOTAL_PARAMS_16,
    N_NODES_16, KATHARA16_EDGES,
    simulate_step_16, simulate_step_16_hebbian, DEFAULT_INHIBIT_16,
    VOICE_NODE,
)
from multi_agent_world_N import MultiAgentCoopWorldN


def brain_step(params, sensors, states, w_adapt=None,
               reward=None, use_hebbian=False, k=3):
    """One step of a single agent's brain. Returns (new_states, firing).

    If use_hebbian=True: applies 3-factor plasticity (w_adapt modified in place).
    """
    if use_hebbian and w_adapt is not None:
        if reward is None:
            reward = 0.0
        new_states, firing, _ = simulate_step_16_hebbian(
            params, sensors, states, w_adapt,
            reward=reward, k=k, reward_scale=10.0
        )
    else:
        new_states, firing = simulate_step_16(
            params, sensors, states, inhibit_sign=DEFAULT_INHIBIT_16
        )
    return new_states, firing


def extract_motor(firing):
    """Extract (nav, speed, voice) from 16-dim firing vector."""
    return (float(firing[5]), float(firing[11]), float(firing[VOICE_NODE]))


def run_coop_match(params_list, seed, n_steps=50, use_hebbian=False,
                   record_traces=False):
    """Run one full episode with N agents in shared world.

    Returns:
      per_agent_score: list of float, one per agent
      per_agent_reached: list of bool
      per_agent_final_dist: list of float
      traces: dict (optional) with voice/dir/action per agent per step
    """
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
            "voices":     [[] for _ in range(N)],
            "dirs":       [[] for _ in range(N)],
            "navs":       [[] for _ in range(N)],
            "speeds":     [[] for _ in range(N)],
            "reached_steps": [-1] * N,
        }

    for step in range(n_steps):
        # Compute rewards (lagged for plasticity)
        rewards = [lag_dists[i] - prev_dists[i] for i in range(N)]

        # Brains act
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

        # Step world
        sensors_list, all_reached, reached_flags = world.step(actions)

        # Update distance tracking
        for i in range(N):
            d = world.get_food_dist(i)
            min_dists[i] = min(min_dists[i], d)
            lag_dists[i] = prev_dists[i]
            prev_dists[i] = d
            if record_traces and reached_flags[i] and traces["reached_steps"][i] == -1:
                traces["reached_steps"][i] = step

        if all_reached:
            break

    # Scores per agent
    own_approaches = [
        max(0.0, initial_dists[i] - min_dists[i]) / (initial_dists[i] + 1e-6)
        for i in range(N)
    ]
    reached = [world.reached[i] for i in range(N)]
    all_r = all(reached)

    # REACH-CENTRIC REWARD (Phase 7b redesign):
    # Reach own food is the MAIN reward (80pt).
    # Partners' reach is secondary bonus (up to 20pt).
    # Approach is small learning-signal nudge (15pt).
    # Synergy bonus if all reached (10pt).
    per_agent_scores = []
    for i in range(N):
        own = own_approaches[i]
        reached_i = reached[i]
        others_reached = [reached[j] for j in range(N) if j != i]
        avg_partner_reached = (float(np.mean(others_reached))
                                if others_reached else 0.0)
        score = (
            (80 if reached_i else 0)
            + avg_partner_reached * 20
            + own * 15
            + (10 if all_r else 0)
        )
        per_agent_scores.append(score)

    final_dists = [world.get_food_dist(i) for i in range(N)]
    if record_traces:
        return per_agent_scores, reached, final_dists, traces
    return per_agent_scores, reached, final_dists


def make_eval_for_agent(agent_idx, other_params, seed_offset,
                        n_episodes=3, use_hebbian=False):
    """eval_fn for training agent agent_idx while others are frozen."""
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            params_list = list(other_params)  # copy
            params_list[agent_idx] = list(params)
            seed = ep * 7 + 13 + seed_offset
            scores, _, _ = run_coop_match(
                params_list, seed=seed, use_hebbian=use_hebbian
            )
            total += scores[agent_idx]
        return total / n_episodes
    return eval_fn


def make_guard_for_agent(agent_idx, other_params, seed_offset,
                         n_episodes=2):
    """guard_fn: ensure agent moves."""
    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            params_list = list(other_params)
            params_list[agent_idx] = list(params)
            seed = ep * 11 + 5 + seed_offset
            world = MultiAgentCoopWorldN(n_agents=len(params_list), seed=seed)
            sensors_list = world.reset()
            states = [np.zeros(N_NODES_16) for _ in range(len(params_list))]
            prev_pos = world.agent_pos[agent_idx].copy()
            move = 0.0
            for _ in range(30):
                actions = []
                for i in range(len(params_list)):
                    states[i], firing = brain_step(
                        params_list[i], sensors_list[i], states[i]
                    )
                    actions.append(extract_motor(firing))
                sensors_list, done, _ = world.step(actions)
                curr = world.agent_pos[agent_idx]
                move += float(np.linalg.norm(curr - prev_pos))
                prev_pos = curr.copy()
                if done: break
            total += move * 5.0
        return total / n_episodes
    return guard_fn


def train_phase7(N_AGENTS=5, n_cycles=3, budget_per_agent=45,
                 use_hebbian=False, verbose=True, seed_base=0,
                 init_params_list=None):
    """Full Phase 7 training: alternating co-evolution N agents x n_cycles.

    init_params_list: optional list of N initial parameter vectors (warm-start).
      If None: all agents start from midpoint.
      If provided: agent i starts from init_params_list[i].
    """
    if init_params_list is not None:
        assert len(init_params_list) == N_AGENTS, (
            f"init_params_list length {len(init_params_list)} != N_AGENTS {N_AGENTS}"
        )
        agent_params = [list(p) for p in init_params_list]
        if verbose:
            print(f"  [init] warm-started from provided {N_AGENTS}-agent params")
    else:
        mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_16]
        agent_params = [list(mid) for _ in range(N_AGENTS)]
        if verbose:
            print(f"  [init] midpoint (no warm-start)")
    history = []
    t_start = time.time()

    for cycle in range(n_cycles):
        for agent_i in range(N_AGENTS):
            round_idx = cycle * N_AGENTS + agent_i + 1
            if verbose:
                print(f"\n  === cycle {cycle+1}/{n_cycles}  "
                      f"agent {agent_i}/{N_AGENTS}  "
                      f"(round {round_idx}/{N_AGENTS*n_cycles}) ===")
            eval_fn = make_eval_for_agent(
                agent_i, agent_params, seed_offset=seed_base + cycle * 100,
                n_episodes=3, use_hebbian=use_hebbian
            )
            guard_fn = make_guard_for_agent(
                agent_i, agent_params, seed_offset=seed_base + cycle * 100,
                n_episodes=2
            )
            t_round = time.time()
            result = Sentinel(
                eval_fn=eval_fn, guard_fn=guard_fn,
                param_ranges=PARAM_RANGES_16, param_names=PARAM_NAMES_16,
                experience_id=f"phase7_agent{agent_i}_cycle{cycle}",
                initial_params=list(agent_params[agent_i]),
                learn=True,
            ).run(time_budget=budget_per_agent, verbose=False)

            best = result.get("best_ever_params") or result.get("best_params")
            if best is not None:
                agent_params[agent_i] = list(best)

            round_time = time.time() - t_round
            total_time = time.time() - t_start
            eval_s = result.get("best_ever_score", 0.0)
            if verbose:
                print(f"  eval={eval_s:.2f}  "
                      f"round_t={round_time:.0f}s  "
                      f"total_t={total_time:.0f}s")
            history.append({
                "cycle": cycle + 1,
                "agent": agent_i,
                "round": round_idx,
                "eval_score": round(float(eval_s), 2),
                "verdict": result.get("verdict"),
                "round_elapsed_s": round(round_time, 1),
                "total_elapsed_s": round(total_time, 1),
            })

    return agent_params, history


if __name__ == "__main__":
    # Smoke test: N=3 agents, 1 cycle, minimal budget
    print("=" * 60)
    print("  phase7_coop_sentinel.py smoke test (N=3, 1 cycle)")
    print("=" * 60)

    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_16]
    # Test run_coop_match
    t0 = time.time()
    scores, reached, finals = run_coop_match([mid] * 3, seed=42, n_steps=40)
    print(f"\n  midpoint 3-agent match: "
          f"scores={[round(s,1) for s in scores]}  "
          f"reached={reached}  "
          f"time={(time.time()-t0)*1000:.0f}ms")

    # Test Hebbian path
    t0 = time.time()
    scores, reached, finals = run_coop_match([mid] * 3, seed=42, n_steps=40,
                                              use_hebbian=True)
    print(f"  midpoint 3-agent + Hebbian: "
          f"scores={[round(s,1) for s in scores]}  "
          f"time={(time.time()-t0)*1000:.0f}ms")

    # Quick training (1 cycle, 15s per agent) for smoke
    print(f"\n  Training smoke: 3 agents, 1 cycle, 15s/agent (~60s total)")
    t0 = time.time()
    params, history = train_phase7(
        N_AGENTS=3, n_cycles=1, budget_per_agent=15,
        use_hebbian=False, verbose=True
    )
    total_t = time.time() - t0
    print(f"\n  Total smoke elapsed: {total_t:.0f}s")
    print(f"  Final history: {len(history)} rounds")
    for h in history:
        print(f"    round {h['round']}: agent {h['agent']} "
              f"eval={h['eval_score']:.2f} t={h['round_elapsed_s']:.0f}s")
