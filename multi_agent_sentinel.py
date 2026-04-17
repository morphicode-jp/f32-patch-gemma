"""multi_agent_sentinel.py - Co-evolution via alternating Sentinel runs.

Red Queen dynamics:
  Gen 1: Freeze B (random), optimize A via Sentinel for 60s.
  Gen 2: Freeze A (best), optimize B via Sentinel for 60s.
  Gen 3: Freeze B (best), optimize A ...
  Repeat N generations.

Each agent's eval_fn = win_rate against opponent on 3 episodes.
Each agent's guard_fn = survival (not getting stuck at wall).

Expected emergent phenomena:
  - Agents learn to USE opponent's position (node 8 signal)
  - Specialization (aggressive vs cautious strategies)
  - Red Queen: both keep improving because the other does
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v8 import PARAM_RANGES, PARAM_NAMES, simulate_step
from twelve.agent.sentinel import Sentinel
from multi_agent_world import MultiAgentFlyWorld


def run_match(params_a, params_b, seed, n_steps=40):
    """One match. Returns (agent_0_score, agent_1_score).

    Score structure:
      +60 if agent reached food first
      +30 * approach_ratio (closer at end = better)
      +0..10 move bonus (active agent > stuck)
    """
    pa = np.array(params_a, dtype=np.float64)
    pb = np.array(params_b, dtype=np.float64)
    world = MultiAgentFlyWorld(seed=seed)
    s0, s1 = world.reset()
    initial_d0 = world.get_food_dist(0)
    initial_d1 = world.get_food_dist(1)
    min_d0 = initial_d0
    min_d1 = initial_d1
    move0 = 0.0
    move1 = 0.0
    states_a = np.zeros(12)
    states_b = np.zeros(12)
    winner = None

    for step in range(n_steps):
        prev_pos0 = world.agent_pos[0].copy()
        prev_pos1 = world.agent_pos[1].copy()
        states_a, fa = simulate_step(pa, s0, states_a)
        states_b, fb = simulate_step(pb, s1, states_b)
        nav0, cen0 = float(fa[5]), float(fa[11])
        nav1, cen1 = float(fb[5]), float(fb[11])
        s0, s1, done, winner = world.step(nav0, cen0, nav1, cen1)
        move0 += float(np.linalg.norm(world.agent_pos[0] - prev_pos0))
        move1 += float(np.linalg.norm(world.agent_pos[1] - prev_pos1))
        min_d0 = min(min_d0, world.get_food_dist(0))
        min_d1 = min(min_d1, world.get_food_dist(1))
        if done:
            break

    approach0 = max(0.0, initial_d0 - min_d0) / (initial_d0 + 1e-6)
    approach1 = max(0.0, initial_d1 - min_d1) / (initial_d1 + 1e-6)
    score0 = (60.0 if winner == 0 else 0.0) + 30.0 * approach0 + min(10.0, move0 * 2.0)
    score1 = (60.0 if winner == 1 else 0.0) + 30.0 * approach1 + min(10.0, move1 * 2.0)
    return score0, score1


def make_eval_fn(opponent_params, is_agent_0, n_episodes=3):
    """eval_fn for optimizing one agent while opponent is frozen."""
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
    """guard_fn = movement activity (avoid stuck). Protects against dead agents."""
    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            seed = ep * 11 + 5
            pa = np.array(params if is_agent_0 else opponent_params)
            pb = np.array(opponent_params if is_agent_0 else params)
            world = MultiAgentFlyWorld(seed=seed)
            s0, s1 = world.reset()
            states_a = np.zeros(12); states_b = np.zeros(12)
            move = 0.0
            for step in range(30):
                prev = world.agent_pos[0 if is_agent_0 else 1].copy()
                states_a, fa = simulate_step(pa, s0, states_a)
                states_b, fb = simulate_step(pb, s1, states_b)
                s0, s1, done, _ = world.step(float(fa[5]), float(fa[11]),
                                             float(fb[5]), float(fb[11]))
                curr = world.agent_pos[0 if is_agent_0 else 1]
                move += float(np.linalg.norm(curr - prev))
                if done: break
            total += move * 5.0  # scale to roughly 0-100
        return total / n_episodes
    return guard_fn


def co_evolve(n_generations=4, gen_budget=90, verbose=True):
    """Alternating Sentinel optimization."""
    # Init: both agents random (midpoint)
    mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES]
    params_a = list(mid)
    params_b = list(mid)

    history = []
    t_start = time.time()

    for gen in range(n_generations):
        target = gen % 2  # 0 or 1 alternates
        opponent = params_b if target == 0 else params_a

        if verbose:
            print(f"\n{'='*70}")
            print(f"  Generation {gen+1}/{n_generations}: optimizing Agent {target}")
            print(f"{'='*70}")

        eval_fn = make_eval_fn(opponent, is_agent_0=(target == 0), n_episodes=3)
        guard_fn = make_guard_fn(opponent, is_agent_0=(target == 0), n_episodes=2)

        initial = params_a if target == 0 else params_b
        result = Sentinel(
            eval_fn=eval_fn,
            guard_fn=guard_fn,
            param_ranges=PARAM_RANGES,
            param_names=PARAM_NAMES,
            experience_id=f"coevo_agent_{target}",
            initial_params=initial,
            learn=True,
        ).run(time_budget=gen_budget, verbose=False)

        best = result.get("best_ever_params") or result.get("best_params")
        if best is not None:
            if target == 0:
                params_a = list(best)
            else:
                params_b = list(best)

        # Measure: A vs B win rate on 20 fresh seeds
        wins_a = 0
        wins_b = 0
        draws = 0
        for seed in range(100, 120):
            s0, s1 = run_match(params_a, params_b, seed)
            if s0 > s1 + 2:
                wins_a += 1
            elif s1 > s0 + 2:
                wins_b += 1
            else:
                draws += 1

        gen_elapsed = time.time() - t_start
        if verbose:
            print(f"\n  Gen {gen+1} result:")
            print(f"    Optimized Agent {target}: eval={result.get('best_ever_score', 0):.2f} "
                  f"guard={result.get('guard_score', 0):.2f} verdict={result.get('verdict')}")
            print(f"    20-match outcome: A wins={wins_a}  B wins={wins_b}  draws={draws}")
            print(f"    Total elapsed: {gen_elapsed:.0f}s")

        history.append({
            "generation": gen + 1,
            "optimized_agent": target,
            "eval_score": result.get("best_ever_score"),
            "guard_score": result.get("guard_score"),
            "verdict": result.get("verdict"),
            "a_wins": wins_a,
            "b_wins": wins_b,
            "draws": draws,
            "elapsed_total_s": round(gen_elapsed, 1),
        })

    return params_a, params_b, history


def main():
    print("=" * 70)
    print("  Phase 2 MVP: Multi-agent co-evolution")
    print("=" * 70)

    pa, pb, history = co_evolve(n_generations=4, gen_budget=90, verbose=True)

    print(f"\n{'='*70}")
    print(f"  Red Queen Dynamics Summary")
    print(f"{'='*70}")
    print(f"{'Gen':>4s} {'opt':>4s} {'eval':>8s} {'A wins':>8s} {'B wins':>8s} {'draws':>7s}")
    for h in history:
        print(f"{h['generation']:>4d} {h['optimized_agent']:>4d} "
              f"{(h['eval_score'] or 0):>8.2f} {h['a_wins']:>8d} {h['b_wins']:>8d} {h['draws']:>7d}")

    # Check: does optimizing agent consistently win?
    target_wins = [h["a_wins"] if h["optimized_agent"] == 0 else h["b_wins"]
                   for h in history]
    print(f"\n  Optimized-agent win counts per gen: {target_wins}")
    if all(w >= 10 for w in target_wins[-2:]):
        print("  -> Red Queen CONFIRMED (each gen dominant after training)")
    elif sum(target_wins) / len(target_wins) > 8:
        print("  -> Red Queen PARTIAL (training gives advantage most gens)")
    else:
        print("  -> Red Queen WEAK (training advantage unclear)")

    with open("coevolution_result.json", "w") as f:
        json.dump({
            "history": history,
            "final_params_a": [float(x) for x in pa],
            "final_params_b": [float(x) for x in pb],
        }, f, indent=2)
    print("\n  Saved: coevolution_result.json")


if __name__ == "__main__":
    main()
