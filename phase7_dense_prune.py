"""phase7_dense_prune.py - Dense-then-Prune: biological brain development analog.

Two-stage training:
  Stage 1 (過剰結合期): Train 120-edge dense brain for baseline Phase 7 task
  Stage 2 (剪定期):      Identify bottom 60% edges by |weight|, mask them
  Stage 3 (成熟期):      Retrain with sparse mask, only surviving 48 edges adapt

Hypothesis: Starting dense gives richer search space, enabling reach > 9%.

Uses the same Reigen-discovered optimal config:
  dale_law=OFF, hebbian=OFF, reach_w=0.60
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twelve.agent.sentinel import Sentinel
import kathara_dense_brain as kdb
from kathara_dense_brain import (
    PARAM_RANGES_DENSE, PARAM_NAMES_DENSE, TOTAL_PARAMS_DENSE,
    N_NODES_DENSE, DENSE_EDGES, simulate_step_dense,
    DEFAULT_INHIBIT_DENSE, compute_edge_mask_from_magnitudes,
)
from multi_agent_world_N import MultiAgentCoopWorldN


# Apply Reigen-optimal config: dale=OFF
kdb.DEFAULT_INHIBIT_DENSE = np.ones(N_NODES_DENSE, dtype=np.float64)

REACH_W = 0.60
_r_bonus = REACH_W * 100
_p_bonus = (1 - REACH_W) * 40
_a_bonus = (1 - REACH_W) * 30
_s_bonus = (1 - REACH_W) * 20

# Global edge mask (set between stages)
_EDGE_MASK = None


def brain_step_dense(params, sensors, states, edge_mask=None):
    new_states, firing = simulate_step_dense(
        params, sensors, states,
        inhibit_sign=kdb.DEFAULT_INHIBIT_DENSE,
        edge_mask=edge_mask,
    )
    return new_states, firing


def extract_motor(firing):
    return float(firing[5]), float(firing[11]), float(firing[1])


def run_coop_match_dense(params_list, seed, n_steps=50, record_traces=False):
    N = len(params_list)
    world = MultiAgentCoopWorldN(n_agents=N, seed=seed)
    sensors_list = world.reset()
    initial_dists = [world.get_food_dist(i) for i in range(N)]
    min_dists = list(initial_dists)
    states = [np.zeros(N_NODES_DENSE) for _ in range(N)]
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
            states[i], firing = brain_step_dense(
                params_list[i], sensors_list[i], states[i],
                edge_mask=_EDGE_MASK,
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
    all_r = all(reached)
    per_agent_scores = []
    for i in range(N):
        others_reached = [reached[j] for j in range(N) if j != i]
        avg_partner = float(np.mean(others_reached)) if others_reached else 0.0
        score = (
            (_r_bonus if reached[i] else 0)
            + avg_partner * _p_bonus
            + own_approaches[i] * _a_bonus
            + (_s_bonus if all_r else 0)
        )
        per_agent_scores.append(score)
    final_dists = [world.get_food_dist(i) for i in range(N)]
    if record_traces:
        return per_agent_scores, reached, final_dists, traces
    return per_agent_scores, reached, final_dists


def make_eval_for_agent(agent_idx, other_params, seed_offset, n_episodes=3):
    def eval_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            params_list = list(other_params)
            params_list[agent_idx] = list(params)
            seed = ep * 7 + 13 + seed_offset
            scores, _, _ = run_coop_match_dense(params_list, seed=seed)
            total += scores[agent_idx]
        return total / n_episodes
    return eval_fn


def make_guard_for_agent(agent_idx, other_params, seed_offset, n_episodes=2):
    def guard_fn(params):
        total = 0.0
        for ep in range(n_episodes):
            params_list = list(other_params)
            params_list[agent_idx] = list(params)
            seed = ep * 11 + 5 + seed_offset
            world = MultiAgentCoopWorldN(n_agents=len(params_list), seed=seed)
            sensors_list = world.reset()
            states = [np.zeros(N_NODES_DENSE) for _ in range(len(params_list))]
            prev_pos = world.agent_pos[agent_idx].copy()
            move = 0.0
            for _ in range(30):
                actions = []
                for i in range(len(params_list)):
                    states[i], firing = brain_step_dense(
                        params_list[i], sensors_list[i], states[i],
                        edge_mask=_EDGE_MASK,
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


def train_dense_phase7(N_AGENTS=3, n_cycles=1, budget_per_agent=30,
                       init_params_list=None, verbose=True, stage_id=""):
    """Reuses Phase 7 structure but on dense brain."""
    if init_params_list is not None:
        agent_params = [list(p) for p in init_params_list]
    else:
        mid = [(lo + hi) / 2 for lo, hi in PARAM_RANGES_DENSE]
        agent_params = [list(mid) for _ in range(N_AGENTS)]

    history = []
    t_start = time.time()
    for cycle in range(n_cycles):
        for agent_i in range(N_AGENTS):
            round_idx = cycle * N_AGENTS + agent_i + 1
            eval_fn = make_eval_for_agent(
                agent_i, agent_params, seed_offset=cycle * 100
            )
            guard_fn = make_guard_for_agent(
                agent_i, agent_params, seed_offset=cycle * 100
            )
            t_round = time.time()
            result = Sentinel(
                eval_fn=eval_fn, guard_fn=guard_fn,
                param_ranges=PARAM_RANGES_DENSE,
                param_names=PARAM_NAMES_DENSE,
                experience_id=f"dense_{stage_id}_agent{agent_i}_cycle{cycle}",
                initial_params=list(agent_params[agent_i]),
                learn=True,
            ).run(time_budget=budget_per_agent, verbose=False)
            best = result.get("best_ever_params") or result.get("best_params")
            if best is not None:
                agent_params[agent_i] = list(best)
            history.append({
                "cycle": cycle + 1,
                "agent": agent_i,
                "round": round_idx,
                "eval": round(float(result.get("best_ever_score") or 0), 2),
                "round_s": round(time.time() - t_round, 1),
                "total_s": round(time.time() - t_start, 1),
            })
            if verbose:
                h = history[-1]
                print(f"  [{stage_id}] c{cycle+1}/a{agent_i}: "
                      f"eval={h['eval']:.1f} round={h['round_s']:.0f}s")

    return agent_params, history


def evaluate_dense(agent_params, n_seeds=15):
    """Simplified reach-focused evaluation for dense brain."""
    reaches = []
    for seed in range(2000, 2000 + n_seeds):
        _, reached, _ = run_coop_match_dense(agent_params, seed=seed, n_steps=50)
        reaches.append(reached)
    reaches_arr = np.array(reaches)
    per_agent = reaches_arr.mean(axis=0)
    return {
        "per_agent_reach": [float(r) for r in per_agent],
        "mean_reach": float(per_agent.mean()),
    }


def main():
    print("=" * 70)
    print("  DENSE-THEN-PRUNE: biological brain development analog")
    print("=" * 70)
    print(f"  Total edges: 120 (vs 48 sparse Kathara(16))")
    print(f"  Param dim: 201 (vs 129 sparse)")
    print(f"  Config: dale=OFF, hebb=OFF, reach_w=0.60 (Reigen-optimal)")

    t_all = time.time()
    global _EDGE_MASK
    _EDGE_MASK = None  # Stage 1: all edges active

    # ==========
    # STAGE 1: Dense training (過剰結合期)
    # ==========
    print(f"\n{'='*70}\n  STAGE 1: Dense training (all 120 edges active)\n{'='*70}")
    t1 = time.time()
    params_list_s1, hist_s1 = train_dense_phase7(
        N_AGENTS=3, n_cycles=2, budget_per_agent=30,
        verbose=True, stage_id="s1dense",
    )
    print(f"  Stage 1 elapsed: {time.time()-t1:.0f}s")
    eval_s1 = evaluate_dense(params_list_s1, n_seeds=15)
    print(f"  Stage 1 reach: {eval_s1['mean_reach']*100:.0f}% "
          f"(per-agent {[f'{r*100:.0f}%' for r in eval_s1['per_agent_reach']]})")

    # ==========
    # STAGE 2: Pruning (剪定期)
    # ==========
    print(f"\n{'='*70}\n  STAGE 2: Pruning 60% weakest edges\n{'='*70}")
    # Average edge magnitudes across agents to decide what to prune
    n_edges = len(DENSE_EDGES)
    edge_mags = np.zeros(n_edges)
    for p in params_list_s1:
        edge_mags += np.abs(np.asarray(p[:n_edges]))
    edge_mags /= len(params_list_s1)
    threshold = np.percentile(edge_mags, 60)  # prune bottom 60%
    mask = (edge_mags >= threshold).astype(np.float64)
    _EDGE_MASK = mask
    print(f"  Edges kept: {int(mask.sum())} / {n_edges} "
          f"({int(mask.sum())/n_edges*100:.0f}%)")
    print(f"  Kept edges have |weight| >= {threshold:.3f}")

    eval_s2_masked = evaluate_dense(params_list_s1, n_seeds=15)
    print(f"  Same params + mask: reach {eval_s2_masked['mean_reach']*100:.0f}%")

    # ==========
    # STAGE 3: Retrain pruned brain (成熟期)
    # ==========
    print(f"\n{'='*70}\n  STAGE 3: Retraining {int(mask.sum())}-edge sparse brain\n{'='*70}")
    t3 = time.time()
    params_list_s3, hist_s3 = train_dense_phase7(
        N_AGENTS=3, n_cycles=2, budget_per_agent=30,
        init_params_list=params_list_s1,  # warm-start from stage 1
        verbose=True, stage_id="s3pruned",
    )
    print(f"  Stage 3 elapsed: {time.time()-t3:.0f}s")
    eval_s3 = evaluate_dense(params_list_s3, n_seeds=15)
    print(f"  Stage 3 reach: {eval_s3['mean_reach']*100:.0f}% "
          f"(per-agent {[f'{r*100:.0f}%' for r in eval_s3['per_agent_reach']]})")

    # Summary
    total = time.time() - t_all
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  Stage 1 (dense 120 edges): reach {eval_s1['mean_reach']*100:.0f}%")
    print(f"  Stage 2 (pruned {int(mask.sum())} edges, no retrain): "
          f"reach {eval_s2_masked['mean_reach']*100:.0f}%")
    print(f"  Stage 3 (pruned {int(mask.sum())} edges, retrained): "
          f"reach {eval_s3['mean_reach']*100:.0f}%")
    print(f"  Total elapsed: {total:.0f}s ({total/60:.1f} min)")

    # Compare to baseline (sparse Kathara(16) N=5 was 7%)
    baseline_sparse_n5 = 0.07
    baseline_sparse_n3 = 0.17  # approximate from Reigen
    print(f"\n  Baselines:")
    print(f"    sparse Kathara(16) N=5 reach:    7%")
    print(f"    sparse Kathara(16) N=3 reach:   ~17% (Reigen peak)")
    print(f"    dense then pruned Stage 3:       {eval_s3['mean_reach']*100:.0f}%")

    improvement = eval_s3["mean_reach"] - baseline_sparse_n3
    print(f"    delta vs N=3 sparse: {improvement*100:+.0f}pt")

    if eval_s3["mean_reach"] > 0.25:
        verdict = "BREAKTHROUGH: dense-then-prune beats sparse baseline"
    elif eval_s3["mean_reach"] > 0.17:
        verdict = "IMPROVEMENT: dense-then-prune modestly better"
    elif eval_s3["mean_reach"] >= 0.12:
        verdict = "COMPARABLE: similar to sparse baseline"
    else:
        verdict = "NO IMPROVEMENT"
    print(f"\n  Verdict: {verdict}")

    out = {
        "config": {"dale_law": False, "hebbian": False,
                   "reward_reach_weight": REACH_W},
        "stage1_reach": eval_s1["mean_reach"],
        "stage2_masked_reach": eval_s2_masked["mean_reach"],
        "stage3_pruned_retrained_reach": eval_s3["mean_reach"],
        "stage1_history": hist_s1,
        "stage3_history": hist_s3,
        "n_edges_kept": int(mask.sum()),
        "edge_mask": [int(x) for x in mask],
        "total_elapsed_s": round(total, 1),
        "verdict": verdict,
    }
    with open("phase7_dense_prune_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_dense_prune_result.json")


if __name__ == "__main__":
    main()
