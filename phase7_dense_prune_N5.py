"""phase7_dense_prune_N5.py - Dense-then-Prune at N=5 scale with full evaluation.

Extends phase7_dense_prune.py to N=5 with the full 4-criterion
evaluation (reach, MI, ToM, scales).

Also monkey-patches phase7_coop_sentinel.run_coop_match to route through
dense brain, enabling phase7_evaluation suite to run on dense agents.

Saves intermediate state after each stage (crash-resilient).
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Monkey-patch dense brain into phase7_coop_sentinel
# ============================================================
import kathara_dense_brain as kdb
from kathara_dense_brain import (
    PARAM_RANGES_DENSE, PARAM_NAMES_DENSE, TOTAL_PARAMS_DENSE,
    N_NODES_DENSE, DENSE_EDGES, simulate_step_dense,
)
from multi_agent_world_N import MultiAgentCoopWorldN

# Reigen-optimal: dale=OFF
kdb.DEFAULT_INHIBIT_DENSE = np.ones(N_NODES_DENSE, dtype=np.float64)

REACH_W = 0.60
_r = REACH_W * 100
_p = (1 - REACH_W) * 40
_a = (1 - REACH_W) * 30
_s = (1 - REACH_W) * 20
_EDGE_MASK = None


def _brain_step_dense(params, sensors, states):
    return simulate_step_dense(
        params, sensors, states,
        inhibit_sign=kdb.DEFAULT_INHIBIT_DENSE,
        edge_mask=_EDGE_MASK,
    )


def _extract_motor(firing):
    return float(firing[5]), float(firing[11]), float(firing[1])


def patched_run_coop_match(params_list, seed, n_steps=50,
                           use_hebbian=False, record_traces=False):
    """Replaces phase7_coop_sentinel.run_coop_match with dense version.
    Also used by phase7_evaluation.evaluate_mi_panel / evaluate_tom.
    """
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
            states[i], firing = _brain_step_dense(
                params_list[i], sensors_list[i], states[i]
            )
            nav, speed, voice = _extract_motor(firing)
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
            (_r if reached[i] else 0)
            + avg_partner * _p
            + own_approaches[i] * _a
            + (_s if all_r else 0)
        )
        per_agent_scores.append(score)
    final_dists = [world.get_food_dist(i) for i in range(N)]
    if record_traces:
        return per_agent_scores, reached, final_dists, traces
    return per_agent_scores, reached, final_dists


# ============================================================
# Patch brain_step so train_phase7 uses dense brain
# ============================================================
import phase7_coop_sentinel as pcs

def _patched_brain_step(params, sensors, states, w_adapt=None,
                         reward=None, use_hebbian=False, k=3):
    return _brain_step_dense(params, sensors, states)


pcs.brain_step = _patched_brain_step
pcs.run_coop_match = patched_run_coop_match

# Now imports use patched versions
from phase7_coop_sentinel import train_phase7
from phase7_evaluation import (
    evaluate_standard, evaluate_mi_panel, evaluate_tom, evaluate_scales
)


# ============================================================
# Main
# ============================================================

def main():
    global _EDGE_MASK
    print("=" * 70)
    print("  DENSE-THEN-PRUNE N=5 (full evaluation)")
    print("=" * 70)
    print(f"  Config: dale=OFF, hebb=OFF, reach_w={REACH_W}")

    t_all = time.time()
    N_AGENTS = 5
    BUDGET = 30  # per agent per round
    N_CYCLES = 1  # per stage

    def save_snapshot(stage, params, mask, history, elapsed):
        """Crash-resilient intermediate save."""
        out = {
            "stage": stage,
            "n_agents": N_AGENTS,
            "elapsed_s": round(elapsed, 1),
            "edge_mask": ([int(x) for x in mask] if mask is not None else None),
            "history": history,
            "params": [[float(x) for x in p] for p in params],
        }
        with open(f"phase7_dense_prune_N5_{stage}.json", "w") as f:
            json.dump(out, f, indent=2, default=str)

    # ==========
    # STAGE 1: dense (120 edges)
    # ==========
    _EDGE_MASK = None
    print(f"\n{'='*70}\n  STAGE 1: Dense (120 edges), {N_AGENTS} agents\n{'='*70}")
    t1 = time.time()
    # We need to override the param ranges used by train_phase7 since it uses
    # sparse Kathara(16) by default. Patch the module's PARAM_RANGES / PARAM_NAMES:
    pcs.PARAM_RANGES_16 = PARAM_RANGES_DENSE
    pcs.PARAM_NAMES_16 = PARAM_NAMES_DENSE

    params_s1, hist_s1 = train_phase7(
        N_AGENTS=N_AGENTS, n_cycles=N_CYCLES, budget_per_agent=BUDGET,
        use_hebbian=False, verbose=True,
    )
    print(f"\n  Stage 1 elapsed: {time.time()-t1:.0f}s")
    save_snapshot("stage1", params_s1, None, hist_s1, time.time() - t_all)

    # ==========
    # STAGE 2: prune
    # ==========
    print(f"\n{'='*70}\n  STAGE 2: Prune (keep top 40% by |weight|)\n{'='*70}")
    n_edges = len(DENSE_EDGES)
    edge_mags = np.zeros(n_edges)
    for p in params_s1:
        edge_mags += np.abs(np.asarray(p[:n_edges]))
    edge_mags /= N_AGENTS
    threshold = np.percentile(edge_mags, 60)
    mask = (edge_mags >= threshold).astype(np.float64)
    _EDGE_MASK = mask
    n_kept = int(mask.sum())
    print(f"  Edges kept: {n_kept}/{n_edges} (threshold |w|>={threshold:.3f})")

    # ==========
    # STAGE 3: retrain pruned
    # ==========
    print(f"\n{'='*70}\n  STAGE 3: Retrain pruned ({n_kept} edges)\n{'='*70}")
    t3 = time.time()
    params_s3, hist_s3 = train_phase7(
        N_AGENTS=N_AGENTS, n_cycles=N_CYCLES, budget_per_agent=BUDGET,
        use_hebbian=False, verbose=True,
        init_params_list=params_s1,
    )
    print(f"\n  Stage 3 elapsed: {time.time()-t3:.0f}s")
    save_snapshot("stage3", params_s3, mask, hist_s3, time.time() - t_all)

    # ==========
    # FULL EVALUATION on Stage 3 params
    # ==========
    print(f"\n{'='*70}\n  FULL EVALUATION (Stage 3 pruned brain)\n{'='*70}")

    print("\n[1/4] Standard task...")
    std = evaluate_standard(params_s3, n_seeds=15)
    print(f"  mean reach: {std['mean_agent_reach_rate']*100:.0f}%")
    print(f"  per-agent: {[f'{r*100:.0f}%' for r in std['per_agent_reach_rate']]}")

    print("\n[2/4] MI panel...")
    mi = evaluate_mi_panel(params_s3, n_episodes=20, n_perm=50)
    print(f"  mean MI gain: {mi['mean_gain']:+.4f}")
    print(f"  max MI gain: {mi['max_gain']:+.4f}")

    print("\n[3/4] ToM test...")
    tom = evaluate_tom(params_s3, n_episodes=15)
    print(f"  predictive pairs: {tom['n_pairs_predictive']}/{tom['n_pairs_measured']} "
          f"({tom['pair_fraction_predictive']*100:.0f}%)")
    print(f"  max gain: {tom['max_gain']:+.4f}")

    print("\n[4/4] Scales test (N=5 -> N=10)...")
    sc = evaluate_scales(params_s3, target_N=10, n_seeds=8)
    print(f"  N=10 mean reach: {sc['mean_agent_reach_rate']*100:.0f}%")

    # Criteria
    c1 = std["mean_agent_reach_rate"] > 0.8
    c2 = mi["mean_gain"] > 0.08
    c3 = tom["pair_fraction_predictive"] > 0.45
    c4 = sc["mean_agent_reach_rate"] > 0.6
    n_met = sum(1 for x in [c1, c2, c3, c4] if x)

    print(f"\n{'='*70}\n  PRE-REGISTERED CRITERIA\n{'='*70}")
    print(f"  [{'v' if c1 else 'x'}] 1. Reach > 80%:    {std['mean_agent_reach_rate']*100:.0f}%")
    print(f"  [{'v' if c2 else 'x'}] 2. MI > 0.08:      {mi['mean_gain']:+.4f}")
    print(f"  [{'v' if c3 else 'x'}] 3. ToM > 45%:      {tom['pair_fraction_predictive']*100:.0f}%")
    print(f"  [{'v' if c4 else 'x'}] 4. Scales > 60%:   {sc['mean_agent_reach_rate']*100:.0f}%")
    print(f"\n  CRITERIA MET: {n_met}/4")

    if n_met >= 2:
        verdict = "PHASE 7 PARTIAL SUCCESS (>=2/4)"
    else:
        verdict = "PHASE 7 INCOMPLETE"
    print(f"  Verdict: {verdict}")

    total = time.time() - t_all
    out = {
        "config": {"dale_law": False, "hebbian": False, "reach_w": REACH_W,
                   "n_agents": N_AGENTS, "n_cycles_per_stage": N_CYCLES,
                   "budget_per_agent_s": BUDGET},
        "n_edges_dense": n_edges,
        "n_edges_kept": n_kept,
        "edge_mask": [int(x) for x in mask],
        "stage1_history": hist_s1,
        "stage3_history": hist_s3,
        "evaluation": {
            "standard": std,
            "mi_panel": mi,
            "tom": tom,
            "scales": sc,
        },
        "criteria_met": n_met,
        "verdict": verdict,
        "total_elapsed_s": round(total, 1),
        "final_params": [[float(x) for x in p] for p in params_s3],
    }
    with open("phase7_dense_prune_N5_full.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Total: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Saved: phase7_dense_prune_N5_full.json")


if __name__ == "__main__":
    main()
