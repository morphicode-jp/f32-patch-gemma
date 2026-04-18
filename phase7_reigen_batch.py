"""phase7_reigen_batch.py - Reigen × Phase 7 with multiprocessing batch_eval_fn.

Uses ProcessPoolExecutor to run inner Phase 7 evaluations in parallel.
Each worker process independently applies its own Dale's law / reward patches.

Per CLAUDE.md Rule 11: batch_eval_fn works for EXTERNAL params (our
dale_law, hebbian, reach_w are all external config). No shared model state
between workers → safe to parallelize.

Expected speedup: i7-12700K with 6 workers → 4-6x wall time reduction.
"""
import os
import sys
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Worker function (module-level for pickleability)
# ============================================================

def _patch_reward_in_worker(reach_weight):
    """Per-process monkey-patch of phase7_coop_sentinel.run_coop_match."""
    import numpy as np
    import phase7_coop_sentinel as pcs
    from phase7_coop_sentinel import brain_step, extract_motor
    from kathara16_brain import N_NODES_16, KATHARA16_EDGES
    from multi_agent_world_N import MultiAgentCoopWorldN

    reach_w = reach_weight * 100
    partner_w = (1 - reach_weight) * 40
    approach_w = (1 - reach_weight) * 30
    synergy_w = (1 - reach_weight) * 20

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
                (reach_w if reached_i else 0)
                + avg_partner_reached * partner_w
                + own_approaches[i] * approach_w
                + (synergy_w if all_r else 0)
            )
            per_agent_scores.append(score)
        final_dists = [world.get_food_dist(i) for i in range(N)]
        if record_traces:
            return per_agent_scores, reached, final_dists, traces
        return per_agent_scores, reached, final_dists

    pcs.run_coop_match = patched_run_coop_match


def worker_eval(user_params_tuple):
    """Pickleable worker. Runs full Phase 7 inner eval.

    Each worker independently patches Dale + reward per its params.
    Returns composite fitness score.
    """
    import numpy as np
    import kathara16_brain
    from phase7_coop_sentinel import train_phase7
    from phase7_evaluation import (
        evaluate_standard, evaluate_mi_panel, evaluate_tom
    )

    try:
        dale_val, hebb_val, reach_w = user_params_tuple
        dale = dale_val > 0.5
        hebb = hebb_val > 0.5

        # Patch Dale's law (always restore original at start)
        if dale:
            # Default Kathara(16) Dale's law: nodes 0, 4, 8 inhibitory
            sign = np.ones(16, dtype=np.float64)
            for n in [0, 4, 8]:
                sign[n] = -1.0
            kathara16_brain.DEFAULT_INHIBIT_16 = sign
        else:
            kathara16_brain.DEFAULT_INHIBIT_16 = np.ones(16, dtype=np.float64)

        # Patch reward
        _patch_reward_in_worker(reach_w)

        # Train
        params_list, _ = train_phase7(
            N_AGENTS=3, n_cycles=1, budget_per_agent=15,
            use_hebbian=hebb, verbose=False,
        )

        # Evaluate
        std = evaluate_standard(params_list, n_seeds=10)
        mi = evaluate_mi_panel(params_list, n_episodes=15, n_perm=50)
        tom = evaluate_tom(params_list, n_episodes=10)

        reach = float(std["mean_agent_reach_rate"])
        mi_gain = max(0.0, float(mi["mean_gain"]))
        tom_pair = float(tom["pair_fraction_predictive"])
        composite = 0.5 * reach + 0.3 * tom_pair + 0.2 * mi_gain
        return {
            "composite": composite,
            "reach": reach, "tom": tom_pair, "mi": mi_gain,
            "dale": bool(dale), "hebb": bool(hebb), "reach_w": float(reach_w),
        }
    except Exception as e:
        return {"composite": 0.0, "error": str(e)}


# ============================================================
# Reigen driver (main process)
# ============================================================

# Global state: dispatcher for batch calls
_BATCH_WORKERS = 6
_BATCH_EXECUTOR = None


def get_batch_executor():
    """Lazy singleton executor."""
    global _BATCH_EXECUTOR
    if _BATCH_EXECUTOR is None:
        _BATCH_EXECUTOR = ProcessPoolExecutor(max_workers=_BATCH_WORKERS)
    return _BATCH_EXECUTOR


def batch_eval_fn(params_list_of_lists):
    """Reigen batch interface.

    params_list_of_lists: list of N user_param vectors, shape (N, 3)
    Returns: list of N composite scores (floats)
    """
    executor = get_batch_executor()
    tuples = [tuple(float(v) for v in p) for p in params_list_of_lists]

    # Submit and collect in original order
    futures = [executor.submit(worker_eval, t) for t in tuples]
    results = [f.result() for f in futures]

    for i, r in enumerate(results):
        score = r.get("composite", 0.0)
        if "error" in r:
            print(f"  [batch][{i}] ERROR: {r['error']}")
        else:
            print(f"  [batch][{i}] dale={r['dale']} hebb={r['hebb']} "
                  f"reach_w={r['reach_w']:.2f} -> "
                  f"reach={r['reach']*100:.0f}% tom={r['tom']*100:.0f}% "
                  f"mi={r['mi']:+.3f} composite={score:.3f}")
    return [r.get("composite", 0.0) for r in results]


def scalar_eval_fn(user_params):
    """Fallback scalar version (called when batch unavailable)."""
    r = worker_eval(tuple(user_params))
    return r.get("composite", 0.0)


def guard_fn(user_params):
    return 0.0


def main():
    from twelve.agent.reigen import reigen

    print("=" * 70)
    print("  REIGEN × PHASE 7 with multiprocessing batch_eval_fn")
    print("=" * 70)
    print(f"  Workers: {_BATCH_WORKERS}")
    print(f"  User params: dale_law, hebbian, reward_reach_weight")

    user_ranges = [(0.0, 1.0), (0.0, 1.0), (0.3, 0.9)]
    user_names = ["dale_law", "hebbian", "reward_reach_weight"]

    t0 = time.time()
    result = reigen(
        eval_fn=scalar_eval_fn,
        guard_fn=guard_fn,
        user_param_ranges=user_ranges,
        user_param_names=user_names,
        experience_id="genesis",
        inner_time_budget=2,
        time_budget=1200,
        wall_time_factor=2.0,
        batch_eval_fn=batch_eval_fn,
        batch_size=_BATCH_WORKERS,
    )
    elapsed = time.time() - t0

    # Cleanup executor
    global _BATCH_EXECUTOR
    if _BATCH_EXECUTOR is not None:
        _BATCH_EXECUTOR.shutdown(wait=True)

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Verdict: {result.get('verdict')}")
    best_user = result.get("user_best_params") or []
    print(f"\n  Best user params:")
    for n, v in zip(user_names, best_user):
        print(f"    {n} = {v:.3f}")
    print(f"  Best eval: {result.get('user_best_score', 0):.3f}")

    # Derive best human-readable
    if best_user:
        dale, hebb, rw = best_user
        print(f"\n  Decoded: dale_law={'ON' if dale>0.5 else 'OFF'}, "
              f"hebbian={'ON' if hebb>0.5 else 'OFF'}, "
              f"reach_w={rw:.2f}")

    self_best = result.get("self_best_params") or {}
    diag = result.get("self_diagnostic") or {}

    out = {
        "elapsed_s": round(elapsed, 1),
        "workers": _BATCH_WORKERS,
        "user_best_params": [float(x) for x in (best_user or [])],
        "user_best_names": user_names,
        "user_best_score": float(result.get("user_best_score", 0) or 0),
        "self_best_params": {k: float(v) for k, v in self_best.items()},
        "self_diagnostic": diag,
        "verdict": result.get("verdict"),
    }
    with open("phase7_reigen_batch_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_reigen_batch_result.json")


if __name__ == "__main__":
    main()
