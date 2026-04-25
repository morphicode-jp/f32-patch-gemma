"""
kathara_mimir() — 12 environment simultaneous optimization via Kathara 12 × Zenron.

Runs 12 mimir instances in parallel, sharing via Kathara 30 edges.
Best for: similar-but-different problems (LLM prompts for 12 use cases,
portfolio of strategies, multi-tissue drug pharmacokinetics, etc.).

Complexity vs serial mimir:
  Serial:   12 × T_single  (run mimir 12 times)
  Parallel: T_single + T_share, where T_share ≈ 30 × dim × N_iter × tiny

Empirical speedup (expected):
  - If tasks independent: factor 12× (nothing to share → degenerate to parallel)
  - If tasks share structure: factor >12× (shared learning, faster convergence)

Usage:
    from twelve.agent.kathara_mimir import kathara_mimir

    eval_fns = [fn_task_0, fn_task_1, ..., fn_task_11]  # 12 evaluators
    param_ranges = [(lo, hi)] * n_dims  # shared parameter space
    r = kathara_mimir(eval_fns, param_ranges, time_budget=300)

    # r[i]["best_params"], r[i]["best_score"] for each task
"""
import os
import sys
import time
import math
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    from .mimir import mimir
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.mimir import mimir


# Module-level worker for ProcessPool (must be pickle-able)
def _process_worker(args):
    """Runs a single mimir in a child process. Args must be pickle-able."""
    import pickle as _pickle
    if len(args) == 8:
        (i, eval_fn_pkl, param_ranges, param_names, curated, budget_this,
         experience_id, extra_kwargs) = args
        use_odin = False
        curated_needs_rescore = False
    elif len(args) == 9:
        (i, eval_fn_pkl, param_ranges, param_names, curated, budget_this,
         experience_id, extra_kwargs, use_odin) = args
        curated_needs_rescore = False
    else:
        (i, eval_fn_pkl, param_ranges, param_names, curated, budget_this,
         experience_id, extra_kwargs, use_odin, curated_needs_rescore) = args
    eval_fn = _pickle.loads(eval_fn_pkl)
    if curated is not None and curated_needs_rescore:
        curated = _curated_for_node(eval_fn, curated)
    if use_odin:
        from twelve.agent.mimir_odin_stable import mimir_odin_stable as _inner_fn
    else:
        from twelve.agent.mimir import mimir as _inner_fn
    kwargs = dict(extra_kwargs or {})
    kwargs.setdefault("experience_id", experience_id)
    if curated is not None:
        kwargs["curated_measurements"] = curated
    r = _inner_fn(eval_fn, param_ranges, param_names=param_names,
                  time_budget=budget_this, verbose=False, **kwargs)
    return i, r, curated is not None


# Kathara 12 = Circulant(12, {1,4,6})
def _kathara_adjacency():
    N = 12
    A = np.zeros((N, N), dtype=np.int8)
    for s in [1, 4, 6]:
        for i in range(N):
            j = (i + s) % N
            A[i, j] = 1; A[j, i] = 1
    return A


KATHARA_ADJ = _kathara_adjacency()
KATHARA_NEIGHBORS = {i: list(np.where(KATHARA_ADJ[i] == 1)[0]) for i in range(12)}
KATHARA_N_TASKS = 12
ODIN_DEFAULT_SPECIALISTS = 4


def _effective_max_workers(max_workers, *, use_odin, mimir_kwargs,
                           auto_throttle_odin=True):
    """Cap outer workers to avoid ODIN's inner 4-way council oversubscription."""
    requested = KATHARA_N_TASKS if max_workers is None else int(max_workers)
    requested = max(1, min(KATHARA_N_TASKS, requested))
    if not use_odin or not auto_throttle_odin:
        return requested

    specialists = mimir_kwargs.get("specialists")
    inner_width = len(specialists) if specialists is not None else ODIN_DEFAULT_SPECIALISTS
    inner_width = max(1, inner_width)
    cpu_count = os.cpu_count() or KATHARA_N_TASKS
    cpu_cap = max(1, cpu_count // inner_width)
    return max(1, min(requested, cpu_cap))


def _node_budget_for_round(round_budget, remaining, max_workers, rounds_left=1):
    """Per-node budget that respects wall budget when nodes run in waves."""
    waves = math.ceil(KATHARA_N_TASKS / max(1, max_workers))
    return min(round_budget, (remaining * 0.95) / (waves * max(1, rounds_left)))


def _share_via_kathara(all_best_params, all_best_scores, node_i, share_weight=0.3):
    """
    Node i receives a share vector from its Kathara neighbors (5 nodes).
    share vector = weighted avg of neighbors' best params, weights ∝ their score.
    """
    neighbors = KATHARA_NEIGHBORS[node_i]
    self_params = np.asarray(all_best_params[node_i], dtype=float)
    nbr_scores = np.array([all_best_scores[j] for j in neighbors], dtype=float)
    nbr_params = np.array([all_best_params[j] for j in neighbors], dtype=float)
    finite = np.isfinite(nbr_scores)
    if finite.any():
        floor = float(nbr_scores[finite].min())
        safe_scores = np.where(finite, nbr_scores, floor)
        w = safe_scores - safe_scores.min() + 1e-6
        w = w / w.sum()
    else:
        w = np.full(len(neighbors), 1.0 / len(neighbors), dtype=float)
    share = (w[:, None] * nbr_params).sum(axis=0)  # (n_dims,)
    # Blend: (1-share_weight) × self + share_weight × share
    return (1.0 - share_weight) * self_params + share_weight * share


def _curated_for_node(eval_fn, candidates):
    """Re-score shared params with this node's own eval_fn before curation."""
    curated = []
    seen = set()
    for params in candidates:
        params_list = np.asarray(params, dtype=float).tolist()
        key = tuple(round(float(v), 12) for v in params_list)
        if key in seen:
            continue
        seen.add(key)
        try:
            score = float(eval_fn(params_list))
        except Exception:
            continue
        if not math.isfinite(score):
            continue
        curated.append({"params": params_list, "score": score})
    return curated or None


def _share_candidates_for_node(all_best_params, all_best_scores, node_i,
                               share_weight, max_curated_per_node):
    """Build a small candidate set: self, blended share, and top neighbors."""
    share_vec = _share_via_kathara(
        all_best_params, all_best_scores, node_i,
        share_weight=share_weight,
    )
    candidates = [all_best_params[node_i], share_vec.tolist()]
    neighbors = KATHARA_NEIGHBORS[node_i]
    if max_curated_per_node is None:
        selected_neighbors = neighbors
    else:
        room = max(0, int(max_curated_per_node) - len(candidates))
        selected_neighbors = sorted(
            neighbors,
            key=lambda j: (
                all_best_scores[j]
                if math.isfinite(float(all_best_scores[j]))
                else float("-inf")
            ),
            reverse=True,
        )[:room]
    candidates.extend(all_best_params[j] for j in selected_neighbors)
    return candidates


def kathara_mimir(
    eval_fns,
    param_ranges,
    *,
    param_names=None,
    time_budget=300.0,
    share_rounds=3,
    share_weight=0.3,
    max_workers=12,
    verbose=False,
    mimir_kwargs=None,
    use_processes=False,
    use_odin=False,
    max_curated_per_node=4,
    auto_throttle_odin=True,
):
    """
    Run 12 mimir instances in parallel on Kathara 12 topology, sharing learning.

    Args:
      eval_fns: list of 12 callables, each (params: list[float]) -> score
      param_ranges: [(lo, hi)] * n_dims, shared across tasks
      param_names: optional list of dim names
      time_budget: total wall-time budget (seconds)
      share_rounds: number of share phases (per-round budget = total / share_rounds)
      share_weight: [0,1] how much to blend neighbor info (0=none, 1=all)
      max_workers: parallel thread count (12 recommended)
      mimir_kwargs: extra kwargs passed to each mimir() call
      max_curated_per_node: max re-scored share points per node/round.
        Default 4 = self + blended share + top 2 neighbors.
      auto_throttle_odin: cap outer workers for use_odin=True to avoid
        12 nodes × 4 specialists oversubscription.

    Returns:
      list of 12 dicts, each with mimir()-style keys plus "kathara_shares_received"
    """
    if len(eval_fns) != 12:
        raise ValueError(f"Need exactly 12 eval_fns (Kathara 12), got {len(eval_fns)}")
    if not 0.0 <= share_weight <= 1.0:
        raise ValueError(f"share_weight must be in [0, 1], got {share_weight}")
    if max_curated_per_node is not None and max_curated_per_node < 2:
        raise ValueError(
            "max_curated_per_node must be None or >= 2 "
            f"(got {max_curated_per_node})"
        )

    mimir_kwargs = dict(mimir_kwargs or {})
    if use_odin and "executor" not in mimir_kwargs:
        mimir_kwargs["executor"] = "thread"
    effective_max_workers = _effective_max_workers(
        max_workers,
        use_odin=use_odin,
        mimir_kwargs=mimir_kwargs,
        auto_throttle_odin=auto_throttle_odin,
    )

    # Per-round budget (reserve tiny overhead for share ops)
    round_budget = max(2.0, (time_budget * 0.95) / share_rounds)

    # Initialize: current best params per task (start from midpoint)
    lo = np.array([r[0] for r in param_ranges])
    hi = np.array([r[1] for r in param_ranges])
    current_best_params = [((lo + hi) / 2).tolist() for _ in range(12)]
    current_best_scores = [-1e30] * 12
    shares_received = [0] * 12
    all_results = [None] * 12
    rounds_completed = 0
    last_budget_this = 0.0
    eval_fn_pickles = [pickle.dumps(fn) for fn in eval_fns] if use_processes else None

    t_start = time.time()

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with Executor(max_workers=effective_max_workers) as pool:
      for round_idx in range(share_rounds):
        elapsed = time.time() - t_start
        remaining = time_budget - elapsed
        if remaining < 1.0:
            break
        budget_this = _node_budget_for_round(
            round_budget,
            remaining,
            effective_max_workers,
            rounds_left=share_rounds - round_idx,
        )
        last_budget_this = budget_this

        if verbose:
            print(f"[kathara_mimir] Round {round_idx + 1}/{share_rounds}, "
                  f"budget {budget_this:.1f}s each, "
                  f"{effective_max_workers} workers...")

        # For each node, prepare warm-start:
        #   use share vector from Kathara neighbors (if round > 0)
        candidates_per_node = []
        for i in range(12):
            if round_idx == 0:
                candidates_per_node.append(None)
            else:
                candidates = _share_candidates_for_node(
                    current_best_params, current_best_scores, i,
                    share_weight=share_weight,
                    max_curated_per_node=max_curated_per_node,
                )
                candidates_per_node.append(candidates)

        # Pick inner optimizer: mimir (default) or mimir_odin_stable (council)
        if use_odin:
            try:
                from .mimir_odin_stable import mimir_odin_stable as _inner_fn
            except ImportError:
                from twelve.agent.mimir_odin_stable import mimir_odin_stable as _inner_fn
        else:
            _inner_fn = mimir

        def _run_node_thread(i):
            kwargs = dict(mimir_kwargs)
            kwargs.setdefault("experience_id", f"kmimir_{i}_round{round_idx}")
            curated = None
            if candidates_per_node[i] is not None:
                curated = _curated_for_node(eval_fns[i], candidates_per_node[i])
            if curated is not None:
                kwargs["curated_measurements"] = curated
            return i, _inner_fn(
                eval_fns[i], param_ranges,
                param_names=param_names,
                time_budget=budget_this,
                verbose=False,
                **kwargs,
            ), curated is not None

        if use_processes:
            # ProcessPool: true parallelism, bypasses GIL
            # eval_fns must be pickle-able (top-level functions, not lambdas/closures)
            worker_args = [
                (i, eval_fn_pickles[i], param_ranges, param_names,
                 candidates_per_node[i], budget_this,
                 f"kmimir_{i}_round{round_idx}", mimir_kwargs, use_odin, True)
                for i in range(12)
            ]
            futures = [pool.submit(_process_worker, args) for args in worker_args]
            for fut in as_completed(futures):
                i, r, share_counted = fut.result()
                if r["best_score"] > current_best_scores[i]:
                    current_best_scores[i] = r["best_score"]
                    current_best_params[i] = (
                        list(r["best_params"].values())
                        if isinstance(r["best_params"], dict)
                        else list(r["best_params"])
                    )
                if share_counted:
                    shares_received[i] += 1
                all_results[i] = r
        else:
            # ThreadPool: GIL-limited but avoids pickling constraint
            futures = {pool.submit(_run_node_thread, i): i for i in range(12)}
            for fut in as_completed(futures):
                i, r, share_counted = fut.result()
                if r["best_score"] > current_best_scores[i]:
                    current_best_scores[i] = r["best_score"]
                    current_best_params[i] = (
                        list(r["best_params"].values())
                        if isinstance(r["best_params"], dict)
                        else list(r["best_params"])
                    )
                if share_counted:
                    shares_received[i] += 1
                all_results[i] = r

        if verbose:
            best_global = max(current_best_scores)
            worst_global = min(current_best_scores)
            print(f"  Round {round_idx + 1} done. Best: {best_global:.4f}, "
                  f"Worst: {worst_global:.4f}, "
                  f"elapsed: {time.time() - t_start:.1f}s")
        rounds_completed += 1

    # Attach share count to each result
    for i in range(12):
        if all_results[i] is None:
            all_results[i] = {
                "best_params": current_best_params[i],
                "best_score": current_best_scores[i],
                "tool_used": "kathara_mimir_incomplete",
            }
        all_results[i]["kathara_shares_received"] = shares_received[i]
        all_results[i]["kathara_node"] = i
        all_results[i]["kathara_effective_max_workers"] = effective_max_workers
        all_results[i]["kathara_rounds_completed"] = rounds_completed
        all_results[i]["kathara_last_node_budget_s"] = last_budget_this
        all_results[i]["kathara_max_curated_per_node"] = max_curated_per_node
        if use_odin:
            all_results[i]["kathara_inner_executor"] = mimir_kwargs.get("executor")

    total_time = time.time() - t_start
    if verbose:
        print(f"[kathara_mimir] Done. Total wall time: {total_time:.1f}s "
              f"(budget was {time_budget}s)")

    return all_results


# Complexity analysis helper
def compute_expected_speedup(n_tasks=12, dim=10, share_cost_per_share=None):
    """
    Estimate speedup of kathara_mimir vs serial mimir.

    Let T_single = single-task optimization time.
    Serial: N × T_single
    Kathara: T_single × (1 + share_rounds × share_overhead)
            where share_overhead ≈ dim × 30_edges / eval_count_per_round

    Typical: share_overhead ~ 10^-4 (share is negligible)
    So speedup ≈ N / (1 + small) ≈ N
    """
    if share_cost_per_share is None:
        share_cost_per_share = dim * 30e-6  # microseconds per share op
    share_rounds = 3
    total_share_cost = share_rounds * share_cost_per_share
    # Approximate single-task mimir time: 300s typical
    T_single = 300
    serial_cost = n_tasks * T_single
    parallel_cost = T_single + total_share_cost
    speedup = serial_cost / parallel_cost
    return {
        "serial_seconds": serial_cost,
        "parallel_seconds": parallel_cost,
        "share_overhead_seconds": total_share_cost,
        "speedup_factor": speedup,
    }
