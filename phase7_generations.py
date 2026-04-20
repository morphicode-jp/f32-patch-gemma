"""phase7_generations.py - Generational evolution with brain param inheritance.

DNA/copy primitive made central. Unlike prior experiments where each eval
starts from scratch (midpoint), here offspring INHERIT parent brain params
with Gaussian mutation. Selection pressure across generations accumulates
prior.

Design (Option-α from アンダーワールド計画 discussion):

  Individual = a team of N=3 agent brains (3 × 129D params)
  Population = 12 teams
  Gen 0: 12 random teams (midpoint + Gaussian noise)
  Gen 1+: top-4 teams × 3 children = 12 new teams
    - Child params = parent team params + Gaussian(0, sigma * param_range)
    - Each child team trains short (30s/agent, 1 cycle)
    - Evaluated on reach/ToM/scales like prior runs

Fast settings:
  - Kathara(16, {3,6,7}) sparse substrate (129D, original Phase 7)
  - No Dense-Prune (too complex to combine with gen evolution)
  - train_phase7: n_cycles=1, budget=30s/agent → ~90s per team
  - 12 teams / 6 workers = 2 batches per gen
  - 2 × 90s = 3min/gen
  - 8 generations × 3min = ~24 min total
"""
import os
import sys
import json
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


POP_SIZE = 12
N_GENS = 8
TOP_K = 4
N_CHILDREN = 3  # 4 × 3 = 12
N_AGENTS = 3
MUTATION_SIGMA = 0.10  # fraction of each param's range
BUDGET_PER_AGENT_S = 30
N_CYCLES = 1


def make_midpoint_team(seed):
    """Return N_AGENTS param vectors, each = midpoint + small jitter."""
    from kathara16_brain import PARAM_RANGES_16
    rng = np.random.default_rng(seed)
    team = []
    for a in range(N_AGENTS):
        params = []
        for lo, hi in PARAM_RANGES_16:
            mid = (lo + hi) / 2
            jitter = rng.normal(0, (hi - lo) * 0.1)
            params.append(max(lo, min(hi, mid + jitter)))
        team.append(params)
    return team


def mutate_team(parent_team, seed):
    """Gaussian mutation on each param, clipped to range."""
    from kathara16_brain import PARAM_RANGES_16
    rng = np.random.default_rng(seed)
    child = []
    for parent_params in parent_team:
        cp = []
        for v, (lo, hi) in zip(parent_params, PARAM_RANGES_16):
            step = rng.normal(0, (hi - lo) * MUTATION_SIGMA)
            cp.append(max(lo, min(hi, v + step)))
        child.append(cp)
    return child


def worker_eval_team(args):
    """Train one team from inherited params, return trained + metrics."""
    team_init, gen_idx, indiv_idx = args
    try:
        import numpy as np
        import time
        import kathara16_brain
        kathara16_brain.DEFAULT_INHIBIT_16 = np.ones(16, dtype=np.float64)

        from phase7_coop_sentinel import train_phase7
        from phase7_evaluation import evaluate_standard, evaluate_tom, evaluate_scales

        t0 = time.time()
        trained_params, _hist = train_phase7(
            N_AGENTS=N_AGENTS, n_cycles=N_CYCLES,
            budget_per_agent=BUDGET_PER_AGENT_S,
            use_hebbian=False, verbose=False,
            init_params_list=team_init,
        )
        train_elapsed = time.time() - t0

        std = evaluate_standard(trained_params, n_seeds=6)
        tom = evaluate_tom(trained_params, n_episodes=8)
        sc = evaluate_scales(trained_params, target_N=5, n_seeds=4)

        reach = float(std["mean_agent_reach_rate"])
        tom_pair = float(tom["pair_fraction_predictive"])
        scales = float(sc["mean_agent_reach_rate"])
        composite = 0.5 * reach + 0.3 * tom_pair + 0.2 * scales

        return {
            "gen": gen_idx, "indiv": indiv_idx,
            "trained_params": [list(p) for p in trained_params],
            "reach": reach, "tom": tom_pair, "scales": scales,
            "composite": composite,
            "train_elapsed": round(train_elapsed, 1),
        }
    except Exception as e:
        import traceback
        return {"gen": gen_idx, "indiv": indiv_idx,
                "error": str(e)[:150],
                "traceback": traceback.format_exc()[:400]}


def batch_evaluate_gen(teams_with_ids, workers=6):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_eval_team, args) for args in teams_with_ids]
        results = [f.result() for f in futures]
    elapsed = time.time() - t0
    return results, elapsed


def main():
    print("=" * 70, flush=True)
    print(f"  PHASE 7 GENERATIONS (DNA/copy pipeline)", flush=True)
    print(f"  Pop={POP_SIZE}, Gens={N_GENS}, Top-{TOP_K} × {N_CHILDREN} children",
          flush=True)
    print(f"  N_AGENTS={N_AGENTS}, budget={BUDGET_PER_AGENT_S}s/agent × "
          f"{N_CYCLES} cycle, σ={MUTATION_SIGMA}", flush=True)
    print("=" * 70, flush=True)

    t_all = time.time()
    rng = random.Random(42)

    # Gen 0: random population
    population_params = [make_midpoint_team(seed=rng.randint(0, 10**9))
                         for _ in range(POP_SIZE)]

    gen_history = []
    all_results = []

    for gen in range(N_GENS):
        print(f"\n[GEN {gen}] Evaluating {POP_SIZE} teams", flush=True)

        # Build args list
        args_list = [(team, gen, i) for i, team in enumerate(population_params)]

        # Evaluate in batches of 6
        gen_results = []
        BATCH = 6
        for batch_idx in range(0, len(args_list), BATCH):
            batch = args_list[batch_idx:batch_idx + BATCH]
            results, elapsed = batch_evaluate_gen(batch, workers=BATCH)
            gen_results.extend(results)
            print(f"  batch {batch_idx//BATCH+1}: {elapsed:.0f}s ({elapsed/60:.1f}min)",
                  flush=True)

        # Log per-indiv
        valid_gen = [r for r in gen_results if "error" not in r]
        for r in valid_gen:
            print(f"    [G{gen}-i{r['indiv']:02d}] "
                  f"reach={r['reach']*100:.0f}% tom={r['tom']*100:.0f}% "
                  f"scales={r['scales']*100:.0f}% C={r['composite']:.3f}",
                  flush=True)

        # Rank + stats
        valid_gen.sort(key=lambda r: -r.get("composite", 0))
        reaches = [r["reach"] for r in valid_gen]
        composites = [r["composite"] for r in valid_gen]
        print(f"\n  [GEN {gen} STATS] "
              f"n_valid={len(valid_gen)}/{len(gen_results)}, "
              f"reach: mean {np.mean(reaches)*100:.1f}% "
              f"max {np.max(reaches)*100:.0f}%, "
              f"composite: mean {np.mean(composites):.3f} "
              f"max {np.max(composites):.3f}", flush=True)

        gen_history.append({
            "gen": gen,
            "n_valid": len(valid_gen),
            "reach_mean": float(np.mean(reaches)) if reaches else 0.0,
            "reach_max": float(np.max(reaches)) if reaches else 0.0,
            "tom_mean": float(np.mean([r["tom"] for r in valid_gen])) if valid_gen else 0.0,
            "composite_mean": float(np.mean(composites)) if composites else 0.0,
            "composite_max": float(np.max(composites)) if composites else 0.0,
        })
        all_results.extend(gen_results)

        # Stop if last gen
        if gen == N_GENS - 1:
            break

        # Select top-K, breed children for next gen
        top_k = valid_gen[:TOP_K]
        print(f"\n  [BREED] Top-{TOP_K} → {N_CHILDREN} children each = "
              f"{TOP_K*N_CHILDREN} new teams", flush=True)
        next_pop = []
        for ki, parent in enumerate(top_k):
            for ci in range(N_CHILDREN):
                child = mutate_team(parent["trained_params"],
                                    seed=rng.randint(0, 10**9))
                next_pop.append(child)
        # Fill if short
        while len(next_pop) < POP_SIZE:
            next_pop.append(make_midpoint_team(seed=rng.randint(0, 10**9)))
        population_params = next_pop[:POP_SIZE]

    # Final
    total = time.time() - t_all
    valid_all = [r for r in all_results if "error" not in r]
    valid_all.sort(key=lambda r: -r.get("composite", 0))
    best = valid_all[0] if valid_all else None

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL ({total:.0f}s = {total/60:.1f}min, "
          f"{len(all_results)} evals, {len(valid_all)} valid)", flush=True)
    print(f"{'='*70}", flush=True)

    if best:
        print(f"\n  BEST: gen {best['gen']} indiv {best['indiv']}", flush=True)
        print(f"    reach = {best['reach']*100:.0f}%", flush=True)
        print(f"    tom   = {best['tom']*100:.0f}%", flush=True)
        print(f"    scales= {best['scales']*100:.0f}%", flush=True)
        print(f"    comp  = {best['composite']:.3f}", flush=True)

    # Reach evolution curve
    print(f"\n  REACH EVOLUTION (does prior accumulate?):", flush=True)
    print(f"    gen | reach_mean | reach_max | comp_mean", flush=True)
    for h in gen_history:
        print(f"    {h['gen']:3d} |  {h['reach_mean']*100:6.1f}%  |  "
              f"{h['reach_max']*100:5.0f}%  | {h['composite_mean']:.3f}",
              flush=True)

    # Comparison
    if gen_history:
        first_mean = gen_history[0]["reach_mean"]
        last_mean = gen_history[-1]["reach_mean"]
        first_max = gen_history[0]["reach_max"]
        last_max = gen_history[-1]["reach_max"]
        print(f"\n  DELTA: mean {first_mean*100:.1f}% → "
              f"{last_mean*100:.1f}% "
              f"({'+' if last_mean>first_mean else ''}"
              f"{(last_mean-first_mean)*100:.1f}pt)", flush=True)
        print(f"         max  {first_max*100:.0f}% → {last_max*100:.0f}% "
              f"({'+' if last_max>first_max else ''}"
              f"{(last_max-first_max)*100:.0f}pt)", flush=True)
        print(f"  Prior Kathara(16) seed-avg ceiling: 15-17% mean, ~22% max",
              flush=True)

    out = {
        "total_elapsed_s": round(total, 1),
        "pop_size": POP_SIZE, "n_gens": N_GENS,
        "top_k": TOP_K, "n_children": N_CHILDREN,
        "mutation_sigma": MUTATION_SIGMA,
        "budget_per_agent_s": BUDGET_PER_AGENT_S, "n_cycles": N_CYCLES,
        "gen_history": gen_history,
        "best": {k: v for k, v in best.items() if k != "trained_params"} if best else None,
        "best_trained_params": best["trained_params"] if best else None,
    }
    with open("phase7_generations_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: phase7_generations_result.json", flush=True)


if __name__ == "__main__":
    main()
