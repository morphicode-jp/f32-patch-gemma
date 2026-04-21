"""Priority 5: Cardinal vs mimir head-to-head on standard benchmarks.

Compares:
  - mimir (our explicit meta-dispatcher optimizer)
  - Cardinal-opt (our meta-evolution of populations + mut_rate)
  - CMA-ES (baseline adaptive ES)

Problems (5D, all maximization, true optimum = 0 or ~97.91):
  - Rastrigin:  多峰 + 格子状 local optima
  - Ackley:     plateau + funnel
  - Styblinski: deep basins (max ≈ 97.91)
  - Rosenbrock: banana valley

Metric: final best_score (closer to true optimum = better, "gap" = optimum - best).
Budget: 500 eval calls, 25s wall.
Seeds: 3 per tool per problem (median reported).

This directly tests the claim in 全論の公式の活用.md §5:
  "Cardinal と mimir は別パラダイム、Cardinal が mimir の turf で勝てるか?"
"""
import json
import math
import os
import sys
import time
import traceback
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


# ---------- Problem set (from benchmark_owl_vs_world.py) ----------
def rastrigin(p):
    A = 10.0
    return -(A * len(p) + sum(x * x - A * math.cos(2 * math.pi * x) for x in p))


def ackley(p):
    n = len(p)
    A, B, C = 20.0, 0.2, 2 * math.pi
    s1 = sum(x * x for x in p)
    s2 = sum(math.cos(C * x) for x in p)
    return -(-A * math.exp(-B * math.sqrt(s1 / n)) - math.exp(s2 / n) + A + math.e)


def styblinski(p):
    return -sum(x ** 4 - 16 * x ** 2 + 5 * x for x in p) / 2.0


def rosenbrock(p):
    s = 0.0
    for i in range(len(p) - 1):
        s += 100 * (p[i + 1] - p[i] ** 2) ** 2 + (1 - p[i]) ** 2
    return -s


PROBLEMS = [
    ("rastrigin_5d",  rastrigin,  [(-5.12, 5.12)] * 5, 0.0),
    ("ackley_5d",     ackley,     [(-5.0, 5.0)] * 5,   0.0),
    ("styblinski_5d", styblinski, [(-5.0, 5.0)] * 5,   39.166 * 5 / 2),
    ("rosenbrock_5d", rosenbrock, [(-2.0, 2.0)] * 5,   0.0),
]


# ---------- Eval counter ----------
class Counter:
    def __init__(self, fn):
        self.fn = fn
        self.n = 0
    def __call__(self, p):
        self.n += 1
        return self.fn(list(p))


EVAL_BUDGET = 500
TIME_LIMIT = 25


# ---------- Tool: mimir ----------
def run_mimir(problem_eval, ranges, budget, seed=42):
    from twelve.agent.mimir import mimir
    counter = Counter(problem_eval)
    t0 = time.time()
    try:
        result = mimir(
            eval_fn=lambda p: counter(p),
            param_ranges=list(ranges),
            time_budget=min(TIME_LIMIT, 20),
            experience_id=f"bench_mimir_{seed}",
        )
        best = result.get("best_score", float("-inf"))
    except Exception as e:
        return {"best_score": float("-inf"), "eval_count": counter.n,
                "wall_s": time.time() - t0, "error": str(e)[:100]}
    return {"best_score": float(best), "eval_count": counter.n,
            "wall_s": round(time.time() - t0, 2)}


# ---------- Tool: Cardinal-opt (pure optimization variant) ----------
def run_cardinal_opt(problem_eval, ranges, budget, seed=42):
    """Cardinal applied to optimization: N populations with own mut_rates,
    meta-evolve mut_rate across populations."""
    counter = Counter(problem_eval)
    t0 = time.time()
    rng = np.random.default_rng(seed)

    N_universes = 4
    pop_size = 8
    epochs = 5

    # Initialize universes with random mut_rates
    universes = []
    for u_id in range(N_universes):
        mut_rate = float(rng.uniform(0.05, 0.30))
        pop = [[float(rng.uniform(lo, hi)) for lo, hi in ranges]
               for _ in range(pop_size)]
        universes.append({
            "id": u_id, "mut_rate": mut_rate, "pop": pop,
            "best_score": float("-inf"), "best_x": None,
        })

    best_global = float("-inf")
    best_x_global = None

    for epoch in range(epochs):
        if counter.n >= budget or time.time() - t0 > TIME_LIMIT:
            break
        # Each universe: evaluate population, top-half mutate
        for u in universes:
            if counter.n >= budget:
                break
            scores = []
            for x in u["pop"]:
                if counter.n >= budget:
                    break
                s = counter(x)
                scores.append(s)
                if s > best_global:
                    best_global = s
                    best_x_global = list(x)
                if s > u["best_score"]:
                    u["best_score"] = s
                    u["best_x"] = list(x)
            if not scores:
                continue
            # Top-half selection + Gaussian mutation with universe's mut_rate
            ranked = sorted(zip(u["pop"][:len(scores)], scores),
                            key=lambda t: -t[1])
            top = [x for x, _ in ranked[:max(2, pop_size // 2)]]
            new_pop = [list(x) for x in top]
            while len(new_pop) < pop_size:
                parent = top[rng.integers(0, len(top))]
                child = [p + rng.normal(0, u["mut_rate"] * (hi - lo))
                         for p, (lo, hi) in zip(parent, ranges)]
                child = [float(np.clip(c, lo, hi))
                         for c, (lo, hi) in zip(child, ranges)]
                new_pop.append(child)
            u["pop"] = new_pop

        # Meta-evolve: replace worst universe with variant of best
        if epoch < epochs - 1 and counter.n < budget:
            universes.sort(key=lambda u: -u["best_score"])
            best_u = universes[0]
            worst_u = universes[-1]
            # Mutate worst's mut_rate toward best's
            tau = 0.1
            new_mut_rate = best_u["mut_rate"] + float(
                rng.normal(0, tau * 0.3))
            new_mut_rate = float(np.clip(new_mut_rate, 0.01, 0.5))
            worst_u["mut_rate"] = new_mut_rate
            # Seed worst's population with best's best + perturbations
            if best_u["best_x"] is not None:
                seed_x = list(best_u["best_x"])
                worst_u["pop"] = [seed_x]
                for _ in range(pop_size - 1):
                    child = [x + rng.normal(0, new_mut_rate * (hi - lo))
                             for x, (lo, hi) in zip(seed_x, ranges)]
                    child = [float(np.clip(c, lo, hi))
                             for c, (lo, hi) in zip(child, ranges)]
                    worst_u["pop"].append(child)
                worst_u["best_score"] = float("-inf")
                worst_u["best_x"] = None

    return {
        "best_score": float(best_global) if best_global > float("-inf") else float("-inf"),
        "eval_count": counter.n,
        "wall_s": round(time.time() - t0, 2),
        "final_mut_rates": [round(u["mut_rate"], 4) for u in universes],
    }


# ---------- Tool: CMA-ES (baseline) ----------
def run_cma_es(problem_eval, ranges, budget, seed=42):
    try:
        import cma
    except ImportError:
        return {"best_score": float("-inf"), "eval_count": 0,
                "wall_s": 0, "error": "cma not installed"}
    counter = Counter(problem_eval)
    t0 = time.time()
    x0 = [(lo + hi) / 2 for lo, hi in ranges]
    sigma = np.mean([(hi - lo) / 4 for lo, hi in ranges])
    try:
        es = cma.CMAEvolutionStrategy(x0, sigma, {
            "bounds": [[lo for lo, _ in ranges], [hi for _, hi in ranges]],
            "maxfevals": budget,
            "verbose": -9, "seed": seed,
        })
        while not es.stop() and time.time() - t0 < TIME_LIMIT:
            X = es.ask()
            fits = [-counter(list(x)) for x in X]
            es.tell(X, fits)
        best_x, best_val, _ = es.best.get()
        return {"best_score": float(-best_val), "eval_count": counter.n,
                "wall_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"best_score": float("-inf"), "eval_count": counter.n,
                "wall_s": round(time.time() - t0, 2), "error": str(e)[:100]}


# ---------- Benchmark runner ----------
TOOLS = {
    "mimir": run_mimir,
    "cardinal": run_cardinal_opt,
    "cma_es": run_cma_es,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,7,13")
    ap.add_argument("--budget", type=int, default=EVAL_BUDGET)
    ap.add_argument("--tools", type=str, default="mimir,cardinal,cma_es")
    ap.add_argument("--output", type=str,
                    default="benchmark_cardinal_vs_mimir.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tools = [t.strip() for t in args.tools.split(",")]

    print("=" * 72, flush=True)
    print("  Cardinal vs mimir Benchmark (Priority 5)", flush=True)
    print(f"  Problems: {[p[0] for p in PROBLEMS]}", flush=True)
    print(f"  Tools: {tools}", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Budget: {args.budget} evals, {TIME_LIMIT}s wall", flush=True)
    print("=" * 72, flush=True)

    results = {}  # {problem: {tool: [runs...]}}
    t_all = time.time()

    for p_name, p_fn, p_ranges, p_opt in PROBLEMS:
        print(f"\n{'='*72}", flush=True)
        print(f"  Problem: {p_name}  (true optimum = {p_opt:.3f})", flush=True)
        print(f"{'='*72}", flush=True)
        results[p_name] = {"optimum": p_opt, "tools": {}}
        for tool_name in tools:
            if tool_name not in TOOLS:
                print(f"  Unknown tool: {tool_name}", flush=True)
                continue
            runner = TOOLS[tool_name]
            runs = []
            for s in seeds:
                t_run = time.time()
                r = runner(p_fn, p_ranges, args.budget, seed=s)
                gap = p_opt - r["best_score"]
                r["gap"] = float(gap)
                r["seed"] = s
                runs.append(r)
                print(f"  {tool_name:10s} seed={s:3d}: "
                      f"best={r['best_score']:+9.4f} gap={gap:9.4f} "
                      f"evals={r.get('eval_count',0):4d} "
                      f"wall={r.get('wall_s',0):5.2f}s"
                      + (f" err={r['error'][:30]}" if "error" in r else ""),
                      flush=True)
            results[p_name]["tools"][tool_name] = runs

    # Summary: median gap per tool per problem
    total = time.time() - t_all
    print(f"\n{'='*72}", flush=True)
    print(f"  SUMMARY ({total:.0f}s)", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"\n  Median gap (smaller = better, 0 = perfect):", flush=True)
    print(f"  {'Problem':15s}", end="")
    for t in tools:
        print(f"  {t:>10s}", end="")
    print()
    for p_name in results:
        print(f"  {p_name:15s}", end="")
        for t in tools:
            runs = results[p_name]["tools"].get(t, [])
            if runs:
                gaps = [r["gap"] for r in runs if r["gap"] != float("inf")]
                med_gap = float(np.median(gaps)) if gaps else float("nan")
                print(f"  {med_gap:10.3f}", end="")
            else:
                print(f"  {'—':>10s}", end="")
        print()

    # Head-to-head: Cardinal vs mimir
    print(f"\n  Head-to-head Cardinal vs mimir:", flush=True)
    c_wins = 0; m_wins = 0; ties = 0
    for p_name in results:
        c_runs = results[p_name]["tools"].get("cardinal", [])
        m_runs = results[p_name]["tools"].get("mimir", [])
        if not c_runs or not m_runs:
            continue
        c_med = float(np.median([r["gap"] for r in c_runs]))
        m_med = float(np.median([r["gap"] for r in m_runs]))
        if abs(c_med - m_med) < 0.01:
            winner = "TIE"; ties += 1
        elif c_med < m_med:
            winner = "Cardinal WINS"; c_wins += 1
        else:
            winner = "mimir WINS"; m_wins += 1
        print(f"    {p_name:15s}: Cardinal gap={c_med:.3f}  mimir gap={m_med:.3f}  → {winner}",
              flush=True)
    print(f"\n  Score: Cardinal {c_wins} / mimir {m_wins} / ties {ties}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary_wins": {
            "cardinal": c_wins, "mimir": m_wins, "ties": ties}}, f,
            indent=2, default=str)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
