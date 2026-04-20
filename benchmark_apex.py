"""apex vs 世界 benchmark. 対決: cma_es, basinhopping, reigen, owl, apex.

optuna_tpe / skopt_gp は前回の benchmark で圧倒敗北しているため除外。
25s budget × 3 seeds × 4 BBOB problems.
"""
import json, math, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")


def rastrigin(p):
    A = 10.0
    return -(A * len(p) + sum(x * x - A * math.cos(2 * math.pi * x) for x in p))

def ackley(p):
    n = len(p); A, B, C = 20.0, 0.2, 2 * math.pi
    s1 = sum(x * x for x in p); s2 = sum(math.cos(C * x) for x in p)
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
    ("styblinski_5d", styblinski, [(-5.0, 5.0)] * 5,   195.83),
    ("rosenbrock_5d", rosenbrock, [(-2.0, 2.0)] * 5,   0.0),
]
EVAL_BUDGET = 500
TIME_LIMIT = 25


class Counter:
    def __init__(self, fn): self.fn = fn; self.n = 0
    def __call__(self, p): self.n += 1; return self.fn(p)


def run_cma(fn, ranges, seed=42):
    import cma
    c = Counter(fn); t0 = time.time()
    x0 = [(lo+hi)/2 for lo,hi in ranges]
    sigma = np.mean([(hi-lo)/4 for lo,hi in ranges])
    es = cma.CMAEvolutionStrategy(x0, sigma, {
        "bounds": [[lo for lo,_ in ranges], [hi for _,hi in ranges]],
        "maxfevals": EVAL_BUDGET, "verbose": -9, "seed": seed})
    while not es.stop() and time.time() - t0 < TIME_LIMIT:
        X = es.ask(); fits = [-c(list(x)) for x in X]; es.tell(X, fits)
    _, best_val, _ = es.best.get()
    return {"best_score": -best_val, "eval_count": c.n, "wall_s": time.time() - t0}


def run_basin(fn, ranges, seed=42):
    from scipy.optimize import basinhopping
    c = Counter(fn); t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = [rng.uniform(lo, hi) for lo, hi in ranges]
    def f(p): return -c(list(p))
    bounds = list(ranges)
    r = basinhopping(f, x0,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds},
        niter=100, seed=seed,
        callback=lambda x, v, ac: c.n >= EVAL_BUDGET or time.time()-t0 >= TIME_LIMIT)
    return {"best_score": -r.fun, "eval_count": c.n, "wall_s": time.time() - t0}


def run_reigen(fn, ranges, seed=42):
    from twelve.agent.reigen import reigen
    c = Counter(fn); t0 = time.time()
    tb = max(3, min(EVAL_BUDGET // 5, TIME_LIMIT))
    try:
        r = reigen(c, lambda p: 0.0, user_param_ranges=ranges,
            experience_id=f"bhrei_{seed}",
            time_budget=tb, inner_time_budget=1, wall_time_factor=1.5,
            self_dim_preset="kathara_17_adaptive", verbose=False)
        return {"best_score": r["user_best_score"],
                "eval_count": c.n, "wall_s": time.time() - t0}
    except Exception as e:
        return {"error": str(e), "best_score": float("-inf"),
                "eval_count": c.n, "wall_s": time.time() - t0}


def run_owl(fn, ranges, seed=42):
    from twelve.optimize import owl
    c = Counter(fn); t0 = time.time()
    rng = np.random.default_rng(seed)
    curated = [{"params": [rng.uniform(lo,hi) for lo,hi in ranges], "score": fn([rng.uniform(lo,hi) for lo,hi in ranges])} for _ in range(0)]
    # deterministic 20-point curated seed
    rng = np.random.default_rng(seed)
    curated = []
    for _ in range(20):
        p = [rng.uniform(lo, hi) for lo, hi in ranges]
        curated.append({"params": p, "score": float(fn(p))})
    c.n = 20
    try:
        r = owl(measurements=curated, verify_fn=c, param_ranges=ranges,
            autonomous=True, max_iterations=30, time_budget=TIME_LIMIT,
            min_r_squared=0.1, experience_id=f"bhowl_{seed}",
            use_lbfgs_refinement=True, use_multistart_fallback=True,
            random_restart_count=5, verbose=False)
        vs = r.get("verified_score"); bs = r.get("best_score")
        best = max(m["score"] for m in curated)
        if vs is not None and vs > best: best = vs
        if bs is not None and bs > best: best = bs
        return {"best_score": best, "eval_count": c.n, "wall_s": time.time() - t0}
    except Exception as e:
        return {"error": str(e), "best_score": float("-inf"),
                "eval_count": c.n, "wall_s": time.time() - t0}


def run_apex(fn, ranges, seed=42):
    from twelve.agent.apex import apex
    c = Counter(fn); t0 = time.time()
    try:
        r = apex(c, ranges, time_budget=TIME_LIMIT,
            experience_id=f"bhhgn_{seed}", verbose=False)
        return {"best_score": r["best_score"], "eval_count": c.n,
                "wall_s": time.time() - t0, "tool_used": r.get("tool_used")}
    except Exception as e:
        return {"error": str(e), "best_score": float("-inf"),
                "eval_count": c.n, "wall_s": time.time() - t0}


TOOLS = [
    ("cma_es",       run_cma),
    ("basinhopping", run_basin),
    ("reigen_k17",   run_reigen),
    ("owl_direct",   run_owl),
    ("apex",        run_apex),
]


def main():
    results = []
    N_REPEATS = 3
    total = len(PROBLEMS) * len(TOOLS) * N_REPEATS
    idx = 0
    for prob_name, fn, ranges, opt in PROBLEMS:
        for tool_name, tool_fn in TOOLS:
            for rep in range(N_REPEATS):
                idx += 1
                seed = 42 + rep
                print(f"[{idx}/{total}] {prob_name:>15} {tool_name:>15} rep{rep+1}...", flush=True)
                try:
                    res = tool_fn(fn, ranges, seed=seed)
                    gap = opt - res["best_score"]
                    res.update({"problem": prob_name, "tool": tool_name, "rep": rep,
                                "gap": gap, "optimum": opt})
                    results.append(res)
                    extra = f" tool={res.get('tool_used')}" if tool_name == "apex" else ""
                    print(f"      gap={gap:+.4f} evals={res['eval_count']} wall={res['wall_s']:.1f}s{extra}", flush=True)
                except Exception as e:
                    results.append({"problem": prob_name, "tool": tool_name, "rep": rep,
                        "error": f"{type(e).__name__}: {e}",
                        "gap": float("inf"), "best_score": float("-inf"),
                        "eval_count": 0, "wall_s": 0})
                    print(f"      ERROR: {type(e).__name__}", flush=True)
                json.dump(results, open("benchmark_apex.json", "w"), indent=2, default=str)

    # Summary
    print("\n" + "=" * 90)
    print(f"{'problem':>16} {'tool':>16} {'gap_median':>12} {'evals':>10} {'wall':>8}")
    print("=" * 90)
    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        if not r.get("error"):
            g[(r["problem"], r["tool"])].append(r)
    for prob_name, _, _, _ in PROBLEMS:
        for tool_name, _ in TOOLS:
            rs = g.get((prob_name, tool_name), [])
            if not rs:
                print(f"{prob_name:>16} {tool_name:>16} {'N/A':>12}")
                continue
            gaps = sorted(r["gap"] for r in rs)
            walls = sorted(r["wall_s"] for r in rs)
            evals = sorted(r["eval_count"] for r in rs)
            print(f"{prob_name:>16} {tool_name:>16} {gaps[len(gaps)//2]:>+12.4f} "
                  f"{evals[len(evals)//2]:>10d} {walls[len(walls)//2]:>7.1f}s")
        print()


if __name__ == "__main__":
    main()
