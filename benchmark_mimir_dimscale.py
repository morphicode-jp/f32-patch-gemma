"""Dimension-scaling benchmark: mimir vs cma_es vs basinhopping on 5d/10d/20d.

5d だけの既存 benchmark を補完。高次元で mimir がスケールするか検証する。
reigen_k17 は mimir の cascade 内部で動くので別途は計測しない。
owl_direct も mimir 内部 Phase 1 と同じなので別途計測しない (単純化)。

TIME_LIMIT は次元に応じて増やす (5d=25s, 10d=40s, 20d=60s)。
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


PROBLEM_DEFS = [
    ("rastrigin",  rastrigin,  (-5.12, 5.12), lambda d: 0.0),
    ("ackley",     ackley,     (-5.0, 5.0),   lambda d: 0.0),
    ("styblinski", styblinski, (-5.0, 5.0),   lambda d: 39.166 * d),  # optimum at x=-2.9035 all dims
    ("rosenbrock", rosenbrock, (-2.0, 2.0),   lambda d: 0.0),
]

DIMS = [5, 10, 20]

def budget_for_dim(d):
    return {5: 25, 10: 40, 20: 60}.get(d, 30)


class Counter:
    def __init__(self, fn): self.fn = fn; self.n = 0
    def __call__(self, p): self.n += 1; return self.fn(p)


def run_cma(fn, ranges, seed=42, time_limit=30):
    import cma
    c = Counter(fn); t0 = time.time()
    x0 = [(lo+hi)/2 for lo,hi in ranges]
    sigma = np.mean([(hi-lo)/4 for lo,hi in ranges])
    es = cma.CMAEvolutionStrategy(x0, sigma, {
        "bounds": [[lo for lo,_ in ranges], [hi for _,hi in ranges]],
        "maxfevals": 2000, "verbose": -9, "seed": seed})
    while not es.stop() and time.time() - t0 < time_limit:
        X = es.ask(); fits = [-c(list(x)) for x in X]; es.tell(X, fits)
    _, best_val, _ = es.best.get()
    return {"best_score": -best_val, "eval_count": c.n, "wall_s": time.time() - t0}


def run_basin(fn, ranges, seed=42, time_limit=30):
    from scipy.optimize import basinhopping
    c = Counter(fn); t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = [rng.uniform(lo, hi) for lo, hi in ranges]
    def f(p): return -c(list(p))
    r = basinhopping(f, x0,
        minimizer_kwargs={"method": "L-BFGS-B", "bounds": list(ranges)},
        niter=200, seed=seed,
        callback=lambda x, v, ac: c.n >= 2000 or time.time()-t0 >= time_limit)
    return {"best_score": -r.fun, "eval_count": c.n, "wall_s": time.time() - t0}


def run_mimir(fn, ranges, seed=42, time_limit=30, prob_name="prob"):
    from twelve.agent.mimir import mimir
    c = Counter(fn); t0 = time.time()
    try:
        r = mimir(c, ranges, time_budget=time_limit,
            experience_id=f"dsm_{prob_name}_{len(ranges)}_{seed}", verbose=False)
        return {"best_score": r["best_score"], "eval_count": c.n,
                "wall_s": time.time() - t0, "tool_used": r.get("tool_used"),
                "proxy_r2": r.get("proxy_r2"), "confidence": r.get("confidence")}
    except Exception as e:
        return {"error": str(e), "best_score": float("-inf"),
                "eval_count": c.n, "wall_s": time.time() - t0}


TOOLS = [
    ("cma_es", run_cma),
    ("basinhopping", run_basin),
    ("mimir", run_mimir),
]


def main():
    results = []
    N_REPEATS = 3
    all_runs = [
        (pname, pfn, lo_hi, opt_fn, dim)
        for (pname, pfn, lo_hi, opt_fn) in PROBLEM_DEFS
        for dim in DIMS
    ]
    total = len(all_runs) * len(TOOLS) * N_REPEATS
    idx = 0
    for pname, pfn, (lo, hi), opt_fn, dim in all_runs:
        ranges = [(lo, hi)] * dim
        optimum = opt_fn(dim)
        tb = budget_for_dim(dim)
        for tname, tfn in TOOLS:
            for rep in range(N_REPEATS):
                idx += 1
                seed = 42 + rep
                print(f"[{idx}/{total}] {pname}_{dim}d {tname} rep{rep+1} budget={tb}s...",
                      flush=True)
                try:
                    kwargs = dict(seed=seed, time_limit=tb)
                    if tname == "mimir":
                        kwargs["prob_name"] = pname
                    res = tfn(pfn, ranges, **kwargs)
                    gap = optimum - res["best_score"]
                    res.update({"problem": pname, "dim": dim, "tool": tname,
                                "rep": rep, "gap": gap, "optimum": optimum})
                    results.append(res)
                    extra = f" tool={res.get('tool_used')}" if tname == "mimir" else ""
                    print(f"      gap={gap:+.4f} evals={res['eval_count']} "
                          f"wall={res['wall_s']:.1f}s{extra}", flush=True)
                except Exception as e:
                    results.append({"problem": pname, "dim": dim, "tool": tname,
                        "rep": rep, "error": f"{type(e).__name__}: {e}",
                        "gap": float("inf"), "best_score": float("-inf"),
                        "eval_count": 0, "wall_s": 0})
                    print(f"      ERROR: {type(e).__name__}", flush=True)
                json.dump(results, open("benchmark_mimir_dimscale.json", "w"),
                          indent=2, default=str)

    # Summary
    print("\n" + "=" * 110)
    print(f"{'problem':>12} {'dim':>4} {'tool':>14} {'gap_med':>10} {'gap_all':>25} {'wall_med':>8}")
    print("=" * 110)
    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        if not r.get("error"): g[(r["problem"], r["dim"], r["tool"])].append(r)
    for pname, _, _, _ in PROBLEM_DEFS:
        for dim in DIMS:
            for tname, _ in TOOLS:
                rs = g.get((pname, dim, tname), [])
                if not rs:
                    print(f"{pname:>12} {dim:>4} {tname:>14} N/A")
                    continue
                gaps = sorted(r["gap"] for r in rs)
                walls = sorted(r["wall_s"] for r in rs)
                gs = '[' + ', '.join(f'{x:+.2f}' for x in gaps) + ']'
                print(f"{pname:>12} {dim:>4} {tname:>14} {gaps[len(gaps)//2]:>+10.3f} "
                      f"{gs:>25} {walls[len(walls)//2]:>7.1f}s")
            print()


if __name__ == "__main__":
    main()
