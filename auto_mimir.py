"""auto_mimir.py — Cardinal-driven self-tuning of mimir's HPs.

Pattern B from docs/全論の公式の活用.md:
  mimir has internal HPs (mimir_params.json) that control its dispatch
  behavior. These are currently hand-tuned. auto_mimir uses Cardinal-style
  meta-evolution to discover better HPs automatically.

Architecture:
  N parallel "universes", each = one mimir HP configuration.
  Each universe solves the benchmark problem suite.
  Quality = mean of (-gap) across problems (higher better).
  Meta-evolve: worst universe's HP replaced by variant of best's.
  Repeat K epochs → discover strong HP config.

Compare to:
  mimir-default: fixed HPs from mimir_params.json
  Goal: auto_mimir-found HPs beat default on same benchmark suite
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------- Benchmark suite (from benchmark_cardinal_vs_mimir.py) ----------
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


# ---------- mimir HP search space ----------
# Each HP is a tunable parameter in mimir_params.json dispatching logic
MIMIR_HP_RANGES = {
    "random_restart_count": (1, 10, "int"),          # 何回 random restart
    "owl_share": (0.2, 0.8, "float"),                 # owl の budget 割合
    "eval_cost_threshold": (0.1, 2.0, "float"),       # owl 直行 vs cascade の境
    "confidence_skip_threshold": (0.4, 0.9, "float"), # confidence 以上で Reigen skip
    "reigen_inner_time_budget": (0.5, 5.0, "float"),  # Reigen 1 回 budget
    "reigen_wall_time_factor": (0.5, 3.0, "float"),   # Reigen wall 倍率
}


def random_hp(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hp = {}
    for k, (lo, hi, typ) in MIMIR_HP_RANGES.items():
        if typ == "float":
            hp[k] = float(rng.uniform(lo, hi))
        elif typ == "int":
            hp[k] = int(round(rng.uniform(lo, hi)))
    return hp


def default_hp() -> dict:
    """Current mimir_params.json defaults (as of 2026-04-22)."""
    return {
        "random_restart_count": 5,
        "owl_share": 0.4,
        "eval_cost_threshold": 0.5,
        "confidence_skip_threshold": 0.7,
        "reigen_inner_time_budget": 2.0,
        "reigen_wall_time_factor": 1.0,
    }


def mutate_hp(parent: dict, sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    child = dict(parent)
    for k, (lo, hi, typ) in MIMIR_HP_RANGES.items():
        if rng.random() < 0.5:  # mutate ~50% of HPs
            continue
        delta = rng.normal(0, sigma * (hi - lo))
        val = parent[k] + delta
        val = max(lo, min(hi, val))
        if typ == "int":
            val = int(round(val))
        child[k] = val
    return child


# ---------- Evaluate one HP config on benchmark suite ----------
def evaluate_hp(hp: dict, problems, time_per_problem: float,
                seed: int = 42) -> dict:
    """Run mimir with given HP on each problem, return quality metrics."""
    from twelve.agent.mimir import mimir
    gaps = []
    times = []
    per_problem = []
    for p_name, p_fn, p_ranges, p_opt in problems:
        t0 = time.time()
        try:
            result = mimir(
                eval_fn=lambda p: p_fn(p),
                param_ranges=list(p_ranges),
                time_budget=time_per_problem,
                experience_id=f"auto_mimir_{p_name}_{seed}",
                **hp,
            )
            best = result.get("best_score", float("-inf"))
            gap = p_opt - best
        except Exception as e:
            gap = float("inf")
            best = float("-inf")
        elapsed = time.time() - t0
        gaps.append(gap if gap != float("inf") else 1e6)  # cap
        times.append(elapsed)
        per_problem.append({"name": p_name, "gap": float(gap) if gap != float("inf") else 1e6,
                            "best_score": float(best), "time_s": round(elapsed, 2)})
    # Quality: negative mean log(1+gap), higher better
    # Use log to handle huge gaps (e.g. Rosenbrock)
    mean_log_gap = float(np.mean([np.log1p(max(0, g)) for g in gaps]))
    quality = -mean_log_gap
    return {"quality": quality, "mean_log_gap": mean_log_gap,
            "per_problem": per_problem, "total_time_s": round(sum(times), 1)}


# ---------- Cardinal meta-evolution over mimir HPs ----------
def run_auto_mimir(n_universes: int = 6, n_epochs: int = 3,
                    time_per_problem: float = 6.0, seed: int = 42,
                    output: str = "auto_mimir_result.json"):
    print("=" * 72, flush=True)
    print("  AUTO-MIMIR: Cardinal tunes mimir's HP", flush=True)
    print(f"  {n_universes} universes × {n_epochs} epochs × "
          f"{time_per_problem}s/problem × {len(PROBLEMS)} problems", flush=True)
    print("=" * 72, flush=True)

    rng = np.random.default_rng(seed)
    t_all = time.time()

    # Universe 0 = defaults (baseline anchor); others = random
    universes = [default_hp()]
    for i in range(1, n_universes):
        universes.append(random_hp(seed + i * 7))

    history = []
    best_hp_ever = None
    best_quality_ever = float("-inf")

    for epoch in range(n_epochs):
        print(f"\n{'='*72}", flush=True)
        print(f"  EPOCH {epoch+1}/{n_epochs}", flush=True)
        print(f"{'='*72}", flush=True)

        # Evaluate each universe
        results = []
        for u_id, hp in enumerate(universes):
            print(f"\n  u{u_id}: random_restart={hp['random_restart_count']} "
                  f"owl_share={hp['owl_share']:.2f} "
                  f"cost_thr={hp['eval_cost_threshold']:.2f} "
                  f"conf_skip={hp['confidence_skip_threshold']:.2f}",
                  flush=True)
            t0 = time.time()
            r = evaluate_hp(hp, PROBLEMS, time_per_problem, seed=seed + epoch * 13 + u_id)
            elapsed = time.time() - t0
            r["hp"] = dict(hp)
            r["u_id"] = u_id
            r["elapsed_s"] = round(elapsed, 1)
            results.append(r)
            gap_summary = ", ".join(f"{p['name']}={p['gap']:.2f}"
                                    for p in r["per_problem"])
            print(f"    quality={r['quality']:.3f} gaps=[{gap_summary}] "
                  f"time={elapsed:.0f}s", flush=True)
            if r["quality"] > best_quality_ever:
                best_quality_ever = r["quality"]
                best_hp_ever = dict(hp)

        # Rank
        ranked = sorted(results, key=lambda r: -r["quality"])
        print(f"\n  Ranking by quality:", flush=True)
        for r in ranked:
            print(f"    u{r['u_id']}: quality={r['quality']:.3f}", flush=True)

        history.append({
            "epoch": epoch,
            "results": [{
                "u_id": r["u_id"], "hp": r["hp"],
                "quality": r["quality"], "mean_log_gap": r["mean_log_gap"],
                "per_problem": r["per_problem"],
            } for r in results],
            "best_this_epoch": ranked[0]["hp"],
        })

        # Meta-evolve: bottom 2 replaced by variants of top 1
        if epoch < n_epochs - 1:
            best_hp = ranked[0]["hp"]
            for worst_result in ranked[-2:]:
                new_hp = mutate_hp(best_hp, sigma=0.2,
                                    seed=seed + epoch * 31 + worst_result["u_id"])
                universes[worst_result["u_id"]] = new_hp
                print(f"  META: u{worst_result['u_id']} ← mutate(u{ranked[0]['u_id']})",
                      flush=True)

    total = time.time() - t_all
    print(f"\n{'='*72}", flush=True)
    print(f"  AUTO-MIMIR DONE ({total:.0f}s = {total/60:.1f}min)", flush=True)
    print(f"{'='*72}", flush=True)

    # Compare: default vs discovered best
    default = default_hp()
    print(f"\n  Default HP vs Discovered HP:", flush=True)
    print(f"    {'param':30s} {'default':>10s} {'discovered':>12s}", flush=True)
    for k in MIMIR_HP_RANGES:
        d = default[k]
        b = best_hp_ever[k] if best_hp_ever else d
        print(f"    {k:30s} {d:>10} {b:>12}", flush=True)

    # Evaluate default explicitly for comparison (in case it wasn't u0 throughout)
    print(f"\n  Final evaluation: default vs discovered HP on same suite",
          flush=True)
    print(f"  Evaluating default HP...", flush=True)
    default_result = evaluate_hp(default, PROBLEMS, time_per_problem, seed=seed + 9999)
    print(f"    default: quality={default_result['quality']:.3f} "
          f"gaps={[round(p['gap'],2) for p in default_result['per_problem']]}",
          flush=True)
    print(f"  Evaluating discovered HP...", flush=True)
    disc_result = evaluate_hp(best_hp_ever, PROBLEMS, time_per_problem, seed=seed + 9999)
    print(f"    discov.: quality={disc_result['quality']:.3f} "
          f"gaps={[round(p['gap'],2) for p in disc_result['per_problem']]}",
          flush=True)

    improvement = disc_result['quality'] - default_result['quality']
    print(f"\n  Quality improvement (discovered - default): {improvement:+.3f}",
          flush=True)
    if improvement > 0.05:
        print(f"  ✓ Discovered HP is BETTER than defaults", flush=True)
    elif improvement > -0.05:
        print(f"  ≈ Discovered HP TIES defaults", flush=True)
    else:
        print(f"  × Discovered HP is WORSE than defaults", flush=True)

    out = {
        "total_elapsed_s": round(total, 1),
        "n_universes": n_universes,
        "n_epochs": n_epochs,
        "time_per_problem_s": time_per_problem,
        "default_hp": default,
        "discovered_hp": best_hp_ever,
        "default_result": default_result,
        "discovered_result": disc_result,
        "improvement": float(improvement),
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=5)
    ap.add_argument("--n_epochs", type=int, default=3)
    ap.add_argument("--time_per_problem", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="auto_mimir_result.json")
    args = ap.parse_args()
    run_auto_mimir(
        n_universes=args.n_universes,
        n_epochs=args.n_epochs,
        time_per_problem=args.time_per_problem,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
