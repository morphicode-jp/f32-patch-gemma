"""K² meta-optimization: use mimir to tune mimir's own dispatch params.

⚠️ STATUS (2026-04-20): pathological runtime under naive self-application.
   First attempt: 3h+ wall with no output before kill (memory 20GB growth).
   Baseline measurement worked cleanly (meta-score = -1.4145 with current defaults).
   The K² inner loop hit slow-config dead-ends and never returned.
   See "教訓" section in the commit message for remediation ideas.

全論 Ch18 K² pattern の具現化:
  K1 = BBOB 問題の解
  K2 = K1 の最適化手法 (mimir の dispatch params)

mimir を使って mimir の dispatch params を最適化 — 自己参照 / fixed point of
self-application (Ch17 "The formula that discovers formulas IS the formula").

対象 params (6 dim continuous、int は round 後扱い):
  - eval_cost_threshold       (0.05, 3.0)   default 0.5
  - owl_share                 (0.15, 0.85)  default 0.4
  - escalation_min_remaining  (1.0, 20.0)   default 5.0
  - confidence_skip_threshold (0.3, 0.95)   default 0.7
  - random_restart_count      (0, 20)  int   default 5
  - reigen_inner_time_budget  (0.5, 8.0)    default 2.0

Eval fn: mimir on BBOB 4 問題 × 2 seeds → sum of -|gap| → maximize
Budget: meta-eval 1 回 ≈ 25s × 4 × 2 = 200s。20-30 meta-evals で ~60-90 min。
"""
import json
import math
import os
import random
import sys
import time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from twelve.agent.mimir import mimir


# ---- K1: BBOB benchmark problems ----
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


# ---- K2 param space ----
MIMIR_PARAM_NAMES = [
    "eval_cost_threshold",       # continuous
    "owl_share",                 # continuous
    "escalation_min_remaining",  # continuous
    "confidence_skip_threshold", # continuous
    "random_restart_count",      # int (rounded from continuous)
    "reigen_inner_time_budget",  # continuous
]

MIMIR_PARAM_RANGES = [
    (0.05, 3.0),   # eval_cost_threshold
    (0.15, 0.85),  # owl_share
    (1.0, 20.0),   # escalation_min_remaining
    (0.3, 0.95),   # confidence_skip_threshold
    (0.0, 20.0),   # random_restart_count (will int(round()) before use)
    (0.5, 8.0),    # reigen_inner_time_budget
]

HARDCODE_DEFAULTS = [0.5, 0.4, 5.0, 0.7, 5.0, 2.0]


def _cfg_from_params(params):
    cfg = dict(zip(MIMIR_PARAM_NAMES, params))
    cfg["random_restart_count"] = int(round(cfg["random_restart_count"]))
    return cfg


def _eval_mimir_config(params, seeds=(42, 43), meta_run_budget=25, verbose=False):
    """Meta eval: score mimir's performance with given dispatch params across problems."""
    cfg = _cfg_from_params(params)
    total = 0.0
    n_runs = 0
    for prob_name, fn, ranges, optimum in PROBLEMS:
        for seed in seeds:
            try:
                r = mimir(fn, ranges, time_budget=meta_run_budget,
                          experience_id=f"meta_{prob_name}_{seed}",
                          mimir_cfg=cfg,
                          verbose=False)
                gap = abs(optimum - r["best_score"])
                # Penalize high gap, normalize by problem scale
                # Rosenbrock range ~1000, Rastrigin ~50, Styblinski ~100 — use log1p
                total += -math.log1p(gap)
            except Exception:
                total += -10.0  # heavy penalty
            n_runs += 1
    # Normalize: higher = better
    return total / max(1, n_runs)


def main():
    print("=" * 70)
    print("K² meta-optimization of mimir dispatch params")
    print("=" * 70)
    print(f"Param ranges: {dict(zip(MIMIR_PARAM_NAMES, MIMIR_PARAM_RANGES))}")

    # Baseline: current hardcoded defaults
    t0 = time.time()
    print(f"\n[Baseline] current defaults ({dict(zip(MIMIR_PARAM_NAMES, HARDCODE_DEFAULTS))})")
    print("  measuring...", flush=True)
    baseline_score = _eval_mimir_config(HARDCODE_DEFAULTS,
                                         seeds=(42, 43),
                                         meta_run_budget=25)
    baseline_wall = time.time() - t0
    print(f"  baseline meta-score = {baseline_score:.4f}  (wall {baseline_wall:.0f}s for 4×2 runs)")

    # Meta-optimize: use mimir on itself
    print(f"\n[K² meta-opt] budget = 30 min, calling mimir on _eval_mimir_config")
    print(f"  Note: each meta-eval ≈ 4 problems × 2 seeds × 25s ≈ 200s")
    print(f"  Expected meta-evals: 8-10 in 30 min")
    print(f"  Setting eval_cost_hint=200 → expensive route (no inner Reigen to avoid recursion)")

    def meta_eval_wrapped(p_list):
        return _eval_mimir_config(p_list, seeds=(42, 43), meta_run_budget=25)

    t_meta = time.time()
    r_meta = mimir(
        eval_fn=meta_eval_wrapped,
        param_ranges=MIMIR_PARAM_RANGES,
        param_names=MIMIR_PARAM_NAMES,
        time_budget=1800,      # 30 min
        eval_cost_hint=200.0,  # force expensive route → owl-only, no inner Reigen
        experience_id="k2_meta_opt",
        verbose=False,
    )
    meta_wall = time.time() - t_meta
    print(f"\n[K² meta-opt complete] wall {meta_wall/60:.1f} min")
    print(f"  tool_used: {r_meta.get('tool_used')}")
    print(f"  best_score = {r_meta['best_score']:.4f}")
    print(f"  confidence = {r_meta.get('confidence')}")

    best_cfg = _cfg_from_params(r_meta["best_params"])
    print(f"\n[Best dispatch config discovered]:")
    for k, v in best_cfg.items():
        print(f"  {k:>30}: {v}")

    # Re-verify best config
    print(f"\n[Verification] re-run best config on 4 × 2 seeds")
    verify_score = _eval_mimir_config(r_meta["best_params"],
                                       seeds=(42, 43),
                                       meta_run_budget=25)
    print(f"  verified meta-score = {verify_score:.4f}  (baseline {baseline_score:.4f})")

    # Detailed gap per problem with best config
    print(f"\n[Per-problem verification with best config]:")
    for prob_name, fn, ranges, optimum in PROBLEMS:
        gaps = []
        for seed in (42, 43):
            r = mimir(fn, ranges, time_budget=25,
                      experience_id=f"vrfy_{prob_name}_{seed}",
                      mimir_cfg=best_cfg)
            gaps.append(abs(optimum - r["best_score"]))
        print(f"  {prob_name:>15}: median gap = {sorted(gaps)[len(gaps)//2]:+.4f}  "
              f"all = {[round(g,4) for g in gaps]}")

    # Save
    out = {
        "baseline": {
            "params": HARDCODE_DEFAULTS,
            "score": baseline_score,
            "wall_s": baseline_wall,
        },
        "best": {
            "params": r_meta["best_params"],
            "named": best_cfg,
            "score": verify_score,
            "confidence": r_meta.get("confidence"),
        },
        "meta_wall_s": meta_wall,
        "improvement": verify_score - baseline_score,
    }
    with open("benchmark_mimir_meta.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[Summary]")
    print(f"  baseline score: {baseline_score:+.4f}")
    print(f"  best verified:  {verify_score:+.4f}")
    print(f"  improvement:    {verify_score - baseline_score:+.4f}")
    print(f"  recommendation: {'UPDATE mimir_params.json' if verify_score > baseline_score else 'keep defaults'}")


if __name__ == "__main__":
    main()
