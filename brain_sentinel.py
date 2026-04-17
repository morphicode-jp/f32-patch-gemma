"""brain_sentinel.py — Sentinel x Brain Evolution

Applies Sentinel (eval_fn + guard_fn cross-validation) to the Kathara 12-node
brain simulation. Four modes:

  biological  : FlyWorldV3 performance vs 20% effective inhibition (Dale's law)
  progressive : V3+predator performance vs V1 food-finding (catastrophic forgetting)
  intelligent : ISS structural score vs V1 survival (structure vs function)
  evolution   : Multi-generation loop with experience_id carry-over

Usage:
    python brain_sentinel.py --mode biological --time-budget 300
    python brain_sentinel.py --mode progressive --time-budget 300
    python brain_sentinel.py --mode intelligent --time-budget 300
    python brain_sentinel.py --mode evolution --n-gens 3 --time-per-gen 300
"""

import numpy as np
import json
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kathara_brain_sim_v6 import compute_iss_from_firing
from kathara_brain_sim_v8 import (
    FlyWorldV1,
    FlyWorldV3,
    simulate_step,
    PARAM_RANGES,
    PARAM_NAMES,
    NODE_NAMES,
)
from twelve.agent.sentinel import Sentinel


# ============================================================
# eval_fn / guard_fn builders
# ============================================================

def make_flyworld_eval(world_cls, n_episodes=5, n_steps=60,
                       has_predator=False, seed_offset=13):
    """Build eval_fn running episodes in the given world. Returns score 0-100.

    Same scoring formula as kathara_brain_sim_v6.eval_fly:
        approach*60 + 30*reached + min(10, move_sum*2), with *0.3 if caught.
    """
    def eval_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            kwargs = {"seed": ep * 7 + seed_offset}
            if world_cls is FlyWorldV3:
                kwargs["has_predator"] = has_predator
            world = world_cls(**kwargs)
            sensors = world.reset()
            initial_dist = world.get_food_dist()
            min_dist = initial_dist
            reached = False
            caught = False
            move_sum = 0.0
            states = np.zeros(12, dtype=np.float64)

            for _ in range(n_steps):
                prev_pos = world.fly_pos.copy()
                states, firing = simulate_step(p, sensors, states)
                nav = float(firing[5])
                cen = float(firing[11])
                sensors, dist, reached_now, caught_now = world.step(nav, cen)
                move_sum += float(np.linalg.norm(world.fly_pos - prev_pos))
                if dist < min_dist:
                    min_dist = dist
                if reached_now:
                    reached = True
                    break
                if caught_now:
                    caught = True
                    break

            approach = max(0.0, initial_dist - min_dist) / (initial_dist + 1e-6)
            ep_score = approach * 60.0
            if reached:
                ep_score += 30.0
            ep_score += min(10.0, move_sum * 2.0)
            if caught:
                ep_score *= 0.3
            total += ep_score
        return total / n_episodes
    return eval_fn


def make_inhibition_guard(target_ratio=0.20, sigma=0.20,
                          n_episodes=3, n_steps=30):
    """Build guard_fn measuring effective inhibition (fraction of quiet nodes
    with firing < 0.3). Gaussian peak at target_ratio. Returns 0-100.

    sigma=0.20 gives graded scoring across the full ratio range [0, 1].
    """
    def guard_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=ep * 11 + 7)
            sensors = world.reset()
            firing_accum = np.zeros(12, dtype=np.float64)
            states = np.zeros(12, dtype=np.float64)
            steps_done = 0
            for _ in range(n_steps):
                states, firing = simulate_step(p, sensors, states)
                firing_accum += firing
                nav = float(firing[5])
                cen = float(firing[11])
                sensors, _, reached, _ = world.step(nav, cen)
                steps_done += 1
                if reached:
                    break
            avg_firing = firing_accum / max(1, steps_done)
            inh_ratio = float(np.sum(avg_firing < 0.3) / 12.0)
            ep_score = 100.0 * float(
                np.exp(-((inh_ratio - target_ratio) ** 2) / (2.0 * sigma ** 2))
            )
            total += ep_score
        return total / n_episodes
    return guard_fn


def make_structural_guard(n_episodes=3, n_steps=30):
    """Build guard_fn using compute_iss_from_firing (4-component structural score).

    Combines: inhibition balance (30pt), hub concentration (20pt),
    sensory->motor propagation (30pt), firing diversity (20pt).
    More robust than pure inhibition ratio - non-zero for most brains.
    """
    def guard_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=ep * 11 + 7)
            sensors = world.reset()
            firing_accum = np.zeros(12, dtype=np.float64)
            states = np.zeros(12, dtype=np.float64)
            steps_done = 0
            for _ in range(n_steps):
                states, firing = simulate_step(p, sensors, states)
                firing_accum += firing
                nav = float(firing[5])
                cen = float(firing[11])
                sensors, _, reached, _ = world.step(nav, cen)
                steps_done += 1
                if reached:
                    break
            avg_firing = firing_accum / max(1, steps_done)
            total += float(compute_iss_from_firing(avg_firing, p))
        return total / n_episodes
    return guard_fn


def make_iss_eval(n_episodes=3, n_steps=30):
    """Build eval_fn computing ISS-from-firing. Returns 0-100.

    Uses kathara_brain_sim_v6.compute_iss_from_firing which scores:
      - 30pt: inhibition ratio close to 20%
      - 20pt: hub concentration close to 12.4%
      - 30pt: sensory->motor propagation efficiency
      - 20pt: firing diversity (std)
    """
    def eval_fn(params):
        p = np.array(params, dtype=np.float64)
        total = 0.0
        for ep in range(n_episodes):
            world = FlyWorldV1(seed=ep * 11 + 23)
            sensors = world.reset()
            firing_accum = np.zeros(12, dtype=np.float64)
            states = np.zeros(12, dtype=np.float64)
            steps_done = 0
            for _ in range(n_steps):
                states, firing = simulate_step(p, sensors, states)
                firing_accum += firing
                nav = float(firing[5])
                cen = float(firing[11])
                sensors, _, reached, _ = world.step(nav, cen)
                steps_done += 1
                if reached:
                    break
            avg_firing = firing_accum / max(1, steps_done)
            total += float(compute_iss_from_firing(avg_firing, p))
        return total / n_episodes
    return eval_fn


# ============================================================
# Mode dispatch
# ============================================================

def _get_mode_fns(mode):
    """Return (eval_fn, guard_fn, description) for the mode."""
    if mode == "biological":
        return (
            make_flyworld_eval(FlyWorldV3, n_episodes=4),
            make_structural_guard(),
            "FlyWorldV3 performance vs ISS structural health (4-component)",
        )
    if mode == "progressive":
        return (
            make_flyworld_eval(FlyWorldV3, has_predator=True, n_episodes=4),
            make_flyworld_eval(FlyWorldV1, n_episodes=5),
            "V3+predator performance vs V1 food-finding (catastrophic forgetting)",
        )
    if mode == "intelligent":
        return (
            make_iss_eval(),
            make_flyworld_eval(FlyWorldV1, n_episodes=4),
            "ISS structural intelligence vs V1 survival",
        )
    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Reporting
# ============================================================

_PNAMES = ["gain", "bias", "input_w", "leak", "threshold"]


def _interpret_dim(d):
    """Map dim index (0-90) to brain parameter meaning."""
    if d < 30:
        return f"Edge {d}"
    if d < 90:
        node_idx = (d - 30) // 5
        param_idx = (d - 30) % 5
        return f"Node {node_idx} ({NODE_NAMES.get(node_idx, '?')}).{_PNAMES[param_idx]}"
    return "n_steps"


def report_result(result, mode_name):
    """Print human-readable Sentinel verdict with dim interpretation."""
    v = result["verdict"]
    print(f"\n{'='*60}")
    print(f"Mode: {mode_name}  |  Verdict: {v.upper()}")
    print(f"{'='*60}")
    print(f"eval_score:     {result['eval_score']:.2f}")
    print(f"guard_score:    {result['guard_score']:.2f}")
    print(f"baseline_guard: {result['baseline_guard']:.2f}")
    print(f"proxy_r2:       {result['proxy_r2']:.3f}")
    print(f"elapsed:        {result['elapsed_s']:.1f}s")

    if v == "approved":
        print("\n[approved] Optimization improved eval without breaking guard.")
    elif v == "pivoted":
        safe = result.get("safe_dims") or []
        conflict = result.get("conflict_dims") or []
        print(f"\n[pivoted] Recovered via safe_dims re-optimization.")
        print(f"  Safe dims ({len(safe)}): {safe[:10]}{'...' if len(safe) > 10 else ''}")
        print(f"  Conflict dims ({len(conflict)}): {conflict[:10]}"
              f"{'...' if len(conflict) > 10 else ''}")
        print(f"\n  Top conflict interpretations (these break the guard):")
        for d in conflict[:5]:
            print(f"    [{d:2d}] {_interpret_dim(d)}")
    elif v == "failed":
        print("\n[failed] Could not recover. Guard incompatible with eval.")


def save_result(result, mode_name, out_dir="."):
    """Save result as JSON."""
    out = {
        "mode": mode_name,
        "verdict": result["verdict"],
        "eval_score": round(float(result["eval_score"]), 2),
        "guard_score": round(float(result["guard_score"]), 2),
        "baseline_guard": round(float(result["baseline_guard"]), 2),
        "proxy_r2": round(float(result["proxy_r2"]), 3),
        "elapsed_s": float(result["elapsed_s"]),
        "best_params": [round(float(p), 4) for p in result["best_params"]],
        "safe_dims": result.get("safe_dims"),
        "conflict_dims": result.get("conflict_dims"),
    }
    fname = os.path.join(out_dir, f"brain_sentinel_{mode_name}_result.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {fname}")
    return fname


# ============================================================
# Runners
# ============================================================

def run_mode(mode, time_budget=1800, initial_params=None, learn=False, meta=False):
    """Run a single Sentinel mode on the brain simulation.

    Args:
        mode: biological / progressive / intelligent
        time_budget: seconds
        initial_params: warm-start (optional, 91-D list)
        learn: accumulate experience across runs
        meta: enable Phase3 meta-evolution in optimize() fallback
    """
    eval_fn, guard_fn, desc = _get_mode_fns(mode)
    print(f"\n[brain_sentinel] Mode: {mode}")
    print(f"  {desc}")
    print(f"  time_budget={time_budget}s, n_dims={len(PARAM_RANGES)}")
    if initial_params is not None:
        print(f"  initial_params: [...]  (warm-start from {len(initial_params)}D seed)")
    if learn:
        print(f"  learn=True (experience accumulation enabled)")
    if meta:
        print(f"  meta=True (Phase3 K7-K12 enabled on optimize fallback)")
    print()

    result = Sentinel(
        eval_fn=eval_fn,
        guard_fn=guard_fn,
        param_ranges=PARAM_RANGES,
        param_names=PARAM_NAMES,
        experience_id=f"brain_{mode}",
        initial_params=initial_params,
        learn=learn,
        meta=meta,
    ).run(time_budget=time_budget, verbose=True)

    report_result(result, mode)
    save_result(result, mode)
    return result


def run_evolution(mode="biological", n_gens=5, time_per_gen=600):
    """Multi-generation Sentinel loop. Same experience_id across gens -> owl
    carries dead_dims + warm_start. Structural safe-dim map compounds.
    """
    eval_fn, guard_fn, desc = _get_mode_fns(mode)
    print(f"\n[brain_sentinel] Evolution: inner-mode={mode}")
    print(f"  {desc}")
    print(f"  n_gens={n_gens}, time_per_gen={time_per_gen}s\n")

    history = []
    for gen in range(n_gens):
        print(f"\n{'#'*60}")
        print(f"#  GENERATION {gen + 1}/{n_gens}  (mode={mode})")
        print(f"{'#'*60}\n")

        result = Sentinel(
            eval_fn=eval_fn,
            guard_fn=guard_fn,
            param_ranges=PARAM_RANGES,
            param_names=PARAM_NAMES,
            experience_id=f"brain_evolution_{mode}",
        ).run(time_budget=time_per_gen, verbose=True)

        entry = {
            "gen": gen + 1,
            "verdict": result["verdict"],
            "eval_score": round(float(result["eval_score"]), 2),
            "guard_score": round(float(result["guard_score"]), 2),
            "baseline_guard": round(float(result["baseline_guard"]), 2),
            "safe_dims": result.get("safe_dims"),
            "conflict_dims": result.get("conflict_dims"),
        }
        history.append(entry)
        print(f"\n  Gen {gen + 1}: verdict={result['verdict']}, "
              f"eval={result['eval_score']:.1f}, guard={result['guard_score']:.1f}")

    fname = f"brain_evolution_{mode}_result.json"
    with open(fname, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Evolution history saved: {fname}")

    print(f"\n{'='*60}")
    print(f"  Evolution summary ({mode}):")
    print(f"{'='*60}")
    for h in history:
        print(f"  Gen {h['gen']}: {h['verdict']:>9} | "
              f"eval={h['eval_score']:6.1f}, guard={h['guard_score']:6.1f}")

    return history


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Sentinel x Brain Evolution")
    ap.add_argument(
        "--mode", default="biological",
        choices=["biological", "progressive", "intelligent", "evolution"],
    )
    ap.add_argument("--time-budget", type=int, default=1800,
                    help="Time budget for single-run modes (seconds)")
    ap.add_argument("--n-gens", type=int, default=3,
                    help="Number of generations (evolution mode)")
    ap.add_argument("--time-per-gen", type=int, default=600,
                    help="Time per generation (evolution mode)")
    ap.add_argument("--evolution-mode", default="biological",
                    choices=["biological", "progressive", "intelligent"],
                    help="Inner mode for evolution")
    ap.add_argument("--learn", action="store_true",
                    help="Enable experience accumulation across runs")
    ap.add_argument("--meta", action="store_true",
                    help="Enable Phase3 meta-evolution on optimize fallback")
    args = ap.parse_args()

    if args.mode == "evolution":
        run_evolution(mode=args.evolution_mode,
                      n_gens=args.n_gens,
                      time_per_gen=args.time_per_gen)
    else:
        run_mode(args.mode, time_budget=args.time_budget,
                 learn=args.learn, meta=args.meta)


if __name__ == "__main__":
    main()
