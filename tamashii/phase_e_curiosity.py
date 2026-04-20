"""Phase E — runtime Hebbian self-learning + curiosity-driven behavior.

Uses HebbianCoreBrain (replaces static CoreBrain) in a full 7-shell
Tamashii. Intrinsic reward = salience + sensor novelty. No external
reward, no reach target. Watch what emerges.

Tracked per episode per time-window (50-step bins):
  - w_adapt magnitude growth      (self-learning happening?)
  - n_changed_edges               (how broadly does it adapt?)
  - motor_std                     (is behavior becoming more varied?)
  - unique S basins visited       (is exploration expanding?)
  - cumulative reward             (is agent finding novelty?)
  - reach rate                    (emergent task success?)

SUCCESS signals (for curiosity-driven self-learning):
  1. w_adapt grows from 0 over time (learning actually happens)
  2. Later windows show HIGHER motor_std than earlier (more exploration)
  3. Unique basins grows across windows (not stuck)
  4. Reach emerges occasionally (novelty → exploration → finding food)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.core import Tamashii
from tamashii.shell_base import load_shell_from_config
from tamashii.shells.brainstem import Brainstem
from tamashii.shells.cerebellum import Cerebellum
from tamashii.shells.dmn import DMN
from tamashii.shells.hebbian_core import HebbianCoreBrain
from tamashii.shells.hippocampus import Hippocampus
from tamashii.shells.prefrontal import Prefrontal
from tamashii.shells.salience import Salience


CONFIGS = os.path.join(THIS_DIR, "configs")


def build_hebbian_agent(use_trained: bool = True) -> Tamashii:
    """Build 7-shell Tamashii with HebbianCoreBrain instead of CoreBrain.

    If use_trained: load kathara_params from core_brain_trained.json
    as the BASE (before Hebbian updates start accumulating w_adapt).
    """
    def _load(name: str, cls):
        base = os.path.join(CONFIGS, f"{name}.json")
        trained = os.path.join(CONFIGS, f"{name}_trained.json") if use_trained else None
        if trained and not os.path.exists(trained):
            trained = None
        return load_shell_from_config(cls, base, trained)

    # For hebbian_core, we need to load base config from hebbian_core.json
    # but inherit the trained kathara_params from core_brain_trained.json
    base_hebbian = os.path.join(CONFIGS, "hebbian_core.json")
    with open(base_hebbian, "r", encoding="utf-8") as f:
        hcfg = json.load(f)

    if use_trained:
        trained_core = os.path.join(CONFIGS, "core_brain_trained.json")
        if os.path.exists(trained_core):
            with open(trained_core, "r", encoding="utf-8") as f:
                tcore = json.load(f)
            kparams = tcore.get("params", {}).get("kathara_params")
            if kparams is not None:
                hcfg["params"]["kathara_params"] = kparams

    shells = [
        HebbianCoreBrain(hcfg),
        _load("brainstem", Brainstem),
        _load("cerebellum", Cerebellum),
        _load("salience", Salience),
        _load("hippocampus", Hippocampus),
        _load("prefrontal", Prefrontal),
        _load("dmn", DMN),
    ]
    return Tamashii(shells=shells, D=192)


def run_long_episode(agent: Tamashii, world, n_steps: int = 200,
                     ticks_per_step: int = 3, window: int = 50) -> dict:
    """Run a single long episode, collect windowed metrics."""
    N_AGENTS = 1
    sensors_list = world.reset()
    initial_dist = world.get_food_dist(0)
    min_dist = initial_dist
    agent.reset_episode()

    motor_series = []
    S_traj = []
    w_adapt_traj = []
    n_changed_traj = []
    reward_traj = []
    reached_at = -1

    for step in range(n_steps):
        with agent._lock:
            agent.S[0:16] = np.asarray(sensors_list[0], dtype=np.float64)
        for _ in range(ticks_per_step):
            agent.tick_once()
            S_traj.append(agent.read_state())
        S = agent.read_state()
        nav = float(np.clip(S[16], 0.0, 1.0))
        speed = float(np.clip(S[17], 0.0, 1.0))
        voice = float(np.clip(S[18], 0.0, 1.0))
        motor_series.append((nav, speed, voice))

        # Capture Hebbian stats
        hb = agent.shells[0].hebbian_stats()
        w_adapt_traj.append(hb["w_adapt_mean_abs"])
        n_changed_traj.append(hb["n_changed_edges"])
        reward_traj.append(hb["mean_reward"])

        sensors_list, all_reached, flags = world.step([(nav, speed, voice)])
        d = world.get_food_dist(0)
        if d < min_dist:
            min_dist = d
        if flags[0] and reached_at < 0:
            reached_at = step
        if all_reached:
            break

    motor_series = np.array(motor_series)
    S_traj = np.array(S_traj)
    reached = bool(world.reached[0])
    approach = max(0.0, initial_dist - min_dist) / (initial_dist + 1e-6)

    # Windowed analysis (each window = `window` steps)
    n_windows = max(1, len(motor_series) // window)
    windows = []
    for w_i in range(n_windows):
        s = w_i * window
        e = min((w_i + 1) * window, len(motor_series))
        if e - s < 2:
            break
        m_win = motor_series[s:e]
        S_win = S_traj[s * 3:e * 3] if S_traj.shape[0] > e * 3 else S_traj[s:e]

        # Unique bins in this window
        bins = np.round(S_win / 0.5).astype(int)
        u_bins = len({tuple(b) for b in bins})

        windows.append({
            "window": w_i,
            "motor_std": float(m_win.std()),
            "unique_bins": int(u_bins),
            "w_adapt_mean": float(np.mean(w_adapt_traj[s:e])),
            "n_changed_edges_mean": float(np.mean(n_changed_traj[s:e])),
            "reward_mean": float(np.mean(reward_traj[s:e])),
        })

    return {
        "n_steps": len(motor_series),
        "reached": reached,
        "approach": approach,
        "reached_at": reached_at,
        "windows": windows,
        "final_hebbian_stats": agent.shells[0].hebbian_stats(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--use_trained_base", action="store_true", default=True)
    ap.add_argument("--no_trained_base", dest="use_trained_base",
                    action="store_false")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_e_curiosity_result.json")
    args = ap.parse_args()

    from multi_agent_world_N import MultiAgentCoopWorldN

    print("=" * 70, flush=True)
    print("  TAMASHII Phase E - Hebbian self-learning + curiosity", flush=True)
    print(f"  7 shells (hebbian_core + 6 trained), {args.episodes} eps × "
          f"{args.n_steps} steps", flush=True)
    print(f"  Intrinsic reward: 0.7*salience + 0.3*sensor_novelty", flush=True)
    print(f"  w_adapt_carry_over: True (cumulative across episodes)", flush=True)
    print("=" * 70, flush=True)

    # ONE agent built, episodes reuse it so w_adapt carries across
    agent = build_hebbian_agent(use_trained=args.use_trained_base)

    all_episodes = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed_base + ep * 7
        world = MultiAgentCoopWorldN(n_agents=1, seed=seed)
        ep_result = run_long_episode(
            agent, world, n_steps=args.n_steps, window=args.window)

        print(f"\n[ep {ep} seed={seed}] reach={ep_result['reached']} "
              f"approach={ep_result['approach']*100:.1f}% "
              f"steps={ep_result['n_steps']}", flush=True)
        print(f"  final hebbian: "
              f"|w|={ep_result['final_hebbian_stats']['w_adapt_mean_abs']:.4f} "
              f"max|w|={ep_result['final_hebbian_stats']['w_adapt_max_abs']:.3f} "
              f"changed={ep_result['final_hebbian_stats']['n_changed_edges']}/48 "
              f"avg_reward={ep_result['final_hebbian_stats']['mean_reward']:.4f}",
              flush=True)
        print(f"  windows:", flush=True)
        for w in ep_result["windows"]:
            print(f"    [w{w['window']} steps {w['window']*args.window}-"
                  f"{(w['window']+1)*args.window}] "
                  f"motor_std={w['motor_std']:.3f} "
                  f"bins={w['unique_bins']} "
                  f"|w|={w['w_adapt_mean']:.4f} "
                  f"reward={w['reward_mean']:.4f}", flush=True)
        all_episodes.append(ep_result)

    elapsed = time.time() - t0

    # Aggregate: growth of each metric across windows (pooled across eps)
    window_counts = max(len(e["windows"]) for e in all_episodes)
    print(f"\n{'='*70}", flush=True)
    print(f"  CROSS-EPISODE CUMULATIVE EFFECT ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    first_eps = all_episodes[:max(1, len(all_episodes) // 2)]
    last_eps = all_episodes[max(1, len(all_episodes) // 2):]

    def agg_key(eps, metric):
        vals = []
        for e in eps:
            for w in e["windows"]:
                vals.append(w[metric])
        return float(np.mean(vals)) if vals else 0.0

    for metric in ["motor_std", "unique_bins", "w_adapt_mean",
                   "n_changed_edges_mean", "reward_mean"]:
        first_v = agg_key(first_eps, metric)
        last_v = agg_key(last_eps, metric)
        pct = ((last_v - first_v) / abs(first_v) * 100) if first_v != 0 else 0.0
        print(f"  {metric:22s} first-half={first_v:8.4f} "
              f"late-half={last_v:8.4f}  Δ={pct:+.1f}%", flush=True)

    # Final w_adapt (accumulated across all episodes)
    final_hb = all_episodes[-1]["final_hebbian_stats"]
    print(f"\n  FINAL w_adapt (carried across eps):", flush=True)
    print(f"    mean |w|: {final_hb['w_adapt_mean_abs']:.4f}", flush=True)
    print(f"    max  |w|: {final_hb['w_adapt_max_abs']:.4f}", flush=True)
    print(f"    edges changed: {final_hb['n_changed_edges']}/48", flush=True)

    reach_count = sum(1 for e in all_episodes if e["reached"])
    print(f"\n  Emergent reach: {reach_count}/{len(all_episodes)} episodes",
          flush=True)

    out = {
        "use_trained_base": args.use_trained_base,
        "episodes": args.episodes,
        "n_steps": args.n_steps,
        "window": args.window,
        "elapsed_s": round(elapsed, 1),
        "all_episodes": [
            {
                "reached": e["reached"], "approach": e["approach"],
                "n_steps": e["n_steps"], "reached_at": e["reached_at"],
                "windows": e["windows"],
                "final_hebbian_stats": e["final_hebbian_stats"],
            }
            for e in all_episodes
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
