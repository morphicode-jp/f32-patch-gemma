"""Phase C — 5-shell MVP demo with structure-level metrics.

Runs full 5-shell Tamashii in MultiAgentCoopWorldN (N=3) and captures:
  - Shell delta_norm time series (per shell per tick)
  - S trajectory (all 192 dims over time)
  - Motor trajectory (nav, speed, voice)
  - Sensor input time series

Outputs tamashii_phase_c_result.json with structure metrics + behavior.

SUCCESS (pre-registered):
  - Shell coupling off_diag_mean > 0.05 (shells interact through S)
  - State diversity unique_bins > 10 (not trapped in single basin)
  - Behavioral coherence autocorr_lag1 > 0.2 (not random flailing)
  - reach_rate is reported but NOT optimized — just observation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tamashii.core import Tamashii
from tamashii.metrics import summarize
from tamashii.runner_multiagent import (
    SHELL_REGISTRY, build_agent, extract_action, write_sensors,
)


def run_instrumented_episode(agents: list[Tamashii], world, N_AGENTS: int,
                             n_steps: int, shell_ticks_per_step: int = 3) -> dict:
    """Run episode while recording everything for metrics."""
    sensors_list = world.reset()
    for a in agents:
        a.reset_episode()

    initial_dists = [world.get_food_dist(i) for i in range(N_AGENTS)]
    min_dists = list(initial_dists)

    # Logs: per-tick per-agent
    delta_log_per_agent = [[] for _ in range(N_AGENTS)]
    S_traj_per_agent = [[] for _ in range(N_AGENTS)]
    motor_per_agent = [[] for _ in range(N_AGENTS)]
    sensor_per_agent = [[] for _ in range(N_AGENTS)]

    for step in range(n_steps):
        for i, agent in enumerate(agents):
            write_sensors(agent, sensors_list[i])
            sensor_per_agent[i].append(np.asarray(sensors_list[i]).copy())

        for _ in range(shell_ticks_per_step):
            for i, agent in enumerate(agents):
                agent.tick_once()
                delta_log_per_agent[i].append(agent.shell_delta_norms())
                S_traj_per_agent[i].append(agent.read_state())

        actions = [extract_action(a) for a in agents]
        for i, act in enumerate(actions):
            motor_per_agent[i].append(np.asarray(act))

        sensors_list, all_reached, reached_flags = world.step(actions)
        for i in range(N_AGENTS):
            d = world.get_food_dist(i)
            if d < min_dists[i]:
                min_dists[i] = d
        if all_reached:
            break

    reached = [bool(world.reached[i]) for i in range(N_AGENTS)]
    return {
        "reached": reached,
        "reach_rate": float(np.mean(reached)),
        "n_steps_run": len(motor_per_agent[0]),
        "initial_dists": initial_dists,
        "min_dists": min_dists,
        "delta_log_per_agent": delta_log_per_agent,
        "S_traj_per_agent": [np.array(s) for s in S_traj_per_agent],
        "motor_per_agent": [np.array(m) for m in motor_per_agent],
        "sensor_per_agent": [np.array(s) for s in sensor_per_agent],
    }


def metrics_per_agent(ep_result: dict, N_AGENTS: int) -> list[dict]:
    """Run metrics.summarize() for each agent separately."""
    out = []
    for i in range(N_AGENTS):
        m = summarize(
            delta_log=ep_result["delta_log_per_agent"][i],
            S_trajectory=ep_result["S_traj_per_agent"][i],
            motor_series=ep_result["motor_per_agent"][i],
            input_series=ep_result["sensor_per_agent"][i],
        )
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", type=str,
                    default="core_brain,brainstem,cerebellum,salience,hippocampus")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--n_agents", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=60)
    ap.add_argument("--ticks_per_step", type=int, default=3)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--output", type=str, default="tamashii_phase_c_result.json")
    ap.add_argument("--trained_dir", type=str, default=None,
                    help="directory with *_trained.json overlays")
    args = ap.parse_args()

    from multi_agent_world_N import MultiAgentCoopWorldN

    shells = [s.strip() for s in args.shells.split(",") if s.strip()]
    print("=" * 70, flush=True)
    print(f"  TAMASHII Phase C - {len(shells)}-shell MVP demo", flush=True)
    print(f"  shells: {shells}", flush=True)
    print(f"  N_AGENTS={args.n_agents}, episodes={args.episodes}, "
          f"steps={args.n_steps}, ticks/step={args.ticks_per_step}", flush=True)
    print("=" * 70, flush=True)

    configs_dir = os.path.join(THIS_DIR, "configs")

    all_episodes = []
    per_ep_metrics = []
    t0 = time.time()

    for ep in range(args.episodes):
        seed = args.seed_base + ep * 7
        # Fresh agents per episode (deterministic per seed)
        agents = [
            build_agent(shells, configs_dir, trained_dir=args.trained_dir, D=192)
            for _ in range(args.n_agents)
        ]
        world = MultiAgentCoopWorldN(n_agents=args.n_agents, seed=seed)
        ep_result = run_instrumented_episode(
            agents, world, args.n_agents, args.n_steps, args.ticks_per_step,
        )
        ep_metrics = metrics_per_agent(ep_result, args.n_agents)

        print(f"\n[ep {ep} seed={seed}] reach={ep_result['reach_rate']*100:.0f}% "
              f"steps={ep_result['n_steps_run']}", flush=True)
        for i, m in enumerate(ep_metrics):
            coup = m["shell_coupling"]["off_diag_mean"]
            div_bins = m["state_diversity"]["unique_bins"]
            coh = m["behavioral_coherence"]["autocorr_lag1"]
            nov = m.get("novelty_response", {}).get("response_ratio", 0.0)
            print(f"  agent{i}: coupling={coup:.3f} unique_bins={div_bins} "
                  f"motor_autocorr={coh:.3f} novelty_resp={nov:.3f}",
                  flush=True)

        # Strip numpy arrays for JSON output (keep summary)
        ep_summary = {
            "episode": ep,
            "seed": seed,
            "reach_rate": ep_result["reach_rate"],
            "reached": ep_result["reached"],
            "n_steps": ep_result["n_steps_run"],
            "metrics_per_agent": [
                {
                    "coupling_off_diag_mean": m["shell_coupling"]["off_diag_mean"],
                    "shell_names": m["shell_coupling"]["shell_names"],
                    "state_diversity": m["state_diversity"],
                    "behavioral_coherence": m["behavioral_coherence"],
                    "novelty_response": m.get("novelty_response", {}),
                }
                for m in ep_metrics
            ],
        }
        all_episodes.append(ep_summary)
        per_ep_metrics.append(ep_metrics)

    elapsed = time.time() - t0

    # Aggregate
    reach_mean = float(np.mean([e["reach_rate"] for e in all_episodes]))
    coupling_all = [
        m["coupling_off_diag_mean"]
        for e in all_episodes for m in e["metrics_per_agent"]
    ]
    bins_all = [
        m["state_diversity"]["unique_bins"]
        for e in all_episodes for m in e["metrics_per_agent"]
    ]
    coh_all = [
        m["behavioral_coherence"]["autocorr_lag1"]
        for e in all_episodes for m in e["metrics_per_agent"]
    ]
    nov_all = [
        m["novelty_response"].get("response_ratio", 0.0)
        for e in all_episodes for m in e["metrics_per_agent"]
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE C SUMMARY ({elapsed:.1f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  reach_rate mean:           {reach_mean*100:.1f}% (NOT optimized)", flush=True)
    print(f"  coupling off_diag_mean:    {np.mean(coupling_all):.3f} ± {np.std(coupling_all):.3f}  (target > 0.05)", flush=True)
    print(f"  unique basins mean:        {np.mean(bins_all):.1f} ± {np.std(bins_all):.1f}  (target > 10)", flush=True)
    print(f"  motor autocorr lag1:       {np.mean(coh_all):.3f} ± {np.std(coh_all):.3f}  (target > 0.2)", flush=True)
    print(f"  novelty response ratio:    {np.mean(nov_all):.3f} ± {np.std(nov_all):.3f}", flush=True)

    # Pre-registered pass check
    pass_coupling = np.mean(coupling_all) > 0.05
    pass_diversity = np.mean(bins_all) > 10
    pass_coherence = np.mean(coh_all) > 0.2
    total_pass = int(pass_coupling) + int(pass_diversity) + int(pass_coherence)
    print(f"\n  PRE-REGISTERED CRITERIA: {total_pass}/3 passed", flush=True)
    print(f"    coupling   > 0.05: {'PASS' if pass_coupling else 'FAIL'}", flush=True)
    print(f"    diversity  > 10:   {'PASS' if pass_diversity else 'FAIL'}", flush=True)
    print(f"    coherence  > 0.2:  {'PASS' if pass_coherence else 'FAIL'}", flush=True)

    out = {
        "shells": shells,
        "n_agents": args.n_agents,
        "episodes": all_episodes,
        "aggregate": {
            "reach_rate_mean": reach_mean,
            "coupling_mean": float(np.mean(coupling_all)),
            "coupling_std": float(np.std(coupling_all)),
            "unique_bins_mean": float(np.mean(bins_all)),
            "coherence_mean": float(np.mean(coh_all)),
            "novelty_mean": float(np.mean(nov_all)),
            "pass_coupling": pass_coupling,
            "pass_diversity": pass_diversity,
            "pass_coherence": pass_coherence,
            "total_pass": total_pass,
        },
        "elapsed_s": round(elapsed, 1),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
