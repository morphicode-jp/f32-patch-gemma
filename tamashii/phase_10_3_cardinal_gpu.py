"""Phase 10.3 Cardinal — GPU cross-universe batched version.

Key optimization vs phase_10_3_cardinal.py:
  Previous: each universe runs its agents sequentially on CPU, universes run sequentially
  This:     all agents from ALL universes are batched together via GPUBatchRunner,
            world.step() remains per-universe (cheap)

Speedup expectation:
  CPU serial (previous):   N_universes × agents_per_u × steps × per_agent_time
  GPU cross-batch:         1 × (N_u × agents_per_u) × steps × batched_time (GPU cheap)

  4 universes × 15 agents = 60 total agents. GPU batched N=60 ticks in ~2ms.
  Vs CPU sequential 60 × 0.17ms = 10ms per full step.
  Expected per-step 5x; full run 11min -> ~2-3min.
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

from tamashii.gpu_runner import GPUBatchRunner
from tamashii.phase_9_ecology import make_child_factory, SHELLS_DEFAULT
from tamashii.phase_10_3_cardinal import (
    Universe, make_universe_params, mutate_universe_params,
)


def run_cardinal_gpu(
    n_universes: int = 4,
    agents_per_universe: int = 15,
    epoch_steps: int = 2500,
    n_epochs: int = 4,
    base_seed: int = 42,
    output: str = "tamashii_phase_10_3_cardinal_gpu.json",
    device: str = "cuda",
):
    shells = SHELLS_DEFAULT
    configs_dir = os.path.join(THIS_DIR, "configs")
    trained_dir = "tamashii/configs"

    print("=" * 72, flush=True)
    print(f"  CARDINAL-GPU — cross-universe batched", flush=True)
    print(f"  {n_universes} universes × {agents_per_universe} agents = "
          f"{n_universes * agents_per_universe} total agents batched", flush=True)
    print(f"  {epoch_steps} steps/epoch × {n_epochs} epochs on {device}", flush=True)
    print("=" * 72, flush=True)

    # Initialize N universes
    universes = []
    for u_id in range(n_universes):
        wp = make_universe_params(base_seed + u_id * 17)
        u = Universe(
            universe_id=u_id, world_params=wp,
            agents_init=agents_per_universe,
            world_seed=base_seed + u_id * 101,
            shells=shells, configs_dir=configs_dir, trained_dir=trained_dir,
        )
        universes.append(u)

    history = []
    t_total = time.time()

    for epoch in range(n_epochs):
        print(f"\n{'='*72}", flush=True)
        print(f"  EPOCH {epoch+1}/{n_epochs}", flush=True)
        print(f"{'='*72}", flush=True)
        t_epoch = time.time()

        # Build megapool: all alive agents from all universes
        # Cross-universe GPU batched runner
        all_agents = []
        agent_to_universe = []  # index of universe for each agent in megapool
        for u in universes:
            for a in u.agents:
                all_agents.append(a)
                agent_to_universe.append(u)

        # GPUBatchRunner handles core_brain GPU + other shells batched CPU
        runner = GPUBatchRunner(all_agents, device=device)

        # Reset all universes + agents
        for u in universes:
            u.world.reset()
            for a in u.agents:
                a.reset_episode()
        runner.reset()

        for step in range(epoch_steps):
            # 1. Gather sensors from each world for each agent
            sensors_mega = np.zeros((len(all_agents), 16), dtype=np.float64)
            idx = 0
            for u in universes:
                N_u = u.world.n_agents
                for i in range(N_u):
                    if idx < len(all_agents) and u.world.agent_alive[i]:
                        sensors_mega[idx] = u.world.get_sensors(i)
                    idx += 1

            # 2. Batched tick: all agents' brains tick on GPU simultaneously
            runner.tick_all(sensors_mega)

            # 3. Extract actions per universe, step each world
            idx = 0
            for u in universes:
                N_u = u.world.n_agents
                actions = []
                for i in range(N_u):
                    if idx < len(all_agents):
                        S = all_agents[idx].read_state()
                        actions.append((
                            float(np.clip(S[16], 0.0, 1.0)),
                            float(np.clip(S[17], 0.0, 1.0)),
                            float(np.clip(S[18], 0.0, 1.0)),
                        ))
                        idx += 1
                    else:
                        actions.append((0.5, 0.0, 0.0))
                u.world.step(actions)
                # Children spawned? Add them
                if len(u.world.agents_external) > len(u.agents):
                    new_agents = u.world.agents_external[len(u.agents):]
                    for na in new_agents:
                        na.reset_episode()
                    u.agents.extend(new_agents)
                    # NOTE: runner does not dynamically add; children won't be
                    # batched this epoch. They'll be included on next epoch's pool rebuild.

            if step % (epoch_steps // 4) == 0:
                stats_str = " ".join(
                    f"u{u.id}=alive{u.world.stats()['n_alive']}"
                    for u in universes)
                print(f"    step {step}: {stats_str}", flush=True)

        # End-of-epoch: record trajectory + compute quality + meta-evolve
        t_u_done = time.time() - t_epoch
        for u in universes:
            u.trajectory.append({"step": epoch_steps, **u.world.stats(), "final": True})
            s = u.world.stats()
            q = u.quality()
            print(f"  u{u.id}: alive={s['n_alive']}/{s['n_total_agents']} "
                  f"births={s['n_births']} deaths={s['n_deaths']} "
                  f"gen={s['max_generation']} dna_div={s['dna_diversity']:.4f} "
                  f"quality={q:.2f}", flush=True)

        ranked = sorted(universes, key=lambda u: -u.quality())
        qualities = [u.quality() for u in ranked]
        print(f"\n  Ranking: {[u.id for u in ranked]} "
              f"qualities: {[round(q,2) for q in qualities]}", flush=True)

        history.append({
            "epoch": epoch,
            "universes": [{
                "id": u.id, "world_params": u.world_params,
                "stats": u.world.stats(), "quality": u.quality(),
                "trajectory": list(u.trajectory),
            } for u in ranked],
        })

        # Meta-evolve: replace worst with variant of best
        if epoch < n_epochs - 1 and len(ranked) >= 2:
            best = ranked[0]
            worst = ranked[-1]
            new_params = mutate_universe_params(
                best.world_params, sigma=0.25, seed=base_seed + epoch * 31)
            print(f"\n  META-EVOLVE: u{worst.id} <- variant of u{best.id} "
                  f"(mut_rate={new_params['mutation_rate']:.3f})", flush=True)
            new_u = Universe(
                universe_id=worst.id, world_params=new_params,
                agents_init=agents_per_universe,
                world_seed=base_seed + worst.id * 101 + epoch * 7,
                shells=shells, configs_dir=configs_dir, trained_dir=trained_dir,
            )
            idx_ = universes.index(worst)
            universes[idx_] = new_u

        print(f"  Epoch {epoch+1} elapsed: {t_u_done:.0f}s", flush=True)

    total_elapsed = time.time() - t_total
    print(f"\n{'='*72}", flush=True)
    print(f"  CARDINAL-GPU SUMMARY "
          f"({total_elapsed:.0f}s = {total_elapsed/60:.1f}min)", flush=True)
    print(f"{'='*72}", flush=True)

    final_ranked = sorted(universes, key=lambda u: -u.quality())
    for u in final_ranked:
        s = u.world.stats()
        print(f"  u{u.id}: quality={u.quality():.2f} alive={s['n_alive']} "
              f"births={s['n_births']} gen_max={s['max_generation']}", flush=True)

    out = {
        "total_elapsed_s": round(total_elapsed, 1),
        "device": device,
        "n_universes": n_universes,
        "agents_per_universe": agents_per_universe,
        "epoch_steps": epoch_steps,
        "n_epochs": n_epochs,
        "history": history,
        "final_ranking": [{
            "id": u.id, "quality": u.quality(),
            "world_params": u.world_params,
            "stats": u.world.stats(),
        } for u in final_ranked],
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents_per_universe", type=int, default=15)
    ap.add_argument("--epoch_steps", type=int, default=2500)
    ap.add_argument("--n_epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_10_3_cardinal_gpu.json")
    args = ap.parse_args()

    run_cardinal_gpu(
        n_universes=args.n_universes,
        agents_per_universe=args.agents_per_universe,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
        output=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
