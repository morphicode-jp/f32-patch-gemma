"""Phase 12 GPU v2 — uses optimized GPUBatchRunnerV2 (shared S_batch).

Expected: 3-5x speedup on top of v1 GPU.
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

from tamashii.gpu_runner_v2 import GPUBatchRunnerV2
from tamashii.phase_9_ecology import SHELLS_DEFAULT
from tamashii.phase_11_cardinal_3d import (
    GravityUniverse, make_3d_universe_params, mutate_3d_universe_params,
)


def _dna_stats(u):
    params_list = []
    for i in range(u.world.n_agents):
        if u.world.agent_alive[i]:
            try:
                p = np.asarray(u.world.agents_external[i].shells[0].kathara_params,
                                dtype=np.float64)
                params_list.append(p)
            except Exception:
                pass
    if len(params_list) < 1:
        return {"n": 0, "centroid_norm": 0.0, "spread": 0.0}
    stacked = np.stack(params_list)
    return {"n": len(params_list),
            "centroid_norm": float(np.linalg.norm(stacked.mean(axis=0))),
            "spread": float(np.linalg.norm(stacked.std(axis=0)))}


def _beh_stats(u):
    alive = [i for i in range(u.world.n_agents) if u.world.agent_alive[i]]
    if not alive:
        return {"n_alive": 0}
    zs = [u.world.agent_positions[i][2] for i in alive]
    gens = [u.world.agent_generation[i] for i in alive]
    return {
        "n_alive": len(alive),
        "mean_z": float(np.mean(zs)),
        "max_z": float(max(zs)),
        "gen_max": int(max(gens)),
    }


def run_emergence_gpu_v2(
    n_universes: int = 4,
    agents_per_universe: int = 5,
    epoch_steps: int = 800,
    n_epochs: int = 50,
    base_seed: int = 42,
    output: str = "phase_12_emergence_gpu_v2.json",
    device: str = "cuda",
    continue_from: str = None,
    pristine_tag: str = None,
    save_run_dir: str = "auto",
    run_label: str = "emergence_v2",
):
    shells = SHELLS_DEFAULT
    cfg = os.path.join(THIS_DIR, "configs")

    # Lineage
    if continue_from or pristine_tag:
        sys.path.insert(0, os.path.join(REPO_ROOT, "dna_archive"))
        from dna_io import load_dna_pool
        sources = []
        if continue_from: sources.append(continue_from)
        if pristine_tag: sources.append(f"pristine:{pristine_tag}")
        pool = load_dna_pool(sources)
        print(f"[lineage] Loaded {len(pool)} DNA vectors", flush=True)
        GravityUniverse.dna_pool = pool
    else:
        GravityUniverse.dna_pool = None

    if save_run_dir == "auto":
        sys.path.insert(0, os.path.join(REPO_ROOT, "dna_archive"))
        from dna_io import new_run_dir
        save_run_dir = new_run_dir(label=run_label)
        print(f"[lineage] Will save to: {save_run_dir}", flush=True)
    elif save_run_dir == "none":
        save_run_dir = None

    print("=" * 72, flush=True)
    print(f"  CARDINAL-GPU-V2 3D EMERGENCE (shared S_batch)", flush=True)
    print(f"  {n_universes}u × {agents_per_universe}agents × "
          f"{n_epochs}ep × {epoch_steps}step", flush=True)
    total_target = n_universes * agents_per_universe * n_epochs * epoch_steps
    print(f"  Target: {total_target:,} agent-steps", flush=True)
    print("=" * 72, flush=True)

    # Build universes
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(base_seed + u_id * 17)
        u = GravityUniverse(
            u_id, wp, agents_per_universe, base_seed + u_id * 101,
            shells, cfg, cfg,
        )
        universes.append(u)

    # Aggregate all agents
    all_agents = []
    for u in universes:
        all_agents.extend(u.agents)
    runner = GPUBatchRunnerV2(all_agents, device=device)

    history = []
    selection_events = []
    t_all = time.time()

    for epoch in range(n_epochs):
        t_ep = time.time()
        for step in range(epoch_steps):
            # Gather sensors
            agent_map = []
            for u in universes:
                for i in range(u.world.n_agents):
                    agent_map.append((u, i))
            current_N = len(runner.agents)
            sensors = np.zeros((current_N, 16), dtype=np.float64)
            for mega_idx in range(min(current_N, len(agent_map))):
                u_ref, local_i = agent_map[mega_idx]
                if u_ref.world.agent_alive[local_i]:
                    sensors[mega_idx] = u_ref.world.get_sensors(local_i)

            runner.tick_all(sensors)

            # Extract 4 actions per universe
            mega_idx = 0
            for u in universes:
                N_u = u.world.n_agents
                actions = []
                for i in range(N_u):
                    if mega_idx < current_N:
                        ag = runner.agents[mega_idx]
                        S = ag.read_state()
                        actions.append((
                            float(np.clip(S[16], 0.0, 1.0)),
                            float(np.clip(S[17], 0.0, 1.0)),
                            float(np.clip(S[18], 0.0, 1.0)),
                            float(np.clip(S[19], 0.0, 1.0)) if len(S) > 19 else 0.0,
                        ))
                        mega_idx += 1
                    else:
                        actions.append((0.5, 0.0, 0.0, 0.0))
                u.world.step(actions)
                # Register children
                while len(u.world.agents_external) > len(u.agents):
                    new_child = u.world.agents_external[len(u.agents)]
                    new_child.reset_episode()
                    u.agents.append(new_child)
                    runner.add_agent(new_child)

        # Epoch snapshot
        snaps = []
        for u in universes:
            snaps.append({
                "u_id": u.id,
                "quality": float(u.quality()),
                "world_params": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                  for k, v in u.world_params.items()},
                "dna": _dna_stats(u),
                "behavior": _beh_stats(u),
            })
        ranked = sorted(snaps, key=lambda x: -x["quality"])
        best_id = ranked[0]["u_id"]; worst_id = ranked[-1]["u_id"]
        history.append({"epoch": epoch, "snapshots": snaps,
                         "best": best_id, "worst": worst_id,
                         "elapsed_ep_s": time.time() - t_ep})

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"\n  ep{epoch} ({time.time() - t_ep:.1f}s)", flush=True)
            for s in ranked:
                print(f"    u{s['u_id']}: q={s['quality']:+.2f} "
                      f"alive={s['behavior'].get('n_alive', 0)} "
                      f"gen={s['behavior'].get('gen_max', 0)} "
                      f"mz={s['behavior'].get('mean_z', 0):.2f}", flush=True)

        # Meta-evolve
        if epoch < n_epochs - 1:
            best_u = next(u for u in universes if u.id == best_id)
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=base_seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == worst_id)
            old_id = universes[worst_idx].id
            selection_events.append({
                "epoch": epoch, "replaced_uid": old_id, "parent_uid": best_id,
                "old_quality": ranked[-1]["quality"],
                "best_quality": ranked[0]["quality"],
            })
            new_u = GravityUniverse(
                old_id, new_params, agents_per_universe,
                base_seed + old_id * 101 + epoch * 7,
                shells, cfg, cfg,
            )
            universes[worst_idx] = new_u
            # Rebuild runner (simpler than differential update)
            all_agents = []
            for u in universes:
                all_agents.extend(u.agents)
            runner = GPUBatchRunnerV2(all_agents, device=device)

    total = time.time() - t_all
    print(f"\n  DONE in {total:.0f}s ({total/60:.1f} min)", flush=True)
    print(f"  Effective rate: {total_target/total:.0f} agent-steps/sec",
          flush=True)

    max_gen = max(s["behavior"].get("gen_max", 0)
                    for h in history for s in h["snapshots"])
    max_z = max(s["behavior"].get("mean_z", 0)
                 for h in history for s in h["snapshots"])
    from collections import Counter
    best_counts = Counter(h["best"] for h in history)
    print(f"  Max gen: {max_gen}, Max z: {max_z:.2f}", flush=True)
    print(f"  Winners: {dict(best_counts)}", flush=True)

    out = {
        "config": {
            "n_universes": n_universes, "agents_per_universe": agents_per_universe,
            "epoch_steps": epoch_steps, "n_epochs": n_epochs,
            "base_seed": base_seed, "device": device,
            "continue_from": continue_from, "pristine_tag": pristine_tag,
            "version": "v2",
        },
        "total_elapsed_s": round(total, 1),
        "effective_rate_per_sec": round(total_target / total, 1),
        "history": history,
        "selection_events": selection_events,
        "summary": {
            "max_generation_ever": max_gen,
            "max_z_reached": float(max_z),
            "best_universe_tally": {str(k): v for k, v in best_counts.items()},
        },
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {output}", flush=True)

    if save_run_dir is not None:
        sys.path.insert(0, os.path.join(REPO_ROOT, "dna_archive"))
        from dna_io import save_run
        save_run(save_run_dir, universes, metadata={
            "n_universes": n_universes,
            "agents_per_universe": agents_per_universe,
            "epoch_steps": epoch_steps,
            "n_epochs": n_epochs,
            "base_seed": base_seed,
            "total_elapsed_s": round(total, 1),
            "effective_rate": round(total_target / total, 1),
            "max_generation_ever": max_gen,
            "continue_from": continue_from,
            "pristine_tag": pristine_tag,
            "results_json": output,
            "runner_version": "v2",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=5)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=50)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_12_emergence_gpu_v2.json")
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--continue_from", type=str, default=None)
    ap.add_argument("--pristine",    type=str, default=None)
    ap.add_argument("--save_run",    type=str, default="auto")
    ap.add_argument("--run_label",   type=str, default="emergence_v2")
    args = ap.parse_args()
    run_emergence_gpu_v2(
        n_universes=args.n_universes,
        agents_per_universe=args.agents,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
        output=args.output,
        device=args.device,
        continue_from=args.continue_from,
        pristine_tag=args.pristine,
        save_run_dir=args.save_run,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
