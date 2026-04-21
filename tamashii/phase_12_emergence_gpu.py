"""Phase 12 GPU — Cross-universe batched Cardinal with 3D gravity world.

Combines:
  - phase_10_3_cardinal_gpu.py: cross-universe GPU batching via GPUBatchRunner
  - phase_11_cardinal_3d.py: GravityUniverse, 4-action (nav/speed/voice/jump)
  - phase_12_emergence.py: emergence observation style (qualitative snapshots)

Expected speedup:
  CPU (Phase 12): 800K agent-steps / 20 min = ~670 agent-steps/sec
  GPU target:    800K agent-steps / 2 min  = ~6700 agent-steps/sec (10x)

With RTX 5090 32GB, we can batch N=1000+ agents in single forward pass.
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
from tamashii.phase_9_ecology import SHELLS_DEFAULT
from tamashii.phase_11_cardinal_3d import (
    GravityUniverse, make_3d_universe_params, mutate_3d_universe_params,
)


def dna_centroid_and_spread(u: GravityUniverse) -> dict:
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
        return {"n": 0, "centroid_norm": 0.0, "spread": 0.0,
                 "centroid_first8": []}
    stacked = np.stack(params_list)
    centroid = stacked.mean(axis=0)
    spread = float(np.linalg.norm(stacked.std(axis=0)))
    return {
        "n": len(params_list),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "spread": spread,
        "centroid_first8": [float(x) for x in centroid[:8]],
    }


def behavioral_signature(u: GravityUniverse) -> dict:
    alive = [i for i in range(u.world.n_agents) if u.world.agent_alive[i]]
    if not alive:
        return {"n_alive": 0}
    zs = [u.world.agent_positions[i][2] for i in alive]
    xs = [u.world.agent_positions[i][0] for i in alive]
    ys = [u.world.agent_positions[i][1] for i in alive]
    gens = [u.world.agent_generation[i] for i in alive]
    energies = [u.world.agent_energy[i] for i in alive]
    return {
        "n_alive": len(alive),
        "mean_z": float(np.mean(zs)),
        "max_z": float(max(zs)),
        "pos_spread": float(np.linalg.norm([np.std(xs), np.std(ys), np.std(zs)])),
        "mean_energy": float(np.mean(energies)),
        "energy_spread": float(np.std(energies)),
        "gen_max": int(max(gens)),
        "gen_spread": float(np.std(gens)),
        "n_distinct_gens": len(set(gens)),
    }


def run_emergence_gpu(
    n_universes: int = 4,
    agents_per_universe: int = 5,
    epoch_steps: int = 800,
    n_epochs: int = 50,
    base_seed: int = 42,
    output: str = "phase_12_emergence_gpu.json",
    device: str = "cuda",
    continue_from: str = None,   # path to prior run_dir (lineage continuation)
    pristine_tag: str = None,    # "16N_3d" → load pristine baseline
    save_run_dir: str = None,    # if set, persist final DNA pool here
    run_label: str = "emergence",
):
    shells = SHELLS_DEFAULT
    configs_dir = os.path.join(THIS_DIR, "configs")
    trained_dir = "tamashii/configs"

    # Optional lineage / pristine loading
    if continue_from or pristine_tag:
        sys.path.insert(0, os.path.join(REPO_ROOT, "dna_archive"))
        from dna_io import load_dna_pool, new_run_dir, save_run
        sources = []
        if continue_from:
            sources.append(continue_from)
        if pristine_tag:
            sources.append(f"pristine:{pristine_tag}")
        pool = load_dna_pool(sources, n_wanted=None)
        print(f"[lineage] Loaded {len(pool)} DNA vectors from sources: {sources}",
              flush=True)
        GravityUniverse.dna_pool = pool
    else:
        GravityUniverse.dna_pool = None

    # If asked to save run, auto-create run_dir
    if save_run_dir is None:
        pass
    elif save_run_dir == "auto":
        sys.path.insert(0, os.path.join(REPO_ROOT, "dna_archive"))
        from dna_io import new_run_dir
        save_run_dir = new_run_dir(label=run_label)
        print(f"[lineage] Will save DNA pool to: {save_run_dir}", flush=True)

    print("=" * 72, flush=True)
    print(f"  CARDINAL-GPU 3D EMERGENCE", flush=True)
    print(f"  {n_universes}u × {agents_per_universe}agents = "
          f"{n_universes * agents_per_universe} total agents BATCHED on {device}",
          flush=True)
    print(f"  {epoch_steps} steps/epoch × {n_epochs} epochs", flush=True)
    total_agent_steps = (n_universes * agents_per_universe *
                          n_epochs * epoch_steps)
    print(f"  Total agent-steps target: {total_agent_steps:,}", flush=True)
    print("=" * 72, flush=True)

    # Build 3D universes
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(base_seed + u_id * 17)
        u = GravityUniverse(
            universe_id=u_id, world_params=wp,
            agents_init=agents_per_universe,
            world_seed=base_seed + u_id * 101,
            shells=shells, configs_dir=configs_dir, trained_dir=trained_dir,
        )
        universes.append(u)
        print(f"  u{u_id}: gravity={wp['gravity']:+.3f} "
              f"vf={wp['vertical_food_frac']:.2f} "
              f"jump={wp['jump_impulse']:.2f}", flush=True)

    # Build cross-universe megapool
    all_agents = []
    for u in universes:
        for a in u.agents:
            all_agents.append(a)

    runner = GPUBatchRunner(all_agents, device=device)

    history = []
    selection_events = []
    t_total = time.time()

    for epoch in range(n_epochs):
        t_epoch = time.time()

        for step in range(epoch_steps):
            # Gather sensors from all universes
            agent_map = []
            for u in universes:
                for i in range(u.world.n_agents):
                    agent_map.append((u, i))
            current_N = len(runner.agents)
            sensors_mega = np.zeros((current_N, 16), dtype=np.float64)
            for mega_idx in range(min(current_N, len(agent_map))):
                u_ref, local_i = agent_map[mega_idx]
                if u_ref.world.agent_alive[local_i]:
                    sensors_mega[mega_idx] = u_ref.world.get_sensors(local_i)

            # Batched GPU tick
            runner.tick_all(sensors_mega)

            # Extract 4-action tuples per universe, step each world
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
                # Register children with runner
                while len(u.world.agents_external) > len(u.agents):
                    new_child = u.world.agents_external[len(u.agents)]
                    new_child.reset_episode()
                    u.agents.append(new_child)
                    runner.add_agent(new_child)

        # Per-epoch snapshot
        snapshots = []
        for u in universes:
            wp = u.world_params
            dna = dna_centroid_and_spread(u)
            beh = behavioral_signature(u)
            snapshots.append({
                "u_id": u.id,
                "quality": float(u.quality()),
                "world_params": {
                    "gravity": float(wp.get("gravity", 0)),
                    "jump_impulse": float(wp.get("jump_impulse", 0)),
                    "max_height": float(wp.get("max_height", 0)),
                    "vertical_food_frac": float(wp.get("vertical_food_frac", 0)),
                    "mutation_rate": float(wp.get("mutation_rate", 0)),
                    "n_food": int(wp.get("n_food", 0)),
                },
                "dna": dna,
                "behavior": beh,
            })
        ranked = sorted(snapshots, key=lambda x: -x["quality"])
        best_id = ranked[0]["u_id"]
        worst_id = ranked[-1]["u_id"]
        history.append({
            "epoch": epoch,
            "snapshots": snapshots,
            "best": best_id, "worst": worst_id,
            "elapsed_ep_s": time.time() - t_epoch,
        })

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"\n  === ep{epoch} ({time.time() - t_epoch:.1f}s) ===", flush=True)
            for s in ranked:
                b = s["behavior"]; w = s["world_params"]; d = s["dna"]
                print(f"    u{s['u_id']}: q={s['quality']:+.2f} "
                      f"alive={b.get('n_alive', 0)} gen={b.get('gen_max', 0)} "
                      f"mz={b.get('mean_z', 0):.2f} "
                      f"mut={w['mutation_rate']:.3f}", flush=True)

        # Meta-evolve: worst replaced by mutated variant of best
        if epoch < n_epochs - 1:
            best_u = next(u for u in universes if u.id == best_id)
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=base_seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == worst_id)
            old_id = universes[worst_idx].id
            selection_events.append({
                "epoch": epoch,
                "replaced_uid": old_id,
                "parent_uid": best_id,
                "old_quality": ranked[-1]["quality"],
                "best_quality": ranked[0]["quality"],
                "new_gravity": float(new_params.get("gravity", 0)),
                "new_mut_rate": float(new_params.get("mutation_rate", 0)),
            })
            new_u = GravityUniverse(
                universe_id=old_id, world_params=new_params,
                agents_init=agents_per_universe,
                world_seed=base_seed + old_id * 101 + epoch * 7,
                shells=shells, configs_dir=configs_dir, trained_dir=trained_dir,
            )
            # Rebuild runner: remove old universe's agents, add new ones
            # (simplest: full rebuild on meta-evolve — cheap at epoch boundary)
            universes[worst_idx] = new_u
            all_agents = []
            for u in universes:
                for a in u.agents:
                    all_agents.append(a)
            runner = GPUBatchRunner(all_agents, device=device)

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 72}", flush=True)
    print(f"  GPU RUN DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
          flush=True)
    print(f"  Effective rate: {total_agent_steps/total_elapsed:.0f} agent-steps/sec",
          flush=True)
    print(f"{'=' * 72}", flush=True)

    # Summary metrics
    max_gen = max(s["behavior"].get("gen_max", 0)
                    for h in history for s in h["snapshots"])
    max_z = max(s["behavior"].get("mean_z", 0)
                 for h in history for s in h["snapshots"])
    from collections import Counter
    best_counts = Counter(h["best"] for h in history)
    print(f"\n  Max generation: {max_gen}", flush=True)
    print(f"  Max mean z reached: {max_z:.2f}", flush=True)
    print(f"  Best-universe tally: {dict(best_counts)}", flush=True)

    out = {
        "config": dict(
            n_universes=n_universes, agents_per_universe=agents_per_universe,
            epoch_steps=epoch_steps, n_epochs=n_epochs,
            base_seed=base_seed, device=device,
            continue_from=continue_from, pristine_tag=pristine_tag,
        ),
        "total_elapsed_s": round(total_elapsed, 1),
        "effective_rate_per_sec": round(total_agent_steps / total_elapsed, 1),
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
    print(f"\n  Saved: {output}", flush=True)

    # Persist DNA pool
    if save_run_dir is not None:
        sys.path.insert(0, os.path.join(REPO_ROOT, "dna_archive"))
        from dna_io import save_run as save_run_fn
        save_run_fn(save_run_dir, universes, metadata={
            "n_universes": n_universes,
            "agents_per_universe": agents_per_universe,
            "epoch_steps": epoch_steps,
            "n_epochs": n_epochs,
            "base_seed": base_seed,
            "device": device,
            "continue_from": continue_from,
            "pristine_tag": pristine_tag,
            "total_elapsed_s": round(total_elapsed, 1),
            "max_generation_ever": max_gen,
            "max_z_reached": float(max_z),
            "results_json": output,
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
                     default="phase_12_emergence_gpu.json")
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--continue_from", type=str, default=None,
                     help="path to prior run_dir for lineage continuation")
    ap.add_argument("--pristine", type=str, default=None,
                     help="tag like '16N_3d' to load pristine baseline DNA")
    ap.add_argument("--save_run", type=str, default="auto",
                     help="'auto' | path | 'none' — where to archive DNA")
    ap.add_argument("--run_label", type=str, default="emergence")
    args = ap.parse_args()
    save_run_dir = None if args.save_run == "none" else args.save_run
    run_emergence_gpu(
        n_universes=args.n_universes,
        agents_per_universe=args.agents,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
        output=args.output,
        device=args.device,
        continue_from=args.continue_from,
        pristine_tag=args.pristine,
        save_run_dir=save_run_dir,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
