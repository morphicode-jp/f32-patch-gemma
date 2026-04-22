"""Phase 14 — GPU-accelerated DNA re-training under hierarchical 20-shell.

Goal: re-train DNA (kathara_params in core_brain) under the new 20-shell
hierarchical architecture, so behavior recovers to flat baseline and
ideally exceeds it. The 13g honest test showed flat=27.4 food, 13d PEAK=24.2
(-12%) because DNA was trained for flat execution.

Uses GPUBatchRunner for 2x speedup; Cardinal with BEHAVIOR quality (food +
births + alive) as the selection pressure.

Start from current pristine DNA, evolve over 20 epochs, compare final
population behavior to flat baseline.
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

from tamashii.runner_3d import build_fluctlight
from tamashii.gpu_runner import GPUBatchRunner
from tamashii.phase_11_cardinal_3d import (
    GravityUniverse, make_3d_universe_params, mutate_3d_universe_params,
)
from world_3d_gravity import GravityVoxelWorld3D


SHELLS_20 = [
    "core_brain", "brainstem", "cerebellum", "salience",
    "hippocampus", "prefrontal", "dmn", "taboo", "mimir_shell",
    "inhibition_L1", "inhibition_L3", "inhibition_L4",
    "inhibition_L0_a", "inhibition_L0_b",
    "inhibition_L2_a", "inhibition_L2_b",
    "inhibition_L3_b",
    "inhibition_L4_a", "inhibition_L4_b",
    "inhibition_L5",
]


class HierarchicalUniverse(GravityUniverse):
    """GravityUniverse but uses 20-shell hierarchical architecture."""
    shells_override = SHELLS_20

    def initialize_agents(self):
        from tamashii.runner_3d import build_fluctlight
        from kathara16_brain import PARAM_RANGES_16
        agents = []
        rng = np.random.default_rng(self.world_seed + 1000)
        pool = self.__class__.dna_pool
        for i in range(self.agents_init):
            a = build_fluctlight(
                self.shells_override, self.configs_dir,
                trained_dir=self.trained_dir,
                use_hebbian_core=False, use_3d_brain=True)
            if pool is not None and len(pool) > 0:
                parent_dna = pool[int(rng.integers(len(pool)))]
                dna = np.asarray(parent_dna, dtype=np.float64).copy()
                for g in range(min(len(dna), len(PARAM_RANGES_16))):
                    if rng.random() < 0.1:
                        lo, hi = PARAM_RANGES_16[g]
                        dna[g] += rng.normal(0, 0.05 * (hi - lo))
                        dna[g] = np.clip(dna[g], lo, hi)
            else:
                dna = np.asarray(a.shells[0].kathara_params,
                                 dtype=np.float64).copy()
                for g in range(min(len(dna), len(PARAM_RANGES_16))):
                    if rng.random() < 0.1:
                        lo, hi = PARAM_RANGES_16[g]
                        dna[g] += rng.normal(0, 0.12 * (hi - lo))
                        dna[g] = np.clip(dna[g], lo, hi)
            a.shells[0].kathara_params = dna
            agents.append(a)
        self.agents = agents

        from tamashii.phase_11_cardinal_3d import _make_child_factory_ext
        from world_3d_gravity import GravityVoxelWorld3D
        child_factory = _make_child_factory_ext(
            self.configs_dir, self.trained_dir, self.shells_override,
            use_cortical=False)
        self.world = GravityVoxelWorld3D(
            n_agents=self.agents_init, seed=self.world_seed,
            child_factory=child_factory,
            **self.world_params,
        )
        self.world.register_agents(agents)
        self.world.reset()
        for a in agents:
            a.reset_episode()


def run_dna_retrain_gpu(
    n_universes: int = 4,
    agents_per_universe: int = 6,
    epoch_steps: int = 800,
    n_epochs: int = 20,
    base_seed: int = 42,
    device: str = "cuda",
    output: str = "phase_14_dna_retrain_gpu.json",
):
    print("=" * 72, flush=True)
    print(f"  PHASE 14 - DNA re-training on 20-shell hierarchy (GPU)",
          flush=True)
    print(f"  {n_universes}u x {agents_per_universe}a x {n_epochs}ep x "
          f"{epoch_steps}step", flush=True)
    print(f"  Quality: BEHAVIOR (food + births + alive), not ISS", flush=True)
    print("=" * 72, flush=True)

    shells = SHELLS_20
    configs_dir = os.path.join(THIS_DIR, "configs")

    # Build 4 universes
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(base_seed + u_id * 17)
        u = HierarchicalUniverse(
            u_id, wp, agents_per_universe,
            base_seed + u_id * 101, shells, configs_dir, configs_dir)
        universes.append(u)
        print(f"  u{u_id}: gravity={wp['gravity']:+.3f} "
              f"vf={wp['vertical_food_frac']:.2f}", flush=True)

    # Cross-universe batched runner
    all_agents = []
    for u in universes:
        all_agents.extend(u.agents)
    runner = GPUBatchRunner(all_agents, device=device)
    print(f"  Built runner with {len(all_agents)} total agents on {device}",
          flush=True)

    history = []
    t_all = time.time()

    for epoch in range(n_epochs):
        t_ep = time.time()
        for step in range(epoch_steps):
            agent_map = [(u, i) for u in universes for i in range(u.world.n_agents)]
            current_N = len(runner.agents)
            sensors = np.zeros((current_N, 16), dtype=np.float64)
            for mi in range(min(current_N, len(agent_map))):
                u_ref, li = agent_map[mi]
                if u_ref.world.agent_alive[li]:
                    sensors[mi] = u_ref.world.get_sensors(li)
            runner.tick_all(sensors)
            # 4 actions (nav/speed/voice/jump)
            mi = 0
            for u in universes:
                actions = []
                for i in range(u.world.n_agents):
                    if mi < current_N:
                        ag = runner.agents[mi]
                        S = ag.read_state()
                        actions.append((
                            float(np.clip(S[16], 0, 1)),
                            float(np.clip(S[17], 0, 1)),
                            float(np.clip(S[18], 0, 1)),
                            float(np.clip(S[19], 0, 1)) if len(S) > 19 else 0.0,
                        ))
                        mi += 1
                    else:
                        actions.append((0.5, 0, 0, 0))
                u.world.step(actions)
                while len(u.world.agents_external) > len(u.agents):
                    nc = u.world.agents_external[len(u.agents)]
                    nc.reset_episode()
                    u.agents.append(nc)
                    runner.add_agent(nc)

        # Per-epoch metrics
        snaps = []
        for u in universes:
            s = u.world.stats()
            food = sum(u.world.agent_food_eaten[i]
                        for i in range(u.world.n_agents))
            snaps.append({
                "u_id": u.id, "quality": float(u.quality()),
                "food_total": int(food),
                "n_alive": s["n_alive"],
                "n_births": s["n_births"],
                "n_deaths": s["n_deaths"],
                "max_gen": s["max_generation"],
                "mutation_rate": float(u.world_params.get("mutation_rate", 0)),
            })
        ranked = sorted(snaps, key=lambda x: -x["quality"])
        history.append({
            "epoch": epoch, "snapshots": snaps,
            "best": ranked[0]["u_id"],
            "elapsed_ep_s": time.time() - t_ep,
        })

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            print(f"\n  ep{epoch} ({time.time() - t_ep:.1f}s)", flush=True)
            for s in ranked:
                print(f"    u{s['u_id']}: q={s['quality']:+.2f} "
                      f"food={s['food_total']} alive={s['n_alive']} "
                      f"b/d={s['n_births']}/{s['n_deaths']} gen={s['max_gen']}",
                      flush=True)

        # Meta-evolve
        if epoch < n_epochs - 1:
            best_u = next(u for u in universes if u.id == ranked[0]["u_id"])
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=base_seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == ranked[-1]["u_id"])
            old_id = universes[worst_idx].id
            universes[worst_idx] = HierarchicalUniverse(
                old_id, new_params, agents_per_universe,
                base_seed + old_id * 101 + epoch * 7,
                shells, configs_dir, configs_dir)
            # Rebuild runner
            all_agents = []
            for u in universes:
                all_agents.extend(u.agents)
            runner = GPUBatchRunner(all_agents, device=device)

    total = time.time() - t_all
    print(f"\n  DONE in {total:.0f}s ({total/60:.1f} min)", flush=True)

    # Final comparison
    final = history[-1]["snapshots"]
    print(f"\n  === FINAL COMPARISON ===", flush=True)
    print(f"  FLAT baseline (Phase 13g): food=27.4/5agents", flush=True)
    for s in sorted(final, key=lambda x: -x["food_total"]):
        food_per_agent = s["food_total"] / max(1, s["n_alive"])
        marker = "★" if s["food_total"] / 6.0 > 27.4 / 5.0 else "  "
        print(f"  {marker} u{s['u_id']} 20-shell: food={s['food_total']}/"
              f"{s['n_alive']}ag ({food_per_agent:.1f}/ag) "
              f"births={s['n_births']} q={s['quality']:.2f}",
              flush=True)

    out = {
        "config": {
            "n_universes": n_universes, "agents_per_universe": agents_per_universe,
            "epoch_steps": epoch_steps, "n_epochs": n_epochs,
            "base_seed": base_seed, "device": device,
            "architecture": "20-shell hierarchical",
        },
        "total_elapsed_s": round(total, 1),
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=6)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=20)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--output",      type=str,
                     default="phase_14_dna_retrain_gpu.json")
    args = ap.parse_args()
    run_dna_retrain_gpu(
        n_universes=args.n_universes,
        agents_per_universe=args.agents,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        base_seed=args.seed,
        device=args.device,
        output=args.output,
    )


if __name__ == "__main__":
    main()
