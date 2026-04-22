"""Phase 17 (H2) - 親から子へ w_adapt 継承 (文化機構)

H1 (Phase 16) で HebbianCoreBrain による生涯学習を実現。
H2 では、子が生まれる時に親の学習済 w_adapt を継承する。

意味:
  - 進化 (Cardinal): DNA (kathara_params) が世代を超えて進化 (生物遺伝)
  - 学習 (Hebbian): 個体の一生の中で w_adapt が変わる (後天的経験)
  - 文化 (H2 新): 親の w_adapt が子に渡る (後天的継承、教育)

これはラマルク主義的な継承 (獲得形質の遺伝)。生物界ではまれ。でも:
  - 人間社会: 教育・文化・言語がこれに相当
  - 我々 Fluctlight: 設計可能 → 文化実装

比較実験:
  A. 継承 OFF (Phase 16 ベース): 子は w_adapt=0 から始める
  B. 継承 ON: 子は 親平均 w_adapt (+ノイズ) から始める

  予想: B の第 N 世代は A より高効率 (世代進むほど顕著)
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
from tamashii.phase_11_cardinal_3d import (
    make_3d_universe_params, mutate_3d_universe_params,
)
from tamashii.phase_14_dna_retrain_gpu import SHELLS_20
from tamashii.phase_16_hebbian_hierarchical import HebbianHierarchicalUniverse
from world_3d_gravity import GravityVoxelWorld3D


def make_hebbian_child_factory(configs_dir: str, trained_dir: str,
                                 shells: list[str],
                                 inherit_rate: float = 0.8,
                                 noise_sigma: float = 0.05):
    """Child factory: Hebbian agent + parent w_adapt 継承.

    inherit_rate: 0.0=無継承(=Phase 16), 1.0=完全コピー
    """
    rng = np.random.default_rng(42)

    def child_factory(parent_i, parent_j, child_dna):
        child = build_fluctlight(
            shells, configs_dir, trained_dir=trained_dir,
            use_hebbian_core=True, use_3d_brain=True)
        # DNA (既存)
        child.shells[0].kathara_params = np.asarray(child_dna, dtype=np.float64)
        # Enable lifetime learning persistence
        if hasattr(child.shells[0], "w_adapt_carry_over"):
            child.shells[0].w_adapt_carry_over = True
        # ★ H2: 親の w_adapt を継承 ★
        if hasattr(child.shells[0], "inherit_w_adapt_from"):
            parent_shells = []
            if parent_i is not None and hasattr(parent_i, "shells"):
                parent_shells.append(parent_i.shells[0])
            if parent_j is not None and hasattr(parent_j, "shells"):
                parent_shells.append(parent_j.shells[0])
            if parent_shells:
                child.shells[0].inherit_w_adapt_from(
                    parent_shells,
                    inherit_rate=inherit_rate,
                    noise_sigma=noise_sigma,
                    rng=rng)
        return child
    return child_factory


class CultureHebbianUniverse(HebbianHierarchicalUniverse):
    """HebbianHierarchicalUniverse + 親→子 w_adapt 継承."""

    inherit_rate: float = 0.8  # クラス属性、サブクラスで上書き可
    noise_sigma: float = 0.05

    def initialize_agents(self):
        """Hebbian core + culture child factory."""
        from kathara16_brain import PARAM_RANGES_16
        agents = []
        rng = np.random.default_rng(self.world_seed + 1000)
        pool = self.__class__.dna_pool

        for i in range(self.agents_init):
            a = build_fluctlight(
                self.shells_override, self.configs_dir,
                trained_dir=self.trained_dir,
                use_hebbian_core=True, use_3d_brain=True)
            if pool is not None and len(pool) > 0:
                parent_dna = pool[int(rng.integers(len(pool)))]
                dna = np.asarray(parent_dna, dtype=np.float64).copy()
                noise_scale = 0.05
            else:
                dna = np.asarray(a.shells[0].kathara_params,
                                 dtype=np.float64).copy()
                noise_scale = 0.12
            for g in range(min(len(dna), len(PARAM_RANGES_16))):
                if rng.random() < 0.1:
                    lo, hi = PARAM_RANGES_16[g]
                    dna[g] += rng.normal(0, noise_scale * (hi - lo))
                    dna[g] = np.clip(dna[g], lo, hi)
            a.shells[0].kathara_params = dna
            if hasattr(a.shells[0], "w_adapt_carry_over"):
                a.shells[0].w_adapt_carry_over = True
            agents.append(a)
        self.agents = agents

        # ★ Culture child factory ★
        child_factory = make_hebbian_child_factory(
            self.configs_dir, self.trained_dir, self.shells_override,
            inherit_rate=self.__class__.inherit_rate,
            noise_sigma=self.__class__.noise_sigma)
        self.world = GravityVoxelWorld3D(
            n_agents=self.agents_init, seed=self.world_seed,
            child_factory=child_factory,
            **self.world_params,
        )
        self.world.register_agents(agents)
        self.world.reset()
        for a in agents:
            a.reset_episode()


class NoCultureHebbianUniverse(CultureHebbianUniverse):
    """Control: Hebbian but no culture inheritance (inherit_rate=0)."""
    inherit_rate: float = 0.0


def run_comparison(n_universes=2, agents_per_u=6, epoch_steps=800,
                    n_epochs=15, seed=42, output="phase_17_culture_compare.json"):
    """Compare: A (no inheritance) vs B (inheritance)."""
    print("=" * 72, flush=True)
    print(f"  PHASE 17 (H2) - Culture inheritance A/B test", flush=True)
    print(f"  {n_universes}u x {agents_per_u}a x {n_epochs}ep x "
          f"{epoch_steps}step", flush=True)
    print("=" * 72, flush=True)

    shells = SHELLS_20
    configs_dir = os.path.join(THIS_DIR, "configs")

    results = {}
    for label, U_class in [("A_no_culture", NoCultureHebbianUniverse),
                             ("B_culture", CultureHebbianUniverse)]:
        print(f"\n  === {label} (inherit_rate={U_class.inherit_rate}) ===",
              flush=True)
        universes = []
        for u_id in range(n_universes):
            wp = make_3d_universe_params(seed + u_id * 17)
            u = U_class(u_id, wp, agents_per_u, seed + u_id * 101,
                         shells, configs_dir, configs_dir)
            universes.append(u)

        trajectory = []
        t_ep_total = time.time()

        for epoch in range(n_epochs):
            t_ep = time.time()
            for u in universes:
                u.run_epoch(epoch_steps, log_every=epoch_steps)

            snaps = []
            for u in universes:
                s = u.world.stats()
                food = sum(u.world.agent_food_eaten[i]
                            for i in range(u.world.n_agents))
                # Track generation stats
                alive = [i for i in range(u.world.n_agents)
                          if u.world.agent_alive[i]]
                gens = [u.world.agent_generation[i] for i in alive]
                gen_mean = float(np.mean(gens)) if gens else 0.0
                # w_adapt across alive
                w_norms = []
                for i in alive:
                    sh = u.world.agents_external[i].shells[0]
                    if hasattr(sh, "hebbian_stats"):
                        w_norms.append(sh.hebbian_stats()["w_adapt_mean_abs"])
                snaps.append({
                    "u_id": u.id,
                    "quality": float(u.quality()),
                    "food_total": int(food),
                    "n_alive": s["n_alive"],
                    "n_births": s["n_births"],
                    "n_deaths": s["n_deaths"],
                    "max_gen": s["max_generation"],
                    "mean_gen": gen_mean,
                    "w_adapt_mean": float(np.mean(w_norms)) if w_norms else 0.0,
                })
            trajectory.append({"epoch": epoch, "snapshots": snaps,
                                "elapsed_ep_s": time.time() - t_ep})

            if epoch % 3 == 0 or epoch == n_epochs - 1:
                best = max(snaps, key=lambda x: x["quality"])
                print(f"    ep{epoch} ({time.time() - t_ep:.1f}s) "
                      f"best u{best['u_id']} q={best['quality']:.2f} "
                      f"food={best['food_total']} "
                      f"mean_gen={best['mean_gen']:.2f} "
                      f"w_norm={best['w_adapt_mean']:.3f}",
                      flush=True)

            # Meta-evolve
            if epoch < n_epochs - 1:
                ranked = sorted(snaps, key=lambda x: -x["quality"])
                best_u = next(u for u in universes if u.id == ranked[0]["u_id"])
                new_params = mutate_3d_universe_params(
                    best_u.world_params, seed=seed + epoch * 37)
                worst_idx = next(i for i, u in enumerate(universes)
                                  if u.id == ranked[-1]["u_id"])
                old_id = universes[worst_idx].id
                universes[worst_idx] = U_class(
                    old_id, new_params, agents_per_u,
                    seed + old_id * 101 + epoch * 7,
                    shells, configs_dir, configs_dir)

        total_ep = time.time() - t_ep_total
        results[label] = {
            "trajectory": trajectory,
            "total_elapsed_s": round(total_ep, 1),
            "inherit_rate": U_class.inherit_rate,
        }
        print(f"  {label} total: {total_ep:.0f}s", flush=True)

    # Compare final epochs
    print(f"\n{'=' * 72}\n  COMPARISON\n{'=' * 72}", flush=True)
    for label in ["A_no_culture", "B_culture"]:
        r = results[label]
        last = r["trajectory"][-1]["snapshots"]
        total_food = sum(s["food_total"] for s in last)
        total_births = sum(s["n_births"] for s in last)
        total_alive = sum(s["n_alive"] for s in last)
        mean_gen = float(np.mean([s["mean_gen"] for s in last]))
        mean_w = float(np.mean([s["w_adapt_mean"] for s in last]))
        print(f"  {label}: food={total_food} births={total_births} "
              f"alive={total_alive} mean_gen={mean_gen:.2f} w_norm={mean_w:.3f}",
              flush=True)

    # Honest verdict
    a_food = sum(s["food_total"] for s in results["A_no_culture"]["trajectory"][-1]["snapshots"])
    b_food = sum(s["food_total"] for s in results["B_culture"]["trajectory"][-1]["snapshots"])
    delta_pct = 100 * (b_food - a_food) / max(a_food, 1)
    print(f"\n  Culture effect: B vs A = {delta_pct:+.1f}% food", flush=True)
    if delta_pct > 20:
        print(f"  ★ CULTURE WORKS: parent→child w_adapt transfer accelerates behavior",
              flush=True)
    elif delta_pct > 5:
        print(f"  ~ Modest culture effect", flush=True)
    else:
        print(f"  - No clear culture effect (may need longer runs or different inherit_rate)",
              flush=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=2)
    ap.add_argument("--agents",      type=int, default=6)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=15)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_17_culture_compare.json")
    args = ap.parse_args()
    run_comparison(
        n_universes=args.n_universes,
        agents_per_u=args.agents,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
