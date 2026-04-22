"""Phase 19 - H2 culture inheritance inside Phase 18 2-stage framework

H1b で Hebbian 単独は DNA 撹乱で Phase 14 を超えない。
H1c (Phase 18) で 2 段階 (Cardinal → Hebbian) なら Hebbian が +7.6% 助ける。
H2 (Phase 17) で culture 機構は動くが Hebbian 未熟で効果見えない。

Phase 19 は **H1c 安定 + H2 文化継承** を合体:
  Stage 1 (ep 0-9): Hebbian OFF、純 Cardinal、culture 継承も無意味 (w_adapt=0)
  Stage 2 (ep 10+): Hebbian ON + 子は親の w_adapt 継承

A/B test:
  A: Stage 2 で culture OFF (child starts with w_adapt=0)
  B: Stage 2 で culture ON (child inherits parents' w_adapt with inherit_rate=0.8)

期待: Stage 2 で B の agent は親の食料獲得 reflex を即座に継承
      → B の後半世代が A より効率的 (cumulative differentiation)
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
from tamashii.phase_17_culture_inheritance import make_hebbian_child_factory
from tamashii.phase_18_two_stage import set_hebbian_state
from world_3d_gravity import GravityVoxelWorld3D


class CultureAware2StageUniverse(HebbianHierarchicalUniverse):
    """Phase 18 2-stage + Phase 17 culture child factory.

    inherit_rate 0.0 = no culture (A)
    inherit_rate 0.8 = full culture (B)
    """

    inherit_rate: float = 0.0  # Override per condition
    noise_sigma: float = 0.05

    def initialize_agents(self):
        """Hebbian + culture child factory."""
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
            # Phase 18 Hebbian settings (initially OFF)
            hb = a.shells[0]
            if hasattr(hb, "hebbian_lr"):
                hb.hebbian_lr = 0.0
                hb.hebbian_decay = 0.01
                hb.reward_scale = 1.5
                hb.violation_penalty_weight = 2.0
                hb.w_clip = 0.5
                hb.noise_base = 0.05
                hb.external_reward_weight = 0.0
                hb.salience_weight = 0.05
                hb.sensor_novelty_weight = 0.05
                hb.pred_error_weight = 0.5
                hb.w_adapt_carry_over = True
            # Dampen inhibitory
            for shell in a.shells[1:]:
                if getattr(shell, "shell_sign", 1) == -1:
                    shell.gain *= 0.3
            agents.append(a)
        self.agents = agents

        # Culture child factory
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


class CultureOFF(CultureAware2StageUniverse):
    inherit_rate: float = 0.0


class CultureON(CultureAware2StageUniverse):
    inherit_rate: float = 0.8


def run_condition(U_class, label, n_universes, agents_per_u,
                    epoch_steps, n_epochs, stage1_end, seed):
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}", flush=True)
    shells = SHELLS_20
    configs_dir = os.path.join(THIS_DIR, "configs")

    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = U_class(u_id, wp, agents_per_u, seed + u_id * 101,
                     shells, configs_dir, configs_dir)
        universes.append(u)

    # Stage 1: Hebbian OFF
    set_hebbian_state(universes, active=False)

    trajectory = []
    t_start = time.time()

    for epoch in range(n_epochs):
        if epoch == stage1_end:
            print(f"  *** STAGE 2 (ep {epoch}): Hebbian ON ***", flush=True)
            set_hebbian_state(universes, active=True)

        for u in universes:
            u.run_epoch(epoch_steps, log_every=epoch_steps)
            if epoch >= stage1_end:
                set_hebbian_state([u], active=True)

        snaps = []
        for u in universes:
            s = u.world.stats()
            food = sum(u.world.agent_food_eaten[i]
                        for i in range(u.world.n_agents))
            w_norms = []
            gens = []
            for i in range(u.world.n_agents):
                if u.world.agent_alive[i]:
                    gens.append(u.world.agent_generation[i])
                    sh = u.world.agents_external[i].shells[0]
                    if hasattr(sh, "hebbian_stats"):
                        w_norms.append(sh.hebbian_stats()["w_adapt_mean_abs"])
            snaps.append({
                "u_id": u.id, "quality": float(u.quality()),
                "food_total": int(food),
                "n_alive": s["n_alive"],
                "n_births": s["n_births"],
                "n_deaths": s["n_deaths"],
                "max_gen": s["max_generation"],
                "mean_gen": float(np.mean(gens)) if gens else 0.0,
                "w_adapt_mean": float(np.mean(w_norms)) if w_norms else 0.0,
            })
        stage = "S1" if epoch < stage1_end else "S2"
        trajectory.append({"epoch": epoch, "stage": stage,
                             "snapshots": snaps})

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            best = max(snaps, key=lambda x: x["quality"])
            tot_food = sum(s["food_total"] for s in snaps)
            tot_b = sum(s["n_births"] for s in snaps)
            mean_gen = float(np.mean([s["mean_gen"] for s in snaps]))
            mean_w = float(np.mean([s["w_adapt_mean"] for s in snaps]))
            print(f"  {stage} ep{epoch:2d}: food={tot_food} births={tot_b} "
                  f"mean_gen={mean_gen:.2f} w_norm={mean_w:.3f} "
                  f"best_q={best['quality']:.2f}", flush=True)

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
            set_hebbian_state([universes[worst_idx]],
                               active=(epoch + 1 >= stage1_end))

    total = time.time() - t_start
    return {"label": label, "elapsed_s": round(total, 1),
             "trajectory": trajectory,
             "inherit_rate": U_class.inherit_rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=3)
    ap.add_argument("--agents",      type=int, default=6)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=20)
    ap.add_argument("--stage1_end",  type=int, default=10)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_19_culture_in_stage2.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print(f"  PHASE 19 - Culture in 2-stage: A (no culture) vs B (culture)",
          flush=True)
    print(f"  {args.n_universes}u x {args.agents}a x {args.n_epochs}ep x "
          f"{args.epoch_steps}step, stage1_end={args.stage1_end}", flush=True)
    print("=" * 72, flush=True)

    result_A = run_condition(
        CultureOFF, "A. Culture OFF (inherit_rate=0)",
        args.n_universes, args.agents, args.epoch_steps,
        args.n_epochs, args.stage1_end, args.seed)

    result_B = run_condition(
        CultureON, "B. Culture ON (inherit_rate=0.8)",
        args.n_universes, args.agents, args.epoch_steps,
        args.n_epochs, args.stage1_end, args.seed)

    # Compare stages
    def stage_rate(result, start_ep, end_ep):
        snaps_start = result["trajectory"][start_ep]["snapshots"]
        snaps_end = result["trajectory"][end_ep]["snapshots"]
        food_start = sum(s["food_total"] for s in snaps_start)
        food_end = sum(s["food_total"] for s in snaps_end)
        span_epochs = end_ep - start_ep
        total_agent_steps = (span_epochs * args.epoch_steps *
                               args.n_universes * args.agents)
        return (food_end - food_start) / max(total_agent_steps, 1)

    print(f"\n{'=' * 72}\n  COMPARISON (A vs B, Stage 2 only)\n{'=' * 72}",
          flush=True)

    for r in [result_A, result_B]:
        s1_rate = stage_rate(r, 0, args.stage1_end - 1)
        s2_rate = stage_rate(r, args.stage1_end, args.n_epochs - 1)
        # Final stats
        last = r["trajectory"][-1]["snapshots"]
        tot_food = sum(s["food_total"] for s in last)
        tot_b = sum(s["n_births"] for s in last)
        mean_gen = float(np.mean([s["mean_gen"] for s in last]))
        mean_w = float(np.mean([s["w_adapt_mean"] for s in last]))
        print(f"  {r['label']}", flush=True)
        print(f"    Stage 1 rate: {s1_rate:.4f} food/agent-step", flush=True)
        print(f"    Stage 2 rate: {s2_rate:.4f} food/agent-step "
              f"({(s2_rate/s1_rate - 1)*100:+.1f}% vs S1)", flush=True)
        print(f"    Final: food={tot_food} births={tot_b} "
              f"mean_gen={mean_gen:.2f} w_norm={mean_w:.3f}", flush=True)

    # A/B delta in Stage 2
    a_s2 = stage_rate(result_A, args.stage1_end, args.n_epochs - 1)
    b_s2 = stage_rate(result_B, args.stage1_end, args.n_epochs - 1)
    delta_pct = (b_s2 / a_s2 - 1) * 100 if a_s2 > 0 else 0
    print(f"\n  CULTURE EFFECT (B vs A in Stage 2):", flush=True)
    print(f"    A (no culture) S2 rate:  {a_s2:.4f}", flush=True)
    print(f"    B (with culture) S2 rate: {b_s2:.4f}", flush=True)
    print(f"    Delta: {delta_pct:+.1f}%", flush=True)
    if delta_pct > 10:
        print(f"    [WIN] Culture significantly helps", flush=True)
    elif delta_pct > 2:
        print(f"    [modest] Culture provides small benefit", flush=True)
    else:
        print(f"    [no effect] Culture does not help here", flush=True)

    out = {
        "config": vars(args),
        "result_A": result_A,
        "result_B": result_B,
        "culture_delta_pct": delta_pct,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
