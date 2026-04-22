"""Phase 18 (H1c) - 2 段階訓練: Cardinal 先行 → Hebbian 後発

H1b の問題: Hebbian が Cardinal 進化を撹乱、Phase 14 の 29% しか出ない。

biological 発想:
  - 動物は DNA で基本反射を持って生まれる (胎児期 = Cardinal 進化)
  - 生後、経験で fine-tune する (Hebbian 学習)
  - → 2 つの学習は timescale が違う、**重ねるタイミング**が重要

H1c 設計:
  Phase 1 (ep 0 〜 stage1_end): Hebbian OFF (lr=0) → 純 Cardinal 進化
                                 = Phase 14 相当の DNA 訓練
  Phase 2 (ep stage1_end +):   Hebbian ON (lr=normal) → 個体が学ぶ
                                 = 大人 agent の生涯学習

期待: Phase 1 で Phase 14 級 base 確立、Phase 2 で更に向上。
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

from tamashii.phase_11_cardinal_3d import (
    make_3d_universe_params, mutate_3d_universe_params,
)
from tamashii.phase_14_dna_retrain_gpu import SHELLS_20
from tamashii.phase_16_hebbian_hierarchical import HebbianHierarchicalUniverse


def set_hebbian_state(universes, active: bool):
    """Toggle Hebbian learning on all agents (including children)."""
    for u in universes:
        for i in range(u.world.n_agents):
            if u.world.agent_alive[i]:
                shell = u.world.agents_external[i].shells[0]
                if hasattr(shell, "hebbian_lr"):
                    if active:
                        shell.hebbian_lr = 0.003  # moderate (from H1b)
                        shell.external_reward_weight = 10.0
                    else:
                        shell.hebbian_lr = 0.0      # effectively static
                        shell.external_reward_weight = 0.0


def run_two_stage(n_universes=4, agents_per_u=6, epoch_steps=800,
                    n_epochs_total=25, stage1_end=12,
                    seed=42, output="phase_18_two_stage.json"):
    """Run 2-stage training:
      Stage 1 (ep 0 to stage1_end): Hebbian OFF, pure Cardinal
      Stage 2 (ep stage1_end to n_epochs_total): Hebbian ON, lifetime learning
    """
    print("=" * 72, flush=True)
    print(f"  PHASE 18 (H1c) - 2 段階訓練 (Cardinal → Hebbian)", flush=True)
    print(f"  Stage 1 (ep 0 - {stage1_end-1}): Hebbian OFF, pure Cardinal",
          flush=True)
    print(f"  Stage 2 (ep {stage1_end} - {n_epochs_total-1}): Hebbian ON",
          flush=True)
    print(f"  {n_universes}u x {agents_per_u}a x {epoch_steps}step/ep",
          flush=True)
    print("=" * 72, flush=True)

    shells = SHELLS_20
    configs_dir = os.path.join(THIS_DIR, "configs")

    # Build universes with Hebbian INFRASTRUCTURE but initially INACTIVE
    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = HebbianHierarchicalUniverse(
            u_id, wp, agents_per_u, seed + u_id * 101,
            shells, configs_dir, configs_dir)
        universes.append(u)
    print(f"  Built {n_universes} universes (Hebbian infra ready)", flush=True)

    # Start with Hebbian OFF
    set_hebbian_state(universes, active=False)

    history = []
    t_all = time.time()

    for epoch in range(n_epochs_total):
        t_ep = time.time()

        # Stage 2 trigger: turn Hebbian ON at stage1_end
        if epoch == stage1_end:
            print(f"\n  *** STAGE 2 TRANSITION (ep {epoch}): Hebbian ACTIVATED ***",
                  flush=True)
            set_hebbian_state(universes, active=True)

        # Run each universe for this epoch
        for u in universes:
            u.run_epoch(epoch_steps, log_every=epoch_steps)
            # Re-apply Hebbian state to any new children
            if epoch >= stage1_end:
                set_hebbian_state([u], active=True)

        # Per-epoch snapshot
        snaps = []
        for u in universes:
            s = u.world.stats()
            food = sum(u.world.agent_food_eaten[i]
                        for i in range(u.world.n_agents))
            w_norms = []
            for i in range(u.world.n_agents):
                if u.world.agent_alive[i]:
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
                "w_adapt_mean": float(np.mean(w_norms)) if w_norms else 0.0,
            })
        ranked = sorted(snaps, key=lambda x: -x["quality"])
        stage = "S1 (DNA)" if epoch < stage1_end else "S2 (Heb)"
        history.append({"epoch": epoch, "stage": stage,
                         "snapshots": snaps,
                         "best": ranked[0]["u_id"],
                         "elapsed_ep_s": time.time() - t_ep})

        if epoch % 2 == 0 or epoch == n_epochs_total - 1:
            b = ranked[0]
            print(f"\n  {stage} ep{epoch} ({time.time() - t_ep:.1f}s) "
                  f"best u{b['u_id']} q={b['quality']:.2f} "
                  f"food={b['food_total']} "
                  f"b/d={b['n_births']}/{b['n_deaths']} "
                  f"alive={b['n_alive']} w_norm={b['w_adapt_mean']:.3f}",
                  flush=True)

        # Meta-evolve
        if epoch < n_epochs_total - 1:
            best_u = next(u for u in universes if u.id == ranked[0]["u_id"])
            new_params = mutate_3d_universe_params(
                best_u.world_params, seed=seed + epoch * 37)
            worst_idx = next(i for i, u in enumerate(universes)
                              if u.id == ranked[-1]["u_id"])
            old_id = universes[worst_idx].id
            universes[worst_idx] = HebbianHierarchicalUniverse(
                old_id, new_params, agents_per_u,
                seed + old_id * 101 + epoch * 7,
                shells, configs_dir, configs_dir)
            # Set new universe to current stage's Hebbian state
            set_hebbian_state([universes[worst_idx]],
                               active=(epoch + 1 >= stage1_end))

    total = time.time() - t_all
    print(f"\n  DONE in {total:.0f}s ({total/60:.1f} min)", flush=True)

    # Stage comparison
    stage1_last = stage1_end - 1
    s1_snap = history[stage1_last]["snapshots"]
    s2_final_snap = history[-1]["snapshots"]
    s1_food = sum(s["food_total"] for s in s1_snap)
    s2_food = sum(s["food_total"] for s in s2_final_snap)
    s1_rate = (s1_food - sum(s["food_total"] for s in history[max(0, stage1_last-1)]["snapshots"])) / \
              (epoch_steps * n_universes * agents_per_u)
    # Rate at end of Stage 2
    if len(history) > 1:
        s2_rate = (s2_food - sum(s["food_total"] for s in history[-2]["snapshots"])) / \
                  (epoch_steps * n_universes * agents_per_u)
    else:
        s2_rate = 0
    print(f"\n  === STAGE COMPARISON ===", flush=True)
    print(f"  Stage 1 end (ep {stage1_last}): food_cum={s1_food} "
          f"rate~={s1_rate:.4f}/agent-step", flush=True)
    print(f"  Stage 2 end (ep {n_epochs_total-1}): food_cum={s2_food} "
          f"rate~={s2_rate:.4f}/agent-step", flush=True)
    if s2_rate > s1_rate * 1.2:
        print(f"  [WIN] Hebbian STAGE 2 HELPED (+{(s2_rate/s1_rate - 1)*100:.1f}%)",
              flush=True)
    elif s2_rate > s1_rate * 0.8:
        print(f"  ~ Hebbian neutral (stage 2 ~= stage 1)", flush=True)
    else:
        print(f"  [LOST] Hebbian HURT (stage 2 -{(1 - s2_rate/s1_rate)*100:.1f}%)",
              flush=True)

    # Phase 14 reference
    print(f"\n  === vs Phase 14 (20-ep static, pretrained) ===", flush=True)
    print(f"  Phase 14 food rate: 0.17/agent-step (best known so far)",
          flush=True)
    if s2_rate > 0.17:
        print(f"  [BEAT] Phase 18 BEAT Phase 14 ({s2_rate/0.17*100:.1f}%)",
              flush=True)
    else:
        print(f"  Phase 18 = {s2_rate/0.17*100:.1f}% of Phase 14",
              flush=True)

    out = {
        "config": {
            "n_universes": n_universes, "agents_per_universe": agents_per_u,
            "epoch_steps": epoch_steps, "n_epochs_total": n_epochs_total,
            "stage1_end": stage1_end, "seed": seed,
        },
        "total_elapsed_s": round(total, 1),
        "stage1_rate": round(s1_rate, 4),
        "stage2_rate": round(s2_rate, 4),
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=6)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=25)
    ap.add_argument("--stage1_end",  type=int, default=12)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      type=str,
                     default="phase_18_two_stage.json")
    args = ap.parse_args()
    run_two_stage(
        n_universes=args.n_universes,
        agents_per_u=args.agents,
        epoch_steps=args.epoch_steps,
        n_epochs_total=args.n_epochs,
        stage1_end=args.stage1_end,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
