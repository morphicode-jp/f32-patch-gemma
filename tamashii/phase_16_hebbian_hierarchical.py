"""Phase 16 (H1) - HebbianCoreBrain を 20-shell 階層に wire-in

User vision: 人間的な学習 = 個体の一生の中で学ぶ。
HebbianCoreBrain が既に STDP + 内因報酬の runtime 学習を実装済。
これを 20-shell 階層に投入 → 進化 (Cardinal) + 生涯学習 (Hebbian) の 2 層。

比較対象:
  Phase 14 (static CoreBrain):  食料 0.17/agent-step、生涯学習なし
  Phase 16 (Hebbian core):      期待 0.17+、agent 内で reflex 獲得

Hebbian のポイント:
  - w_adapt が個体の生涯中に更新される
  - 報酬 = salience attention + sensor novelty + 予測誤差 - 違反 penalty
  - reset() 時 w_adapt_carry_over=True で episode 越え継承
  - 子の出生時は parent の w_adapt を継承 (H2 で利用、今は baseline)

20-shell 階層構成は Phase 14 と同一。core_brain だけ HebbianCoreBrain に置換。
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
    make_3d_universe_params, mutate_3d_universe_params,
)
from tamashii.phase_14_dna_retrain_gpu import HierarchicalUniverse, SHELLS_20
from world_3d_gravity import GravityVoxelWorld3D


class HebbianHierarchicalUniverse(HierarchicalUniverse):
    """Phase 14 の階層 + HebbianCoreBrain (runtime 学習).

    [H1b] run_epoch を override して、食料獲得信号を各 tick 前に
    S[189] (external_reward_slot) に注入 → Hebbian が survival と alignment.
    """

    def run_epoch(self, n_steps: int, log_every: int = 500):
        """Custom run that injects food-gain reward signal into agents' S[189]
        before each tick, so HebbianCoreBrain sees survival reward."""
        import numpy as np

        sensors_list = [self.world.get_sensors(i)
                        for i in range(self.world.n_agents)]
        # Track per-agent food eaten to compute delta
        prev_food = [int(self.world.agent_food_eaten[i])
                     for i in range(self.world.n_agents)]

        for step in range(n_steps):
            N = self.world.n_agents
            # Pad prev_food if children added
            while len(prev_food) < N:
                prev_food.append(0)

            # Inject sensors + survival reward signal
            for i in range(N):
                if self.world.agent_alive[i]:
                    cur_food = int(self.world.agent_food_eaten[i])
                    food_delta = max(0, cur_food - prev_food[i])
                    prev_food[i] = cur_food
                    with self.agents[i]._lock:
                        self.agents[i].S[0:16] = np.asarray(
                            sensors_list[i] if isinstance(sensors_list, list)
                            else sensors_list, dtype=np.float64)
                        # [H1b] Food eaten signal at S[189] (1.0 if ate, else decay)
                        prev_reward = float(self.agents[i].S[189])
                        # EMA decay + immediate signal on eat
                        new_signal = 0.7 * prev_reward + (1.0 if food_delta > 0 else 0.0)
                        self.agents[i].S[189] = new_signal

            # Tick × 3 (Phase 11 standard) + NaN/clip guard per tick
            for _ in range(3):
                for i in range(N):
                    if self.world.agent_alive[i]:
                        self.agents[i].tick_once()
                        # Safety: clip S, kill NaN/Inf (prevents feedback blowup)
                        S = self.agents[i].S
                        np.nan_to_num(S, copy=False, nan=0.0,
                                       posinf=3.0, neginf=-3.0)
                        np.clip(S, -3.0, 3.0, out=S)

            # Extract 4 actions (nav, speed, voice, jump)
            actions = []
            for i in range(N):
                if self.world.agent_alive[i]:
                    S = self.agents[i].read_state()
                    actions.append((
                        float(np.clip(S[16], 0.0, 1.0)),
                        float(np.clip(S[17], 0.0, 1.0)),
                        float(np.clip(S[18], 0.0, 1.0)),
                        float(np.clip(S[19], 0.0, 1.0)) if len(S) > 19 else 0.0,
                    ))
                else:
                    actions.append((0.5, 0.0, 0.0, 0.0))
            sensors_list, ate, done = self.world.step(actions)

            # Children may have been added
            if len(self.world.agents_external) > len(self.agents):
                self.agents = list(self.world.agents_external)
                for new_a in self.agents[len(actions):]:
                    new_a.reset_episode()

            if step % log_every == 0:
                self.trajectory.append({"step": step, **self.world.stats()})
            if self.world.stats()["n_alive"] == 0:
                break

        self.trajectory.append({"step": n_steps, **self.world.stats(), "final": True})

    def initialize_agents(self):
        """HebbianCoreBrain を使って agent を build。"""
        from kathara16_brain import PARAM_RANGES_16
        agents = []
        rng = np.random.default_rng(self.world_seed + 1000)
        pool = self.__class__.dna_pool

        for i in range(self.agents_init):
            a = build_fluctlight(
                self.shells_override, self.configs_dir,
                trained_dir=self.trained_dir,
                use_hebbian_core=True,  # ★ Hebbian 有効化
                use_3d_brain=True)
            # [H1b] Hebbian with EXTERNAL SURVIVAL REWARD aligned to food
            hb = a.shells[0]
            if hasattr(hb, "hebbian_lr"):
                hb.hebbian_lr = 0.003       # moderate learning rate
                hb.hebbian_decay = 0.01
                hb.reward_scale = 1.5
                hb.violation_penalty_weight = 2.0
                hb.w_clip = 0.5              # tighter clip for stability
                hb.noise_base = 0.05
                # ★ External survival reward: big weight on food-eating ★
                hb.external_reward_weight = 10.0  # food_delta * 10 drives learning
                # Reduce intrinsic rewards (they're noise relative to food)
                hb.salience_weight = 0.05
                hb.sensor_novelty_weight = 0.05
                hb.pred_error_weight = 0.5
            # Reduce inhibition gain to prevent feedback instability
            for shell in a.shells[1:]:
                if getattr(shell, "shell_sign", 1) == -1:
                    shell.gain *= 0.3
            # DNA 揺らぎ
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
            # Enable long-term lifetime learning
            if hasattr(a.shells[0], "w_adapt_carry_over"):
                a.shells[0].w_adapt_carry_over = True
            agents.append(a)
        self.agents = agents

        from tamashii.phase_11_cardinal_3d import _make_child_factory_ext
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


def run_phase16(n_universes=4, agents_per_u=6, epoch_steps=800,
                n_epochs=20, seed=42, device="cuda",
                output="phase_16_hebbian_hierarchical.json"):
    print("=" * 72, flush=True)
    print(f"  PHASE 16 (H1) - HebbianCoreBrain in 20-shell hierarchy (GPU)",
          flush=True)
    print(f"  {n_universes}u x {agents_per_u}a x {n_epochs}ep x "
          f"{epoch_steps}step", flush=True)
    print(f"  Brain: HebbianCoreBrain (runtime STDP) + 19 shells",
          flush=True)
    print("=" * 72, flush=True)

    shells = SHELLS_20
    configs_dir = os.path.join(THIS_DIR, "configs")

    universes = []
    for u_id in range(n_universes):
        wp = make_3d_universe_params(seed + u_id * 17)
        u = HebbianHierarchicalUniverse(
            u_id, wp, agents_per_u, seed + u_id * 101,
            shells, configs_dir, configs_dir)
        universes.append(u)
        print(f"  u{u_id}: gravity={wp['gravity']:+.3f} "
              f"vf={wp['vertical_food_frac']:.2f}", flush=True)

    # NOTE: GPU runner uses GPU batched core_brain kernel which BYPASSES
    # HebbianCoreBrain.step() — so Hebbian learning never fires.
    # For H1, we use CPU tick_once via Universe.run_epoch (Phase 11 base).
    # Slower but shell.step() is invoked correctly, Hebbian fires.
    print(f"  Using CPU tick_once (Hebbian compatible, ~5x slower than GPU)",
          flush=True)

    history = []
    t_all = time.time()

    for epoch in range(n_epochs):
        t_ep = time.time()
        # Run each universe via Phase 11 run_epoch (CPU, full shell.step chain)
        for u in universes:
            u.run_epoch(epoch_steps, log_every=epoch_steps)

        # Per-epoch snapshot with Hebbian stats
        snaps = []
        for u in universes:
            s = u.world.stats()
            food = sum(u.world.agent_food_eaten[i]
                        for i in range(u.world.n_agents))
            # Aggregate Hebbian stats across alive agents
            w_norms = []
            n_changed = []
            for i in range(u.world.n_agents):
                if u.world.agent_alive[i]:
                    shell = u.world.agents_external[i].shells[0]
                    if hasattr(shell, "hebbian_stats"):
                        hs = shell.hebbian_stats()
                        w_norms.append(hs["w_adapt_mean_abs"])
                        n_changed.append(hs["n_changed_edges"])
            snaps.append({
                "u_id": u.id, "quality": float(u.quality()),
                "food_total": int(food),
                "n_alive": s["n_alive"],
                "n_births": s["n_births"],
                "n_deaths": s["n_deaths"],
                "max_gen": s["max_generation"],
                "hebbian_w_norm_mean": float(np.mean(w_norms)) if w_norms else 0.0,
                "hebbian_w_norm_max": float(np.max(w_norms)) if w_norms else 0.0,
                "hebbian_n_changed_edges": float(np.mean(n_changed)) if n_changed else 0.0,
            })
        ranked = sorted(snaps, key=lambda x: -x["quality"])
        history.append({"epoch": epoch, "snapshots": snaps,
                         "best": ranked[0]["u_id"],
                         "elapsed_ep_s": time.time() - t_ep})

        if epoch % 2 == 0 or epoch == n_epochs - 1:
            b = ranked[0]
            print(f"\n  ep{epoch} ({time.time() - t_ep:.1f}s) "
                  f"best u{b['u_id']} q={b['quality']:.2f} food={b['food_total']} "
                  f"b/d={b['n_births']}/{b['n_deaths']} alive={b['n_alive']}",
                  flush=True)
            print(f"    Hebbian: w_norm={b['hebbian_w_norm_mean']:.4f} "
                  f"max={b['hebbian_w_norm_max']:.4f} "
                  f"changed_edges={b['hebbian_n_changed_edges']:.1f}/48",
                  flush=True)

        # Meta-evolve (no runner rebuild — CPU path)
        if epoch < n_epochs - 1:
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

    total = time.time() - t_all
    print(f"\n  DONE in {total:.0f}s ({total/60:.1f} min)", flush=True)

    # Final comparison to Phase 14
    print(f"\n  === COMPARISON vs Phase 14 (static CoreBrain) ===", flush=True)
    print(f"  Phase 14 ep19: food/agent-step = 0.17, births/epoch = 76",
          flush=True)
    final = history[-1]["snapshots"]
    total_food = sum(s["food_total"] for s in final)
    total_births = sum(s["n_births"] for s in final)
    total_alive = sum(s["n_alive"] for s in final)
    # Last-epoch delta rate
    if len(history) > 1:
        prev_food = sum(s["food_total"] for s in history[-2]["snapshots"])
        delta_food = total_food - prev_food
        agent_step_last_ep = epoch_steps * n_universes * agents_per_u
        rate = delta_food / agent_step_last_ep
        print(f"  Phase 16 last ep: food rate = {rate:.4f}/agent-step "
              f"({rate/0.17*100:.1f}% of Phase 14)", flush=True)
    print(f"  Phase 16 ep{n_epochs-1}: total_food={total_food} "
          f"births={total_births} alive={total_alive}", flush=True)
    # Hebbian learning evidence
    final_w = np.mean([s["hebbian_w_norm_mean"] for s in final])
    print(f"  Hebbian learning: w_adapt mean abs = {final_w:.4f} "
          f"(0 = no learning, >0 = agents DID learn within their life)",
          flush=True)

    out = {
        "config": {"n_universes": n_universes,
                    "agents_per_universe": agents_per_u,
                    "epoch_steps": epoch_steps, "n_epochs": n_epochs,
                    "seed": seed, "device": device,
                    "architecture": "20-shell + HebbianCoreBrain"},
        "total_elapsed_s": round(total, 1),
        "history": history,
        "final_summary": {
            "total_food_final": total_food,
            "total_births_final": total_births,
            "total_alive_final": total_alive,
            "hebbian_w_adapt_mean": final_w,
        },
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--agents",      type=int, default=6)
    ap.add_argument("--epoch_steps", type=int, default=800)
    ap.add_argument("--n_epochs",    type=int, default=20)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--device",      type=str, default="cuda")
    ap.add_argument("--output",      type=str,
                     default="phase_16_hebbian_hierarchical.json")
    args = ap.parse_args()
    run_phase16(
        n_universes=args.n_universes,
        agents_per_u=args.agents,
        epoch_steps=args.epoch_steps,
        n_epochs=args.n_epochs,
        seed=args.seed,
        device=args.device,
        output=args.output,
    )


if __name__ == "__main__":
    main()
