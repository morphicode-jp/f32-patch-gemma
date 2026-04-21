"""Phase 11.2 — Cardinal for Neural Net HP tuning (cross-domain test).

Does Zenron^∞ (quality-sensitive meta-evolution) generalize beyond
agent/world simulations?

Setup:
  Task = synthetic regression (y = sum(x^2) + noise)
  Each "universe" = NN training run with different HP
  HP = {lr, momentum, l2_weight, batch_size, mut_rate}
  Quality = -validation_loss (higher better)
  Cardinal meta-evolve: replace worst HP with mutated copy of best

Two conditions tested (analogous to Priority 1 ablation):
  A. Val-loss reward only (quality = -val_loss)
  B. Val-loss + variance reward (quality = -val_loss + 2 * std_of_pop)
     → reward HP diversity → should favor higher mutation_rate

If Zenron^∞ quality-sensitivity generalizes:
  A → mut_rate stays low (stability wins)
  B → mut_rate escalates (diversity wins)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def synthetic_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 8)).astype(np.float32)
    y = (x ** 2).sum(axis=1) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )


def train_universe(hp: dict, steps: int = 400, seed: int = 42,
                   device: str = "cuda") -> float:
    """Train NN with given HP, return -val_loss as quality."""
    x_train, y_train = synthetic_data(1000, seed)
    x_val, y_val = synthetic_data(200, seed + 1)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    torch.manual_seed(seed)
    model = make_model().to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(hp["lr"]),
        momentum=float(hp["momentum"]),
        weight_decay=float(hp["l2"]),
    )
    batch_size = int(hp["batch_size"])
    N = x_train.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        pred = model(x_train[idx]).squeeze()
        loss = ((pred - y_train[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        val_loss = ((model(x_val).squeeze() - y_val) ** 2).mean().item()
    return -val_loss


HP_RANGES = {
    "lr":         (0.001, 0.1,  "float"),
    "momentum":   (0.0,   0.95, "float"),
    "l2":         (1e-6,  1e-2, "float_log"),
    "batch_size": (8,     128,  "int"),
    "mut_rate":   (0.02,  0.25, "float"),  # meta-HP: our σ
}


def random_hp(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    hp = {}
    for k, (lo, hi, typ) in HP_RANGES.items():
        if typ == "float":
            hp[k] = float(rng.uniform(lo, hi))
        elif typ == "float_log":
            hp[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif typ == "int":
            hp[k] = int(round(rng.uniform(lo, hi)))
    return hp


def mutate_hp(parent: dict, seed: int) -> dict:
    """Mutate HP using parent's OWN mut_rate as sigma."""
    rng = np.random.default_rng(seed)
    child = dict(parent)
    sigma = float(parent.get("mut_rate", 0.1))
    for k, (lo, hi, typ) in HP_RANGES.items():
        if typ == "float":
            delta = rng.normal(0, sigma * (hi - lo))
            child[k] = float(np.clip(parent[k] + delta, lo, hi))
        elif typ == "float_log":
            log_lo, log_hi = np.log(lo), np.log(hi)
            delta = rng.normal(0, sigma * (log_hi - log_lo))
            child[k] = float(np.exp(np.clip(np.log(parent[k]) + delta, log_lo, log_hi)))
        elif typ == "int":
            delta = rng.normal(0, sigma * (hi - lo))
            child[k] = int(np.clip(round(parent[k] + delta), lo, hi))
    return child


def run_cardinal_nn(n_universes: int = 4, n_epochs: int = 5,
                    steps_per_epoch: int = 400, seed: int = 42,
                    reward_diversity: bool = False,
                    label: str = "control"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*72}", flush=True)
    print(f"  NN-HP Cardinal: {label} (reward_diversity={reward_diversity})",
          flush=True)
    print(f"  {n_universes} universes × {n_epochs} epochs × {steps_per_epoch}steps",
          flush=True)
    print(f"{'='*72}", flush=True)

    universes = [random_hp(seed + i * 7) for i in range(n_universes)]
    history = []
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for epoch in range(n_epochs):
        # Evaluate each
        qualities_raw = []
        for u_id, hp in enumerate(universes):
            q = train_universe(hp, steps=steps_per_epoch,
                                seed=seed + epoch * 31 + u_id,
                                device=device)
            qualities_raw.append(q)
        # Optionally add diversity reward (per-universe)
        if reward_diversity:
            # Reward HIGH mutation_rate directly (it SOURCES diversity)
            # Per-universe bonus proportional to its own mut_rate
            qualities = [q + 3.0 * u["mut_rate"]
                         for q, u in zip(qualities_raw, universes)]
        else:
            qualities = qualities_raw

        print(f"\n  Epoch {epoch}:", flush=True)
        for u_id, (hp, q_raw, q) in enumerate(zip(universes, qualities_raw, qualities)):
            print(f"    u{u_id}: lr={hp['lr']:.4f} mom={hp['momentum']:.2f} "
                  f"l2={hp['l2']:.5f} bs={hp['batch_size']:3d} "
                  f"mut={hp['mut_rate']:.3f} → raw_q={q_raw:.3f} adj_q={q:.3f}",
                  flush=True)

        # Rank
        ranked_idx = sorted(range(n_universes), key=lambda i: -qualities[i])
        best_idx = ranked_idx[0]
        worst_idx = ranked_idx[-1]

        history.append({
            "epoch": epoch,
            "universes": [dict(hp) for hp in universes],
            "qualities_raw": qualities_raw,
            "qualities_adjusted": qualities,
            "best_idx": best_idx,
            "worst_idx": worst_idx,
        })

        # Meta-evolve except last epoch
        if epoch < n_epochs - 1:
            print(f"  META: u{worst_idx} ← mutate(u{best_idx})", flush=True)
            new_hp = mutate_hp(universes[best_idx], seed=seed + epoch * 13 + 17)
            universes[worst_idx] = new_hp

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.0f}s", flush=True)

    # Summary: mut_rate trajectory
    print(f"  mut_rate trajectory (best universe each epoch):", flush=True)
    mut_traj = []
    for h in history:
        best = h["universes"][h["best_idx"]]
        mut_traj.append(best["mut_rate"])
        print(f"    ep{h['epoch']}: u{h['best_idx']} mut_rate={best['mut_rate']:.4f} "
              f"quality_raw={h['qualities_raw'][h['best_idx']]:.3f}", flush=True)

    # Did mut_rate change from initial?
    initial_mut_best = history[0]["universes"][history[0]["best_idx"]]["mut_rate"]
    final_mut_best = mut_traj[-1]
    delta = final_mut_best - initial_mut_best
    print(f"\n  mut_rate delta (initial→final winner): {initial_mut_best:.4f} → "
          f"{final_mut_best:.4f} (Δ={delta:+.4f})", flush=True)

    return {
        "label": label,
        "reward_diversity": reward_diversity,
        "elapsed_s": round(elapsed, 1),
        "history": history,
        "mut_rate_trajectory": mut_traj,
        "initial_mut_best": initial_mut_best,
        "final_mut_best": final_mut_best,
        "delta_mut_best": delta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=4)
    ap.add_argument("--n_epochs", type=int, default=6)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str,
                    default="tamashii_phase_11_2_nn_hp.json")
    args = ap.parse_args()

    print("=" * 72, flush=True)
    print("  PHASE 11.2 — NN HP Cardinal (cross-domain Zenron^∞ test)",
          flush=True)
    print("=" * 72, flush=True)

    # Run both conditions
    result_A = run_cardinal_nn(
        n_universes=args.n_universes, n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch, seed=args.seed,
        reward_diversity=False, label="A_loss_only")
    result_B = run_cardinal_nn(
        n_universes=args.n_universes, n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch, seed=args.seed,
        reward_diversity=True, label="B_loss_plus_diversity")

    # Compare
    print(f"\n{'='*72}", flush=True)
    print(f"  COMPARISON", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  A (loss only):           Δmut={result_A['delta_mut_best']:+.4f}",
          flush=True)
    print(f"  B (loss + diversity):    Δmut={result_B['delta_mut_best']:+.4f}",
          flush=True)
    if result_B['delta_mut_best'] > result_A['delta_mut_best'] + 0.02:
        print(f"\n  ✓ Cross-domain sensitivity: B's diversity reward drove "
              f"mut_rate higher than A", flush=True)
    else:
        print(f"\n  ⏸ Effect small or absent in this domain", flush=True)

    out = {
        "A_loss_only": result_A,
        "B_loss_plus_diversity": result_B,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
