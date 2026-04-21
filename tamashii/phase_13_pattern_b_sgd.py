"""Phase 13 — Pattern B on SGD (HP-sensitive optimizer test).

Context: Pattern B on mimir was null (mimir defaults already robust).
Hypothesis: on raw SGD (HP-sensitive), Cardinal can BEAT defaults.

Setup:
  Task = synthetic regression (y = sum(x^2) + noise, 8D)
  Optimizer = SGD with manually-chosen defaults (lr=0.01, mom=0.9, wd=1e-4, bs=32)
  HP search space = {lr, momentum, weight_decay, batch_size}
  Quality = -val_loss (higher better)

3 conditions, same eval budget:
  A. SGD defaults (reference, 1 run × N seeds)
  B. Random search over HP_RANGES (baseline, matched budget)
  C. Cardinal Pattern B (our target)

Comparison: final best val_loss on held-out test set.

Expected:
  If raw SGD is HP-sensitive → C > A (Cardinal finds better HP than hand-tuned)
  If null → C ≈ A (defaults were already near-optimal, same as mimir)
  If Cardinal overfits → C > random_search B marginally

Key difference from Phase 11.2:
  11.2 asked "does σ escalate with diversity reward?" (Level-1 sensitivity)
  13 asks "does Pattern B auto-tune SGD better than defaults?" (practical value)
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


def train_sgd(hp: dict, steps: int, seed: int, device: str,
              x_train, y_train, x_val, y_val) -> float:
    """Train NN with SGD using given HP, return -val_loss."""
    torch.manual_seed(seed)
    model = make_model().to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(hp["lr"]),
        momentum=float(hp["momentum"]),
        weight_decay=float(hp["weight_decay"]),
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
        if not torch.isfinite(loss):
            return -1e6
    with torch.no_grad():
        val_loss = ((model(x_val).squeeze() - y_val) ** 2).mean().item()
    if not np.isfinite(val_loss):
        return -1e6
    return -val_loss


HP_RANGES = {
    "lr":           (0.0001, 0.3,   "float_log"),
    "momentum":     (0.0,    0.99,  "float"),
    "weight_decay": (1e-6,   1e-2,  "float_log"),
    "batch_size":   (8,      128,   "int"),
}


DEFAULT_HP = {
    "lr":           0.01,
    "momentum":     0.9,
    "weight_decay": 1e-4,
    "batch_size":   32,
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


def mutate_hp(parent: dict, sigma: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    child = dict(parent)
    for k, (lo, hi, typ) in HP_RANGES.items():
        if rng.random() < 0.5:
            continue
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


def evaluate_on_test(hp: dict, steps: int, device: str,
                      test_seeds: list[int]) -> dict:
    """Train from scratch, eval on test set. Multiple seeds for stats."""
    val_losses = []
    test_losses = []
    for s in test_seeds:
        x_tr, y_tr = synthetic_data(1000, s)
        x_va, y_va = synthetic_data(200, s + 1)
        x_te, y_te = synthetic_data(500, s + 7777)
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        x_va, y_va = x_va.to(device), y_va.to(device)
        x_te, y_te = x_te.to(device), y_te.to(device)

        torch.manual_seed(s)
        model = make_model().to(device)
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(hp["lr"]),
            momentum=float(hp["momentum"]),
            weight_decay=float(hp["weight_decay"]),
        )
        bs = int(hp["batch_size"])
        N = x_tr.shape[0]
        for step in range(steps):
            idx = torch.randint(0, N, (bs,), device=device)
            pred = model(x_tr[idx]).squeeze()
            loss = ((pred - y_tr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not torch.isfinite(loss):
                break
        with torch.no_grad():
            va = ((model(x_va).squeeze() - y_va) ** 2).mean().item()
            te = ((model(x_te).squeeze() - y_te) ** 2).mean().item()
        if not np.isfinite(va): va = 1e6
        if not np.isfinite(te): te = 1e6
        val_losses.append(va)
        test_losses.append(te)
    return {
        "val_mean": float(np.mean(val_losses)),
        "val_std":  float(np.std(val_losses)),
        "test_mean": float(np.mean(test_losses)),
        "test_std":  float(np.std(test_losses)),
        "val_losses": val_losses,
        "test_losses": test_losses,
    }


def run_cardinal_pattern_b(n_universes: int, n_epochs: int,
                             steps_per_epoch: int, sigma: float,
                             seed: int, device: str):
    print(f"\n  CARDINAL PATTERN B: {n_universes}u × {n_epochs}ep × {steps_per_epoch}step",
          flush=True)

    # Shared train/val data for within-epoch training (fixed across universes to ensure fair ranking)
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    universes = [random_hp(seed + i * 7) for i in range(n_universes)]
    # Universe 0 = defaults (anchor)
    universes[0] = dict(DEFAULT_HP)

    history = []
    best_hp_ever = None
    best_quality_ever = -np.inf
    total_evals = 0

    for epoch in range(n_epochs):
        qs = []
        for u_id, hp in enumerate(universes):
            q = train_sgd(hp, steps_per_epoch, seed + epoch * 31 + u_id,
                           device, x_tr, y_tr, x_va, y_va)
            qs.append(q)
            total_evals += 1
            if q > best_quality_ever:
                best_quality_ever = q
                best_hp_ever = dict(hp)

        # Rank
        ranked = sorted(range(n_universes), key=lambda i: -qs[i])
        best_idx = ranked[0]
        worst_idx = ranked[-1]

        print(f"  ep{epoch}: best u{best_idx} lr={universes[best_idx]['lr']:.4f} "
              f"mom={universes[best_idx]['momentum']:.2f} q={qs[best_idx]:.4f}",
              flush=True)

        history.append({
            "epoch": epoch,
            "universes": [dict(hp) for hp in universes],
            "qualities": qs,
            "best_idx": best_idx,
        })

        # Meta-evolve: replace bottom 2 with mutants of top 1
        if epoch < n_epochs - 1:
            for bad_idx in ranked[-2:]:
                universes[bad_idx] = mutate_hp(
                    universes[best_idx], sigma,
                    seed + epoch * 53 + bad_idx)

    print(f"  Cardinal total evals: {total_evals}", flush=True)
    return best_hp_ever, best_quality_ever, history, total_evals


def run_random_search(n_evals: int, steps_per_epoch: int, seed: int, device: str):
    print(f"\n  RANDOM SEARCH: {n_evals} evals", flush=True)
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    best_hp = None
    best_q = -np.inf
    for i in range(n_evals):
        hp = random_hp(seed + i * 137)
        q = train_sgd(hp, steps_per_epoch, seed + i * 31, device,
                       x_tr, y_tr, x_va, y_va)
        if q > best_q:
            best_q = q
            best_hp = dict(hp)
    print(f"  Random best: lr={best_hp['lr']:.4f} q={best_q:.4f}", flush=True)
    return best_hp, best_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=5)
    ap.add_argument("--n_epochs",    type=int, default=5)
    ap.add_argument("--steps",       type=int, default=400)
    ap.add_argument("--sigma",       type=float, default=0.25)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--n_test_seeds", type=int, default=5)
    ap.add_argument("--output", type=str, default="phase_13_pattern_b_sgd.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72, flush=True)
    print(f"  PHASE 13: Pattern B on SGD (HP-sensitive test)", flush=True)
    print(f"  Device: {device}", flush=True)
    print("=" * 72, flush=True)

    t_all = time.time()
    test_seeds = [args.seed + i * 997 for i in range(args.n_test_seeds)]

    # --- A. Defaults ---
    print(f"\n{'='*72}\n  A. DEFAULTS\n{'='*72}", flush=True)
    print(f"  HP: {DEFAULT_HP}", flush=True)
    t0 = time.time()
    eval_defaults = evaluate_on_test(DEFAULT_HP, args.steps, device, test_seeds)
    t_A = time.time() - t0
    print(f"  Test: mean={eval_defaults['test_mean']:.4f} ± {eval_defaults['test_std']:.4f}",
          flush=True)
    print(f"  Val:  mean={eval_defaults['val_mean']:.4f} ± {eval_defaults['val_std']:.4f}",
          flush=True)

    # --- B. Cardinal Pattern B ---
    print(f"\n{'='*72}\n  B. CARDINAL PATTERN B\n{'='*72}", flush=True)
    t0 = time.time()
    card_hp, card_q, card_hist, card_evals = run_cardinal_pattern_b(
        args.n_universes, args.n_epochs, args.steps, args.sigma,
        args.seed, device)
    t_B_search = time.time() - t0
    print(f"  Cardinal discovered: {card_hp}", flush=True)
    print(f"  Search time: {t_B_search:.1f}s ({card_evals} evals)", flush=True)

    # Eval Cardinal's discovered HP on test seeds
    t0 = time.time()
    eval_cardinal = evaluate_on_test(card_hp, args.steps, device, test_seeds)
    t_B_eval = time.time() - t0
    print(f"  Test: mean={eval_cardinal['test_mean']:.4f} ± {eval_cardinal['test_std']:.4f}",
          flush=True)

    # --- C. Random search (matched budget) ---
    print(f"\n{'='*72}\n  C. RANDOM SEARCH (budget={card_evals})\n{'='*72}",
          flush=True)
    t0 = time.time()
    rand_hp, rand_q = run_random_search(card_evals, args.steps, args.seed, device)
    t_C_search = time.time() - t0
    print(f"  Random discovered: {rand_hp}", flush=True)
    eval_random = evaluate_on_test(rand_hp, args.steps, device, test_seeds)
    t_C_eval = time.time() - t0
    print(f"  Test: mean={eval_random['test_mean']:.4f} ± {eval_random['test_std']:.4f}",
          flush=True)

    # --- Compare ---
    total = time.time() - t_all
    print(f"\n{'='*72}\n  RESULTS (test_loss, lower=better)\n{'='*72}", flush=True)
    print(f"  {'method':25s} {'test_mean':>10s} {'test_std':>10s} {'evals':>6s}",
          flush=True)
    print(f"  {'A. Defaults':25s} {eval_defaults['test_mean']:>10.4f} "
          f"{eval_defaults['test_std']:>10.4f} {'-':>6s}", flush=True)
    print(f"  {'B. Cardinal Pattern B':25s} {eval_cardinal['test_mean']:>10.4f} "
          f"{eval_cardinal['test_std']:>10.4f} {card_evals:>6d}", flush=True)
    print(f"  {'C. Random (matched)':25s} {eval_random['test_mean']:>10.4f} "
          f"{eval_random['test_std']:>10.4f} {card_evals:>6d}", flush=True)

    improve_B_over_A = eval_defaults['test_mean'] - eval_cardinal['test_mean']
    improve_B_over_C = eval_random['test_mean'] - eval_cardinal['test_mean']
    print(f"\n  Cardinal vs Defaults: Δ = {improve_B_over_A:+.4f} "
          f"({'B WINS' if improve_B_over_A > 0 else 'A WINS'})", flush=True)
    print(f"  Cardinal vs Random:   Δ = {improve_B_over_C:+.4f} "
          f"({'B WINS' if improve_B_over_C > 0 else 'C WINS'})", flush=True)

    print(f"\n  Total elapsed: {total:.0f}s", flush=True)

    out = {
        "device": device,
        "total_elapsed_s": round(total, 1),
        "config": vars(args),
        "defaults_hp": DEFAULT_HP,
        "defaults_eval": eval_defaults,
        "cardinal_hp": card_hp,
        "cardinal_eval": eval_cardinal,
        "cardinal_evals": card_evals,
        "cardinal_history": card_hist,
        "random_hp": rand_hp,
        "random_eval": eval_random,
        "improvement_B_vs_A": improve_B_over_A,
        "improvement_B_vs_C": improve_B_over_C,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
