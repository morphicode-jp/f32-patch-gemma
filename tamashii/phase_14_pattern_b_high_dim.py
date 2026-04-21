"""Phase 14 — Pattern B on high-dim HP space (Random Search 弱点領域).

Phase 13 結論: 4D HP 空間では Cardinal ≈ Random Search。
仮説: 高次元 (10+ HPs) で HP 間 interaction が増えると、Random が弱くなり
     Cardinal (structured search) が有利になる。

Setup:
  Task = synthetic regression (y = Σx² + noise, 8D)
  Optimizer = Adam + LR schedule + 構造 HP
  12 HPs tuned:
    1. lr (log)
    2. beta1 (0.8-0.999)
    3. beta2 (0.9-0.9999)
    4. epsilon (log, 1e-10 to 1e-4)
    5. weight_decay (log, 1e-6 to 1e-2)
    6. batch_size (int, 8-128)
    7. warmup_steps (int, 0-200)
    8. lr_decay_gamma (0.9-1.0)
    9. gradient_clip (log, 1e-2 to 1e2)
    10. dropout (0.0-0.5)
    11. hidden_dim (int, 16-128)
    12. n_layers (int, 2-5)

3 conditions, same eval budget:
  A. Typical manual defaults
  B. Cardinal Pattern B (same 5u × 5ep = 25 evals)
  C. Random Search (25 evals)

Expected:
  12D interactions → Random Search 弱体化
  Cardinal's top-1-variant strategy できれば明確に勝つ
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def synthetic_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 8)).astype(np.float32)
    y = (x ** 2).sum(axis=1) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_model(hidden_dim: int, n_layers: int, dropout: float):
    layers = [torch.nn.Linear(8, hidden_dim), torch.nn.ReLU(),
              torch.nn.Dropout(dropout)]
    for _ in range(n_layers - 2):
        layers += [torch.nn.Linear(hidden_dim, hidden_dim),
                   torch.nn.ReLU(),
                   torch.nn.Dropout(dropout)]
    layers += [torch.nn.Linear(hidden_dim, 1)]
    return torch.nn.Sequential(*layers)


def train_adam(hp: dict, steps: int, seed: int, device: str,
                x_train, y_train, x_val, y_val) -> float:
    torch.manual_seed(seed)
    model = make_model(
        hidden_dim=int(hp["hidden_dim"]),
        n_layers=int(hp["n_layers"]),
        dropout=float(hp["dropout"]),
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(hp["lr"]),
        betas=(float(hp["beta1"]), float(hp["beta2"])),
        eps=float(hp["epsilon"]),
        weight_decay=float(hp["weight_decay"]),
    )
    bs = int(hp["batch_size"])
    warmup = int(hp["warmup_steps"])
    decay_gamma = float(hp["lr_decay_gamma"])
    clip = float(hp["gradient_clip"])

    N = x_train.shape[0]
    base_lr = float(hp["lr"])
    for step in range(steps):
        # LR schedule: warmup then exponential decay
        if step < warmup:
            lr = base_lr * (step + 1) / max(warmup, 1)
        else:
            lr = base_lr * (decay_gamma ** ((step - warmup) // 100))
        for pg in opt.param_groups:
            pg["lr"] = lr

        idx = torch.randint(0, N, (bs,), device=device)
        model.train()
        pred = model(x_train[idx]).squeeze()
        loss = ((pred - y_train[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if not torch.isfinite(loss):
            return -1e6

    model.eval()
    with torch.no_grad():
        val_loss = ((model(x_val).squeeze() - y_val) ** 2).mean().item()
    if not np.isfinite(val_loss):
        return -1e6
    return -val_loss


HP_RANGES = {
    "lr":              (1e-5, 0.3,    "float_log"),
    "beta1":           (0.5,  0.9999, "float"),
    "beta2":           (0.9,  0.99999,"float"),
    "epsilon":         (1e-10, 1e-4,  "float_log"),
    "weight_decay":    (1e-7, 1e-2,   "float_log"),
    "batch_size":      (8,    128,    "int"),
    "warmup_steps":    (0,    200,    "int"),
    "lr_decay_gamma":  (0.5,  1.0,    "float"),
    "gradient_clip":   (1e-2, 1e2,    "float_log"),
    "dropout":         (0.0,  0.5,    "float"),
    "hidden_dim":      (16,   128,    "int"),
    "n_layers":        (2,    5,      "int"),
}


DEFAULT_HP = {
    "lr":             1e-3,
    "beta1":          0.9,
    "beta2":          0.999,
    "epsilon":        1e-8,
    "weight_decay":   1e-4,
    "batch_size":     32,
    "warmup_steps":   0,
    "lr_decay_gamma": 1.0,
    "gradient_clip":  1.0,
    "dropout":        0.1,
    "hidden_dim":     32,
    "n_layers":       3,
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


def evaluate_on_test(hp: dict, steps: int, device: str, test_seeds: list[int]) -> dict:
    test_losses = []
    val_losses = []
    for s in test_seeds:
        x_tr, y_tr = synthetic_data(1000, s)
        x_va, y_va = synthetic_data(200, s + 1)
        x_te, y_te = synthetic_data(500, s + 7777)
        x_tr, y_tr = x_tr.to(device), y_tr.to(device)
        x_va, y_va = x_va.to(device), y_va.to(device)
        x_te, y_te = x_te.to(device), y_te.to(device)

        # Fresh train
        neg_val = train_adam(hp, steps, s, device, x_tr, y_tr, x_va, y_va)
        va = -neg_val if neg_val > -1e5 else 1e6
        val_losses.append(va)
        # Retrain for test eval (use same seed, eval on test)
        # Cheaper: reuse model - but we don't return it. Retrain once more.
        # To keep code simple and honest, we train again and eval on test.
        torch.manual_seed(s)
        model = make_model(int(hp["hidden_dim"]), int(hp["n_layers"]),
                            float(hp["dropout"])).to(device)
        opt = torch.optim.Adam(
            model.parameters(),
            lr=float(hp["lr"]),
            betas=(float(hp["beta1"]), float(hp["beta2"])),
            eps=float(hp["epsilon"]),
            weight_decay=float(hp["weight_decay"]),
        )
        bs = int(hp["batch_size"])
        warmup = int(hp["warmup_steps"])
        decay_gamma = float(hp["lr_decay_gamma"])
        clip = float(hp["gradient_clip"])
        base_lr = float(hp["lr"])
        N = x_tr.shape[0]
        failed = False
        for step in range(steps):
            if step < warmup:
                lr = base_lr * (step + 1) / max(warmup, 1)
            else:
                lr = base_lr * (decay_gamma ** ((step - warmup) // 100))
            for pg in opt.param_groups:
                pg["lr"] = lr
            idx = torch.randint(0, N, (bs,), device=device)
            model.train()
            pred = model(x_tr[idx]).squeeze()
            loss = ((pred - y_tr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if not torch.isfinite(loss):
                failed = True
                break
        model.eval()
        if failed:
            te = 1e6
        else:
            with torch.no_grad():
                te = ((model(x_te).squeeze() - y_te) ** 2).mean().item()
            if not np.isfinite(te):
                te = 1e6
        test_losses.append(te)

    return {
        "val_mean": float(np.mean(val_losses)),
        "val_std":  float(np.std(val_losses)),
        "test_mean": float(np.mean(test_losses)),
        "test_std":  float(np.std(test_losses)),
        "val_losses": val_losses,
        "test_losses": test_losses,
    }


def run_cardinal(n_universes: int, n_epochs: int, steps: int, sigma: float,
                  seed: int, device: str):
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    universes = [random_hp(seed + i * 7) for i in range(n_universes)]
    universes[0] = dict(DEFAULT_HP)

    best_hp = None
    best_q = -np.inf
    total_evals = 0
    history = []

    for epoch in range(n_epochs):
        qs = []
        for u_id, hp in enumerate(universes):
            q = train_adam(hp, steps, seed + epoch * 31 + u_id, device,
                            x_tr, y_tr, x_va, y_va)
            qs.append(q)
            total_evals += 1
            if q > best_q:
                best_q = q
                best_hp = dict(hp)

        ranked = sorted(range(n_universes), key=lambda i: -qs[i])
        print(f"  ep{epoch}: best u{ranked[0]} q={qs[ranked[0]]:.4f} "
              f"(worst q={qs[ranked[-1]]:.4f})", flush=True)
        history.append({"epoch": epoch, "qualities": qs,
                         "universes": [dict(hp) for hp in universes]})

        if epoch < n_epochs - 1:
            for bad_idx in ranked[-2:]:
                universes[bad_idx] = mutate_hp(
                    universes[ranked[0]], sigma,
                    seed + epoch * 53 + bad_idx)

    return best_hp, best_q, total_evals, history


def run_random(n_evals: int, steps: int, seed: int, device: str):
    x_tr, y_tr = synthetic_data(1000, seed)
    x_va, y_va = synthetic_data(200, seed + 1)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)

    best_hp = None
    best_q = -np.inf
    for i in range(n_evals):
        hp = random_hp(seed + i * 137)
        q = train_adam(hp, steps, seed + i * 31, device,
                        x_tr, y_tr, x_va, y_va)
        if q > best_q:
            best_q = q
            best_hp = dict(hp)
    return best_hp, best_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_universes", type=int, default=5)
    ap.add_argument("--n_epochs",    type=int, default=5)
    ap.add_argument("--steps",       type=int, default=400)
    ap.add_argument("--sigma",       type=float, default=0.25)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--n_test_seeds", type=int, default=5)
    ap.add_argument("--output", type=str, default="phase_14_pattern_b_high_dim.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72, flush=True)
    print(f"  PHASE 14: Pattern B on HIGH-DIM HP (12 HPs, interactions)", flush=True)
    print(f"  Device: {device}", flush=True)
    print("=" * 72, flush=True)

    test_seeds = [args.seed + i * 997 for i in range(args.n_test_seeds)]
    t_all = time.time()

    # A. Defaults
    print(f"\n  A. DEFAULTS", flush=True)
    print(f"  HP: lr={DEFAULT_HP['lr']} b1={DEFAULT_HP['beta1']} "
          f"bs={DEFAULT_HP['batch_size']} h={DEFAULT_HP['hidden_dim']} "
          f"L={DEFAULT_HP['n_layers']}", flush=True)
    t0 = time.time()
    ev_A = evaluate_on_test(DEFAULT_HP, args.steps, device, test_seeds)
    print(f"  Test: mean={ev_A['test_mean']:.4f} ± {ev_A['test_std']:.4f}",
          flush=True)

    # B. Cardinal
    print(f"\n  B. CARDINAL PATTERN B", flush=True)
    t0 = time.time()
    card_hp, card_q, card_evals, card_hist = run_cardinal(
        args.n_universes, args.n_epochs, args.steps, args.sigma,
        args.seed, device)
    print(f"  Cardinal best: lr={card_hp['lr']:.5f} bs={card_hp['batch_size']} "
          f"h={card_hp['hidden_dim']} L={card_hp['n_layers']}", flush=True)
    ev_B = evaluate_on_test(card_hp, args.steps, device, test_seeds)
    print(f"  Test: mean={ev_B['test_mean']:.4f} ± {ev_B['test_std']:.4f}",
          flush=True)

    # C. Random
    print(f"\n  C. RANDOM SEARCH ({card_evals} evals)", flush=True)
    rand_hp, rand_q = run_random(card_evals, args.steps, args.seed, device)
    print(f"  Random best: lr={rand_hp['lr']:.5f} bs={rand_hp['batch_size']} "
          f"h={rand_hp['hidden_dim']} L={rand_hp['n_layers']}", flush=True)
    ev_C = evaluate_on_test(rand_hp, args.steps, device, test_seeds)
    print(f"  Test: mean={ev_C['test_mean']:.4f} ± {ev_C['test_std']:.4f}",
          flush=True)

    # Compare
    total = time.time() - t_all
    print(f"\n{'='*72}\n  RESULTS (test_loss, lower=better)\n{'='*72}", flush=True)
    print(f"  {'method':25s} {'test_mean':>10s} {'test_std':>10s} {'evals':>6s}",
          flush=True)
    print(f"  {'A. Defaults':25s} {ev_A['test_mean']:>10.4f} "
          f"{ev_A['test_std']:>10.4f} {'-':>6s}", flush=True)
    print(f"  {'B. Cardinal Pattern B':25s} {ev_B['test_mean']:>10.4f} "
          f"{ev_B['test_std']:>10.4f} {card_evals:>6d}", flush=True)
    print(f"  {'C. Random (matched)':25s} {ev_C['test_mean']:>10.4f} "
          f"{ev_C['test_std']:>10.4f} {card_evals:>6d}", flush=True)

    d_BA = ev_A['test_mean'] - ev_B['test_mean']
    d_BC = ev_C['test_mean'] - ev_B['test_mean']
    print(f"\n  Cardinal vs Defaults: Δ = {d_BA:+.4f} "
          f"({'B WINS' if d_BA > 0 else 'A WINS'})", flush=True)
    print(f"  Cardinal vs Random:   Δ = {d_BC:+.4f} "
          f"({'B WINS' if d_BC > 0 else 'C WINS'})", flush=True)
    print(f"\n  Total elapsed: {total:.0f}s", flush=True)

    out = {
        "config": vars(args),
        "total_elapsed_s": round(total, 1),
        "defaults_hp": DEFAULT_HP,
        "defaults_eval": ev_A,
        "cardinal_hp": card_hp,
        "cardinal_eval": ev_B,
        "cardinal_evals": card_evals,
        "cardinal_history": card_hist,
        "random_hp": rand_hp,
        "random_eval": ev_C,
        "improvement_B_vs_A": d_BA,
        "improvement_B_vs_C": d_BC,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
